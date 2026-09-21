# Train.jl
#
# The training loop, the two compiled programs it drives, and the eval entry points.
#
# Gradient accumulation, the part easiest to get silently wrong:
#
#   | Forward and backward | unconditional, every micro-batch      |
#   | Accumulator reset    | traced `ifelse.` select, both arms run |
#   | Optimizer step       | HOST branch in the driver; skipped     |
#
# The optimizer is a separate program because a traced select cannot skip a stateful update: fused
# and gated, Adam's moments would decay once per micro-batch. `Optimisers.AccumGrad` has the same
# defect under tracing (it returns `zero.(dx)` rather than `nothing`, so the base rule still runs)
# and a CPU test that does not go through `@compile` will not catch it. There is no fused program
# at `accum = 1` either; one shape for every `accum` is the point.

# ── The two programs ────────────────────────────────────────────────────────────────

"""
    ReactantNitro.grad_program(ev, model, ps, st, batch, g_accum::NTuple{G}, is_first, inv_n)
        -> (loss, g_accum_new::NTuple{G}, st_new, stats)

One per run, invoked once per micro-batch.

`ps` crosses as the Lux tree `build_model` returned; the gradient accumulator and `opt_state`
cross flat, as `NTuple{G}` of per-group buffers, flattened on entry and unflattened on exit inside
the trace. `ev` is `compile_view(e)`, never `e`: Enzyme walks the whole `Const` argument, so a
dataset reachable from `e` would be traversed element-wise at every compile.

The accumulator reset is a broadcast select,
`ifelse.(is_first .> 0f0, g_scaled, g_accum .+ g_scaled)`: `ifelse.` because scalar `ifelse` is
defined for traced scalars only; `is_first` a 1-element array because a replicated scalar cannot
round-trip out of a program on a multi-device mesh; a select rather than a multiply by zero because
a `NaN` left in `g_accum` survives a multiply.

`inv_n = 1/accum` is a trace-time constant, which is why `accum` is in the compile cache key.

The Enzyme shadow `dps` is allocated inside the program on every invocation and is never an
argument, output or field of `Nitro`: Enzyme reverse accumulates into the shadow it is handed, so a
hoisted `dps` would add every past micro-batch's gradient to the current one with no error. Under
trace `make_zero` folds to a constant, so the allocation costs nothing at runtime.

The fourth result is `stats` under `:device` metrics residency and the raw `outputs` under `:host`,
where the driver calls `train_metrics` on the transferred outputs. `TM` is type-level, so the two
are separate programs and cache entries.
"""
function grad_program(
        ev, model, ps, st, batch, g_accum, is_first, inv_n, routers, lref,
        ::Val{TM} = Val(:device)
    ) where {TM}
    # Allocated here on every invocation, never hoisted: Enzyme accumulates into the shadow it is
    # handed, and a reused `dps` would sum every past micro-batch's gradient with no error.
    dps = Enzyme.make_zero(ps)
    _, (l, st_new, aux) = Enzyme.autodiff(
        Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI),
        Enzyme.Const(objective_wrapper), Enzyme.Duplicated,
        Enzyme.Const(ev), Enzyme.Const(model),
        Enzyme.Duplicated(ps, dps), Enzyme.Const(st), Enzyme.Const(batch),
        Enzyme.Const(routers), Enzyme.Const(Val(TM))
    )
    # `Base.eltype(x)(inv_n)`: `1/1` is a Float64, and a bare `inv_n` would promote a Float32
    # accumulator, so the second call would fail the thunk's argument-type guard.
    g_scaled = map(x -> x .* Base.eltype(x)(inv_n), flatten(dps, resolve_layout(lref)))
    # A broadcast select (see the docstring): `ifelse.` not `ifelse`, an array flag not a scalar,
    # a select not a multiply by zero.
    g_new = map((s, a) -> ifelse.(is_first .> 0.0f0, s, a .+ s), g_scaled, g_accum)
    return l, g_new, st_new, aux
end

"""
    ReactantNitro.opt_program(ev, g_accum::NTuple{G}, ps, opt_state::NTuple{G}, hp)
        -> (ps_new, states_new::NTuple{G})

One per run, invoked only on the last micro-step from the driver's host branch, which is what makes
it genuinely skipped rather than merely discarded.

Clipping runs first, on the fully accumulated `g_accum`, at [`gradient_clip_norm`](@ref)`(ev)`; the
threshold is type-level, so at its default `0` it emits no ops. `ev` is taken even though nothing is
differentiated here, so every trace site sees the stripped view.

Returns the per-group optimizer STATE, not the `Leaf`: a `Leaf` carries its rule, whose
hyperparameters are device scalars, and a replicated scalar cannot leave a compiled program on a
multi-device mesh. The driver re-attaches the rules it passed in.
"""
function opt_program(ev, g_accum, ps, opt_state, lref, ::Val{CLIP}) where {CLIP}
    layout = resolve_layout(lref)
    # `CLIP` is type-level so the default 0 folds away at trace time. As a traced value, `threshold
    # > 0` would be a host branch on a traced Bool, and a traced 0 would scale the gradient to NaN.
    g = clip_by_global_norm(g_accum, CLIP)
    p_flat = flatten(ps, layout)
    both = map(apply_group, opt_state, p_flat, g)
    # State only, never the `Leaf`: its rule's scalars cannot be program outputs on a mesh (every
    # multi-device run died in output codegen, even for a stateless `Descent`), and the driver
    # rebuilds the rules host-side each step anyway.
    return unflatten(map(last, both), ps, layout), map(t -> first(t).state, both)
end

"""
    ReactantNitro.objective_wrapper(ev, model, ps, st, batch)

The differentiated wrapper: `forward`, then `loss`, then `train_metrics`, returning
`(l, ignore_derivatives(st_new), ignore_derivatives(stats))`.

The return activity is `Enzyme.Duplicated` rather than `Active`, with
`set_abi(ReverseWithPrimal, ReactantABI)`, and every non-loss element is wrapped in
`EnzymeCore.ignore_derivatives`. That wrapper is load-bearing: Reactant seeds every `OUT` result
with ones, which is the correct `dL/dL = 1` for the loss and gradient corruption for anything else,
with no error. (`ignore_derivatives` is not exported from Reactant, so it is qualified. A `Ref`
stash for aux values fails silently under tracing.)

`ev` is a `Const` argument rather than a closure capture so its `Device` fields are traced inputs;
the routers are captures because they are genuine trace-time constants the cache key already covers.
"""
function objective_wrapper(ev, model, ps, st, batch, routers, ::Val{TM} = Val(:device)) where {TM}
    fns = hook_fns(routers)
    outputs, st_new = call_hook(
        hook_fn(fns, :forward, forward), :forward, routers.forward, batch, ev, model, ps, st
    )
    l = call_hook(hook_fn(fns, :loss, loss), :loss, routers.loss, batch, ev, outputs)
    # Under `:host` the hook is outside every program: what leaves is the primal and the driver
    # calls `train_metrics` on it. `TM` is type-level, so this branch folds at trace time.
    aux = TM === :host ? outputs :
        routers.train_metrics === nothing ? (;) :
        call_hook(
            hook_fn(fns, :train_metrics, train_metrics), :train_metrics,
            routers.train_metrics, batch, ev, outputs
        )
    # `ignore_derivatives` is load-bearing: every `OUT` is seeded with ones, and unbarriered state
    # would corrupt the gradient with no error. The suite carries a control that detects it.
    return (
        l,
        Enzyme.EnzymeCore.ignore_derivatives(st_new),
        Enzyme.EnzymeCore.ignore_derivatives(aux),
    )
end

# ── Mutating layer state ────────────────────────────────────────────────────────────
#
# `forward` returns `(outputs, st_new)`, so state always threads; a stateless model returns `st`.
# In train mode LuxLib's Reactant extension ignores running statistics, so threading changes only
# what is validated and exported. State updates once per layer APPLICATION: N times per optimizer
# step under accumulation, and more inside a weight-tied layer or an ODE solver, which matches
# PyTorch. Multi-device statistics are per device (unsynced), also PyTorch's default; `st_new`
# carries the same sharding annotation as `st` via `Reactant.Ops.sharding_group`. Keep constant
# data out of `st`, since it is returned every step; a `Device` field is converted once.

# ── Manual mode: the backward helper ────────────────────────────────────────────

"""
    backward(f, ps_sub, consts...) -> (loss, grads)

One reverse-mode pass over an objective with respect to one parameter subtree, for use inside a
[`train_step`](@ref) closure. `f(ps_sub, consts...)` returns the scalar loss; `grads` matches
`ps_sub`, and everything in `consts...` is `Const`, so no gradient flows into it.

Every traced value the objective reads must be an argument, never a closure capture: a captured
parameter subtree silently zeroes the gradient while the primal loss stays correct. A struct holding
no arrays, such as the model, is safe to capture.

The non-saturating GAN formulation falls out of the activities: with the discriminator's parameters
in `consts...`, the generator's loss adjoint flows through the discriminator as a function of the
fake data and never into its parameters. The shadow is allocated per call, never hoisted, as in the
gradient program.
"""
function backward(f, ps_sub, consts...)
    dps = Enzyme.make_zero(ps_sub)
    _, (l,) = Enzyme.autodiff(
        Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI),
        Enzyme.Const(_objective_wrapper), Enzyme.Duplicated,
        Enzyme.Const(f), Enzyme.Duplicated(ps_sub, dps), map(Enzyme.Const, consts)...
    )
    return l, dps
end

# A named wrapper with the objective as a `Const` argument, the shape Lux's Reactant extension
# verifies; the one-tuple is the `Duplicated`-return contract.
function _objective_wrapper(f, ps_sub, consts...)
    return (f(ps_sub, consts...),)
end

# ── Entry points ─────────────────────────────────────────────────────────────────────

"""
    train!(nitro) -> Nitro
    train!(e; kwargs...) -> Nitro

Train. Blocks and returns the [`Nitro`](@ref).

`train!(nitro)` takes no keywords; every keyword belongs to the `Nitro` constructor, and
`train!(e; kwargs...)` is sugar for `train!(Nitro(e; kwargs...))`. The handle is also reachable
through `run_ref::Ref{Nitro}`, filled before the loop starts, and as `info.nitro` in a phase
monitor. Re-training under a different stopping rule means constructing another `Nitro`; the
compile cache is module-level, so it reuses every compiled program.

There is no non-finite rollback: a non-finite loss stops the run with an error naming step and
epoch, and recovery is a resume from the last checkpoint at a lower learning rate. Every scalar
the framework branches on (the loss, the checkpoint metric, the early-stopping metric) is validated
on readback, because a failed `BufferToHost` on this stack returns garbage without raising;
purely-logged metrics are not.

Ctrl+C is a graceful stop: the loop runs on a worker thread, ^C becomes [`request_stop!`](@ref),
the step loop breaks at its next boundary, validation and the checkpoint still run, and the call
returns with `stop_reason = :requested`.
"""
train!(nitro::Nitro) = with_repl(() -> _train!(nitro), nitro; on_interrupt = _stop_on_interrupt)

# Ctrl+C lands on the parked caller, never on the worker, so it becomes the graceful stop; a
# failure during the wind-down surfaces its own exception rather than the interrupt.
function _stop_on_interrupt(t::Task, nitro::Nitro)
    request_stop!(nitro)
    try
        return fetch(t)
    catch e
        throw(unwrap_task_exc(e))
    end
end

# The eval entry points stop at the next batch boundary (`honor_stop`) and then surface the
# interrupt. The flag is reset because this stop was for this eval only.
function _eval_on_interrupt(t::Task, nitro::Nitro)
    request_stop!(nitro)
    try
        wait(t)
    finally
        nitro.stop_requested = false
    end
    rethrow()
end

function _train!(nitro::Nitro)
    nitro.opt_state === nothing && error("ReactantNitro: this `Nitro` was built without a `train` \
        split, so setup skipped the optimizer steps and there is no optimizer state to train with. \
        `Nitro(e; data = ...)` with no `train` key is the evaluation construction.")
    # Manual mode owns the step body and shares the epoch skeleton; the driver was chosen at
    # construction.
    get(nitro.frozen, :manual, false) && return _train_manual!(nitro)
    # Every keyword was resolved at construction, so a revised accessor is inert here; the report
    # says so once. The world-closure guard runs first so the report's staleness section can read
    # its result.
    world_closure_staleness()
    report_fixed_config(nitro, :train)
    ev = compile_view(nitro.e)
    layout = nitro.layout
    inv_n = one(Float32) / nitro.accum
    # 1-element arrays, not traced scalars: a replicated scalar cannot round-trip out of a compiled
    # program on a multi-device mesh.
    is_first = place_replicated(Float32[1], nitro.mesh)
    is_next = place_replicated(Float32[0], nitro.mesh)
    # Frozen at construction so an existing handle cannot recompile between two `train!` calls
    # after a redefinition; `worlds_opt` is separate so a rule edit stays off the gradient program.
    worlds = nitro.frozen.worlds_train
    worlds_opt = nitro.frozen.worlds_opt
    # Read, not rebuilt. `no_decay` is a user hook, so recomputing the mask here would let a
    # revision take effect on an existing handle, which is exactly what freezing forbids.
    masks, anchors = nitro.decay.masks, nitro.decay.anchors
    # One-slot memo so a constant `eta` crosses to the device once per run rather than once per
    # group per step; a scheduled one misses every step.
    scalar_memo = ScalarMemo()
    lref = layout_ref(layout)
    clipv = Val(nitro.gradient_clip_norm)
    # An experiment with no `train_metrics` stays on the `:device` program whatever the residency
    # says; there is nothing to move to the host.
    tmv = Val(nitro.frozen.tm_residency)
    # The per-run copy of the module-level registry, taken here rather than at construction, so
    # that a monitor registered between `Nitro(e)` and `train!` is in this run.
    adopt_monitors!(nitro)
    t_started = time()
    last_metrics = (;)
    # The final checkpoint rewrite applies only to an epoch THIS call wrote. `nitro.epoch > 0` is
    # not that condition: a resume restores the counter.
    wrote_epoch = false

    try
        while nitro.epoch < nitro.max_epochs
            nitro.epoch += 1
            set_phase!(nitro, TrainStepping())
            st = Lux.trainmode(nitro.st)
            seen = 0
            # The epoch's mean train loss, for `history`: summed over micro-batches as host
            # values the step already read back, so it costs no transfer of its own.
            loss_sum, loss_n = 0.0, 0
            # Host-wait accounting: two `time()` calls per micro-batch, and the one generic guard
            # that catches every variant of "the loader is the bottleneck".
            t_wait = 0.0
            t_step = 0.0
            # One loop body for all three prefetch paths (inline, one producer, fan-out). Planning
            # is its own reported stretch because `begin_epoch!` may be a server round trip, and
            # the label is "planning" because the reporter appends "epoch N/M" itself.
            stream = with_progress_stretch("planning", 0, nitro.epoch, nitro.max_epochs) do
                batch_stream(nitro.data.train, nitro.routing, nitro.mesh)
            end
            # One bar per epoch: the epoch position goes in the label. `div` is exact because
            # setup checked `length(train) % accum == 0`.
            progress_begin!(
                "train", _epoch_steps(nitro), nitro.epoch, nitro.max_epochs
            )
            try
                # An explicit `iterate` loop rather than `for`, so the pull can be timed.
                t0 = time()
                it = iterate(stream)
                t_wait += time() - t0
                while it !== nothing
                    ((batch, b), st_stream) = it
                    t_body = time()
                    seen += 1
                    check_batch_schema(batch, nitro.schema, :train, seen)
                    check_train_batch_shape(batch, nitro.batch_size, :train, seen, nitro.routing)

                    micro = (seen - 1) % nitro.accum
                    if micro == 0
                        # The experiment half of the per-step rebuild, here rather than beside
                        # `rebuild_rules` because the scheduled fields are read by the GRADIENT
                        # program. Guarded on `micro == 0` so every micro-batch of one step sees one
                        # set of values. `step_experiment` returns `e` unchanged when nothing
                        # device-side is scheduled.
                        e_next = step_experiment(
                            nitro.e, nitro.schedules, nitro.step + 1; mesh = nitro.mesh
                        )
                        if e_next !== nitro.e
                            nitro.e = e_next
                            ev = compile_view(e_next)
                        end
                    end
                    gthunk = compile_cached(
                        grad_program, ev, ev, nitro.model, nitro.ps, st, b,
                        nitro.g_accum, is_first, inv_n, nitro.routing, lref,
                        tmv; phase = GradCompiling(), worlds, nitro,
                        baked = (; accum = nitro.accum)
                    )
                    l, nitro.g_accum, st, aux = gthunk(
                        ev, nitro.model, nitro.ps, st, b,
                        nitro.g_accum,
                        micro == 0 ? is_first : is_next, inv_n,
                        nitro.routing, lref, tmv
                    )
                    # The readback is kept, not repeated: the divergence check already paid the
                    # D2H for this scalar, and logging it again would pay it twice for one number.
                    lh = check_finite(l, nitro.step, nitro.epoch)
                    loss_sum += Float64(lh)
                    loss_n += 1
                    # Freed here, not when the compiled call returns: XLA is asynchronous and the
                    # executable holds its inputs, but the loss readback above awaited the step, so
                    # `b`'s buffers are dead. A use-after-free would surface as
                    # `AssertionError: buffer.buffer !== C_NULL`.
                    free_batch!(b)
                    # The host path for `train_metrics`: `aux` is the primal, and the hook runs
                    # in ordinary Julia on it. The transfer is the cost the user chose by asking.
                    stats = tmv isa Val{:host} ?
                        call_host_hook(
                            train_metrics, :train_metrics, nitro.routing.train_metrics,
                            batch, nitro.e, host_tree(aux)
                        ) : aux

                    if micro == nitro.accum - 1
                        # A host branch, the only construct that genuinely skips: gated by a traced
                        # select, Adam's moments would decay once per micro-batch.
                        state = rebuild_rules(nitro, layout, masks, anchors, scalar_memo)
                        othunk = compile_cached(
                            opt_program, ev, ev, nitro.g_accum, nitro.ps, state,
                            lref, clipv; phase = OptCompiling(),
                            worlds = worlds_opt, nitro
                        )
                        nitro.ps, new_states = othunk(
                            ev, nitro.g_accum, nitro.ps, state, lref,
                            clipv
                        )
                        # Re-attach this step's rules to the returned states, so `opt_state` stays
                        # an NTuple of `Leaf`s for the checkpoint path.
                        nitro.opt_state = map(
                            (l, s) -> Optimisers.Leaf(l.rule, s, l.frozen), state, new_states
                        )
                        nitro.step += 1
                        # One completed unit of work, for a monitor measuring time since progress.
                        note_progress!()
                        # Once per optimizer step, the cadence the metrics contract names. The stats
                        # are the closing micro-batch's, unreduced.
                        log_metrics!(
                            nitro.logger, finite_only((; loss = lh, stats...));
                            step = nitro.step, epoch = nitro.epoch, context = "train"
                        )
                    end
                    # The step half ends after `check_finite`, which is where the device is awaited;
                    # timing `gthunk` alone would report a near-zero step. Micro-batch 1 is excluded
                    # from both halves because it carries the fan-out's spin-up.
                    seen > 1 && (t_step += time() - t_body)
                    nitro.stop_requested && break
                    t0 = time()
                    it = iterate(stream, st_stream)
                    seen > 1 && (t_wait += time() - t0)
                end
                # The exactly-once ledger, only on an epoch that ran to completion. It catches a
                # right count delivered with one index twice and another never.
                nitro.stop_requested || check_prefetch_delivery(stream)
            finally
                # Named because the bar sits full while the producers stop.
                progress_phase!("closing data stream")
                close_stream!(stream)
                # Taken back off: closing a full bar is a no-op, so a label left set would stay on
                # screen through validation setup and get the blame for it.
                progress_phase!("")
                # In the `finally`: a non-finite loss, a stop and an error all leave early.
                progress_end!()
            end
            nitro.st = st
            check_epoch_length(
                seen, length(nitro.data.train), :train;
                horizon_dependent = nitro.schedules !== nothing &&
                    nitro.schedules.horizon_dependent
            )
            report_data_wait!(nitro, t_wait, t_step)

            metrics_out = haskey(nitro.data, :val) ? run_eval(nitro, :val; report = false) : (;)
            last_metrics = metrics_out
            # On the handle too, which is what a REPL holds afterwards.
            nitro.last_metrics = metrics_out
            push!(nitro.history, history_row(nitro, loss_sum, loss_n, metrics_out))
            isempty(metrics_out) || log_metrics!(
                nitro.logger, finite_only(metrics_out);
                step = nitro.step, epoch = nitro.epoch,
                context = "validate"
            )
            set_phase!(nitro, Checkpointing())
            save_checkpoint_reported!(nitro, nitro.epoch, metrics_out)
            wrote_epoch = true

            # Both stopping routes set one flag, checked after validation and the checkpoint, so a
            # stop is a finished epoch exiting through `Done`. `should_stop` runs even when a stop
            # is already requested so a stateful policy stays consistent.
            stopped = should_stop(nitro.early_stop, nitro.epoch, metrics_out)
            if stopped || nitro.stop_requested
                nitro.stop_reason = stopped ? :early_stop : :requested
                break
            end
        end
    catch
        nitro.stop_reason = :error
        nitro.elapsed = time() - t_started
        progress_done!()
        # Through `try`: a run that died on I/O is the one whose manifest may be unreadable, and a
        # display helper must not replace the real exception.
        nitro.best_checkpoint = try
            selected_checkpoint(nitro.checkpointer, nitro.run_dir)
        catch
            nothing
        end
        set_phase!(nitro, Failed())
        finish!(nitro.logger, :error)
        rethrow()
    end
    nitro.stop_reason === nothing && (nitro.stop_reason = :completed)
    # Checkpoints are written per epoch, before the outcome is known, so the final epoch's record
    # is rewritten with `stop_reason` once it is: same file, one manifest entry replaced. Gated on
    # THIS call having written an epoch, not on `nitro.epoch > 0`: a resume into a finished run
    # restores the counter, exits the loop at once, and would otherwise overwrite the previous
    # process's final record with empty metrics, dropping it out of the top-K ranking.
    wrote_epoch && save_checkpoint_reported!(nitro, nitro.epoch, last_metrics)
    # `Done` for both stopping routes; `stop_reason` is where the difference lives.
    nitro.elapsed = time() - t_started
    # Per entry point, not per epoch: only here is it known that nothing follows.
    progress_done!()
    # After the final rewrite above, so the winning entry is the one the manifest ends up holding.
    nitro.best_checkpoint = selected_checkpoint(nitro.checkpointer, nitro.run_dir)
    set_phase!(nitro, Done())
    finish!(nitro.logger, nitro.stop_reason === :completed ? :completed : :early_stop)
    return nitro
end

# ── Manual mode: the driver ─────────────────────────────────────────────────────────

"""
    ReactantNitro.manual_program(ev, model, ps, opt_state, st, batch, router)

Manual mode's traced entry: [`train_step`](@ref) compiled as the closure's program, once per run.
The batch is narrowed to the closure's own routed fields and handed to `train_step` as keywords
through the same `call_hook` machinery as `forward`, so the missing-keyword error names the closure.
"""
function manual_program(ev, model, ps, opt_state, st, batch, router)
    return call_hook(train_step, :train_step, router, batch, ev, model, ps, opt_state, st)
end

"""
    ReactantNitro.check_train_step_return(out) -> out

The return-contract check for [`train_step`](@ref): a `NamedTuple` with `loss`, `ps`, `st` and
`opt_state` (`stats` defaults to `(;)`). Checked loudly so a misspelled key does not surface as a
`KeyError` or a silently defaulted `stats`.
"""
@noinline function check_train_step_return(out)
    out isa NamedTuple || error(
        """
        ReactantNitro: `train_step` returned a `$(typeof(out))`; it must return a `NamedTuple`
        `(; loss, ps, st, opt_state, stats)`. `stats` may be omitted and
        defaults to `(;)`."""
    )
    for k in (:loss, :ps, :st, :opt_state)
        haskey(out, k) || error(
            """
            ReactantNitro: `train_step` returned keys $(keys(out)) and is missing `$k`; it must
            return `(; loss, ps, st, opt_state, stats)`. A misspelled key would
            otherwise surface as a `KeyError` naming nothing about the contract."""
        )
    end
    return out
end

"""
    ReactantNitro._train_manual!(nitro) -> Nitro

Manual mode's driver: the automatic loop's epoch skeleton (prefetch, checks, device schedules,
the finite check, logging, `data_wait_frac`, validation, checkpointing, early stopping, phases,
`request_stop!`) with the step body replaced by one call to the user's [`train_step`](@ref)
closure, compiled once per run. The driver checks the closure's return, validates `loss`, stores
`ps` and `st`, and re-attaches the rules it handed in via `merge_rules`, since a `Leaf` cannot
leave a compiled program on a mesh.
"""
function _train_manual!(nitro::Nitro)
    world_closure_staleness()
    report_fixed_config(nitro, :train)
    ev = compile_view(nitro.e)
    worlds = nitro.frozen.worlds_manual
    router = nitro.routing.train_step
    adopt_monitors!(nitro)
    t_started = time()
    last_metrics = (;)
    wrote_epoch = false
    try
        while nitro.epoch < nitro.max_epochs
            nitro.epoch += 1
            set_phase!(nitro, TrainStepping())
            st = Lux.trainmode(nitro.st)
            seen = 0
            # The epoch's mean train loss, for `history`: summed over micro-batches as host
            # values the step already read back, so it costs no transfer of its own.
            loss_sum, loss_n = 0.0, 0
            # Per-epoch host-wait accounting, identical to the automatic loop.
            t_wait = 0.0
            t_step = 0.0
            # Planning is its own stretch: `begin_epoch!` may be a server round trip.
            stream = with_progress_stretch("planning", 0, nitro.epoch, nitro.max_epochs) do
                batch_stream(nitro.data.train, nitro.routing, nitro.mesh)
            end
            # One bar per epoch; `div` is exact because setup checked the divisibility.
            progress_begin!(
                "train", _epoch_steps(nitro), nitro.epoch, nitro.max_epochs
            )
            try
                t0 = time()
                it = iterate(stream)
                t_wait += time() - t0
                while it !== nothing
                    ((batch, b), st_stream) = it
                    t_body = time()
                    seen += 1
                    check_batch_schema(batch, nitro.schema, :train, seen)
                    check_train_batch_shape(batch, nitro.batch_size, :train, seen, nitro.routing)
                    # Device schedules write `e`'s fields; the closure reads them as traced inputs.
                    e_next = step_experiment(
                        nitro.e, nitro.schedules, nitro.step + 1; mesh = nitro.mesh
                    )
                    if e_next !== nitro.e
                        nitro.e = e_next
                        ev = compile_view(e_next)
                    end
                    # Narrowed to the closure's routed fields, as `eval_forward` does for `forward`.
                    bf = NamedTuple{keys(router)}(b)
                    # The per-step host rebuild of scheduled rules, type-preserving so one program
                    # serves every step.
                    opt_state = rebuild_scheduled_rules(nitro, nitro.step + 1)
                    thunk = compile_cached(
                        manual_program, ev, ev, nitro.model, nitro.ps, opt_state, st,
                        bf, router; phase = GradCompiling(), worlds, nitro
                    )
                    out = check_train_step_return(
                        thunk(ev, nitro.model, nitro.ps, opt_state, st, bf, router)
                    )
                    # One D2H serves both the divergence check and the log line.
                    lh = check_finite(out.loss, nitro.step, nitro.epoch)
                    loss_sum += Float64(lh)
                    loss_n += 1
                    nitro.ps = out.ps
                    st = out.st
                    # Re-attach the rules the closure was handed to the states it returned. The
                    # rules are not donated, so reuse across steps is safe.
                    nitro.opt_state = merge_rules(opt_state, out.opt_state)
                    # One train line per optimizer step, which in manual mode is one per
                    # batch (accum is fixed at 1 by construction).
                    log_metrics!(
                        nitro.logger,
                        finite_only((; loss = lh, get(out, :stats, (;))...));
                        step = nitro.step, epoch = nitro.epoch, context = "train"
                    )
                    # Safe to free: `check_finite` awaited the step.
                    free_batch!(b)
                    nitro.step += 1
                    note_progress!()
                    nitro.stop_requested && break
                    t0 = time()
                    it = iterate(stream, st_stream)
                    seen > 1 && (t_wait += time() - t0)
                end
                nitro.stop_requested || check_prefetch_delivery(stream)
            finally
                # Named because the bar sits full while the producers stop, then taken back off
                # (closing a full bar is a no-op, so a label left set would stay on screen).
                progress_phase!("closing data stream")
                close_stream!(stream)
                progress_phase!("")
                progress_end!()
            end
            nitro.st = st
            check_epoch_length(
                seen, length(nitro.data.train), :train;
                horizon_dependent = nitro.schedules !== nothing &&
                    nitro.schedules.horizon_dependent
            )
            report_data_wait!(nitro, t_wait, t_step)

            metrics_out = haskey(nitro.data, :val) ? run_eval(nitro, :val; report = false) : (;)
            last_metrics = metrics_out
            # On the handle too, which is what a REPL holds afterwards.
            nitro.last_metrics = metrics_out
            push!(nitro.history, history_row(nitro, loss_sum, loss_n, metrics_out))
            isempty(metrics_out) || log_metrics!(
                nitro.logger, finite_only(metrics_out);
                step = nitro.step, epoch = nitro.epoch,
                context = "validate"
            )
            set_phase!(nitro, Checkpointing())
            save_checkpoint_reported!(nitro, nitro.epoch, metrics_out)
            wrote_epoch = true

            stopped = should_stop(nitro.early_stop, nitro.epoch, metrics_out)
            if stopped || nitro.stop_requested
                nitro.stop_reason = stopped ? :early_stop : :requested
                break
            end
        end
    catch
        nitro.stop_reason = :error
        nitro.elapsed = time() - t_started
        progress_done!()
        # Through `try`: a display helper must not replace the real exception.
        nitro.best_checkpoint = try
            selected_checkpoint(nitro.checkpointer, nitro.run_dir)
        catch
            nothing
        end
        set_phase!(nitro, Failed())
        finish!(nitro.logger, :error)
        rethrow()
    end
    nitro.stop_reason === nothing && (nitro.stop_reason = :completed)
    # The final rewrite, as in the automatic loop, gated on this call having written an epoch.
    wrote_epoch && save_checkpoint_reported!(nitro, nitro.epoch, last_metrics)
    nitro.elapsed = time() - t_started
    # Per entry point, not per epoch.
    progress_done!()
    # After the final rewrite above, so the winning entry is the one the manifest ends up holding.
    nitro.best_checkpoint = selected_checkpoint(nitro.checkpointer, nitro.run_dir)
    set_phase!(nitro, Done())
    finish!(nitro.logger, nitro.stop_reason === :completed ? :completed : :early_stop)
    return nitro
end

# One entry point rather than two so the inner `Nitro` construction does not publish `Repl` on its
# way out, a microsecond before the first training phase.
train!(e; kwargs...) = with_repl_result(() -> train!(Nitro(e; kwargs...)))

"""
    ReactantNitro.DATA_WAIT_WARN

The `data_wait_frac` above which [`report_data_wait!`](@ref) warns. Below a quarter the loader is
comfortably ahead of the device; above it the run is paying for host work.
"""
const DATA_WAIT_WARN = 0.25

"""
    ReactantNitro.report_data_wait!(nitro, t_wait, t_step) -> nothing

The per-epoch host-wait fraction `t_wait / (t_wait + t_step)`, logged as `data_wait_frac` under
`context = "data"` and warned on (once per process) past [`DATA_WAIT_WARN`](@ref).

This is the generic guard: every other data-path check targets a specific mistake, while this one
measures the outcome, so it catches a source that is simply too slow, a cache cap, or a contended
sample server. `t_wait` is time blocked pulling from the stream; `t_step` is the rest of the
micro-batch body through the loss readback, where the device work is awaited. Both exclude each
epoch's first micro-batch. A third context because the `"train"` and `"validate"` cadences are
asserted by the suite.
"""
function report_data_wait!(nitro::Nitro, t_wait::Float64, t_step::Float64)
    total = t_wait + t_step
    total > 0 || return nothing
    frac = t_wait / total
    log_metrics!(
        nitro.logger, (; data_wait_frac = frac);
        step = nitro.step, epoch = nitro.epoch, context = "data"
    )
    frac < DATA_WAIT_WARN && return nothing
    pf = prefetch_config(nitro.data.train)
    @warn """
    ReactantNitro: the training loop spent $(round(100 * frac; digits = 1))% of epoch \
    $(nitro.epoch) BLOCKED waiting for the data source. The device is idle for
    that fraction of the epoch, so this is close to a $(round(1 / max(1 - frac, 1.0e-3); digits = 1))x
    wall-clock penalty against a run whose loader keeps up.
    Resolved prefetch for the `train` split: $(pf.workers) worker(s), $(pf.device_batches) batch(es)
    on device, $(pf.host_batches) on host, path $(pf.path). $(
        (pf.path === :fanout || pf.path === :fanout_unordered) ?
            "The fan-out is active, so the source itself is the limit: profile it in a CPU gate by iterating `build_data(e, nothing).train` directly, and check for a decoded-cache cap or a contended sample server." :
            "There is no host-side concurrency here; see the prefetch warning at setup for the two methods that enable it."
    )""" maxlog = 1
    return nothing
end

"""
    ReactantNitro.rebuild_rules(nitro, layout, masks, anchors, memo = nothing) -> NTuple{G}

The per-optimizer-step rebuild on the optimizer side: evaluate the schedules at this step, wrap the
results as device scalars, rebuild each group's rule chain, and put it back in its `Leaf` beside the
state the optimizer owns. Each group's `eta_t` is its own path key if configured, else the bare
`eta` key, else nothing; the ratio math lives in `effective_lr`.

A rebuilt rule travels inside `opt_state`, which is why the optimizer program takes no separate
`hp` argument. Rebuilding with a fresh `ConcretePJRTNumber` re-enters the same compiled thunk,
because the rule's type is unchanged; the [`ScalarMemo`](@ref) changes which buffer the rule
carries and nothing in the cache key.
"""
function rebuild_rules(
        nitro::Nitro, layout::FlatLayout{G}, masks, anchors, memo = nothing
    ) where {G}
    sched = nitro.schedules
    step = nitro.step + 1
    return ntuple(Val(G)) do gi
        # The path key if configured, else the bare `eta` key, else the base rate.
        eta_t = if sched === nothing
            nothing
        else
            pk = Symbol(layout.groups[gi], ".eta")
            haskey(sched.opt, pk) ? sched.opt[pk](step) :
                haskey(sched.opt, :eta) ? sched.opt.eta(step) : nothing
        end
        # `mesh` is required: without it the rule's scalars are placed for one device while the
        # moments are replicated across the mesh, and the program refuses to rebuild the `Leaf`.
        # Invisible on CPU, where every device count is 1.
        hp = resolve_hp(nitro.e, layout, gi; eta_t, anchors, masks, mesh = nitro.mesh, memo)
        Optimisers.Leaf(
            build_chain(nitro.e, layout.groups[gi], hp),
            nitro.opt_state[gi].state, nitro.opt_state[gi].frozen
        )
    end
end

"""
    ReactantNitro.to_device_batch(batch, routing) -> NamedTuple

Transfer only the routed fields to device. A field no hook declares reaches nobody, and a
bookkeeping field such as a `case_id` could not be transferred at all.
"""
to_device_batch(batch::NamedTuple, routing, mesh = nothing) =
    NamedTuple{routed_fields(routing)}(
    map(
        x -> place_batch(x, mesh),
        NamedTuple{routed_fields(routing)}(batch)
    )
)

"""
    ReactantNitro.check_finite(l, step, epoch) -> Float64

No non-finite rollback: a non-finite loss stops the run with an error naming step and epoch, and
recovery is a resume at a lower learning rate. The readback itself is validated because a failed
`BufferToHost` on this stack returns garbage without raising.
"""
function check_finite(l, step, epoch)
    v = l isa Number ? Float64(l) : Float64(only(Array(l)))
    # Returned so the one D2H serves both the divergence check and the train-metric line.
    isfinite(v) && return v
    error(
        """
        ReactantNitro: the training loss is $v at step $step, epoch $epoch.
        There is no non-finite rollback by design: a diverged run fails fast so
        you resume from the last checkpoint with a lower learning rate, rather than continuing from
        poisoned state."""
    )
end

"""
    ReactantNitro.check_control_readback(v, what, name) -> Float64

The validated readback for scalars the framework branches on (the early-stopping metric and the
checkpoint selection metric). A logged `NaN` costs a missing point on a chart; acting on one
truncates a healthy run or lets a stalled one continue.
"""
function check_control_readback(v, what::AbstractString, name::Symbol)
    x = v isa Real ? Float64(v) : nothing
    (x !== nothing && isfinite(x)) || error(
        """
        ReactantNitro: $what reads metric `$name`, whose value is $(repr(v)), and that value drives
        control flow.
        The framework validates every scalar readback it branches on, because a failed device-to-host
        readback returns garbage on this stack WITHOUT raising, and a non-finite or non-numeric
        value here would either truncate a healthy run or let a diverged one continue.
        A metric used for control flow must be a finite real number; `count === nothing` metrics that
        accumulate an array are for reporting, not for stopping."""
    )
    return x
end

"""
    ReactantNitro.finite_only(metrics) -> NamedTuple

The logging contract's "the framework drops non-finite values before calling". Values are read
back to host here, so a backend receives host numbers. Dropped rather than zeroed or passed
through: a `NaN` at a backend is rejected or plotted as a gap, and the framework cannot tell a
diverged metric from a failed readback at this point.
"""
function finite_only(metrics::NamedTuple)
    ks = Symbol[]
    vs = Any[]
    for k in keys(metrics)
        # `host_tree` rather than `_host_value`: a metric may be a tuple or NamedTuple of leaves.
        v = host_tree(getproperty(metrics, k))
        v isa Real && !isfinite(v) && continue
        push!(ks, k)
        push!(vs, v)
    end
    # The last boundary assertion before user-supplied backend code.
    return assert_host(
        NamedTuple{Tuple(ks)}(Tuple(vs)), "the metrics being handed to the logger"
    )
end

"""
    ReactantNitro.save_checkpoint_reported!(nitro, epoch, metrics) -> nothing

[`save_checkpoint!`](@ref) as one reported stretch, so a watcher can tell a slow write to a remote
filesystem from an epoch that finished and hung. Nothing is reported without a checkpointer, since
`save_checkpoint!(::Nothing, ...)` is a no-op. Every write gets the one label `"checkpoint"`,
including the final rewrite; a watcher has nothing to do differently about which is in flight.
"""
function save_checkpoint_reported!(nitro::Nitro, epoch, metrics)
    write() = save_checkpoint!(nitro.checkpointer, epoch, metrics, snapshot(nitro))
    nitro.checkpointer === nothing && return write()
    return with_progress_stretch(write, "checkpoint", 0, epoch, nitro.max_epochs)
end

"""
    ReactantNitro.snapshot(nitro) -> NamedTuple

The record handed to [`save_checkpoint!`](@ref): everything but `metrics` and the two
framework-stamped fields, which the checkpointer adds.

`ps`, `st` and `opt_state` are all converted to host values here. JLD2 serializes a
`ConcretePJRTArray` field by field, raw device pointer included, without raising on write or read;
the failure lands much later as `AssertionError: buffer.buffer !== C_NULL` inside a readback. The
rule for the whole record is write host, read host, normalize on the way in.
"""
function snapshot(nitro::Nitro)
    lgr = nitro.logger
    state = logger_state(lgr)
    # A non-`nothing` `logger_state` makes `reattach!` required; checked now rather than hours
    # later at the resume.
    if state !== nothing
        check_logger_state_serializable(state, lgr)
        applicable(reattach!, lgr, state) || error(
            """
            ReactantNitro: `logger_state` on this `$(nameof(typeof(lgr)))` returned
            $(repr(state)), which makes `reattach!` REQUIRED.
            The pair is self-describing: `logger_state` defaults to `nothing`, meaning "I have no
            resumable state", and then nothing is stored and no reattachment is attempted. Returning
            anything else is a promise this logger cannot keep.
            Define `ReactantNitro.reattach!(lgr::$(nameof(typeof(lgr))), state)`, or return
            `nothing`."""
        )
    end
    snap = (;
        ps = to_host(nitro.ps), st = to_host(nitro.st),
        opt_state = to_host(nitro.opt_state),
        flat_permutation = nitro.layout.permutation,
        step = nitro.step, epoch = nitro.epoch, seed = nitro.seed,
        config = graphconst_fields(nitro.e), devices = device_values(nitro.e),
        run_id = run_id(lgr), run_url = run_url(lgr),
        # Both or neither: storing the type alone would be a refusal waiting to happen on resume.
        logger_state = state,
        logger_type = state === nothing ? nothing : string(nameof(typeof(lgr))),
        anchor_checksum = nitro.anchor_checksum,
        stop_reason = nitro.stop_reason,
        preset = nitro.preset,                  # the named configuration, if any
    )
    # A structural walk over leaves, not elements: about a hundred checks per epoch.
    assert_host_record(snap)
    return snap
end

"""
    ReactantNitro.check_logger_state_serializable(state, lgr) -> nothing

`logger_state` must return plain serializable data (a `String`, `NamedTuple` or `Dict`), never the
live backend object: the record goes through JLD2 and outlives the process. Checked here rather
than failing on the resume in another process.
"""
function check_logger_state_serializable(state, lgr)
    state isa Union{AbstractString, NamedTuple, AbstractDict, Number, Symbol, Tuple} && return nothing
    error(
        """
        ReactantNitro: `logger_state` on this `$(nameof(typeof(lgr)))` returned a
        `$(typeof(state))`. It must return plain serializable data, a `String`, `NamedTuple`, or
        `Dict`, and never the live backend object.
        The record goes through JLD2 and outlives the process, so a stored handle is either
        unserializable or dead on arrival. Return the few identifiers your backend needs to
        reattach: an experiment key, a run id plus project and entity, a tracking URI."""
    )
end

"""
    ReactantNitro.HostRNG

A host surrogate for `Reactant.ReactantRNG`, whose type parameter forbids host contents, so there
is no host-resident `ReactantRNG` to rebuild. A value whose type cannot hold host data cannot
round-trip through a record as itself; the record stores a surrogate and [`from_host`](@ref) turns
it back on the way in. Found as `st.<layer>.rng.seed` on any Lux model carrying a `Dropout`.
"""
struct HostRNG
    seed::Vector{UInt64}
    algorithm::String
end

to_host(x::Optimisers.Leaf) = Optimisers.Leaf(to_host_rule(x.rule), to_host(x.state), x.frozen)
to_host(x::Reactant.RNumber) = Reactant.to_number(x)
to_host(x::Reactant.AbstractConcreteArray) = Array(x)
to_host(r::Reactant.ReactantRNG) = HostRNG(Array(r.seed), r.algorithm)

# Identity-preserving, so an unchanged container returns the SAME object. That is what lets the
# struct method below decide "nothing moved" reliably and skip a reconstruction it does not need.
function to_host(x::Union{Tuple, NamedTuple})
    y = map(to_host, x)
    return all(i -> y[i] === x[i], 1:length(x)) ? x : y
end

"""
    ReactantNitro.to_host(x)

The generic fallback: walk any struct's fields, not only the containers, reconstructing only when
something actually moved so structs with inner constructors are never rebuilt needlessly. A struct
that holds a device value and cannot be rebuilt raises here, in the writing process. Types that
cannot be rebuilt from host values at all get a surrogate (see [`HostRNG`](@ref)).
"""
function to_host(x)
    T = typeof(x)
    isstructtype(T) || return x
    fs = fieldnames(T)
    isempty(fs) && return x
    vals = ntuple(i -> to_host(getfield(x, fs[i])), length(fs))
    all(i -> vals[i] === getfield(x, fs[i]), 1:length(fs)) && return x
    return T.name.wrapper(vals...)
end

"""
    ReactantNitro.from_host(x)

The mirror of [`to_host`](@ref) on the restore path: every surrogate back into the device-resident
value it stood for, everything else left for `to_rarray`. A general walk so the next surrogate
needs a method rather than a mechanism.
"""
from_host(h::HostRNG) = Reactant.ReactantRNG(Reactant.to_rarray(h.seed), h.algorithm)

function from_host(x::Union{Tuple, NamedTuple})
    y = map(from_host, x)
    return all(i -> y[i] === x[i], 1:length(x)) ? x : y
end

function from_host(x)
    T = typeof(x)
    isstructtype(T) || return x
    fs = fieldnames(T)
    isempty(fs) && return x
    vals = ntuple(i -> from_host(getfield(x, fs[i])), length(fs))
    all(i -> vals[i] === getfield(x, fs[i]), 1:length(fs)) && return x
    return T.name.wrapper(vals...)
end

"""
    ReactantNitro.device_paths(x, path = "") -> Vector{String}

Every device-resident value reachable from `x`, as `path :: Type` strings, which is exactly what
neither JLD2 nor the eventual `AssertionError` will tell you. The sibling of
[`assert_device_state`](@ref), pointed the other way.
"""
function device_paths(x, path::AbstractString = "", out::Vector{String} = String[])
    if x isa Reactant.RNumber || x isa Reactant.AbstractConcreteArray
        push!(out, string(path, " :: ", typeof(x)))
    elseif x isa Union{Tuple, NamedTuple}
        for (k, v) in pairs(x)
            device_paths(v, string(path, ".", k), out)
        end
    elseif x isa AbstractArray
        # A wrapper (`view`, `reshape`, `vec`) may hide a device array the test above cannot see.
        p = _hidden_array(x)
        if p !== nothing
            device_paths(p, string(path, " (", nameof(typeof(x)), " parent)"), out)
        elseif eltype(x) <: AbstractArray
            for (i, v) in pairs(x)
                device_paths(v, string(path, "[", i, "]"), out)
            end
        end
        return out
    elseif x isa AbstractString || x isa Symbol || x isa Number
        return out
    elseif isstructtype(typeof(x))
        for f in fieldnames(typeof(x))
            device_paths(getfield(x, f), string(path, ".", f), out)
        end
    end
    return out
end

"""
    ReactantNitro.assert_host(x, what) -> x

The host/device boundary assertion. The framework converts every value it hands to host-side user
code; this asserts the conversion was total and names the path of anything that survived.

It exists because on a CPU backend a "device" array is host memory and scalar indexing into one is
legal, so a residency mistake is nearly unobservable there and a green CPU suite proves nothing
about it. Reaching this assertion means the conversion has a gap, not that the user forgot an
`Array(...)`. The cost is a walk over leaves, not elements. [`assert_host_record`](@ref) is the same
idea at the checkpoint boundary; both share [`device_paths`](@ref).
"""
function assert_host(x, what::AbstractString)
    leaked = device_paths(x, "")
    isempty(leaked) && return x
    error(
        """
        ReactantNitro: $what still holds DEVICE-resident values, at:
            $(join(leaked, "\n    "))
        Everything crossing to host-side user code is converted by the framework first, so this
        is a GAP IN THE CONVERSION rather than a missing `Array(...)` in your hook. Two such gaps
        have been found: `to_host` reached containers but not structs, and
        `host_tree` covered the outputs the framework owns while the routed batch fields did not.
        A leaf whose type CANNOT hold a host value needs a surrogate plus a `from_host` method;
        `ReactantNitro.HostRNG` is the worked example.
        If the path above starts at a value your own hook RETURNED, convert it there: the framework
        cannot convert what it has not seen yet."""
    )
end

"""
    ReactantNitro.assert_host_record(snap) -> nothing

"Every record value is a host value", asserted when the record is built, the one moment the
failure is cheap. Without it a device value is written and read back by JLD2 without complaint and
kills a fresh process on `AssertionError: buffer.buffer !== C_NULL`, naming neither the field nor
serialization.
"""
function assert_host_record(snap)
    leaked = String[]
    for k in (:ps, :st, :opt_state, :devices)
        hasproperty(snap, k) && device_paths(getproperty(snap, k), string(k), leaked)
    end
    isempty(leaked) && return nothing
    error(
        """
        ReactantNitro: this checkpoint record still holds DEVICE-resident values, at:
            $(join(leaked, "\n    "))
        Every value in a record must be a HOST value. A device array survives JLD2:
        it is written field by field, raw pointer included, and read back without complaint, so
        the failure lands much later, in another process, as
        `AssertionError: buffer.buffer !== C_NULL` from inside a readback that names neither the
        field nor serialization.
        `to_host` walks containers and structs. A type that CANNOT hold host values needs a
        surrogate and a `from_host` method to turn it back; `ReactantNitro.HostRNG` is the worked
        example."""
    )
end

"""
    ReactantNitro.to_host_rule(r) -> r

The mirror of [`to_device_rule`](@ref). An optimizer rule is a struct, not a container, so before
`to_host` walked structs a rule reached the record still holding the `ConcretePJRTNumber` the
per-step rebuild put there.
"""
function to_host_rule(r)
    T = typeof(r)
    isstructtype(T) || return to_host(r)
    return T.name.wrapper(map(f -> to_host(getfield(r, f)), fieldnames(T))...)
end
to_host_rule(c::Optimisers.OptimiserChain) = Optimisers.OptimiserChain(map(to_host_rule, c.opts)...)
to_host_rule(x::Union{Tuple, NamedTuple}) = map(to_host_rule, x)

"""
    validate(nitro) -> NamedTuple

Run the `:val` split and return the finalized metrics. This is what the training loop calls each
epoch, and it works on a `Nitro` that has never trained. Eval mode throughout; `st_new` is
discarded, and each batch's device buffers are freed explicitly rather than left to the GC, which
is the documented cause of device OOM during validation on this stack. Runs on a worker thread;
Ctrl+C stops at the next batch boundary.
"""
validate(nitro::Nitro) = with_repl(
    () -> run_eval(nitro, :val; honor_stop = true), nitro; on_interrupt = _eval_on_interrupt
)

"""
    evaluate(nitro; split = :test) -> NamedTuple

Run any named split from the data collection, erroring on a name `build_data` did not return and
listing the available ones. Shares one compiled `forward` and one compiled `metrics` with
validation and inference, so moving between them never recompiles. Runs on a worker thread; Ctrl+C
stops at the next batch boundary.
"""
evaluate(nitro::Nitro; split::Symbol = :test) = with_repl(
    () -> run_eval(nitro, split; honor_stop = true), nitro; on_interrupt = _eval_on_interrupt
)

"""
    predict(nitro, batch::NamedTuple) -> outputs
    predict(nitro, loader) -> iterator

Inference: [`forward`](@ref) alone, in eval mode.

Takes a batch `NamedTuple` or anything iterating them, never a bare array: `forward` is
keyword-routed from the batch's field names, and the error for a bare array names the fields it
declares. Only `forward`'s fields are required, so a prediction batch needs no labels.

One batch in, one output tree out, sliced to the real sample count, as host arrays. A loader in, a
lazy iterator out, one element per batch; `collect` it for all of them. An output leaf whose last
dimension is not the batch raises rather than slicing the wrong axis.
"""
predict(nitro::Nitro, batch::NamedTuple) = with_repl(
    () -> _predict(nitro, batch), nitro; on_interrupt = _eval_on_interrupt
)

function _predict(nitro::Nitro, batch::NamedTuple)
    # The world-closure guard, as in every entry point.
    world_closure_staleness()
    routing, batch_size = predict_routing!(nitro, batch)
    ev = compile_view(nitro.e)
    st = Lux.testmode(nitro.st)
    padded, n_real = pad_batch(batch, batch_size, routing)
    b = to_device_batch(padded, routing, nitro.mesh)
    outputs = eval_forward(nitro, ev, st, b, routing.forward)
    host = host_tree(outputs)
    free_device_buffers!(protected_buffers(nitro, st), b, outputs)
    # Sliced on the host: `predict` transfers the outputs anyway, so slicing on device would only
    # buy a second compiled program per short shape. The assertion matters here because `predict`
    # is the one host crossing with no hook behind it; a leak would land in the caller's code.
    return assert_host(slice_outputs(host, n_real, batch_size), "the outputs `predict` is returning")
end

# An array is iterable, so without this a bare array would reach the loader form and fail
# somewhere unhelpful.
function predict(nitro::Nitro, x::AbstractArray)
    ks = nitro.routing === nothing ? nothing : keys(nitro.routing.forward)
    how = ks === nothing ? "Wrap it in a NamedTuple whose fields are the ones `forward` declares." :
        "Call it as `predict(nitro, (; " * join(("$k = ..." for k in ks), ", ") * "))`."
    error(
        """
        ReactantNitro: `predict` takes a batch `NamedTuple` or something iterating them, not a bare
        `$(typeof(x))`. `forward` is keyword-routed from the batch's field names, so the framework
        needs those names to route.
        $how"""
    )
end

predict(nitro::Nitro, loader) = Iterators.map(b -> predict(nitro, b), loader)

"""
    ReactantNitro.predict_routing!(nitro, batch) -> (routing, batch_size)

Deferred routing. A `Nitro` built with no split has no first batch to resolve routing from, so
resolution moves from setup to the first `predict` call, and `batch_size` is then that batch's
width. A `Nitro` that does have a split keeps setup's routing, and this only checks that the batch
carries `forward`'s fields; a prediction batch legitimately lacks labels, so the full schema is not
checked.
"""
function predict_routing!(nitro::Nitro, batch::NamedTuple)
    if nitro.routing === nothing
        ev = compile_view(nitro.e)
        routing = resolve_routing(
            ev, batch; nitro.model, nitro.ps, nitro.st, hooks = hook_fns(nitro.routing)
        )
        validate_batch(batch, routing)
        nitro.routing = routing
        nitro.schema = keys(batch)
        nitro.batch_size = batch_size_of(batch, routing)
        return ((; forward = routing.forward), nitro.batch_size)
    end
    fwd = nitro.routing.forward
    missed = Tuple(k for k in keys(fwd) if !haskey(batch, k))
    isempty(missed) || error(
        """
        ReactantNitro: the batch handed to `predict` has fields $(keys(batch)) and is missing
        $(missed), which `forward` was routed at setup.
        Only the fields `forward` declares are required, so a prediction batch does not need labels;
        these are `forward`'s own."""
    )
    # Route to `forward`'s fields alone, so a batch without labels still pads and transfers.
    return ((; forward = fwd), nitro.batch_size)
end

# ── The eval programs, and the pad-and-slice around them ────────────────────────────

"""
    ReactantNitro.fwd_program(ev, model, ps, st, batch, routers) -> outputs

The eval-mode forward shared by `predict`, `validate` and `evaluate`. It always runs at
`batch_size`, since the framework pads before and slices after, so it compiles once. `st_new` is
discarded inside the program. It takes `forward`'s own router and a batch narrowed to `forward`'s
fields, so a `predict` batch without labels and a `validate` batch with them share one cache entry.
"""
function fwd_program(ev, model, ps, st, batch, fwd_router, fwd_fn = forward)
    outputs, _ = call_hook(fwd_fn, :forward, fwd_router, batch, ev, model, ps, st)
    return outputs
end

"""
    ReactantNitro.eval_metric_program(ev, outputs, batch, router, ::Val{NREAL}, ::Val{HOOK})

The traced metric path: `metrics` compiled as its own program on outputs and batch fields sliced to
`NREAL` inside the trace. `NREAL` is type-level because a traced length cannot slice, which also
puts the two shapes (full and remainder) in the cache key. `HOOK` selects the user's `metrics` or
the framework's `val_loss` substitution for an experiment defining none; the substitution is
traced whatever `metrics_residency` says, because it calls `loss`, a traced hook by contract.
"""
function eval_metric_program(
        ev, outputs, batch, router, ::Val{NREAL}, ::Val{HOOK}, fns = (;)
    ) where {NREAL, HOOK}
    o = slice_last(outputs, NREAL)
    b = slice_last(batch, NREAL)
    return HOOK === :metrics ?
        call_hook(hook_fn(fns, :metrics, metrics), :metrics, router, b, ev, o) :
        (; val_loss = (call_hook(hook_fn(fns, :loss, loss), :loss, router, b, ev, o), 1))
end

"""
    ReactantNitro.eval_forward(nitro, ev, st, b, fwd_router) -> outputs

Compile-or-reuse and invoke [`fwd_program`](@ref): the one place `predict` and the metric loop share
the entry, with the one narrowing of the batch that makes the sharing survive a label-free batch.
"""
function eval_forward(nitro::Nitro, ev, st, b, fwd_router)
    bf = NamedTuple{keys(fwd_router)}(b)
    # Frozen against an eval-mode `st`, which every caller passes. The handle-local memo means a
    # hit never reaches the phase machinery, so only the first compile publishes `EvalCompiling`.
    thunk = compile_cached(
        fwd_program, ev, ev, nitro.model, nitro.ps, st, bf, fwd_router,
        hook_fn(hook_fns(nitro.routing), :forward, forward);
        phase = EvalCompiling(), worlds = nitro.frozen.worlds_eval,
        gc_hash = nitro.frozen.graphconst_hash, nitro
    )
    return thunk(
        ev, nitro.model, nitro.ps, st, bf, fwd_router,
        hook_fn(hook_fns(nitro.routing), :forward, forward)
    )
end

"""
    ReactantNitro.run_eval(nitro, split; report = true, honor_stop = false) -> NamedTuple

The body of [`validate`](@ref) and [`evaluate`](@ref). `honor_stop = true`, which the standalone
entry points pass, breaks at the next batch boundary after [`request_stop!`](@ref); the training
loop's own per-epoch validation keeps `false`, so the validation after a requested stop runs in
full.

Per batch: pad to `batch_size`, transfer the routed fields, run the shared eval `forward`, compute
the metric on the real samples, accumulate `(sum, count)` on the host, free the batch's device
buffers. Then divide and hand the result to [`finalize_metrics`](@ref). Under `:host` residency
(the default) the outputs are transferred and sliced and `metrics` runs in ordinary Julia on the
split's own host batch; under `:device` the slice and the call happen inside
[`eval_metric_program`](@ref) and only the metric scalars cross.
"""
function run_eval(nitro::Nitro, split::Symbol; report::Bool = true, honor_stop::Bool = false)
    # `report = false` for the training loop's per-epoch validation, which already printed the
    # banner; the world-closure guard runs with it.
    if report
        world_closure_staleness()
        report_fixed_config(nitro, :eval)
    end
    haskey(nitro.data, split) || error(
        """
        ReactantNitro: there is no `$split` split. `build_data` returned $(keys(nitro.data)).
        `evaluate` names the split it wants and errors on one that does not exist, which is why
        the collection is a NamedTuple rather than a positional tuple: a positional one would have
        silently evaluated a different split."""
    )
    e, ev = nitro.e, compile_view(nitro.e)
    routing = nitro.routing
    st = Lux.testmode(nitro.st)                 # the framework owns the switch
    # With no `metrics` the framework substitutes the validation loss through `loss`'s own router.
    hook = routing.metrics === nothing ? :val_loss : :metrics
    router = hook === :val_loss ? routing.loss : routing.metrics
    # The substitution is traced whatever the residency says.
    traced = hook === :val_loss || nitro.frozen.metrics_residency === :device
    # A host metric reads the split's own host batch, so only `forward`'s fields are transferred.
    xfer = traced ? routing : (; forward = routing.forward)
    protected = protected_buffers(nitro, st)
    acc, prev_phase = nothing, nitro.phase
    set_phase!(nitro, EvalStepping())
    # Planned first, then counted, then the bar: `begin_epoch!` may be a server round trip, and a
    # source whose plan changes its batch count is only counted correctly afterwards. The stream
    # lives for this pass only and is closed in the `finally`, so training and evaluation prefetch
    # memory never coexist.
    stream = with_progress_stretch("planning", 0, nitro.epoch, nitro.max_epochs) do
        eval_stream(getproperty(nitro.data, split), xfer, nitro.batch_size, nitro.mesh)
    end
    progress_begin!(
        String(split), _split_length(getproperty(nitro.data, split)),
        nitro.epoch, nitro.max_epochs
    )
    try
        for (idx, (batch, b)) in enumerate(stream)
            # Standalone entry points stop here after Ctrl+C; the training loop's validation does
            # not.
            honor_stop && nitro.stop_requested && break
            # `step` does not move during evaluation, so this is what a stall watchdog sees.
            note_progress!()
            check_batch_schema(batch, nitro.schema, split, idx)
            # Recomputed from the same host batch the producer padded from, so it cannot disagree.
            n_real = batch_size_of(batch, xfer)
            outputs = eval_forward(nitro, ev, st, b, routing.forward)
            m = if traced
                # Host-side and only when there is a slice to protect; inside the trace the same
                # assertion would compare each leaf against itself.
                n_real == nitro.batch_size || check_output_batch_dim(outputs, nitro.batch_size)
                thunk = compile_cached(
                    eval_metric_program, ev, ev, outputs, b, router,
                    Val(n_real), Val(hook), hook_fns(nitro.routing); phase = EvalCompiling(),
                    worlds = nitro.frozen.worlds_eval, nitro
                )
                thunk(ev, outputs, b, router, Val(n_real), Val(hook), hook_fns(nitro.routing))
            else
                # `metrics` NEVER sees padding: the outputs are sliced back to `n_real`, and the
                # batch handed to it is the split's own host batch, which was never padded.
                call_host_hook(
                    metrics, :metrics, router, batch, e,
                    slice_outputs(host_tree(outputs), n_real, nitro.batch_size)
                )
            end
            acc = accumulate_metrics(acc, host_metrics(m, split), split)
            # Freed eagerly: leaving eval outputs to the GC is what OOMs validation on this stack.
            free_device_buffers!(protected, b, outputs)
        end
        # The exactly-once ledger, on a pass that ran to completion.
        (honor_stop && nitro.stop_requested) || check_prefetch_delivery(stream)
    catch
        set_phase!(nitro, prev_phase)
        progress_end!()
        report && progress_done!()
        rethrow()
    finally
        # One teardown for every exit. Named as a phase because the bar sits full while producers
        # stop and buffered device batches are freed.
        progress_phase!("closing data stream")
        close_stream!(stream)
        # Taken back off: closing a full bar is a no-op, so the label would otherwise stay up.
        progress_phase!("")
    end
    progress_end!()
    # A stretch rather than a phase, since the bar has closed. `finalize_metrics` is user code (a
    # confusion matrix, an AUC) and unreported it is a silent stall between the bar and the
    # checkpoint.
    out = with_progress_stretch("finalize metrics", 0, nitro.epoch, nitro.max_epochs) do
        # The boundary assertion on the output: `finalize_metrics` is user code and its result
        # reaches the logger, the phase monitors and the checkpoint metric.
        assert_host(
            finalize_metrics(e, reduce_metrics(acc), split),
            "the metrics `finalize_metrics` returned for the `$split` split"
        )
    end
    # After the finalize stretch: `:done` prints the newline that closes the reused terminal line.
    report && progress_done!()
    # The transition out of `EvalStepping` carries `info.metrics`, which is where a user logs
    # something custom at validation time; there is no dedicated hook.
    set_phase!(nitro, prev_phase; metrics = out)
    return out
end

# Optimizer steps in one epoch, for the training bar. Exact because setup checked the
# divisibility; zero when the length is unknowable, which the reporter reads as indeterminate.
function _epoch_steps(nitro::Nitro)
    return try
        div(_split_length(nitro.data.train), max(nitro.accum, 1))
    catch
        0
    end
end

# A split's batch count without iterating it; a source that promises no length gets zero.
function _split_length(v)
    return try
        Base.IteratorSize(typeof(v)) isa Union{Base.HasLength, Base.HasShape} ? length(v) : 0
    catch
        0
    end
end

# ── Metric accumulation, host-side ───────────────────────────────────────────────────

"""
    ReactantNitro.host_metrics(m, split) -> NamedTuple of (sum, count)

Read one batch's metric result back to host values and check the metric contract: every metric is
a `(sum, count)` pair, where `count === nothing` means accumulate by summation without dividing,
which is what a confusion matrix needs. A bare number is refused rather than misaccumulated.
"""
function host_metrics(m, split::Symbol)
    m isa NamedTuple || error(
        """
        ReactantNitro: `metrics` returned a `$(typeof(m))` for the `$split` split; it must return a
        NamedTuple of `(sum, count)` pairs, as in
        `(; err = (sum_abs_err, n_items), acc = (n_correct, n_images))`."""
    )
    return NamedTuple{keys(m)}(map(k -> _host_metric_entry(getproperty(m, k), k, split), keys(m)))
end

function _host_metric_entry(v, name::Symbol, split::Symbol)
    (v isa Tuple && length(v) == 2) || error(
        """
        ReactantNitro: metric `$name` on the `$split` split is a `$(typeof(v))`; every metric must
        report its own numerator and denominator as a `(sum, count)` pair. The
        framework adds the pairs up over the split and divides at the end, which is why a
        framework-supplied sample count would be the wrong denominator: different metrics have
        different natural ones, per sample, per image, per object.
        Write `$name = (the sum, the count)`, or `(the sum, nothing)` to accumulate by summation
        without dividing, which is what a confusion matrix wants."""
    )
    s, c = v
    # `host_tree` because a numerator may be a container of leaves, not a single one.
    return (host_tree(s), c === nothing ? nothing : host_tree(c))
end

_host_value(x::Reactant.RNumber) = Reactant.to_number(x)
_host_value(x::Reactant.AbstractConcreteArray) = Array(x)
_host_value(x) = x

"""
    ReactantNitro.accumulate_metrics(acc, m, split) -> NamedTuple

Add one batch's `(sum, count)` pairs into the accumulator. The key set is fixed by the first batch:
a metric measured on some batches and not others has no honest denominator.
"""
function accumulate_metrics(acc, m::NamedTuple, split::Symbol)
    acc === nothing && return m
    keys(acc) === keys(m) || error(
        """
        ReactantNitro: `metrics` returned $(keys(m)) on one batch of the `$split` split and
        $(keys(acc)) on an earlier one. The metric set must be the same for every batch, because the
        framework accumulates each metric's own (sum, count) across the split, and a
        metric present on only some batches has no honest denominator."""
    )
    return NamedTuple{keys(acc)}(
        map(keys(acc)) do k
            (as, ac), (ms, mc) = getproperty(acc, k), getproperty(m, k)
            (ac === nothing) == (mc === nothing) || error(
                """
                ReactantNitro: metric `$k` on the `$split` split reported a `count` on one batch and
                `nothing` on another. `count === nothing` means accumulate by summation without
                dividing, so the two cannot be mixed within one metric."""
            )
            (as .+ ms, ac === nothing ? nothing : ac + mc)
        end
    )
end

"""
    ReactantNitro.reduce_metrics(acc) -> NamedTuple

The divide: `sum / count`, or the bare `sum` where `count === nothing`. Runs **before**
[`finalize_metrics`](@ref), which is what makes an `f1 = 2tp / (2tp + fp + fn)` finalizer work:
its `tp`, `fp`, and `fn` are count-less sums and arrive at the finalizer unmodified.
"""
reduce_metrics(acc::NamedTuple) =
    NamedTuple{keys(acc)}(map(v -> v[2] === nothing ? v[1] : v[1] / v[2], values(acc)))

# ── Device OOM during validation ─────────────────────────────────────────────────────

"""
    ReactantNitro.protected_buffers(nitro, st) -> Set{Ptr{Cvoid}}

The device buffers [`free_device_buffers!`](@ref) must never touch: parameters, both copies of the
layer state, `w0`, the optimizer state and the gradient accumulator, all of which outlive the
batch. A `forward` returning one of its inputs unchanged would otherwise hand the eval loop a live
parameter buffer to free. The experiment is not walked: it may carry a materialized dataset.
Computed once per eval call, not per batch.
"""
function protected_buffers(nitro::Nitro, st = nothing)
    out = Set{Ptr{Cvoid}}()
    for tree in (nitro.ps, nitro.st, st, nitro.w0, nitro.opt_state, nitro.g_accum)
        tree === nothing && continue
        _each_concrete_leaf(tree) do x
            for buf in _device_buffers(x)
                buf.buffer == C_NULL || push!(out, buf.buffer)
            end
        end
    end
    return out
end

"""
    ReactantNitro.free_device_buffers!(protected, trees...) -> nothing

Free the device buffers of one eval batch and its outputs explicitly rather than leaving them to
the GC, the documented cause of device OOM during validation on this stack. A buffer in
`protected`, a buffer XLA donated, and a pointer already freed in this call are each skipped.
Nulling the pointer is what lets Reactant's own finalizer compose with the early free.
"""
function free_device_buffers!(protected::Set{Ptr{Cvoid}}, trees...)
    freed = Set{Ptr{Cvoid}}()
    for tree in trees
        _each_concrete_leaf(tree) do x
            for buf in _device_buffers(x)
                p = buf.buffer
                (p == C_NULL || p in protected || p in freed) && continue
                push!(freed, p)
                Reactant.XLA.free_buffer(buf)
                buf.buffer = C_NULL
            end
        end
    end
    return nothing
end

# A read-only walk rather than `Functors.fmap`, which would rebuild every container it descends.
# Containers this framework threads (NamedTuples, `NTuple{G}`, `Leaf`) are reached; an output inside
# some other struct is not, which costs a late GC-timed free rather than a wrong one.
_is_concrete_leaf(x) = x isa Reactant.AbstractConcreteArray || x isa Reactant.AbstractConcreteNumber

function _each_concrete_leaf(f, x)
    _is_concrete_leaf(x) && return (f(x); nothing)
    if x isa Optimisers.Leaf
        _each_concrete_leaf(f, x.rule)
        _each_concrete_leaf(f, x.state)
    elseif x isa Union{Tuple, NamedTuple}
        for v in x
            _each_concrete_leaf(f, v)
        end
    elseif x isa AbstractArray && !(eltype(x) <: Number)
        for v in x
            _each_concrete_leaf(f, v)
        end
    elseif _walkable_struct(x)
        for fld in fieldnames(typeof(x))
            _each_concrete_leaf(f, getfield(x, fld))
        end
    end
    return nothing
end

# Rules and output structs are worth reaching; numbers, strings, symbols, functions and types are
# not containers.
_walkable_struct(x) = isstructtype(typeof(x)) &&
    !(x isa Union{Number, AbstractString, Symbol, Function, Type, AbstractArray})

# PJRT only: `data` is an NTuple of AsyncBuffer there. On IFRT this returns nothing and the eager
# free degrades to GC-timed collection.
function _device_buffers(x)
    getfield(x, :donated) && return ()
    data = getfield(x, :data)
    data isa Tuple || return ()
    return map(d -> d.buffer, data)
end

"""
    ReactantNitro.host_tree(tree) -> tree

Every device leaf of a tree read back to host values, structure preserved. This is what a `:host`
metric hook and `predict` receive.

It walks structs as well as containers: the earlier `Functors.fmap` form treated an unregistered
struct as a leaf, so a device array inside a plain struct came back unconverted on the path feeding
every `:host` metric hook. Identity is preserved, so a subtree with nothing device-resident comes
back `===` what it was. Unlike [`to_host`](@ref) it has no surrogate types: a hook wants the value.
"""
host_tree(x::Reactant.RNumber) = Reactant.to_number(x)
host_tree(x::Reactant.AbstractConcreteArray) = Array(x)

# `ReactantRNG` cannot hold a host value (its constructor rejects a `Vector{UInt64}` seed), so it is
# passed through unchanged. An RNG's seed is not model data, no metric hook is handed layer state,
# and export traces `st` rather than reading it. Found by the first export of a model with a
# `Dropout`.
host_tree(r::Reactant.ReactantRNG) = r

# A host array is a leaf: an array of arrays is walked, an array of numbers is not.
function host_tree(x::AbstractArray)
    # A wrapper over a device array: convert the parent, rebuild the wrapper, then materialize.
    # `Array(view_of_device_array)` would copy elementwise, legal on CPU and fatal on a GPU.
    p = _hidden_array(x)
    if p !== nothing
        hp = host_tree(p)
        hp === p && return x            # a host wrapper over host memory: nothing moved
        return Array(_rewrap(x, hp))
    end
    return _is_array_leaf(x) ? x : _host_tree_container(x)
end

"""
    ReactantNitro._rewrap(x, p) -> x_over_p

Rebuild array wrapper `x` over a new parent `p`, for [`host_tree`](@ref). Only the wrappers this
stack produces are covered; anything else refuses by name rather than handing traced code a device
array. Adding a wrapper is one method.
"""
_rewrap(x::SubArray, p) = view(p, parentindices(x)...)
_rewrap(x::Base.ReshapedArray, p) = reshape(p, size(x))
_rewrap(x::LinearAlgebra.Adjoint, p) = LinearAlgebra.adjoint(p)
_rewrap(x::LinearAlgebra.Transpose, p) = LinearAlgebra.transpose(p)
_rewrap(x::PermutedDimsArray{T, N, perm}, p) where {T, N, perm} = PermutedDimsArray(p, perm)
@noinline _rewrap(x, _) = error(
    """
    ReactantNitro: `$(nameof(typeof(x)))` wraps a DEVICE array and the framework does not know how
    to rebuild it over a host parent, so it cannot convert it for you.
    Every array wrapper the framework has met is covered: `SubArray`, `ReshapedArray`, `Adjoint`,
    `Transpose`, `PermutedDimsArray`. This one is new, and adding it is one `_rewrap` method.
    Refusing rather than passing it through, because a device array reaching host code raises
    "Scalar indexing is disallowed" on a GPU and silently works on CPU."""
)

host_tree(x::Union{Tuple, NamedTuple}) = _host_tree_container(x)

function _host_tree_container(x)
    y = map(host_tree, x)
    return all(i -> y[i] === x[i], eachindex(x)) ? x : y
end

# The generic fallback: reconstruct only when something moved. Strings, symbols and numbers are
# struct types to `isstructtype` and must not be walked.
function host_tree(x)
    (x isa AbstractString || x isa Symbol || x isa Number) && return x
    T = typeof(x)
    isstructtype(T) || return x
    fs = fieldnames(T)
    isempty(fs) && return x
    vals = ntuple(i -> host_tree(getfield(x, fs[i])), length(fs))
    all(i -> vals[i] === getfield(x, fs[i]), 1:length(fs)) && return x
    return T.name.wrapper(vals...)
end
