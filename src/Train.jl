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
    ReactantNitro.grad_program(ev, model, ps, st, batch, g_accum::NTuple{G}, is_first, inv_n,
                               routers, lref, ::Val{TM}) -> (loss, g_accum::NTuple{G}, st_new, aux)

The gradient program: compiled once per run, invoked once per micro-batch. It writes the scaled
gradient into `g_accum` in place and returns it.

`ps` crosses as the Lux tree `build_model` returned; the gradient accumulator crosses flat, as an
`NTuple{G}` of per-group buffers, and the gradient is flattened into it inside the trace. `ev` is `compile_view(e)`, never `e`: Enzyme walks the whole `Const` argument, so a
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

`aux` is the `train_metrics` result when `TM === :device` and the raw `outputs` when
`TM === :host`, where the driver calls `train_metrics` on the transferred outputs. `TM` is
type-level, so the two are separate programs and cache entries.
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
    # Written into `g_accum` rather than returned as a new tree: Reactant donates only the arguments
    # a trace mutates, so a fresh result would leave the old accumulator for the GC every micro-batch.
    foreach((s, a) -> (a .= ifelse.(is_first .> 0.0f0, s, a .+ s)), g_scaled, g_accum)
    return l, g_accum, program_out(st_new), program_out(aux)
end

# ── Gradient rules in the driver ───────────────────────────────────────────────────

"""
    ReactantNitro.micro_grad!(rule, nitro, gctx, ev, st, b, flag) -> (l, st_out, aux, extra)

One micro-batch's gradient into `nitro.g_accum` under `rule`. `extra` is what the deferred half
needs beyond the loss: empty for [`SingleGrad`](@ref), `(; l2, ps_adv)` for a
[`PerturbRule`](@ref).
"""
function micro_grad!(::SingleGrad, nitro, gctx, ev, st, b, flag)
    l, nitro.g_accum, st_out, aux = grad_call(nitro, gctx, ev, nitro.ps, st, b, nitro.g_accum, flag)
    return l, st_out, aux, (;)
end

function micro_grad!(rule::PerturbRule, nitro, gctx, ev, st, b, flag)
    # Pass 1 overwrites the scratch; its loss, state and outputs are the micro-batch's.
    l, gctx.scratch[], st_out, aux =
        grad_call(nitro, gctx, ev, nitro.ps, st, b, gctx.scratch[], gctx.is_first)
    ps_adv = compile_cached(
        eps_program, ev, rule, nitro.ps, gctx.scratch[], gctx.masks, gctx.lref;
        phase = GradCompiling(), nitro,
        worlds = (method_world(perturbation, Tuple{typeof(rule), Any, Any, Any}),)
    )(rule, nitro.ps, gctx.scratch[], gctx.masks, gctx.lref)
    # The same input `st`: the gradient program does not donate it, and reusing it gives pass 2
    # the same RNG key, so dropout draws the same masks.
    l2, nitro.g_accum, _, _ = grad_call(nitro, gctx, ev, ps_adv, st, b, nitro.g_accum, flag)
    return l, st_out, aux, (; l2, ps_adv)
end

function grad_call(nitro, gctx, ev, ps, st, b, acc, flag)
    thunk = compile_cached(
        grad_program, ev, ev, nitro.model, ps, st, b, acc, gctx.is_first, gctx.inv_n,
        nitro.routing, gctx.lref, gctx.tmv; phase = GradCompiling(), worlds = gctx.worlds, nitro,
        baked = (; accum = nitro.accum)
    )
    return thunk(
        ev, nitro.model, ps, st, b, acc, flag, gctx.inv_n, nitro.routing, gctx.lref, gctx.tmv
    )
end

# This step's rule on the device, refused if its type moved since construction.
function step_grad_rule(nitro, grad_raw, bind_raw, memo)
    r = build_grad_rule(nitro.e, grad_hp(grad_raw, bind_raw))
    typeof(r) == nitro.frozen.grad_rule_type || error(
        "ReactantNitro: `grad_rule` now returns a `$(typeof(r))`, but this handle was built \
         with a `$(nitro.frozen.grad_rule_type)`. The rule's type is fixed: vary its values only, \
         or construct a new `Nitro` to change the rule."
    )
    return place_grad_rule(r, memo; mesh = nitro.mesh)
end

"""
    ReactantNitro.opt_program(ev, g_accum::NTuple{G}, ps, opt_state::NTuple{G}, lref, ::Val{CLIP})
        -> (ps, states::NTuple{G})

The optimizer program: compiled once per run and called from a host branch on the last
micro-batch of each step only, so the other micro-batches never run it. The step's
hyperparameters travel inside the rules in `opt_state` (see [`rebuild_rules`](@ref)).

Clipping runs first, on the fully accumulated `g_accum`, at the global norm `CLIP`
([`gradient_clip_norm`](@ref)); the threshold is type-level, so at its default `0` it emits no ops.
`ev` is passed although nothing is differentiated, so every trace site sees the stripped view.
The new parameters are written into `ps` in place.

Returns the per-group optimizer state without its `Leaf`: a `Leaf` carries its rule, whose
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
    # Written into `ps` leaf for leaf, for the same donation reason as the accumulator: the
    # parameters stay allocated once instead of leaving one superseded tree per step for the GC.
    Functors.fmap(ps, unflatten(map(last, both), ps, layout)) do p, p_new
        p isa AbstractArray && (p .= p_new)
        return p
    end
    return ps, map(t -> first(t).state, both)
end

"""
    ReactantNitro.objective_wrapper(ev, model, ps, st, batch, routers, ::Val{TM} = Val(:device))

The differentiated function: `forward`, then `loss`, then (when `TM === :device`) `train_metrics`,
returning `(l, ignore_derivatives(st_new), ignore_derivatives(aux))`. Under `TM === :host`, `aux`
is the raw `outputs`.

The return activity is `Enzyme.Duplicated` with `set_abi(ReverseWithPrimal, ReactantABI)`, and
every element but the loss is wrapped in `EnzymeCore.ignore_derivatives`. Reactant seeds every
`OUT` result with ones, which is the correct `dL/dL = 1` for the loss; without the wrapper the
seed on any other result corrupts the gradient with no error. A `Ref` stash for aux values does
not work under tracing either.

`ev` is a `Const` argument rather than a closure capture so its `Device` fields are traced inputs.
The routers are trace-time constants that the cache key already covers.
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
# PyTorch. On a mesh the program is one global computation, so batch statistics are over the
# global batch (synced, unlike PyTorch's per-device default) and the returned state is replicated;
# its `ReactantRNG`s leave through `program_out`. Keep constant data out of `st`, since it is
# returned every step; a `Device` field is converted once.

# ── Manual mode: the backward helper ────────────────────────────────────────────

"""
    backward(f, ps_sub, consts...) -> (loss, grads)

One reverse-mode pass over an objective with respect to one parameter subtree, for use inside a
[`train_step`](@ref) closure. `f(ps_sub, consts...)` returns the scalar loss; `grads` matches
`ps_sub`, and everything in `consts...` is `Const`, so no gradient flows into it.

Every traced value the objective reads must be an argument, never a closure capture: a captured
parameter subtree gets a zero gradient with no error, while the loss is still correct. A struct
holding no arrays, such as the model, is safe to capture.

For a GAN generator step, pass the discriminator's parameters in `consts...`: the generator's
gradient flows through the discriminator as a function of the fake data and never into its
parameters. The gradient buffer is allocated per call, as in the gradient program.
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

Run the training loop until `max_epochs`, an early stop or a requested stop, and return the
[`Nitro`](@ref). The call blocks.

`train!(nitro)` takes no keywords; every keyword belongs to the `Nitro` constructor, and
`train!(e; kwargs...)` is sugar for `train!(Nitro(e; kwargs...))`. The handle is also reachable
through `run_ref::Ref{Nitro}`, filled before the loop starts, and as `info.nitro` in a phase
monitor. To retrain with a different stopping rule, construct another `Nitro`; the compile cache
is module-level, so the new handle reuses every compiled program.

A non-finite loss stops the run with an error naming step and epoch; there is no rollback, and
recovery is a resume from the last checkpoint at a lower learning rate. The automatic loop reads
the loss back one micro-batch late, so the device runs the next micro-batch while the host waits;
the error is therefore raised after at most one more micro-batch (and, at a step's last
micro-batch, its optimizer step) has been dispatched, and still names the step the loss belonged
to. [`check_divergence`](@ref)` = false` turns the stop off: the loss is logged and training
continues. Every scalar the framework branches on (the loss, the checkpoint metric, the
early-stopping metric) is validated on readback, because a failed `BufferToHost` on this stack
returns garbage without raising; metrics that are only logged are not, nor is the loss under
`check_divergence = false`.

The train line is written every [`log_every_n_steps`](@ref) optimizer steps and carries the loss,
the `train_metrics` of the step's last micro-batch and, under [`autolog_schedules`](@ref), each
varying scalar schedule under a `schedule/` key. At the end, `stop_reason` is `:completed`,
`:early_stop`, `:requested` or, if the loop threw, `:error`.

Ctrl+C is a graceful stop: the loop runs on a worker thread, ^C becomes [`request_stop!`](@ref),
the step loop breaks at its next boundary, validation and the checkpoint still run, and the call
returns with `stop_reason = :requested`.

A handle built with `restore_best = true` ends every `train!` that did not fail (completion, early
stop, a requested stop) by loading its own run's selected checkpoint into `ps` and `st`; step,
epoch, `history` and the last metrics stay as training ended. A further `train!` on that handle
is refused, since those weights beside the final optimizer state and step match no state any run
was in; the error lists the constructions that do. With no scored checkpoint it warns and keeps
the final weights.
"""
function train!(nitro::Nitro)
    check_trainable_after_restore(nitro)
    return with_repl(() -> _train!(nitro), nitro; on_interrupt = _stop_on_interrupt)
end

"""
    ReactantNitro.check_trainable_after_restore(nitro) -> nothing

Refuse `train!` on a handle whose weights `restore_best` replaced: the best epoch's parameters
beside the final optimizer state, step and schedule position would train from a state no run was
ever in. The error gives the three valid constructions as code, with this handle's experiment,
preset, run directory and run id filled in.
"""
function check_trainable_after_restore(nitro::Nitro)
    rb = nitro.restored_best
    rb === nothing && return nothing
    T = string(nameof(typeof(nitro.e)))
    head = nitro.preset === nothing ? "$(T)()" : "$(T), :$(nitro.preset)"
    id = run_key(rb.run)
    dir = repr(nitro.run_dir)
    haskey(rb, :average) && return error(
        """
        ReactantNitro: this handle's weights were replaced at the end of `train!` by
        `restore_best = $(rb.average)`: epochs $(join(rb.epochs, ", ")) of run `$(id)`, averaged.
        It cannot train again: averaged weights beside the optimizer state, step and schedule
        position training ended in are not a training state any run was ever in.
        For the averaged weights with a fresh optimizer (a new run):
            Nitro($(head); run_dir = $(dir), weights = "$(id)" => $(rb.average))"""
    )
    return error(
        """
        ReactantNitro: this handle's weights were replaced at the end of `train!` by its best
        checkpoint (`restore_best = true`): epoch $(rb.epoch) of run `$(id)`, `$(rb.path)`.
        It cannot train again. Its optimizer state, step and schedule position are still where
        training ended, at epoch $(nitro.epoch), so the best weights beside them are not a training
        state any run was ever in, and training on would continue from a mixture.
        Build a new handle for the state you want:
          the best weights with a fresh optimizer (a new run):
            Nitro($(head); run_dir = $(dir), weights = "$(id)" => :best)
            (or, with this handle in hand, `run_dir = "<new dir>", weights = n => :best`)
          the best weights and their optimizer state, with a new schedule:
            Nitro($(head); run_dir = $(dir), weights = "$(id)" => :best, restore_optimizer = true)
          the exact training state at the best epoch (continues as a branch of run `$(id)`):
            Nitro($(head); run_dir = $(dir), resume = "$(id)" => :best)"""
    )
end

"""
    ReactantNitro.restore_best!(nitro) -> nothing

The last step of `train!` under `restore_best = true`: load `ps` and `st` from this run's selected
checkpoint, `nitro.best_checkpoint`, which is scoped to the handle's own run and the
checkpointer's current metric. `checkpoint_source` becomes that file, so export provenance names it, and
`restored_best` records it for `show` and for the refusal of a further `train!`. A run that saved
no scored checkpoint warns and keeps its final weights.
"""
function restore_best!(nitro::Nitro)
    nitro.restore_best isa Average && return restore_average!(nitro, nitro.restore_best)
    nitro.restore_best || return nothing
    bc = nitro.best_checkpoint
    if bc === nothing
        @warn "ReactantNitro: `restore_best = true`, and this run saved no scored checkpoint (no \
               `val` split, no checkpointer, or no epoch under the checkpointer's metric), so the \
               handle keeps the weights training ended with."
        return nothing
    end
    record = load_checkpoint(nitro.checkpointer, bc.path)
    nitro.ps = place_replicated(from_host(to_host(record.ps)), nitro.mesh)
    nitro.st = place_replicated(from_host(to_host(record.st)), nitro.mesh)
    nitro.checkpoint_source = bc.path
    nitro.restored_best = (;
        path = bc.path, epoch = bc.epoch, metric = bc.metric, score = bc.score,
        run = hasproperty(bc, :run) ? bc.run : checkpoint_run(nitro),
    )
    @info "ReactantNitro: `restore_best = true`: the handle now holds the weights of its best \
           checkpoint, epoch $(bc.epoch) ($(bc.metric) $(_shown(bc.score))), `$(bc.path)`; step and \
           epoch stay at $(nitro.step) and $(nitro.epoch)."
    return nothing
end

# `restore_best = Average(sel)`: the run's own checkpoints, averaged as `weights` would load them.
function restore_average!(nitro::Nitro, avg)
    ck = nitro.checkpointer
    what = "`restore_best = $(avg)`"
    src = try
        resolve_average(ck, nitro.run_dir, run_key(ck.run), avg; what)
    catch err
        @warn "ReactantNitro: $(what), and this run cannot supply it, so the handle keeps the \
               weights training ended with. $(first(split(sprint(showerror, err), '\n')))"
        return nothing
    end
    record = average_records(ck, src)
    nitro.ps = place_replicated(from_host(to_host(record.ps)), nitro.mesh)
    nitro.st = place_replicated(from_host(to_host(record.st)), nitro.mesh)
    nitro.checkpoint_source = describe(src)
    nitro.averaged_from = record.averaged_from
    nitro.restored_best = (;
        path = describe(src), epoch = maximum(src.epochs), average = avg, epochs = src.epochs,
        run = ck.run,
    )
    @info "ReactantNitro: $(what): the handle now holds $(describe(src)); step and epoch stay at \
           $(nitro.step) and $(nitro.epoch)."
    return nothing
end

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
    rule = step_grad_rule(
        nitro, grad_schedule_values(nitro.schedules, nitro.step + 1),
        bind_schedule_values(nitro.schedules, nitro.step + 1), scalar_memo
    )
    gctx = (;
        is_first, inv_n, lref, tmv, worlds, masks,
        scratch = Ref{Any}(rule isa PerturbRule ? map(zero, nitro.g_accum) : nothing),
    )
    # The per-run copy of the module-level registry, taken here rather than at construction, so
    # that a monitor registered between `Nitro(e)` and `train!` is in this run.
    adopt_monitors!(nitro)
    t_started = time()
    last_metrics = (;)
    # The final checkpoint rewrite applies only to an epoch THIS call wrote. `nitro.epoch > 0` is
    # not that condition: a resume restores the counter. Nor is having written some earlier epoch:
    # under `keep_latest = false` the checkpointer declines an epoch outside the top K, and the
    # rewrite then has no record of the final epoch to rewrite.
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
            # The one dispatched micro-batch whose loss readback, batch free and (if it closed a
            # step) train line are still owed; see `complete_micro_batch!`. Depth 1, and empty at
            # every epoch edge: drained after the step loop on both exits, completion and a stop.
            pending = nothing
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
                # Set on a step's first micro-batch and logged on its closing one.
                device_raw = grad_raw = bind_raw = (;)
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
                        device_raw = device_schedule_values(nitro.schedules, nitro.step + 1)
                        e_next = step_experiment(
                            nitro.e, nitro.schedules, nitro.step + 1; mesh = nitro.mesh,
                            raw = device_raw
                        )
                        if e_next !== nitro.e
                            nitro.e = e_next
                            ev = compile_view(e_next)
                        end
                        grad_raw = grad_schedule_values(nitro.schedules, nitro.step + 1)
                        bind_raw = bind_schedule_values(nitro.schedules, nitro.step + 1)
                        rule = step_grad_rule(nitro, grad_raw, bind_raw, scalar_memo)
                    end
                    l, st_out, aux, extra = micro_grad!(
                        rule, nitro, gctx, ev, st, b, micro == 0 ? is_first : is_next
                    )
                    st, aux = program_in(st_out), program_in(aux)
                    # What this micro-batch's deferred half needs, captured now: the step it
                    # belongs to (before the increment below, which is what the divergence error
                    # names) and the `e` its host metrics see (the next step's `step_experiment`
                    # may replace it before the record is completed).
                    # `sched` is the step's schedule values, set below on a logged closing
                    # micro-batch and written with its train line.
                    rec = (;
                        l, b, batch, aux, e = nitro.e, step = nitro.step, epoch = nitro.epoch,
                        closes = micro == nitro.accum - 1, sched = (;), extra,
                    )

                    if rec.closes
                        # A host branch, the only construct that genuinely skips: gated by a traced
                        # select, Adam's moments would decay once per micro-batch.
                        applied = applied_hp(nitro, layout)
                        state = rebuild_rules(nitro, layout, masks, anchors, scalar_memo; applied)
                        if logs_schedules(nitro) && step_line_due(nitro, nitro.step + 1)
                            rec = merge(
                                rec, (;
                                    sched = schedule_log(
                                        nitro.schedules, device_raw; groups = layout.groups,
                                        applied, fields = nitro.bindings.fields, grad_raw,
                                        bind_raw
                                    ),
                                )
                            )
                        end
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
                    end
                    # The previous micro-batch's readback, one behind, so the host waits on
                    # program k-1 while program k (and this step's optimizer program) is already
                    # queued behind it, and the device is not idle through the host work between.
                    if pending !== nothing
                        loss_sum += complete_micro_batch!(nitro, pending, tmv)
                        loss_n += 1
                    end
                    pending = rec
                    # The step half ends after the readback, which is where the device is awaited;
                    # timing the dispatch alone would report a near-zero step. Lagged, it awaits the
                    # previous micro-batch, so over an epoch it still sums the device time the host
                    # waited through. Micro-batch 1 is excluded from both halves because it
                    # carries the fan-out's spin-up.
                    seen > 1 && (t_step += time() - t_body)
                    nitro.stop_requested && break
                    t0 = time()
                    it = iterate(stream, st_stream)
                    seen > 1 && (t_wait += time() - t0)
                end
                # The drain, on both ways out of the step loop (exhausted, or a requested stop), so
                # `loss_sum`, the last step's train line and the last batch's free all land before
                # anything after the epoch reads them. An exception skips it on purpose: the record's
                # program may still be in flight, so its batch is left to the GC, as before.
                if pending !== nothing
                    t_drain = time()
                    loss_sum += complete_micro_batch!(nitro, pending, tmv)
                    loss_n += 1
                    pending = nothing
                    seen > 1 && (t_step += time() - t_drain)
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
            nitro.stop_requested || check_epoch_length(
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
            wrote_epoch = save_checkpoint_reported!(nitro, nitro.epoch, metrics_out) !== false

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
            own_selected_checkpoint(nitro.checkpointer, nitro.run_dir)
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
    # THIS call having written the final epoch, not on `nitro.epoch > 0`: a resume into a finished
    # run restores the counter, exits the loop at once, and would otherwise overwrite the previous
    # process's final record with empty metrics, dropping it out of the top-K ranking. An epoch the
    # checkpointer declined (`keep_latest = false`, outside the top K) is not written here either.
    wrote_epoch && save_checkpoint_reported!(nitro, nitro.epoch, last_metrics)
    # `Done` for both stopping routes; `stop_reason` is where the difference lives.
    nitro.elapsed = time() - t_started
    # Per entry point, not per epoch: only here is it known that nothing follows.
    progress_done!()
    # After the final rewrite above, so the winning entry is the one the manifest ends up holding.
    nitro.best_checkpoint = own_selected_checkpoint(nitro.checkpointer, nitro.run_dir)
    # Before `Done`, so a Terminal monitor (an export, say) sees the weights the handle will hold.
    restore_best!(nitro)
    set_phase!(nitro, Done())
    finish!(nitro.logger, nitro.stop_reason)
    return nitro
end

"""
    ReactantNitro.complete_micro_batch!(nitro, rec, tmv) -> Float64

The deferred half of one micro-batch of the automatic loop, run one micro-batch late, after the
next micro-batch's gradient program (and its step's optimizer program) has been dispatched: read the
loss back ([`check_finite`](@ref) under `check_divergence`, else [`read_loss`](@ref)), run a `:host`
`train_metrics` on the transferred outputs, emit the train line if the micro-batch closed a
logged optimizer step (see [`log_every_n_steps`](@ref)), and free its batch. Returns the host
loss, for the epoch mean.

`rec` is `(; l, b, batch, aux, e, step, epoch, closes, sched, extra)`, captured at dispatch.
`step` is the counter before that micro-batch's optimizer step: the divergence error names it, and
the train line is logged at `step + 1`, the step the micro-batch belonged to, whatever the counter
reads by now. `sched` holds the step's `schedule/` values when the line is due, else `(;)`.
`extra` is [`micro_grad!`](@ref)'s: under a [`PerturbRule`](@ref), reading pass 2's loss awaits
pass 2 before `b` and `ps_adv` are freed, and `sam/gap`, pass 2's loss minus pass 1's, joins the
train line.

Freeing the batch last is safe: the loss readback awaited the record's program, and no later
program reads its batch. Holding `aux` across the later dispatches is safe because no later program
donates it: the gradient program donates only `g_accum`, which the objective never reads, and the
optimizer program donates `ps` and the optimizer state. The exception is a `forward` or
`train_metrics` that returns a parameter array unchanged, which Reactant may hand back as the
input buffer itself.
"""
function complete_micro_batch!(nitro::Nitro, rec, tmv)
    # One D2H serves the divergence check, the epoch mean and the train line.
    lh = nitro.check_divergence ? check_finite(rec.l, rec.step, rec.epoch) : read_loss(rec.l)
    # A perturbation rule's pass 2 also reads `b` and `ps_adv`; this readback awaits it.
    gap = (;)
    if haskey(rec.extra, :l2)
        gap = NamedTuple{(Symbol("sam/gap"),)}((read_loss(rec.extra.l2) - lh,))
        free_batch!(rec.extra.ps_adv)
    end
    # The host path for `train_metrics`: `aux` is the primal, and the hook runs in ordinary Julia
    # on it. The transfer is the cost the user chose by asking. Every micro-batch, as the contract
    # says, though only a logged step's closing one is logged.
    stats = tmv isa Val{:host} ?
        call_host_hook(
            train_metrics, :train_metrics, nitro.routing.train_metrics,
            rec.batch, rec.e, host_tree(rec.aux)
        ) : rec.aux
    # Once per logged optimizer step, the cadence the metrics contract names. The stats are the
    # closing micro-batch's, unreduced. `finite_only` drops a non-finite loss when the check is off.
    rec.closes && step_line_due(nitro, rec.step + 1) && log_metrics!(
        nitro.logger, finite_only((; loss = lh, gap..., stats..., rec.sched...));
        step = rec.step + 1, epoch = rec.epoch, context = "train"
    )
    # Freed here, not when the compiled call returns: XLA is asynchronous and the executable holds
    # its inputs, but the readback above awaited this record's program, so `b`'s buffers are dead.
    # A use-after-free would surface as `AssertionError: buffer.buffer !== C_NULL`.
    free_batch!(rec.b)
    return lh
end

# ── Manual mode: the driver ─────────────────────────────────────────────────────────

"""
    ReactantNitro.manual_program(ev, model, ps, opt_state, st, batch, router)

Manual mode's traced entry: [`train_step`](@ref) compiled as the closure's program, once per run.
The batch is narrowed to the closure's own routed fields and handed to `train_step` as keywords
through the same `call_hook` machinery as `forward`, so the missing-keyword error names the closure.
"""
function manual_program(ev, model, ps, opt_state, st, batch, router)
    out = call_hook(train_step, :train_step, router, batch, ev, model, ps, opt_state, st)
    # Malformed returns pass through untouched for `check_train_step_return` to name.
    return out isa NamedTuple && haskey(out, :st) ? merge(out, (; st = program_out(out.st))) : out
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

Manual mode's driver. It shares the automatic loop's epoch skeleton (prefetch, checks, device
schedules, the finite check under `check_divergence`, logging, `data_wait_frac`, validation,
checkpointing, early stopping, phases, `request_stop!`) and replaces the step body with one call
to the user's [`train_step`](@ref) closure, compiled once per run. Each batch is one optimizer
step. The loss readback is synchronous, since the closure's `stats` may alias buffers the next
call donates. The driver rebuilds the scheduled rules before each call, checks the closure's
return, validates `loss`, stores `ps` and `st`, and re-attaches the rules it handed in via
`merge_rules`, since a `Leaf` cannot leave a compiled program on a mesh. A due train line carries
each varying `opt` and device schedule under `schedule/`.
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
                    device_raw = device_schedule_values(nitro.schedules, nitro.step + 1)
                    e_next = step_experiment(
                        nitro.e, nitro.schedules, nitro.step + 1; mesh = nitro.mesh,
                        raw = device_raw
                    )
                    if e_next !== nitro.e
                        nitro.e = e_next
                        ev = compile_view(e_next)
                    end
                    # Narrowed to the closure's routed fields, as `eval_forward` does for `forward`.
                    bf = NamedTuple{keys(router)}(b)
                    # The per-step host rebuild of scheduled rules, type-preserving so one program
                    # serves every step.
                    opt_raw = opt_schedule_values(nitro.schedules, nitro.step + 1)
                    opt_state = rebuild_scheduled_rules(nitro, nitro.step + 1; raw = opt_raw)
                    sched_line = logs_schedules(nitro) && step_line_due(nitro, nitro.step) ?
                        schedule_log(nitro.schedules, device_raw, opt_raw) : (;)
                    thunk = compile_cached(
                        manual_program, ev, ev, nitro.model, nitro.ps, opt_state, st,
                        bf, router; phase = GradCompiling(), worlds, nitro
                    )
                    out = check_train_step_return(
                        thunk(ev, nitro.model, nitro.ps, opt_state, st, bf, router)
                    )
                    # One D2H serves both the divergence check and the log line. Synchronous here,
                    # not lagged as in the automatic loop: the closure's `stats` are the user's and
                    # may alias the `ps` or optimizer state the next call donates.
                    lh = nitro.check_divergence ? check_finite(out.loss, nitro.step, nitro.epoch) :
                        read_loss(out.loss)
                    loss_sum += Float64(lh)
                    loss_n += 1
                    nitro.ps = out.ps
                    st = program_in(out.st)
                    # Re-attach the rules the closure was handed to the states it returned. The
                    # rules are not donated, so reuse across steps is safe.
                    nitro.opt_state = merge_rules(opt_state, out.opt_state)
                    # One train line per logged optimizer step, which in manual mode is one per
                    # batch (accum is fixed at 1 by construction).
                    step_line_due(nitro, nitro.step) && log_metrics!(
                        nitro.logger,
                        finite_only((; loss = lh, get(out, :stats, (;))..., sched_line...));
                        step = nitro.step, epoch = nitro.epoch, context = "train"
                    )
                    # Safe to free: the loss readback awaited the step.
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
            nitro.stop_requested || check_epoch_length(
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
            wrote_epoch = save_checkpoint_reported!(nitro, nitro.epoch, metrics_out) !== false

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
            own_selected_checkpoint(nitro.checkpointer, nitro.run_dir)
        catch
            nothing
        end
        set_phase!(nitro, Failed())
        finish!(nitro.logger, :error)
        rethrow()
    end
    nitro.stop_reason === nothing && (nitro.stop_reason = :completed)
    # The final rewrite, as in the automatic loop, gated on this call having written the final
    # epoch.
    wrote_epoch && save_checkpoint_reported!(nitro, nitro.epoch, last_metrics)
    nitro.elapsed = time() - t_started
    # Per entry point, not per epoch.
    progress_done!()
    # After the final rewrite above, so the winning entry is the one the manifest ends up holding.
    nitro.best_checkpoint = own_selected_checkpoint(nitro.checkpointer, nitro.run_dir)
    # Before `Done`, so a Terminal monitor (an export, say) sees the weights the handle will hold.
    restore_best!(nitro)
    set_phase!(nitro, Done())
    finish!(nitro.logger, nitro.stop_reason)
    return nitro
end

# One entry point rather than two so the inner `Nitro` construction does not publish `Repl` on its
# way out, a microsecond before the first training phase.
train!(e; kwargs...) = with_repl_result(() -> train!(Nitro(e; kwargs...)))

"""
    ReactantNitro.DATA_WAIT_WARN

The `data_wait_frac` above which [`report_data_wait!`](@ref) warns: `0.25`.
"""
const DATA_WAIT_WARN = 0.25

"""
    ReactantNitro.report_data_wait!(nitro, t_wait, t_step) -> nothing

Log the epoch's host-wait fraction `t_wait / (t_wait + t_step)` as `data_wait_frac` under
`context = "data"`, and warn (once per process) when it reaches [`DATA_WAIT_WARN`](@ref).

Other data-path checks catch specific mistakes; this one measures the result, so it also catches a
source that is too slow, a cache cap or a contended sample server. `t_wait` is the time blocked
pulling from the stream; `t_step` is the rest of the micro-batch body through the loss readback,
where the device work is awaited, plus the epoch-end drain. Both exclude each epoch's first
micro-batch. Since the device runs micro-batch k while the host pulls k+1, the fraction is an
upper bound on device idle time. It has its own context so the `"train"` and `"validate"` lines
keep their cadence.
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
    $(nitro.epoch) BLOCKED waiting for the data source. The device is idle for up to
    that fraction of the epoch (each wait overlaps one queued micro-batch), so this is up to a
    $(round(1 / max(1 - frac, 1.0e-3); digits = 1))x wall-clock penalty against a run whose loader keeps up.
    Resolved prefetch for the `train` split: $(pf.workers) worker(s), $(pf.device_batches) batch(es)
    on device, $(pf.host_batches) on host, path $(pf.path). $(
        (pf.path === :fanout || pf.path === :fanout_unordered) ?
            "The fan-out is active, so the source itself is the limit: profile it in a CPU gate by iterating `build_data(e, nothing).train` directly, and check for a decoded-cache cap or a contended sample server." :
            "There is no host-side concurrency here; see the prefetch warning at setup for the two methods that enable it."
    )""" maxlog = 1
    return nothing
end

"""
    ReactantNitro.rebuild_rules(nitro, layout, masks, anchors, memo = nothing; applied) -> NTuple{G}

Rebuild every group's rule chain for the next optimizer step. Each group's `applied` values (from
[`applied_hp`](@ref)) become device scalars in its `hp`, `build_chain` calls the factory, and
[`place_slots`](@ref) moves any value the factory computed from `hp` (a slot recorded by
[`schedule_bindings`](@ref)) back onto the device. Each chain goes into a `Leaf` beside the state
the optimizer owns.

The rules travel inside `opt_state`, so the optimizer program takes no separate `hp` argument. A
rule rebuilt with a fresh `ConcretePJRTNumber` has the same type and so re-enters the same compiled
thunk; the [`ScalarMemo`](@ref) avoids re-uploading a value that has not changed.
"""
function rebuild_rules(
        nitro::Nitro, layout::FlatLayout{G}, masks, anchors, memo = nothing;
        applied = applied_hp(nitro, layout)
    ) where {G}
    sched = nitro.schedules
    return ntuple(Val(G)) do gi
        g = layout.groups[gi]
        keep_decay = sched !== nothing && any(
            k -> _binds(sched, k, g) &&
                last(_schedule_target(k)) === :lambda, keys(sched.opt)
        )
        # `mesh` is required: without it the rule's scalars are placed for one device while the
        # moments are replicated across the mesh, and the program refuses to rebuild the `Leaf`.
        # Invisible on CPU, where every device count is 1.
        hp = resolve_hp(
            nitro.e, layout, gi; host = applied[gi], keep_decay, anchors, masks,
            mesh = nitro.mesh, memo
        )
        chain = place_slots(
            build_chain(nitro.e, g, hp), nitro.bindings.slots[gi], gi; memo, mesh = nitro.mesh
        )
        Optimisers.Leaf(chain, nitro.opt_state[gi].state, nitro.opt_state[gi].frozen)
    end
end

"""
    ReactantNitro.applied_hp(nitro, layout) -> NTuple{G, NamedTuple}

Each group's [`hp_host`](@ref) values for the upcoming optimizer step, `nitro.step + 1`: what
[`rebuild_rules`](@ref) applies and the schedule log reports.
"""
applied_hp(nitro::Nitro, layout::FlatLayout{G}) where {G} = ntuple(Val(G)) do gi
    g = layout.groups[gi]
    hp_host(nitro.e, g, group_schedule_values(nitro.schedules, g, nitro.step + 1))
end

# Whether the per-step lines are written at the logged step `step`; see `log_every_n_steps`.
step_line_due(nitro::Nitro, step::Integer) = step % nitro.log_every_n_steps == 0

# Whether the train line carries the schedule values; see `autolog_schedules`.
logs_schedules(nitro::Nitro) = nitro.autolog_schedules && nitro.schedules !== nothing

"""
    ReactantNitro.to_device_batch(batch, routing, mesh = nothing) -> NamedTuple

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

Read the loss back and raise, naming step and epoch, if it is not finite. There is no rollback;
recovery is a resume at a lower learning rate. The check also catches a failed `BufferToHost`,
which on this stack returns garbage without raising. Called only under
[`check_divergence`](@ref); the automatic loop calls it one micro-batch late (see
[`complete_micro_batch!`](@ref)), with the step and epoch captured at dispatch.
"""
function check_finite(l, step, epoch)
    v = read_loss(l)
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
    ReactantNitro.read_loss(l) -> Float64

The loss readback with no validation, for `check_divergence = false`: a non-finite value passes
through to the epoch mean and to `finite_only`, which drops it from the train line. So does a
failed `BufferToHost`, which on this stack returns garbage without raising; that is the cost of
turning the check off.
"""
read_loss(l) = l isa Number ? Float64(l) : Float64(only(Array(l)))

"""
    ReactantNitro.check_control_readback(v, what, name) -> Float64

The validated readback for scalars the framework branches on (the early-stopping metric and the
checkpoint selection metric): returns `Float64(v)` and raises if `v` is not a finite real. A
logged `NaN` costs a missing point on a chart; acting on one could stop a healthy run or keep a
stalled one going.
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
        A metric used for control flow must be a finite real number; a `:sum` or `:concat` metric
        that accumulates an array is for reporting, not for stopping."""
    )
    return x
end

"""
    ReactantNitro.finite_only(metrics) -> NamedTuple

Read `metrics` back to host values and drop every non-finite real, as the logging contract
promises backends. Dropped rather than zeroed or passed through: backends reject a `NaN` or plot
it as a gap, and at this point the framework cannot tell a diverged metric from a failed readback.
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
    ReactantNitro.save_checkpoint_reported!(nitro, epoch, metrics)

[`save_checkpoint!`](@ref) as one reported progress stretch, so a watcher can tell a slow write to
a remote filesystem from a hang. Nothing is reported without a checkpointer, since
`save_checkpoint!(::Nothing, ...)` is a no-op. Every write, the final rewrite included, uses the
label `"checkpoint"`. Returns what `save_checkpoint!` returned, so `false` means the epoch was
declined.
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
        derived = nitro.derived,
        early_stop = early_stop_state(nitro.early_stop),
    )
    # A structural walk over leaves, not elements: about a hundred checks per epoch.
    assert_host_record(snap)
    return snap
end

"""
    ReactantNitro.check_logger_state_serializable(state, lgr) -> nothing

Check that `logger_state` returned plain serializable data (a string, `NamedTuple`, `Dict`,
number, `Symbol` or `Tuple`), never the live backend object: the record goes through JLD2 and
outlives the process. Checked when the record is built, rather than failing at a resume in another
process.
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

The checkpoint-record surrogate for a `Reactant.ReactantRNG`, whose type parameter forbids host
contents. [`to_host`](@ref) stores the seed and algorithm here and [`from_host`](@ref) rebuilds the
RNG on restore. Lux layer state holds one at `st.<layer>.rng` for any `Dropout`.
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

Convert every device value in `x` to a host value, for a checkpoint record. The generic method
walks any struct's fields, not only containers, and rebuilds a struct only when a field changed,
so structs with inner constructors are left alone when they hold nothing on device. A struct that
holds a device value and cannot be rebuilt raises here, in the writing process. Types that cannot
hold host values at all get a surrogate (see [`HostRNG`](@ref)).
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

The inverse of [`to_host`](@ref) on the restore path: turn every surrogate back into the value it
stands for and leave everything else for `to_rarray`. A new surrogate needs only a `from_host`
method.
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
    ReactantNitro.ProgramRNG
    ReactantNitro.ProgramPair

How a `Reactant.ReactantRNG` and a `Pair` leave a compiled program. Reactant rebuilds a returned
struct with the type its trace recorded, which names one-device buffers, so a struct holding a
value replicated on a mesh fails `new` after the program has run. Untyped fields hold either;
[`program_out`](@ref) wraps inside the program and [`program_in`](@ref) unwraps after.
"""
struct ProgramRNG
    seed::Any
    algorithm::String
end

struct ProgramPair
    first::Any
    second::Any
end

"""
    ReactantNitro.program_out(x) -> x
    ReactantNitro.program_in(x) -> x

Every `ReactantRNG` and `Pair` in a program's returned outputs, layer state or metrics to its
surrogate ([`ProgramRNG`](@ref), [`ProgramPair`](@ref)) and back. Walks tuples and named tuples, which
is what Lux state, `forward` outputs and the metrics contract are made of. Any other struct still
cannot leave a program on a mesh.

Every array leaf also leaves as a fresh buffer: a lazy wrapper (`reshape`, `view`) is materialized,
and an `optimization_barrier` keeps a leaf that is an unchanged input from being returned as that
input. On a mesh, PJRT rebuilds either with one-device types (EnzymeAD/Reactant.jl#2989).
"""
program_out(r::Reactant.ReactantRNG) = ProgramRNG(program_out(r.seed), r.algorithm)
program_out(p::Pair) = ProgramPair(program_out(p.first), program_out(p.second))
program_out(x::Union{Tuple, NamedTuple}) = map(program_out, x)
program_out(a::Reactant.AnyTracedRArray) =
    only(Reactant.Ops.optimization_barrier(Reactant.TracedUtils.materialize_traced_array(a)))
program_out(x) = x
program_in(r::ProgramRNG) = Reactant.ReactantRNG(r.seed, r.algorithm)
program_in(p::ProgramPair) = program_in(p.first) => program_in(p.second)
program_in(x::Union{Tuple, NamedTuple}) = map(program_in, x)
program_in(x) = x

"""
    ReactantNitro.device_paths(x, path = "") -> Vector{String}

Every device-resident value reachable from `x`, as `path :: Type` strings, for the errors of
[`assert_host`](@ref) and [`assert_host_record`](@ref). The inverse check of
[`assert_device_state`](@ref).
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

On a CPU backend a "device" array is host memory and scalar indexing into one is legal, so a
residency mistake does not show up in a CPU test without this check. A failure means the
framework's conversion has a gap, not that the user forgot an `Array(...)`. It walks leaves, not
elements. [`assert_host_record`](@ref) is the same check at the checkpoint boundary; both use
[`device_paths`](@ref).
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

Assert that every value in a checkpoint record is a host value, when the record is built. JLD2
writes and reads a device value without complaint, and the failure surfaces later, in another
process, as `AssertionError: buffer.buffer !== C_NULL`, naming neither the field nor serialization.
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

The inverse of [`to_device_rule`](@ref): rebuild an optimizer rule, or each rule of a chain, with
its fields converted by [`to_host`](@ref), so the record does not hold the `ConcretePJRTNumber`s
the per-step rebuild put there.
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

Run the `:val` split and return the finalized metrics. The training loop runs the same pass each
epoch, and `validate` also works on a `Nitro` that has never trained. Eval mode throughout; `st_new` is
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
    ReactantNitro.fwd_program(ev, model, ps, st, batch, fwd_router, fwd_fn = forward) -> outputs

The eval-mode forward shared by `predict`, `validate` and `evaluate`. It always runs at
`batch_size`, since the framework pads before and slices after, so it compiles once. `st_new` is
discarded inside the program. It takes `forward`'s own router and a batch narrowed to `forward`'s
fields, so a `predict` batch without labels and a `validate` batch with them share one cache entry.
"""
function fwd_program(ev, model, ps, st, batch, fwd_router, fwd_fn = forward)
    outputs, _ = call_hook(fwd_fn, :forward, fwd_router, batch, ev, model, ps, st)
    return program_out(outputs)
end

"""
    ReactantNitro.eval_metric_program(ev, outputs, batch, router, ::Val{NREAL}, ::Val{HOOK}, fns = (;))

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
    m = HOOK === :metrics ?
        call_hook(hook_fn(fns, :metrics, metrics), :metrics, router, b, ev, o) :
        (; val_loss = call_hook(hook_fn(fns, :loss, loss), :loss, router, b, ev, o) => 1)
    return program_out(m)
end

"""
    ReactantNitro.eval_forward(nitro, ev, st, b, fwd_router) -> outputs

Compile or reuse [`fwd_program`](@ref) and call it, for both `predict` and the eval loop. The batch
is narrowed to `forward`'s fields, so a batch without labels hits the same cache entry.
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
    return program_in(
        thunk(
            ev, nitro.model, nitro.ps, st, bf, fwd_router,
            hook_fn(hook_fns(nitro.routing), :forward, forward)
        )
    )
end

"""
    ReactantNitro.run_eval(nitro, split; report = true, honor_stop = false) -> NamedTuple

The body of [`validate`](@ref) and [`evaluate`](@ref). `honor_stop = true`, which the standalone
entry points pass, breaks at the next batch boundary after [`request_stop!`](@ref); the training
loop's own per-epoch validation keeps `false`, so the validation after a requested stop runs in
full.

Per batch: pad to `batch_size`, transfer the routed fields, run the shared eval `forward`, compute
the metric on the real samples, accumulate each `value => mode` on the host, free the batch's device
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
                program_in(
                    thunk(ev, outputs, b, router, Val(n_real), Val(hook), hook_fns(nitro.routing))
                )
            else
                # `metrics` NEVER sees padding: the outputs are sliced back to `n_real`, and the
                # batch handed to it is the split's own host batch, which was never padded.
                call_host_hook(
                    metrics, :metrics, router, batch, e,
                    slice_outputs(host_tree(outputs), n_real, nitro.batch_size)
                )
            end
            acc = accumulate_metrics(acc, host_metrics(m, split, n_real), split)
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
        reduced = reduce_metrics(acc)
        assert_host(
            check_concat_reduced(acc, reduced, finalize_metrics(e, reduced, split), split),
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
The reduction modes a metric may name besides a count. `nothing` is the older spelling of `:sum`.
"""
const METRIC_MODES = (:sum, :max, :min, :concat)

"""
    ReactantNitro.MetricEntry(value, mode)

One `value => mode` of a metric, read back to the host. `mode` is a `Real` count or a `Symbol`
from `METRIC_MODES`; `value` is a leaf or a `NamedTuple`/`Tuple` the mode covers leaf by leaf.
"""
struct MetricEntry{V, M}
    value::V
    mode::M
end

"""
    ReactantNitro.host_metrics(m, split, n_real) -> NamedTuple of MetricEntry trees

Read one batch's metric result back to host values and check the metric contract: every value is
covered by exactly one `value => mode`, on itself or on a group above it. The older `(value, mode)`
tuple is accepted at the top of a key. A bare number is refused rather than misaccumulated.
"""
function host_metrics(m, split::Symbol, n_real::Integer)
    m isa NamedTuple || error(
        """
        ReactantNitro: `metrics` returned a `$(typeof(m))` for the `$split` split; it must return a
        NamedTuple of `value => mode` pairs, as in
        `(; err = sum_abs_err => n_items, cm = confusion => :sum)`."""
    )
    return NamedTuple{keys(m)}(
        map(keys(m)) do k
            v = getproperty(m, k)
            # The older spelling: a 2-tuple whose second element can only be a mode.
            v isa Tuple && length(v) == 2 && v[2] isa Union{Number, Symbol, Nothing} &&
                (v = v[1] => v[2])
            _parse_metric(v, string(k), split, n_real)
        end
    )
end

function _parse_metric(x, path::String, split::Symbol, n_real::Integer)
    if x isa Pair
        mode = _parse_mode(x.second, path, split)
        _has_pair(x.first) && error(
            """
            ReactantNitro: metric `$path` on the `$split` split has a mode, and so does a value
            inside it. Each value takes exactly one mode: put it on the group or on its values."""
        )
        value = host_tree(x.first)
        mode === :concat && _leafmap(v -> _check_concat_leaf(v, path, split, n_real), value)
        return MetricEntry(value, mode)
    elseif x isa NamedTuple
        return NamedTuple{keys(x)}(map(k -> _parse_metric(x[k], "$path.$k", split, n_real), keys(x)))
    elseif x isa Tuple
        return ntuple(i -> _parse_metric(x[i], "$path[$i]", split, n_real), length(x))
    end
    error(
        """
        ReactantNitro: metric `$path` on the `$split` split is a `$(typeof(x))` with no mode. Write
        `value => mode`, with a count (to divide by) or one of $(join(repr.(METRIC_MODES), ", ")),
        on the value or on a group above it. A ternary binds looser than `=>`, so
        `c ? a : b => :sum` gives `a` no mode; write `(c ? a : b) => :sum`."""
    )
end

_has_pair(x::Pair) = true
_has_pair(x::Union{NamedTuple, Tuple}) = any(_has_pair, x)
_has_pair(x) = false

function _parse_mode(c, path::String, split::Symbol)
    mode = c === nothing ? :sum : c isa Symbol ? c : host_tree(c)
    (mode isa Real || mode in METRIC_MODES) && return mode
    error(
        """
        ReactantNitro: metric `$path` on the `$split` split has the mode $(repr(mode)). A mode is a
        count, which divides the summed value at the end, or one of
        $(join(repr.(METRIC_MODES), ", ")). `nothing` is accepted as the older spelling of `:sum`."""
    )
end

# Within one entry, a mode applies leaf by leaf through `NamedTuple` and `Tuple` values; `.+`
# broadcasts over neither a `NamedTuple` nor a tuple of arrays the way a sum needs.
_leafmap(f, x::Union{NamedTuple, Tuple}, ys...) = map((a, bs...) -> _leafmap(f, a, bs...), x, ys...)
_leafmap(f, x, ys...) = f(x, ys...)

# Walks the groups of one key down to its entries, in step with trees of the same shape above them.
_entrymap(f, path, x::MetricEntry, ys...) = f(path, x, ys...)
function _entrymap(f, path, x::Union{NamedTuple, Tuple}, ys...)
    for y in ys
        (typeof(y) <: (x isa NamedTuple ? NamedTuple{keys(x)} : Tuple) && length(y) == length(x)) ||
            _shape_changed(path)
    end
    sub(i) = x isa NamedTuple ? "$path.$(keys(x)[i])" : "$path[$i]"
    vals = ntuple(i -> _entrymap(f, sub(i), x[i], map(y -> y[i], ys)...), length(x))
    return x isa NamedTuple ? NamedTuple{keys(x)}(vals) : vals
end
_entrymap(f, path, x, ys...) = _shape_changed(path)

_shape_changed(path) = error(
    """
    ReactantNitro: metric `$path` changed shape between batches. A metric's groups, and the mode
    of each value, must be the same on every batch."""
)

function _check_concat_leaf(x, path::String, split::Symbol, n_real::Integer)
    (x isa AbstractArray && ndims(x) >= 1 && size(x, ndims(x)) == n_real) && return nothing
    shape = x isa AbstractArray ? "an array of size $(size(x))" : "a `$(typeof(x))`"
    error(
        """
        ReactantNitro: metric `$path` on the `$split` split is `:concat`, and one of its values is
        $shape on a batch of $n_real samples.
        `:concat` appends along the LAST axis, which must hold one entry per sample, so that every
        `:concat` value lines up sample for sample. Move the sample axis last with `permutedims`,
        or use `:sum` for a value that is not per sample."""
    )
end

"""
    ReactantNitro.accumulate_metrics(acc, m, split) -> NamedTuple

Fold one batch's entries into the accumulator. The first batch fixes the key set, since a metric
reported on only some batches has no consistent denominator. `:concat` leaves are kept as a
vector of per-batch arrays and joined once in [`reduce_metrics`](@ref).
"""
function accumulate_metrics(acc, m::NamedTuple, split::Symbol)
    if acc === nothing
        return map(t -> _entrymap(_start_entry, "", t), m)
    end
    keys(acc) === keys(m) || error(
        """
        ReactantNitro: `metrics` returned $(keys(m)) on one batch of the `$split` split and
        $(keys(acc)) on an earlier one. The metric set must be the same for every batch, because the
        framework accumulates each metric across the split, and a metric present on only some
        batches has no honest denominator."""
    )
    return NamedTuple{keys(acc)}(
        map(k -> _entrymap(_add_entry, string(k), getproperty(acc, k), getproperty(m, k)), keys(acc))
    )
end

_start_entry(path, e::MetricEntry) =
    e.mode === :concat ? MetricEntry(_leafmap(x -> Any[x], e.value), :concat) : e

function _add_entry(path, a::MetricEntry, e)
    e isa MetricEntry || _shape_changed(path)
    (a.mode isa Real && e.mode isa Real) || a.mode === e.mode || error(
        """
        ReactantNitro: metric `$path` reported $(_mode_label(a.mode)) on one batch and
        $(_mode_label(e.mode)) on another. A value's mode decides how its batches combine, so it
        must be the same on every batch."""
    )
    a.mode === :concat &&
        return MetricEntry(_leafmap((p, x) -> _push_concat!(p, x, path), a.value, e.value), :concat)
    f = a.mode === :max ? max : a.mode === :min ? min : +
    return MetricEntry(_leafmap((p, x) -> f.(p, x), a.value, e.value), a.mode isa Real ? a.mode + e.mode : a.mode)
end

_mode_label(c) = c isa Real ? "a count" : repr(c)

function _push_concat!(parts::Vector{Any}, x, path::String)
    p = first(parts)
    (ndims(x) == ndims(p) && size(x)[1:(end - 1)] == size(p)[1:(end - 1)]) || error(
        """
        ReactantNitro: metric `$path` is `:concat`, and a value of size $(size(x)) follows one of
        size $(size(p)). Every axis but the last must match across batches, since the batches are
        joined along the last."""
    )
    return push!(parts, x)
end

"""
    ReactantNitro.reduce_metrics(acc) -> NamedTuple

The divide: `value / count` where the mode is a count, the joined array for `:concat`, and the bare
value otherwise. The result has the shape the hook returned, with the modes removed. It runs
before [`finalize_metrics`](@ref), so a finalizer such as `f1 = 2tp / (2tp + fp + fn)` receives
its `:sum` values `tp`, `fp` and `fn` as split totals.
"""
reduce_metrics(acc::NamedTuple) = map(t -> _entrymap((_, e) -> _finish(e), "", t), acc)

_finish(e::MetricEntry) =
    e.mode isa Real ? _leafmap(x -> x / e.mode, e.value) :
    e.mode === :concat ? _leafmap(_concat_last, e.value) : e.value

# One allocation for the whole split, rather than `cat` over thousands of splatted arguments.
function _concat_last(parts::Vector{Any})
    p = first(parts)
    N = ndims(p)
    out = similar(
        p, mapreduce(eltype, promote_type, parts), (size(p)[1:(N - 1)]..., sum(x -> size(x, N), parts))
    )
    o = 0
    for x in parts
        n = size(x, N)
        selectdim(out, N, (o + 1):(o + n)) .= x
        o += n
    end
    return out
end

"""
    ReactantNitro.check_concat_reduced(acc, reduced, out, split) -> out

Refuse a `:concat` array that reaches `finalize_metrics`' output unreduced. It would otherwise be
stored in History every epoch, sent to the logger, and written into every checkpoint record. An
identity check: it catches the default identity finalizer and plain pass-through.
"""
function check_concat_reduced(acc::NamedTuple, reduced::NamedTuple, out, split::Symbol)
    joined = IdDict{Any, String}()
    for k in keys(acc)
        _entrymap(string(k), getproperty(acc, k), getproperty(reduced, k)) do path, e, r
            e.mode === :concat && _leafmap(x -> (joined[x] = path; nothing), r)
            nothing
        end
    end
    isempty(joined) && return out
    _leafmap(out) do x
        path = get(joined, x, nothing)
        path === nothing || error(
            """
            ReactantNitro: `finalize_metrics` for the `$split` split returned the `:concat` array of
            metric `$path` unreduced. A `:concat` value holds one entry per sample of the split, so
            it is an input to `finalize_metrics` only: reduce it there. Per-sample outputs for
            analysis come from `predict`."""
        )
        nothing
    end
    return out
end

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

It walks plain structs as well as containers, and a subtree with nothing on device comes back
`===` what it was. Unlike [`to_host`](@ref) it uses no surrogate types, since a hook wants the
value; a `ReactantRNG` passes through unchanged.
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
