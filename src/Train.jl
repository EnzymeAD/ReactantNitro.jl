# Train.jl
#
# The training loop, the two compiled programs it drives, and the eval entry points.
#
# ── The three-line summary an implementer must not get wrong ───────────────────────
#
# What is conditional, and what only LOOKS conditional. These are different mechanisms and
# confusing them is how an implementation goes wrong:
#
#   | | Branch lives | Genuinely skipped? |
#   | Forward and backward | nowhere; unconditional | NO, and must not be. Every micro-batch carries
#   |                      |                        | different data; skipping one drops it |
#   | Accumulator reset    | traced `ifelse.` select| NO. Both arms evaluate and one is selected |
#   | Optimizer step       | HOST branch in driver  | YES. Not invoked at all on micro-steps 1..N-1 |
#
# THE OPTIMIZER MUST BE A SEPARATE PROGRAM PRECISELY BECAUSE A SELECT CANNOT SKIP A STATEFUL UPDATE.
# Fused and gated by an `ifelse.`, the update would be computed on every micro-step and N-1 results
# discarded, which is merely wasteful, but its STATE would advance unconditionally: a select cannot
# un-write `opt_state`, so Adam's moments would decay and its bias correction would step forward once
# per micro-batch instead of once per optimizer step.
#
# DO NOT REACH FOR `Optimisers.AccumGrad`, even though it traces. Measured: the Reactant
# extension returns `zero.(dx)` on non-boundary micro-steps, because under tracing it cannot return
# `nothing` the way the host method does. A zero gradient is not a skipped step: the base rule still
# runs, decaying Adam's moments every micro-batch. Against the Lightning reference,
# `OptimiserChain(AccumGrad(2), Adam(1f-2))` over four micro-batches diverges by 4.4e-3, with no
# error, and the equivalent HOST chain matches exactly, so a CPU test that does not go through
# `@compile` will not catch it.
#
# THERE IS NO FUSED PROGRAM, INCLUDING AT `accum = 1`, which is a deliberate divergence from an
# earlier framework: an implementer porting from that code will find its host branch on
# `accumulate_grad_batches` and copy it. The reason is uniformity. Accumulation is where this stack
# is easiest to get subtly and silently wrong, so `accum = 1` is not a special case with its own
# program, signature, and cache entry. The saving is real rather than dismissed; note that adding
# fusion would relink every trace-time constant the optimizer program reads, so the clip's
# residency and the fusion decision are coupled and neither should be revisited alone.

# ── The two programs ────────────────────────────────────────────────────────────────

"""
    ReactantNitro.grad_program(ev, model, ps, st, batch, g_accum::NTuple{G}, is_first, inv_n)
        -> (loss, g_accum_new::NTuple{G}, st_new, stats)

One per run, invoked once per micro-batch.

**Parameters are canonically the Lux tree** that `build_model` returned. `ps` crosses both program
boundaries as a tree; the gradient accumulator and `opt_state` cross **flat**, as `NTuple{G}` of
per-group buffers, and each program flattens on entry and unflattens on exit, inside the trace. This
is the residency an earlier framework shipped in production, it keeps `Duplicated(ps, dps)` in
exactly the form Lux's own Reactant extension verifies, and every other consumer of `ps` already
wants a tree. The flat-canonical alternative is cheaper on paper and is not worth taking unmeasured,
in the one place where a mistake is a wrong gradient.

**`ev` is `compile_view(e)` and never `e`.** Enzyme walks the whole `Const` argument, so
a dataset reachable from `e` would be traversed element-wise here, at O(n_train) per compile, with no
visible effect on the emitted graph.

**The accumulator reset is a broadcast SELECT:**

```julia
g_scaled    = flatten(dps) .* inv_n
g_accum_new = ifelse.(is_first .> 0f0, g_scaled, g_accum .+ g_scaled)
```

Three details, none cosmetic:

  * **`ifelse.`, not `ifelse`.** Reactant 0.2.270 defines scalar `ifelse` for traced *scalars* only;
    the accumulator is an array, so the non-broadcast form is a `MethodError` at trace time.
  * **`is_first` is a 1-element device array tested with `.> 0f0`, not a traced scalar `Bool`.**
    Reactant cannot round-trip a replicated scalar out of a compiled program on a multi-device mesh,
    though it can for arrays, so a scalar formulation works on one device and fails on several.
  * **A SELECT, not a multiply by zero.** They differ on non-finite input: a `g_accum` left holding
    `NaN` from a diverged group would survive a multiply and cannot survive a select.

`inv_n = 1/accum` is a **trace-time host constant**, which is why `accum` is in the compile cache
key. One gradient program serves every micro-step of a run; two runs at different `accum` are two
different programs.

**The Enzyme shadow is program-internal.** `dps` is allocated INSIDE the traced program on every
invocation, is neither an argument nor an output, and never persists:

```julia
dps = Enzyme.make_zero(ps)
_, (l, st_new, stats) = Enzyme.autodiff(..., Duplicated(ps, dps), ...)
g = flatten(dps)
```

Enzyme reverse **accumulates into** the shadow it is handed, so a `dps` hoisted out of the program
and reused, which is the natural thing to do for a program invoked millions of times, would add every
past micro-batch's gradient to the current one, on top of the deliberate accumulator. No error is
raised and the loss curve reads as a badly chosen learning rate. Under trace `make_zero` produces a
trace-time zero that XLA folds into the adjoint seeds, so per-invocation allocation costs nothing at
runtime and there is no efficiency argument for hoisting it. **[`Nitro`](@ref) deliberately holds
no `dps` field**, and the accumulation tests assert it.

**Verified** (Julia 1.12.6, Reactant 0.2.270, Enzyme 0.13.198, CPU): a program allocating
its own shadow this way, invoked three times on the same inputs with `is_first = 1`, returns
bitwise-identical accumulators and the analytically correct gradient, while the same program at
`is_first = 0` accumulates exactly 2x and 3x over three calls. Both halves matter: the second proves
the select was not simply stuck on the reset arm.

**The fourth result is `stats` under `:device` and the raw `outputs` under `:host`.** `TM` is
type-level, like `Val{CLIP}`, so the two are separate programs with separate cache entries rather
than one program carrying a runtime switch, and the `:device` graph is exactly the one this program
emitted before host residency existed. Host residency is the user asking for a full output batch per
micro-batch across the boundary, which the framework declines to make the default and makes
available on request; the driver calls `train_metrics` in ordinary Julia on the transferred outputs.
"""
function grad_program(
        ev, model, ps, st, batch, g_accum, is_first, inv_n, routers, lref,
        ::Val{TM} = Val(:device)
    ) where {TM}
    # ALLOCATED INSIDE THE PROGRAM, ON EVERY INVOCATION. Never an argument, never an output, never
    # persisted. Enzyme reverse ACCUMULATES INTO the shadow it is handed, so a hoisted, reused
    # `dps` would add every past micro-batch's gradient to the current one, on top of the
    # deliberate accumulator, with no error and a loss curve that reads as a bad learning rate.
    dps = Enzyme.make_zero(ps)
    _, (l, st_new, aux) = Enzyme.autodiff(
        Enzyme.set_abi(Enzyme.ReverseWithPrimal, Reactant.ReactantABI),
        Enzyme.Const(objective_wrapper), Enzyme.Duplicated,
        Enzyme.Const(ev), Enzyme.Const(model),
        Enzyme.Duplicated(ps, dps), Enzyme.Const(st), Enzyme.Const(batch),
        Enzyme.Const(routers), Enzyme.Const(Val(TM))
    )
    # `Base.eltype(x)(inv_n)`, not a bare `inv_n`. The scaling is `flatten(dps) .* inv_n` with
    # `inv_n = 1/accum`, and `1/1` in Julia is a Float64, so the bare form promotes a Float32
    # accumulator to Float64: the accumulator is then not a type fixed point, the thunk guard
    # rejects the SECOND call, and the run dies on step 2 with a message about argument types. Same
    # family as the rule-construction eltype rule, in a third place.
    g_scaled = map(x -> x .* Base.eltype(x)(inv_n), flatten(dps, resolve_layout(lref)))
    # A broadcast SELECT, not `ifelse` and not a multiply by zero:
    #   * scalar `ifelse` is defined for traced SCALARS only, so the non-broadcast form is a
    #     MethodError at trace time on an array accumulator;
    #   * `is_first` is a 1-element device array tested with `.> 0f0`, not a traced scalar Bool,
    #     because Reactant cannot round-trip a replicated scalar out of a compiled program on a
    #     multi-device mesh though it can for arrays;
    #   * a select and a multiply differ on non-finite input: a `g_accum` left holding NaN from a
    #     diverged group survives a multiply and cannot survive a select.
    g_new = map((s, a) -> ifelse.(is_first .> 0.0f0, s, a .+ s), g_scaled, g_accum)
    return l, g_new, st_new, aux
end

"""
    ReactantNitro.opt_program(ev, g_accum::NTuple{G}, ps, opt_state::NTuple{G}, hp)
        -> (ps_new, opt_state_new::NTuple{G})

One per run, invoked **only on the last micro-step**, from the driver's host branch, which is what
makes it genuinely skipped rather than merely discarded.

**Clipping lives at the top of this program**, on the fully accumulated `g_accum`, before any
`apply!`. The threshold is [`gradient_clip_norm`](@ref)`(ev)`, a trace-time host constant; at
its default `0` the clip emits no ops and the compiled program is identical to one from a run that
never configured clipping.

`hp` is the per-group hyperparameter carrier, holding the scheduled device scalars. **It is the
only argument whose values change from one optimizer step to the next without changing the program.**

`ev` is taken here too, even though this program is not differentiated, so the rule stays uniform:
every trace site sees the stripped view, and the dataset guard has one statement rather than a list
of exceptions.

**`g_accum` is the accumulator, not `accum`.** `accum` is the micro-batch count, a `Nitro` keyword;
the accumulator is a different object with a nearly identical name, and conflating them is how
"accumulation is keyed by nothing" gets written.

The residency property the programs tests assert: `typeof(ps_new) === typeof(ps)` across a step,
`ps` a tree on both sides, `opt_state` and the accumulator `NTuple{G}` on both sides, and no
recompile on the second step.

**It returns `(ps_new, states_new)`, not `(ps_new, opt_state_new)`.** The second value is the
per-group optimizer STATE, all arrays; the driver re-attaches the rules it passed in. See the
comment at the return for why a `Leaf` cannot leave a compiled program on a mesh.
"""
function opt_program(ev, g_accum, ps, opt_state, lref, ::Val{CLIP}) where {CLIP}
    layout = resolve_layout(lref)
    # Clipping at the TOP, on the fully accumulated gradient, before any `apply!`.
    # `CLIP` is a TYPE-LEVEL host constant, so at its default 0 the branch folds at trace time and
    # this emits no ops at all. As a value argument it would be promoted to a traced scalar, and
    # `threshold > 0` would then be a host branch on a traced Bool, which raises; that is the same
    # promotion ruled out for a second reason, that a traced 0 scales the gradient to zero norm and
    # yields NaN rather than meaning "off".
    g = clip_by_global_norm(g_accum, CLIP)
    p_flat = flatten(ps, layout)
    both = map(apply_group, opt_state, p_flat, g)
    # Return the STATE, never the `Leaf`. A `Leaf` carries its RULE, and a rule's
    # hyperparameters are device SCALARS; Reactant cannot reconstruct a replicated scalar as a
    # program OUTPUT on a multi-device mesh, though it can for arrays. So returning the leaf made
    # every multi-device run die in Reactant's output codegen, for EVERY rule including a stateless
    # `Descent`, whose `Leaf{Descent{ConcretePJRTNumber},Nothing}` still carries `eta`.
    #
    # Returning it was redundant as well as fatal: `rebuild_rules` reconstructs every rule
    # host-side at the top of the next step, so the rule that came back was thrown away
    # unread. The driver re-attaches the rules it passed in, which is where they were already
    # authoritative. On one device this changes nothing observable.
    return unflatten(map(last, both), ps, layout), map(t -> first(t).state, both)
end

"""
    ReactantNitro.objective_wrapper(ev, model, ps, st, batch)

The differentiated wrapper. Five parameters, five activity-wrapped arguments at the
`Enzyme.autodiff` call.

```julia
function objective_wrapper(ev, model, ps, st, batch)
    outputs, st_new = forward(ev, model, ps, st; R_FWD(batch)...)
    l     = loss(ev, outputs; R_LOSS(batch)...)
    stats = train_metrics(ev, outputs; R_STATS(batch)...)
    return (l,
            EnzymeCore.ignore_derivatives(st_new),
            EnzymeCore.ignore_derivatives(stats))
end
```

**The mechanism**, verified in Lux's Reactant extension: return a tuple with the loss first, pass
`Enzyme.Duplicated` as the **return activity** instead of `Active`, keep
`set_abi(ReverseWithPrimal, ReactantABI)`, and wrap every non-loss element in
`EnzymeCore.ignore_derivatives`.

**`ignore_derivatives` is load-bearing, not decorative.** Reactant maps a `Duplicated` return to an
`OUT` result and seeds *every* `OUT` with ones. For the scalar loss that is the correct `dL/dL = 1`;
for an unbarriered state output those ones flow into `dps` and corrupt the gradient **with no
error**. The programs tests cover it, and **the control matters as much as the test**: delete
`ignore_derivatives` and assert the test FAILS. Its naive form is vacuous, because on a stateless
model there are no array `OUT` results and the barrier is a no-op, so it must use a state-carrying
model.

Note it is `EnzymeCore.ignore_derivatives`, re-exported by Reactant but **not exported** from it, so
it must be qualified. Do not copy Lux's *other* aux mechanism, a `Ref` stash, which fails silently
under tracing because the `Ref` holds a traced handle that is never a program output.

`ev` is a `Const` **argument**, not a closure capture: its `Device` fields must be traced inputs,
and a capture would bake them and break the cache key. The routers, by contrast, are closure
captures, which is correct: they are genuinely trace-time constants and the cache key already
covers them.
"""
function objective_wrapper(ev, model, ps, st, batch, routers, ::Val{TM} = Val(:device)) where {TM}
    outputs, st_new = call_hook(forward, :forward, routers.forward, batch, ev, model, ps, st)
    l = call_hook(loss, :loss, routers.loss, batch, ev, outputs)
    # Under `:host` the hook is part of no program, so what leaves here is the PRIMAL, and
    # the driver calls `train_metrics` on it in ordinary Julia. `TM` is a type-level constant, so
    # this branch folds at trace time and neither arm's ops reach the other's graph.
    aux = TM === :host ? outputs :
        routers.train_metrics === nothing ? (;) :
        call_hook(train_metrics, :train_metrics, routers.train_metrics, batch, ev, outputs)
    # `ignore_derivatives` is LOAD-BEARING, not decorative: Reactant maps a `Duplicated` return to an
    # OUT result and seeds EVERY OUT with ones. For the scalar loss that is the correct dL/dL = 1;
    # for an unbarriered state output those ones flow into `dps` and corrupt the gradient with no
    # error. The suite carries a control that proves it can detect it.
    return (
        l,
        Enzyme.EnzymeCore.ignore_derivatives(st_new),
        Enzyme.EnzymeCore.ignore_derivatives(aux),
    )
end

# ── Mutating layer state ────────────────────────────────────────────────────────────
#
# `forward` returns `(outputs, st_new)`, so state threading is inherent to the contract. There is no
# `mutable_state` trait and no parallel implementation: state always threads, and for a stateless
# model `st_new` is `st`.
#
# Training numerics do not change: in train mode LuxLib's Reactant extension calls
# `batch_norm_training(x, γ, β)` and ignores the running statistics, so they are WRITE-ONLY during
# training and threading state changes only what is validated and exported. Do not generalize that to
# weight EMA or recurrent carries.
#
# Caveat, documented with no runtime warning: state updates once per layer APPLICATION, and a step
# may contain more applications than expected. Accumulation supplies them via micro-batches; a
# weight-tied layer or an ODE solver supplies them within a single forward. Running statistics
# therefore update N times per optimizer step on micro-batch statistics, matching PyTorch, which is
# what makes a ported model reproduce its reference numbers.
#
# Multi-device: per-device (unsynced) statistics only, matching PyTorch's default. `st_new` must
# leave the program carrying the same sharding annotation as the `st` that entered, via
# `Reactant.Ops.sharding_group(input; group_id)`, one id per state leaf derived from its position in
# the state tree, applied on the way in and the way out. (An earlier draft named
# `mark_same_sharding_group`, which DOES NOT EXIST in Reactant 0.2.270.) At 4 devices with global
# batch 32, BatchNorm normalizes over 8
# samples per device; the answer is GroupNorm or LayerNorm.
#
# Keep constant data out of `st`: a large fixed array there is now returned every step, and belongs
# in a `Device` field, converted once.
#
# Precision: Float32 storage, StableHLO dot precision left at its default so XLA may use TF32, no
# user-facing precision knob. Parameterize the element type in the flat buffer, optimizer state,
# masks, and the scheduled-scalar carrier anyway: nearly free now, expensive to retrofit.

# ── Manual mode: the backward helper ────────────────────────────────────────────

"""
    backward(f, ps_sub, consts...) -> (loss, grads)

One reverse-mode pass over an objective, w.r.t. one parameter subtree, for use inside a
[`train_step`](@ref) closure. `f(ps_sub, consts...)` returns the **scalar loss**; the returned
`grads` is a tree matching `ps_sub`, and everything in `consts...` is `Const`, so no gradient ever
flows into it.

**Every traced value the objective reads must be an argument, never a closure capture.** Measured:
a captured parameter subtree silently zeroed the gradient, with the primal loss correct, which is
the worst failure mode, no error at any point. A plain struct like the model, which holds no
arrays, is safe to capture; parameters, batches, and state go in `consts...`.

**The non-saturating GAN formulation falls out of the activities.** In the generator's backward,
the discriminator's parameters are a `Const` argument: the loss's adjoint flows through the
discriminator's forward **as a function of the fake data** and never into the discriminator's
parameters. No explicit stop-gradient op is needed.

**The mechanism** is the framework's own objective-wrapper pattern, verified by the same
measurement as `objective_wrapper`: the objective is wrapped in a named function returning a
one-tuple, the return activity is `Enzyme.Duplicated` rather than `Active`, and the wrapper is
passed `Const` with the objective as a `Const` argument. `dps = Enzyme.make_zero(ps_sub)` is
allocated inside this call on every invocation, never hoisted, so gradients never accumulate across
calls, which is the same discipline the gradient program follows.

The whole `train_step` body is one traced program, so `backward` is compiled as part of it; the
host-side analog is exactly what the framework's own tests use as their reference.
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

# The named wrapper behind [`backward`](@ref): a NAMED function with the objective passed as a
# Const ARGUMENT, exactly the shape Lux's Reactant extension and this framework's own
# `objective_wrapper` verify. The one-tuple is the Duplicated-return contract; the scalar
# loss is element one, and there is nothing else to barrier.
function _objective_wrapper(f, ps_sub, consts...)
    return (f(ps_sub, consts...),)
end

# ── Entry points ─────────────────────────────────────────────────────────────────────

"""
    train!(nitro) -> Nitro
    train!(e; kwargs...) -> Nitro

Train. `train!` **blocks** and returns the [`Nitro`](@ref).

**`train!(nitro)` takes no keywords**; every keyword belongs to the `Nitro` constructor, and
`train!(e; kwargs...)` is pure sugar for `train!(Nitro(e; kwargs...))`. Three ways to reach the
handle: the return value, `run_ref::Ref{Nitro}` filled before the loop starts, or `info.nitro`
inside a phase monitor.

Re-training one `Nitro` under a different stopping rule means constructing another one, which is
cheap: the compile cache is module-level, so a second `Nitro` over the same experiment reuses every
compiled program.

On **divergence**: there is **no non-finite rollback**. A non-finite loss stops the run with an
error naming step and epoch, at whatever cadence the loss is already read. Recovery is resume from
the last checkpoint with a lower learning rate. The read itself needs validating because this
stack has a documented bug where a failed `BufferToHost` readback returns **garbage without
raising**, so the framework validates every scalar readback it uses for control flow (the loss, the
checkpoint metric, the early-stopping metric) and lets purely-logged metrics through unvalidated.
That asymmetry is deliberate.

**Ctrl+C is a graceful stop.** The loop runs on a worker thread when one is available, and ^C
interrupts the parked caller rather than the run; this entry point turns it into
[`request_stop!`](@ref): the step loop breaks at its next boundary, the epoch's validation and
checkpoint still run, and the call returns the finished `Nitro` with `stop_reason = :requested`,
exactly the `nitro_stop` wind-down. A run that fails surfaces its own exception, not a task
wrapper.
"""
train!(nitro::Nitro) = with_repl(() -> _train!(nitro), nitro; on_interrupt = _stop_on_interrupt)

# `with_repl`'s Ctrl+C hook for the training entry point. SIGINT is delivered into the parked
# caller on the interactive thread, never into the worker the loop runs on, so an interrupt is
# converted into the framework's own graceful stop, the same `nitro_stop` wind-down: the step
# loop breaks at its next boundary, validation runs, the checkpoint is written, and the call
# returns the finished `Nitro` with `stop_reason = :requested`. A failure during the wind-down
# surfaces its own exception rather than the interrupt.
function _stop_on_interrupt(t::Task, nitro::Nitro)
    request_stop!(nitro)
    try
        return fetch(t)
    catch e
        throw(unwrap_task_exc(e))
    end
end

# `with_repl`'s Ctrl+C hook for the eval entry points: stop the split at its next batch boundary
# (the `honor_stop` the entry points pass to `run_eval`), then surface the interrupt exactly as an
# inline abort would have. The flag is reset afterwards, because this stop was for THIS eval only:
# leaving it set would make a later `train!` on the same handle stop at its first step.
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
    # The driver is FROZEN at construction. Manual mode owns the step body (forwards,
    # backwards, optimizer steps, device metrics) and shares the epoch skeleton: batching, prefetch,
    # timing, validation, checkpointing, early stopping, phases. It does not share the accumulation
    # machinery, which is exactly what the closure replaces.
    get(nitro.frozen, :manual, false) && return _train_manual!(nitro)
    # The constructor's keywords were resolved at construction and every read since goes to a
    # field, so a revised `accum`/`max_epochs` accessor is inert here. Say so, once, before anything
    # expensive. The world-closure guard runs BEFORE the report: it poisons module entries whose
    # dependency closures moved since compile (so NEW `Nitro`s miss and rebuild against current
    # dispatch), and
    # the report's staleness section reads the scan's result to tell an existing `Nitro` that its
    # own programs are among the poisoned and that it keeps them, now stale.
    world_closure_staleness()
    report_fixed_config(nitro, :train)
    ev = compile_view(nitro.e)
    layout = nitro.layout
    inv_n = one(Float32) / nitro.accum
    # Replicated, like every other carrier. These are 1-element arrays rather than traced
    # scalars precisely because a replicated SCALAR cannot round-trip out of a compiled program on
    # a multi-device mesh though an array can, so the array form is what makes the mesh work at all.
    is_first = place_replicated(Float32[1], nitro.mesh)
    is_next = place_replicated(Float32[0], nitro.mesh)
    # FROZEN AT CONSTRUCTION, not resolved here. These were the last two components of the
    # compile-cache key that came from live dispatch, so this is where an existing handle could
    # once notice a redefinition and silently recompile between two `train!` calls. The two are
    # not interchangeable: the rules' `apply!` worlds are in `worlds_opt` alone, which keeps a rule
    # edit off the expensive gradient program. `stale_hooks` is what tells the user instead, through
    # the report `report_fixed_config` just printed.
    worlds = nitro.frozen.worlds_train
    worlds_opt = nitro.frozen.worlds_opt
    # Read, not rebuilt. `no_decay` is a user hook, so recomputing the mask here would let a
    # revision take effect on an existing handle, which is exactly what freezing forbids.
    masks, anchors = nitro.decay.masks, nitro.decay.anchors
    # The one-slot scalar memo, created here so it lives exactly as long as this call and inherits
    # `nitro.mesh` as an invariant rather than as a key. A constant `eta` crosses to the device once
    # per run instead of once per group per optimizer step; a scheduled one misses on every step and
    # costs what it always cost.
    scalar_memo = ScalarMemo()
    lref = layout_ref(layout)
    clipv = Val(nitro.gradient_clip_norm)
    # An experiment with no `train_metrics` method stays on the `:device` program whatever
    # `metrics_residency` says, so the graph and the cache entry are identical to a run from before
    # host residency existed. There is nothing to move to the host when the hook does not exist.
    # That rule is applied once, at construction, and this reads its answer.
    tmv = Val(nitro.frozen.tm_residency)
    # The per-run copy of the module-level registry, taken here rather than at construction, so
    # that a monitor registered between `Nitro(e)` and `train!` is in this run.
    adopt_monitors!(nitro)
    t_started = time()
    last_metrics = (;)
    # The final checkpoint rewrite below applies to the checkpoint THIS CALL wrote, and there may be
    # none. `nitro.epoch > 0` is not that condition: a resume restores the epoch counter, so a
    # `train!` that exits the loop without an iteration passes it.
    wrote_epoch = false

    try
        while nitro.epoch < nitro.max_epochs
            nitro.epoch += 1
            set_phase!(nitro, TrainStepping())
            st = Lux.trainmode(nitro.st)
            seen = 0
            # The per-epoch host-wait accounting, which is the generic guard on the whole data
            # path. Two `time()` calls per micro-batch, about 25 ns each, so under 50 us across an
            # epoch of 500. It catches EVERY variant of "the loader is the bottleneck", including the
            # ones no default prevents: a wrapper present but the source too slow, a cache cap, a
            # sample server contending with something else. The regression that motivated it would
            # have read about 0.8 here on its first epoch.
            t_wait = 0.0
            t_step = 0.0
            # One loop body for all three paths. Inline it is a lazy generator transferring on
            # this task; with one producer it is a depth-N channel; with the fan-out it is N workers, a
            # transfer task, and the same channel. The `finally` is not optional: `request_stop!`, a
            # non-finite loss, and any error mid-epoch all leave the loop with a full channel behind
            # it, and closing it is what stops the producers.
            stream = batch_stream(nitro.data.train, nitro.routing, nitro.mesh)
            # One bar per EPOCH, not per run: "steps left in this epoch" is the number a person
            # watching a run actually wants, and the epoch position goes in the label as text.
            # `div` is exact here because setup checked `length(train) % accum == 0` to resolve the
            # schedule horizon, so this is the same arithmetic `total` was built from.
            progress_begin!(
                "train", _epoch_steps(nitro), nitro.epoch, nitro.max_epochs
            )
            try
                # An explicit `iterate` loop rather than `for`, for one reason: the pull is what has to
                # be timed, and `for` gives nowhere to put the clock around it. Everything inside is
                # what the `for` body was.
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
                        # The EXPERIMENT half of the per-step rebuild. It has to
                        # happen here rather than beside `rebuild_rules` below: the scheduled fields
                        # are read by `loss` and `forward`, which run in the GRADIENT program, and
                        # by the time the optimizer step runs this step's gradient is already
                        # accumulated. Guarded on `micro == 0` so every micro-batch of one optimizer
                        # step sees one set of values, which is what makes accumulation equivalent
                        # to a larger batch.
                        #
                        # `nitro.step + 1` is the UPCOMING step, matching `rebuild_rules` exactly.
                        # `step_experiment` returns `e` unchanged when nothing device-side is
                        # scheduled, so the common run pays one `isempty` per optimizer step.
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
                    # ── SEPARABLE EDIT, revert this one line and this comment together ──────
                    #
                    # The batch free, on the beat that makes it safe. `free_batch!`'s own note
                    # says a batch handed to a compiled program cannot be freed when the call
                    # returns, because XLA is asynchronous and the executable holds its inputs. The
                    # line ABOVE is what changes that: reading the loss back awaits the execution, so
                    # by here the step is complete and `b`'s buffers are dead. Without this the loop
                    # drops roughly one batch of device memory per micro-batch to the GC on EVERY
                    # path, including the prefetch one, which frees only what is left in its channel
                    # at teardown.
                    #
                    # An earlier framework did exactly this, on this same Reactant pin, for the
                    # whole life of its reference runs: read the loss back to host, then free the
                    # batch. That is the evidence it is safe here, and it is evidence rather than
                    # proof: a use-after-free surfaces as the
                    # `AssertionError: buffer.buffer !== C_NULL` that nulling the pointer exists to
                    # make loud, so this wants one GPU run before it is trusted.
                    free_batch!(b)
                    # ── end separable edit ─────────────────────────────────────────────────
                    # The host path for `train_metrics`: `aux` is the primal, and the hook runs
                    # in ordinary Julia on it. The transfer is the cost the user chose by asking.
                    stats = tmv isa Val{:host} ?
                        call_host_hook(
                            train_metrics, :train_metrics, nitro.routing.train_metrics,
                            batch, nitro.e, host_tree(aux)
                        ) : aux

                    if micro == nitro.accum - 1
                        # A HOST branch, which is the only construct that GENUINELY skips.
                        # Gated by a traced select, the update would be computed every micro-step
                        # and its STATE would advance unconditionally, so Adam's moments would decay
                        # and its bias correction step forward once per micro-batch.
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
                        # Re-attach the rules the program was handed. `state` is what
                        # `rebuild_rules` just built, so its rules are this step's, and the program
                        # returns only the moments. `nitro.opt_state` stays a proper NTuple of
                        # `Leaf`s, which the checkpoint path and the residency test require.
                        nitro.opt_state = map(
                            (l, s) -> Optimisers.Leaf(l.rule, s, l.frozen), state, new_states
                        )
                        nitro.step += 1
                        # One completed unit of work, for a monitor measuring time since progress.
                        note_progress!()
                        # Train metrics carry `step`, validation metrics carry `epoch`, and both
                        # carry the other so the two overlay in a backend. Logged once per OPTIMIZER
                        # step rather than per micro-batch, which is the cadence the metrics
                        # contract names and the one the step counter counts. The stats are the
                        # closing micro-batch's, unreduced: the contract's reduction is none, and
                        # averaging the group would be a reduction the user did not ask for.
                        log_metrics!(
                            nitro.logger, finite_only((; loss = lh, stats...));
                            step = nitro.step, epoch = nitro.epoch, context = "train"
                        )
                    end
                    # The step half ends AFTER `check_finite`, not after `gthunk`: the compiled call
                    # returns before the device is done and only the loss readback awaits it, so
                    # timing `gthunk` alone would report a near-zero step and a `data_wait_frac`
                    # close to 1 on every healthy run.
                    #
                    # Micro-batch 1 is excluded from both halves. It carries the fan-out's task
                    # spin-up, which is one full batch build with no overlap available, and including
                    # it would fire the warning below on a short epoch.
                    seen > 1 && (t_step += time() - t_body)
                    nitro.stop_requested && break
                    t0 = time()
                    it = iterate(stream, st_stream)
                    seen > 1 && (t_wait += time() - t0)
                end
                # The exactly-once ledger, and only on an epoch that ran to completion: an early
                # exit leaves it legitimately partial. `check_epoch_length` below catches a SHORT
                # epoch; this catches the right count delivered with one index twice and another
                # never, which no count can see.
                nitro.stop_requested || check_prefetch_delivery(stream)
            finally
                close_stream!(stream)
                # In the `finally` with the stream: a non-finite loss, a `request_stop!`, and an
                # error all leave the epoch early, and a bar left open would sit on the terminal
                # underneath whatever the run printed next.
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
            # On the handle too, not only in this local: the local exists for the final checkpoint
            # rewrite and dies with the call, and the handle is what a REPL is holding afterwards.
            nitro.last_metrics = metrics_out
            isempty(metrics_out) || log_metrics!(
                nitro.logger, finite_only(metrics_out);
                step = nitro.step, epoch = nitro.epoch,
                context = "validate"
            )
            set_phase!(nitro, Checkpointing())
            save_checkpoint_reported!(nitro, "checkpoint", nitro.epoch, metrics_out)
            wrote_epoch = true

            # Both stopping routes set the same flag and are checked once per epoch, AFTER
            # validation and after the checkpoint. Stopping is graceful by construction: the epoch
            # has finished, validation has run, the checkpoint is written, and the exit is through
            # the normal `Done` path. `should_stop` is called even when a stop is already
            # requested, so a policy tracking its own state stays consistent with the epochs the run
            # actually saw.
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
        # A failed run still wrote epochs, and which one survived is the first thing asked of it.
        # Through `try`, because a run that died on I/O is exactly the run whose manifest may not
        # be readable, and a display helper must not replace the real exception with its own.
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
    # Stopping is RECORDED as `stop_reason` in the checkpoint record, and a checkpoint is written
    # per epoch, BEFORE the run knows how it ended, so every record would otherwise carry `nothing`
    # there. The final epoch's checkpoint is rewritten once the outcome is known: same epoch, same
    # filename, one manifest entry replaced rather than a second file. That is what makes a resume
    # able to say "this run ended at epoch 37 on patience" instead of exiting silently.
    #
    # GATED ON THIS CALL HAVING WRITTEN AN EPOCH, not on the epoch counter (found by the defaults
    # audit exercising `resume = :auto` against `max_epochs = 1`). A resume into a finished
    # run restores `epoch` and then exits the loop immediately, so the old `nitro.epoch > 0` fired
    # with `last_metrics` still `(;)` and REWROTE the previous process's final record with empty
    # metrics: the recorded `val_loss` was erased and the manifest entry's score became `nothing`,
    # dropping that epoch out of the top-K ranking. Silent, on the default path, and it needs only a
    # second `train!` in the same directory. There is nothing of this call's to rewrite when this
    # call wrote nothing; the record already carries the outcome of the process that did.
    # LABELLED APART from the per-epoch writes, because it immediately follows the last one and
    # two identical `checkpoint` stretches back to back read as a stutter rather than as the two
    # different writes they are: the epoch's own, and this rewrite carrying the final metrics.
    wrote_epoch && save_checkpoint_reported!(nitro, "final checkpoint", nitro.epoch, last_metrics)
    # `Done` for both stopping routes: exiting through the normal Done path is what makes an early
    # stop a finished run rather than a failure. `stop_reason` is where the difference lives, and
    # the checkpoint record keeps it, since "completed 40/40" and "stopped at 37 on patience" are
    # different outcomes that a resume with `:auto` would otherwise have to guess at.
    nitro.elapsed = time() - t_started
    # The last stretch of this entry point is over, so a reporter reusing one terminal line can
    # close it. Per entry point, not per epoch: an epoch is followed by a validation pass and then
    # by the next epoch, and only here is it known that nothing follows.
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

The return-contract check for [`train_step`](@ref), run once per optimizer step: it must be a
`NamedTuple` carrying `loss`, `ps`, `st`, and `opt_state` (with `stats` defaulting to `(;)`).
Checked loudly rather than indexed blindly, because a misspelled key would otherwise surface as a
`KeyError` naming nothing about the contract, or worse as a silently defaulted `stats`.
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

Manual mode's driver: the automatic loop's epoch skeleton with the step body replaced by one call
to the user's [`train_step`](@ref) closure, compiled once per run.

What is shared, unchanged: `batch_stream` and prefetch, per-batch schema and shape checks,
`step_experiment` for device schedules, the loss readback and the fail-fast finite check,
`free_batch!` after the readback, per-step logging of `(; loss, stats...)`,
`data_wait_frac` accounting, validation, checkpointing, early stopping, phases, and
`request_stop!`.

What is the closure's, per step: forwards, backwards, optimizer steps, and device metrics. The
closure returns `(; loss, ps, st, opt_state, stats)`; the driver checks the contract, validates
`loss`, stores `ps` and `st`, and re-attaches the rules it handed in to the returned states,
because a `Leaf` cannot leave a compiled program on a mesh, via `merge_rules`.
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
            # Per-epoch host-wait accounting, identical to the automatic loop.
            t_wait = 0.0
            t_step = 0.0
            stream = batch_stream(nitro.data.train, nitro.routing, nitro.mesh)
            # One bar per EPOCH, not per run: "steps left in this epoch" is the number a person
            # watching a run actually wants, and the epoch position goes in the label as text.
            # `div` is exact here because setup checked `length(train) % accum == 0` to resolve the
            # schedule horizon, so this is the same arithmetic `total` was built from.
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
                    # The experiment side of the per-step rebuild: device schedules write `e`'s
                    # fields, and the closure reads them as traced inputs, so it never recompiles.
                    e_next = step_experiment(
                        nitro.e, nitro.schedules, nitro.step + 1; mesh = nitro.mesh
                    )
                    if e_next !== nitro.e
                        nitro.e = e_next
                        ev = compile_view(e_next)
                    end
                    # The batch is narrowed to the closure's own routed fields, exactly as
                    # `eval_forward` narrows for `forward`, so a field only the closure reads is
                    # transferred and routed only when it declares it.
                    bf = NamedTuple{keys(router)}(b)
                    # The per-step HOST rebuild of the scheduled rules, the manual-mode
                    # counterpart of the automatic loop's `rebuild_rules`. Runs before every call
                    # so the closure steps with THIS step's values; it is TYPE-preserving (fresh
                    # scalar values, same rule types), so the same program serves every step.
                    opt_state = rebuild_scheduled_rules(nitro, nitro.step + 1)
                    thunk = compile_cached(
                        manual_program, ev, ev, nitro.model, nitro.ps, opt_state, st,
                        bf, router; phase = GradCompiling(), worlds, nitro
                    )
                    out = check_train_step_return(
                        thunk(ev, nitro.model, nitro.ps, opt_state, st, bf, router)
                    )
                    # The divergence check, on the readback that also feeds the log line, so the
                    # D2H is paid once for one number, exactly as in the automatic loop.
                    lh = check_finite(out.loss, nitro.step, nitro.epoch)
                    nitro.ps = out.ps
                    st = out.st
                    # Rules in, states out: re-attach the rules the closure was handed (which are
                    # THIS step's rebuilt rules when schedules exist) to the states it returned
                    # The rules were not donated (measured), so reusing them across
                    # steps is safe.
                    nitro.opt_state = merge_rules(opt_state, out.opt_state)
                    # One train line per optimizer step, which in manual mode is one per
                    # batch (accum is fixed at 1 by construction).
                    log_metrics!(
                        nitro.logger,
                        finite_only((; loss = lh, get(out, :stats, (;))...));
                        step = nitro.step, epoch = nitro.epoch, context = "train"
                    )
                    # The batch free, on the beat that makes it safe: `check_finite` read the loss
                    # back, so the step is complete and `b`'s buffers are dead.
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
                close_stream!(stream)
                # In the `finally` with the stream: a non-finite loss, a `request_stop!`, and an
                # error all leave the epoch early, and a bar left open would sit on the terminal
                # underneath whatever the run printed next.
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
            # On the handle too, not only in this local: the local exists for the final checkpoint
            # rewrite and dies with the call, and the handle is what a REPL is holding afterwards.
            nitro.last_metrics = metrics_out
            isempty(metrics_out) || log_metrics!(
                nitro.logger, finite_only(metrics_out);
                step = nitro.step, epoch = nitro.epoch,
                context = "validate"
            )
            set_phase!(nitro, Checkpointing())
            save_checkpoint_reported!(nitro, "checkpoint", nitro.epoch, metrics_out)
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
        # A failed run still wrote epochs, and which one survived is the first thing asked of it.
        # Through `try`, because a run that died on I/O is exactly the run whose manifest may not
        # be readable, and a display helper must not replace the real exception with its own.
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
    # The final rewrite, identical to the automatic loop: the last epoch's record is rewritten
    # once the outcome is known, gated on this call having written an epoch at all.
    # LABELLED APART from the per-epoch writes, because it immediately follows the last one and
    # two identical `checkpoint` stretches back to back read as a stutter rather than as the two
    # different writes they are: the epoch's own, and this rewrite carrying the final metrics.
    wrote_epoch && save_checkpoint_reported!(nitro, "final checkpoint", nitro.epoch, last_metrics)
    nitro.elapsed = time() - t_started
    # The last stretch of this entry point is over, so a reporter reusing one terminal line can
    # close it. Per entry point, not per epoch: an epoch is followed by a validation pass and then
    # by the next epoch, and only here is it known that nothing follows.
    progress_done!()
    # After the final rewrite above, so the winning entry is the one the manifest ends up holding.
    nitro.best_checkpoint = selected_checkpoint(nitro.checkpointer, nitro.run_dir)
    set_phase!(nitro, Done())
    finish!(nitro.logger, nitro.stop_reason === :completed ? :completed : :early_stop)
    return nitro
end

# Wrapped as one entry point rather than two so the `Nitro` construction inside does not
# publish `Repl` on its way out, only to be superseded a microsecond later by the first
# training phase. Both inner calls nest under this depth and stay quiet.
train!(e; kwargs...) = with_repl_result(() -> train!(Nitro(e; kwargs...)))

"""
    ReactantNitro.DATA_WAIT_WARN

The `data_wait_frac` above which [`report_data_wait!`](@ref) warns. Roughly a quarter: below that the
loader is comfortably ahead of the device and the remainder is jitter, and above it the run is paying
for host work that a worker count or a faster source would hide.
"""
const DATA_WAIT_WARN = 0.25

"""
    ReactantNitro.report_data_wait!(nitro, t_wait, t_step) -> nothing

The per-epoch host-wait fraction: `t_wait / (t_wait + t_step)`, logged as `data_wait_frac` and
warned on past [`DATA_WAIT_WARN`](@ref).

**This is the generic guard, and it is the one thing that would have caught the 4.2x regression on day
one.** Every other check here is about a specific mistake; this one measures the outcome, so it catches
the variants no default prevents: a prefetch wrapper present but the source too slow, a decoded-cache
cap, a sample server contending with another user, a model whose host augmentation genuinely costs more
than its step. It converts a silent several-fold slowdown into a number in the metrics and a line in
the log.

`t_wait` is time blocked pulling from [`batch_stream`](@ref) and `t_step` is everything else in the
micro-batch body, measured through the loss readback because that is where the device work is actually
awaited. Both exclude each epoch's first micro-batch, which carries task spin-up.

Logged under `context = "data"` rather than `"train"`, and the logging contract names that third
context. The two established ones are each pinned to a cadence this would falsify: a `"train"` line
is exactly one per optimizer step and a `"validate"` line exactly one per epoch of metrics, and the
test suite asserts both counts. Carrying both axes keeps it overlayable against either series in a
backend.

`maxlog = 1` rather than a flag on the handle: one warning per process is what a reader needs, and a
training process is one run.
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

The per-optimizer-step rebuild, on the optimizer side: evaluate the schedules at this step, wrap
the results as device scalars, rebuild each group's rule chain, and put it back in its `Leaf`
alongside the state the optimizer already owns.

**Each group's `eta_t` is its own path key if one is configured, else the bare `eta` key, else
nothing.** The per-group lookup inside the group loop is the whole of the path-bound mechanism on
this side: the ratio math lives in `effective_lr`, which this only feeds.

**This is why `hp` is not a separate argument to the optimizer program.** The driver rebuilds the
optimizer rules every step, and a rebuilt rule travels inside `opt_state`, so passing the same
scheduled scalars a second time as `hp` would be two transports for one set of values.

**Verified, and re-verified by the cache tests:** rebuilding a rule each step with a fresh
`ConcretePJRTNumber` re-enters the **same** compiled thunk, because the rule's TYPE is
unchanged. The same holds for the memoized scalar, and for the same reason: a reused
`ConcretePJRTNumber` has the type a fresh one has, so [`ScalarMemo`](@ref) changes which BUFFER the
rule carries and nothing about the compile-cache key.

`memo` is the run's [`ScalarMemo`](@ref), or `nothing` to upload fresh every step. It is a positional
argument with a default so the two `train!`-adjacent call sites read the same as they did.
"""
function rebuild_rules(
        nitro::Nitro, layout::FlatLayout{G}, masks, anchors, memo = nothing
    ) where {G}
    sched = nitro.schedules
    step = nitro.step + 1
    return ntuple(Val(G)) do gi
        # Each group's `eta_t` is its own path key (`group.eta`) if one is configured, else
        # the bare `eta` key, else nothing (the base rate). The per-group lookup is the whole of
        # the path-bound mechanism on this side; the ratio math lives in `effective_lr`, untouched.
        eta_t = if sched === nothing
            nothing
        else
            pk = Symbol(layout.groups[gi], ".eta")
            haskey(sched.opt, pk) ? sched.opt[pk](step) :
                haskey(sched.opt, :eta) ? sched.opt.eta(step) : nothing
        end
        # `mesh` is not optional here. Without it the rebuilt rule's scalars are placed
        # with a device count of 1 while `opt_state`'s moments are replicated across the mesh, and
        # the optimizer program then refuses to reconstruct its `Leaf` because the two halves of
        # one leaf disagree about how many devices they live on. Found by the first 4-device run;
        # invisible on CPU, where every device count is 1 and the mismatch cannot exist.
        hp = resolve_hp(nitro.e, layout, gi; eta_t, anchors, masks, mesh = nitro.mesh, memo)
        Optimisers.Leaf(
            build_chain(nitro.e, layout.groups[gi], hp),
            nitro.opt_state[gi].state, nitro.opt_state[gi].frozen
        )
    end
end

"""
    ReactantNitro.to_device_batch(batch, routing) -> NamedTuple

Transfer **only the routed fields** to device, by routing rule 1. A field no hook declares reaches
nobody, so transferring it would cost bandwidth for something nothing reads, and would fail outright
for the `case_id`-style bookkeeping rule 1 explicitly permits.
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

**No non-finite rollback**: a non-finite loss stops the run with an error naming step and epoch.
Recovery is resume from the last checkpoint with a lower learning rate.

The readback itself is validated, because this stack has a documented bug where a failed
`BufferToHost` returns **garbage without raising**, which would let the divergence check read a
plausible-looking number and let a broken run continue. Metrics that only get logged are **not**
validated, and that asymmetry is deliberate.
"""
function check_finite(l, step, epoch)
    v = l isa Number ? Float64(l) : Float64(only(Array(l)))
    # Returned rather than discarded, so the one D2H this scalar costs serves both the divergence
    # check and the train-metric line. They are the same number.
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

The validated readback, for the scalars the framework **branches on**: the early-stopping metric
here, and the checkpoint selection metric. This stack has a documented bug where a failed
`BufferToHost` returns garbage without raising, so a value that reaches a decision is checked and a
value that is only logged is not. That asymmetry is deliberate: dropping a logged `NaN` costs a
missing point on a chart, while acting on one truncates a healthy run or lets a stalled one continue.
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

The logging contract's "the framework drops non-finite values before calling". Readback happens
here too, so a backend receives host numbers rather than device scalars it would have to unwrap.

**Dropped, not zeroed and not passed through.** A `NaN` reaching a backend is either rejected by its
API or plotted as a gap in a curve that then reads as a broken run, and the framework cannot tell a
diverged metric from a failed readback at this point. The ones that matter are validated instead,
by [`check_control_readback`](@ref); these are the ones that only get logged.
"""
function finite_only(metrics::NamedTuple)
    ks = Symbol[]
    vs = Any[]
    for k in keys(metrics)
        # `host_tree` rather than `_host_value`: this is the LAST conversion before a value leaves
        # for a logger backend, and `_host_value` reaches one leaf while a metric may perfectly well
        # be a tuple or a NamedTuple of them. The `:device` train-metrics path arrives here still
        # device-resident by design, which is why the conversion is here and the assertion is after.
        v = host_tree(getproperty(metrics, k))
        v isa Real && !isfinite(v) && continue
        push!(ks, k)
        push!(vs, v)
    end
    # The boundary assertion, and the last one before user-supplied backend code. A device value
    # reaching a backend is the worst-placed instance of this class: the framework is out of the
    # stack, and a backend serializing it repeats the checkpoint path's raw-pointer failure in
    # someone else's package.
    return assert_host(
        NamedTuple{Tuple(ks)}(Tuple(vs)), "the metrics being handed to the logger"
    )
end

"""
    ReactantNitro.save_checkpoint_reported!(nitro, label, epoch, metrics) -> nothing

[`save_checkpoint!`](@ref) as one REPORTED stretch of work, so a progress reporter can say that a
checkpoint is being written.

It emits no units, so this is an unknown-length stretch rather than a counted one, and what a
watcher gets from it is the label. That is the whole value: the write is the one per-epoch stretch
that used to be silent, and an epoch whose weights are going to a slow or remote filesystem
otherwise looks exactly like an epoch that finished and then hung.

**Nothing is reported when there is no checkpointer.** `save_checkpoint!(::Nothing, ...)` is a
no-op, and a stretch announcing a write that never happens is worse than no stretch: it puts a bar
on the terminal, and an open/close pair into the event stream, for a run with checkpointing
switched off.
"""
function save_checkpoint_reported!(nitro::Nitro, label::AbstractString, epoch, metrics)
    write() = save_checkpoint!(nitro.checkpointer, epoch, metrics, snapshot(nitro))
    nitro.checkpointer === nothing && return write()
    return with_progress_stretch(write, label, 0, epoch, nitro.max_epochs)
end

"""
    ReactantNitro.snapshot(nitro) -> NamedTuple

The concrete `NamedTuple` handed to [`save_checkpoint!`](@ref). **`opt_state` is converted to HOST
values here**, which is what keeps the flow uniform with exactly one normalization point per path:
write host, read host, normalize on the way in.

Everything the record requires except `metrics` and the two framework-stamped fields, which the
checkpointer adds: `metrics` because the driver hands it the epoch's own, and the version stamps
because they are facts about the writer rather than about the run.

**`ps` and `st` are converted to host values here too, not only `opt_state`**, which the original
rule named alone. JLD2 serializes a `ConcretePJRTArray` **field by field**, which includes the raw
device pointer inside its buffer, and it neither raises on write nor on read. The pointer is
meaningless in the reading process, and the failure surfaces much later as
`AssertionError: buffer.buffer !== C_NULL` from inside a readback, naming nothing that would lead
anyone back to serialization. So the rule stated for optimizer state is the rule for the whole
record: **write host, read host, normalize on the way in.**
"""
function snapshot(nitro::Nitro)
    lgr = nitro.logger
    state = logger_state(lgr)
    # `logger_state` returning non-`nothing` makes `reattach!` REQUIRED. Checked here, at
    # the moment the state is produced, rather than at resume: a logger that claims resumable state
    # and cannot restore it is a bug, and discovering it hours later at the resume is the worst time
    # to find out.
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
        # BOTH or NEITHER. A logger returning `nothing` has no resumable state,
        # so nothing is stored and no reattachment is attempted; storing the type alone would
        # be a refusal waiting to happen on a resume that has nothing to refuse about.
        logger_state = state,
        logger_type = state === nothing ? nothing : string(nameof(typeof(lgr))),
        anchor_checksum = nitro.anchor_checksum,
        stop_reason = nitro.stop_reason,
        preset = nitro.preset,                  # the named configuration, if any
    )
    # The control. Cheap: a structural walk over leaves, not elements, so it is on the
    # order of a hundred checks per epoch against a step that costs seconds.
    assert_host_record(snap)
    return snap
end

"""
    ReactantNitro.check_logger_state_serializable(state, lgr) -> nothing

**`logger_state` must return plain serializable data**, a `String`, `NamedTuple`, or `Dict`, never
the live backend object. The record goes through JLD2 and outlives the process, so a stored handle
is either unserializable or dead on arrival.

This is the one contract detail a backend author is likely to get wrong, which is why the framework
checks it and names it rather than writing a dead handle and failing on the resume, in another
process, hours later.
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

A **host surrogate** for `Reactant.ReactantRNG`, which cannot be host-ified in place: its type
parameter is constrained `S <: Union{<:AbstractConcreteArray{UInt64,1}, TracedRArray{UInt64,1}}`,
so `ReactantRNG(::Vector{UInt64}, ::String)` is a `MethodError` and there is no such thing as a
host-resident `ReactantRNG` to rebuild (measured).

This is the general shape of the problem, not a one-off: a value whose *type* forbids host contents
cannot round-trip through a record as itself, so the record stores a surrogate and
[`from_host`](@ref) turns it back on the way in. It was found as `st.layer_2.rng.seed` on any
Lux model carrying a `Dropout`, and the failure was the usual one: JLD2 wrote the device
pointer without complaint and the restore died in a **fresh process** on
`AssertionError: buffer.buffer !== C_NULL`, naming nothing that leads back to serialization.
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

The generic fallback: **walk any struct's fields**, not only the containers.

It used to be `to_host(x) = x`, which is why [`to_host_rule`](@ref) exists as a special case for
optimizer rules, and why the same hole then turned up again in layer state. A struct is the shape
this keeps recurring in, so the walk is general now and the special cases are only for types that
cannot be rebuilt from host values at all (see [`HostRNG`](@ref)).

**It reconstructs only when something actually moved.** If every field comes back `===` what it was,
the original object is returned untouched, which keeps this safe for structs with inner
constructors, validation, or fields that could not be passed positionally. A struct that genuinely
holds a device value and cannot be rebuilt raises here, loudly, in the writing process, rather than
producing a record that fails on read hours later somewhere else.
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

The mirror of [`to_host`](@ref), applied on the **restore** path: turns every surrogate back into
the device-resident value it stood for, and leaves everything else alone for `to_rarray` to place.

Only [`HostRNG`](@ref) needs it today. It exists as a general walk rather than one line because the
reason surrogates exist, a type that forbids host contents, is a property of the ecosystem rather
than of this framework, and the next one should need a method rather than a new mechanism.
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

Every device-resident value reachable from `x`, as `path :: Type` strings. The paths are the whole
point: "which leaf is still on the device" is the only useful thing to say, and it is exactly what
neither JLD2 nor the eventual `AssertionError` will tell you.

The sibling of [`assert_device_state`](@ref), pointed the other way. That one asserts optimizer
state IS device-resident; this asserts a record is NOT.
"""
function device_paths(x, path::AbstractString = "", out::Vector{String} = String[])
    if x isa Reactant.RNumber || x isa Reactant.AbstractConcreteArray
        push!(out, string(path, " :: ", typeof(x)))
    elseif x isa Union{Tuple, NamedTuple}
        for (k, v) in pairs(x)
            device_paths(v, string(path, ".", k), out)
        end
    elseif x isa AbstractArray
        # A WRAPPER may hide a device array underneath it, and the test above cannot see one:
        # `view`, `reshape`, and `vec` of a `ConcretePJRTArray` produce a `SubArray` or a
        # `ReshapedArray`, neither of which is an `AbstractConcreteArray`. The walk used to stop
        # here and report nothing.
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

**The host/device boundary assertion.** Every value crossing from the framework to host-side user
code has already been converted by the framework; this asserts the conversion was **total**, and
names the path of anything that survived it.

It exists because of what a residency audit measured: on a CPU backend a "device" array **is** host
memory and scalar indexing into one is legal rather than fatal, so a residency mistake is nearly
unobservable there. Five of that audit's six defects were invisible to a fully green 1243-test CPU
suite for that one reason. The suite's authority stops at this line and adding CPU tests does not
extend it, so the boundary needs an assertion instead.

**Converted first, asserted second, and the order is the point.** The framework does the transfer
rather than requiring the hook to, so reaching this assertion means the conversion has a **gap**,
not that the user forgot an `Array(...)`. The two gaps found so far were both of that shape:
`to_host` walked containers but not structs, and `host_tree` covers the outputs the framework owns
while the routed batch fields came straight from the loader.

The generalization of [`assert_host_record`](@ref), which is the same idea at the checkpoint
boundary and the only mechanism in this framework that has caught its class **at the point of the
mistake** rather than hours later in an unrelated stack. Both share [`device_paths`](@ref).

**Cost is a walk over LEAVES, not elements**, so it is O(number of arrays) at a crossing rather than
O(data). The per-epoch crossings are free, and the per-optimizer-step one already pays a
device-to-host transfer that dwarfs it.
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

**"Every record value is a HOST value", asserted at the moment the record is built**, which is the
only moment the failure is cheap.

This is the control the checkpoint path was missing. Without it, a device value in a record is
written by JLD2 without complaint, read back without complaint, and kills a **fresh process** on
`AssertionError: buffer.buffer !== C_NULL` from inside an unrelated readback, naming neither the
field nor serialization. That was paid for once with optimizer rules and again with layer state;
each time the fix was one method and the diagnosis was a day. The assertion is what makes the third
one a legible error in the writing process instead.
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

The mirror of [`to_device_rule`](@ref), and it exists because **an optimizer rule is a struct, not a
container**: `to_host`'s `Tuple`/`NamedTuple` method does not reach inside one, so a rule reached
the record still holding the `ConcretePJRTNumber` the per-step rebuild put there.

That failed the way device pointers in a serialized file always fail: JLD2 wrote the struct field by
field including the raw pointer, read it back without complaint, and the restore raised
`AssertionError: buffer.buffer !== C_NULL` from inside a readback several steps later. The rule for
optimizer state is "write host, read host, normalize on the way in", and this is the half of "write
host" that the `Leaf`'s *state* got and its *rule* did not.
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
epoch, and it works on a `Nitro` that has never trained.

Eval mode throughout: the framework calls `Lux.testmode(st)` and the `st_new` returned is
**discarded**, so nothing accumulates during validation. The loop **frees each batch's device
buffers explicitly** after its metrics are accumulated, rather than leaving it to the GC, which is
the documented cause of device OOM during validation on this stack.

Like every entry point, this runs its loop on a worker thread when one is available, and Ctrl+C
stops the split at its next batch boundary and surfaces the interrupt.
"""
validate(nitro::Nitro) = with_repl(
    () -> run_eval(nitro, :val; honor_stop = true), nitro; on_interrupt = _eval_on_interrupt
)

"""
    evaluate(nitro; split = :test) -> NamedTuple

Run any named split from the data collection. **Distinct from [`validate`](@ref) rather than a second
name for it**, and it **errors on a split name `build_data` did not return, naming the available
ones**, which is the reason the data collection is a `NamedTuple` rather than a positional tuple.

Shares one compiled `forward` and one compiled `metrics` with validation and inference, so moving
between them never recompiles.

Like every entry point, this runs its loop on a worker thread when one is available, and Ctrl+C
stops the split at its next batch boundary and surfaces the interrupt.
"""
evaluate(nitro::Nitro; split::Symbol = :test) = with_repl(
    () -> run_eval(nitro, split; honor_stop = true), nitro; on_interrupt = _eval_on_interrupt
)

"""
    predict(nitro, batch::NamedTuple) -> outputs
    predict(nitro, loader) -> iterator

Inference: [`forward`](@ref) alone, in eval mode.

**It takes a batch `NamedTuple` or anything iterating them**, which is the same data-source contract
as the data contract, so a `DataLoader` works unchanged. **It does not take a bare array**:
`forward` is keyword-routed from the batch schema, so the framework needs the field names, and the
error for a bare array says exactly that, naming the fields `forward` declares. Only the fields
`forward` declares are required, so a prediction batch does not need labels; that falls out of
routing rather than being a special case, and is what lets one loader serve training and inference.

**Returns.** One batch in, one output tree out, sliced to the real sample count, as **host**
arrays. A loader in, a **lazy** iterator out, one element per batch, so predicting over a large set
does not materialize every output at once; `collect` it if you want them all. Host rather than
device because the caller is leaving the framework, and a device array that outlives its run is a
footgun.

The eval tests pin both halves: a partial batch returns exactly `n_real` outputs, and an output leaf
whose last dimension is not the batch **raises rather than slicing the wrong axis**.
"""
predict(nitro::Nitro, batch::NamedTuple) = with_repl(
    () -> _predict(nitro, batch), nitro; on_interrupt = _eval_on_interrupt
)

function _predict(nitro::Nitro, batch::NamedTuple)
    # The world-closure guard, same as the other entry points: a downstream redefinition must
    # poison the affected entries before this call reuses them.
    world_closure_staleness()
    routing, batch_size = predict_routing!(nitro, batch)
    ev = compile_view(nitro.e)
    st = Lux.testmode(nitro.st)
    padded, n_real = pad_batch(batch, batch_size, routing)
    b = to_device_batch(padded, routing, nitro.mesh)
    outputs = eval_forward(nitro, ev, st, b, routing.forward)
    host = host_tree(outputs)
    free_device_buffers!(protected_buffers(nitro, st), b, outputs)
    # Sliced on the HOST rather than on device, which is the one place this path differs from the
    # traced metric path. `predict` is transferring the outputs anyway, so slicing first would buy
    # nothing but a second compiled program per short shape; the extra compile is budgeted for
    # `metrics`, which has no such transfer.
    # The boundary assertion. `predict` is the one host crossing with no hook behind it, so a
    # device leaf surviving `host_tree` here would be returned to the CALLER, and the failure would
    # land in their code with the framework nowhere in the stack.
    return assert_host(slice_outputs(host, n_real, batch_size), "the outputs `predict` is returning")
end

# An array is iterable, so without this method a bare array would reach the loader form and fail
# somewhere unhelpful. The error names the fields `forward` declares, because that is exactly what
# the caller has to wrap it in.
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

Deferred routing. A `Nitro` built with no split at all has no first batch to resolve routing from,
so **resolution moves from setup to the first `predict` call**, which supplies one. That is the
single case where setup's ordering does not fully apply, and it is why routing is described as
resolved once per run rather than once at setup.

`batch_size` is then the width of the batch `predict` was handed, and no padding is needed, because
the caller supplied that batch whole.

A `Nitro` that DOES have a split keeps the routing resolved at setup, and this only checks that the
batch carries the fields `forward` was routed. A prediction batch legitimately lacks labels, so
the schema is **not** checked against the training one here: only `forward`'s own fields matter.
"""
function predict_routing!(nitro::Nitro, batch::NamedTuple)
    if nitro.routing === nothing
        ev = compile_view(nitro.e)
        routing = resolve_routing(ev, batch; nitro.model, nitro.ps, nitro.st)
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
    # Route to forward's fields alone, so a prediction batch that drops the label fields still pads
    # and transfers coherently. `pad_batch`, `to_device_batch`, and `batch_size_of` all take a
    # routing and read only the fields it names.
    return ((; forward = fwd), nitro.batch_size)
end

# ── The eval programs, and the pad-and-slice around them ────────────────────────────

"""
    ReactantNitro.fwd_program(ev, model, ps, st, batch, routers) -> outputs

The eval-mode forward, shared by `predict`, `validate`, and `evaluate`, which is what makes moving
between inference, validation, and testing free. It always runs at `batch_size`, because the
framework pads a short final batch before it and slices after it, so this program compiles **once**.

**`st_new` is discarded inside the program**, not returned and thrown away by the caller. It is a
program output that nothing reads, so not returning it keeps it out of the boundary entirely.


**It takes `forward`'s own router and a batch narrowed to `forward`'s own fields**, rather than the
whole routing and the whole batch. Both are in the compile-cache key by type, so a `predict` batch
carrying no labels would otherwise be a different key from a `validate` batch carrying them, and the
sharing this program exists for would be lost to a difference `forward` cannot see.
"""
function fwd_program(ev, model, ps, st, batch, fwd_router)
    outputs, _ = call_hook(forward, :forward, fwd_router, batch, ev, model, ps, st)
    return outputs
end

"""
    ReactantNitro.eval_metric_program(ev, outputs, batch, router, ::Val{NREAL}, ::Val{HOOK})

The traced metric path: `metrics` compiled as its own program, on outputs and batch fields already
sliced to `NREAL` **inside the trace**, which is where the device arrays are.

`NREAL` is type-level for the reason `LayoutRef` and `Val{CLIP}` are: a host number passed by value
arrives traced, and a traced length cannot slice. It also puts the two shapes in the compile-cache
key for free, which is what makes `metrics` compile twice per split, once at `batch_size` and once
at the remainder.

`HOOK` selects between the user's `metrics` and the framework's substitution for an experiment that
defines none, `val_loss = (loss(e, outputs; R_LOSS(batch)...), 1)`. The branch is on a type-level
constant, so it folds at trace time and the two are separate cache entries rather than one program
with a switch.

**The substitution is traced even when `metrics_residency` says `:host`**, because it calls `loss`,
which is a traced hook by contract: it is what the gradient program differentiates, and nothing says
it works on host arrays. It also reads `e`'s `Device` fields, which are device scalars from setup
onward. The residency choice is about the user's `metrics`, and an experiment that defines none has
not made one.
"""
function eval_metric_program(ev, outputs, batch, router, ::Val{NREAL}, ::Val{HOOK}) where {NREAL, HOOK}
    o = slice_last(outputs, NREAL)
    b = slice_last(batch, NREAL)
    return HOOK === :metrics ? call_hook(metrics, :metrics, router, b, ev, o) :
        (; val_loss = (call_hook(loss, :loss, router, b, ev, o), 1))
end

"""
    ReactantNitro.eval_forward(nitro, ev, st, b, fwd_router) -> outputs

Compile-or-reuse and invoke [`fwd_program`](@ref). One place, so `predict` and the metric loop
provably share the entry, which is what "the three eval-mode paths share one compiled `forward`"
means, and one narrowing of the device batch, which is what makes that sharing survive a prediction
batch with no labels in it.
"""
function eval_forward(nitro::Nitro, ev, st, b, fwd_router)
    bf = NamedTuple{keys(fwd_router)}(b)
    # Frozen at construction against an EVAL-mode `st`, which is the mode every caller of this
    # function passes, so the resolution matches the compile.
    # `nitro` IS passed: the handle-local memo means a hit never reaches the phase machinery, so a
    # per-batch phase flap is impossible; only the first compile of a program publishes
    # `EvalCompiling`, which is honest. `gc_hash` is still read directly, since the frozen hash is
    # wanted here and `nitro` is what phases are keyed on.
    thunk = compile_cached(
        fwd_program, ev, ev, nitro.model, nitro.ps, st, bf, fwd_router;
        phase = EvalCompiling(), worlds = nitro.frozen.worlds_eval,
        gc_hash = nitro.frozen.graphconst_hash, nitro
    )
    return thunk(ev, nitro.model, nitro.ps, st, bf, fwd_router)
end

"""
    ReactantNitro.run_eval(nitro, split; report = true, honor_stop = false) -> NamedTuple

The body of [`validate`](@ref) and [`evaluate`](@ref), which differ only in which split they name.

`honor_stop = true` (what the standalone entry points pass) breaks the batch loop at its next
boundary when [`request_stop!`](@ref) has been called, which is how Ctrl+C stops an eval at the
next batch; the framework's own per-epoch validation keeps the default `false`, so the validation
that follows a requested training stop always runs in full.

Per batch: pad to `batch_size`, transfer the routed fields, run the shared eval `forward`, compute
the metric on exactly the real samples, accumulate `(sum, count)` on the host, and **free the
batch's device buffers explicitly**. Then divide and hand the result to
[`finalize_metrics`](@ref).

Two metric paths, one per residency the user can choose:

  * **`:host`, the default.** Transfer the outputs, slice to `n_real` on the host, and call `metrics`
    in ordinary Julia with the split's own **host** batch, which is already the real samples and
    needs no slicing. Nothing about this hook is in a compiled program, so editing it recompiles
    nothing.
  * **`:device`.** Slice and call inside [`eval_metric_program`](@ref), so the outputs never cross
    the boundary and only the metric scalars do.

Eval mode throughout, and `st_new` is discarded: nothing accumulates.
"""
function run_eval(nitro::Nitro, split::Symbol; report::Bool = true, honor_stop::Bool = false)
    # `report = false` for `train!`'s own per-epoch validation: it already printed the banner once for
    # this call, and reprinting it every epoch would bury the metrics it exists to contextualize.
    # The world-closure guard runs with the report: the standalone entry points poison before
    # running, while train!'s per-epoch validation inherits the check its entry already made.
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
    # When the user defines no `metrics`, the framework substitutes the validation loss, reusing
    # LOSS's own router rather than routing the batch through a second hook. A default method
    # declaring `kwargs...` would take routing rule 3's whole batch and splat all of it into
    # `loss`, which declares a subset: a MethodError on exactly the path the default exists for.
    hook = routing.metrics === nothing ? :val_loss : :metrics
    router = hook === :val_loss ? routing.loss : routing.metrics
    # The substitution is traced whatever the residency says, and the residency itself was
    # resolved at construction, so this whole line is now frozen state.
    traced = hook === :val_loss || nitro.frozen.metrics_residency === :device
    # A host metric reads the split's own host batch, so transferring its fields would be bandwidth
    # spent on something nothing on device reads. `forward`'s fields are the only ones it needs.
    xfer = traced ? routing : (; forward = routing.forward)
    protected = protected_buffers(nitro, st)
    acc, prev_phase = nothing, nitro.phase
    set_phase!(nitro, EvalStepping())
    # The eval bar counts BATCHES, which is what `note_progress!` bumps here, and carries the split
    # name so a validation pass inside a training run is distinguishable from the epoch it sits in.
    progress_begin!(
        String(split), _split_length(getproperty(nitro.data, split)),
        nitro.epoch, nitro.max_epochs
    )
    # BUILT HERE, WHEN THE PHASE STARTS, and closed in the `finally` below. The producers, their
    # channels and every buffered batch therefore exist only for the duration of this pass, which is
    # the same lifetime the training loop gives its own stream: `train!` closes the epoch's stream
    # before it calls this, so training prefetch memory and evaluation prefetch memory never
    # coexist.
    stream = eval_stream(getproperty(nitro.data, split), xfer, nitro.batch_size, nitro.mesh)
    try
        for (idx, (batch, b)) in enumerate(stream)
            # `honor_stop`: the standalone eval entry points stop the split at its next batch
            # boundary when Ctrl+C requested a stop (their `with_repl` interrupt handler).
            # `train!`'s own per-epoch validation keeps the default `false`, so the epoch that
            # follows a requested stop still validates and checkpoints in full before the run
            # ends; the graceful stop's "validation runs" is not a partial pass.
            honor_stop && nitro.stop_requested && break
            # Eval progress is invisible on the handle: `idx` is local and `step` does not move
            # during evaluation, so without this a long validation or test pass looks motionless to
            # a watchdog measuring time since progress.
            note_progress!()
            check_batch_schema(batch, nitro.schema, split, idx)
            # Recomputed rather than carried through the stream, and it cannot disagree with what the
            # producer padded to: `pad_batch` derives it from this same host batch and the same
            # routing, so both sides call one function on one object. `eval_stream` says why it is
            # not a third element of the pair.
            n_real = batch_size_of(batch, xfer)
            outputs = eval_forward(nitro, ev, st, b, routing.forward)
            m = if traced
                # Host-side, against the width the program was compiled at, and only when there is
                # a slice to protect: inside the trace the same assertion would compare each leaf
                # against itself, and it would raise from a place with no batch index in scope.
                n_real == nitro.batch_size || check_output_batch_dim(outputs, nitro.batch_size)
                thunk = compile_cached(
                    eval_metric_program, ev, ev, outputs, b, router,
                    Val(n_real), Val(hook); phase = EvalCompiling(),
                    worlds = nitro.frozen.worlds_eval, nitro
                )
                thunk(ev, outputs, b, router, Val(n_real), Val(hook))
            else
                # `metrics` NEVER sees padding: the outputs are sliced back to `n_real`, and the
                # batch handed to it is the split's own host batch, which was never padded.
                call_host_hook(
                    metrics, :metrics, router, batch, e,
                    slice_outputs(host_tree(outputs), n_real, nitro.batch_size)
                )
            end
            acc = accumulate_metrics(acc, host_metrics(m, split), split)
            # Device OOM during validation is caused on this stack by per-batch outputs not
            # being freed eagerly. Do not leave it to the GC.
            free_device_buffers!(protected, b, outputs)
        end
        # The exactly-once ledger, on a pass that ran to completion. An `honor_stop` break leaves it
        # legitimately partial, exactly as a requested stop does on the training side.
        (honor_stop && nitro.stop_requested) || check_prefetch_delivery(stream)
    catch
        set_phase!(nitro, prev_phase)
        progress_end!()
        report && progress_done!()
        rethrow()
    finally
        # ONE teardown for both paths, in the `finally` rather than duplicated into the `catch` and
        # the fall-through, exactly as the training loop's epoch stream is closed. `honor_stop`, an
        # error in a hook, and a clean pass all leave here, and each of them leaves producers to
        # stop and buffered device batches to free.
        close_stream!(stream)
    end
    progress_end!()
    # `report` is already the standalone-versus-inside-a-training-epoch distinction: `train!`'s own
    # per-epoch validation passes `false`, and the last stretch of THAT entry point is the run.
    report && progress_done!()
    # The boundary assertion, placed on the OUTPUT rather than the input because the input is
    # `host_metrics`' product and already host. `finalize_metrics` is user code, and what it returns
    # reaches three consumers that all assume host values: the logger, the phase monitors, and the
    # checkpoint metric. One assertion covers all three, and it is here
    # rather than at each of them because here is where the value is produced.
    out = assert_host(
        finalize_metrics(e, reduce_metrics(acc), split),
        "the metrics `finalize_metrics` returned for the `$split` split"
    )
    # The transition OUT of `EvalStepping` is the one that carries `info.metrics`, and this is the
    # first moment those numbers exist. That transition is what replaces an earlier design's
    # `on_validation_end` hook: a user wanting to log something custom at validation time has the
    # metrics and `info.logger` here, so the framework needs no dedicated hook for it.
    set_phase!(nitro, prev_phase; metrics = out)
    return out
end

# Optimizer steps in one epoch, which is what the training bar counts. Setup checked
# `length(train) % accum == 0` to resolve the schedule horizon, so the division is exact and this
# is the same arithmetic `total` came from. Zero when the length is not knowable, which the
# reporter contract reads as an indeterminate bar rather than an empty one.
function _epoch_steps(nitro::Nitro)
    return try
        div(_split_length(nitro.data.train), max(nitro.accum, 1))
    catch
        0
    end
end

# A split's batch count WITHOUT iterating it, on the same rule the handle's display uses: a source
# that promises no length gets zero rather than a number nobody measured.
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

Read one batch's metric result back to host values and check the metric contract on it.

**The contract is checked rather than assumed**, because the natural mistake is returning a bare
number instead of a pair, and a bare number would otherwise be accumulated as though it were a
`(sum, count)` tuple's first element and divided by nothing recognizable. The error names the metric
and shows the shape it wanted.

`count === nothing` is legal and means accumulate by summation without dividing, which is what
a confusion matrix needs.
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
    # `host_tree` rather than `_host_value`, for the same reason `finite_only` uses it: a metric's
    # numerator is one value in the common case and nothing requires it to be a single leaf, so
    # the conversion walks rather than reaching one level.
    return (host_tree(s), c === nothing ? nothing : host_tree(c))
end

_host_value(x::Reactant.RNumber) = Reactant.to_number(x)
_host_value(x::Reactant.AbstractConcreteArray) = Array(x)
_host_value(x) = x

"""
    ReactantNitro.accumulate_metrics(acc, m, split) -> NamedTuple

Add one batch's `(sum, count)` pairs into the accumulator. The **key set is fixed by the first
batch**, and a later batch introducing or dropping a metric is an error rather than a silently
ragged average: with per-metric denominators there is no honest number to report for a metric that
was measured on some batches and not others.
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

The device buffers [`free_device_buffers!`](@ref) must never touch: the parameters, both copies of
the layer state, `w0`, the optimizer state, and the gradient accumulator. All of them outlive the
batch, and a `forward` that returns one of its inputs unchanged would otherwise hand the eval loop a
live parameter buffer to free.

**The experiment is deliberately not walked.** Its `Device` values are device-resident too, and an
experiment may legitimately carry a materialized dataset, so walking it would be O(n_train) per
call to protect values no output can be.

Computed once per `validate`/`evaluate` call rather than per batch: it walks the parameter tree,
which is cheap once and not cheap thousands of times.
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

Free the device buffers of one eval batch and its outputs **explicitly**, rather than leaving them
to the GC, which is the documented cause of device OOM during validation on this stack. The data
path applies the same principle to prefetch buffers.

Three guards, none of them decorative:

  * a buffer in `protected` is skipped, so an output that IS an input is never freed under the run;
  * an argument XLA **donated** is skipped, since its storage now belongs to a result and reading
    its `data` field raises "has already been donated";
  * a pointer already freed in this call is skipped, so two leaves sharing one buffer free once.

Reactant's own `Buffer` finalizer is a no-op on a null pointer, so nulling here is what makes the
early free and the eventual finalization compose instead of colliding.
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

# The concrete (device-resident) leaves of a tree. A READ-ONLY walk rather than `Functors.fmap`,
# which would rebuild every container it descends: nothing here needs a rebuilt tree, and rebuilding
# `opt_state`'s `Leaf`s or a user's output struct to look at their buffers is work that can fail for
# no benefit. Containers are the ones this framework actually threads: parameter and state trees are
# nested NamedTuples of arrays, the accumulator is an `NTuple{G}`, and `opt_state` is `NTuple{G}` of
# `Leaf`. An output inside some other struct is simply not reached, which costs a late GC-timed free
# rather than a wrong one.
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

# Optimizer rules and a user's output struct are ordinary structs holding device values, and both
# are worth reaching. Numbers, strings, symbols, functions, and types are not containers and walking
# their internals would descend into the runtime rather than into the run.
_walkable_struct(x) = isstructtype(typeof(x)) &&
    !(x isa Union{Number, AbstractString, Symbol, Function, Type, AbstractArray})

# PJRT only, deliberately. `data` is an NTuple of AsyncBuffer there, one per device. On IFRT the
# layout differs and this returns nothing, so the eager free degrades to GC-timed collection, which
# is a performance property rather than a correctness one. The framework's scope is single node,
# and the runtime it is developed and tested against is PJRT.
function _device_buffers(x)
    getfield(x, :donated) && return ()
    data = getfield(x, :data)
    data isa Tuple || return ()
    return map(d -> d.buffer, data)
end

"""
    ReactantNitro.host_tree(tree) -> tree

Every device leaf of a tree read back to host values, structure preserved. This is what a `:host`
metric hook and `predict` receive, and it is the whole of what "what changes for the hook author is
only what `outputs` is" means.

**It walks structs, and it did not used to.** This was `Functors.fmap` with an
`exclude`, and `fmap` descends `Tuple`, `NamedTuple`, and arrays but treats an **unregistered struct
as a leaf**; `exclude` then rejected that leaf, so a device array held inside a plain struct came back
**unconverted**, silently, on the path that feeds every `:host` metric hook and `predict`.

That is [`to_host`](@ref)'s history repeating on the other walker. It turned up first in optimizer
rules, then again in layer state, where `to_host` was fixed by making the walk general, and the same
fix was never applied here because nothing pointed at the second walker. **The framework had two
functions that both claim to read a tree back to host and one of them could not see inside a struct.**
Building `assert_host` is what surfaced it: the assertion walks any struct through
[`device_paths`](@ref), so it is strictly more thorough than the conversion it guards, and the gap
between the two was a real defect rather than a theoretical one.

The shape mirrors `to_host` deliberately, including **identity preservation**: a subtree with nothing
device-resident in it comes back `===` what it was, so the common case allocates nothing and structs
with inner constructors or validation are never rebuilt needlessly. It differs from `to_host` in one
respect only, that it has no surrogate types: a hook wants the value, and there is nothing sensible to
hand it for a type that cannot hold one.
"""
host_tree(x::Reactant.RNumber) = Reactant.to_number(x)
host_tree(x::Reactant.AbstractConcreteArray) = Array(x)

# `Reactant.ReactantRNG` IS THE TYPE THAT CANNOT HOLD A HOST VALUE, and "no surrogate types" above is
# only half an answer for it: the generic struct walk at the bottom of this section reaches its
# `seed`, converts that to a `Vector{UInt64}`, and then rebuilds the struct with
# `ReactantRNG(::Vector{UInt64}, ::String)`, which is precisely the `MethodError` [`HostRNG`](@ref)
# exists to document. So it is passed through UNCHANGED, which is the only thing available and is also
# the right answer here: an RNG's seed is not model data, no `:host` metric hook is ever handed layer
# state, and the one path that walks a whole `st` is export, where the state is TRACED rather
# than read and a `Dropout` in test mode never draws from it.
#
# Found by the first real export of a Lux model carrying a `Dropout`, which is any model whose front
# end has one: `export_model`'s `host_tree(Lux.testmode(nitro.st))` died in the struct walk, naming
# `ReactantRNG` and nothing about export. `st.<layer>.rng` is the same leaf the residency audit
# found on the record path, so this is that hole in the second walker.
host_tree(r::Reactant.ReactantRNG) = r

# A host array is a LEAF, not a container to descend into, which is `_is_array_leaf`'s rule carried
# over unchanged: an array of arrays is walked, an array of numbers is not. Descending into a plain
# `Matrix{Float32}` would be O(elements) on a path that runs per eval batch.
function host_tree(x::AbstractArray)
    # Same blind spot as `device_paths`, and the same fix: a wrapper is not a leaf for residency
    # purposes even though it is one for slicing. Convert the PARENT, rebuild the wrapper over the
    # host parent, then materialize. Materializing after the rebuild is what keeps this off the
    # device: `Array(view_of_device_array)` would copy elementwise, which is legal on CPU and
    # "Scalar indexing is disallowed" on a GPU, i.e. exactly the defect being fixed.
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

Rebuild array wrapper `x` over a new parent `p`, for [`host_tree`](@ref).

Only the wrappers this stack actually produces are covered. Anything else **refuses by name**
rather than silently handing traced code a device array, which is the failure mode the whole
boundary assertion exists to end. Adding a wrapper is one method.
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

# The generic fallback, and the half that was missing. Same contract as `to_host`'s: reconstruct only
# when something actually moved, so this is safe for structs that could not be rebuilt if nothing had
# to be. Strings, symbols, and numbers are struct types to `isstructtype` and must not be walked.
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
