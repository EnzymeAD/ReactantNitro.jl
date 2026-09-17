# Interface.jl
#
# Hook definitions and their defaults. Four hooks are required (`build_data`, `build_model`,
# `forward`, `loss`); every other hook here has a default.
#
# ── Why every hook here is a bare `function f end` ──────────────────────────────────
#
# Two mechanisms read the METHOD TABLE rather than calling the hook, and a catch-all placeholder
# method would break both:
#
#   * batch routing resolves each hook by `Base.kwarg_decl(which(f, argtypes))`. A catch-all
#     `forward(e, model, ps, st; kwargs...)` would be what `which` resolves to, its declared
#     keywords would be the single symbol `kwargs...`, and routing rule 3 would then route the
#     WHOLE batch to every hook, silently.
#   * the missing-`metrics` fallback is framework behavior rather than a default method, detected
#     by `metrics` having no method for this experiment type. A catch-all makes that detection
#     impossible, and the method the obvious version would write is the WRONG one: declaring
#     `kwargs...` routes the whole batch into it, which it then splats into `loss`, which declares
#     a subset, so an experiment defining only the four required hooks MethodErrors on the very
#     path the default exists to support.
#
# So the defaulted accessors below get their values as real methods on `::Any`, and `metrics`
# never gets one at all.

# ── The four required hooks ─────────────────────────────────────────────────────────

"""
    build_data(e, dist) -> NamedTuple

**Required.** Return the run's named data collection, `(; train, val)` or `(; train, val, test)`.
Named and extensible, so calling [`evaluate`](@ref) with no `test` supplied is a clear error rather
than a positional mistake.

**A data source is anything iterable that yields concrete `NamedTuple`s of host arrays, and
supports `length`.** A `Vector` of batches, an `MLUtils.DataLoader`, or a bespoke sampler all
qualify. Loaders yield **host** arrays; the framework transfers each batch, applies batch-dimension
sharding, and owns prefetching.

Three requirements on the source, each with a reason:

  * **Restartable.** Setup draws one batch to learn the schema and resolve routing, then discards
    it, and the loop iterates from scratch. A one-shot source such as a bare `Channel` would
    silently lose its first batch.
  * **`length`.** Read once, at setup, to fix the schedule horizon. It counts **batches**.
  * **The `train` split drops its partial final batch** (`partial = false` for
    `MLUtils.DataLoader`, `drop_last = true` elsewhere), and its batch count must divide by
    `accum`. Eval splits should NOT drop theirs: the framework pads and slices.

There is **no `prepare_epoch!` hook**: `for batch in loader` calls `iterate` afresh each epoch, so a
loader that reshuffles or regenerates does so in its own iteration initialization.

`dist` is `nothing` and nothing dispatches on it; it is the seam a distribution layer would use.
Write it untyped.

`build_data` sees the **pre-conversion** experiment, where `Device` fields hold plain host values.
It is skipped entirely when `Nitro(e; data = ...)` supplies the collection directly, which is what
makes serving possible.
"""
function build_data end

"""
    build_model(e, rng) -> (model, ps, st)

**Required.** Standard Lux.

**Pretrained loading must happen inside `build_model`**, because the framework captures `w0`
immediately after it returns; loading afterwards would anchor `decay_anchor = :w0` to the random
init, silently.

It sees the **post-conversion** experiment, so `Device` fields already hold device values.

For a model with variant branches, lift the selector to a type with
[`dispatch_variant`](@ref) so the branches fold at trace time rather than leaving the return type a
union of every variant:

```julia
build_model(e, rng) = _build_model(e, dispatch_variant(e, :model_kind), rng)
_build_model(e, ::Val{:bit_r50}, rng) = ...
```

Cost is one dynamic dispatch at a boundary crossed once per run. Each variant gets its own Julia
specialization, so many variants multiply **Julia** compile time, which is a different budget from
XLA compile time.
"""
function build_model end

"""
    forward(e, model, ps, st; <declared batch fields>) -> (outputs, st_new)

**Required.** One definition of the model's computation, three consumers: training traces `forward`
then [`loss`](@ref), validation traces `forward` then [`metrics`](@ref), and [`predict`](@ref) is
`forward` alone.

**Discipline: `forward` takes inputs, not targets.** That is what makes `predict` possible. A model
genuinely needing targets in its forward (teacher forcing) cannot be predicted from inputs alone,
which is true rather than a limitation.

**Declare exactly the batch fields you want as keywords**; the framework resolves the method once
at setup and passes exactly that subset. A keyword with a default is an optional batch field and
works as you would expect. A method ending in `kwargs...` receives the **whole batch** and nothing
is checked for it.

**`outputs` is passed on as ONE positional argument.** Whatever goes in the first slot of this
return is exactly what [`loss`](@ref), [`metrics`](@ref) and [`train_metrics`](@ref) receive in
their second; nothing is splatted or unpacked in between. For more than one output, return a
`NamedTuple` and destructure it by name downstream:

```julia
forward(e, model, ps, st; tokens) = ((; logits, energy), st_new)
loss(e, out; target) = ce(out.logits, target) + e.w * mean(abs2, out.energy)
```

A `Tuple` or a nested structure works too, since the framework walks the output tree with
`Functors`, but a `NamedTuple` is what [`export_outputs`](@ref) names leaves by.

**Every array leaf of the output tree must have the batch dimension last, and must be an array.**
Both follow from the short final eval batch, which is padded to the compiled width and sliced back:
the framework asserts the last axis is the batch rather than slicing the wrong one, and a scalar you
already reduced over the batch has nothing to slice. Reduce in [`loss`](@ref) or [`metrics`](@ref)
instead of returning the scalar here.

State threading is inherent to the contract: for a stateless model `st_new` is `st`. The framework
owns the train/eval mode switch (`Lux.trainmode` / `Lux.testmode`) and users never call either.
The `st_new` returned from an eval-mode call is discarded.

`forward` is also the exportable program: export uses the same function as `predict`, so there is
no second model definition.
"""
function forward end

"""
    loss(e, outputs; <declared batch fields>) -> scalar

**Required.** The scalar Enzyme differentiates, traced inside the gradient program together with
[`forward`](@ref).

`outputs` is the first element of `forward`'s return; the framework has already stripped `st_new`.
**Do not unpack it again**: `first(outputs)` is element one of the prediction array, and every
operation after that stays broadcast-legal, so the run would train on one sample out of the batch
without raising.

Keyword routing is [`forward`](@ref)'s. `loss`'s own router is also what the default validation
metric reuses when an experiment defines no [`metrics`](@ref).
"""
function loss end

# ── Metrics ─────────────────────────────────────────────────────────────────────────

"""
    metrics(e, outputs; <declared batch fields>) -> NamedTuple of (sum, count)

Traced, once per eval batch. **A metric reports its own numerator and denominator**; the framework
adds them up and divides at the end:

```julia
metrics(e, outputs; lab) = (; err = (sum_abs_err, n_items),
                              acc = (n_correct,   n_images))
```

A framework-supplied sample count would be the wrong denominator, since different metrics have
different natural ones (per-sample, per-image, per-object). **`count === nothing` means accumulate
by summation without dividing**, which covers confusion matrices.

`metrics` **never sees padding**: the framework slices a padded short final batch back to `n_real`
before calling it, so there is no `mask` batch field and no silent-miscount failure to guard
against.

**When the user defines no `metrics`, the framework reports validation loss**, as
`val_loss = (loss(e, outputs; R_LOSS(batch)...), 1)`, reusing `loss`'s own router. That is framework
behavior rather than a default method, and the distinction is load-bearing: see this file's header.
The `count` of `1` makes it the mean over *batches* rather than over samples, which is a
reporting-only difference and never touches training numerics.

The `:validation` / `:testing` distinction lives **host-side**, in
[`finalize_metrics`](@ref), because as an argument here it would produce two compiled programs even
when the user's code ignored it.
"""
function metrics end

"""
    metrics_residency(e, hook::Symbol) -> :host | :device

Where a metric hook runs. `hook` is `:metrics` or `:train_metrics`. Defaults: **`:host` for
`metrics`, `:device` for `train_metrics`.**

**The default follows the cadence, which is the whole argument.** `train_metrics` runs once per
**micro-batch**, so running it on the host means transferring a full output batch every micro-batch,
which for anything image-shaped dwarfs the scalars you wanted. `metrics` runs once per eval batch,
once per **epoch**, against a training epoch that has just done thousands of steps, so the same
transfer is noise. Tracing `metrics` by default would be optimizing the cadence that does not need
it, at the cost of the one that does.

**And the flexibility runs the other way.** A traced metric must be expressible as a traced program,
which rules out a great deal that is ordinary in evaluation code: data-dependent control flow, a
matching or assignment step, connected components, sorting with tie-breaking, anything reaching a
library that knows nothing about Reactant. Those are common in validation and rare in a per-step
diagnostic. Computing evaluation metrics on the host is the common case: an evaluation metric is
usually ordinary Julia over transferred arrays, and the transfer it costs is noise against the epoch
it follows. That is why the host default is the workable one rather than a concession.

Both hooks accept both values, so the choice is the user's:

```julia
# A per-step diagnostic too expensive to trace, on a small model where the transfer is affordable.
ReactantNitro.metrics_residency(::MyExp, hook) = hook === :train_metrics ? :host : :host

# A validation metric that IS traceable, on a large eval set where the transfer is not free.
ReactantNitro.metrics_residency(::MyExp, ::Symbol) = :device
```

**What changes for the hook author** is what `outputs` is: device arrays under `:device`, host
arrays under `:host`. Everything else is identical, including the keyword routing, the
`(sum, count)` contract, and the guarantee that a metric never sees padding.

**What changes for the framework** is the compile cache. A `:host` hook is part of no program, so
its `primary_world` is **not** in the cache key and editing it recompiles nothing. That is the
point: it is what makes adding a diagnostic mid-session free.
"""
function metrics_residency end
metrics_residency(e, hook::Symbol) = hook === :train_metrics ? :device : :host

"""
    ReactantNitro.check_residency(e, hook) -> Symbol

Read [`metrics_residency`](@ref) and reject anything but the two legal values, since a typo would
otherwise silently select the default and change where the metric runs.
"""
function check_residency(e, hook::Symbol)
    r = metrics_residency(e, hook)
    r in (:host, :device) && return r
    error(
        """
        ReactantNitro: `metrics_residency(e, $(repr(hook)))` returned $(repr(r)); it must be
        `:host` or `:device`.
        `:device` traces the hook into a compiled program, so its outputs never cross the device
        boundary and editing it recompiles that program. `:host` runs it in ordinary Julia on
        transferred arrays, so it can do anything Julia can and editing it recompiles nothing."""
    )
end

"""
    train_metrics(e, outputs; <declared batch fields>) -> NamedTuple of scalars

Traced, once per micro-batch, inside the gradient program. Default `(;)`, which **must fold away
entirely**.

Train and validation metrics have deliberately different contracts: these are **scalars per step**
with no reduction, for an instantaneous diagnostic, while [`metrics`](@ref) is `(sum, count)` per
batch, accumulated over a set once per epoch. Smoothing is host-side in the user's logger.

**Readback is lazy**, so a value returned from a compiled thunk is a device array and costs D2H only
when read: compute these in-graph every step and read them only on logged steps, which avoids
separate programs for logged and unlogged steps. See [`check_control_readback`](@ref) for the
readback hazard that creates.

The primal is exposed to metrics code **inside** the traced program, not transferred: the step
returns `(loss, st_new, stats)` and only scalars cross the boundary.
"""
function train_metrics end

"""
    finalize_metrics(e, acc, split) -> NamedTuple

Host-side, once per split per epoch. Default is identity.

Needed because a derived metric is not the mean of per-batch values:

```julia
finalize_metrics(e, acc, split) = (; f1 = 2acc.tp / (2acc.tp + acc.fp + acc.fn))
```

`split` is the `Symbol` naming the split, so this is where branching between validation and testing
belongs: it is free here and costs a second compiled program in [`metrics`](@ref).
"""
function finalize_metrics end

finalize_metrics(e, acc, split) = acc

# ── Manual training mode ────────────────────────────────────────────────────────────

"""
    train_step(e, model, ps, opt_state, st; <declared batch fields>)
        -> (; loss, ps, st, opt_state, stats)

**Define this method for an experiment type to switch `train!` to manual mode.** The automatic
loop has no `train_step`: its step is framework-owned, `grad_program` plus `opt_program` plus
`rebuild_rules` sequenced by the driver, and the user never writes one. So the act of writing this
method is what selects manual mode; [`manual_training`](@ref)`(e) = false` declines it while
keeping the method. An experiment that defines neither trains automatically, and an experiment
that defines it but is only evaluated never consults it.

**Manual mode hands you the whole optimizer step.** The framework keeps everything outside the
step: the batch stream and prefetch, per-epoch accounting, validation, checkpointing, logging,
phases, and early stopping. Inside one call you own the forwards, the backwards, the optimizer
steps, and the device metrics.

**The call.** Once per optimizer step, in train mode (the framework owns the `Lux.trainmode`
switch), on one transferred **device** batch. The batch's fields are routed from the keywords this
method declares, so a field only the closure reads (the GAN's noise, below) is
transferred only when declared. The method is traced and compiled by the framework, so it must be
a stable named method, never an anonymous closure built per step.

**The objective discipline.** The gradients inside are computed with [`backward`](@ref)`(f,
ps_sub, consts...)`, whose objective must take every **traced** value it reads as an argument.
Closure captures of traced values are silently treated as constants, which is how a zero gradient
happens without an error (measured). A plain struct like the model, which holds no arrays, is safe
to capture; parameters, batches, and state go in `consts...`.

**A very simple GAN.** Generator and discriminator as two Lux chains in one parameter tree, each
with its own optimizer; noise `z` rides in the batch. The generator's backward re-runs its own
forward from the Duplicated subtree, and the discriminator's parameters are a `Const` argument, so
the generator's gradient flows through the discriminator's forward as a function of the fake data
and never into the discriminator's parameters (the non-saturating formulation):

```julia
using Statistics: mean   # or `using Statistics` at the top of the file

@experiment struct ToyGAN
    max_epochs::Host{Int} = 50
end

ReactantNitro.build_data(e::ToyGAN, dist) = (;
    train = [(; x = randn(Float32, 4, 32), z = randn(Float32, 2, 32)) for _ in 1:10])

ReactantNitro.build_model(e::ToyGAN, rng) = begin
    model = (; gen  = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 4)),
               disc = Lux.Chain(Lux.Dense(4 => 16, tanh), Lux.Dense(16 => 1)))
    (model, Lux.setup(rng, model)...)
end

ReactantNitro.setup_optimizers(e::ToyGAN, model, ps, st, mesh) = (;
    gen  = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.gen),
    disc = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.disc))

function ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z)
    # forwards: the generator's output is the discriminator's fake input. Each sub-network
    # applies with its own state subtree; `st` stays the WHOLE tree for the backwards and
    # the return.
    fake, st_gen = Lux.apply(model.gen, z, ps.gen, st.gen)
    d_real, _ = Lux.apply(model.disc, x, ps.disc, st.disc)
    d_fake, _ = Lux.apply(model.disc, fake, ps.disc, st.disc)

    # discriminator backward: real -> 1, fake -> 0
    l_d, g_d = ReactantNitro.backward(ps.disc, x, fake, st) do ps_d, xc, fc, stc
        d1, _ = Lux.apply(model.disc, xc, ps_d, stc.disc)
        d2, _ = Lux.apply(model.disc, fc, ps_d, stc.disc)
        mean(abs2, d1 .- 1.0f0) + mean(abs2, d2)
    end

    # generator backward: fake -> 1, through the discriminator's forward, never into its params
    l_g, g_g = ReactantNitro.backward(ps.gen, ps.disc, z, st) do ps_g, ps_d, zc, stc
        f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
        d3, _ = Lux.apply(model.disc, f2, ps_d, stc.disc)
        mean(abs2, d3 .- 1.0f0)
    end

    # one optimizer step per network; states out, rules re-attached by the driver
    s_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
    s_d, ps_d = ReactantNitro.step_optimizer(opt_state.disc, ps.disc, g_d)

    return (; loss = l_d + l_g,
              ps = (; gen = ps_g, disc = ps_d),
              st = (; gen = st_gen, disc = st.disc),
              opt_state = (; gen = s_g, disc = s_d),
              stats = (; l_d, l_g))
end

train!(Nitro(ToyGAN()))
```

**The return.** A checked `NamedTuple`:

| Key | Meaning |
| --- | --- |
| `loss` | **Required.** The scalar the driver validates (fail-fast on non-finite) and logs |
| `ps` | The updated parameter tree |
| `st` | The updated layer state; a stateless model returns `st` unchanged |
| `opt_state` | The new optimizer **states only**, not the leaves. The measured sharding rule: a replicated scalar (a rule's device hyperparameters) cannot leave a compiled program as an output on a mesh, though arrays can. [`step_optimizer`](@ref) returns states; the driver re-attaches the rules it handed in |
| `stats` | Device scalars logged once per step (train metrics), `(;)` allowed |

The driver checks the keys, validates `loss` against the no-non-finite rule, logs
`(; loss, stats...)`, and stores `ps`, `st`, and the merged `opt_state` for the next call.

**Schedules.** [`schedules`](@ref) works in manual mode with one difference: `opt` keys are
**path-bound** into the `opt_state` [`setup_optimizers`](@ref) returned, because the framework has
no parameter groups to bind them to. A bare key (`opt.eta`) binds every rule with that field, at
the absolute value; a nested key (`opt.gen.eta`) is a path into the tree and binds the rules at
that subtree. The driver rebuilds the named rules between calls, type-preserving, so the same
program serves every step. `device` keys work exactly as in the automatic loop. `accum > 1`
remains a setup error (accumulation is the automatic loop's mechanism; the closure owns its own
multi-batch work). The [`train_metrics`](@ref) hook is unused in manual mode: `stats` come from
the closure itself.

**What the automatic loop still does for you in manual mode**, namely validation, checkpointing,
early stopping, `request_stop!`, phases, and the device schedules, is unchanged; [`loss`](@ref)
becomes optional (the closure computes its own losses; define [`metrics`](@ref) for validation, or
`loss` for the `val_loss` substitution when `metrics` is also absent).
"""
function train_step end

"""
    setup_optimizers(e, model, ps, st, mesh) -> opt_state

**Required for a manual experiment.** Build the optimizer states the user owns, one per network or
group:

```julia
ReactantNitro.setup_optimizers(e::MyGAN, model, ps, st, mesh) = (;
    gen  = Optimisers.setup(Optimisers.AdamW(1.0f-4), ps.gen),
    disc = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.disc))
```

Called once at construction, in place of the automatic loop's own optimizer-state build, after
device conversion and `build_model`. The framework normalizes the result to device residency
through the `to_device_leaf` walk and asserts the no-host-`Number` property on it, so a rule's
scalars are traced and an integer step counter that freezes under trace is promoted before it can.
The result is what [`train_step`](@ref) receives as `opt_state`.

Rules are fixed for the run: there is no per-step rebuild, so rule-field schedules are a
setup error. Fixed rules are safe across steps because the closure's program does not donate their
scalars (measured).
"""
function setup_optimizers end

"""
    manual_training(e) -> Bool

The manual-mode flag. Default: whether the experiment type has a [`train_step`](@ref) method,
which is how defining the hook selects manual mode. Override to `false` to keep the method in
source while using the automatic loop:

```julia
ReactantNitro.manual_training(::MyExp) = false
```

Read once, at construction, and frozen: the driver a `Nitro` uses is fixed at construction, so
flipping this on an existing handle is inert and reported by `fixed_config_report` rather than
applied silently.
"""
function manual_training end

manual_training(e) =
    hasmethod(train_step, Tuple{typeof(compile_view(e)), Any, Any, Any, Any})

# ── Derived values ──────────────────────────────────────────────────────────────────

"""
    derive(e, data) -> NamedTuple

Values that genuinely depend on the dataset. Default `(;)`. The framework merges the result into
the experiment:

```julia
derive(e::MyExp, data) = (; class_weights = inv_freq(data))
```

It runs **before** device conversion, so it always returns host values, and it sees the
**pre-conversion** experiment. A derived [`Device`](@ref) becomes a device value excluded from the
cache key; a derived `GraphConst` field becomes a baked constant included in it.

**Prefer constants for structural values; derive only numerics.** Deriving a shape-determining value
makes the compiled shape a function of the data, so a different split silently changes the program
and the cache key. Use a constant with an explicit bounds check at data load instead.

**Distinguish derived from merely computed-late.** A value computable from config alone belongs in an
ordinary computed default, not here.

**Resume recomputes, it does not restore**, so resuming against changed data picks up new values;
they are recorded in the checkpoint record so a change is visible after the fact.
"""
function derive end

derive(e, data) = (;)

"""
    dispatch_variant(e, field) -> Val

Lift a `Symbol` config selector to a type, so variant branches below the barrier fold at trace
time. Default `Val(getproperty(e, field))`.
"""
function dispatch_variant end

dispatch_variant(e, field) = Val(getproperty(e, field))

# ── Optimizer accessors ─────────────────────────────────────────────────────────────

"""
    param_group(e, keypath) -> Symbol

The parameter group a leaf belongs to, as a function of its keypath. **Defaulted accessor, not
mandatory dispatch**: the default returns `:default` for every keypath, which is the single-group
case and is what most experiments want.

```julia
param_group(::MyExp, ks) = ks[1] === :backbone ? :backbone : :default
```

The group is **the unit at which optimizer behavior is defined**. `:default` is group 1; remaining
groups follow first-appearance order, which is stable for a fixed model and bakes into the compiled
program. Per-layer groups (`Symbol(join(ks, "_"))`) are available if finer resolution is ever
wanted, at a cost that is visible since `G` is a trace-time constant.
"""
function param_group end

# The universal fallback: the single-group case, which is what most experiments want, so that a
# bare experiment never has to define `param_group` at all.
param_group(e, keypath) = :default

"""
    optimizer(e) -> Type{<:Optimisers.AbstractRule}
    optimizer(e, group::Symbol, hp) -> Optimisers.AbstractRule

The two upper levels of the three-level optimizer surface.

**Level 0**: declare nothing and get **`RAdam`**, with per-group hyperparameters from
[`learning_rate`](@ref) and [`lambda`](@ref). RAdam rather than Adam deliberately: it rectifies the
adaptive variance term over the first steps instead of leaving the user to hand-tune a warmup that
does the same job by feel. `Adam` is one line away at Level 1. The cost is a hard floor of
Optimisers 0.4.8, which is the first release whose Reactant extension can trace RAdam.

**Level 1**, pick the rule. The hook returns a rule **type** and the framework constructs it,
splatting resolved values in by field name, so every configured hyperparameter is applied by
construction:

```julia
optimizer(::MyExp) = Optimisers.Momentum
optimizer(::MyExp, ::Val{:backbone}) = Optimisers.Momentum
```

The framework-composed [`Decay`](@ref) tail, per-group accessors, `w0` capture, and no-decay mask are
untouched. **Level 1 rejects any rule declaring a `lambda` field** (`AdamW`), because that rule plus
the framework's own decay tail would decay twice, silently, and `opt.lambda` would then resolve
against two rules in one chain.

**Level 2**, supply the chain. Same function, higher arity, receiving the group and its already
resolved, already device-converted hyperparameters:

```julia
optimizer(::MyExp, group::Symbol, hp) = Optimisers.OptimiserChain(
    Optimisers.RAdam(; eta = hp.eta, beta = hp.beta),
    Decay(hp.lambda, hp.anchor, hp.no_decay_mask),
)
```

`hp` is a `NamedTuple` with `eta` (this group's effective learning rate), `lambda` (already
multiplied by that eta), `anchor` (a flat device array or `nothing`), `no_decay_mask`, and one
key per scheduled rule field. `group` is a bare `Symbol` here and a `Val` in the accessors: the
accessors dispatch, this branches host-side.

**Level 2 owns hyperparameter application, and the framework will not check that the factory read
what it was given.** A factory building `RAdam(; eta = hp.eta, beta = hp.beta)` while `epsilon` is
scheduled passes the name check and the schedule then does nothing for the whole run. This is the one
place the framework knowingly permits a silent no-op rather than a loud error, and it is confined
to the level a user opts into explicitly. It is not invisible: the binding report lists, per Level
2 group, which scheduled values it found in the returned chain and which it did not.

**No path accepts a fully constructed chain with baked hyperparameter values**, because rules carry
tracked device scalars rebuilt every step.
"""
function optimizer end

# Level 0: RAdam, deliberately. Needs Optimisers >= 0.4.8 for the traced `apply!`.
optimizer(e) = Optimisers.RAdam
# Level 1 per group, defaulting to the same rule everywhere.
optimizer(e, ::Val{g}) where {g} = optimizer(e)
# Level 1 IS the default method of Level 2, so there is one concept with a shorthand.
optimizer(e, group::Symbol, hp) = level1_chain(e, group, hp)

"""
    learning_rate(e) -> Real
    learning_rate(e, ::Val{group}) -> Real

The base learning rate, and the per-group bases. Defaults are `1f-3` and `learning_rate(e)`, i.e.
ratio 1.0 for every group.

**Per-group accessors define ratios; the schedule sets the absolute value of the default group**:

    η_g(t) = eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))

so `η_default(t) == eta_sched(t)` exactly, and a backbone at `1f-4` against a default of `1f-3`
stays a tenth of it for the whole run. **`learning_rate(e) == 0` is a setup error**, since the ratio
is undefined; that is also why the default is `1f-3` rather than `0`.
"""
function learning_rate end

# These are plain values rather than `_field` reads. That is deliberate and not an oversight: an
# experiment may declare a `lambda::Device{Float32}` that is an auxiliary LOSS weight, and a
# `_field(e, :lambda, 0)` default would silently read it as the optimizer's decay coefficient.
# Only `gradient_clip_norm` and `max_epochs` read a field of the same name.
learning_rate(e) = 1.0f-3
learning_rate(e, ::Val{g}) where {g} = learning_rate(e)

"""
    lambda(e) -> Real
    lambda(e, ::Val{group}) -> Real

The decay coefficient, per group. Defaults are `0` and `lambda(e)`, so by default there is **no**
[`Decay`](@ref) in the chain at all.

**There is one decay coefficient per group and what it decays *toward* is set separately by that
group's [`decay_anchor`](@ref).** Decaying toward zero and decaying toward the pretrained
weights are one rule with different anchors, so the configuration surface is the rule's own two
fields rather than two competing coefficients that could contradict each other.

Across groups it can be both at once, and one `opt.lambda` schedule drives both: the intended setup
is `:w0` on the pretrained backbone and `:zero` on the fresh head. That is usually what you want,
since both are the same regularization-strength knob, and the binding report names the anchor per
group so it is visible.

Decay is **decoupled**, so the value reaching the rule arrives pre-multiplied by that group's
effective learning rate, and an LR schedule modulates regularization strength.

**Scheduling `opt.lambda` while also defining per-group `lambda` accessors is a setup error naming
both**, since the schedule supplies the base.
"""
function lambda end

lambda(e) = 0.0f0
lambda(e, ::Val{g}) where {g} = lambda(e)

"""
    decay_anchor(e, ::Val{group}) -> :zero | :w0 | AbstractArray

What this group's decay pulls toward. Default `:zero`, i.e. ordinary weight decay.

`:w0` decays toward the parameters exactly as `build_model` returned them, before any training step
or restore. The literature calls this L2-SP, for "L2 distance to Starting Point" (Li,
Grandvalet, and Davoine, ICML 2018, [arXiv:1802.01483](https://arxiv.org/abs/1802.01483)); the name
is recorded once so the technique stays findable, and `decay_anchor = :w0` is used everywhere else
because it says what it does.

An explicit array is the extension point: initializing from A while anchoring to B is
`decay_anchor` returning B.

**`decay_anchor = :w0` on a randomly initialized group anchors it to a random point.** That is why
`:w0` goes on the pretrained backbone and `:zero` on the fresh head, why `:zero` is the default, and
why the framework cannot warn: it cannot detect which groups are pretrained.

Anchoring anywhere but `:zero` puts a per-group `anchor_checksum` in the checkpoint record and makes
resume **verify it and refuse on mismatch**. The arrays themselves are not stored, which
keeps a checkpoint from roughly doubling for a `:w0`-anchored backbone, and the accepted cost is that
a run whose anchored parameters come from a nondeterministic source cannot be resumed at all.
"""
function decay_anchor end

# `:zero` rather than `:w0`, because anchoring blind would be wrong for any group that was not
# pretrained, and the framework cannot detect which groups are.
decay_anchor(e, ::Val{g}) where {g} = :zero

"""
    ReactantNitro.default_no_decay(keypath, param) -> Bool

The framework's built-in decay exclusions: **every 1-D parameter**. In Lux that is biases and every
normalization layer's affine scale and shift, which is why one predicate covers all three.

**Structural rather than name-based**, so it does not depend on a layer naming convention. Exported
so that [`no_decay`](@ref) can be *extended* rather than replaced, which is the common case:

```julia
ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || (:fc in ks)
```
"""
default_no_decay(ks, x) = ndims(x) == 1

"""
    no_decay(e, keypath::Tuple, param) -> Bool

Whether this parameter leaf is **excluded from weight decay entirely**. Default
[`default_no_decay`](@ref), so a bare experiment excludes biases and norm affines and decays
everything else, which is the conventional policy and is exactly what the framework did before this
hook existed.

It receives the **leaf itself**, not merely its keypath, because a useful exclusion rule needs both:
the keypath says *which* parameter, and the array says what shape it is. `ndims`, `size`, and
`eltype` are all fair game. This costs nothing, because the hook runs **host-side, once, at setup**,
and never enters a trace, so it can do anything ordinary Julia can.

Four modes, from one hook:

```julia
# 1. DEFAULT: write nothing at all.

# 2. ADD to the defaults, which is what most experiments want.
ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || (:fc in ks)

# 3. REPLACE them with your own system.
ReactantNitro.no_decay(::MyExp, ks, x) = (:fc in ks) && (:weight in ks)

# 4. DISABLE exclusion entirely, decaying every parameter.
ReactantNitro.no_decay(::MyExp, ks, x) = false
```

Composition is a plain `||` against an exported function rather than an implicit merge the framework
performs behind you, so what a run actually excludes is readable in the user's own source.

**Exclusion dominates the anchor, for free.** An excluded leaf gets a mask of 0, and [`Decay`](@ref)
computes `no_decay_mask * lambda * (x - anchor)`, so the term vanishes whether that group's
[`decay_anchor`](@ref) is `:zero` or `:w0`. "Excluded from either" needs no separate mechanism.

**Exclusion is per LEAF; the coefficient and the anchor are per GROUP.** That split is deliberate:
biases and norm affines occur inside every group, so expressing their exclusion through
[`param_group`](@ref) would force a group split that also silently splits the learning-rate ratio,
which is a different knob. Groups stay the unit at which optimizer *behavior* is defined; this is
the one per-parameter refinement.

**Not in the compile cache key, deliberately.** The mask is a parameter-sized device buffer passed
into `Decay` as a **value**, so its contents change no graph and a world entry for this hook could
only ever fire spuriously, which is the same reason `accum` and the clip carry no world entry
either. It is resolved once at construction, so revising it takes effect on the **next `Nitro`**,
like everything else a handle freezes.
"""
function no_decay end

no_decay(e, ks, x) = default_no_decay(ks, x)

"""
    gradient_clip_norm(e) -> Real

The global-norm clip threshold. Default `0f0`, meaning **off**. Also a `Nitro` keyword defaulting to
this accessor, so `train!(e; gradient_clip_norm = 1f0)` overrides it for one run.

**Global norm over the fully accumulated gradient, applied at the top of the optimizer program,
never a chain member.** A chain member would clip **per group**, which is a different
algorithm producing different updates; a user who genuinely wants that can put an admitted
`ClipNorm(ω, p; throw = false)` in a Level 2 chain.

It is **not** a [`Device`](@ref), not schedulable, and not per group, matching Lightning. It is a
**trace-time host constant**, which is what buys the property that a disabled clip emits no ops at
all. The price is that changing it recompiles the optimizer program, and *only* the optimizer
program, so a clip sweep re-pays the cheap compile rather than the expensive one.

**`0` means off, and a traced threshold could not keep that**: with the threshold device-resident,
`0` does not disable the clip, it scales the gradient to zero norm, and on an all-zero gradient
`0/0` yields `NaN` silently. "Off" would have to be spelled `Inf`.

The name matches Lightning's `gradient_clip_*` family.
"""
function gradient_clip_norm end

gradient_clip_norm(e) = _field(e, :gradient_clip_norm, 0.0f0)

"""
    nonschedulable(::Type{R}) -> NTuple{N,Symbol}

The fields of rule type `R` that may **not** be scheduled, and therefore are **not** promoted to
device residency by `to_device_rule`. Default `()`.

**One declaration drives both, and it must**: a field that cannot be promoted cannot be scheduled,
and a field that is scheduled must be promoted. Declaring them separately would let them drift.

The shipped methods:

```julia
nonschedulable(::Type{<:Optimisers.Adam})     = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.RAdam})    = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.AdamW})    = (:beta, :epsilon, :couple)
nonschedulable(::Type{<:Optimisers.ClipNorm}) = (:p, :throw)
nonschedulable(::Type{<:Decay})               = (:anchor, :no_decay_mask)
```

`beta` is excluded because it is a `Tuple` and a scalar schedule cannot produce one; `p` because it
is structural, and promoting it breaks `_norm`'s `::Real` dispatch, which is what once made
`ClipNorm` look unable to trace at all. **The general rule: any parameter-sized rule field is
non-schedulable**, since scheduling one would push a parameter-sized buffer to device every step.

**Declare one method per concrete rule type, never a `Union`.** A `Union` method is shadowed by any
more specific one, silently and with no ambiguity warning.

The set for a chain is the union over its rules, computed automatically, so a custom chain declares
nothing.
"""
function nonschedulable end

nonschedulable(::Type) = ()

# ONE METHOD PER CONCRETE RULE TYPE, NEVER A UNION. A `Union` method is shadowed by any more
# specific one, silently and with no ambiguity warning: with both `nonschedulable(::Type{<:AdamW})`
# and `nonschedulable(::Type{<:Union{Adam,RAdam,AdamW}})` defined, AdamW resolves to the first alone,
# so `beta` silently becomes schedulable for that one rule and a scalar schedule gets splatted into a
# Tuple field. The repetition below is the price of not being able to fail that way.
#
# `beta` is excluded because it is a `Tuple` and a scalar schedule cannot produce one.
# `epsilon` is excluded as a numerical floor rather than a knob. `couple` is a `Bool`.
# ClipNorm's `p` is STRUCTURAL: promoting it breaks `_norm`'s ::Real dispatch, which is what once
# made this project record "ClipNorm cannot trace".
nonschedulable(::Type{<:Optimisers.Adam}) = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.RAdam}) = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.AdamW}) = (:beta, :epsilon, :couple)
nonschedulable(::Type{<:Optimisers.ClipNorm}) = (:p, :throw)

# ── Schedules ───────────────────────────────────────────────────────────────────────

"""
    schedules(e) -> NamedTuple

Every entry is a **factory of the horizon**. Default `(;)`, i.e. everything constant.

```julia
schedules(e::MyExp) = (;
    eta        = total -> OneCycle(total, 1f-3),               # factory of the horizon
    aux_weight = _ -> (t -> max(0f0, 1f0 - t / 5000)),         # ignores the horizon
    epsilon    = 1f-8,                                         # a bare Number is a constant
)
```

The framework calls `f(total)` **once**, at setup, then `sched(step)` once per optimizer step. A
bare `Number` normalizes to `_ -> (_ -> value)`.

**The schedule belongs to the experiment by default**, because it is part of the recipe. The `Nitro`
keyword of the same name is the per-run override and **replaces wholesale; it does not merge**. To
merge, say so: `schedules = merge(schedules(e), (; eta = ...))`. The binding report names the
source of every entry.

**A schedule key is either a [`Device`](@ref) field name or a rule field name**, resolved
against the union of the two. `opt` keys are applied to the optimizer, `device` keys are written
into the experiment and reach traced code as `e.field`; there is no third destination. Both names
are **reserved** at the top level and are how an ambiguous key is qualified:

```julia
schedules = (; eta = total -> OneCycle(total, 1f-3),
               device = (; lambda = _ -> t -> 0.5f0),    # e.lambda
               opt     = (; lambda = _ -> t -> 1f-4))     # the decay coefficient
```

Rule field names are the rule's **actual** field names, so the learning-rate key is `eta` rather than
`lr`.

**A nested `opt` key is a parameter-group path.** `opt = (; backbone = (; eta = ...))` names the
`:backbone` group and binds that group's chain only, so per-group learning-rate curves work: the
backbone anneals while the head warms up. The path value is the BASE curve for those groups and the
per-group ratio is still applied,
`η_g(t) = opt.backbone.eta(t) · (learning_rate(e, Val(:backbone)) / learning_rate(e))`, exactly like
a bare key; a group with no path key falls back to the bare key, then to its base rate. Paths are
ONE level (`group.field`), because parameter groups are flat, and qualified-only: an unqualified
dotted key is a resolution error, since the automatic loop cannot know a top-level key is a group
path. Manual mode is deliberately asymmetric: its path values are absolute, because manual mode has
no base rate and no ratios.

**`step` is the optimizer step, not the micro-batch.** With `accum = N` they differ by a factor of
N, and confusing them shifts the whole curve by N. The horizon is
`total = max_epochs * div(steps_per_epoch, accum)`, an exact division because a training batch
count not divisible by `accum` is a setup error.

Schedules are host-side, so ordinary Julia control flow is fine inside them. A schedule need not be
a pure function of the step, but one closing over training state does not resume exactly, since
resume restores the step counter and not the closure.

`ParameterSchedulers.jl` is a documented recommendation with **no dependency**, not even a weakdep,
because the framework never dispatches on a schedule.
"""
function schedules end

schedules(e) = (;)

# ── The driver-only accessors ───────────────────────────────────────────────────────

"""
    max_epochs(e) -> Int

How many epochs to train for. Default `_field(e, :max_epochs, 1)`, i.e. an experiment's own
`max_epochs` field if it has one, else `1`.

**It is also a `Nitro` keyword defaulting to this accessor**, because raising it on resume is the
normal case. Resolution order is keyword, then a user method on the experiment type, then a field
on `e`, then the framework default; the binding report is where a reader sees which source won.

This runs **host-side against the real `e`**, never against [`compile_view`](@ref)'s stripped view,
which is why the field is normally marked [`Host`](@ref): a driver knob has no business in the
compile cache key.

**A bare experiment therefore finishes after one epoch.** Stated because a first run stopping there
reads as a bug otherwise.
"""
function max_epochs end

max_epochs(e) = _field(e, :max_epochs, 1)

# ── The run accessors ───────────────────────────────────────────────────────────────
#
# Every one of these is a defaulted accessor whose `Nitro` keyword DEFAULTS TO THE ACCESSOR CALL, so
# the two compose exactly as `max_epochs`, `schedules`, and `gradient_clip_norm` already do: declare
# it on the experiment and omit the keyword, or pass the keyword to replace it for one run.
#
# WHY THIS IS THE GENERAL RULE. The composition was once a per-value decision, on two grounds: the
# per-group accessors have no sensible keyword form, and a bare `learning_rate` keyword would
# compete with `schedules` for the same quantity. Neither ground touches the values below. They are
# single-valued, they compete with nothing, and every one of them is a property of the experiment
# that a REPL user would otherwise retype on every call.
#
# WHAT STAYS KEYWORD-ONLY, and why it is a line rather than an omission: `data`, `checkpoint`,
# `resume`, and `run_ref` each name a fact about THIS INVOCATION rather than a property of the
# experiment. `data` already has its accessor and it is called `build_data`; `checkpoint` and
# `resume` point at a file this call should read; `run_ref` is a channel back to the caller.

"""
    seed(e) -> Int

The rng seed. Default `_field(e, :seed, 42)`; also a `Nitro` keyword.

**A `seed` field must be `Host`**, and unmarked already is: the seed is kept out of the config
hash so that a seed sweep shares one compiled program, and a `GraphConst` field of that name would
enter the compile cache key and recompile per seed. That is checked at setup rather than left to
the reader, since the whole point of the sweep is that it is cheap.
"""
function seed end
seed(e) = _field(e, :seed, 42)

"""
    run_dir(e) -> String

The run's output directory. Default `_field(e, :run_dir, joinpath("runs", string(nameof(typeof(e)))))`;
also a `Nitro` keyword. Checkpoints, the manifest, and `resume = :auto` all resolve against it.

Note the name is shared with [`run_dir`](@ref)`(nitro)`, which reads the value a run actually
resolved to. They are the same concept at two moments: what the experiment asks for, and what this
run got. Dispatch separates them.
"""
run_dir(e) = _field(e, :run_dir, default_run_dir(e))

"""
    accum(e) -> Int

Micro-batches per optimizer step. Default `_field(e, :accum, 1)`; also a `Nitro` keyword.

Unlike the other run accessors this one **bakes**: it reaches the gradient program as
`inv_n = 1/accum`, a trace-time host constant, so it is in the cache key through this accessor's
`primary_world` exactly as `gradient_clip_norm` is. A `GraphConst` field of this name is therefore
correct rather than a trap, because it genuinely does change the compiled program.
"""
function accum end
accum(e) = _field(e, :accum, 1)

"""
    n_devs(e) -> Int

Local devices to shard the batch over. Also a `Nitro` keyword. `1` skips the mesh entirely.

**The default is every VISIBLE device**, `length(Reactant.devices())`, so a host with four GPUs
data-parallelizes without being asked and `CUDA_VISIBLE_DEVICES` is the supported way to restrict a
run. That matches how these hosts are actually driven. With no GPU visible Reactant reports a single
CPU device, so a CPU run defaults to 1 and skips the mesh, which is why the test suite is unaffected.

**A session-level pin from [`setup_devices!`](@ref), which is the `nitro_setup` tool in a Kaimon
session, beats the experiment field and the default.** Once a session pins `n_devs = 2`, every run
in that process shards over 2 devices until the pin changes, whatever an experiment declares. An
explicit `n_devs` keyword on `Nitro` still wins for that one run. Without a pin, the experiment's
own field wins over the default.

Note the batch size is GLOBAL and gets split across the mesh, so adding devices buys throughput and
does not change the effective batch.
"""
function n_devs end
# The session-level device-count pin, written by `setup_devices!` (and the `nitro_setup` tool over
# it) and read here. `nothing` = no pin = the documented default below.
const _PINNED_N_DEVS = Ref{Union{Nothing, Int}}(nothing)
function n_devs(e)
    pinned = _PINNED_N_DEVS[]
    return pinned === nothing ? _field(e, :n_devs, length(Reactant.devices())) : pinned
end

"""
    checkpointer(e)

The run's checkpointer. Default `_field(e, :checkpointer, TopKCheckpointer())`; also a
`Nitro` keyword. `nothing` disables checkpointing and is the documented opt-out.

An experiment whose selection metric is one of its own `metrics` keys rather than `:val_loss` should
say so here, so that every run of it selects the same way without the caller remembering.
"""
function checkpointer end
checkpointer(e) = _field(e, :checkpointer, TopKCheckpointer())

"""
    early_stop(e)

The run's stopping rule. Default `_field(e, :early_stop, nothing)`; also a `Nitro` keyword.
`nothing` means no early stopping, which stays the default so that a bare experiment does not
truncate its own run.
"""
function early_stop end
early_stop(e) = _field(e, :early_stop, nothing)

"""
    logger(e)

The run's logger. Default `_field(e, :logger, JSONLogger())`, i.e. an experiment's own
`logger` field if it has one, else the framework's shipped JSON default; also a `Nitro` keyword.
`nothing` is the documented opt-out and stays the public "no logging" value: pass it as a
keyword, or declare a `logger = nothing` field.

**Called once, at the end of setup.** That matters because constructing a logger is often a side
effect: a file logger opens a handle and a hosted tracker registers a run. One call per `Nitro` is
the contract, so `logger(e) = TSVLog(open(...))` is safe. The default [`JSONLogger`](@ref) is
deliberately side-effect-free at construction: it opens its file lazily, after setup has pinned its
path to the run's resolved `run_dir`, the same adoption the default checkpointer gets.
"""
function logger end
logger(e) = _field(e, :logger, JSONLogger())
