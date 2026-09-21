# Interface.jl
#
# Hook definitions and their defaults. Four hooks are required (`build_data`, `build_model`,
# `forward`, `loss`); every other hook has a default.
#
# Every hook is a bare `function f end` rather than a catch-all method, because two mechanisms read
# the method table: batch routing resolves a hook's declared keywords through `which`, and a
# `kwargs...` catch-all would route the whole batch to every hook; and the missing-`metrics`
# substitution is detected by `metrics` having no method. Defaulted accessors get real methods on
# `::Any`, and `metrics` never gets one.

# ── The four required hooks ─────────────────────────────────────────────────────────

"""
    build_data(e, dist) -> NamedTuple

Required. Return the run's named data collection, `(; train, val)` or `(; train, val, test)`.

A data source is anything iterable that yields concrete `NamedTuple`s of host arrays and supports
`length`, which counts batches: a `Vector` of batches, an `MLUtils.DataLoader`, or a bespoke
sampler. The framework transfers each batch, shards it, and owns prefetching. Prefer
`MLUtils.DataLoader`: its settings are checked at setup from its own fields, the MLUtils extension
fans it out across every thread, and batching, shuffling and collation are its job. A bespoke
sampler puts those back on you.

Three requirements: the source is restartable (setup draws one batch for the schema and the loop
iterates from scratch, so a bare `Channel` loses a batch); `length` is read once at setup to fix
the schedule horizon; and the `train` split drops its partial final batch (`partial = false`,
`drop_last = true`) and its batch count divides by `accum`. Eval splits should not drop theirs,
since the framework pads and slices, and `testmode` makes a standard model per-sample along the
batch axis.

There is no `prepare_epoch!` hook; a loader re-plans in its own iteration initialization. `dist`
is `nothing` today; write it untyped. `build_data` sees the pre-conversion experiment, and is
skipped when `Nitro(e; data = ...)` supplies the collection.
"""
function build_data end

"""
    build_model(e, rng) -> (model, ps, st)

Required. Standard Lux. Pretrained loading must happen inside `build_model`, because the framework
captures `w0` immediately after it returns and `decay_anchor = :w0` would otherwise anchor to the
random init. It sees the post-conversion experiment. For a model with variant branches, lift the
selector to a type with [`dispatch_variant`](@ref) so the branches fold at trace time:

```julia
build_model(e, rng) = _build_model(e, dispatch_variant(e, :model_kind), rng)
_build_model(e, ::Val{:bit_r50}, rng) = ...
```
"""
function build_model end

"""
    forward(e, model, ps, st; <declared batch fields>) -> (outputs, st_new)

Required. One definition of the model's computation, three consumers: training traces `forward`
then [`loss`](@ref), validation traces `forward` then [`metrics`](@ref), and [`predict`](@ref) is
`forward` alone, which is also the exported program. `forward` takes inputs, not targets; that is
what makes `predict` possible.

Declare exactly the batch fields you want as keywords; the framework resolves the method once at
setup and passes that subset. A keyword with a default is an optional batch field. A method ending
in `kwargs...` receives the whole batch unchecked.

`outputs` is passed on as one positional argument: whatever is in the first slot here is exactly
what `loss`, `metrics` and `train_metrics` receive. For more than one output, return a
`NamedTuple`, which is also what [`export_outputs`](@ref) names leaves by:

```julia
forward(e, model, ps, st; tokens) = ((; logits, energy), st_new)
loss(e, out; target) = ce(out.logits, target) + e.w * mean(abs2, out.energy)
```

Every array leaf of the output tree must have the batch dimension last, and must be an array: the
short final eval batch is padded and sliced back along the last axis, and a scalar already reduced
over the batch has nothing to slice. State always threads (`st_new` is `st` for a stateless model);
the framework owns the train/eval mode switch, and the eval-mode `st_new` is discarded.
"""
function forward end

"""
    loss(e, outputs; <declared batch fields>) -> scalar

Required. The scalar Enzyme differentiates, traced with [`forward`](@ref). `outputs` is the first
element of `forward`'s return; do not unpack it again, since `first(outputs)` is element one of the
prediction array and the run would train on one sample without raising. `loss`'s router is also
what the default validation metric reuses when an experiment defines no [`metrics`](@ref).
"""
function loss end

# ── Metrics ─────────────────────────────────────────────────────────────────────────

"""
    metrics(e, outputs; <declared batch fields>) -> NamedTuple of (sum, count)

Once per eval batch. A metric reports its own numerator and denominator, and the framework adds
them up and divides at the end, since different metrics have different natural denominators;
`count === nothing` accumulates by summation without dividing, for a confusion matrix.

```julia
metrics(e, outputs; lab) = (; err = (sum_abs_err, n_items),
                              acc = (n_correct,   n_images))
```

`metrics` never sees padding. When the user defines none, the framework reports
`val_loss = (loss(e, outputs; ...), 1)`, the mean over batches. The validation/testing distinction
lives host-side in [`finalize_metrics`](@ref), since an argument here would compile two programs.
"""
function metrics end

"""
    metrics_residency(e, hook::Symbol) -> :host | :device

Where a metric hook runs. `hook` is `:metrics` or `:train_metrics`; the defaults are `:host` for
`metrics` and `:device` for `train_metrics`, following the cadence. `train_metrics` runs once per
micro-batch, where transferring a full output batch dwarfs the scalars wanted; `metrics` runs once
per eval batch against an epoch of stepping, where the transfer is noise, and a host metric can use
anything ordinary Julia can (matching, sorting, a library that knows nothing about Reactant).

```julia
ReactantNitro.metrics_residency(::MyExp, ::Symbol) = :device    # a traceable validation metric
```

What changes for the hook author is whether `outputs` holds device or host arrays. A `:host` hook is
part of no program, so editing it recompiles nothing.
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

Once per micro-batch, inside the gradient program. Default `(;)`, which folds away. Scalars per
step with no reduction, for an instantaneous diagnostic; smoothing is the logger's job. Readback is
lazy, so the values cost a transfer only on logged steps.
"""
function train_metrics end

"""
    finalize_metrics(e, acc, split) -> NamedTuple

Host-side, once per split per epoch; default identity. For a derived metric that is not the mean
of per-batch values, and for branching on `split`, which is free here:

```julia
finalize_metrics(e, acc, split) = (; f1 = 2acc.tp / (2acc.tp + acc.fp + acc.fn))
```
"""
function finalize_metrics end

finalize_metrics(e, acc, split) = acc

# ── Manual training mode ────────────────────────────────────────────────────────────

"""
    train_step(e, model, ps, opt_state, st; <declared batch fields>)
        -> (; loss, ps, st, opt_state, stats)

Define this method for an experiment type to switch `train!` to manual mode; the automatic loop has
no `train_step`, so writing one is what selects the mode, and [`manual_training`](@ref)`(e) =
false` declines it. Manual mode hands you the whole optimizer step (forwards, backwards, optimizer
steps, device metrics) while the framework keeps everything outside it: the batch stream and
prefetch, validation, checkpointing, logging, phases, early stopping.

Called once per optimizer step, in train mode, on one transferred device batch whose fields are
routed from the declared keywords. It is traced and compiled, so it must be a stable named method.
Gradients inside are computed with [`backward`](@ref)`(f, ps_sub, consts...)`, whose objective must
take every traced value it reads as an argument: a closure capture of a traced value is silently a
constant and yields a zero gradient with no error.

A complete minimal GAN, generator and discriminator with their own optimizers and the
non-saturating formulation falling out of the activities, is worked through in the manual-mode page
(see [The very simple GAN](@ref "The very simple GAN")).

The return is a checked `NamedTuple`: `loss` (required; validated and logged), `ps`, `st`,
`opt_state` (the new optimizer STATES only, since a rule's replicated device scalars cannot leave a
compiled program on a mesh; the driver re-attaches the rules), and `stats` (device scalars logged
once per step, `(;)` allowed).

[`schedules`](@ref) works in manual mode with one difference: `opt` keys are path-bound into the
`opt_state` [`setup_optimizers`](@ref) returned, since there are no parameter groups. A bare key
(`opt.eta`) binds every rule with that field at the absolute value; a nested key (`opt.gen.eta`) is
a path into the tree. `accum > 1` is a setup error, and [`train_metrics`](@ref) is unused; `stats`
come from the closure. `loss` becomes optional.
"""
function train_step end

"""
    setup_optimizers(e, model, ps, st, mesh) -> opt_state

Required for a manual experiment. Build the optimizer states the user owns, one per network or
group, called once at construction after device conversion and `build_model`. The framework
normalizes the result to device residency and asserts no host `Number` survives in any leaf's
state, so an integer step counter cannot freeze under trace.

```julia
ReactantNitro.setup_optimizers(e::MyGAN, model, ps, st, mesh) = (;
    gen  = Optimisers.setup(Optimisers.AdamW(1.0f-4), ps.gen),
    disc = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.disc))
```
"""
function setup_optimizers end

"""
    manual_training(e) -> Bool

The manual-mode flag. Default: whether the experiment type has a [`train_step`](@ref) method.
Override to `false` to keep the method in source while using the automatic loop. Read once at
construction and frozen; flipping it on an existing handle is reported by `fixed_config_report`.
"""
function manual_training end

manual_training(e) =
    hasmethod(train_step, Tuple{typeof(compile_view(e)), Any, Any, Any, Any})

# ── Derived values ──────────────────────────────────────────────────────────────────

"""
    derive(e, data) -> NamedTuple

Values that genuinely depend on the dataset, merged into the experiment. Default `(;)`. Runs before
device conversion, so it returns host values and sees the pre-conversion experiment; a derived
[`Device`](@ref) is excluded from the cache key and a derived `GraphConst` is baked into it. Prefer
constants for structural values and derive only numerics, since a shape-determining derived value
makes the compiled program a function of the data. Resume recomputes rather than restores.

```julia
derive(e::MyExp, data) = (; class_weights = inv_freq(data))
```
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

The parameter group a leaf belongs to, as a function of its keypath. Default `:default` for every
keypath, the single-group case. The group is the unit at which optimizer behavior is defined;
`:default` is group 1 and the rest follow first-appearance order, which bakes into the program.

```julia
param_group(::MyExp, ks) = ks[1] === :backbone ? :backbone : :default
```
"""
function param_group end

# The single-group case, so a bare experiment never defines `param_group`.
param_group(e, keypath) = :default

"""
    optimizer(e) -> Type{<:Optimisers.AbstractRule}
    optimizer(e, group::Symbol, hp) -> Optimisers.AbstractRule

The three-level optimizer surface.

Level 0: declare nothing and get `RAdam`, with per-group hyperparameters from
[`learning_rate`](@ref) and [`lambda`](@ref). RAdam rectifies the adaptive variance term instead
of leaving the user to hand-tune a warmup; it needs Optimisers 0.4.8, the first release whose
Reactant extension traces it.

Level 1: return a rule type, and the framework constructs it, splatting the resolved values in by
field name, with the [`Decay`](@ref) tail, per-group accessors and no-decay mask untouched. A rule
declaring its own `lambda` (`AdamW`) is rejected here, since it would decay twice.

```julia
optimizer(::MyExp) = Optimisers.Momentum
optimizer(::MyExp, ::Val{:backbone}) = Optimisers.Momentum
```

Level 2: supply the chain, receiving the group and its resolved, device-converted `hp`:

```julia
optimizer(::MyExp, group::Symbol, hp) = Optimisers.OptimiserChain(
    Optimisers.RAdam(; eta = hp.eta, beta = hp.beta),
    Decay(hp.lambda, hp.anchor, hp.no_decay_mask),
)
```

`hp` carries `eta` (the group's effective learning rate), `lambda` (already multiplied by it),
`anchor`, `no_decay_mask`, and one key per scheduled rule field. Level 2 owns hyperparameter
application: a factory that ignores a scheduled `epsilon` passes the name check and the schedule
does nothing, which the binding report shows per group rather than the framework erroring on. No
path accepts a fully constructed chain with baked values, since rules carry device scalars rebuilt
every step.
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

The base learning rate (default `1f-3`) and the per-group bases (default `learning_rate(e)`).
Per-group accessors define ratios and the schedule sets the default group's absolute value:
`η_g(t) = eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))`. `learning_rate(e) == 0`
is a setup error, since the ratio is undefined.
"""
function learning_rate end

# Plain values rather than `_field` reads: an experiment may declare a `lambda::Device{Float32}`
# that is a loss weight, which a `_field` default would silently read as the decay coefficient.
learning_rate(e) = 1.0f-3
learning_rate(e, ::Val{g}) where {g} = learning_rate(e)

"""
    lambda(e) -> Real
    lambda(e, ::Val{group}) -> Real

The decoupled decay coefficient, per group; default `0`, so there is no [`Decay`](@ref) in the
chain at all. What it decays toward is that group's [`decay_anchor`](@ref), so `:w0` on a
pretrained backbone and `:zero` on a fresh head are one rule with different anchors, driven by one
`opt.lambda` schedule. The value reaching the rule is pre-multiplied by the group's effective
learning rate. Scheduling `opt.lambda` while also defining per-group `lambda` accessors is a setup
error.
"""
function lambda end

lambda(e) = 0.0f0
lambda(e, ::Val{g}) where {g} = lambda(e)

"""
    decay_anchor(e, ::Val{group}) -> :zero | :w0 | AbstractArray

What this group's decay pulls toward. Default `:zero`, ordinary weight decay. `:w0` decays toward
the parameters as `build_model` returned them, which is L2-SP (Li, Grandvalet and Davoine, ICML
2018, [arXiv:1802.01483](https://arxiv.org/abs/1802.01483)); an explicit array anchors to B while
initializing from A. `:w0` on a randomly initialized group anchors it to a random point, and the
framework cannot detect which groups are pretrained, which is why `:zero` is the default. Anchoring
puts a per-group checksum in the checkpoint record, which resume verifies; the arrays are not
stored, so an anchored run with a nondeterministic init cannot be resumed.
"""
function decay_anchor end

# `:zero` rather than `:w0`, because anchoring blind would be wrong for any group that was not
# pretrained, and the framework cannot detect which groups are.
decay_anchor(e, ::Val{g}) where {g} = :zero

"""
    ReactantNitro.default_no_decay(keypath, param) -> Bool

The built-in decay exclusion: every 1-D parameter, which in Lux is biases and every normalization
layer's scale and shift. Exported so [`no_decay`](@ref) can extend it:
`ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || (:fc in ks)`.
"""
default_no_decay(ks, x) = ndims(x) == 1

"""
    no_decay(e, keypath::Tuple, param) -> Bool

Whether this parameter leaf is excluded from weight decay. Default [`default_no_decay`](@ref). It
receives the leaf itself as well as its keypath, so `ndims` and `size` are fair game; it runs
host-side once at setup.

```julia
ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || (:fc in ks)   # extend
ReactantNitro.no_decay(::MyExp, ks, x) = (:fc in ks) && (:weight in ks)          # replace
ReactantNitro.no_decay(::MyExp, ks, x) = false                                   # decay everything
```

An excluded leaf gets a mask of 0, and [`Decay`](@ref) computes `mask * lambda * (x - anchor)`, so
exclusion dominates the anchor for free. Exclusion is per leaf while the coefficient and anchor are
per group, because biases occur inside every group and splitting groups for them would also split
the learning-rate ratio. Not in the compile cache key: the mask is a device buffer passed as a
value, so it changes no graph. Resolved once at construction.
"""
function no_decay end

no_decay(e, ks, x) = default_no_decay(ks, x)

"""
    gradient_clip_norm(e) -> Real

The global-norm clip threshold; default `0f0`, off. Also a `Nitro` keyword. Applied to the fully
accumulated gradient at the top of the optimizer program, never as a chain member, which would clip
per group. A trace-time host constant, so a disabled clip emits no ops and changing it recompiles
the optimizer program only; a traced threshold could not make `0` mean off, since `0/0` on an
all-zero gradient yields `NaN`. The name matches Lightning's.
"""
function gradient_clip_norm end

gradient_clip_norm(e) = _field(e, :gradient_clip_norm, 0.0f0)

"""
    nonschedulable(::Type{R}) -> NTuple{N,Symbol}

The fields of rule type `R` that may not be scheduled and are therefore not promoted to device
residency by `to_device_rule`. Default `()`. One declaration drives both, since a field that cannot
be promoted cannot be scheduled. The shipped methods:

```julia
nonschedulable(::Type{<:Optimisers.Adam})     = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.RAdam})    = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.AdamW})    = (:beta, :epsilon, :couple)
nonschedulable(::Type{<:Optimisers.ClipNorm}) = (:p, :throw)
nonschedulable(::Type{<:Decay})               = (:anchor, :no_decay_mask)
```

`beta` is a `Tuple`; `p` is structural and promoting it breaks `_norm`'s `::Real` dispatch; any
parameter-sized field is non-schedulable. Declare one method per concrete rule type, never a
`Union`, which a more specific method shadows silently.
"""
function nonschedulable end

nonschedulable(::Type) = ()

# One method per concrete rule type, never a `Union`: a `Union` method is shadowed by any more
# specific one with no ambiguity warning, and `beta` would silently become schedulable.
nonschedulable(::Type{<:Optimisers.Adam}) = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.RAdam}) = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.AdamW}) = (:beta, :epsilon, :couple)
nonschedulable(::Type{<:Optimisers.ClipNorm}) = (:p, :throw)

# ── Schedules ───────────────────────────────────────────────────────────────────────

"""
    schedules(e) -> NamedTuple

Every entry is a factory of the horizon, called once at setup as `f(total)`, after which
`sched(step)` runs once per optimizer step; a bare `Number` is a constant. Default `(;)`.

```julia
schedules(e::MyExp) = (;
    eta        = total -> OneCycle(total, 1f-3),               # factory of the horizon
    aux_weight = _ -> (t -> max(0f0, 1f0 - t / 5000)),         # ignores the horizon
    epsilon    = 1f-8,                                         # a bare Number is a constant
)
```

The `Nitro` keyword of the same name replaces this wholesale; to merge, say so with `merge`. A key
is either a [`Device`](@ref) field name or a rule field name (`eta`, not `lr`), and `device` and
`opt` are reserved at the top level to qualify an ambiguous one:

```julia
schedules = (; eta = total -> OneCycle(total, 1f-3),
               device = (; lambda = _ -> t -> 0.5f0),    # e.lambda
               opt     = (; lambda = _ -> t -> 1f-4))     # the decay coefficient
```

A nested `opt` key is a parameter-group path: `opt = (; backbone = (; eta = ...))` binds the
`:backbone` group's chain only, as the base curve with the per-group ratio still applied; paths are
one level, since groups are flat. Manual mode's path values are absolute. `step` is the optimizer
step, and `total = max_epochs * div(steps_per_epoch, accum)` is exact. Schedules are host-side, so
ordinary control flow is fine; one closing over training state does not resume exactly.
`ParameterSchedulers.jl` is a recommendation with no dependency.
"""
function schedules end

schedules(e) = (;)

# ── The driver-only accessors ───────────────────────────────────────────────────────

"""
    max_epochs(e) -> Int

How many epochs to train for. Default `_field(e, :max_epochs, 1)`, so a bare experiment finishes
after one epoch. Also a `Nitro` keyword, since raising it on resume is the normal case; resolution
is keyword, then a user method, then a field on `e`, then the default. Runs host-side against the
real `e`, which is why the field is normally `Host`.
"""
function max_epochs end

max_epochs(e) = _field(e, :max_epochs, 1)

# ── The run accessors ───────────────────────────────────────────────────────────────
#
# Each is a defaulted accessor whose `Nitro` keyword defaults to the accessor call: declare it on
# the experiment and omit the keyword, or pass the keyword to replace it for one run. `data`,
# `checkpoint`, `resume` and `run_ref` stay keyword-only, since each names a fact about the
# invocation rather than the experiment.

"""
    seed(e) -> Int

The rng seed. Default `_field(e, :seed, 42)`; also a `Nitro` keyword. A `seed` field must be
`Host` (unmarked already is), so a seed sweep shares one compiled program; setup checks it.
"""
function seed end
seed(e) = _field(e, :seed, 42)

"""
    run_dir(e) -> String

The run's output directory. Default `joinpath("runs", string(nameof(typeof(e))))`; also a `Nitro`
keyword. Checkpoints, the manifest and `resume = :auto` all resolve against it. Shares its name
with [`run_dir`](@ref)`(nitro)`, which reads what a run actually resolved to.
"""
run_dir(e) = _field(e, :run_dir, default_run_dir(e))

"""
    accum(e) -> Int

Micro-batches per optimizer step. Default `_field(e, :accum, 1)`; also a `Nitro` keyword. Unlike
the other run accessors it bakes, reaching the gradient program as `inv_n = 1/accum`, so a
`GraphConst` field of this name is correct.
"""
function accum end
accum(e) = _field(e, :accum, 1)

"""
    n_devs(e) -> Int

Local devices to shard the batch over; also a `Nitro` keyword. `1` skips the mesh. The default is
every visible device, `length(Reactant.devices())`, so `CUDA_VISIBLE_DEVICES` is the supported way
to restrict a run; with no GPU visible a CPU run defaults to 1. A session pin from
[`setup_devices!`](@ref) (the `nitro_setup` tool) beats the experiment field and the default; an
explicit `n_devs` keyword on `Nitro` still wins for that one run. The batch size is global and is
split across the mesh.
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

The run's logger. Default `_field(e, :logger, JSONLogger())`; also a `Nitro` keyword, and `nothing`
is the "no logging" value. Called once, at the end of setup, so a constructor with side effects (a
file handle, a hosted tracker registering a run) is safe. The default [`JSONLogger`](@ref) opens its
file lazily, after setup has pinned its path to the resolved `run_dir`.
"""
function logger end
logger(e) = _field(e, :logger, JSONLogger())
