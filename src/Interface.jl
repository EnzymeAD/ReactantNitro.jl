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

Required. Return a Lux model with its parameters and state. Load pretrained weights here: the
framework captures `w0` as soon as `build_model` returns, so weights loaded later would leave
`decay_anchor = :w0` anchored to the random init. It sees the post-conversion experiment. For a
model with variant branches, lift the selector to a type with [`dispatch_variant`](@ref) so the
branches fold at trace time:

```julia
build_model(e, rng) = _build_model(e, dispatch_variant(e, :model_kind), rng)
_build_model(e, ::Val{:bit_r50}, rng) = ...
```
"""
function build_model end

"""
    forward(e, model, ps, st; <declared batch fields>) -> (outputs, st_new)

Required. The model's computation, from inputs to outputs. Training traces `forward` then
[`loss`](@ref), validation traces `forward` then [`metrics`](@ref), and [`predict`](@ref) runs
`forward` alone, which is also the exported program. Declare inputs only, never targets, so
`predict` can call it without them.

Declare exactly the batch fields you want as keywords; the framework resolves the method once at
setup and passes that subset. A keyword with a default is an optional batch field. A method ending
in `kwargs...` receives the whole batch unchecked.

`loss`, `metrics` and `train_metrics` receive `outputs`, the first element of the return, as one
positional argument. For more than one output, return a
`NamedTuple`, which is also what [`export_outputs`](@ref) names leaves by:

```julia
forward(e, model, ps, st; tokens) = ((; logits, energy), st_new)
loss(e, out; target) = ce(out.logits, target) + e.w * mean(abs2, out.energy)
```

Every leaf of the output tree must be an array with the batch dimension last: the short final eval
batch is padded and sliced back along the last axis, and a scalar already reduced over the batch
has nothing to slice. Always return the state (`st` itself for a stateless model). The framework
switches between train and eval mode and discards the eval-mode `st_new`.
"""
function forward end

"""
    loss(e, outputs; <declared batch fields>) -> scalar

Required. The scalar Enzyme differentiates, traced together with [`forward`](@ref). `outputs` is
already the first element of `forward`'s return; do not unpack it again, since `first(outputs)` is
the first element of the prediction array and the run would train on one sample without an error.
When an experiment defines no [`metrics`](@ref), the default validation metric calls `loss` with
the same declared batch fields.
"""
function loss end

# ── Metrics ─────────────────────────────────────────────────────────────────────────

"""
    metrics(e, outputs; <declared batch fields>) -> NamedTuple of `value => mode`

The validation metrics of one eval batch. Each metric is `value => mode`, where the mode says how
batches combine:

| mode | across batches | `finalize_metrics` receives |
| --- | --- | --- |
| a count | values and counts summed | `value / count` |
| `:sum` (or `nothing`) | summed | the total |
| `:max`, `:min` | elementwise max or min | the extreme |
| `:concat` | joined along the last axis | an `(…, N)` array, to reduce there |

```julia
metrics(e, outputs; lab) = (; err = sum_abs_err => n_items,
                              cm = confusion => :sum,
                              ranking = (; score = p => :concat, n_pos = count(lab) => :sum))
```

A key may be a group (`NamedTuple` or `Tuple`). A mode on a group covers every leaf below it, and
each leaf of a group must be covered by exactly one mode. `finalize_metrics` receives the same
shape with the modes removed. A `:concat` leaf must have one entry per sample on its last axis, and
must not be returned from `finalize_metrics` unreduced. The tuple `(value, mode)` is the older
spelling of `value => mode` and still works at the top of a key.

`metrics` never sees padding. When the experiment defines none, the framework reports
`val_loss = loss(e, outputs; ...) => 1`, the mean over batches. `metrics` takes no split argument,
since that would compile one program per split; branch on the split in
[`finalize_metrics`](@ref).
"""
function metrics end

"""
    metrics_residency(e, hook::Symbol) -> :host | :device

Where a metric hook runs. `hook` is `:metrics` or `:train_metrics`; the defaults are `:host` for
`metrics` and `:device` for `train_metrics`. `train_metrics` runs once per micro-batch, where
transferring the full outputs would cost far more than the scalars it returns. `metrics` runs once
per eval batch, where the transfer is cheap, and on the host it can use any Julia code (matching,
sorting, a library that knows nothing about Reactant).

```julia
ReactantNitro.metrics_residency(::MyExp, ::Symbol) = :device    # a traceable validation metric
```

For the hook author, the difference is whether `outputs` holds device or host arrays. A `:host`
hook is part of no program, so editing it recompiles nothing.
"""
function metrics_residency end
metrics_residency(e, hook::Symbol) = hook === :train_metrics ? :device : :host

"""
    ReactantNitro.check_residency(e, hook) -> Symbol

Read [`metrics_residency`](@ref) and raise on anything but `:host` or `:device`; a typo would
otherwise select the default without an error.
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

Per-step training diagnostics, computed once per micro-batch. Default `(;)`, which compiles to
nothing. At the default residency (`:device`, see [`metrics_residency`](@ref)) it runs inside the
gradient program. Each value is a scalar logged as is, with no reduction across batches; smoothing
is the logger's job. Readback is lazy, so the values cost a transfer only on logged steps.
"""
function train_metrics end

"""
    finalize_metrics(e, acc, split) -> NamedTuple

Turn one split's combined metrics into the reported values. Runs on the host, once per split per
evaluation; the default returns `acc` unchanged. `acc` has the shape of [`metrics`](@ref)' return
with each value combined across batches as its mode says, and `split` is the split's name, such as
`:val`. Use it for a metric that is not a mean of per-batch values, and to branch on `split`, which
costs no compile here:

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

Define this method for an experiment type to switch `train!` to manual mode;
[`manual_training`](@ref)`(e) = false` keeps the automatic loop anyway. In manual mode you write
the whole optimizer step (forward and backward passes, optimizer updates, device metrics), and the
framework keeps everything around it: the batch stream and prefetch, validation, checkpointing,
logging, phases, early stopping.

Called once per optimizer step, in train mode, on one device batch whose fields are routed from the
declared keywords. It is traced and compiled, so it must be a named method. Compute gradients with
[`backward`](@ref)`(f, ps_sub, consts...)`; its objective must take every traced value it reads as
an argument, because a traced value captured by a closure becomes a constant and yields a zero
gradient with no error.

The manual-mode page works through a minimal GAN, with a generator and a discriminator on their own
optimizers (see [A minimal GAN](@ref "A minimal GAN")).

The return is a `NamedTuple`, checked every step: `loss` (required; validated and logged), `ps`,
`st`, `opt_state` (the new optimizer states only, since a rule's replicated device scalars cannot
leave a compiled program on a mesh; the driver re-attaches the rules), and `stats` (device scalars
logged on each train line, `(;)` allowed).

[`schedules`](@ref) works in manual mode, but `opt` keys bind into the `opt_state`
[`setup_optimizers`](@ref) returned, since there are no parameter groups, and every value is
absolute. A bare key (`opt.eta`) sets that field on every rule that has it; a nested key
(`opt.gen.eta`) is a path into the tree. `accum > 1` is a setup error, and
[`train_metrics`](@ref) is unused, since `stats` replaces it. The `loss` hook is optional; it is
needed only for the default `val_loss` when [`metrics`](@ref) is not defined.
"""
function train_step end

"""
    setup_optimizers(e, model, ps, st, mesh) -> opt_state

Required for a manual experiment. Build the optimizer states, typically one per network. Called
once at construction, after `build_model`. The framework moves the result to the device and
asserts that no host `Number` remains in any leaf's state, so an integer step counter cannot freeze
under trace.

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

Field values computed from the dataset, merged into the experiment. Default `(;)`. Every key must
name an existing field. Runs before device conversion, so it sees the pre-conversion experiment and
returns host values; a derived [`Device`](@ref) field stays out of the cache key and a derived
`GraphConst` field is baked into it. Derive numerics and keep structural values as constants, since
a derived shape makes the compiled program depend on the data. Resume recomputes derived values;
`weights` from a checkpoint record restores them and skips `derive`.

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
keypath, a single group. Each group gets its own optimizer chain, learning-rate ratio, decay and
anchor. `:default` is group 1 and the rest are numbered in order of first appearance, which bakes
into the program.

```julia
param_group(::MyExp, ks) = ks[1] === :backbone ? :backbone : :default
```
"""
function param_group end

# The single-group case, so a bare experiment never defines `param_group`.
param_group(e, keypath) = :default

"""
    optimizer(e) -> Type{<:Optimisers.AbstractRule}
    optimizer(e, ::Val{group}) -> Type{<:Optimisers.AbstractRule}
    optimizer(e, group::Symbol, hp) -> Optimisers.AbstractRule

The optimizer rule, at one of three levels of control.

Level 0: declare nothing and get `RAdam`, with per-group hyperparameters from
[`learning_rate`](@ref) and [`lambda`](@ref). RAdam rectifies the adaptive variance term, so no
hand-tuned warmup is needed; it needs Optimisers 0.4.8, the first release whose Reactant extension
traces it.

Level 1: return a rule type, for every group or per group, and the framework constructs it, passing
the resolved values by field name. The [`Decay`](@ref) tail, the per-group accessors and the
no-decay mask still apply. A rule with its own `lambda` field (`AdamW`) is rejected here, since it
would decay twice.

```julia
optimizer(::MyExp) = Optimisers.Momentum
optimizer(::MyExp, ::Val{:backbone}) = Optimisers.Momentum
```

Level 2: build the chain yourself from the group and its resolved, device-converted `hp`. It is
called at setup and again every optimizer step:

```julia
optimizer(::MyExp, group::Symbol, hp) = Optimisers.OptimiserChain(
    Optimisers.ClipNorm(1.0f0, 2.0f0, false),
    Optimisers.Momentum(hp.eta, get(hp, :rho, 0.9f0)),
)
```

`hp` carries `eta` (the group's effective learning rate), `lambda` (already multiplied by it, and
`nothing` when it is zero and not scheduled), `anchor`, `no_decay_mask`, and one key per scheduled
rule field, present from the first call. A scheduled field reaches the rule only if the factory
reads it from `hp`. Setup checks this on the host: a scheduled field whose value does not change
the returned chain is a setup error, and the binding report shows where each scheduled field
landed. No level accepts a prebuilt chain with fixed values, since the rules carry device scalars
that are rebuilt every step.
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

The decoupled weight decay coefficient, per group. Default `0`, which adds no [`Decay`](@ref) to
the chain. Each group decays toward its own [`decay_anchor`](@ref), so `:w0` on a pretrained
backbone and `:zero` on a fresh head share one rule and one `opt.lambda` schedule. The value
reaching the rule is multiplied by the group's effective learning rate. Scheduling `opt.lambda`
while also defining per-group `lambda` accessors is a setup error.
"""
function lambda end

lambda(e) = 0.0f0
lambda(e, ::Val{g}) where {g} = lambda(e)

"""
    decay_anchor(e, ::Val{group}) -> :zero | :w0 | AbstractArray

What this group's decay pulls toward. Default `:zero`, ordinary weight decay. `:w0` decays toward
the parameters as `build_model` returned them, which is L2-SP (Li, Grandvalet and Davoine, ICML
2018, [arXiv:1802.01483](https://arxiv.org/abs/1802.01483)). An explicit array, aligned with the
group's flat parameter slice, anchors to values other than the initialization. `:w0` on a randomly
initialized group anchors it to a random point, and the framework cannot tell which groups are
pretrained, so `:zero` is the default. Anchoring puts a per-group checksum in the checkpoint
record, which resume verifies; the anchor arrays are not stored, so an anchored run whose init is
not deterministic cannot be resumed.
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
receives the leaf as well as its keypath, so it can test `ndims` and `size`; it runs on the host,
once at setup.

```julia
ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || (:fc in ks)   # extend
ReactantNitro.no_decay(::MyExp, ks, x) = (:fc in ks) && (:weight in ks)          # replace
ReactantNitro.no_decay(::MyExp, ks, x) = false                                   # decay everything
```

An excluded leaf gets a mask of 0, and [`Decay`](@ref) computes `mask * lambda * (x - anchor)`, so
exclusion overrides the anchor. Exclusion is per leaf while the coefficient and anchor are per
group, because biases occur in every group and splitting groups for them would also split the
learning-rate ratio. The mask is a device buffer passed as a value, so it is not in the compile
cache key and changing it recompiles nothing. Resolved once at construction.
"""
function no_decay end

no_decay(e, ks, x) = default_no_decay(ks, x)

"""
    grad_rule(e) -> GradRule
    grad_rule(e, hp) -> GradRule

Which gradient each micro-batch contributes: [`SingleGrad`](@ref) by default, or a
[`PerturbRule`](@ref) such as [`SAM`](@ref). Its fields are host numbers, never `Device` fields.

The one-argument form returns the rule, and a scheduled field (`rho = ...` or
`grad = (; rho = ...)`) is filled in by name. The two-argument form mirrors a Level 2
[`optimizer`](@ref): `hp` holds this step's scheduled rule fields by name and `bind` values as
`hp.bind.<name>`, all host numbers, and you place them:

```julia
ReactantNitro.grad_rule(::MyExp) = SAM(0.05f0)
ReactantNitro.grad_rule(::MyExp, hp) = SAM(0.05f0 * hp.bind.sam_warm)
```

Called on the host once per optimizer step; the framework moves the rule's fields to the device, so
a changing value never recompiles. The rule's type must not change. Manual mode supports
`SingleGrad` only.
"""
function grad_rule end

grad_rule(e) = SingleGrad()
grad_rule(e, hp) = grad_rule(e)

"""
    gradient_clip_norm(e) -> Real

The global-norm gradient clip threshold. Default `_field(e, :gradient_clip_norm, 0f0)`, where `0`
means off; also a `Nitro` keyword. Applied to the fully accumulated gradient at the start of the
optimizer program, not as a chain member, which would clip per group. It is a trace-time host
constant, so a disabled clip emits no ops and changing it recompiles the optimizer program only. A
traced threshold could not make `0` mean off, since `0/0` on an all-zero gradient yields `NaN`. The
name matches Lightning's.
"""
function gradient_clip_norm end

gradient_clip_norm(e) = _field(e, :gradient_clip_norm, 0.0f0)

"""
    nonschedulable(::Type{R}) -> NTuple{N,Symbol}

The fields of rule type `R` that may not be scheduled. `to_device_rule` also leaves them on the
host, so one declaration decides both. Default `()`. The shipped methods:

```julia
nonschedulable(::Type{<:Optimisers.Adam})     = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.RAdam})    = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.AdamW})    = (:beta, :epsilon, :couple)
nonschedulable(::Type{<:Optimisers.ClipNorm}) = (:p, :throw)
nonschedulable(::Type{<:Decay})               = (:anchor, :no_decay_mask)
```

`beta` is a `Tuple`; `p` is structural and promoting it breaks `_norm`'s `::Real` dispatch; any
parameter-sized field is non-schedulable. Declare one method per concrete rule type, never a
`Union`: a more specific method shadows a `Union` method with no ambiguity warning.
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

The values that vary with the optimizer step, keyed by rule field or [`Device`](@ref) field.
Default `(;)`. Every entry is a factory of the horizon: setup calls `f(total)` once, and the
returned `sched(step)` runs once per optimizer step. A bare `Number` is a constant.

```julia
schedules(e::MyExp) = (;
    eta        = total -> OneCycle(total, 1f-3),               # factory of the horizon
    aux_weight = _ -> (t -> max(0f0, 1f0 - t / 5000)),         # ignores the horizon
    lambda     = 1f-4,                                         # a bare Number is a constant
)
```

Also a `Nitro` keyword, which replaces this table whole; to extend it, pass
`merge(schedules(e), (; ...))`. Unlike the other run accessors, it reads no field of `e`. A key is
a `Device` field name or a rule field name (`eta`, not `lr`); `device` and `opt` are reserved at
the top level to qualify a name that is both:

```julia
schedules = (; eta = total -> OneCycle(total, 1f-3),
               device = (; lambda = _ -> t -> 0.5f0),    # e.lambda
               opt     = (; lambda = _ -> t -> 1f-4))     # the decay coefficient
```

A nested `opt` key is a parameter-group path: `opt = (; backbone = (; eta = ...))` binds the
`:backbone` group's chain only, with that group's learning-rate ratio still applied. Paths are one
level, since groups are flat. In manual mode a path is a path into `opt_state` and its values are
absolute. `step` is the optimizer step, and `total = max_epochs * div(length(train), accum)`, which
divides exactly because setup requires it. Schedules run on the host, so ordinary control flow is
fine; one that closes over training state does not resume exactly. `ParameterSchedulers.jl` works
here and is not a dependency.
"""
function schedules end

schedules(e) = (;)

# ── The driver-only accessors ───────────────────────────────────────────────────────

"""
    max_epochs(e) -> Int

How many epochs to train for. Default `_field(e, :max_epochs, 1)`, so a bare experiment trains one
epoch. Also a `Nitro` keyword, the usual way to raise it on resume. Resolution order: the keyword,
a user method, a field on `e`, the default. It runs on the host against the real `e`, so a
`max_epochs` field must not be `GraphConst`; leave it unmarked (`Host`). Setup checks this.
"""
function max_epochs end

max_epochs(e) = _field(e, :max_epochs, 1)

# ── The run accessors ───────────────────────────────────────────────────────────────
#
# Each is a defaulted accessor whose `Nitro` keyword defaults to the accessor call: declare it on
# the experiment and omit the keyword, or pass the keyword to replace it for one run. `data`,
# `weights`, `resume` and `run_ref` stay keyword-only, since each names a fact about the
# invocation rather than the experiment.

"""
    seed(e) -> Int

The global RNG seed, set before `build_data`. Default `_field(e, :seed, 42)`; also a `Nitro`
keyword. A `seed` field must not be `GraphConst`, so that a seed sweep shares one compiled program;
leave it unmarked (`Host`). Setup checks this. A resume restores the record's seed, with a warning
if it differs.
"""
function seed end
seed(e) = _field(e, :seed, 42)

"""
    run_dir(e) -> String

The run's output directory. Default `_field(e, :run_dir, joinpath("runs",
string(nameof(typeof(e)))))`, relative to the working directory; also a `Nitro` keyword.
Checkpoints, the manifest, the default logger's file and `resume = :auto` all resolve against it.
[`run_dir`](@ref)`(nitro)` returns the directory a handle resolved to.
"""
run_dir(e) = _field(e, :run_dir, default_run_dir(e))

"""
    accum(e) -> Int

Micro-batches per optimizer step. Default `_field(e, :accum, 1)`; also a `Nitro` keyword. It bakes
into the gradient program as the constant `inv_n = 1/accum`, so changing it recompiles that
program, and a `GraphConst` field of this name is allowed. The `train` split's batch count must
divide by it.
"""
function accum end
accum(e) = _field(e, :accum, 1)

"""
    n_devs(e) -> Int

The number of local devices to shard the batch over; also a `Nitro` keyword. `1` skips the mesh.
Default `_field(e, :n_devs, length(Reactant.devices()))`, every visible device, so set
`CUDA_VISIBLE_DEVICES` to restrict a run; with no GPU visible it is 1. A session pin from
[`setup_devices!`](@ref) (the `nitro_setup` tool) takes precedence over the field and the default,
and an explicit `n_devs` keyword on `Nitro` takes precedence over the pin. The batch size is global
and is split across the devices, so it must divide by `n_devs`.
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

The run's checkpointer. Default `_field(e, :checkpointer, Checkpointer())`; also a `Nitro`
keyword. `nothing` disables checkpointing. Called once per construction, and not at all when the
keyword is given.

To rank checkpoints by one of the experiment's own `metrics` keys instead of `:val_loss`, define
this accessor, so every run ranks the same way:

```julia
ReactantNitro.checkpointer(::MyExp) = Checkpointer(metric = :f1, mode = :max)
```
"""
function checkpointer end
checkpointer(e) = _field(e, :checkpointer, Checkpointer())

"""
    early_stop(e)

The run's early-stopping rule. Default `_field(e, :early_stop, nothing)`, no early stopping; also a
`Nitro` keyword. A rule is checked once per epoch after validation, so setup refuses one on a
handle with no `val` split.
"""
function early_stop end
early_stop(e) = _field(e, :early_stop, nothing)

"""
    logger(e)

The run's logger. Default `_field(e, :logger, JSONLogger())`; also a `Nitro` keyword. `nothing`
turns logging off. Called once per construction, as the keyword's default, so at the start of
construction before `build_data`, and not at all when the `logger` keyword is given. A side effect
in the accessor (opening a file, registering a run with a hosted tracker) therefore happens once per
`Nitro`; a field default instead runs when the experiment is constructed. The default
[`JSONLogger`](@ref) opens its file lazily, after setup has pointed it at the resolved `run_dir`.
"""
function logger end
logger(e) = _field(e, :logger, JSONLogger())

"""
    check_divergence(e) -> Bool

Whether a non-finite training loss stops the run. Default `_field(e, :check_divergence, true)`;
also a `Nitro` keyword. When on, a `NaN` or `Inf` loss raises, naming the step and epoch (see
[`train!`](@ref)). The automatic loop reads each micro-batch's loss back one micro-batch late, so
the error comes after at most one more micro-batch has been dispatched. When off, the loss is
still read back; a non-finite value is dropped from the logged train line like any other, the
epoch's mean train loss in `history` becomes non-finite, and training continues from the diverged
state. Host-side only, so changing it recompiles nothing.
"""
function check_divergence end
check_divergence(e) = _field(e, :check_divergence, true)

"""
    autolog_schedules(e) -> Bool

Whether every varying scalar schedule value is logged on each train line (see
[`log_every_n_steps`](@ref)). Default `_field(e, :autolog_schedules, true)`; also a `Nitro`
keyword. A `Device` field logs as `schedule/<field>` and a rule value as
`schedule/<group>/<field>`, after the per-group scaling (see [Schedules](schedules.md)); constant
schedules are left out. Host-side only: the values are the ones the step already computed.
"""
function autolog_schedules end
autolog_schedules(e) = _field(e, :autolog_schedules, true)

"""
    log_every_n_steps(e) -> Int

Write the train line only on optimizer steps that are a multiple of `n`. Default
`_field(e, :log_every_n_steps, 1)`; also a `Nitro` keyword. The train line carries the loss,
[`train_metrics`](@ref) and the schedule values; thinning it keeps a fast run from flooding its
logger. Only the writing is thinned: validation and the per-epoch lines are unaffected, the loss is
still read back every micro-batch for the divergence check and `history`, and `train_metrics`
still runs every micro-batch.
"""
function log_every_n_steps end
log_every_n_steps(e) = _field(e, :log_every_n_steps, 1)
