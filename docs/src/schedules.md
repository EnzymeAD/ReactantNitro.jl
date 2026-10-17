# Schedules: what varies with the step

A schedule varies one quantity per optimizer step: the learning rate, a decay coefficient, another
rule field, or a `Device` field read inside a traced hook. Everything else is fixed at setup.

## The contract

[`schedules`](@ref)`(e)` returns a NamedTuple; the default is `(;)`, meaning everything constant.
Every entry is a **factory of the horizon**: the framework calls `f(total)` once, at setup, with the
total number of optimizer steps, then calls what it returned once per optimizer step. A bare
`Number` normalizes to `_ -> (_ -> value)`, a constant.

```julia
schedules(e::MyExp) = (;
    eta        = total -> OneCycle(total, 1f-3),               # factory of the horizon
    aux_weight = _ -> (t -> max(0f0, 1f0 - t / 5000)),         # ignores the horizon
    lambda     = 1f-4,                                         # a bare Number is a constant
)
```

`eta` and `lambda` are rule fields and `aux_weight` is a `Device` field on `MyExp`; the next
sections say how each key resolves. `OneCycle(total, 1f-3)` is the schedule called once per step,
`aux_weight`'s factory ignores the horizon, and `lambda` is constant for the whole run.

## Step and horizon

`step` is the optimizer step, not the micro-batch. With [`accum`](@ref) = `N` they differ by a
factor of `N`. The horizon is `total = max_epochs * div(length(train), accum)`, where
`length(train)` is the number of training batches per epoch. The division is exact because a
remainder is a setup error: the training loader must drop its partial final batch, and its batch
count must divide by `accum`.

Schedules run on the host, so ordinary Julia control flow is fine inside them: an `if`, a `for`, a
call into any library. A schedule need not be a pure function of the step, but one that closes over
training state does not resume exactly.

## The two namespaces

A schedule key is either a [`Device`](@ref) field name or a rule field name, resolved against the
union of the two. An `opt` key is applied to the optimizer; a `device` key is written into the
experiment and reaches traced code as `e.field`.

`device` and `opt` are reserved at the top level and name the two namespaces; every other top-level
key is unqualified and must resolve to exactly one of them. A key matching neither is an error
naming the fix. A key matching both is an ambiguity error that shows the qualified form, so you
never have to rename the field. Collisions are common: `lambda` is a natural name for a decay
coefficient and for a loss weight, and `eta`, `rho`, and `epsilon` are plausible model
hyperparameters.

```julia
schedules = (; eta = total -> OneCycle(total, 1f-3),
               device = (; lambda = _ -> t -> 0.5f0),    # e.lambda
               opt     = (; lambda = _ -> t -> 1f-4))     # the decay coefficient
```

Rule field names are the rule's actual field names, so the learning-rate key is `eta`, not `lr`.

Scheduling [`lambda`](@ref) has one extra rule: `opt.lambda` together with per-group `lambda`
accessors is a setup error naming both, because the schedule supplies the decay coefficient for
every group. A path-bound `opt.<group>.lambda` (next section) conflicts only with that group's
accessor. A scheduled `lambda` keeps the [`Decay`](@ref) tail in the chain even while its value is
zero, so a schedule that starts or ends at zero does not change the chain mid-run. `eta` has no
such conflict: per-group [`learning_rate`](@ref) accessors are ratios that multiply an `eta`
schedule ([Optimization](optimization.md)).

## Per-group curves: path-bound `opt` keys

A bare `opt` key applies to every parameter group. A **path-bound** key, `opt.<group>.<field>`,
applies to the named group only, so groups can follow different curves.

```julia
schedules(e::MyExp) = (;
    opt = (;
        eta      = total -> OneCycle(total, 1f-3),               # bare: groups without a path key
        backbone = (; eta = total -> t -> 1f-3 * (1 - t / total)),  # path: the :backbone group
    ))
```

For each group and field, the value comes from the first of these that exists:

1. the path key for that group, `opt.<group>.<field>`;
2. the bare key, `opt.<field>` or an unqualified `<field>` that resolves to the optimizer;
3. the group's accessor: [`learning_rate`](@ref)`(e, Val(g))` for `eta`, [`lambda`](@ref) for
   `lambda`.

So above, `:backbone` decays linearly and every other group follows `OneCycle`. Any other field
with no key keeps the rule's own value.

A path value goes through the same per-group scaling as a bare one:

```julia
# η_g(t) = value(t) * (learning_rate(e, Val(g)) / learning_rate(e))   # eta
# λ_g(t) = η_g(t) * value(t)                                          # lambda
# any other field: value(t), as given
```

so `opt.backbone.eta = 1f-2` against a backbone ratio of `0.1` trains the backbone at `1f-3`. To
give a group an absolute curve, leave its `learning_rate` accessor at the default ratio of 1.

Path keys are **qualified only**: they live under `opt`, and an unqualified dotted key is a
resolution error. Paths are one level (`group.field`), since parameter groups are flat. A path
segment that is not a parameter group, and a field that is not schedulable in that group's chain,
are setup errors naming the offender.

Manual mode uses the same syntax with different values: there are no groups or ratios, a path is a
path into `opt_state`, and every value is absolute ([Manual training](manual.md)).

## How schedules interact with `Device` values

Anything you want to vary inside `loss`, `forward`, or `metrics` must be a `Device` field. A
scheduled `device` key is written into the experiment on each step, and traced code reads `e.field`.
A [`GraphConst`](@ref) cannot be scheduled: its value is baked into the compiled graph and is part
of the compile cache key, so a different value is a different program. A `Host` field is in neither
namespace, so a schedule key naming one is a resolution error naming the fix.

A scheduled value is converted to the field's own type, so a schedule written
`t -> 0.25 * min(1, t / 2000)`, which returns `Float64`, works for a `Device{Float32}` field without
the `f0`. After the conversion the framework asserts that the experiment's type did not change; a
change would alter the compile key and recompile the gradient program on every step.

A constant and a schedule use the same device slot. A bare `Number` normalizes to a constant
factory, so switching a field between a constant and a schedule costs no recompile.

The fields a rule may **not** schedule are declared by [`nonschedulable`](@ref)`(::Type{R})`, which
defaults to `()`:

```julia
nonschedulable(::Type{<:Optimisers.Adam})     = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.RAdam})    = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.AdamW})    = (:beta, :epsilon, :couple)
nonschedulable(::Type{<:Optimisers.ClipNorm}) = (:p, :throw)
nonschedulable(::Type{<:Decay})               = (:anchor, :no_decay_mask)
```

The same declaration also keeps those fields off the device, so what is schedulable and what is
device-resident cannot drift apart. Any parameter-sized rule field is non-schedulable, since
scheduling it would upload a parameter-sized buffer every step; [`Decay`](@ref)'s `anchor` and
`no_decay_mask` are the shipped case. The scalar exclusions have their own reasons: `beta` is a
`Tuple`, which a scalar schedule cannot produce; `couple` is a `Bool`; `p` is structural, and
promoting it breaks the clip norm's dispatch; `epsilon` is a numerical floor rather than a tuning
knob.

Declare one method per concrete rule type, never a `Union`: a more specific method shadows a
`Union` method with no ambiguity warning, so a scalar schedule could be written into a `Tuple`
field.

## The binding report

The binding report lists every schedule key, what it resolved to, and whether it came from a
keyword or the accessor, plus one row per parameter group with that group's rate, ratio, anchor,
decay, rule, and parameter count. At Level 2 it also shows which rule field each scheduled value
landed in.

The report is part of `show(nitro)`: printing a handle shows what the run holds and where each
configured value came from, in one table. Setup prints nothing itself, since the REPL already
displays the handle the constructor returns. From a script, call `display(nitro)`. The same text is
passed to [`log_other!`](@ref), so the run's record carries it.

Read it on the first run of any new config: it shows a group that came out empty, a ratio you did
not intend, or a schedule that bound to the optimizer when you meant your experiment. It reports
and never fails.

The schedule belongs to the experiment by default, and the `schedules` keyword on [`Nitro`](@ref)
and [`train!`](@ref) overrides it for one run. The keyword replaces the accessor's table; it does
not merge with it. To merge, say so:

```julia
train!(e; schedules = merge(schedules(e), (; eta = total -> OneCycle(total, 3f-4))))
```

## Logging

The framework logs every varying scalar schedule on the train line, which is written every
[`log_every_n_steps`](@ref) optimizer steps (default 1). Every key starts with `schedule/`, so none
can collide with a [`train_metrics`](@ref) name.

- A `Device` field logs as `schedule/<field>`.
- A rule value logs as `schedule/<group>/<field>`, for each group whose chain uses it, at the value
  applied: `eta` and `lambda` after the per-group scaling above, so `schedule/backbone/eta` is the
  rate the backbone trained at.
- A group's `lambda` is also logged when only its `eta` varies, if the group decays, since its
  decay is `η_g(t)` times the coefficient.
- Manual mode logs each `opt` key's value as given, with the path joined by `/`
  (`schedule/gen/eta`).

Constants, non-finite values, and values that are not `Real` (an array, a device scalar) are left
out; the binding report records the constants. To turn logging off, pass
`Nitro(e; autolog_schedules = false)` or define [`autolog_schedules`](@ref)`(e) = false`.

## Use the ecosystem

[ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl) is the recommendation.
ReactantNitro does not depend on it, not even as a weakdep: a schedule is any callable of the step,
so there is no interface to satisfy.

The framework calls a schedule with the upcoming optimizer step, counting from 1. ParameterSchedulers
schedules are 1-indexed, so `s(t)` lines up with no offset. Check this for any library you bring: a
0-indexed schedule needs `t -> s(t - 1)`.

Do not check a library's index convention at step 1. Most schedules start on a flat part, so the
first two steps can differ by less than `Float32` eps and round to the same number. Check mid-curve,
where the slope is steepest.

PyTorch's `torch.optim.lr_scheduler.OneCycleLR` curve can be reproduced exactly, but only by
mapping every argument rather than accepting defaults:

| PyTorch `OneCycleLR` | `ParameterSchedulers.OneCycle` |
| --- | --- |
| `total_steps` | first positional argument |
| `max_lr` | second positional argument |
| `pct_start` | `percent_start` |
| `div_factor` | `startval = max_lr / div_factor` |
| `final_div_factor` | `endval = max_lr / div_factor / final_div_factor` |

At `max_lr = 1e-3`, 20 steps, `pct_start = 0.3`, `div_factor = 25`, `final_div_factor = 1e4`, the
two curves agree to a maximum absolute difference of `2.2e-19`, with the same peak step and the
same endpoint. With `OneCycle`'s own defaults the final value differs by four orders of magnitude,
because the two libraries define the end of the cycle differently.

`OneCycle` cannot express a zero-length warmup. It asserts `0 < percent_start < 1`, so a schedule
that starts at `max_lr` and only anneals (`pct_start = 0` in PyTorch) fails to construct.
`percent_start = 1e-6` is accepted but is not the same curve.

Every name on this page is documented on the [API](api.md) page. For the field markers behind
`device` schedules, see [Experiments](experiments.md); for `eta`, parameter groups, and decay, see
[Optimization](optimization.md).
