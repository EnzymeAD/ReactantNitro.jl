# Schedules: what varies with the step

A schedule varies one quantity per optimizer step, the learning rate, a decay coefficient, a
`Device` field read inside a traced hook, and everything else stays fixed at setup. This page covers
the contract, the two namespaces a key can live in, how schedules interact with `Device` values, the
binding report, and the ecosystem packages worth reaching for.

## The contract

[`schedules`](@ref)`(e)` returns a NamedTuple; the default is `(;)`, meaning everything constant.
Every entry is a **factory of the horizon**: the framework calls `f(total)` once, at setup, with the
total number of optimizer steps, then calls what it returned once per optimizer step. A bare
`Number` normalizes to `_ -> (_ -> value)`, which makes constants setup-fixed by construction: a
constant is a `Number` and a schedule is a callable, so the per-step transfer count equals the
number of quantities actually varying.

```julia
schedules(e::MyExp) = (;
    eta        = total -> OneCycle(total, 1f-3),               # factory of the horizon
    aux_weight = _ -> (t -> max(0f0, 1f0 - t / 5000)),         # ignores the horizon
    epsilon    = 1f-8,                                         # a bare Number is a constant;
                                                               # not every field is schedulable, see below
)
```

`eta` is a rule field and `aux_weight` is a `Device` field on `MyExp`; the next sections say how
each key resolves. `OneCycle(total, 1f-3)` is the schedule that runs once per step, `aux_weight`'s
factory ignores the horizon, and `epsilon` is a constant for the whole run.

## Step and horizon

`step` is the optimizer step, not the micro-batch. With [`accum`](@ref) = `N` they differ by a
factor of `N`, and confusing them shifts the whole curve by `N`. The horizon is
`total = max_epochs * div(steps_per_epoch, accum)`, an exact division, never `fld` or `cld`, because
the remainder is a setup error: the training loader must drop its partial final
batch, and its batch count must divide by `accum`.

Schedules are host-side, so ordinary Julia control flow is fine inside them: an `if`, a `for`, a
call into any library, none of it has to trace. A schedule need not be a pure function of the step,
but one closing over training state does not resume exactly. The framework calls the
schedule with the upcoming optimizer step, counting from 1.

## The two namespaces

A schedule key is either a [`Device`](@ref) field name or a rule field name, resolved against the
union of the two. The two route to different places: an `opt` key is applied to the
optimizer, a `device` key is written into the experiment and reaches traced code as `e.field`; there
is no third destination, which is what makes the resolution total.

`device` and `opt` are reserved at the top level and name the two namespaces; every other top-level
key is unqualified and must resolve to exactly one of them. A key matching neither is an error
naming the fix. A key matching both is an ambiguity error naming both candidates and showing
the qualified form for your own key; renaming the field is never required. Collisions are
expected rather than exotic: `lambda` is the obvious name for a decay coefficient and for a loss
weight, and `eta`, `rho`, and `epsilon` are all plausible model hyperparameters.

```julia
schedules = (; eta = total -> OneCycle(total, 1f-3),
               device = (; lambda = _ -> t -> 0.5f0),    # e.lambda
               opt     = (; lambda = _ -> t -> 1f-4))     # the decay coefficient
```

Rule field names are the rule's **actual** field names, so the learning-rate key is `eta`, not `lr`.

Scheduling [`lambda`](@ref) carries one extra rule: `opt.lambda` together with per-group `lambda`
accessors is a setup error naming both, because the schedule supplies the decay base for every
group. A path-bound `opt.<group>.lambda` applies the same rule to the group it names: it conflicts
with that group's accessor and leaves every other group's alone. `eta` is deliberately not
symmetric: per-group [`learning_rate`](@ref) accessors define
ratios that compose with an `eta` schedule, which is the [Optimization](optimization.md) page's
subject.

## Per-group curves: path-bound `opt` keys

A bare `opt` key binds **every** parameter group's chain. A **path-bound** key names a group and
binds that group's chain only, so per-group learning-rate curves work: the backbone anneals while
the head warms up.

```julia
schedules(e::MyExp) = (;
    opt = (;
        eta      = total -> t -> 1.0f-3 * (1 - t / total),   # bare: every group's curve
        backbone = (; eta = total -> t -> 3.0f-4 * ...),     # path: the :backbone group only
        default  = (; eta = total -> t -> 1.0f-3 * ...),     # path: the :default group only
    ))
```

The structure IS the binding, exactly as in manual mode; the difference is the value semantics. A
path value is the **base curve for those groups, and the per-group ratio is still applied**:

```julia
# η_g(t) = opt.backbone.eta(t) * (learning_rate(e, Val(:backbone)) / learning_rate(e))
```

so `opt.backbone.eta = 1f-2` against a backbone ratio of `0.1` trains the backbone at an effective
`1f-3`. A group with no path key falls back to the bare `opt.eta` key, then to its base rate. Bare
keys are unchanged: `opt.eta` remains the all-groups shorthand.

Path keys are **qualified only**: they live under `opt`, and an unqualified dotted key is a
resolution error, because the automatic loop cannot know a top-level key is a group path. Paths are
**one level** (`group.field`), since parameter groups are flat. A path segment that is not a
parameter group, and a field that is not schedulable in that group's chain, are setup errors naming
the offender.

**Manual mode is deliberately asymmetric.** Manual-mode path values are absolute, because manual
mode has no base rate and no ratios; the automatic loop's path values are ratio-scaled against the
`learning_rate` accessors. Same syntax, each mode's native value semantics
([Manual training](manual.md)).

## How schedules interact with `Device` values

Anything you want to vary inside `loss`, `forward`, or `metrics` must be a `Device` field. A
scheduled `device` key is written into the experiment on each step, and traced code reads `e.field`.
A [`GraphConst`](@ref) cannot be scheduled: a different value is a different program, because the
value bakes into the compiled graph and enters the compile cache key. A `Host` field
is in neither namespace, so a schedule key naming one is a resolution error naming the fix.

A scheduled value must return the field's own type. The framework coerces before converting, so a
schedule written `t -> 0.25 * min(1, t / 2000)`, which returns `Float64`, is coerced to a
`Device{Float32}` field's type rather than making you remember the `f0`, and it asserts afterwards.
The assertion is the backstop for a silent failure: an uncoerced `Float64` would move
`typeof(compile_view(e))`, which is `args[1]` at every trace site and therefore part of the compile
key, so every optimizer step would recompile the gradient program instead of reusing it.

Constant and scheduled are the same device slot. A bare `Number` normalizes to a constant
factory, so either way the value lands in the same slot, and switching between a constant and a
schedule costs no recompile: the schedule writes the scalar the constant would have held.

The fields a rule may **not** schedule are declared by [`nonschedulable`](@ref)`(::Type{R})`, which
defaults to `()`:

```julia
nonschedulable(::Type{<:Optimisers.Adam})     = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.RAdam})    = (:beta, :epsilon)
nonschedulable(::Type{<:Optimisers.AdamW})    = (:beta, :epsilon, :couple)
nonschedulable(::Type{<:Optimisers.ClipNorm}) = (:p, :throw)
nonschedulable(::Type{<:Decay})               = (:anchor, :no_decay_mask)
```

A field on that list is not promoted to device residency, and one declaration drives both, and it
must: a field that cannot be promoted cannot be scheduled, and a field that is scheduled must be
promoted; declaring them separately would let them drift. The general rule is that any
parameter-sized rule field is non-schedulable, since scheduling one would push a parameter-sized
buffer to device every optimizer step; [`Decay`](@ref)'s `anchor` and `no_decay_mask`
are the shipped case. The scalar exclusions have their own reasons: `beta` is a `Tuple` and a
scalar schedule cannot produce one, `couple` is a `Bool`, `p` is structural and promoting it
breaks the clip norm's dispatch, and `epsilon` is a numerical floor rather than a knob.

Declare one method per concrete rule type, never a `Union`: a `Union` method is shadowed by any more
specific one, silently and with no ambiguity warning, so a scalar schedule could get splatted into a
`Tuple` field.

## The binding report

The binding report names every schedule, where it resolved to, and whether it came from a keyword
or an accessor, plus a row per parameter group carrying that group's rate, ratio, anchor, decay,
rule, and parameter count.

**It is part of `show(nitro)` rather than a display of its own.** Its sections are appended to the
handle's, so printing a handle answers both what the run currently holds and where each configured
value came from, in one table. Setup prints nothing itself: the REPL already displays the handle
the constructor returns, and a constructor that also printed it would show you the same table
twice. From a script, ask for it with `display(nitro)`.

The plain text is handed to [`log_other!`](@ref) whether or not anything displayed it, so the
run's record carries it and the terminal is never the only copy. [`binding_report`](@ref) returns
that same text.

Read it on the first run of any new config; it is the cheapest way to catch a group that came out
empty, a ratio you did not intend, or a schedule that bound to the optimizer when you meant your
experiment. It is a diagnostic, not a check: it never fails, and it computes nothing the run
does not already compute.

The schedule belongs to the experiment by default, and the `schedules` keyword on [`Nitro`](@ref)
and [`train!`](@ref) is the per-run override. It replaces wholesale; it does not merge, because a
silently retained schedule from the accessor is much harder to explain than an explicitly dropped
one. To merge, say so:

```julia
train!(e; schedules = merge(schedules(e), (; eta = total -> OneCycle(total, 3f-4))))
```

## Use the ecosystem

[ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl) is a documented
recommendation with no dependency, not even a weakdep, because the framework never dispatches on a
schedule: a schedule is any callable of the step, so there is no interface to satisfy and nothing to
import from us.

Its schedules are 1-indexed and align with the framework's step counting: the framework calls a
schedule with the upcoming optimizer step, counting from 1, so `s(t)` lines up with no offset. That
is worth checking for any library you bring; a schedule that is 0-indexed needs `t -> s(t - 1)` and
gets the first step wrong if you forget.

Do not check a library's index convention at step 1. Most schedules start on a flat part: a cosine's
derivative is zero at its endpoint, so the first two steps can differ by less than `Float32` eps and
round to the same number, which is exactly how you conclude the convention does not matter when it
does. Check mid-curve, where the slope is steepest.

If you are reproducing a run from a codebase that used PyTorch's
`torch.optim.lr_scheduler.OneCycleLR`, the curve is reproducible exactly, but only by mapping every
argument rather than accepting defaults:

| PyTorch `OneCycleLR` | `ParameterSchedulers.OneCycle` |
| --- | --- |
| `total_steps` | first positional argument |
| `max_lr` | second positional argument |
| `pct_start` | `percent_start` |
| `div_factor` | `startval = max_lr / div_factor` |
| `final_div_factor` | `endval = max_lr / div_factor / final_div_factor` |

Verified at `max_lr = 1e-3`, 20 steps, `pct_start = 0.3`, `div_factor = 25`,
`final_div_factor = 1e4`: the two curves agree to a maximum absolute difference of `2.2e-19`,
floating-point noise, with the same peak step and the same endpoint. Left at its own defaults it is
not that curve: the final value differs by four orders of magnitude, because the two libraries
disagree about what the end of the cycle means.

One limitation to know before you commit to the swap: `OneCycle` cannot express a zero-length
warmup. It asserts `0 < percent_start < 1`, so a schedule that starts at `max_lr` and only anneals,
which is `pct_start = 0` in PyTorch terms, refuses to construct. That is loud rather than silent,
but it is a surprise if you have already rewritten the hook; `percent_start = 1e-6` is accepted and
is not the same curve.

Every name on this page is documented on the [API](api.md) page. For the field markers behind
`device` schedules, see [Experiments](experiments.md); for `eta`, parameter groups, and decay, see
[Optimization](optimization.md).
