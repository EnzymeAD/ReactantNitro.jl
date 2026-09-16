---
name: reactantnitro-optimizer
description: >
  Configure a ReactantNitro optimizer: parameter groups and per-group learning-rate
  ratios, `Decay` with its `:zero` and `:w0` anchors (L2 and L2-SP), per-leaf decay
  exclusion through `no_decay` and `default_no_decay`, global-norm gradient clipping, and
  schedules with their two namespaces. Also covers the binding report, which is how you
  check what your config actually resolved to. Invoke when setting up or debugging an
  optimizer, a learning-rate schedule, weight decay, or a fine-tuning setup that treats a
  backbone differently from a head.
---

# The optimizer, decay, and schedules

Defaults get you a working run: RAdam at `1f-3` over one parameter group, no decay, no clipping, no
schedule. Everything below is opt-in.

## Use the ecosystem's packages, and know which one you cannot swap

**The framework composes with existing Julia packages rather than reimplementing them.** Reach for
the ecosystem first, unless you have a reason not to. But the three you will meet here are not the
same kind of thing, and the difference matters:

| | Package | Can you substitute? |
| --- | --- | --- |
| **rules** | `Optimisers.jl` | **No.** It is a dependency and the framework dispatches on its types |
| **schedules** | `ParameterSchedulers.jl` | **Yes.** No dependency; a schedule is any callable |
| **batching** | `MLUtils.jl` | **Yes.** No dependency; a data source is any iterable (see `reactantnitro-experiment`) |

**Rules are `Optimisers.jl` rules, and that is structural rather than a recommendation.** The
framework builds `Optimisers.Leaf`s, calls `Optimisers.apply!`, and checks membership of an
**allowlist** of rules verified to trace: `Descent`, `Momentum`, `Nesterov`, `Adam`, `AdamW`,
`RAdam`, and the chain and decay types around them. A rule outside it is refused at setup rather
than discovered inside a trace, and the criterion is mechanical: the rule has to trace, be a fixed
point across a step, and hold no host `Number` in its state.

So write `Optimisers.Adam`, not your own. A custom rule is possible, but it is a rule that must
satisfy the same three properties, not a free hand.

**Schedules and batching are the opposite case**: the framework takes no dependency on either, so
whatever you bring works as long as it satisfies a very small contract. The sections below say what
those contracts are.

**If you are porting, port the EFFECTIVE value of each of those, not the ones your model set
explicitly.** Every framework picks its own defaults, and a port that transcribes only what the model
overrode silently adopts this one's for everything else. The clip is the usual casualty: a source
framework that defaults it ON, with a model that never mentions it, produces a port with no clip and
no error and no warning. Read the source framework's defaults once, write down the four values on
this page that it disagrees with, and set them here explicitly even where they look redundant.

## Parameter groups

```julia
param_group(e, keypath) -> Symbol            # default: :default for every leaf
learning_rate(e) -> Real                     # default: 1f-3, the base
learning_rate(e, ::Val{group}) -> Real       # default: learning_rate(e)
```

A group is the unit for a learning rate, a decay coefficient, a decay anchor, and a rule chain.
Classic fine-tuning is two groups:

```julia
ReactantNitro.param_group(::MyExp, ks) = :backbone in ks ? :backbone : :default
ReactantNitro.learning_rate(::MyExp) = 1.0f-3
ReactantNitro.learning_rate(::MyExp, ::Val{:backbone}) = 1.0f-4      # a tenth
```

**Per-group rates are ratios against the base, not absolute values.** The effective rate is
`eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))`, so a schedule on `eta` moves every
group together and the ratio holds at every step. A base of `0` is therefore a setup error rather
than "everything trains at zero": it makes every ratio undefined. To train one group only, zero that
group's own rate or freeze it.

## Decay

```julia
Decay(lambda)                        # anchor :zero, every parameter
Decay(lambda, anchor)                # anchor :zero, :w0, or an array
Decay(lambda, anchor, no_decay_mask)
```

`decay_anchor(e, ::Val{group})` defaults to `:zero`, which is ordinary L2. **`:w0` anchors to the
parameters exactly as `build_model` returned them**, which is L2-SP: the standard regularizer for
fine-tuning a pretrained backbone, pulling toward the pretrained weights rather than toward zero.
The anchor is per group and it is the right granularity, since it is a property of what a group *is*.

Decay is decoupled: it is applied as its own step in the chain, not folded into the gradient.

### Per-leaf exclusion: `no_decay`

```julia
default_no_decay(ks, x) = ndims(x) == 1        # exported so you can extend it
no_decay(e, keypath, param) -> Bool            # defaults to default_no_decay
```

The default is the conventional policy: decay weight matrices, spare biases and norm parameters.

**Extend it, do not replace it, unless you mean to.** This is the single most common way to get a
port's numerics wrong:

```julia
# ADD form: the conventional exclusion, PLUS this model's head weights.
ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || ((:fc in ks) && (:weights in ks))

# REPLACE form: excludes ONLY the head. Every bias and every norm affine is now decayed,
# which is almost certainly not what the framework you are porting from did.
ReactantNitro.no_decay(::MyExp, ks, x) = (:fc in ks) && (:weights in ks)
```

If you are matching another framework's numbers, check whether it ORs its own bias/norm rule with
your exclusion list. Most do, which makes the ADD form the faithful port and the REPLACE form a
silent change to what gets regularized.

Exclusion is **per leaf**, which a group cannot express: biases occur inside every group, and
splitting a group to exclude them would also split the learning-rate ratio.

## Clipping

`gradient_clip_norm(e)` defaults to `0`, meaning off. It is global, it is not per group, it is not
schedulable, and it is a **trace-time host constant**: a disabled clip emits no operations at all
rather than multiplying by infinity every step. The price is that changing it recompiles the
optimizer program, and only the optimizer program, which is the cheap one.

## Schedules

```julia
schedules(e) -> NamedTuple            # also a `Nitro`/`train!` keyword, which replaces wholesale
```

**Every entry is a factory of the horizon**, called once at setup with the total optimizer-step
count, returning a callable of the step:

```julia
ReactantNitro.schedules(e) = (; eta = total -> OneCycle(total, 1.0f-3))
```

A bare `Number` is accepted and means constant. That is what makes constants free: a constant is a
`Number` and a schedule is a callable, so the per-step transfer count equals the number of quantities
actually varying.

### Use the ecosystem's schedules, not your own

**The framework takes no dependency on any schedule library, and that is so you can use whichever
one you like, not so you write your own.** A schedule is any callable of the step, so the framework
never dispatches on one and has no interface to satisfy. Reach for an existing package first.

[ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl) is the obvious one, and it
composes with the factory contract directly:

```julia
using ParameterSchedulers: OneCycle, CosAnneal, Exp

ReactantNitro.schedules(e) = (; eta = total -> OneCycle(total, 1.0f-3))
```

**Its indexing already matches.** Its schedules are 1-indexed and the framework calls a schedule with
the upcoming optimizer step, counting from 1, so `s(t)` lines up with no offset. That is worth
checking for any library you bring: a schedule that is 0-indexed needs `t -> s(t - 1)` and gets the
first step wrong if you forget.

**Where this bites: a library's defaults are its own, not another framework's.** If you are
reproducing a run from a codebase that used PyTorch's `torch.optim.lr_scheduler.OneCycleLR`, the
curve is reproducible **exactly**, but only by mapping every argument rather than accepting defaults:

| PyTorch `OneCycleLR` | `ParameterSchedulers.OneCycle` |
| --- | --- |
| `total_steps` | first positional argument |
| `max_lr` | second positional argument |
| `pct_start` | `percent_start` |
| `div_factor` | `startval = max_lr / div_factor` |
| `final_div_factor` | `endval = max_lr / div_factor / final_div_factor` |

**Verified**, at `max_lr = 1e-3`, 20 steps, `pct_start = 0.3`, `div_factor = 25`,
`final_div_factor = 1e4`: the two curves agree to a maximum absolute difference of `2.2e-19`, which
is floating-point noise, with the same peak step and the same endpoint.

Left at its own defaults it is **not** that curve: the final value differs by four orders of
magnitude, because the two libraries disagree about what the end of the cycle means. That difference
is invisible in a loss curve until you compare against a reference and miss.

**One limitation to know before you commit to the swap: `OneCycle` cannot express a zero-length
warmup.** It asserts `0 < percent_start < 1`, so a schedule that starts at `max_lr` and only anneals,
which is `pct_start = 0` in PyTorch terms, **refuses to construct**. That is at least loud rather than
silent, but it is a surprise if you have already rewritten the hook. `percent_start = 1e-6` is
accepted and is not the same curve. A model that genuinely runs a zero warmup needs an exact port or
a small adapter, and that is the one case where writing the schedule yourself is right.

So: **ecosystem first for new work, and map every argument when reproducing a reference.** Parity is
usually a keyword-mapping problem rather than a reimplementation problem, so reach for the mapping
before the rewrite. Just check first that the library can express the curve you need at all.

### Checking a schedule library's index convention, without fooling yourself

Any library you bring has an index convention, and getting it wrong shifts the whole curve by a step.
**Do not check it at step 1.** Most schedules start on a flat part: a cosine's derivative is zero at
its endpoint, so the first two steps can differ by less than `Float32` eps and round to the *same
number*. Comparing them and seeing equality is exactly how you conclude the convention does not
matter when it does.

Check **mid-curve**, where the slope is steepest. On a real 27,650-step run the offset left almost
every step different and peaked around the half-way point, while step 1 was identical in `Float32`.

One shape difference to expect: some schedulers return several values, a learning rate and a
momentum say. The framework wants one number per key, so take the one you mean:
`total -> (t -> first(sched(t)))`.

**Two namespaces, and a key resolves against exactly one.** `opt` names optimizer rule fields;
`device` names your experiment's `Device` fields.

```julia
schedules(e) = (;
    opt    = (; eta = total -> OneCycle(total, 1.0f-3)),   # the optimizer's learning rate
    device = (; smoothing = total -> (t -> ramp(t, total))),  # e.smoothing, read inside `loss`
)
```

Unqualified keys resolve automatically when unambiguous. A key matching both namespaces is an error
naming both candidates: `lambda` is the obvious name for a decay coefficient *and* for a loss weight,
so collisions are expected rather than exotic, and qualifying is a two-line fix. Renaming your field
is never required. Note the learning-rate key is `eta`, not `lr`.

**A nested `opt` key is a parameter-group path.** `opt = (; backbone = (; eta = ...))` flattens to
the dotted key `opt.backbone.eta`, which binds the `:backbone` group's chain only. The value is the
BASE curve for those groups and the per-group ratio is still applied,
`η_g(t) = opt.backbone.eta(t) · (learning_rate(e, Val(:backbone)) / learning_rate(e))`, exactly
like a bare key; a group with no path key falls back to the bare key, then to its base rate. Paths
are ONE level (`group.field`), because the group table is flat, and qualified-only: an unqualified
dotted key is a resolution error, since the automatic branch cannot know a top-level key is a group
path.

**Manual mode is deliberately asymmetric: its path values are absolute**, because manual mode has no
base rate and no ratios, and its keys bind into the `opt_state` tree rather than into groups. See
`reactantnitro-manual`.

**A scheduled value must return the field's own type.** A schedule written `t -> 0.25 * min(1, t/2000)`
returns `Float64`; the framework coerces it to a `Device{Float32}` field's type rather than making you
remember the `f0`, and asserts afterwards.

**Anything you want to vary inside `loss`, `forward`, or `metrics` must be a `Device` field.** A
`GraphConst` cannot be scheduled: a different value is a different program.

## Check what you got: the binding report

The framework prints a binding report at setup naming every schedule, where it resolved to, and
whether it came from a keyword or an accessor, plus the per-group table with each group's rate, ratio,
anchor, decay, rule, and parameter count. **Read it on the first run of any new config.** It is the
cheapest way to catch a group that came out empty, a ratio you did not intend, or a schedule that
bound to the optimizer when you meant your experiment.
