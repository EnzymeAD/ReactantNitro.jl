# Optimization: groups, decay, clipping, SAM, and EMA

Rules are `Optimisers.jl` rules, and the framework owns everything around them: which rules may
be traced, parameter groups, decay, per-leaf decay exclusion, gradient clipping, the gradient
rule (SAM and ASAM), and a moving average of the weights (EMA). The learning-rate curve is on the
[Schedules](schedules.md) page.

## The defaults get you a working run

Declare nothing and setup resolves a complete optimizer: [`optimizer`](@ref) is `RAdam` at `1f-3`
over one parameter group, [`lambda`](@ref) is zero so there is no [`Decay`](@ref) in the chain,
[`gradient_clip_norm`](@ref) is zero so clipping is off, and no schedule is configured. Each
accessor below changes one of these.

Gradient accumulation changes none of it: `accum` folds N micro-batches into one optimizer step,
and the optimizer runs once per accumulated gradient, so every accessor here means the same thing
whether `accum` is 1 or 16. The only other place `accum` appears is the schedule horizon
([Schedules](schedules.md)).

`RAdam` is the default rather than `Adam` because it rectifies the adaptive variance term over the
first steps, so you need not tune a warmup by hand. `Adam` is one line away at Level 1. The cost is
a minimum of `Optimisers` 0.4.8, the first release whose Reactant extension can trace `RAdam`.

## Rules are Optimisers.jl rules

The framework builds `Optimisers.Leaf` objects, calls `Optimisers.apply!`, and checks each rule
against an allowlist of rules verified to trace: `Descent`, `Momentum`, `Nesterov`, `Adam`, `AdamW`,
`RAdam`, `WeightDecay`, `ClipGrad`, `ClipNorm`, [`Decay`](@ref), and `OptimiserChain`s of them. A
rule outside the allowlist is refused at setup, with an error naming the admitted set, rather than
failing inside a trace. `ClipNorm` is admitted only with `throw = false`. A rule is admitted once it
traces with device-resident state, keeps its type across a step, and matches the untraced
`Optimisers` result. Two rules are excluded on purpose: `AccumGrad` traces but is numerically wrong
under tracing, and layer-adaptive rules such as LARS and LAMB would compute one trust ratio per
parameter group rather than per layer.

The allowlist has no extension hook: admitting a new rule that meets these criteria is a change to
the package source.

## Three levels of optimizer

The [`optimizer`](@ref) hook has three levels, each handing more of the construction to you:

```julia
# Level 0: declare nothing. RAdam at 1f-3 over one parameter group, per-group hyperparameters
# from `learning_rate` and `lambda`.

# Level 1: return a rule TYPE and the framework constructs it, filling its fields from `hp` by
# name. Shown here per group; `optimizer(::MyExp) = Optimisers.Adam` sets every group.
ReactantNitro.optimizer(::MyExp, ::Val{:backbone}) = Optimisers.RAdam

# Level 2: supply the whole chain. `hp` is this group's resolved, device-converted
# hyperparameters; the method is called at setup and again every optimizer step.
function ReactantNitro.optimizer(::MyExp, group::Symbol, hp)
    Optimisers.OptimiserChain(
        Optimisers.ClipNorm(1.0f0, 2.0f0, false),
        Optimisers.Momentum(hp.eta, get(hp, :rho, 0.9f0)),
    )
end
```

Both levels receive the same `hp`, a `NamedTuple` with keys `eta` (this group's effective learning
rate), `lambda` (already multiplied by that `eta`, or `nothing` when it is zero and not scheduled),
`anchor` (a flat device array or `nothing`), `no_decay_mask`, one key per scheduled rule field,
present from the first call, and `bind` when the schedules have a `bind` table (Level 2 reads it;
[Schedules](schedules.md)). A field is schedulable unless [`nonschedulable`](@ref) lists it.

At Level 1 the framework calls `R()` and replaces each field that `hp` has a key for, so fields
`hp` does not carry keep the rule's constructor defaults. It then appends the
[`Decay`](@ref)`(hp.lambda, hp.anchor, hp.no_decay_mask)` tail when `hp.lambda` is not `nothing`,
so the per-group accessors, the `w0` anchor, and the no-decay mask all work unchanged. **Level 1
rejects any rule declaring a `lambda` field**, which among the admitted rules is `AdamW` and
`WeightDecay`: that rule plus the framework's decay tail would decay twice, at whatever the two
coefficients sum to.

At Level 2 the framework adds nothing to your chain, including no `Decay` tail: to decay, end the
chain with `Decay(hp.lambda, hp.anchor, hp.no_decay_mask)`. It checks that the returned chain is on
the allowlist and that `Decay` is last. Two rules may share a field name: `hp.eta` reaches whichever
rules the factory passes it to, and nothing else.

Because Level 2 places the values itself, setup checks the placement once, on the host: for each
scheduled field it calls the factory with two different values and records which rule fields
change. A scheduled field that changes nothing is a setup error, and the binding report shows where
each one landed. A value computed from `hp`, such as `2 * hp.eta`, is a host number, which would be
baked into the optimizer program at its first value; the framework moves each recorded field to
the device every step, so a computed value follows its schedule too.

No level accepts a fully constructed chain with fixed values, because rules carry device scalars
that are rebuilt every step.

## Parameter groups

[`param_group`](@ref)`(e, keypath)` maps each parameter leaf to the group it belongs to, by its
keypath. The default returns `:default` for every leaf, which is the single-group case and is what
most experiments want. A group is the unit at which optimizer behavior is defined: its learning
rate, its decay coefficient, its decay anchor, and its rule chain.

**Per-group learning rates are ratios against the base, not absolute values.** The effective rate is

```julia
# η_g(t) = eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))
```

where `eta_sched(t)` is the `eta` schedule's value, or `learning_rate(e)` when `eta` is not
scheduled. A schedule on `eta` therefore moves every group together and the ratio holds at every
step; with no schedule, each group trains at its own `learning_rate(e, Val(g))`. A path-bound
`opt.<group>.eta` gives one group its own curve, with the same ratio applied
([Schedules](schedules.md)).

`learning_rate(e) == 0` is a setup error, because it makes every group's ratio undefined; this is
also why the default is `1f-3` rather than `0`. To freeze one group, set that group's own rate to
zero.

The classic fine-tuning setup is two groups, a pretrained backbone at a tenth of the head:

```julia
ReactantNitro.param_group(::MyExp, ks) = :backbone in ks ? :backbone : :default
ReactantNitro.learning_rate(::MyExp) = 1f-3
ReactantNitro.learning_rate(::MyExp, ::Val{:backbone}) = 1f-4   # a tenth of the head
```

## The flat parameter layout

Parameter groups are also the unit the optimizer runs on.

At setup the framework builds a **flat layout**: every parameter array in the tree is assigned to a
group, and the arrays of each group are concatenated into one buffer. The gradient accumulator and
the optimizer state then cross the program boundary as an `NTuple{G}`, one buffer per group, and the
optimizer program applies each rule `G` times rather than once per parameter array.

**Parameters themselves stay a tree.** `ps` is the Lux tree `build_model` returned, everywhere and
at every boundary. Each program flattens on entry and unflattens on exit, inside the trace. That
keeps `Duplicated(ps, dps)` in exactly the form Lux's Reactant extension verifies, and every other
consumer of `ps` already wants a tree.

### Why it matters

**Compile size.** A tree-level `Optimisers.update` emits the rule's operations once per parameter
array. A model with several hundred arrays therefore produces an optimizer program with several
hundred copies of them, and XLA compiles all of it. Flattening reduces that to `G`, which for most
models is one or two.

**`apply!` is array-level.** `Optimisers.apply!` works on an array, not a tree, so a flat group
buffer is what it expects and no tree walk is needed. Because each group is a separate call, each
can carry its own scalar hyperparameters, which one whole-model buffer could not.

### What it costs, and what it does not

`flatten` is a **concatenate**, so it is a real copy of the parameter set. It is not a runtime
gather: the permutation has one row per leaf, not one entry per element, so the operand order is
known at trace time and grouping costs nothing beyond the copy and emits no control flow.

`unflatten` is a **reshape over a contiguous slice**, per leaf. Under trace both the slice and the
reshape fold into the consuming operation, so the reconstruction is free.

### The layout is stable, and the checkpoint depends on it

The layout is a stable sort of the tree's traversal-order leaves by group index: `:default` is group
one, the rest follow first-appearance order, and within a group leaves keep traversal order. Because
the layout is deterministic, a checkpoint stores the permutation and a resume verifies the restored
tree against it, so a model whose structure changed is refused rather than loaded into the wrong
slots. The layout is built for every handle, including an evaluation handle, because it is
host-side bookkeeping and costs no device memory.

## Decay, and what it decays toward

[`Decay`](@ref) is one rule that differs only in the anchor:

```julia
Decay(lambda)                        # toward zero, every parameter
Decay(lambda, anchor)                # anchor: `nothing` (toward zero) or an array
Decay(lambda, anchor, no_decay_mask)
```

An anchor of `nothing` is decay toward zero, ordinary weight decay. An array anchor decays toward
that target. The framework fills the anchor from each group's [`decay_anchor`](@ref)`(e, Val(group))`,
which defaults to `:zero` (anchor `nothing`). `:w0` decays toward the parameters exactly as
`build_model` returned them, before any training step or restore, which the literature calls L2-SP
(Li, Grandvalet, and Davoine, ICML 2018, arXiv:1802.01483). Returning an explicit array, aligned
with that group's flat buffer, lets you initialize from A while anchoring to B.

The usual fine-tuning setup is `:w0` on the pretrained backbone and `:zero` on the fresh head, with
one `opt.lambda` schedule driving both, since both are the same regularization strength.

Decay is decoupled, AdamW-style: it goes after the base rule in the chain, and the coefficient is
scaled by the group's effective learning rate,

```julia
# λ_g(t) = η_g(t) * lambda_sched(t)      # `opt.lambda` scheduled
# λ_g(t) = η_g(t) * lambda(e, Val(g))    # otherwise
```

so an `eta` schedule also modulates regularization strength. `Decay` anywhere but last in a chain is
a setup error. The anchor is an explicit field, never captured at optimizer init, which keeps resume
correct: an anchor captured at init would take the restored weights instead. Anchoring anywhere but
`:zero` puts a per-group `anchor_checksum` in the checkpoint record, and resume verifies it and
refuses on mismatch.

## Per-leaf exclusion: `no_decay`

[`default_no_decay`](@ref)`(ks, x)` excludes every 1-D parameter: in Lux, biases and every
normalization layer's affine scale and shift, which is why one predicate covers all three. It is
exported so that [`no_decay`](@ref) can be extended rather than replaced:

```julia
# ADD form: the conventional exclusion, PLUS this model's head weights.
ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || (:fc in ks)

# REPLACE form: excludes ONLY the head. Every bias and every norm affine is now decayed,
# which is almost certainly not what the framework you are porting from did.
ReactantNitro.no_decay(::MyExp, ks, x) = (:fc in ks) && (:weight in ks)
```

Most frameworks exclude biases and norm affines from decay, so when porting, the ADD form keeps the
original behavior and the REPLACE form changes what gets regularized without any error.

Exclusion overrides the anchor: an excluded leaf gets a mask of 0, and `Decay` computes
`no_decay_mask * lambda * (x - anchor)`, so the term vanishes under either anchor. Exclusion is per
leaf while the coefficient and the anchor are per group, because biases and norm affines occur
inside every group; expressing their exclusion through [`param_group`](@ref) would force a group
split that also splits the learning-rate ratio. The hook runs on the host, once, at setup; it never
enters a trace and is not in the compile cache key.

## Gradient clipping: `gradient_clip_norm`

[`gradient_clip_norm`](@ref)`(e)` is the global-norm clip threshold, default `0f0`, meaning off. The
norm is taken over the fully accumulated gradient of all groups, at the top of the optimizer
program, never as a chain member: a chain member would clip per group, which produces different
updates. It is not per group and not schedulable. It is also a [`Nitro`](@ref) keyword defaulting
to this accessor, so `train!(e; gradient_clip_norm = 1f0)` overrides it for one run:

```julia
# 0 means off, and is the default.
ReactantNitro.gradient_clip_norm(::MyExp) = 1f0
```

The threshold is a trace-time host constant, so a disabled clip emits no ops at all. The cost is
that changing it recompiles the optimizer program, and only that one, so a clip sweep repeats only
the cheap compile. A device-resident threshold could not mean "off" at `0`: it would scale the
gradient to zero norm, and on an all-zero gradient `0/0` yields `NaN`.

## Sharpness-aware minimization: `grad_rule`

[`grad_rule`](@ref)`(e)` decides which gradient each micro-batch adds to the accumulator. The
default, [`SingleGrad`](@ref), is one forward and backward pass. [`SAM`](@ref) takes the gradient
at `ps + rho * g / ‖g‖`, where `g` is the micro-batch's gradient at `ps`:

```julia
ReactantNitro.grad_rule(::MyExp) = SAM(0.05f0)
ReactantNitro.optimizer(::MyExp) = Optimisers.RAdam
# Optional: warm rho up over the first 500 steps.
ReactantNitro.schedules(::MyExp) = (; rho = _ -> t -> 0.05f0 * min(1, t / 500))
```

Like the optimizer, the rule has a second form that places scheduled values itself,
`grad_rule(e, hp)`, where `hp` holds the scheduled rule fields by name and `bind` values as
`hp.bind.<name>` ([Schedules](schedules.md)).

- **Cost.** Two forward and backward passes per micro-batch, plus a small program that builds the
  perturbed parameters. The gradient program is the same compiled program for both passes, so SAM
  adds no compile of it, and the optimizer program does not change.
- **`rho`** is a host number the framework places on the device, so a different value never
  recompiles. A schedule key `rho` binds to it when nothing else is named `rho`; otherwise qualify
  it as `grad = (; rho = ...)` or bind it by hand. A `Device` field passed into the rule is refused.
  Values around 0.01 to 0.1 are typical with Adam-family rules.
- **Masking.** Every leaf is perturbed by default, as in Foret et al. Leaves that
  [`no_perturb`](@ref) excludes are not perturbed and do not count toward `‖g‖`. It is independent
  of [`no_decay`](@ref); to mirror decay exclusion, `no_perturb(e::MyExp, ks, x) = no_decay(e, ks, x)`.
- **Per micro-batch.** `ε` comes from each accumulation micro-batch's own gradient, which is
  m-sharpness with `m` equal to the micro-batch size. A smaller per-device `m` is reserved
  (`SAM(rho; m)`) and refused for now.
- **What is reported.** The loss, the layer state and the `train_metrics` outputs come from the
  unperturbed pass, so curves compare directly with a run without SAM. Pass 2 reuses pass 1's
  input state, so dropout draws the same masks in both. The train line adds `sam/gap`, pass 2's
  loss minus pass 1's.

Manual mode supports `SingleGrad` only; a `train_step` closure can call [`backward`](@ref) at
perturbed parameters itself. A new [`PerturbRule`](@ref) defines [`perturbation`](@ref) and reuses
the same driver path.

### Adaptive SAM: `ASAM`

[`ASAM`](@ref) (Kwon et al. 2021) scales the perturbation by each parameter's own size, so the
sharpness it measures does not change when the weights are rescaled. With `T = |w| + eta`,

```julia
# ε = rho * T² g / ‖T g‖
ReactantNitro.grad_rule(::MyExp) = ASAM(0.5f0)            # eta = 0.01f0 by default
ReactantNitro.grad_rule(::MyExp) = ASAM(1f0; eta = 0.1f0)
```

- **`rho` is relative**, so typical values are about ten times SAM's: around 0.5 to 2 with SGD,
  smaller with Adam-family rules. It schedules and logs exactly as SAM's does.
- **`eta` is not schedulable.** It is structural, and keeping it off the schedule leaves an
  unqualified `eta` key meaning the learning rate.
- **Which leaves scale.** [`adaptive_perturb`](@ref) picks them. The default scales rank-2 and
  higher leaves (weight matrices and kernels) and leaves biases, norm affines and scalars at
  `T = 1`, as SAM treats every leaf. `no_perturb` still applies on top.

```julia
# Scale every leaf, biases included.
ReactantNitro.adaptive_perturb(::MyExp, ks, x) = true
```

Everything else, cost, per-micro-batch `ε`, `sam/gap` and what is reported, is as for SAM.

## Exponential moving average: `ema`

[`ema`](@ref)`(e)` keeps a device copy of the weights averaged after every optimizer step,
`θ_ema ← β θ_ema + (1 - β) θ`. Training never reads it.

```julia
ReactantNitro.ema(::MyExp) = EMA(0.999)
n = Nitro(MyExp(); ema = EMA(0.9999; every = 4))   # or as a keyword
```

- **Warmup.** `warmup = true` (the default) uses `β = min(beta, (1 + t) / (10 + t))` at step `t`,
  so the average is not held at the initial weights for `1 / (1 - beta)` steps. The train line logs
  the `β` applied as `ema/beta`.
- **`every = k`** updates on every `k`-th optimizer step only.
- **What reads it.** Under `eval_weights = :ema`, the default with an EMA, validation, checkpoint
  selection, `restore_best`, [`predict`](@ref) and export use the average. `eval_weights = :raw`
  evaluates the raw weights instead. Training and [`parameters`](@ref) always use the raw weights.
- **Cost.** One copy of the parameters on the device and a small elementwise program per update.
  It works in the automatic loop and in manual mode.

Checkpoints keep the average in the record's `ema` field, and a resume continues it. `load_ema =
true` loads a source's average as the weights of a new handle; see [Checkpoints](checkpoints.md).

## A complete recipe

The per-group pieces together, for an MNIST MLP:

```julia
# Two groups: `layer_1` is the encoder, everything else is the head. Per-group rates are
# RATIOS against the base, so the encoder stays a tenth of the head for the whole curve.
ReactantNitro.param_group(::MnistMLP, ks) = ks[1] === :layer_1 ? :encoder : :default
ReactantNitro.learning_rate(::MnistMLP)   = 3f-4
ReactantNitro.learning_rate(::MnistMLP, ::Val{:encoder}) = 3f-5
ReactantNitro.lambda(::MnistMLP, ::Val{:encoder})        = 1f-4   # decoupled decay, toward zero
```

The schedule that turns the `3f-4` base into a curve is on the [Schedules](schedules.md) page.
Every name on this page is documented on the [API](api.md) page.
