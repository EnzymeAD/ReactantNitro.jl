# Optimization: groups, decay, and clipping

Rules are `Optimisers.jl` rules, and the framework owns everything around them: which rules may
be traced, parameter groups, decay, per-leaf decay exclusion, and gradient clipping. The
learning-rate curve is on the [Schedules](schedules.md) page.

## The defaults get you a working run

Declare nothing and setup resolves a complete optimizer: [`optimizer`](@ref) is `RAdam` at `1f-3`
over one parameter group, [`lambda`](@ref) is zero so there is no [`Decay`](@ref) in the chain,
[`gradient_clip_norm`](@ref) is zero so clipping is off, and no schedule is configured. Every
accessor below changes one part of that picture.

Gradient accumulation changes none of it: `accum` folds N micro-batches into one optimizer step,
and the optimizer runs once per accumulated gradient, so every accessor here means the same thing
whether `accum` is 1 or 16. Its only other appearance is in the schedule horizon
([Schedules](schedules.md)).

`RAdam` is the default rather than `Adam` because it rectifies the adaptive variance term over the
first steps instead of leaving you to hand-tune a warmup. `Adam` is one line away at Level 1. The
price is a hard floor of `Optimisers` 0.4.8, the first release whose Reactant extension can trace
`RAdam`.

## Rules are Optimisers.jl rules, and that is structural

The framework builds `Optimisers.Leaf` objects, calls `Optimisers.apply!`, and checks membership of
an allowlist of rules verified to trace: `Descent`, `Momentum`, `Nesterov`, `Adam`, `AdamW`,
`RAdam`, `WeightDecay`, `ClipGrad`, `ClipNorm`, [`Decay`](@ref), and `OptimiserChain`s of them. A
rule outside the allowlist is refused at setup, with an error naming the admitted set, rather than
discovered inside a trace. The criterion is mechanical: the rule has to trace with device-resident
state, be a type fixed point across a step, and hold no host `Number` in its state. Two exclusions
are deliberate rather than untested: `AccumGrad` traces and is numerically wrong under tracing,
and layer-adaptive rules such as LARS and LAMB would compute one trust ratio per
parameter group rather than per layer.

A custom rule is possible if it satisfies the same three properties.

## Three levels of optimizer

The [`optimizer`](@ref) hook has three levels, each a smaller or larger share of the construction:

```julia
# Level 0: declare nothing. The defaults are the whole configuration: RAdam at 1f-3 over one
# parameter group, per-group hyperparameters from `learning_rate` and `lambda`.

# Level 1: return a rule TYPE and the framework constructs it, splatting the resolved
# hyperparameters in by field name. Momentum and Adam are each one line away; RAdam is shown
# because it is the default and the recommended rule.
ReactantNitro.optimizer(::MyExp, ::Val{:backbone}) = Optimisers.RAdam

# Level 2: supply the whole chain. `hp` is this group's already resolved, already
# device-converted hyperparameters.
function ReactantNitro.optimizer(::MyExp, group::Symbol, hp)
    Optimisers.OptimiserChain(
        Optimisers.RAdam(; eta = hp.eta, beta = hp.beta),
        Decay(hp.lambda, hp.anchor, hp.no_decay_mask),
    )
end
```

At Level 1 the framework composes the [`Decay`](@ref) tail itself, so the per-group accessors, the
`w0` anchor, and the no-decay mask all keep working unchanged. **Level 1 rejects any rule declaring
a `lambda` field, which is `AdamW` today**: that rule plus the framework's own decay tail would
decay twice, silently, at whatever the two coefficients sum to.

At Level 2 the framework supplies `hp`, a `NamedTuple` with keys `eta` (this group's effective
learning rate), `lambda` (already multiplied by that `eta`), `anchor` (a flat device array or
`nothing`), `no_decay_mask`, and one key per scheduled rule field, where schedulable means every
field [`nonschedulable`](@ref) does not list. The framework checks that the returned chain is on
the allowlist, that `Decay` is last, and that no two rules in the chain declare the same schedulable
field. It deliberately does not check that the factory read what it was given: a Level 2 factory
that ignores a scheduled key produces a silent no-op, confined to a level you opted into.

There is no path that accepts a fully constructed chain with baked hyperparameter values, because
rules carry tracked device scalars rebuilt every step.

## Parameter groups

[`param_group`](@ref)`(e, keypath)` maps each parameter leaf to the group it belongs to, by its
keypath. The default returns `:default` for every leaf, which is the single-group case and is what
most experiments want. A group is the unit at which optimizer behavior is defined: its learning
rate, its decay coefficient, its decay anchor, and its rule chain.

**Per-group learning rates are ratios against the base, not absolute values.** The effective rate is

```julia
# η_g(t) = eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))
```

so a schedule on `eta` moves every group together and the ratio holds at every step. Per-group
curves are a separate key shape: a path-bound `opt.<group>.eta` binds one group's chain only, with
the same ratio applied ([Schedules](schedules.md)).
`learning_rate(e) == 0` is therefore a setup error rather than "everything trains at zero": it
makes every group's ratio undefined, which is also why the default is `1f-3` rather than `0`. To
train one group only, zero that group's own rate.

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
hundred copies of them, and XLA compiles all of it. Flattening collapses that to `G`, which for most
models is one or two.

**`apply!` is array-level.** `Optimisers.apply!` works on an array, not a tree, so a flat group
buffer hands it exactly what it wants and skips the tree walk. Per-group calls also solve the
scalar-`eta` limitation that a single whole-model buffer would run into, since each group carries
its own hyperparameters.

### What it costs, and what it does not

`flatten` is a **concatenate**, so it is a real copy of the parameter set rather than a fold into a
consumer. It is not a runtime permuted gather: the permutation is one row per leaf, not one entry
per element, so the operand order is known at trace time and grouping by leaf costs nothing beyond
the copy. That is also why it emits no control flow.

`unflatten` is a **reshape over a contiguous slice**, per leaf. Under trace both the slice and the
reshape fold into the consuming operation, so the reconstruction is free.

### The layout is stable, and the checkpoint depends on it

The layout is a stable sort of the tree's traversal-order leaves by group index: `:default` is group
one, the rest follow first-appearance order, and within a group leaves keep traversal order. That
determinism is what lets a checkpoint store the permutation and a resume verify the restored tree
against it, so a model whose structure changed is refused rather than silently reloaded into the
wrong slots. The layout is built unconditionally, including for an evaluation handle, because it is
host-side bookkeeping rather than device memory.

## Decay, and what it decays toward

[`Decay`](@ref) is one rule that differs only in the anchor:

```julia
Decay(lambda)                        # anchor :zero, every parameter
Decay(lambda, anchor)                # anchor :zero, :w0, or an array
Decay(lambda, anchor, no_decay_mask)
```

An anchor of `nothing` is decay toward zero, ordinary weight decay. An anchor of `:w0` is decay
toward the parameters exactly as `build_model` returned them, before any training step or restore,
which the literature calls L2-SP (Li, Grandvalet, and Davoine, ICML 2018, arXiv:1802.01483). An
explicit array decays toward that target, which is the extension point: initialize from A while
anchoring to B.

The anchor is per group, via [`decay_anchor`](@ref)`(e, Val(group))`, defaulting to `:zero`. The
intended fine-tuning setup is `:w0` on the pretrained backbone and `:zero` on the fresh head, and
one `opt.lambda` schedule drives both, because both are the same regularization-strength knob.

Decay is decoupled, AdamW-style: it goes after the base rule in the chain, scaled by the learning
rate, so `lambda` arrives pre-multiplied by the group's effective rate and an LR schedule modulates
regularization strength. `Decay` anywhere but last in a chain is an error checked at setup. The
anchor is an explicit field, never captured at optimizer init, which is what keeps resume correct:
an init-captured anchor would grab the restored weights instead. Anchoring anywhere but `:zero`
puts a per-group `anchor_checksum` in the checkpoint record, and resume verifies it and refuses on
mismatch.

## Per-leaf exclusion: `no_decay`

[`default_no_decay`](@ref)`(ks, x)` excludes every 1-D parameter: in Lux, biases and every
normalization layer's affine scale and shift, which is why one predicate covers all three. It is
exported so that [`no_decay`](@ref) can be extended rather than replaced:

```julia
# ADD form: the conventional exclusion, PLUS this model's head weights.
ReactantNitro.no_decay(::MyExp, ks, x) = default_no_decay(ks, x) || (:fc in ks)

# REPLACE form: excludes ONLY the head. Every bias and every norm affine is now decayed,
# which is almost certainly not what the framework you are porting from did.
ReactantNitro.no_decay(::MyExp, ks, x) = (:fc in ks) && (:weights in ks)
```

The REPLACE form is the single most common way to get a port's numerics wrong: most frameworks OR
their own bias and norm exclusion in, so the ADD form is the faithful port and the REPLACE form a
silent change to what gets regularized.

Exclusion dominates the anchor for free: an excluded leaf gets a mask of 0, and `Decay` computes
`no_decay_mask * lambda * (x - anchor)`, so the term vanishes under either anchor. Exclusion is per
leaf while the coefficient and the anchor are per group, and that split is deliberate: biases and
norm affines occur inside every group, and expressing their exclusion through
[`param_group`](@ref) would force a group split that also silently splits the learning-rate ratio,
which is a different knob. The hook is resolved host-side, once, at setup; it never enters a trace,
and it is not in the compile cache key.

## Gradient clipping: `gradient_clip_norm`

[`gradient_clip_norm`](@ref)`(e)` is the global-norm clip threshold, default `0f0`, meaning off. It
is a global norm over the fully accumulated gradient, applied at the top of the optimizer program,
never a chain member: a chain member would clip per group, which is a different algorithm producing
different updates. It is not per group and it is not schedulable. It is also a
[`Nitro`](@ref) keyword defaulting to this accessor, so `train!(e; gradient_clip_norm = 1f0)`
overrides it for one run:

```julia
# 0 means off, and is the default.
ReactantNitro.gradient_clip_norm(::MyExp) = 1f0
```

The threshold is a trace-time host constant, so a disabled clip emits no ops at all. The price is
that changing it recompiles the optimizer program, and only that one, so a clip sweep re-pays the
cheap compile rather than the expensive one. A device-resident threshold could not keep "off": `0`
would scale the gradient to zero norm, and on an all-zero gradient `0/0` yields `NaN` silently.

## A complete recipe

Putting the per-group pieces together, from the README's MNIST experiment:

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
