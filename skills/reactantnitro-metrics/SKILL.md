---
name: reactantnitro-metrics
description: >
  Write metrics for a ReactantNitro experiment: the (sum, count) contract and why the
  framework does not supply the denominator, `count === nothing` for confusion matrices
  and other summed quantities, the fixed key set and why an empty stratum must still
  report, `finalize_metrics` for anything that is not an average of per-batch values,
  logging a validation loss and its components, the one reduction the contract cannot
  express, `metrics` versus `train_metrics`, and metric residency (:host or :device) with
  what changes under each. Invoke when adding or debugging a metric, when porting one from
  another framework, when a metric reads wrong or reads NaN, or when deciding whether a
  metric should run traced.
---

# Metrics

Two hooks, both optional. `metrics` runs on an evaluation split, once per epoch. `train_metrics`
runs on the training batch, once per micro-batch.

```julia
metrics(e, outputs; y) -> NamedTuple of (sum, count) pairs
```

With no `metrics` method at all, the framework reports the validation loss as `val_loss`, so a bare
experiment still validates.

## The `(sum, count)` contract

**Every metric reports its own numerator and denominator.** The framework adds the pairs up over the
split and divides at the end.

```julia
function ReactantNitro.metrics(::MyExp, ŷ; y)
    return (;
        mae      = (sum(abs, ŷ .- y), length(y)),   # divided at the end
        accuracy = (count_correct(ŷ, y), size(y, 2)),
        confusion = (confusion_matrix(ŷ, y), nothing),   # summed, never divided
    )
end
```

**Why not a framework-supplied sample count.** Different metrics have different natural denominators:
per sample, per image, per object, per present class. One framework count would be the wrong
denominator for most of them and silently so. The natural mistake here is returning a bare number
instead of a pair; that raises with the metric named rather than being accumulated as though the
number were a numerator.

**`count === nothing` means accumulate by summation and never divide.** That is what a confusion
matrix wants, and a running total of anything.

**The metric set is fixed by the first batch.** A later batch introducing or dropping a key is an
error, not a ragged average, because with per-metric denominators there is no honest number to report
for something measured on only some batches.

**So emit every key on every batch, with `(0.0, 0)` where the stratum is empty.** The natural way to
write a per-class or per-stratum metric is to skip the empty ones (`count(sel) == 0 && continue`,
`if any(mask)`), which is correct when you own a `Dict` and fatal here. It also survives testing:
synthetic data usually populates every stratum, and the first batch that misses one is real data,
often on the run you were not watching.

**An empty count reaches you as `NaN`**, because the reduction is `sum / count` and nothing guards
`0 / 0`. That NaN then flows to three places that all assume a number: the logger, the phase
monitors, and the checkpoint metric, which the checkpointer compares and which must be finite. So
scrub it in `finalize_metrics`, which is the one place that sees every value:

```julia
ReactantNitro.finalize_metrics(e, acc, split) =
    NamedTuple{keys(acc)}(map(v -> isfinite(v) ? v : 0.0, values(acc)))
```

## `finalize_metrics` for anything that is not an average

F1 is not the mean of per-batch F1s. Accumulate the parts, combine at the end:

```julia
ReactantNitro.metrics(::MyExp, ŷ; y) =
    (; tp = (tp(ŷ, y), nothing), fp = (fp(ŷ, y), nothing), fn = (fn(ŷ, y), nothing))

ReactantNitro.finalize_metrics(e, acc, split) =
    (; f1 = 2acc.tp / (2acc.tp + acc.fp + acc.fn))
```

Default is identity. What `finalize_metrics` returns is what reaches the logger, the phase monitors,
and the checkpoint metric, so the key you select a checkpoint on has to appear here.

### The validation loss's components

`metrics` sees `forward`'s outputs, not the loss, which is the price of the three-function split.
To log `val_loss` and its terms, **call your own `loss` on the host arrays the hook already has**:

```julia
ReactantNitro.metrics(e, out; y) =
    (; val_loss = (loss(e, out; y), 1), acc = (n_correct(out, y), size(y, 2)))
```

That is the same function the optimizer differentiates rather than a second copy of it, so the two
numbers cannot drift, and at once-per-epoch cadence the host arithmetic is noise against the epoch
that just ran. Reach for it whenever you want a breakdown: a loss that returns its parts can hand
every term to a separate key.

### A reduction the contract cannot express

`(sum, count)` covers sums and means. `count === nothing` covers sums. **Neither covers a minimum, a
maximum, or a median over the split**, and there is no third form. Summing per-batch minima and
dividing gives an average minimum, which is strictly higher than the real one and drifts with the
batch count, so it is not the number you asked for.

For those, own the reduction yourself: keep the running value on a `Host` field of the experiment,
update it inside a `:host` hook, and **reset it on the transition INTO `EvalStepping`**, which is the
one moment before a split's first batch. `finalize_metrics` then reads it and adds it to what it
returns. That is the sanctioned exception to the rule below, and the reset point is what makes each
split's extremum its own.

## Residency: `:host` or `:device`

```julia
metrics_residency(e, hook::Symbol) -> :host | :device
```

Defaults: **`:host` for `metrics`, `:device` for `train_metrics`.** The default follows the cadence.
`train_metrics` runs per micro-batch, so host residency there transfers a full output batch every
micro-batch, which for anything image-shaped dwarfs the scalars it produces. `metrics` runs once per
epoch against an epoch that just executed thousands of steps, so the same transfer is noise.

**The flexibility runs toward `:host`, which is why the eval default is the host one.** A traced
metric has to be expressible as a traced program, and evaluation code routinely is not: data-dependent
control flow, a matching or assignment step, connected components, sorting with tie-breaking, or any
call into a library that knows nothing about Reactant. Those are common in validation and rare in a
per-step diagnostic.

**What changes for you is only what the arguments are**: device arrays under `:device`, host arrays
under `:host`. Keyword routing, the `(sum, count)` contract, `count === nothing`, and the guarantee
that a metric never sees a padded row are identical either way.

**"Host" means every argument, not just `outputs`.** Under `:host` the framework converts both the
outputs and the routed batch fields before calling you, and asserts the conversion was total. You do
not need `Array(...)` in your hook. If you see an error naming a device-resident path at this
boundary, that is a framework gap and the message says so; see `reactantnitro-device-boundary`.

**Residency also decides whether editing the hook recompiles.** A `:host` hook is part of no compiled
program, so editing it recompiles nothing. A `:device` hook is in the program and editing it re-pays
the compile. That is a real reason to prefer `:host` while iterating on a metric.

## Two things that will not work

- **Do not keep a mutable accumulator on the experiment for anything the framework can reduce.** The
  pairs are the accumulator for every additive quantity, and the framework owns that reduction; a
  field you mutate per batch instead is not visible to the tracer and will not survive the run the
  way you expect. The exception is the non-additive reduction above, which the contract genuinely
  cannot express: there the `Host` field is the mechanism, and the `EvalStepping` reset is what keeps
  it honest.

  **If you are porting a metric from another framework, this is the item to check first.** A
  framework whose metric is a mutable object with `update!` and `finalize` lets you accumulate
  anything in any shape, so a ported metric usually arrives with both habits baked in: keys emitted
  only when their stratum is non-empty, and reductions the pairs cannot carry.
- **Do not unpack `outputs` again.** If `forward` returns a single array, `metrics` receives that
  array. Writing `first(outputs)` gets you element one of the prediction matrix, and every operation
  after it stays broadcast-legal, so the run reports a metric over one sample without raising.
