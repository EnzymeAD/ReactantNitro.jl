---
name: reactantnitro-metrics
description: >
  Write metrics for a ReactantNitro experiment: the `value => mode` contract (a count, :sum,
  :max, :min, :concat) and why the framework does not supply the denominator, groups that
  share or mix modes, the fixed key set and why an empty stratum must still report,
  `finalize_metrics` for anything that is not a mean of per-batch values, `:concat` for
  rank-based metrics such as AUROC, logging a validation loss and its components, `metrics`
  versus `train_metrics`, and metric residency (:host or :device) with what changes under
  each. Invoke when adding or debugging a metric, when porting one from another framework,
  when a metric reads wrong or reads NaN, or when deciding whether a metric should run traced.
---

# Metrics

Two hooks, both optional. `metrics` runs on an evaluation split, once per eval batch, each epoch.
`train_metrics` runs on the training batch, once per micro-batch, and returns plain scalars.

```julia
metrics(e, outputs; y) -> NamedTuple of `value => mode`
```

With no `metrics` method at all, the framework reports the validation loss as `val_loss`, so a bare
experiment still validates. The full reference is the Metrics page of the docs (`docs/src/metrics.md`).

## The `value => mode` contract

**The mode says how the framework combines a value across the batches of a split.**

| Mode | Across batches | `finalize_metrics` receives |
| --- | --- | --- |
| a count | values summed, counts summed | `value / count` |
| `:sum` | summed | the total |
| `:max`, `:min` | elementwise max or min | the extreme |
| `:concat` | joined along the last axis | an `(…, N)` array, one entry per sample |

```julia
function ReactantNitro.metrics(::MyExp, ŷ; y)
    err = abs.(ŷ .- y)
    return (;
        mae       = sum(err) => length(err),            # divided at the end
        accuracy  = count_correct(ŷ, y) => size(y, 2),
        worst     = maximum(err) => :max,               # the batch's own extreme
        confusion = confusion_matrix(ŷ, y) => :sum,     # summed, never divided
    )
end
```

**Why not a framework-supplied sample count.** Different metrics have different natural
denominators: per sample, per image, per object, per present class. One framework count would be
the wrong denominator for most of them, silently. A bare number with no mode raises with the metric
named rather than being accumulated as though it were a numerator.

**`=>` binds tighter than a ternary.** `c ? a : b => :sum` gives `a` no mode; write
`(c ? a : b) => :sum`. A count is any real number, so a weighted mean is `sum(w .* err) => sum(w)`.

**The older spellings still work** at the top of a key: `(value, count)` for `value => count` and
`nothing` for `:sum`. Write new metrics with `=>`.

## Groups

A key can hold a `NamedTuple` or `Tuple` of values. A mode on the group covers every value in it; a
mode on one value covers that value, so a group can mix modes:

```julia
ranking = (; score = p, label = y) => :concat
ranking = (; score = p => :concat, label = y => :concat, n_pos = count(y) => :sum)
```

Every value needs exactly one mode, from itself or a group above it. `finalize_metrics` receives the
same shape with the modes removed.

## The fixed key set, and empty strata

**The metric set and every value's mode are fixed by the first batch.** A later batch that adds or
drops a key, or changes a mode, is an error, because a value measured on only some batches has no
honest denominator.

**So emit every key on every batch, with `0.0 => 0` where the stratum is empty.** The natural way to
write a per-class metric is to skip the empty ones (`count(sel) == 0 && continue`), which is correct
when you own a `Dict` and fatal here. It also survives testing: synthetic data usually populates
every stratum, and the first batch that misses one is real data.

**An empty count reaches you as `NaN`**, because nothing guards `0 / 0`. That flows to the logger,
the phase monitors, and the checkpoint metric, which must be finite. Scrub it in `finalize_metrics`:

```julia
ReactantNitro.finalize_metrics(e, acc, split) =
    map(v -> v isa Real && !isfinite(v) ? 0.0 : v, acc)
```

## `finalize_metrics` for anything that is not a mean

F1 is not the mean of per-batch F1s. Accumulate the parts with `:sum` and combine at the end:

```julia
ReactantNitro.metrics(::MyExp, ŷ; y) = (; f1 = (; tp = tp(ŷ, y), fp = fp(ŷ, y), fn = fn(ŷ, y)) => :sum)

ReactantNitro.finalize_metrics(e, acc, split) =
    (; f1 = 2acc.f1.tp / (2acc.f1.tp + acc.f1.fp + acc.f1.fn))
```

The default is identity. What `finalize_metrics` returns is what reaches the run history, the
logger, the phase monitors, and the checkpoint metric, so the key you select a checkpoint on has to
appear here.

### Which mode a metric needs

A function of sums over samples is exact with counts and `:sum`: MSE, MAE, R², Pearson correlation,
precision, recall, F1, global Dice and IoU, fixed-bin calibration error. A split-wide extreme is
`:max` or `:min`. A metric that depends on ranks or order needs `:concat`, or a binned `:sum`
histogram when the split is large: AUROC, average precision, Spearman correlation, medians,
quantiles.

### `:concat` rules

- **Batch-last.** Each value's last axis must equal the batch's real sample count. Scalars and
  batch-first values are refused; move the sample axis last with `permutedims`.
- **Aligned, not ordered.** Entry `i` of every `:concat` value is the same sample, but dataset order
  is not promised. Concatenate an `id` field and sort by it if you need order.
- **Reduce it in `finalize_metrics`.** A `:concat` array returned unreduced is refused, because it
  would be stored in the run history, logged, and written into every checkpoint. Per-sample outputs
  for analysis come from `predict`.
- **Memory.** `:concat` holds the split's per-sample values in host memory for one evaluation pass.
  The other modes hold one value per key.

### The validation loss's components

`metrics` sees `forward`'s outputs, not the loss. To log `val_loss` and its terms, **call your own
`loss` on the arrays the hook already has**:

```julia
ReactantNitro.metrics(e, out; y) =
    (; val_loss = loss(e, out; y) => 1, acc = n_correct(out, y) => size(y, 2))
```

It is the same function the optimizer differentiates, so the two numbers cannot drift. A loss that
returns its parts can hand every term to a separate key.

## Residency: `:host` or `:device`

```julia
metrics_residency(e, hook::Symbol) -> :host | :device
```

Defaults: **`:host` for `metrics`, `:device` for `train_metrics`.** The default follows the cadence.
`train_metrics` runs per micro-batch, so host residency there transfers a full output batch every
micro-batch. `metrics` runs once per eval batch per epoch, after an epoch of thousands of steps, so
the same transfer is noise.

**The flexibility runs toward `:host`.** A traced metric has to be expressible as a traced program,
and evaluation code often is not: data-dependent control flow, matching, connected components,
sorting with tie-breaking, any library that knows nothing about Reactant.

**What changes for you is only what the arguments are**: device arrays under `:device`, host arrays
under `:host`. Keyword routing, the modes, and the guarantee that a metric never sees a padded row
are identical either way.

**"Host" means every argument, not just `outputs`.** Under `:host` the framework converts the outputs
and the routed batch fields before calling you, and asserts the conversion was total. You do not
need `Array(...)` in your hook. An error naming a device-resident path at this boundary is a
framework gap; see `reactantnitro-device-boundary`.

**Residency also decides whether editing the hook recompiles.** A `:host` hook is part of no
compiled program; a `:device` hook is, and editing it re-pays the compile. Prefer `:host` while
iterating on a metric.

## Two things that will not work

- **Do not keep a mutable accumulator on the experiment.** The modes are the accumulator, and the
  framework owns the reduction; a field you mutate per batch is invisible to the tracer and does not
  survive the run the way you expect.

  **If you are porting a metric from another framework, check this first.** A metric object with
  `update!` and `finalize` lets you accumulate anything, so a ported metric usually arrives with
  keys emitted only when their stratum is non-empty, and with its own running state. Map the state
  to modes: running sums to a count or `:sum`, running extremes to `:max` or `:min`, collected
  predictions to `:concat`.
- **Do not unpack `outputs` again.** If `forward` returns a single array, `metrics` receives that
  array. `first(outputs)` gets you element one of the prediction matrix, and every operation after
  it stays broadcast-legal, so the run reports a metric over one sample without raising.
