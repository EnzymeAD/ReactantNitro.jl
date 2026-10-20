# Metrics: the denominator is yours

A metric returned by [`metrics`](@ref) is `value => mode`. The mode says how the framework
combines the value across the batches of a split. For a mean, the mode is the count to divide by,
and you supply it: the framework has no single right count (samples, pixels, tokens, weights).

| Mode | Across batches | [`finalize_metrics`](@ref) receives |
| --- | --- | --- |
| a count | values summed, counts summed | `value / count` |
| `:sum` | summed | the total |
| `:max`, `:min` | elementwise max or min | the extreme |
| `:concat` | joined along the last axis | an `(…, N)` array, one entry per sample |

```julia
function ReactantNitro.metrics(::MyExp, ŷ; y)
    err = abs.(ŷ .- y)
    return (;
        mae   = sum(err) => length(err),      # a count: a mean per element
        worst = maximum(err) => :max,
        sse   = sum(abs2, err) => :sum,
    )
end
```

!!! info "`=>` is not a lambda"
    `=>` builds a `Pair`, as in `Dict(:a => 1)`; it is not the lambda arrow `->`. The value is
    computed where it is written, so `count(y) => :sum` holds a number. A ternary binds looser than
    `=>`, so wrap it: `(isempty(y) ? 0 : n) => :sum`.

A count is any real number, so a weighted mean is `sum(w .* err) => sum(w)`. Any other mode is
refused on the first batch that returns it.

Metrics never see padding. The framework pads a short final batch, runs the compiled program,
slices the outputs back, and only then calls you. With no `metrics` method the framework reports
`loss = loss(e, outputs; ...) => 1`, the mean of the per-batch losses, so a bare experiment
still validates.

**Names carry no split.** The split is the context a metric is logged under, so the validation loss
is `loss`, as the train loss is, and an accuracy is `acc`, never `val_acc`. A backend shows the
context its own way (`validate_acc` on Comet, `validate/acc` on TensorBoard), and that display name
is not the key. [`monitor`](@ref) and everything that follows it read the validation metrics by
these keys. `:val_loss`, the name before 0.2.0, is accepted as `:loss` with a warning until 1.0.0.

## The loss and its terms

`loss` may return `(l, terms)`, with `terms` a flat `NamedTuple` of scalars. Each term is logged as
`loss/<term>`, the same key on every split: on the train line, read from the gradient program
itself so nothing is recomputed, and on validation and test wherever the loss is reported. Only `l`
is differentiated.

```julia
ReactantNitro.loss(e::MyExp, ŷ; y) = (ce = crossent(ŷ, y); kl = kldiv(ŷ); (ce + 0.1f0 * kl, (; ce, kl)))
```

Without `metrics`, validation reports `loss` and each `loss/<term>`. With `metrics`, it reports
exactly what they return, unless [`eval_loss`](@ref) is on: then each eval batch also runs `loss` in
its own small compiled program, on the device as in training, and `loss` and its terms are added
after `finalize_metrics`. A key both report is an error naming both. Prefer it to calling `loss`
inside a `:host` `metrics`, which runs it on the CPU, outside the trained program.

[`monitor`](@ref) defaults to `:loss => :min` only when that key is guaranteed: no `metrics`, or
`eval_loss` on. With `metrics` and `eval_loss` off there is no default, and setup refuses a ranking
checkpointer, an `EarlyStopping` or a `Plateau` until you declare one.

## Groups

A key can hold a group: a `NamedTuple` (or `Tuple`) of related values. A mode on the group covers
every value in it, and a mode on a single value covers just that value, so a group can mix modes:

```julia
ranking = (; score = p, label = y) => :concat              # one mode for the whole group
ranking = (; score = p => :concat, label = y => :concat,   # a mode per value
             n_pos = count(y) => :sum)
```

Every value in a group needs exactly one mode, from itself or from a group above it. A value with
none, or with two, is refused with its path in the error, such as `ranking.label`.
[`finalize_metrics`](@ref) receives the same shape with the modes removed.

### One split, traced end to end

A binary classifier reporting accuracy, its worst error, and AUROC, over a split of two batches:
three samples, then two.

```julia
function ReactantNitro.metrics(::Classifier, logit; label)
    p = sigmoid.(vec(logit))                  # (B,) predicted probability
    y = vec(label) .> 0.5                     # (B,) true class
    return (;
        accuracy = sum((p .> 0.5) .== y) => length(y),
        worst    = maximum(abs.(p .- y)) => :max,
        ranking  = (; score = p => :concat,
                      label = y => :concat,
                      n_pos = count(y) => :sum),
    )
end
```

Each batch returns:

| Name | Batch 1: `p = [0.9, 0.2, 0.6]`, `y = [1, 0, 0]` | Batch 2: `p = [0.3, 0.8]`, `y = [1, 1]` |
| --- | --- | --- |
| `accuracy` | `2 => 3` | `1 => 2` |
| `worst` | `0.6 => :max` | `0.7 => :max` |
| `ranking.score` | `[0.9, 0.2, 0.6] => :concat` | `[0.3, 0.8] => :concat` |
| `ranking.label` | `[1, 0, 0] => :concat` | `[1, 1] => :concat` |
| `ranking.n_pos` | `1 => :sum` | `2 => :sum` |

The framework combines each value by its own mode. The names do not change:

| Name | Combined | Received by `finalize_metrics` |
| --- | --- | --- |
| `accuracy` | values `2 + 1`, counts `3 + 2` | `0.6` |
| `worst` | `max(0.6, 0.7)` | `0.7` |
| `ranking.score` | joined | `[0.9, 0.2, 0.6, 0.3, 0.8]` |
| `ranking.label` | joined, aligned with `score` | `[1, 0, 0, 1, 1]` |
| `ranking.n_pos` | `1 + 2` | `3` |

```julia
function ReactantNitro.finalize_metrics(::Classifier, acc, split)
    (; score, label, n_pos) = acc.ranking
    n_neg = length(label) - n_pos
    r = tiedrank(score)                       # StatsBase: ties share the average rank
    auroc = (sum(r[label]) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return (; acc.accuracy, acc.worst, auroc)
end
```

The ranks are `[5, 1, 3, 2, 4]` and the positives hold 5, 2, and 4, so
`auroc = (11 - 6) / 6 = 0.833`: the positive scores higher in 5 of the 6 positive-negative pairs.
Only what `finalize_metrics` returns, `(; accuracy = 0.6, worst = 0.7, auroc = 0.833)`, reaches the
run history, the logger, and checkpoint selection, so `monitor(e) = :auroc => :max` works.

`n_pos` is here to show a mixed group; `count(label)` in `finalize_metrics` gives the same number.

## Rules

- **One mode per value, on every batch.** Every batch returns the same keys, and a value keeps its
  mode from batch to batch.
- **`:max` and `:min` combine per-batch extremes.** The hook returns the batch's own
  `maximum(err)`; the framework takes the max across batches.
- **`:concat` is batch-last and strict.** Each value's last axis must equal the batch's real sample
  count. Batch-first values are refused; move the sample axis last with `permutedims`. Scalars are
  refused, since a single number per batch depends on how the split is cut into batches.
- **`:concat` values are aligned, not ordered.** Entry `i` of every `:concat` value is the same
  sample. Dataset order is not promised: concatenate an `id` field and sort by it if you need it.
- **`:concat` is an input to `finalize_metrics` only.** Reduce it there. A `:concat` array returned
  unreduced is refused, because it would be stored in the run history every epoch, logged, and
  written into every checkpoint. Per-sample outputs for analysis come from [`predict`](@ref).
- **Values keep their type.** Each value accumulates in the type the hook returns. Return `Float64` where
  precision matters, such as sums of squares.

`:concat` holds the split's per-sample values in host memory for one evaluation pass. The other
modes hold one value per key.

## Host or device

[`metrics_residency`](@ref)`(e, hook)` sets where each hook runs, `:device` or `:host`:

| | `:device` (traced) | `:host` (ordinary Julia) |
| --- | --- | --- |
| Performance | the model's raw outputs never cross the device boundary; only the metric values come back | every eval batch transfers the outputs to the host |
| Expressiveness | must be expressible as a traced program | anything Julia can do: matching, connected components, sorting with tie-breaking, any library |
| Editing the hook | edits the program, so it recompiles | part of no program, so it recompiles nothing |

The defaults follow how often each hook runs. `train_metrics` runs once per micro-batch, so it is
traced by default: transferring a full output batch that often costs far more than the few scalars
it produces. `metrics` runs once per eval batch, once per epoch, so the same transfer is small
against an epoch of training, and host residency lets evaluation code use any Julia code.

```julia
# The defaults, written out. You would not normally write this line.
ReactantNitro.metrics_residency(::MyExp, hook) =
    hook === :train_metrics ? :device : :host

# Trace the validation metric instead: it is cheap to express and the eval set is large.
ReactantNitro.metrics_residency(::MyExp, ::Symbol) = :device
```

Both hooks accept both values. For the hook author, the only difference is what `outputs` holds:
device arrays under `:device`, host arrays under `:host`. Keyword routing, the modes, and the
no-padding guarantee are the same either way. A traced `train_metrics` is part of the gradient
program, so editing it recompiles that program; a host metric can be edited without a recompile.

[`train_metrics`](@ref) returns a `NamedTuple` of plain scalars, with no modes: nothing is
accumulated. It runs on every micro-batch, also under `log_every_n_steps`, but only the values of
a logged step's last micro-batch are written to the train line.

## A worked pair

Validation metrics for the `MnistMLP` from the [Tutorial](tutorial.md), in host mode, the default
for `metrics`: ordinary Julia on transferred arrays, once per eval batch. `validate(nitro)` reports
these every epoch, and `evaluate(nitro; split = :test)` runs the same hooks on the test split.

```julia
function ReactantNitro.metrics(::MnistMLP, logits; label)
    pred = getindex.(argmax(logits; dims = 1), 1)    # (1, B), the predicted class
    truth = getindex.(argmax(label; dims = 1), 1)    # (1, B), the true class
    cm = zeros(Int, 10, 10)
    for (p, t) in zip(pred, truth)
        cm[p, t] += 1
    end
    return (;
        acc = sum(pred .== truth) => size(label, 2),  # per image: divided at the end
        confusion = cm => :sum,                       # summed across the split, never divided
    )
end

function ReactantNitro.finalize_metrics(::MnistMLP, acc, split)
    recall = [acc.confusion[c, c] / max(sum(acc.confusion[:, c]), 1) for c in 1:10]
    return (; acc.acc, macro_recall = sum(recall) / 10)
end
```

`acc` carries a denominator and `confusion` does not. In `finalize_metrics`, counted keys arrive
already divided and `:sum` keys as raw totals, so `acc.confusion` is the whole split's matrix and
`macro_recall` is computed once at the end rather than averaged per batch. Neither hook is traced,
so editing either recompiles nothing.

Compute anything that is not a mean of per-batch values in `finalize_metrics`. Averaging
per-batch recalls instead of computing recall from the split's confusion matrix gives a different
and wrong number.

## R², exact from sums

R² needs the split's mean target, which no single batch knows. The total sum of squares expands
into sums, so `:sum` is enough:

```math
SS_\text{tot} = \sum (y - \bar y)^2 = \sum y^2 - \frac{(\sum y)^2}{n}
```

```julia
function ReactantNitro.metrics(::MyRegressor, ŷ; y)
    d = Float64.(y .- 0.5)                  # a fixed shift: see below
    r = Float64.(y .- ŷ)
    return (; r2 = (; n = length(y), sy = sum(d), syy = sum(abs2, d), sres = sum(abs2, r)) => :sum)
end

function ReactantNitro.finalize_metrics(::MyRegressor, acc, split)
    (; n, sy, syy, sres) = acc.r2
    return (; r2 = 1 - sres / (syy - sy^2 / n))
end
```

This is the split's R², not a mean of per-batch values. `syy - sy^2 / n` loses precision when the
target mean is large relative to its spread; the `Float64` and the shift reduce that. Any fixed
shift leaves the variance unchanged; the training target mean is a good choice.

## AUROC, binned

The [traced example](@ref "One split, traced end to end") computes AUROC exactly from `:concat`,
which holds every score in host memory. For a large split, a histogram of scores per class is a
`:sum` value of fixed size, and the result is accurate to the bin width:

```julia
function ReactantNitro.metrics(::MyClassifier, logit; label)
    B = 1000
    bin = clamp.(ceil.(Int, sigmoid.(vec(logit)) .* B), 1, B)
    h = zeros(Int, 2, B)                    # row 1 positives, row 2 negatives
    for (b, y) in zip(bin, vec(label))
        h[y > 0.5 ? 1 : 2, b] += 1
    end
    return (; roc = h => :sum)
end

function ReactantNitro.finalize_metrics(::MyClassifier, acc, split)
    tp = [reverse(cumsum(reverse(acc.roc[1, :]))); 0]    # positives at or above each bin
    fp = [reverse(cumsum(reverse(acc.roc[2, :]))); 0]
    tpr, fpr = tp ./ tp[1], fp ./ fp[1]
    return (; auroc = sum((fpr[i] - fpr[i + 1]) * (tpr[i] + tpr[i + 1]) / 2 for i in 1:(length(tpr) - 1)))
end
```

## Which mode a metric needs

A metric that is a function of sums over samples is exact with counts and `:sum`: MSE, MAE, R²,
Pearson correlation (from five sums), precision, recall, F1, global Dice and IoU, and calibration
error over fixed bins. A metric that depends on ranks or order needs `:concat`, or a binned `:sum`
approximation: AUROC, average precision, Spearman correlation, medians, and quantiles.

## Older spellings

`(value, mode)` is the older spelling of `value => mode`, and `nothing` the older spelling of
`:sum`, so `(cm, nothing)` and `cm => :sum` mean the same. The tuple form is accepted only at the
top of a key; `nothing` is accepted wherever a mode is.

## See also

- The [Tutorial](tutorial.md) builds these hooks up in context, against real MNIST.
- [Logging](logging.md) for where the finalized numbers go, including the confusion matrix, which
  the framework never sends on your behalf.
