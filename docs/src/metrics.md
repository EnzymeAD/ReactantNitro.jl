# Metrics: the denominator is yours

A metric is a `(sum, count)` pair, so it carries its own denominator. A per-image accuracy, a
per-token loss, and a confusion matrix coexist in one call, and the framework never supplies a
sample count of its own, because there is no single right one.

`count === nothing` means accumulate by summation and never divide, which is what a confusion
matrix wants.

Metrics never see padding. The framework pads a short final batch, runs the compiled program,
slices the outputs back, and only then calls you. With no `metrics` method at all the framework
substitutes `val_loss`, so a bare experiment still validates.

## Host or device

Where a metric runs is a per-hook choice that trades compile cost, expressiveness, and transfer
cost against each other:

| | `:device` (traced) | `:host` (ordinary Julia) |
| --- | --- | --- |
| Performance | the model's raw outputs never cross the device boundary; only the metric scalars come back | every eval batch transfers the outputs to the host |
| Expressiveness | must be expressible as a traced program | anything Julia can do: matching, connected components, sorting with tie-breaking, any library |
| Editing the hook | edits the program, so it recompiles | part of no program, so it recompiles nothing |

The defaults follow the cadence. `train_metrics` runs once per micro-batch, so it is traced by
default; transferring a full output batch that often would cost far more than the few scalars it
produces. `metrics` runs once per eval batch, once per epoch, after a training epoch of thousands
of steps, so the same transfer is noise, and host residency lets evaluation code use anything Julia
can do.

```julia
# The defaults, written out. You would not normally write this line.
ReactantNitro.metrics_residency(::MyExp, hook) =
    hook === :train_metrics ? :device : :host

# Trace the validation metric instead: it is cheap to express and the eval set is large.
ReactantNitro.metrics_residency(::MyExp, ::Symbol) = :device
```

Both hooks accept both values. The only difference for the hook author is what `outputs` is: device
arrays under `:device`, host arrays under `:host`. Keyword routing, the `(sum, count)` contract,
`count === nothing`, and the no-padding guarantee are the same either way. The cost to weigh is
compilation: editing a traced `train_metrics` repays the gradient compile, while a host metric can
be added mid-session without one.

## A worked pair

Validation metrics for the `MnistMLP` from the quick start, in host mode, the default for
`metrics`: ordinary Julia on transferred arrays, once per eval batch. `validate(nitro)` reports
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
        acc = (sum(pred .== truth), size(label, 2)),  # per image: divided at the end
        confusion = (cm, nothing),                    # summed across the split, never divided
    )
end

function ReactantNitro.finalize_metrics(::MnistMLP, acc, split)
    recall = [acc.confusion[c, c] / max(sum(acc.confusion[:, c]), 1) for c in 1:10]
    return (; acc.acc, macro_recall = sum(recall) / 10)
end
```

`acc` carries a denominator and `confusion` carries none. In `finalize_metrics`, counted keys
arrive already divided and `nothing`-counted keys as raw totals, so `acc.confusion` is the whole
split's matrix and `macro_recall` is computed once at the end rather than averaged per batch.
Neither hook is traced, so editing either recompiles nothing.

The general rule for `finalize_metrics`: reduce host-side for anything that is not a mean of
per-batch values. Averaging per-batch recalls instead of computing recall from the split's
confusion matrix gives a different, wrong number that still looks plausible.

## See also

- The [Tutorial](tutorial.md) builds these hooks up in context, against real MNIST.
- [Logging](logging.md) for where the finalized numbers go, including the confusion matrix, which
  the framework never sends on your behalf.
