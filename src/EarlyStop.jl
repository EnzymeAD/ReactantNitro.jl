# EarlyStop.jl
#
# Early stopping: the policy, its setup-time checks, and why it is independent of the checkpointer.

"""
    EarlyStopping(; metric = :val_loss, mode = :min, patience = 5, min_delta = 1f-4)

Stop training when a validation metric stops improving. Pass it as the `early_stop` keyword or
accessor; the default `nothing` never stops early. It is configured separately from the
checkpointer and may watch a different metric.

`metric` is a key of the validation metrics: `:val_loss` when the experiment defines no `metrics`,
otherwise one of the keys its `metrics` returns. `mode` is `:min` or `:max`. `patience` counts
epochs without improvement, and an epoch improves when it beats the best so far by more than
`min_delta` (absolute, as in Keras and Lightning). Setup requires a `val` split and checks `mode`,
`patience` and, for an experiment without `metrics`, that `metric` is `:val_loss`; a key missing
from a custom `metrics` raises at the first epoch.

```julia
train!(e; early_stop = EarlyStopping(; metric = :val_loss, mode = :min,
                                       patience = 5, min_delta = 1f-4))
```

The stopping epoch still validates and checkpoints, then `train!` returns through `Done` with
`stop_reason = :early_stop`, which the final checkpoint record stores. [`request_stop!`](@ref)
stops at the same point on demand, with `stop_reason = :requested`. The `best` and `wait` counters
live in the instance and are not checkpointed, so a resume does not restore them.
"""
mutable struct EarlyStopping
    metric::Symbol
    mode::Symbol            # :min or :max
    patience::Int
    min_delta::Float64
    best::Union{Float64, Nothing}
    wait::Int
end

EarlyStopping(;
    metric::Symbol = :val_loss, mode::Symbol = :min, patience::Int = 5,
    min_delta::Real = 1.0f-4
) =
    EarlyStopping(metric, mode, patience, Float64(min_delta), nothing, 0)

"""
    should_stop(es, epoch, metrics) -> Bool

Update the policy with one epoch's validation `metrics` and return whether to stop. A custom
policy implements this method for its own type; `should_stop(::Nothing, ...)` returns `false`. For
[`EarlyStopping`](@ref), an epoch improves when it beats the best so far by more than `min_delta`,
and the run stops after `patience` epochs in a row without improvement. The metric is read through
[`check_control_readback`](@ref), which raises on a non-finite or non-real value.
"""
function should_stop(es::EarlyStopping, epoch, metrics)
    haskey(metrics, es.metric) || error(
        """
        ReactantNitro: early stopping is configured on `$(es.metric)`, which `metrics` does not
        emit. The available metrics are $(keys(metrics)).
        `:val_loss` is not a magic name: it is what the framework substitutes when an experiment
        defines no `metrics`. An experiment defining its own names one of its own keys."""
    )
    v = check_control_readback(getproperty(metrics, es.metric), "early stopping", es.metric)
    improved = es.best === nothing ||
        (es.mode === :min ? v < es.best - es.min_delta : v > es.best + es.min_delta)
    if improved
        es.best = v
        es.wait = 0
        return false
    end
    es.wait += 1
    return es.wait >= es.patience
end

should_stop(::Nothing, epoch, metrics) = false

"""
    ReactantNitro.check_early_stop(es, collection, routing) -> nothing

The setup-time early-stopping checks: `mode`, `patience`, the presence of a `val` split, and, when
the experiment defines no `metrics` (so the only key is `:val_loss`), the metric name. A custom
`metrics` has no known keys before it runs, so [`should_stop`](@ref) checks the name at the first
epoch.
"""
check_early_stop(::Nothing, collection, routing) = nothing

function check_early_stop(es::EarlyStopping, collection, routing)
    es.mode in (:min, :max) || error("ReactantNitro: early stopping `mode` is $(repr(es.mode)); it \
        must be `:min` or `:max`.")
    es.patience >= 1 || error("ReactantNitro: early stopping `patience` is $(es.patience); it \
        counts epochs without improvement and must be at least 1.")
    haskey(collection, :val) || error(
        """
        ReactantNitro: early stopping is configured, and there is no `val` split to evaluate it on.
        `build_data` returned $(keys(collection)).
        The stopping condition is checked once per epoch AFTER validation, so without a `val`
        split there is nothing for it to read and the run would silently never stop early."""
    )
    (routing !== nothing && routing.metrics === nothing && es.metric !== :val_loss) && error(
        """
        ReactantNitro: early stopping is configured on `$(es.metric)`, and this experiment defines
        no `metrics` method, so the only metric the framework will emit is `:val_loss`.
        Either stop on `:val_loss`, or define `metrics` emitting `$(es.metric)`."""
    )
    return nothing
end
