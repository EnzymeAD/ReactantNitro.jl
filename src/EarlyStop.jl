# EarlyStop.jl
#
# Early stopping: the policy, its setup-time checks, and why it is independent of the checkpointer.

"""
    EarlyStopping(; metric = :val_loss, mode = :min, patience = 5, min_delta = 1f-4)

The stopping policy, independent of the checkpointer: the metrics can differ and the jobs differ
(retention versus control flow). `:val_loss` is what the default `metrics` emits; an experiment
defining its own `metrics` names one of its keys, and a metric nothing emits is a setup error.
`patience` counts epochs without improvement and `min_delta` is absolute, matching Keras and
Lightning. `early_stop` defaults to `nothing`, so a bare experiment does not early-stop.

```julia
train!(e; early_stop = EarlyStopping(; metric = :val_loss, mode = :min,
                                       patience = 5, min_delta = 1f-4))
```

Stopping is graceful (the epoch finishes, validates, checkpoints, and exits through `Done`) and
recorded as `stop_reason` in the checkpoint record, so a resume into a stopped run says so.
[`request_stop!`](@ref) sets the same flag imperatively.
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

The stopping policy; custom policies implement this for their own type, and the `::Nothing` method
returns `false`. An epoch improves when it beats the best seen by more than `min_delta`. The metric
is read through [`check_control_readback`](@ref), since it drives control flow.
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
the experiment defines no `metrics` (so the only key is `:val_loss`), the metric name. Otherwise
the keys exist only once the hook has run, and [`should_stop`](@ref) raises at the first epoch.
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
