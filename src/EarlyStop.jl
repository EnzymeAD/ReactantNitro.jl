# EarlyStop.jl
#
# Early stopping: the policy, its setup-time checks, and why it is independent of the checkpointer.

"""
    EarlyStopping(; metric = :val_loss, mode = :min, patience = 5, min_delta = 1f-4)

A small dedicated component, **independent of the checkpointer** even though both track improvement.
The metrics can legitimately differ (checkpoint on `mae`, stop on `val_loss`), and the jobs differ
(retention versus control flow); they share only the *convention* of a `metric` name and a
`:min`/`:max` mode. Duplicating three lines of comparison beats the abstraction that would avoid it.

```julia
train!(e; early_stop = EarlyStopping(; metric = :val_loss, mode = :min,
                                       patience = 5, min_delta = 1f-4))
```

`:val_loss` is not a magic name: it is what the default `metrics` emits. An experiment defining its
own `metrics` names one of its own keys instead, and **a metric no `metrics` emits is a setup error
naming the available ones**, checked before training rather than at the first epoch.

`patience` counts **epochs without improvement**; `min_delta` is **absolute**. Both match Keras and
Lightning.

**`early_stop` defaults to `nothing`, so a bare experiment does not early-stop.** A framework that
truncates your run by default is surprising, any patience value would be a guess, and the licence
this framework takes to be opinionated is scoped to where the evidence is one-sided, which here it
is not. The test suite is worded to match and asserts a bare experiment does **not** early-stop.

**Stopping is graceful:** finish the epoch, validate, checkpoint, finalize the logger, exit through
the normal [`Done`](@ref) path. Aborting mid-epoch skips exactly the steps that make the run useful.

**Stopping is recorded** as `stop_reason` in the checkpoint record, since "completed 40/40" and
"stopped at 37 on patience" are different outcomes. That also makes resume comprehensible: a run that
early-stopped and is resumed with `:auto` would immediately re-satisfy the condition, and with the
reason stored it says so instead of exiting silently.

[`request_stop!`](@ref) sets the same flag imperatively, from a REPL or a phase monitor. Both are
checked once per epoch after validation.
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

The stopping policy. Custom policies implement this for their own type; a `::Nothing` method returns
`false`, which is how "no early stopping" stays the default with no branch in the driver.

`patience` counts **epochs without improvement** and `min_delta` is **absolute**, both matching Keras
and Lightning: an epoch improves when it beats the best seen **by more than `min_delta`**, and
anything else, including an equal or slightly better value, counts against patience.

**The metric is read through [`check_control_readback`](@ref)**, because this is control flow and
every scalar the framework branches on must be validated: a failed `BufferToHost` on this stack
returns garbage without raising, and a garbage value here either truncates a healthy run or lets a
stalled one continue.
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

The setup-time early-stopping checks, run before training rather than at the first epoch.

Three of the four things that can be wrong here are visible at setup and are errors here. The fourth,
a metric name no `metrics` emits, is **only** visible at setup when the experiment defines no
`metrics` at all, since the framework then knows the whole key set is `(:val_loss,)`; otherwise the
keys exist only once the hook has run, and [`should_stop`](@ref) raises at the first epoch naming
what was actually emitted. Saying so is better than implying a check the framework cannot make.
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
