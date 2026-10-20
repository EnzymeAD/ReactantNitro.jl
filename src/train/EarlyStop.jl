# EarlyStop.jl
#
# Early stopping: the policy, its setup-time checks, and why it is independent of the checkpointer.

"""
    EarlyStopping(; metric = nothing, mode = nothing, patience = 5, min_delta = 1f-4)

Stop training when a validation metric stops improving. Pass it as the `early_stop` keyword or
accessor; the default `nothing` never stops early.

`metric` and `mode` default to `nothing`, which follows the run's [`monitor`](@ref); give both to
watch something else. `patience` counts epochs without improvement, and an epoch improves when it
beats the best so far by more than `min_delta` (absolute, as in Keras and Lightning). Setup requires
a `val` split and checks `mode`, `patience` and, for an experiment without `metrics`, that `metric`
is `:loss`. With a custom `metrics` the key cannot be checked before it runs, so a misspelled
one raises after the first epoch's validation.

```julia
train!(e; early_stop = EarlyStopping(; patience = 5))                              # the monitor
train!(e; early_stop = EarlyStopping(; metric = :loss, mode = :min, patience = 5))
```

The stopping epoch still validates and checkpoints, then `train!` returns through `Done` with
`stop_reason = :early_stop`, which the final checkpoint record stores. [`request_stop!`](@ref)
stops at the same point on demand, with `stop_reason = :requested`. Setup copies the instance with
the monitor filled in and fresh `best` and `wait` counters, so two handles never share them and the
instance you passed is not mutated; the handle's copy is `nitro.early_stop`. The counters are
stored in each checkpoint, and a resume that watches the same metric and mode continues from them.
"""
mutable struct EarlyStopping
    metric::Union{Symbol, Nothing}
    mode::Union{Symbol, Nothing}  # :min or :max
    patience::Int
    min_delta::Float64
    best::Union{Float64, Nothing}
    wait::Int
end

function EarlyStopping(;
        metric::Union{Symbol, Nothing} = nothing, mode::Union{Symbol, Nothing} = nothing,
        patience::Int = 5, min_delta::Real = 1.0f-4
    )
    own = _own_monitor(metric, mode, "early stopping")
    m, md = own === nothing ? (nothing, nothing) : (first(own), last(own))
    return EarlyStopping(m, md, patience, Float64(min_delta), nothing, 0)
end

"""
    should_stop(es, epoch, metrics) -> Bool

Update the policy with one epoch's validation `metrics` and return whether to stop. A custom
policy implements this method for its own type; `should_stop(::Nothing, ...)` returns `false`. For
[`EarlyStopping`](@ref), an epoch improves when it beats the best so far by more than `min_delta`,
and the run stops after `patience` epochs in a row without improvement. The metric is read through
[`check_control_readback`](@ref), which raises on a non-finite or non-real value.
"""
function should_stop(es::EarlyStopping, epoch, metrics)
    es.metric === nothing && error(
        "ReactantNitro: this `EarlyStopping` follows the run's `monitor` and has not been through \
         `Nitro` setup, which fills it in. Give it `metric` and `mode` to use it on its own."
    )
    v = read_watched_metric(metrics, es.metric, "early stopping")
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

# The resumable counters the checkpoint record stores; `nothing` for a policy without any.
early_stop_state(es) = nothing
early_stop_state(es::EarlyStopping) = (; es.metric, es.mode, es.best, es.wait)

# The handle's own copy, following `monitor` unless it has its own metric, with fresh counters or a
# resumed record's when it watched the same metric and mode. A policy of another type is used as
# given.
setup_early_stop(es, state, monitor) = es
function setup_early_stop(es::EarlyStopping, state, monitor)
    # Left unfilled without a monitor; `check_monitor_needed` refuses that before training.
    es.metric === nothing && monitor === nothing && return es
    m, md = es.metric === nothing ? (first(monitor), last(monitor)) : (es.metric, es.mode)
    own = EarlyStopping(m, md, es.patience, es.min_delta, nothing, 0)
    # A record written before 0.2.0 stores `:val_loss`.
    if state !== nothing && (state.metric === :val_loss ? :loss : state.metric) === own.metric &&
            state.mode === own.mode
        own.best, own.wait = state.best, state.wait
    end
    return own
end

"""
    ReactantNitro.check_early_stop(es, collection, routing) -> nothing

The setup-time early-stopping checks: `mode`, `patience`, the presence of a `val` split, and, when
the experiment defines no `metrics` (so the only key is `:loss`), the metric name. A custom
`metrics` has no known keys before it runs, so [`should_stop`](@ref) checks the name at the first
epoch.
"""
check_early_stop(::Nothing, collection, routing) = nothing

function check_early_stop(es::EarlyStopping, collection, routing)
    es.mode in (:min, :max) || error("ReactantNitro: early stopping `mode` is $(repr(es.mode)); it \
        must be `:min` or `:max`.")
    es.patience >= 1 || error("ReactantNitro: early stopping `patience` is $(es.patience); it \
        counts epochs without improvement and must be at least 1.")
    return check_watched_metric("early stopping", es.metric, collection, routing)
end

"""
    ReactantNitro.check_watched_metric(what, metric, collection, routing) -> nothing

The setup-time checks for a policy that reads a validation metric each epoch: a `val` split exists,
and, when the experiment defines no `metrics`, `metric` is `:loss`. Shared by
[`EarlyStopping`](@ref) and [`Plateau`](@ref).
"""
function check_watched_metric(what, metric::Symbol, collection, routing)
    haskey(collection, :val) || error(
        """
        ReactantNitro: $what is configured, and there is no `val` split to evaluate it on.
        `build_data` returned $(keys(collection)).
        It is checked once per epoch AFTER validation, so without a `val` split there is nothing
        for it to read and it would silently never act."""
    )
    (routing !== nothing && routing.metrics === nothing && metric !== :loss) && error(
        """
        ReactantNitro: $what is configured on `$metric`, and this experiment defines no `metrics`
        method, so the only metric the framework will emit is `:loss`, the validation loss.
        Either watch `:loss`, or define `metrics` emitting `$metric`.$(split_prefix_hint(metric))"""
    )
    return nothing
end

"""
    ReactantNitro.resolve_monitor(monitor, defines_metrics, eval_loss) -> Pair or nothing

The run's `monitor`: as declared, else `:loss => :min` when the validation loss is guaranteed (no
`metrics`, or `eval_loss`), else `nothing`, which [`check_monitor_needed`](@ref) refuses where
something would follow it.
"""
resolve_monitor(monitor::Pair, defines_metrics, eval_loss) = monitor
resolve_monitor(::Nothing, defines_metrics::Bool, eval_loss) =
    (!defines_metrics || eval_loss) ? (:loss => :min) : nothing

"""
    ReactantNitro.check_monitor_needed(monitor, checkpointer, early_stop, sched) -> nothing

With no `monitor`, refuse each component that would follow one: a checkpointer with a `Best`
selector, an `EarlyStopping` and every `Plateau` without their own `metric` and `mode`.
"""
function check_monitor_needed(monitor, checkpointer, early_stop, sched)
    monitor === nothing || return nothing
    who = String[]
    checkpointer isa Checkpointer && getfield(checkpointer, :own) === nothing &&
        any(s -> s isa Best, checkpointer.keep) && push!(who, "the checkpointer's `Best` selector")
    early_stop isa EarlyStopping && early_stop.metric === nothing && push!(who, "early stopping")
    for (k, s) in plateaus(sched)
        s.p.metric === nothing && push!(who, "the plateau schedule `$k`")
    end
    isempty(who) && return nothing
    error(
        """
        ReactantNitro: $(join(who, ", ")) ranks by the run's `monitor`, and this experiment has none.
        It defines `metrics`, so the validation loss is not reported unless asked for, and there is
        no default to fall back on. Declare the metric the run is judged by, a key `metrics` (or
        `finalize_metrics`) returns:
            ReactantNitro.monitor(::MyExp) = :acc => :max
        or report the validation loss as well and keep the default `:loss => :min`:
            ReactantNitro.eval_loss(::MyExp) = true
        A checkpointer that should not rank at all: `Checkpointer(; keep = Last(1))`."""
    )
end

"""
    ReactantNitro.check_monitor_emitted(monitor, collection, routing) -> nothing

Refuse a `monitor` other than `:loss` on an experiment with a `val` split and no `metrics`,
whose only validation key is `:loss`.
"""
function check_monitor_emitted(monitor, collection, routing)
    monitor === nothing && return nothing
    (
        haskey(collection, :val) && routing !== nothing && routing.metrics === nothing &&
            first(monitor) !== :loss
    ) && error(
        """
        ReactantNitro: the run's `monitor` is `$(first(monitor))`, and this experiment defines no
        `metrics` method, so the only metric the framework will emit is `:loss`, the validation loss.
        Either set `monitor = :loss => :min`, or define `metrics` emitting `$(first(monitor))`.$(split_prefix_hint(first(monitor)))"""
    )
    return nothing
end

"""
    ReactantNitro.read_watched_metric(metrics, metric, what) -> Float64

Read the watched `metric` from one epoch's validation `metrics` through
[`check_control_readback`](@ref), raising when `metrics` does not emit it.
"""
function read_watched_metric(metrics, metric::Symbol, what)
    haskey(metrics, metric) || error(
        """
        ReactantNitro: $what is configured on `$metric`, which `metrics` does not emit. The
        available metrics are $(keys(metrics)).
        `:loss` is what the framework supplies when an experiment defines no `metrics`; an
        experiment defining its own names one of its own keys.$(split_prefix_hint(metric))"""
    )
    return check_control_readback(getproperty(metrics, metric), what, metric)
end
