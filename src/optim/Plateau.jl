# Plateau.jl
#
# Metric-driven schedules: `Plateau` scales any schedule down when a validation metric stops
# improving, as PyTorch's `ReduceLROnPlateau` does.

"""
    Plateau(inner; metric = nothing, mode = nothing, factor = 0.1, patience = 10, cooldown = 0,
            threshold = 1f-4, threshold_mode = :rel, min_scale = 0)

A schedule factory that multiplies `inner` by a scale, and reduces the scale when the validation
metric stops improving. `inner` is any schedule factory or a bare `Number`; a bare number is
PyTorch's `ReduceLROnPlateau`. `metric` and `mode` default to `nothing`, which follows the run's
[`monitor`](@ref); give both to watch something else. It works on every schedule key: `opt` (bare and group paths),
`device`, `grad` and `bind`.

```julia
schedules(e::MyExp) = (;
    eta = Plateau(total -> OneCycle(total, 1f-3); patience = 3),
)
```

The value at step `t` is `inner(t) * scale`. The scale starts at 1 and is updated once per epoch,
after validation, with PyTorch's rule and defaults:

- an epoch improves when the metric beats the best so far by more than `threshold`, relative
  (`best * (1 - threshold)` for `:min`) under `threshold_mode = :rel`, or absolute
  (`best - threshold`) under `:abs`
- after more than `patience` epochs in a row without improvement, `scale = max(scale * factor,
  min_scale)`, and the next `cooldown` epochs count no bad epochs

So `patience = 2` reduces at the third bad epoch, as in PyTorch, and `patience = 0` reduces at every
bad one. A reduction applies from the next epoch's first step.

Each value is a pure function of the step and the run's validation history. The counters are stored
in each checkpoint, so a resume continues the same sequence; a `weights` load starts fresh. Each key
gets its own counters. The train line logs the scaled value as usual, the validation line logs
`schedule/plateau/<key>`, and each reduction is reported with `@info`.

Setup requires a `val` split, `mode` in `(:min, :max)`, `0 < factor < 1`, `patience >= 0`, `cooldown >= 0`, and, for an
experiment without `metrics`, `metric = :loss`. With a custom `metrics`, the key cannot be
checked before it runs, so a misspelled one raises after the first epoch's validation.
"""
struct Plateau{F}
    inner::F
    metric::Union{Symbol, Nothing}
    mode::Union{Symbol, Nothing}
    factor::Float64
    patience::Int
    cooldown::Int
    threshold::Float64
    threshold_mode::Symbol
    min_scale::Float64
end

function Plateau(
        inner; metric::Union{Symbol, Nothing} = nothing, mode::Union{Symbol, Nothing} = nothing,
        factor::Real = 0.1, patience::Integer = 10, cooldown::Integer = 0, threshold::Real = 1.0f-4,
        threshold_mode::Symbol = :rel, min_scale::Real = 0
    )
    own = _own_monitor(metric, mode, "`Plateau`")
    m, md = own === nothing ? (nothing, nothing) : (first(own), last(own))
    return Plateau(
        inner, m, md, Float64(factor), Int(patience), Int(cooldown), Float64(threshold),
        threshold_mode, Float64(min_scale)
    )
end

# `p` with the run's `monitor` filled in, unless it has its own metric.
_follow(p::Plateau, monitor) = p.metric !== nothing ? p : Plateau(
        p.inner, first(monitor), last(monitor), p.factor, p.patience, p.cooldown, p.threshold,
        p.threshold_mode, p.min_scale
    )

"""
    ReactantNitro.PlateauSchedule

What a [`Plateau`](@ref) returns when setup calls it with the horizon: the inner step function and
the counters. Mutated once per epoch by [`step_plateau!`](@ref).
"""
mutable struct PlateauSchedule{F}
    f::F
    p::Plateau
    best::Union{Float64, Nothing}
    bad_epochs::Int
    cooldown_left::Int
    scale::Float64
end

(p::Plateau)(total) = PlateauSchedule(_as_factory(p.inner)(total), p, nothing, 0, 0, 1.0)

# `oftype`, so a `Float32` schedule stays `Float32` and a device field keeps its type.
(s::PlateauSchedule)(step) = _scaled(s.f(step), s.scale)
_scaled(v::AbstractFloat, scale) = v * oftype(v, scale)
_scaled(v, scale) = v * scale

function _improved(p::Plateau, v, best)
    best === nothing && return true
    t = p.threshold
    rel = p.threshold_mode === :rel
    return p.mode === :min ? v < (rel ? best * (1 - t) : best - t) :
        v > (rel ? best * (1 + t) : best + t)
end

"""
    ReactantNitro.step_plateau!(s, v) -> Bool

Update a [`PlateauSchedule`](@ref) with one epoch's metric value `v`, in the order of PyTorch's
`ReduceLROnPlateau.step`. Returns whether the scale was reduced.
"""
function step_plateau!(s::PlateauSchedule, v::Real)
    p = s.p
    p.metric === nothing && error(
        "ReactantNitro: this `Plateau` follows the run's `monitor` and has not been through `Nitro` \
         setup, which fills it in. Give it `metric` and `mode` to use it on its own."
    )
    if _improved(p, v, s.best)
        s.best = Float64(v)
        s.bad_epochs = 0
    else
        s.bad_epochs += 1
    end
    if s.cooldown_left > 0
        s.cooldown_left -= 1
        s.bad_epochs = 0
    end
    s.bad_epochs > p.patience || return false
    s.cooldown_left = p.cooldown
    s.bad_epochs = 0
    new = max(s.scale * p.factor, p.min_scale)
    new < s.scale || return false
    s.scale = new
    return true
end

_unguard(f::HorizonGuard) = f.f
_unguard(f) = f

"""
    ReactantNitro.collect_plateaus(device, opt, grad, bind) -> Vector{Pair{Symbol, PlateauSchedule}}

Every [`PlateauSchedule`](@ref) in the resolved tables, keyed by its namespaced key: `opt` keys as
written (`eta`, `backbone.eta`), every other one with its namespace (`device.aux_weight`).
"""
function collect_plateaus(device, opt, grad, bind)
    out = Pair{Symbol, PlateauSchedule}[]
    for (ns, tbl) in ((:opt, opt), (:device, device), (:grad, grad), (:bind, bind)), (k, f) in pairs(tbl)
        s = _unguard(f)
        s isa PlateauSchedule && push!(out, (ns === :opt ? k : Symbol(ns, ".", k)) => s)
    end
    return out
end

plateaus(sched) = sched === nothing ? Pair{Symbol, PlateauSchedule}[] : sched.plateaus

"""
    ReactantNitro.check_plateaus(sched, collection, routing) -> nothing

The setup-time checks for each [`Plateau`](@ref): its own settings, then the watched metric through
[`check_watched_metric`](@ref).
"""
function check_plateaus(sched, collection, routing)
    for (k, s) in plateaus(sched)
        p = s.p
        what = "the plateau schedule `$k`"
        p.threshold_mode in (:rel, :abs) || error(
            "ReactantNitro: $what has `threshold_mode = $(repr(p.threshold_mode))`; it must be \
             `:rel` or `:abs`."
        )
        0 < p.factor < 1 ||
            error("ReactantNitro: $what has `factor = $(p.factor)`; it must be in (0, 1).")
        p.patience >= 0 && p.cooldown >= 0 && p.threshold >= 0 && p.min_scale >= 0 || error(
            "ReactantNitro: $what has a negative `patience`, `cooldown`, `threshold` or \
             `min_scale`."
        )
        check_watched_metric(what, p.metric, collection, routing)
    end
    return nothing
end

# A watched metric named with a split prefix is likely a misreading of the convention, since the
# validation metrics are named without one. A warning, since a `metrics` may really return it.
function warn_split_prefixes(watchers)
    named = unique(first(m) for (_, m) in watchers if !isempty(split_prefix_hint(first(m))))
    isempty(named) && return nothing
    @warn "ReactantNitro: watching $(join(("`$m`" for m in named), ", ")).\
           $(split_prefix_hint(first(named))) Use the key itself, as in `monitor = :acc => :max`; a \
           backend's display name such as `validate_acc` is not the key." maxlog = 1
    return nothing
end

# Fill in the run's `monitor` for every plateau that follows it, at setup.
function follow_monitor!(sched, monitor)
    for (_, s) in plateaus(sched)
        s.p = _follow(s.p, monitor)
    end
    return nothing
end

"""
    ReactantNitro.update_plateaus!(nitro, metrics) -> nothing

Update every plateau with one epoch's validation `metrics`, reporting each reduction with `@info`.
Runs before the epoch's checkpoint, so the record holds the updated counters.
"""
function update_plateaus!(nitro, metrics)
    for (k, s) in plateaus(nitro.schedules)
        old = s.scale
        v = read_watched_metric(metrics, s.p.metric, "the plateau schedule `$k`")
        step_plateau!(s, v) &&
            @info "ReactantNitro: `$k` reduced on plateau of `$(s.p.metric)` at epoch \
                   $(nitro.epoch), scale $old -> $(s.scale)."
    end
    return nothing
end

# Each plateau's scale on the validation line, as `schedule/plateau/<key>`.
function log_plateaus!(nitro)
    ps = plateaus(nitro.schedules)
    isempty(ps) && return nothing
    vals = (; (Symbol("schedule/plateau/", replace(string(k), "." => "/")) => s.scale for (k, s) in ps)...)
    log_metrics!(nitro.logger, vals; step = nitro.step, epoch = nitro.epoch, context = "validate")
    return nothing
end

# The counters the checkpoint record stores; `nothing` without any plateau.
function plateau_state(sched)
    ps = plateaus(sched)
    isempty(ps) && return nothing
    return (;
        (
            k => (; s.p.metric, s.p.mode, s.best, s.bad_epochs, s.cooldown_left, s.scale)
                for (k, s) in ps
        )...,
    )
end

"""
    ReactantNitro.restore_plateaus!(sched, state) -> nothing

Restore each plateau's counters from a resumed record's `schedule_state`, when the key is stored
and watches the same metric and mode. Any other plateau starts fresh, and a stored key the table no
longer has is ignored with an `@info`.
"""
function restore_plateaus!(sched, state)
    state === nothing && return nothing
    ps = Dict(plateaus(sched))
    for (k, st) in pairs(state)
        s = get(ps, k, nothing)
        if s === nothing
            @info "ReactantNitro: the resumed record holds plateau state for `$k`, which this \
                   run's schedules do not have; ignored."
        elseif st.metric === s.p.metric && st.mode === s.p.mode
            s.best, s.bad_epochs, s.cooldown_left, s.scale =
                st.best, st.bad_epochs, st.cooldown_left, st.scale
        else
            @info "ReactantNitro: plateau `$k` watched `$(st.metric)` ($(st.mode)) in the resumed \
                   record and now watches `$(s.p.metric)` ($(s.p.mode)), so it starts fresh."
        end
    end
    return nothing
end

# The binding report's note for a plateau-wrapped key; `""` for any other schedule.
function _plateau_desc(f)
    s = _unguard(f)
    s isa PlateauSchedule || return ""
    p = s.p
    return ", on plateau of $(p.metric) ($(p.mode)): x$(_g(p.factor)), patience $(p.patience)"
end
