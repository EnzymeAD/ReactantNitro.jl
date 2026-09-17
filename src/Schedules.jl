# Schedules.jl
#
# Schedule resolution, the effective learning rate, and the text of the binding report. The
# `binding_report(nitro)` accessor itself lives in Phases.jl with the rest of the `Nitro` accessors;
# this file builds what it returns.

"""
    ReactantNitro.resolve_schedules(e, given, total) -> resolved

Schedule resolution, run once at setup.

**Every entry is a factory of the horizon**, called once here with `total`, after which
`sched(step)` runs once per optimizer step. A bare `Number` normalizes to `_ -> (_ -> value)`,
which is what makes constants setup-fixed by construction: only scheduled entries transfer per step.

`given` is the `Nitro` keyword if one was passed and [`schedules`](@ref)`(e)` otherwise; **the
keyword replaces wholesale and does not merge**, because a silently retained schedule from the
accessor is much harder to explain than an explicitly dropped one.

**Key resolution is against two disjoint namespaces**, `Device` field names and rule field names,
and they route to different places: a scheduled rule field lands in the per-group `hp` carrier and
is applied to the chain, while a scheduled `Device` is written into the experiment by the per-step
rebuild and reaches traced code as `e.aux_weight`. There is no third destination, which is what
makes the resolution total.

`device` and `opt` are **reserved** at the top level and name the two namespaces. A key matching
neither is an error naming the fix; a key matching both is an ambiguity error naming both candidates
and showing the qualified form **for the user's own key** rather than a generic example. Collisions
are expected rather than exotic: `lambda` is the obvious name both for a decay coefficient and for a
loss weight, and `eta`, `rho`, and `epsilon` are all plausible model hyperparameters, so the
collision path is a two-line fix and **renaming the field is never required**.

**`step` is the optimizer step, not the micro-batch**, and
`total = max_epochs * div(steps_per_epoch, accum)` is an **exact** division, never `fld` or `cld`,
because a batch count not divisible by `accum` is a setup error.

**A nested `opt` key is a parameter-group path.** `opt = (; backbone = (; eta = ...))` flattens to
the dotted key `opt.backbone.eta`, which binds the `:backbone` group's chain only. The value is the
BASE curve for those groups and the per-group ratio is still applied,
`η_g(t) = opt.backbone.eta(t) · (learning_rate(e, Val(:backbone)) / learning_rate(e))`, exactly like
a bare key; a group with no path key falls back to the bare key, then to its base rate. Paths are
ONE level (`group.field`), because the group table is flat, and qualified-only: an unqualified
dotted key is a resolution error, since the `:auto` branch cannot know a top-level key is a group
path. Manual mode is deliberately asymmetric: its path values are absolute, because manual mode has
no base rate and no ratios.
"""
struct ResolvedSchedules{T <: NamedTuple, O <: NamedTuple}
    device::T                       # Device field name -> callable(step)
    opt::O                          # rule field name    -> callable(step)
    source::Dict{Symbol, Symbol}     # display key        -> :keyword | :accessor
    constant::Vector{Symbol}        # display keys that were bare Numbers
    horizon_dependent::Bool
end

Base.isempty(r::ResolvedSchedules) = isempty(r.device) && isempty(r.opt)

# A bare `Number` normalizes to `_ -> (_ -> value)`. That is what makes constants setup-fixed BY
# CONSTRUCTION rather than by a check: a constant is a `Number` and a schedule is a `Callable`, so
# the per-step transfer count equals the number of quantities actually being varied.
_as_factory(x::Number) = _ -> (_ -> x)
_as_factory(f) = f

"""
    ReactantNitro.HorizonGuard

A schedule asked for a step PAST its horizon that THROWS there holds its final value and warns once,
instead of ending the run. A schedule that answers past `total` (a step decay, a constant, a cycle) is
left alone: the guard evaluates it as usual and only steps in on the exception, so nothing that does
not depend on the horizon changes. The data contract is that `length(train)` is stable across
epochs, so `total` counts every step the run will take. A source whose batch count drifts between
epochs breaks that quietly.
"""
mutable struct HorizonGuard{F}
    f::F
    total::Int
    key::Symbol
    warned::Bool
end

# A horizon of zero or less means none is known (a manual loop with no `total`): nothing to guard.
guard_horizon(f, total::Integer, key) = total <= 0 ? f : HorizonGuard(f, Int(total), Symbol(key), false)

function (g::HorizonGuard)(step)
    step <= g.total && return g.f(step)
    try
        return g.f(step)
    catch err
        if !g.warned
            g.warned = true
            @warn """
            ReactantNitro: schedule `$(g.key)` threw at step $step, past its horizon `total = $(g.total)`
            ($(typeof(err))). The epoch length drifted from what setup read: a source whose batch
            count varies between epochs, such as a filter that depends on augmentation. Holding the
            schedule at its step-$(g.total) value for the rest of the run. Make the per-epoch count stable
            so `total` is exact.
            """ key = g.key step total = g.total
        end
        return g.f(g.total)
    end
end

# `device` and `opt` are reserved at the top level and name the two namespaces; every other
# top-level key is unqualified and must resolve to exactly one of them. The `opt` table may nest
# one level deeper: a NamedTuple VALUE is a parameter-group path, flattened here to a
# dotted display key (`Symbol("backbone.eta")`) so the per-group binding survives resolution.
function _flatten_schedules(nt::NamedTuple, what::AbstractString)
    out = Tuple{Symbol, Symbol, Any}[]
    for (k, v) in pairs(nt)
        if k === :device || k === :opt
            v isa NamedTuple || error(
                """
                ReactantNitro: `$k` is reserved at the top level of $what and names a namespace, so
                it must be a NamedTuple of schedules. Got a `$(typeof(v))`.
                If you meant a `Device` field literally named `$k`, write `$k = (; $k = ...)`."""
            )
            if k === :opt
                _flatten_opt!(out, v)
            else
                for (k2, v2) in pairs(v)
                    push!(out, (k, k2, v2))
                end
            end
        else
            push!(out, (:auto, k, v))
        end
    end
    return out
end

# Flatten the `opt` table into `(ns, key, spec)` triples: a bare key is a rule field name and a
# nested NamedTuple is a parameter-group PATH whose innermost keys are field names. The mirror of
# manual mode's `_flatten_manual_opt!`, recursing over the user's table rather than an opt_state
# tree; the dotted display key (`Symbol("backbone.eta")`) is the same shape either way, which is
# what keeps the path parsing (`_schedule_target`) shared between the two modes.
function _flatten_opt!(out, spec::NamedTuple, path::Tuple = ())
    for (k, v) in pairs(spec)
        if v isa NamedTuple
            _flatten_opt!(out, v, (path..., k))
        else
            push!(out, (:opt, isempty(path) ? k : Symbol(join((path..., k), ".")), v))
        end
    end
    return nothing
end

_display_key(ns, k) = ns === :auto ? k : Symbol(ns, ".", k)

"""
    ReactantNitro.schedulable_fields(chains) -> Tuple{Vararg{Symbol}}

The `opt` namespace: the union of every rule field in the composed chains, minus each rule's
[`nonschedulable`](@ref) opt-out. The namespace is `opt` rather than `rule` deliberately, because
`rule` is singular while a chain holds several and a key resolves against the union of their
fields.
"""
function schedulable_fields(chains)
    ks = Symbol[]
    for chain in chains, r in _rules_of(chain)
        T = typeof(r)
        ns = nonschedulable(T)
        for f in fieldnames(T)
            f in ns && continue
            f in ks || push!(ks, f)
        end
    end
    return Tuple(ks)
end

function resolve_schedules(
        e, given::NamedTuple, total::Integer;
        chains = (), accessor = nothing, groups = (:default,)
    )
    df = device_fields(typeof(e))
    of = schedulable_fields(chains)
    acc = accessor === nothing ? given : accessor
    from_accessor = Set(_display_key(ns, k) for (ns, k, _) in _flatten_schedules(acc, "`schedules(e)`"))

    tun, opt = Pair{Symbol, Any}[], Pair{Symbol, Any}[]
    source, constant = Dict{Symbol, Symbol}(), Symbol[]
    varying = false

    for (ns, k, spec) in _flatten_schedules(given, "`schedules`")
        display = _display_key(ns, k)
        target = ns
        if ns === :auto
            in_t, in_o = k in df, k in of
            if in_t && in_o
                error(
                    """
                    ReactantNitro: schedule key `$k` is ambiguous. It matches a `Device` field on
                    $(nameof(typeof(e))) AND a schedulable field of the optimizer chain, and the
                    two route to different places: an `opt` key is applied to the optimizer, a
                    `device` key is written into the experiment and reaches traced code as
                    `e.$k`.
                    Qualify it. To schedule the experiment's field:
                        schedules = (; device = (; $k = <factory>))
                    To schedule the optimizer's:
                        schedules = (; opt = (; $k = <factory>))
                    Renaming the field is never required; collisions on `lambda`, `eta`, `rho`, and
                    `epsilon` are expected rather than exotic."""
                )
            elseif in_t
                target = :device
            elseif in_o
                target = :opt
            else
                error(
                    """
                    ReactantNitro: schedule key `$k` matches nothing. A schedule key is either a
                    `Device` field name or a rule field name, resolved against the union of the
                    two.
                    `Device` fields on $(nameof(typeof(e))): $(isempty(df) ? "(none)" : df)
                    Schedulable optimizer fields: $(isempty(of) ? "(none; no chain was supplied)" : of)
                    Anything you want to vary inside `loss`, `forward`, or `metrics` must be a
                    `Device` field; anything on the optimizer must be a rule field. Note the
                    learning-rate key is `eta`, not `lr`."""
                )
            end
        elseif ns === :device
            k in df || error("ReactantNitro: `device.$k` names no `Device` field on \
                              $(nameof(typeof(e))). Its `Device` fields are \
                              $(isempty(df) ? "(none)" : df).")
        else
            # A DOTTED `opt` key is a parameter-group path (`group.field`), one level because
            # the group table is flat; it binds that group's chain only, and the field must be
            # schedulable THERE, not merely somewhere. A bare key binds every group, unchanged.
            path, field = _schedule_target(k)
            if isempty(path)
                k in of || error(
                    "ReactantNitro: `opt.$k` names no schedulable field of the optimizer \
                     chain. Schedulable fields are $(isempty(of) ? "(none)" : of); a \
                     parameter-sized or structural field is excluded by `nonschedulable`."
                )
            else
                length(path) == 1 || error(
                    """
                    ReactantNitro: `opt.$k` is a path of $(length(path) + 1) segments.
                    Automatic-loop paths are ONE level (`group.field`), because parameter groups
                    are flat. Bind the group's curve directly under `opt`."""
                )
                group = only(path)
                gi = findfirst(==(group), groups)
                gi === nothing && error(
                    """
                    ReactantNitro: `opt.$k` names group $(repr(group)), which is not a parameter
                    group of this experiment. Parameter groups are $(Tuple(groups))."""
                )
                (gi > length(chains) || isempty(chains)) && error(
                    """
                    ReactantNitro: `opt.$k` names group $(repr(group)), but no optimizer chain was
                    resolved for it. Path-bound `opt` keys are validated against the group's own
                    chain."""
                )
                gf = schedulable_fields((chains[gi],))
                field in gf || error(
                    """
                    ReactantNitro: `opt.$k` names no schedulable field of the $(repr(group)) group's
                    chain. Schedulable fields of that chain:
                    $(isempty(gf) ? "(none)" : gf)."""
                )
            end
        end

        if spec isa Number
            push!(constant, display)
        else
            varying = true
        end
        # The factory is called ONCE, here, with the horizon; `sched(step)` runs once per step, and
        # past the horizon it holds rather than throws (HorizonGuard).
        sched = _as_factory(spec)(total)
        spec isa Number || (sched = guard_horizon(sched, total, display))
        push!(target === :device ? tun : opt, k => sched)
        source[display] = display in from_accessor ? :accessor : :keyword
    end

    opt_nt = NamedTuple(opt)
    # A BARE `lambda` key conflicts against every group, since the schedule supplies the whole
    # table's decay base; a PATH `lambda` key names its group and conflicts per-group instead.
    haskey(opt_nt, :lambda) && check_lambda_conflict(e, groups)
    for k in keys(opt_nt)
        path, field = _schedule_target(k)
        (field === :lambda && !isempty(path)) || continue
        check_lambda_conflict(e, (only(path),), k)
    end
    return ResolvedSchedules(NamedTuple(tun), opt_nt, source, constant, varying)
end

"""
    ReactantNitro.check_lambda_conflict(e, groups, [key]) -> nothing

**Scheduling `lambda` while also defining per-group `lambda` accessors is a setup error naming
both**, since the schedule supplies the base. A BARE key (`opt.lambda`) applies the rule to every
`groups` entry; a PATH key (`opt.<group>.lambda`) names its group and applies it to that group only,
which is what lets one group's decay be scheduled while another's accessor keeps its own base.

`eta` is deliberately exempt from the same rule, and that asymmetry is the whole point: per-group
`learning_rate` accessors define **ratios** that compose with an `eta` schedule, whereas a scheduled
`lambda` is uniform across groups and gets its per-group scaling from the `η_g(t)` factor instead.
"""
function check_lambda_conflict(e, groups, key::Symbol = :lambda)
    offenders = [g for g in groups if !is_framework_default(lambda, Tuple{typeof(e), Val{g}})]
    isempty(offenders) && return nothing
    whole = key === :lambda
    scope = whole ? "for every group" : "for group $(repr(only(groups)))"
    uniform = whole ? " is uniform across groups and" : ""
    error(
        """
        ReactantNitro: `opt.$key` is scheduled AND per-group `lambda` accessors are defined for
        $(Tuple(offenders)), which is a setup error naming both. The schedule
        supplies the decay base $scope, so a per-group accessor has nothing left to say.
        Drop the accessors and keep the schedule, or drop the schedule and keep the accessors.
        Note `eta` is deliberately NOT symmetric with this: per-group `learning_rate` accessors
        define RATIOS that compose with an `eta` schedule, whereas a scheduled `lambda`$uniform
        takes its per-group scaling from the effective learning rate."""
    )
end

"""
    ReactantNitro.is_framework_default(f, argtypes) -> Bool

Whether the method that would be called is one of the framework's own defaults rather than a user
method. Used where a configuration is an error only if the user actually defined something, which
cannot be answered by calling the accessor: its default and a user method that happens to return
the same value are indistinguishable by value.
"""
is_framework_default(f, argtypes) =
    hasmethod(f, argtypes) && which(f, argtypes).module === @__MODULE__

"""
    ReactantNitro.effective_lr(e, group, eta_t) -> scalar

The definition of "effective learning rate" that the per-group `hp.eta` and `hp.lambda` refer to:

    η_g(t) = eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))

where `eta_sched(t)` is the `eta` schedule if one is configured and `learning_rate(e)` otherwise.
**Per-group accessors define ratios; the schedule sets the absolute value of the default group**, so
`η_default(t) == eta_sched(t)` exactly and a backbone at `1f-4` against a default of `1f-3` stays a
tenth of it for the whole run. Both obvious alternatives are worse: a multiplier makes
`OneCycle(total, 1f-3)` silently yield `1e-6`, and replacing all groups kills per-group ratios
whenever a schedule exists.

`lambda` follows `eta`'s per-group scaling and is the one other key with a per-group base:

    λ_g(t) = η_g(t) * lambda_sched(t)      # `opt.lambda` scheduled: uniform base
    λ_g(t) = η_g(t) * lambda(e, Val(g))    # otherwise: per-group base

**The `η_g(t)` factor is what makes decay per-group, so the scheduled `lambda` itself stays uniform.**
There is no ratio and no division, which matters because the default group's decay is legitimately
`0` in most experiments. This is forced rather than chosen: decoupled decay is *defined* as scaled
by the learning rate.

**Every other scheduled key sets its value absolutely for every group**, and configuring both a
schedule and a per-group accessor for the same quantity is a setup error naming both. Three rules,
`eta`, `lambda`, and everything else, all stated, all loud. Per-group schedules are out of scope.

The formula is complete without schedules: the schedule layer supplies `eta_t` from the resolved
`eta` schedule and changes nothing here. `eta_t === nothing` is "no `eta` schedule configured",
which means `eta_sched(t) = learning_rate(e)`.
"""
function effective_lr(e, group::Symbol, eta_t = nothing)
    base = learning_rate(e)
    iszero(base) && error(
        """
        ReactantNitro: `learning_rate(e)` is 0, which is a setup error. Per-group
        accessors define RATIOS against it, `learning_rate(e, Val(g)) / learning_rate(e)`, so a zero
        base makes every group's ratio undefined rather than making every group's rate zero. To train
        one group only, give the others a `lambda`-style zero through their own rate, or freeze them.
        This is also why the default is `1f-3` rather than `0`."""
    )
    sched = eta_t === nothing ? base : eta_t
    return sched * (learning_rate(e, Val(group)) / base)
end

"""
    ReactantNitro.step_aux(fields, scheds, step) -> NamedTuple

The per-step rebuild of the scheduled `Device` fields, in **one pass** rather than `@set` per key:

```julia
step_aux(fields, scheds, step) =
    merge(fields, NamedTuple{SCHED_KEYS}(map(k -> to_device(scheds[k](step)), SCHED_KEYS)))
```

Per-step cost is one small `NamedTuple` (mostly reference copies), one struct copy, and K device
conversions. Per-group values count once per group, so a scheduled learning rate across G groups is
G transfers; at G of two to four this is noise.

**Verified:** rebuilding an optimizer rule each step with a fresh `ConcretePJRTNumber` `eta`
re-enters the **same** compiled thunk, and the value is live: a flat eta sequence and a ramped one
produce different parameters through one compiled program.

**The scheduled value is coerced to the field's own type before conversion**, which is why this
takes `fields` rather than only the schedules. A schedule written as `t -> 0.25 * min(1, t / 2000)`
returns `Float64` for a `Device{Float32}` field, and the uncoerced value would produce a
`ConcretePJRTNumber{Float64}`, moving `typeof(compile_view(e))`, which is `args[1]` at every trace
site, and so **recompiling the gradient program on the first scheduled step**. That is a
several-hundred-second cost for a missing `f0` suffix, and the framework can simply do the
conversion.
"""
function step_aux(fields::NamedTuple, scheds::NamedTuple, step::Integer; mesh = nothing)
    ks = keys(scheds)
    vals = map(ks) do k
        to_device(
            _coerce_scheduled(getproperty(fields, k), getproperty(scheds, k)(step), k); mesh
        )
    end
    return merge(fields, NamedTuple{ks}(vals))
end
step_aux(fields::NamedTuple, r::ResolvedSchedules, step::Integer; mesh = nothing) =
    step_aux(fields, r.device, step; mesh)

# Match the field's own type. The two cases need two different questions asked, and the trap is that
# `eltype` answers only one of them: `eltype(ConcretePJRTArray{Float32,1,1})` is `Float32`, but
# `eltype(ConcretePJRTNumber{Float32,1})` is `ConcretePJRTNumber{Float32,1}`, i.e. Base's
# `eltype(::Type{T}) = T` fallback for a non-array. So the scalar case converts to `typeof(old)`,
# which is right whether the field is already device-resident (setup has run, the run path) or still
# a host `Number` (before conversion, and in unit tests), and `to_device` is idempotent over both.
_coerce_scheduled(old, new::Number, ::Symbol) = convert(typeof(old), new)
function _coerce_scheduled(old, new::AbstractArray, name::Symbol)
    size(new) == size(old) || error(
        """
        ReactantNitro: the schedule for `$name` returned an array of size $(size(new)) and the field
        holds $(size(old)). Size is NOT in the compile cache's key, so this would hit the existing
        program and fail inside XLA on the shape mismatch rather than recompiling."""
    )
    return convert(AbstractArray{eltype(old)}, new)
end
_coerce_scheduled(_, new, ::Symbol) = new

"""
    ReactantNitro.step_experiment(e, sched, step) -> e

The per-step rebuild on the **experiment** side: the same rebuild `rebuild_rules` does for the
optimizer's rules, applied to the scheduled [`Device`](@ref) fields of `e` itself. This is what makes
`device = (; aux_weight = ...)` in `schedules` actually vary `e.aux_weight` inside `loss`, `forward`,
and `metrics`.

**Returns `e` itself, identically, when nothing device-side is scheduled**, which is the overwhelmingly
common case and the reason the caller can invoke it unconditionally. Only a run that scheduled an
experiment field pays for the struct copy.

**The rebuild itself is [`step_aux`](@ref)**, which this wraps rather than reimplements: `step_aux`
produces the new field values in one pass, and this puts them back into the struct. Two copies of the
same merge would be two things to keep in agreement, and the one with the tests would not be the one
that runs.

`step` is the **upcoming** optimizer step, so callers pass `nitro.step + 1`, matching
[`rebuild_rules`](@ref) exactly: the two halves of one rebuild must not disagree about which step they
are building for.

**The type assertion is the backstop.** `step_aux`'s coercion is meant to make it unreachable, and it
is checked rather than trusted because the failure it prevents is a **silent recompile on every
optimizer step**, which presents as "training got mysteriously slower" rather than as an error.
"""
function step_experiment(e, sched, step::Integer; mesh = nothing)
    (sched === nothing || isempty(sched.device)) && return e
    T = typeof(e)
    ks = keys(sched.device)
    cur = NamedTuple{ks}(map(f -> getfield(e, f), ks))
    stepped = step_aux(cur, sched.device, step; mesh)
    e_new = Base.typename(T).wrapper(
        map(f -> haskey(stepped, f) ? getproperty(stepped, f) : getfield(e, f), fieldnames(T))...
    )
    typeof(compile_view(e_new)) === typeof(compile_view(e)) || error(
        """
        ReactantNitro: a `device` schedule moved `typeof(compile_view(e))`, which is `args[1]`'s type
        at every trace site and therefore part of the compile cache's key, so every optimizer step
        would recompile rather than reuse the program. The scheduled fields are $(join(ks, ", ")). A
        schedule must return the field's own type: check that a `Device{Float32}` field's schedule
        returns `Float32`, and that an array-valued one keeps its element type and its size."""
    )
    return e_new
end

"""
    ReactantNitro.to_device(x)

The device conversion, and it is **not** the obvious incantation. Measured: bare `to_rarray(2.0f0)`
passes the scalar through unchanged and the value **bakes**.

```julia
to_device(x::Number) = Reactant.to_rarray(x; track_numbers = Number)
to_device(x::AbstractArray) = Reactant.to_rarray(x)
```

Prefer a device *number* over a 1-element array: an array requires `[1]` to read, which throws
"Scalar indexing is disallowed" under trace.

The two identity methods keep it idempotent, so a value already on device passes through rather than
being handed back to `to_rarray` a second time. That matters because `to_device_rule` runs
over rules the framework has already built with device hyperparameters.

**A `Device` field may hold a CONTAINER of arrays, not only one value**, which is PyTorch's
"buffers": tensors the graph reads and nothing differentiates, such as the frozen weights of a
backbone a model calls through `Reactant.Ops.hlo_call` inside its `forward`. They belong on a
`Device` field rather than in `ps` (nothing trains them) and rather than on a `GraphConst` (baking
tens of millions of constants into the graph is a compile-time disaster). Because the experiment
reaches Enzyme as `Const(ev)`, a buffer set placed there is constant to the gradient, so Enzyme
differentiates only the head and keeps no tape for the backbone's forward.
The container is mapped element-wise, so each leaf takes one of the methods above and the identity
methods keep the whole thing idempotent.

`Tuple` and `NamedTuple` only, deliberately: `AbstractVector{<:AbstractArray}` is already claimed
by the `AbstractArray` method above, and adding it here would turn a wrong dispatch into a silent
one rather than an error.
"""
to_device(x::Number; mesh = nothing) =
    place_replicated(x, mesh; track_numbers = Number)
to_device(x::AbstractArray; mesh = nothing) = place_replicated(x, mesh)
to_device(x::Reactant.RNumber; mesh = nothing) = x
to_device(x::Reactant.AbstractConcreteArray; mesh = nothing) = x
to_device(x::Union{Tuple, NamedTuple}; mesh = nothing) = map(v -> to_device(v; mesh), x)

# ── Manual mode: path-bound `opt` schedules ─────────────────────────────────────────

"""
    ReactantNitro.resolve_manual_schedules(e, given, total; opt_state, accessor) -> ResolvedSchedules

Schedule resolution for a manual-mode experiment: the `device` keys resolve exactly as in the
automatic loop (via [`resolve_schedules`](@ref), which this delegates to), and the `opt` keys
resolve against the USER's `opt_state` structure instead of the framework's chains.

**The `opt` table's keys are binding descriptors, and the structure IS the binding:**

```julia
schedules(e) = (;
    opt = (;
        eta = total -> t -> 1.0f-3 * (1 - t / total),   # bare: EVERY rule's eta, absolute
        gen  = (; eta = total -> t -> 1.0f-4 * ...),     # path: opt_state.gen's rules' eta
        disc = (; eta = _ -> _ -> 2.0f-3),               # path: opt_state.disc's rules' eta
    ))
```

A bare key is a rule FIELD name and binds to every rule in `opt_state` that has that field (the
absolute value: manual mode has no base rate and no ratios). A nested key is a PATH into
`opt_state`; the innermost keys are field names. A path is validated at setup against the
`opt_state` tree, a field against the rules it reaches, and a `nonschedulable` field is refused,
all with loud errors. This is what makes "we don't know what eta to bind to" impossible: the key
says.

Only the `opt` namespace carries paths; unqualified keys resolve exactly as in the automatic
loop, against `Device` fields alone (manual mode has no chains for the opt side of the `:auto`
resolution).
"""
function resolve_manual_schedules(e, given::NamedTuple, total::Integer; opt_state, accessor = nothing)
    acc = accessor === nothing ? given : accessor
    opt_given = get(given, :opt, (;))
    opt_acc = get(acc, :opt, (;))
    flat, source, constant, varying = flatten_manual_opt(opt_given, opt_state, total, opt_acc)
    # The `opt` namespace is consumed above; `device` (and the `:auto` keys that resolve to it)
    # go through the ordinary machinery, which never sees the nested opt table.
    rest = (; (k => v for (k, v) in pairs(given) if k !== :opt)...)
    r = resolve_schedules(e, rest, total; chains = (), accessor = acc)
    return ResolvedSchedules(
        r.device, flat, merge(r.source, source), vcat(r.constant, constant),
        r.horizon_dependent || varying
    )
end

"""
    ReactantNitro.flatten_manual_opt(opt_table, opt_state, total, accessor_table) -> (flat, source, constant, varying)

Manual mode's opt-table resolution: flatten the user's (possibly nested) `opt` table into
display-keyed step functions, validating each binding against `opt_state`.

* a bare key is a rule field name and binds to every rule in `opt_state` with that field;
* a nested key is a path into `opt_state` and binds to the rules at that subtree.

The display key of a path binding is the dotted path plus the field, `Symbol("gen.eta")`, which is
what the per-step rebuild and the binding report read. `source`/`constant` mirror the automatic
resolution's bookkeeping.
"""
function flatten_manual_opt(opt_table::NamedTuple, opt_state, total::Integer, accessor_table = (;))
    out = Pair{Symbol, Any}[]
    constant = Symbol[]
    varying = Ref(false)
    acc_keys = Set(_manual_opt_keys(accessor_table))
    _flatten_manual_opt!(out, constant, varying, opt_table, opt_state, (), total)
    source = Dict{Symbol, Symbol}(
        k => (k in acc_keys ? :accessor : :keyword) for (k, _) in out
    )
    return (NamedTuple(out), source, constant, varying[])
end

# Walk one level of the (possibly nested) opt table. `target` is the opt_state subtree at `path`.
function _flatten_manual_opt!(out, constant, varying, spec::NamedTuple, target, path, total)
    for (k, v) in pairs(spec)
        if v isa NamedTuple
            target2 = _descend_opt_path(target, k, path)
            _flatten_manual_opt!(out, constant, varying, v, target2, (path..., k), total)
        else
            _check_manual_field(target, path, k)
            display = Symbol(join((path..., k), "."))
            sched = _as_factory(v)(total)
            v isa Number || (sched = guard_horizon(sched, total, display))
            push!(out, display => sched)
            if v isa Number
                push!(constant, display)
            else
                varying[] = true
            end
        end
    end
    return nothing
end

# The display keys of an opt table, leniently (no opt_state validation): source tracking only.
function _manual_opt_keys(spec::NamedTuple, path = ())
    ks = Symbol[]
    for (k, v) in pairs(spec)
        if v isa NamedTuple
            append!(ks, _manual_opt_keys(v, (path..., k)))
        else
            push!(ks, Symbol(join((path..., k), ".")))
        end
    end
    return ks
end

# Descend one path segment into the opt_state tree, loudly.
function _descend_opt_path(target, k::Symbol, path)
    target isa NamedTuple || error(
        """
        ReactantNitro: the `opt` schedule path `$(join((path..., k), "."))` walks off the optimizer
        state tree: `$(isempty(path) ? "the root" : join(path, "."))` is not a `NamedTuple`.
        `opt` schedule keys are paths into the `opt_state` `setup_optimizers` returned; the value
        at a path segment must be a nested `NamedTuple`."""
    )
    haskey(target, k) || error(
        """
        ReactantNitro: the `opt` schedule path `$(join((path..., k), "."))` does not exist in the
        optimizer state tree. Available at `$(isempty(path) ? "the root" : join(path, "."))`:
        $(keys(target))."""
    )
    return getproperty(target, k)
end

# The rules reachable from a target subtree, as a list of DISTINCT CORE rule types in
# first-appearance order, and an error naming the reach if there are none. The core is the
# innermost rule: `Optimisers.setup` wraps rules (with Reactant arrays) in a single-field
# wrapper that delegates `apply!`, and a schedulable field lives on the wrapped rule, so the
# check and the rebuild both look through wrappers.
function _manual_rules(x, seen = Type[], out = Type[])
    if x isa Optimisers.Leaf
        R = typeof(_rule_core(x.rule))
        R in seen || (push!(seen, R); push!(out, R))
    elseif x isa Union{Tuple, NamedTuple}
        for v in x
            _manual_rules(v, seen, out)
        end
    elseif x isa AbstractArray || x isa Number || x isa AbstractString || x isa Symbol
        nothing
    elseif isstructtype(typeof(x))
        for f in fieldnames(typeof(x))
            _manual_rules(getfield(x, f), seen, out)
        end
    end
    return out
end

# Unwrap single-field rule wrappers (the `ReactantOptimiser` of Optimisers' Reactant extension,
# which we do not name because it is not this package's dependency) to the innermost rule.
function _rule_core(r)
    T = typeof(r)
    fs = fieldnames(T)
    if length(fs) == 1 && getfield(r, fs[1]) isa Optimisers.AbstractRule
        return _rule_core(getfield(r, fs[1]))
    end
    return r
end

# A schedule key's field must be a schedulable scalar field of at least one CORE rule at the
# target, and a rule that HAS it must not exclude it via `nonschedulable`.
function _check_manual_field(target, path, field::Symbol)
    rules = _manual_rules(target)
    isempty(rules) && error(
        """
        ReactantNitro: the `opt` schedule key `$(Symbol(join((path..., field), ".")))` reaches no
        optimizer leaves at `$(isempty(path) ? "the root of opt_state" : join(path, "."))`.
        A path must end at a subtree or leaf holding `Optimisers.Leaf`es."""
    )
    for R in rules
        field in fieldnames(R) || continue
        field in nonschedulable(R) && error(
            """
            ReactantNitro: `opt.$(Symbol(join((path..., field), ".")))` schedules `$field` on
            $(nameof(R)), which `nonschedulable` excludes. A parameter-sized or structural rule
            field cannot be scheduled."""
        )
        fieldtype(R, field) <: Number || error(
            """
            ReactantNitro: `opt.$(Symbol(join((path..., field), ".")))` schedules `$field` on
            $(nameof(R)), whose field type is $(fieldtype(R, field)). Only scalar `Number` fields
            are schedulable."""
        )
    end
    any(f -> field in fieldnames(f), rules) || error(
        """
        ReactantNitro: `opt.$(Symbol(join((path..., field), ".")))` names no field of the rules at
        `$(isempty(path) ? "the root of opt_state" : join(path, "."))`:
        $(join(nameof.(rules), ", "))."""
    )
    return nothing
end

"""
    ReactantNitro.rebuild_scheduled_rules(nitro, step) -> opt_state

The per-step host rebuild, the manual-mode counterpart of the automatic loop's `rebuild_rules`:
evaluate every `opt` schedule at `step` and rebuild the rules it names in a fresh `opt_state` tree.

Called by `_train_manual!` before each closure invocation, and the closure's program re-enters the
same cache entry because the rebuild is TYPE-preserving: the scheduled value is coerced to the
field's own element type (the `resolve_hp` rule, applied to a third place) and the rule's type is
unchanged, so only the scalar VALUES move, exactly as the automatic loop's rebuilt rules.
"""
function rebuild_scheduled_rules(nitro::Nitro, step::Integer)
    sched = nitro.schedules
    (sched === nothing || isempty(sched.opt)) && return nitro.opt_state
    opt_state = nitro.opt_state
    for (k, f) in pairs(sched.opt)
        path, field = _schedule_target(k)
        opt_state = apply_schedule(opt_state, path, field, f(step))
    end
    return opt_state
end

# `:eta` -> ((), :eta); `Symbol("gen.eta")` -> ((:gen,), :eta). Rule field names cannot contain
# dots, so the last segment is always the field.
function _schedule_target(k::Symbol)
    parts = Symbol.(split(string(k), '.'))
    return (Tuple(parts[1:(end - 1)]), parts[end])
end

"""
    ReactantNitro.apply_schedule(opt_state, path, field, value) -> opt_state

Rebuild every rule at `opt_state`'s subtree `path` whose rule has `field`, setting it to `value`
(coerced to the field's element type and device-converted). Rules without the field are untouched:
the union semantics of the automatic loop's chain rebuild.
"""
function apply_schedule(x, path::Tuple, field::Symbol, value)
    if isempty(path)
        return map_leaves(l -> rebuild_leaf_rule(l, field, value), x)
    end
    k = first(path)
    x isa NamedTuple && haskey(x, k) || error(
        "ReactantNitro internal: the `opt` schedule path `$k` is not in the optimizer state tree."
    )
    rest = apply_schedule(getproperty(x, k), Base.tail(path), field, value)
    return merge(x, NamedTuple{(k,)}((rest,)))
end

# Map `f` over every Leaf in a plain tree of Leaves, rebuilding the NamedTuple structure.
map_leaves(f, x::Optimisers.Leaf) = f(x)
map_leaves(f, x::Union{Tuple, NamedTuple}) = map(v -> map_leaves(f, v), x)
map_leaves(f, x) = x

"""
    ReactantNitro.rebuild_leaf_rule(leaf, field, value) -> Leaf

One rule's rebuild: set `field` to `value` in a NEW rule of the same type, keeping the state. The
value is coerced to the field's own element type before device conversion, the `resolve_hp` rule
applied in its third place: a Float64 schedule into a Float32 field would otherwise produce a
`ConcretePJRTNumber{Float64}`, moving the rule's type, moving the closure program's argument types,
and recompiling on the first scheduled step.

Rules that do not have the field are untouched, and single-field rule wrappers (the
`ReactantOptimiser` of Optimisers' Reactant extension) and `OptimiserChain`s are unwrapped and
rewrapped, so a scheduled `eta` reaches the Adam inside either.
"""
function rebuild_leaf_rule(leaf::Optimisers.Leaf, field::Symbol, value)
    newrule = rebuild_rule_field(leaf.rule, field, value)
    newrule === leaf.rule && return leaf
    return Optimisers.Leaf(newrule, leaf.state, leaf.frozen)
end

# Set `field` to `value` in a new rule of the same type; recurse into chains and single-field
# wrappers; leave every other rule untouched. The direct-field case coerces the value to the
# field's own element type, then device-converts, exactly like the automatic loop's `resolve_hp`.
function rebuild_rule_field(r, field::Symbol, value)
    T = typeof(r)
    if hasfield(T, field)
        old = getfield(r, field)
        elt = old isa Reactant.RNumber ? typeof(Reactant.to_number(old)) :
            old isa Number ? typeof(old) :
            error(
                """
                ReactantNitro: the scheduled field `$field` on $(nameof(T)) holds a $(typeof(old)),
                which is not a number. Only scalar rule fields are schedulable."""
            )
        return T.name.wrapper(
            map(f -> f === field ? to_device(convert(elt, value)) : getfield(r, f), fieldnames(T))...
        )
    end
    if r isa Optimisers.OptimiserChain
        return Optimisers.OptimiserChain(map(o -> rebuild_rule_field(o, field, value), r.opts)...)
    end
    fs = fieldnames(T)
    return if length(fs) == 1 && getfield(r, fs[1]) isa Optimisers.AbstractRule
        inner = getfield(r, fs[1])
        rebuilt = rebuild_rule_field(inner, field, value)
        rebuilt === inner ? r : T.name.wrapper(rebuilt)
    else
        r
    end
end

"""
    ReactantNitro.build_binding_report(nitro) -> String

The binding report's text. **A diagnostic, not a check**: it computes nothing the run does not
already compute and it never fails. It exists because the rules that resolve a learning rate, a
schedule key, and a per-group accessor are individually simple and jointly hard to hold in your
head.

**One of three reports, and they do not overlap.** This one says where each configured value BOUND
and from which source. What the handle currently holds, including the seed, the batch size, the
split sizes and the preset, is `show(nitro)`. What has been redefined since the handle froze it is
[`fixed_config_report`](@ref).

Rendered through whatever table renderer is installed, so loading `PrettyTables` boxes it; the
copy stored on the handle and sent to the logger stays plain.

```
ReactantNitro: binding report for MyExp

  data
  split  resolved
  train  2944 of 3001 samples; 57 dropped by drop-last; prefetch: 16 workers, depth 1
  val    500 samples; final batch of 52 padded then sliced; prefetch: none; the eval path does not stream

  gradient clip
    global norm 1.0   (optimizer program only; changing it recompiles it)       [train! keyword]

  schedules
  binding        what                                                        source
  eta            optimizer field, all groups, per-group ratio applied         [train! keyword]
  opt.lambda     optimizer field, all groups, per-group ratio applied         [schedules(e)]
  device.lambda  Device field   e.lambda                                      [schedules(e)]

  parameter groups (G = 2)
  group      base eta  ratio  decay toward  lambda  rule   params
  :default   1.0e-3    1.0    zero          1.0e-4  RAdam  1,234,567
  :backbone  1.0e-4    0.1    w0            1.0e-3  RAdam  23,456,789

  level 2 chains
    :backbone   scheduled values present in the returned chain: eta, beta
                NOT present: epsilon        <- scheduled but the factory did not apply it
```

That last block is the point: the framework gives up **erroring** on an unapplied Level 2
hyperparameter, because the identity check that would detect it also rejects a factory that
legitimately transforms a value. As a report the same information costs nothing and rejects nothing.

The `data` block is the home of the one thing the framework deliberately does not raise on: samples
a drop-last training loader discarded are invisible to every check the framework can make. The
parenthetical appears only when the source supports `MLUtils.numobs`, and is **omitted rather than
guessed** otherwise. The `gradient_clip_norm` line's source label is the one place a reader sees
whether the keyword, a method, or a field won.

Emitted through `@info` and handed to `log_other!(lgr, "binding_report", str)` so it lands in the
run's record.
"""
function build_binding_report(nitro; plain::Bool = true)
    lay = nitro.layout
    splits = [
        (;
            name = String(nm), batches = length(getproperty(nitro.data, nm)),
            batch_size = something(nitro.batch_size, 0),
            samples = nothing, dropped = nothing, short_final = nothing,
            prefetch = prefetch_report_entry(nm, getproperty(nitro.data, nm)),
        )
            for nm in keys(nitro.data)
    ]
    if get(nitro.frozen, :manual, false)
        # In manual mode the optimizers are the USER's own, from `setup_optimizers`; the automatic
        # per-group accessor resolution does not apply, and reporting it would claim a chain
        # (the framework's RAdam default, say) the run does not have. One honest line instead;
        # the rules themselves live in the user's `setup_optimizers`.
        groups = [
            (;
                name = :manual, base_eta = 0, ratio = 0, anchor = :user, lambda = 0,
                rule = "user (setup_optimizers)", params = sum(lay.lengths),
            ),
        ]
    else
        base = learning_rate(nitro.e)
        groups = [
            (;
                name = g,
                base_eta = learning_rate(nitro.e, Val(g)),
                ratio = learning_rate(nitro.e, Val(g)) / base,
                anchor = decay_anchor(nitro.e, Val(g)) === :zero ? :zero :
                    decay_anchor(nitro.e, Val(g)) === :w0 ? :w0 : :array,
                lambda = lambda(nitro.e, Val(g)),
                rule = nameof(optimizer(nitro.e, Val(g))),
                params = lay.lengths[gi],
            )
                for (gi, g) in enumerate(lay.groups)
        ]
    end
    return binding_report_text(;
        name = string(nameof(typeof(nitro.e))), splits,
        clip = nitro.gradient_clip_norm,
        clip_source = clip_source(nitro.e, nitro.gradient_clip_norm),
        schedules = nitro.schedules, groups,
        manual = get(nitro.frozen, :manual, false), plain
    )
end

"""
    ReactantNitro.clip_source(e, resolved) -> Symbol

Which route supplied `gradient_clip_norm`: `:keyword` if the value this run resolved to is not the
one the accessor chain would have produced, `:method` if the user defined an accessor, `:field` if
the experiment declares a field of that name, and `:default` for the framework's own `0f0`. The
binding report is the one place a reader sees whether the keyword, a method, or a field won.

**The framework default is its own label.** Both branches of the accessor test used to
answer `:field`, so a bare experiment declaring no such field was reported as taking one, which is
the report naming a source that does not exist. A defaults audit is exactly where that surfaces:
every run of a bare experiment prints this line.

**The keyword is detected by comparing values rather than by a flag**, because `Nitro`'s keyword
defaults to the accessor call, and a keyword that was passed is therefore indistinguishable from
one that was not by the time the constructor body runs. A keyword equal to what the accessor would
have returned reports the accessor, which is true of the value even though it understates the call.
"""
clip_source(e, resolved) =
    resolved != gradient_clip_norm(e) ? :keyword :
    !is_framework_default(gradient_clip_norm, Tuple{typeof(e)}) ? :method :
    hasfield(typeof(e), :gradient_clip_norm) ? :field : :default

_commas(n::Integer) = replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+$)" => ",")
_g(x) = string(round(Float64(x); sigdigits = 3))

"""
    ReactantNitro.prefetch_report_entry(name, split) -> (; device_batches, host_batches, workers, ordered, path)

The per-split prefetch fact, resolved rather than requested (see `prefetch_config`).

Every split reports its own resolved configuration now that the eval path streams too. It did not
always: while `run_eval` iterated its split directly, an eval entry reported `path = :eval`, because a
worker count for a path that never entered `batch_stream` would have been false.
"""
prefetch_report_entry(::Symbol, split) = prefetch_config(split)

# The rule for the data block: state what is known and omit what is not. `path` is the resolved
# one, so `:single_no_trait` reads differently from `:single` on purpose: the first is a capability
# the source does not have and the second is a number somebody chose.
# The two buffer counts read as "on the device / on the host", which is what they are: one batch
# staged past the transfer and `host_batches` allowed to exist before it. `ordered` appears only
# when it is FALSE, because that is the setting that costs bitwise reproducibility and a line that
# says so on every ordinary run would be noise rather than a warning.
_prefetch_buffers(pf) =
    "$(pf.device_batches) on device, $(pf.host_batches) on host" *
    (pf.ordered ? "" : ", UNORDERED")

_prefetch_note(pf) =
    pf.path === :fanout || pf.path === :fanout_unordered ?
    "prefetch: $(pf.workers) workers, $(_prefetch_buffers(pf))" :
    pf.path === :single ? "prefetch: 1 producer, $(_prefetch_buffers(pf))" :
    pf.path === :materialized ?
    "prefetch: 1 producer, $(_prefetch_buffers(pf)); source is a materialized `Vector`" :
    pf.path === :single_no_trait ?
    "prefetch: 1 producer, $(_prefetch_buffers(pf)); source is NOT index-addressable" :
    pf.path === :inline ? "prefetch: none (NoPrefetch)" :
    pf.path === :eval ? "prefetch: none; the eval path does not stream" :
    "prefetch: $(pf.path)"

"""
    ReactantNitro.binding_report_text(; name, splits, clip, clip_source,
                                        schedules, groups, level2 = ()) -> String

The binding report's text, built from explicit pieces so it is testable without a run.
`build_binding_report` assembles these from a [`Nitro`](@ref).

**It reports where values BOUND, and deliberately nothing else.** The seed, the accum, the schedule
horizon, the batch size, the batch counts and the preset are all state the handle carries, so
`show(nitro)` is where they are read; a fact printed by two reports is a fact that can disagree
between them. What is redefined since construction is the third report,
[`fixed_config_report`](@ref).

  * `splits`: one `(; name, batches, batch_size, samples, dropped, short_final[, prefetch])` per
    split. Only what the data path RESOLVED is printed: `samples`, `dropped`, and `short_final` may
    be `nothing` and are then **omitted rather than guessed**, per the rule above for sources that
    do not support `MLUtils.numobs`. `batches` and `batch_size` are accepted and not printed, since
    the handle carries both. `prefetch` is optional and read with `get`, so a caller assembling the
    pieces by hand may leave it out; `build_binding_report` always supplies it, from
    [`prefetch_report_entry`](@ref).
  * `clip_source`: `:keyword`, `:method`, `:field`, or `:default`. **This is the one place a reader
    sees which of them won.** `:default` is the framework's own value and is what a bare
    experiment reports; it is a separate label because "field on e" naming a field the experiment
    does not declare is a report that cannot be checked against the source.
  * `groups`: one `(; name, base_eta, ratio, anchor, lambda, rule, params)` per parameter group.
  * `level2`: one `(; group, present, absent)` per Level 2 chain. The framework gives up
    **erroring** on an unapplied Level 2 hyperparameter, because the identity check that would
    detect it also rejects a factory that legitimately transforms a value. As a report the same
    information costs nothing and rejects nothing.
"""
function binding_report_text(;
        name, splits = (), clip = 0,
        clip_source::Symbol = :default, schedules = nothing, groups = (),
        level2 = (), manual = false, plain::Bool = true
    )
    io = IOBuffer()
    # `plain = true` BY DEFAULT, and that default is load-bearing rather than conservative. This
    # text is stored on the handle, returned by `binding_report`, and sent to `log_other!`: a
    # machine-read artifact whose bytes must not depend on whether some other package in the
    # session happened to load PrettyTables. Setup asks for the rendered version separately, for
    # the human reading the `@info` at that moment, and keeps the plain one for everything else.
    section(title, header, rows) = sprint() do buf
        plain ? _render_table_plain(buf, title, header, rows, nothing) :
            _render_table(buf, title, header, rows)
    end
    # WHERE VALUES BOUND, and nothing else. The seed, the accum, the horizon and the preset used to
    # head this report and are all on the handle, so `show(nitro)` says them; a value repeated in
    # two reports is a value that can disagree between them. A preset was never a per-value source
    # anyway: by the time anything sees `e`, a preset's values ARE struct fields, indistinguishable
    # from hand-set ones, so claiming provenance for them would be a lie the type system cannot
    # back.
    println(io, "ReactantNitro: binding report for $name")

    if !isempty(splits)
        rows = TableRows()
        for sp in splits
            notes = String[]
            if sp.samples !== nothing
                notes = [
                    sp.dropped !== nothing && sp.dropped > 0 ?
                        "$(_commas(sp.samples - sp.dropped)) of $(_commas(sp.samples)) samples; $(sp.dropped) dropped by drop-last" :
                        "$(_commas(sp.samples)) samples",
                ]
            end
            sp.short_final === nothing ||
                push!(notes, "final batch of $(sp.short_final) padded then sliced")
            # `get` rather than `sp.prefetch`: this function is built to be callable from a test with
            # explicit pieces, and the suites that do so construct their splits tuples by hand.
            pf = get(sp, :prefetch, nothing)
            pf === nothing || push!(notes, _prefetch_note(pf))
            # No batch count and no batch size: `show(nitro)` carries both, and what belongs here
            # is what the DATA PATH resolved to, which is drop-last, padding, and prefetch.
            push!(rows, [sp.name, isempty(notes) ? "(nothing resolved)" : join(notes, "; ")])
        end
        print(io, "\n", section("  data", ["split", "resolved"], rows), "\n")
    end

    # A table, like every other section of this report, rather than the two hand-printed lines this
    # used to be. The three pieces were always there (the resolved value, what it implies, and which
    # source won); a table is what lines them up with the columns the rest of the report already
    # uses, and it is the column a reader scans that makes `[train! keyword]` versus
    # `[framework default]` findable in the same place here as in the schedules block.
    clip_rows = TableRows(
        [
            [
                clip > 0 ? "global norm $(_g(clip))" : "none (threshold 0)",
                clip > 0 ? "optimizer program only; changing it recompiles it" :
                    "a run with no clipping",
                "[" * _source_label(clip_source) * "]",
            ],
        ]
    )
    print(io, "\n", section("  gradient clip", ["value", "effect", "source"], clip_rows), "\n")

    if schedules !== nothing && !isempty(schedules)
        rows = TableRows()
        for (ns, tbl) in ((:opt, schedules.opt), (:device, schedules.device))
            for k in keys(tbl)
                disp = haskey(schedules.source, k) ? k : Symbol(ns, ".", k)
                what = ns === :opt ?
                    (manual ? _manual_opt_desc(k) : _auto_opt_desc(k)) :
                    "Device field   e.$k"
                note = disp in schedules.constant ? "  (constant)" : ""
                # The source keeps its brackets. It is the column a reader scans for, and "[train!
                # keyword]" is the token that appears in every other report and in the docs.
                push!(
                    rows, [
                        string(disp), what * note,
                        "[" * _source_label(get(schedules.source, disp, :accessor)) * "]",
                    ]
                )
            end
        end
        print(io, "\n", section("  schedules", ["binding", "what", "source"], rows), "\n")
    end

    if !isempty(groups)
        rows = TableRows[]
        rows = TableRows(
            [
                [
                    repr(g.name), _g(g.base_eta),
                    string(round(Float64(g.ratio); digits = 2)), string(g.anchor),
                    _g(g.lambda), string(g.rule), _commas(g.params),
                ] for g in groups
            ]
        )
        print(
            io, "\n",
            section(
                "  parameter groups (G = " * string(length(groups)) * ")",
                ["group", "base eta", "ratio", "decay toward", "lambda", "rule", "params"],
                rows
            ), "\n"
        )
    end

    if !isempty(level2)
        println(io, "\n  level 2 chains")
        for l in level2
            println(
                io, "    ", rpad(repr(l.group), 12),
                "scheduled values present in the returned chain: ",
                isempty(l.present) ? "(none)" : join(l.present, ", ")
            )
            isempty(l.absent) || println(
                io, "    ", " "^12, "NOT present: ",
                join(l.absent, ", "),
                "        <- scheduled but the factory did not apply it"
            )
        end
    end
    return String(take!(io))
end

_source_label(s::Symbol) = s === :keyword ? "train! keyword" :
    s === :accessor ? "schedules(e)" :
    s === :method ? "method on the experiment type" :
    s === :field ? "field on e" :
    s === :default ? "framework default" : string(s)

# Automatic-loop opt schedule keys: a bare key is a rule field name binding every group's
# chain; a dotted key is `group.field`, binding that group's chain only, with the per-group ratio
# still applied. The report says which a key is and which group a path binds.
function _auto_opt_desc(k::Symbol)
    parts = split(string(k), '.')
    field = Symbol(last(parts))
    scope = length(parts) == 1 ? "all groups" : "group :$(first(parts))"
    return field === :eta ? "optimizer field, $scope, per-group ratio applied" :
        field === :lambda ? "optimizer field, $scope, times eta per group" :
        "optimizer field, $scope, absolute"
end

# Manual mode: the opt schedule keys are binding descriptors (a bare field name, or a dotted
# path into the user's opt_state). The report says what the key binds to.
function _manual_opt_desc(k::Symbol)
    parts = split(string(k), '.')
    length(parts) == 1 && return "optimizer field, every rule, absolute"
    return "optimizer field at opt_state path $(join(parts[1:(end - 1)], "."))"
end
