# Schedules.jl
#
# Schedule resolution, the effective learning rate, and the binding report. The report's sections
# are appended to `show(nitro)` in Phases.jl and its text goes to the run's logger.

"""
    ReactantNitro.resolve_schedules(e, given, total) -> resolved

Schedule resolution, run once at setup. Every entry is a factory of the horizon, called once here
with `total`, after which `sched(step)` runs once per optimizer step; a bare `Number` normalizes to
`_ -> (_ -> value)`, so constants are setup-fixed by construction. `given` is the `Nitro` keyword
if passed and [`schedules`](@ref)`(e)` otherwise, replacing wholesale rather than merging.

Keys resolve against two disjoint namespaces, `Device` field names and rule field names, reserved
at the top level as `device` and `opt`. A scheduled rule field lands in the per-group `hp` carrier;
a scheduled `Device` is written into the experiment by the per-step rebuild and reaches traced code
as `e.aux_weight`. A key matching neither is an error naming the fix; one matching both is an
ambiguity error showing the qualified form, since `lambda`, `eta` and `epsilon` collide routinely.

`step` is the optimizer step, and `total = max_epochs * div(steps_per_epoch, accum)` is exact. A
nested `opt` key is a parameter-group path: `opt = (; backbone = (; eta = ...))` binds the
`:backbone` group's chain only, as the base curve with the per-group ratio still applied; paths are
one level, since the group table is flat. Manual mode's path values are absolute instead, since it
has no base rate.
"""
struct ResolvedSchedules{T <: NamedTuple, O <: NamedTuple}
    device::T                       # Device field name -> callable(step)
    opt::O                          # rule field name    -> callable(step)
    source::Dict{Symbol, Symbol}     # display key        -> :keyword | :accessor
    constant::Vector{Symbol}        # display keys that were bare Numbers
    horizon_dependent::Bool
end

Base.isempty(r::ResolvedSchedules) = isempty(r.device) && isempty(r.opt)

# A constant is a `Number` and a schedule is a `Callable`, so the per-step transfer count equals the
# number of quantities actually being varied.
_as_factory(x::Number) = _ -> (_ -> x)
_as_factory(f) = f

"""
    ReactantNitro.HorizonGuard

A schedule that throws when asked for a step past its horizon holds its final value and warns once,
instead of ending the run. One that answers past `total` is left alone. The horizon is exact only
while `length(train)` is stable across epochs, which the data contract requires.
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

# `device` and `opt` are reserved at the top level; every other key is unqualified and must resolve
# to exactly one namespace. A NamedTuple value under `opt` is a parameter-group path, flattened to a
# dotted display key (`Symbol("backbone.eta")`).
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

# Flatten the `opt` table into `(ns, key, spec)` triples: a bare key is a rule field, a nested
# NamedTuple a parameter-group path. The dotted display key is the same shape as manual mode's, so
# `_schedule_target` is shared.
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
[`nonschedulable`](@ref) opt-out.
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
            # A dotted `opt` key is a one-level `group.field` path binding that group's chain only;
            # the field must be schedulable there.
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

Scheduling `lambda` while also defining per-group `lambda` accessors is a setup error, since the
schedule supplies the base. A bare key applies to every group; a path key to its group only. `eta`
is exempt because per-group `learning_rate` accessors define ratios that compose with an `eta`
schedule, while a scheduled `lambda` gets its per-group scaling from `η_g(t)`.
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

Whether the method that would be called is the framework's own default rather than a user method.
Calling the accessor cannot answer this, since a user method may return the same value.
"""
is_framework_default(f, argtypes) =
    hasmethod(f, argtypes) && which(f, argtypes).module === @__MODULE__

"""
    ReactantNitro.effective_lr(e, group, eta_t) -> scalar

The effective learning rate the per-group `hp.eta` and `hp.lambda` refer to:

    η_g(t) = eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))

Per-group accessors define ratios and the schedule sets the default group's absolute value, so a
backbone at `1f-4` against a default of `1f-3` stays a tenth of it for the whole run. `lambda`
follows the same per-group scaling, with a uniform base from the schedule or a per-group one from
the accessor:

    λ_g(t) = η_g(t) * lambda_sched(t)      # `opt.lambda` scheduled
    λ_g(t) = η_g(t) * lambda(e, Val(g))    # otherwise

Decoupled decay is defined as scaled by the learning rate, so there is no ratio and no division,
which matters because the default group's decay is legitimately `0`. Every other scheduled key is
absolute for every group. `eta_t === nothing` means no `eta` schedule is configured.
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

The per-step rebuild of the scheduled `Device` fields, in one pass: one small `NamedTuple`, one
struct copy, and K device conversions. The scheduled value is coerced to the field's own type
before conversion, which is why this takes `fields`: a schedule written as
`t -> 0.25 * min(1, t / 2000)` returns `Float64` for a `Device{Float32}` field, and the uncoerced
value would move `typeof(compile_view(e))` and recompile the gradient program on the first
scheduled step.
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

# `eltype(ConcretePJRTNumber{Float32,1})` is the type itself (Base's non-array fallback), so the
# scalar case converts to `typeof(old)`, which is right whether the field is already on device or
# still a host `Number`; `to_device` is idempotent over both.
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

The per-step rebuild on the experiment side, the counterpart of `rebuild_rules`: the scheduled
[`Device`](@ref) fields of `e` are rebuilt through [`step_aux`](@ref), which is what makes
`device = (; aux_weight = ...)` vary `e.aux_weight` inside `loss` and `forward`. Returns `e` itself
when nothing device-side is scheduled. `step` is the upcoming optimizer step, `nitro.step + 1`,
matching `rebuild_rules`. The type assertion is a backstop against a silent recompile on every
step, which presents as training getting mysteriously slower.
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

The device conversion. Not the obvious incantation: bare `to_rarray(2.0f0)` passes a scalar
through unchanged and the value bakes, so a `Number` needs `track_numbers = Number`. A device
number is preferred over a 1-element array, whose `[1]` throws under trace. The identity methods
keep it idempotent, since `to_device_rule` runs over rules already built with device scalars.

A `Device` field may hold a `Tuple` or `NamedTuple` of arrays, PyTorch's "buffers": tensors the
graph reads and nothing differentiates, such as a frozen backbone's weights. They belong on a
`Device` field rather than in `ps` (nothing trains them) or a `GraphConst` (baking millions of
constants into the graph). Reaching Enzyme as `Const(ev)`, they are constant to the gradient.
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

Schedule resolution for a manual-mode experiment: `device` keys resolve as in the automatic loop,
and `opt` keys resolve against the user's `opt_state` structure. A bare key is a rule field name
and binds every rule with that field, absolutely; a nested key is a path into `opt_state`, with
the innermost keys field names. Paths and fields are validated at setup.

```julia
schedules(e) = (;
    opt = (;
        eta = total -> t -> 1.0f-3 * (1 - t / total),   # bare: EVERY rule's eta, absolute
        gen  = (; eta = total -> t -> 1.0f-4 * ...),     # path: opt_state.gen's rules' eta
        disc = (; eta = _ -> _ -> 2.0f-3),               # path: opt_state.disc's rules' eta
    ))
```
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

Manual mode's opt-table resolution: flatten the possibly nested `opt` table into display-keyed step
functions, validating each binding against `opt_state`. The display key of a path binding is the
dotted path plus the field, `Symbol("gen.eta")`.
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

# The distinct core rule types reachable from a subtree. `Optimisers.setup` wraps rules in a
# single-field wrapper that delegates `apply!`, and a schedulable field lives on the wrapped rule.
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

The per-step host rebuild, manual mode's counterpart of `rebuild_rules`: evaluate every `opt`
schedule at `step` and rebuild the rules it names. Type-preserving, since the value is coerced to
the field's own element type, so the closure's program re-enters the same cache entry.
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

One rule's rebuild: set `field` to `value` in a new rule of the same type, keeping the state, with
the value coerced to the field's element type so the rule's type does not move. Chains and
single-field rule wrappers are unwrapped and rewrapped, so a scheduled `eta` reaches the Adam
inside either.
"""
function rebuild_leaf_rule(leaf::Optimisers.Leaf, field::Symbol, value)
    newrule = rebuild_rule_field(leaf.rule, field, value)
    newrule === leaf.rule && return leaf
    return Optimisers.Leaf(newrule, leaf.state, leaf.frozen)
end

# Set `field` in a new rule of the same type; recurse into chains and single-field wrappers; leave
# every other rule untouched.
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
    ReactantNitro.binding_report_pieces(nitro) -> NamedTuple

Everything the binding report says, read off a [`Nitro`](@ref) and handed to
[`binding_report_sections`](@ref) or [`binding_report_text`](@ref). A diagnostic, not a check: it
computes nothing the run does not already compute and never fails. It says where each configured
value bound and from which source; what the handle holds is the `state` band of `show(nitro)`, and
what has been redefined since is [`fixed_config_report`](@ref).

```
  data
  split  resolved
  train  46 batches; 2,944 of 3,001 samples; 57 dropped by drop-last; prefetch: 16 workers
  val    8 batches; 500 samples; final batch of 52 padded then sliced; prefetch: 1 producer

  bindings
  binding        what                                                   source
  gradient clip  global norm 1.0; optimizer program only, and changing  [train! keyword]
  eta            optimizer field, all groups, per-group ratio applied    [train! keyword]
  device.lambda  Device field   e.lambda                                 [schedules(e)]

  parameter groups (G = 2)
  group      settings                                              params
  :default   eta 0.001 (x1.0), lambda 0.0001 toward zero, RAdam    1,234,567
  :backbone  eta 0.0001 (x0.1), lambda 0.001 toward w0, RAdam      23,456,789

  level 2 chains
  group      binding                                      values
  :backbone  present in the returned chain                eta, beta
             NOT present, the factory did not apply it    epsilon
```

The `level 2` block is where the framework reports rather than errors on an unapplied
hyperparameter, since the identity check that would detect it also rejects a factory that
legitimately transforms a value. The `data` block reports samples a drop-last loader discarded,
which no check can raise on, and only when the source supports `MLUtils.numobs`.
"""
function binding_report_pieces(nitro)
    lay = nitro.layout
    splits = [
        (;
            name = String(nm), batches = _nitro_split(getproperty(nitro.data, nm)),
            batch_size = something(nitro.batch_size, 0),
            samples = nothing, dropped = nothing, short_final = nothing,
            prefetch = prefetch_report_entry(nm, getproperty(nitro.data, nm)),
        )
            for nm in keys(nitro.data)
    ]
    if get(nitro.frozen, :manual, false)
        # Manual mode's optimizers are the user's own, so the per-group resolution does not apply.
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
    return (;
        name = string(nameof(typeof(nitro.e))), splits,
        clip = nitro.gradient_clip_norm,
        clip_source = clip_source(nitro.e, nitro.gradient_clip_norm),
        schedules = nitro.schedules, groups,
        manual = get(nitro.frozen, :manual, false),
    )
end

"""
    ReactantNitro.build_binding_report(nitro) -> String

The binding report's text, for the run's logger.
"""
build_binding_report(nitro) = binding_report_text(; binding_report_pieces(nitro)...)

"""
    ReactantNitro.build_binding_sections(nitro) -> Vector{TableSection}

The binding report's [`TableSection`](@ref)s, which `show(nitro)` appends to the handle's own
bands so that a run displays as one table.
"""
build_binding_sections(nitro) = binding_report_sections(; binding_report_pieces(nitro)...)

"""
    ReactantNitro.clip_source(e, resolved) -> Symbol

Which route supplied `gradient_clip_norm`: `:keyword` if the resolved value differs from what the
accessor chain would produce, `:method` for a user accessor, `:field` for a declared field, and
`:default` for the framework's own `0f0`. The keyword is detected by value, since a passed keyword
is indistinguishable from the accessor default by the time the constructor body runs.
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
"""
prefetch_report_entry(::Symbol, split) = prefetch_config(split)

# The data block states what is known and omits what is not. `ordered` appears only when false,
# since that is the setting that costs bitwise reproducibility.
_prefetch_buffers(pf) =
    "$(pf.device_batches) on device, $(pf.host_batches) on host" *
    (pf.ordered ? "" : ", UNORDERED")

"""
    ReactantNitro._batches_note(batches) -> String

The data band's leading note: a count, or the word [`_nitro_split`](@ref) reports for a loader
with no length.
"""
_batches_note(b) = b in ("streaming", "?") ? String(b) : "$b batches"

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
    ReactantNitro.binding_report_sections(; name, splits, clip, clip_source,
                                            schedules, groups, level2 = ()) -> Vector{TableSection}

The binding report's [`TableSection`](@ref)s, built from explicit pieces so they are testable
without a run. They report where values bound and nothing else; the seed, accum, horizon, batch
size and preset are the `state` band's, since a fact printed twice can disagree.

  * `splits`: one `(; name, batches, batch_size, samples, dropped, short_final[, prefetch])` per
    split. `samples`, `dropped` and `short_final` may be `nothing` and are then omitted rather than
    guessed; `prefetch` is optional; `batch_size` is accepted and not printed.
  * `clip_source`: `:keyword`, `:method`, `:field` or `:default`, the one place a reader sees which
    won.
  * `groups`: one `(; name, base_eta, ratio, anchor, lambda, rule, params)` per parameter group.
  * `level2`: one `(; group, present, absent)` per Level 2 chain.

`name` is accepted and not printed, since the table's title names the experiment.
"""
function binding_report_sections(;
        name = "", splits = (), clip = 0,
        clip_source::Symbol = :default, schedules = nothing, groups = (),
        level2 = (), manual = false
    )
    sections = TableSection[]

    if !isempty(splits)
        rows = TableRows()
        styles = CellStyles()
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
            # `get`, so a test can build its splits tuples by hand.
            pf = get(sp, :prefetch, nothing)
            pf === nothing || push!(notes, _prefetch_note(pf))
            # The batch count leads, as the first thing anyone asks of a split; a note rather than a
            # column, since every section shares its columns and a count would sit in one sized by
            # the settings text.
            pushfirst!(notes, _batches_note(sp.batches))
            push!(rows, [sp.name, join(notes, "; ")])
            # UNORDERED is the one resolved data setting worth a colour: it costs reproducibility.
            occursin("UNORDERED", last(rows)[2]) &&
                (styles[(length(rows), 2)] = :warn)
        end
        push!(sections, TableSection("data", ["split", "resolved"], rows, styles))
    end

    # One band for everything that has a source. The clip leads and is always present, where a
    # schedule may not be, and it answers the same three questions a binding does.
    rows = TableRows()
    srcs = Symbol[clip_source]
    push!(
        rows, [
            "gradient clip",
            clip > 0 ?
                "global norm $(_g(clip)); optimizer program only, and changing it recompiles it" :
                "none, threshold 0; a run with no clipping",
            "[" * _source_label(clip_source) * "]",
        ]
    )

    if schedules !== nothing && !isempty(schedules)
        for (ns, tbl) in ((:opt, schedules.opt), (:device, schedules.device))
            for k in keys(tbl)
                disp = haskey(schedules.source, k) ? k : Symbol(ns, ".", k)
                what = ns === :opt ?
                    (manual ? _manual_opt_desc(k) : _auto_opt_desc(k)) :
                    "Device field   e.$k"
                note = disp in schedules.constant ? "  (constant)" : ""
                # The source keeps its brackets; `[train! keyword]` is the token used everywhere.
                src = get(schedules.source, disp, :accessor)
                push!(rows, [string(disp), what * note, "[" * _source_label(src) * "]"])
                push!(srcs, src)
            end
        end
    end
    # A threshold of zero is the absence of a setting, so its cell is muted.
    styles = _source_styles(srcs, 3)
    styles[(1, 2)] = clip > 0 ? :accent : :muted
    push!(sections, TableSection("bindings", ["binding", "what", "source"], rows, styles))

    if !isempty(groups)
        # Seven facts in three columns, since every section shares one column structure and a
        # seven-column band would strand the two-column bands across it.
        rows = TableRows(
            [
                [
                    repr(g.name),
                    string(
                        "eta ", _g(g.base_eta), " (x",
                        string(round(Float64(g.ratio); digits = 2)), "), lambda ",
                        _g(g.lambda), " toward ", g.anchor, ", ", g.rule
                    ),
                    _commas(g.params),
                ] for g in groups
            ]
        )
        push!(
            sections, TableSection(
                "parameter groups (G = " * string(length(groups)) * ")",
                ["group", "settings", "params"], rows
            )
        )
    end

    if !isempty(level2)
        rows = TableRows()
        for l in level2
            push!(
                rows, [
                    repr(l.group), "present in the returned chain",
                    isempty(l.present) ? "(none)" : join(l.present, ", "),
                ]
            )
            isempty(l.absent) || push!(
                rows, [
                    "", "NOT present, the factory did not apply it", join(l.absent, ", "),
                ]
            )
        end
        push!(sections, TableSection("level 2 chains", ["group", "binding", "values"], rows))
    end

    return sections
end

"""
    ReactantNitro.binding_report_text(; kwargs...) -> String

[`binding_report_sections`](@ref) rendered as plain text for `log_other!`. An `IOBuffer` declares
no `:color`, so the text carries the frame and none of the colour.
"""
function binding_report_text(; name = "", kwargs...)
    io = IOBuffer()
    _render_sections(
        io, MIME"text/plain"(), "ReactantNitro: binding report for $name",
        binding_report_sections(; name, kwargs...); note = nothing
    )
    return String(take!(io))
end


"""
    ReactantNitro._source_styles(sources, col) -> CellStyles

The role for each row's source cell: `:accent` for a value a `train!` keyword set (what somebody
changed for this run) and `:muted` for the framework's own default (a value nobody chose). The
accessor and method sources stay plain, since a value from the experiment is the ordinary case.
"""
_source_styles(sources, col::Int) = CellStyles(
    (i, col) => (src === :keyword ? :accent : :muted)
        for (i, src) in enumerate(sources) if src === :keyword || src === :default
)

_source_label(s::Symbol) = s === :keyword ? "train! keyword" :
    s === :accessor ? "schedules(e)" :
    s === :method ? "method on the experiment type" :
    s === :field ? "field on e" :
    s === :default ? "framework default" : string(s)

# Automatic-loop opt keys: a bare key binds every group's chain, a dotted `group.field` key binds
# that group's only, with the per-group ratio still applied.
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
