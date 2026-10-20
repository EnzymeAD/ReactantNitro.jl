# Schedules.jl
#
# Schedule resolution, the effective learning rate, and the binding report. The report's sections
# are appended to `show(nitro)` in Phases.jl and its text goes to the run's logger.

"""
    ReactantNitro.resolve_schedules(e, given, total) -> resolved

Resolve the schedule table once, at setup. Each entry is a factory of the horizon, called here with
`total`; the step function it returns is then called once per optimizer step. A bare `Number`
becomes a `Constant` callable. `given` is the `schedules` keyword if passed and
[`schedules`](@ref)`(e)` otherwise; the keyword replaces the accessor's table and does not merge
with it. `accessor` is the accessor's table, used only to label each key's source in the binding
report.

Keys resolve against two namespaces, `Device` field names and the rule field names of `chains`,
reserved at the top level as `device` and `opt`. A scheduled rule field reaches the per-group `hp`;
a scheduled `Device` field is written into the experiment each step and traced code reads it as
`e.<field>`. A third, `grad`, holds the [`grad_rule`](@ref)'s fields (`grad_fields`). An
unqualified key resolves to the one namespace it matches. A key matching none is an error naming the
fix; a key matching several is refused, and the error shows the qualified forms and `bind`.

`bind` holds free names that resolve against nothing: their values reach `hp.bind` in a Level 2
`optimizer(e, group, hp)` and in `grad_rule(e, hp)`, and the user places them.

`step` is the optimizer step, and `total = max_epochs * div(steps_per_epoch, accum)` is exact. A
nested `opt` key is a one-level parameter-group path: `opt = (; backbone = (; eta = ...))` binds
the `:backbone` group only, and must name a group in `groups` and a schedulable field of its chain.
For each group a path key overrides the bare key, which overrides the accessor. Path values get the
same per-group scaling as bare ones (see [`effective_lr`](@ref)). Manual mode's path values are
absolute ([`resolve_manual_schedules`](@ref)).
"""
struct ResolvedSchedules{T <: NamedTuple, O <: NamedTuple, R <: NamedTuple, B <: NamedTuple}
    device::T                       # Device field name -> callable(step)
    opt::O                          # rule field name    -> callable(step)
    grad::R                         # grad_rule field    -> callable(step)
    bind::B                         # free name          -> callable(step)
    source::Dict{Symbol, Symbol}     # display key        -> :keyword | :accessor
    constant::Vector{Symbol}        # display keys that were bare Numbers
    horizon_dependent::Bool
    plateaus::Vector                # key => PlateauSchedule, from `collect_plateaus`
end

ResolvedSchedules(device, opt, grad, bind, source, constant, horizon_dependent) = ResolvedSchedules(
    device, opt, grad, bind, source, constant, horizon_dependent,
    collect_plateaus(device, opt, grad, bind)
)

Base.isempty(r::ResolvedSchedules) =
    isempty(r.device) && isempty(r.opt) && isempty(r.grad) && isempty(r.bind)

# A constant is a `Number` and a schedule is a `Callable`, so the per-step transfer count equals the
# number of quantities actually being varied.
_as_factory(x::Number) = _ -> Constant(x)
_as_factory(f) = f

# A bare `Number` schedule, typed so the schedule log can skip it.
struct Constant{T}
    x::T
end
(c::Constant)(_) = c.x

"""
    ReactantNitro.HorizonGuard

Wraps a schedule so that a throw past its horizon holds the step-`total` value and warns once,
instead of ending the run. A schedule that returns a value past `total` is used as is. The horizon
is exact only while `length(train)` is stable across epochs, which the data contract requires.
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

# `device`, `opt`, `grad` and `bind` are reserved at the top level; every other key is unqualified
# and must resolve to exactly one of `device`, `opt` and `grad`. A NamedTuple value under `opt` is a
# parameter-group path, flattened to a dotted display key (`Symbol("backbone.eta")`).
function _flatten_schedules(nt::NamedTuple, what::AbstractString)
    out = Tuple{Symbol, Symbol, Any}[]
    for (k, v) in pairs(nt)
        if k === :device || k === :opt || k === :grad || k === :bind
            v isa NamedTuple || error(
                """
                ReactantNitro: `$k` is reserved at the top level of $what and names a namespace, so
                it must be a NamedTuple of schedules. Got a `$(typeof(v))`.
                If you meant a `Device` field literally named `$k`, write `$k = (; $k = ...)`."""
            )
            if k === :opt
                _flatten_opt!(out, v)
            elseif k === :bind
                for (k2, v2) in pairs(v)
                    v2 isa NamedTuple && error(
                        "ReactantNitro: `bind.$k2` is a NamedTuple. `bind` names are flat: each \
                         is a schedule, read as `hp.bind.$k2`."
                    )
                    push!(out, (k, k2, v2))
                end
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
        chains = (), accessor = nothing, groups = (:default,), grad_fields = ()
    )
    df = device_fields(typeof(e))
    of = schedulable_fields(chains)
    acc = accessor === nothing ? given : accessor
    from_accessor = Set(_display_key(ns, k) for (ns, k, _) in _flatten_schedules(acc, "`schedules(e)`"))

    tun, opt, grad, bind = ntuple(_ -> Pair{Symbol, Any}[], 4)
    source, constant = Dict{Symbol, Symbol}(), Symbol[]
    varying = false

    for (ns, k, spec) in _flatten_schedules(given, "`schedules`")
        display = _display_key(ns, k)
        target = ns
        if ns === :auto
            spaces = ((:device, df), (:opt, of), (:grad, grad_fields))
            hits = Symbol[ns2 for (ns2, fs) in spaces if k in fs]
            if length(hits) > 1
                where = Dict(
                    :device => "a `Device` field on $(nameof(typeof(e))), read as `e.$k`",
                    :opt => "a schedulable field of the optimizer chain",
                    :grad => "a field of the gradient rule",
                )
                error(
                    """
                    ReactantNitro: schedule key `$k` is ambiguous, so it is refused rather than
                    guessed. It names:
                    $(join(("  - " * where[h] for h in hits), "\n"))
                    Qualify it:
                    $(join(("    schedules = (; $h = (; $k = <factory>))" for h in hits), "\n"))
                    or bind it yourself under a name of your choosing:
                        schedules = (; bind = (; my_$k = <factory>))
                    and read `hp.bind.my_$k` in `optimizer(e, group, hp)` or `grad_rule(e, hp)`.
                    Renaming the field is never required; collisions on `lambda`, `eta`, `rho`, and
                    `epsilon` are expected rather than exotic."""
                )
            elseif length(hits) == 1
                target = only(hits)
            else
                error(
                    """
                    ReactantNitro: schedule key `$k` matches nothing. An unqualified key is a
                    `Device` field name, a rule field name, or a gradient rule field name.
                    `Device` fields on $(nameof(typeof(e))): $(isempty(df) ? "(none)" : df)
                    Schedulable optimizer fields: $(isempty(of) ? "(none; no chain was supplied)" : of)
                    Gradient rule fields: $(isempty(grad_fields) ? "(none)" : grad_fields)
                    Anything you want to vary inside `loss`, `forward`, or `metrics` must be a
                    `Device` field; anything on the optimizer must be a rule field. Note the
                    learning-rate key is `eta`, not `lr`. A free name goes under `bind`."""
                )
            end
        elseif ns === :bind
            # Free names: the probe at setup checks that something reads each one.
        elseif ns === :grad
            k in grad_fields || error(
                "ReactantNitro: `grad.$k` names no schedulable field of the gradient rule. \
                 Its schedulable fields are $(isempty(grad_fields) ? "(none)" : grad_fields)."
            )
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
        dest = target === :device ? tun : target === :grad ? grad : target === :bind ? bind : opt
        push!(dest, k => sched)
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
    return ResolvedSchedules(
        NamedTuple(tun), opt_nt, NamedTuple(grad), NamedTuple(bind), source, constant, varying
    )
end

"""
    ReactantNitro.check_lambda_conflict(e, groups, [key]) -> nothing

Error if `lambda` is scheduled and any of `groups` also defines a per-group `lambda` accessor, since
the schedule supplies the coefficient. Called with every group for a bare key and with one group for
a path key. `eta` has no such check: per-group `learning_rate` accessors are ratios that multiply an
`eta` schedule, while a scheduled `lambda` gets its per-group scaling from `η_g(t)`.
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

The effective learning rate of `group`, which `hp.eta` holds and `hp.lambda` is scaled by:

    η_g(t) = eta_t * (learning_rate(e, Val(g)) / learning_rate(e))

`eta_t` is the group's `eta` schedule value at the step (its path key, else the bare key), or
`nothing` when neither is set, in which case `learning_rate(e)` stands in. Per-group accessors are
ratios, so a backbone at `1f-4` against a default of `1f-3` stays a tenth of it for the whole run.
`lambda` is multiplied by the effective rate rather than by a ratio, so a default group with zero
decay needs no division:

    λ_g(t) = η_g(t) * lambda_t             # `lambda` scheduled for the group
    λ_g(t) = η_g(t) * lambda(e, Val(g))    # otherwise

Every other scheduled field is absolute. Errors if `learning_rate(e)` is zero.
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

Evaluate the `Device` schedules at `step` and return `fields` with each scheduled entry replaced by
its device value. Each value is first converted to the current field's type, which is why this takes
`fields`: `t -> 0.25 * min(1, t / 2000)` returns `Float64` for a `Device{Float32}` field, and the
unconverted value would change `typeof(compile_view(e))` and recompile the gradient program. An
array value must keep the field's size.
"""
step_aux(fields::NamedTuple, scheds::NamedTuple, step::Integer; mesh = nothing) =
    _step_aux(fields, map(f -> f(step), scheds); mesh)
step_aux(fields::NamedTuple, r::ResolvedSchedules, step::Integer; mesh = nothing) =
    step_aux(fields, r.device, step; mesh)

# `raw` is the schedules' output at one step, before coercion.
function _step_aux(fields::NamedTuple, raw::NamedTuple; mesh = nothing)
    ks = keys(raw)
    vals = map(k -> to_device(_coerce_scheduled(getproperty(fields, k), getproperty(raw, k), k); mesh), ks)
    return merge(fields, NamedTuple{ks}(vals))
end

# The `device` schedules evaluated at `step`, uncoerced; `(;)` when none are configured.
device_schedule_values(sched, step::Integer) =
    sched === nothing ? (;) : map(f -> f(step), sched.device)

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

Return a copy of `e` with its scheduled [`Device`](@ref) fields rebuilt through
[`step_aux`](@ref), so `device = (; aux_weight = ...)` varies `e.aux_weight` inside `loss` and
`forward`. The experiment-side counterpart of `rebuild_rules`. Returns `e` itself when no `device`
key is scheduled. `step` is the upcoming optimizer step, `nitro.step + 1`. `raw` is the step's
schedule output; the loop passes it in so the schedule log reads the values applied here. Errors if
the rebuild changes `typeof(compile_view(e))`, which would recompile on every step.
"""
function step_experiment(
        e, sched, step::Integer; mesh = nothing, raw = device_schedule_values(sched, step)
    )
    (sched === nothing || isempty(sched.device)) && return e
    T = typeof(e)
    ks = keys(sched.device)
    cur = NamedTuple{ks}(map(f -> getfield(e, f), ks))
    stepped = _step_aux(cur, raw; mesh)
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

Move a value to the device, replicated over `mesh` when one is given. A `Number` needs
`track_numbers = Number`: bare `to_rarray(2.0f0)` returns the scalar unchanged, and it would bake
into the trace as a constant. A device number is used instead of a 1-element array, whose `[1]`
throws under trace. Values already on the device are returned as is, since `to_device_rule` runs
over rules built with device scalars.

A `Tuple` or `NamedTuple` is converted elementwise. This is how a `Device` field holds what PyTorch
calls buffers: arrays the graph reads and nothing differentiates, such as a frozen backbone's
weights. In `ps` they would be trained; in a `GraphConst` they would be baked into the graph. They
reach Enzyme as `Const(ev)`, so the gradient treats them as constants.
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

Resolve the schedule table for a manual-mode experiment. `device` keys resolve as in
[`resolve_schedules`](@ref); `opt` keys resolve against the user's `opt_state`. A bare key is a rule
field name and sets that field on every rule that has it. A nested key is a path into `opt_state`
whose innermost key is the field name; a path key overrides a bare key for its subtree. Every value
is absolute. Paths and fields are validated at setup.

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
        r.device, flat, r.grad, r.bind, merge(r.source, source), vcat(r.constant, constant),
        r.horizon_dependent || varying
    )
end

"""
    ReactantNitro.flatten_manual_opt(opt_table, opt_state, total, accessor_table) -> (flat, source, constant, varying)

Flatten manual mode's possibly nested `opt` table into step functions keyed by display key,
validating each binding against `opt_state`. A path binding's display key is the dotted path plus
the field, `Symbol("gen.eta")`.
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

Evaluate every `opt` schedule at `step` and rebuild the rules it names, on the host; manual mode's
counterpart of `rebuild_rules`. Values are absolute. Shallower keys are applied first, so a path key
overrides a bare key whatever the order they were written in. Each value is converted to the field's
element type, so the rule types do not change and the closure's program is reused.
"""
function rebuild_scheduled_rules(
        nitro::Nitro, step::Integer; raw = opt_schedule_values(nitro.schedules, step)
    )
    isempty(raw) && return nitro.opt_state
    opt_state = nitro.opt_state
    # Shallower keys first, so a path key overrides a bare key for its subtree whatever the order
    # they were written in, as in the automatic loop.
    for k in sort(collect(keys(raw)); by = k -> count('.', string(k)))
        path, field = _schedule_target(k)
        opt_state = apply_schedule(opt_state, path, field, raw[k])
    end
    return opt_state
end

# The `opt` schedules evaluated at `step`, keyed as resolved; `(;)` when none are configured.
opt_schedule_values(sched, step::Integer) =
    sched === nothing ? (;) : map(f -> f(step), sched.opt)

"""
    ReactantNitro.group_schedule_values(sched, group, step) -> NamedTuple

The automatic loop's `opt` schedule values for one parameter group at `step`, keyed by rule field.
A path key (`backbone.eta`) overrides the bare key (`eta`) for its group. A field with neither is
absent, and [`hp_host`](@ref) falls back to the accessor.
"""
function group_schedule_values(sched, group::Symbol, step::Integer)
    sched === nothing && return (;)
    vals = Pair{Symbol, Any}[]
    for (k, f) in pairs(sched.opt)
        _binds(sched, k, group) && push!(vals, last(_schedule_target(k)) => f(step))
    end
    isempty(sched.bind) || push!(vals, :bind => bind_schedule_values(sched, step))
    return (; vals...)
end

# The `bind` schedules evaluated at `step`; `(;)` when none are configured.
bind_schedule_values(sched, step::Integer) =
    sched === nothing ? (;) : map(f -> f(step), sched.bind)

# Whether `opt` key `k` sets its field for `group`: a path key binds its group, and a bare key every
# group without a path key for the same field.
function _binds(sched, k::Symbol, group::Symbol)
    path, field = _schedule_target(k)
    return isempty(path) ? !haskey(sched.opt, Symbol(group, ".", field)) : only(path) === group
end

"""
    ReactantNitro.setup_schedule_values(e, given, group; elt = Float32) -> NamedTuple

Zero placeholders for the rule fields `given` schedules for `group`, read from the unresolved table.
Setup must build each group's chain before resolving the keys, since resolution reads the chains, so
these make `hp` carry every scheduled field from the first call: a Level 2 factory that reads
`hp.rho` would otherwise fail. `eta` is always in `hp` and is left out. `device` keys, unqualified
keys naming a `Device` field, unqualified dotted keys, other groups' path keys, and paths deeper
than one level are skipped.
"""
function setup_schedule_values(e, given::NamedTuple, group::Symbol; elt::Type = Float32)
    df = device_fields(typeof(e))
    fs, bs = Symbol[], Symbol[]
    for (ns, k, _) in _flatten_schedules(given, "`schedules`")
        ns === :bind && (push!(bs, k); continue)
        (ns === :device || ns === :grad) && continue
        ns === :auto && (k in df || occursin('.', string(k))) && continue
        path, field = _schedule_target(k)
        length(path) <= 1 && (isempty(path) || only(path) === group) || continue
        field === :eta || field in fs || push!(fs, field)
    end
    sv = NamedTuple{Tuple(fs)}(ntuple(_ -> zero(elt), length(fs)))
    isempty(bs) && return sv
    return merge(sv, (; bind = NamedTuple{Tuple(bs)}(ntuple(_ -> zero(elt), length(bs)))))
end

# The rule fields one group's train line logs: each varying schedule binding the group, plus
# `lambda` when only `eta` varies and the group decays.
function _varying_fields(sched, group::Symbol, decays::Bool)
    fs = Symbol[
        last(_schedule_target(k)) for (k, f) in pairs(sched.opt)
            if !(f isa Constant) && _binds(sched, k, group)
    ]
    :eta in fs && !(:lambda in fs) && decays && push!(fs, :lambda)
    return fs
end

"""
    ReactantNitro.schedule_bindings(e, layout, sched, chains; anchors, masks) -> NamedTuple

Record, once at setup, where each scheduled rule field lands. Returns `(; fields, slots, level2)`:

- `fields[gi]`: the rule fields group `gi`'s train line logs, read each step without reflection
- `slots[gi]`: the `(rule position, field)` pairs of a Level 2 chain that hold a varying value, for
  [`place_slots`](@ref)
- `level2`: one `(; group, present, absent)` per group when `optimizer(e, group, hp)` is
  user-defined, for the binding report; `present` lists `field: Rule.field` entries and `absent` is
  always empty, since an absent field is an error

At Level 1 a scheduled field reaches the rule by name, so `fields` is the varying fields the chain
has. At Level 2 the factory places the values, and a matching field name does not show it read
`hp.rho`. So each scheduled field is probed on the host: the factory is called with two `hp`s that
differ only in that field, and the field is bound to the rule slots that differ. A scheduled field,
constant or varying, that lands in no slot is a setup error.
"""
function schedule_bindings(e, layout, sched, chains; anchors = nothing, masks = nothing)
    G = length(layout.groups)
    none = (; fields = ntuple(_ -> (), G), slots = ntuple(_ -> (), G), level2 = (), landed = Dict())
    sched === nothing && return none
    bkeys = Tuple(keys(sched.bind))
    # Where each `bind` key landed, across groups; the grad rule's landings are added at setup.
    landed_all = Dict{Symbol, Vector{String}}(Symbol("bind.", k) => String[] for k in bkeys)
    level2 = !is_framework_default(optimizer, Tuple{typeof(e), Symbol, Any})
    fields, slots = Vector{Tuple}(undef, G), Vector{Tuple}(undef, G)
    report = []
    for (gi, g) in enumerate(layout.groups)
        decays = !iszero(lambda(e, Val(g))) || any(
            k -> _binds(sched, k, g) &&
                last(_schedule_target(k)) === :lambda, keys(sched.opt)
        )
        fs = _varying_fields(sched, g, decays)
        slots[gi] = ()
        if !level2
            present = schedulable_fields((chains[gi],))
            fields[gi] = Tuple(f for f in fs if f in present)
            continue
        end
        scheduled = Symbol[last(_schedule_target(k)) for k in keys(sched.opt) if _binds(sched, k, g)]
        landed = Dict(
            f => _probe_field(e, layout, gi, scheduled, f; anchors, masks, bind_keys = bkeys)
                for f in unique(vcat(scheduled, fs))
        )
        blanded = Dict(
            k => _probe_field(
                e, layout, gi, scheduled, k; anchors, masks, bind_keys = bkeys, bind = true
            )
                for k in bkeys
        )
        for k in bkeys, sl in blanded[k]
            push!(landed_all[Symbol("bind.", k)], "$(repr(g)) $(sl.rule).$(sl.field)")
        end
        absent = [f for f in scheduled if isempty(landed[f])]
        isempty(absent) || error(
            """
            ReactantNitro: group $(repr(g)) schedules $(join(absent, ", ")), and the chain returned
            by `optimizer(e, $(repr(g)), hp)` does not use $(length(absent) == 1 ? "it" : "them"):
            calling it with two different values of $(join(("hp." * string(f) for f in absent), ", "))
            returned identical rules, so the schedule would compute a value every step and discard
            it. Read the value from `hp` (`Optimisers.Momentum(hp.eta, hp.rho)`), or remove the
            schedule."""
        )
        fields[gi] = Tuple(f for f in fs if !isempty(landed[f]))
        # Only the varying fields' slots: a slot fed by a constant is correct as a baked constant.
        varying_bind = [k for k in bkeys if !(sched.bind[k] isa Constant)]
        slots[gi] = Tuple(
            unique(
                vcat(
                    [(s.pos, s.field) for f in fs for s in landed[f]],
                    [(s.pos, s.field) for k in varying_bind for s in blanded[k]],
                )
            )
        )
        present = Tuple(
            "$f: " * join(("$(s.rule).$(s.field)" for s in landed[f]), ", ")
                for f in unique(vcat(scheduled, fs)) if !isempty(landed[f])
        )
        push!(report, (; group = g, present, absent = ()))
    end
    return (; fields = Tuple(fields), slots = Tuple(slots), level2 = report, landed = landed_all)
end

# The rule slots `hp.<field>` reaches in group `gi`'s chain, as `(; pos, rule, field)`: the factory
# called on two host `hp`s that differ only in `field`. Every scheduled field is present, as at setup.
function _probe_field(
        e, layout, gi, scheduled, field; anchors, masks, elt::Type = Float32, bind_keys = (),
        bind::Bool = false
    )
    g = layout.groups[gi]
    names = Tuple(f for f in scheduled if f !== :eta)
    sv = NamedTuple{names}(ntuple(_ -> elt(0.5), length(names)))
    isempty(bind_keys) ||
        (sv = merge(sv, (; bind = NamedTuple{bind_keys}(ntuple(_ -> elt(0.5), length(bind_keys))))))
    base = hp_host(e, g, sv; elt)
    one_key = NamedTuple{(field,)}
    with(v) = bind ? merge(base, (; bind = merge(base.bind, one_key((elt(v),))))) :
        merge(base, one_key((elt(v),)))
    keep_decay = :lambda in scheduled
    chain(v) = optimizer(
        e, g, resolve_hp(e, layout, gi; host = with(v), keep_decay, anchors, masks, device = false)
    )
    ra, rb = _rules_of(chain(0.25)), _rules_of(chain(0.75))
    length(ra) == length(rb) || error(
        """
        ReactantNitro: `optimizer(e, $(repr(g)), hp)` returns a chain of $(length(ra)) rules for
        one value of `hp.$(bind ? "bind." : "")$field` and $(length(rb)) for another. The chain's
        structure is fixed at setup, since the optimizer state is built from it; vary values only."""
    )
    at = NamedTuple{(:pos, :rule, :field), Tuple{Int, Symbol, Symbol}}[]
    for (i, (x, y)) in enumerate(zip(ra, rb)), f in fieldnames(typeof(x))
        u, v = getfield(x, f), getfield(y, f)
        u isa Number && v isa Number && !isequal(u, v) &&
            push!(at, (; pos = i, rule = nameof(typeof(x)), field = f))
    end
    return at
end

"""
    ReactantNitro.place_slots(chain, slots, gi; memo, mesh, elt = Float32) -> chain

Move host values in a Level 2 chain's recorded slots to the device, each step. A value the factory
computed from `hp`, such as `2 * hp.eta`, is a host number, and a host number in a rule bakes into
the optimizer program at its first value without entering the cache key, so the schedule would stop
taking effect with no error. Each slot from [`schedule_bindings`](@ref) holding a host number is
converted through `memo`, so an unchanged value is not uploaded again. A slot already on the device
is left alone.
"""
function place_slots(chain, slots, gi::Int; memo = nothing, mesh = nothing, elt::Type = Float32)
    isempty(slots) && return chain
    rules = collect(Any, _rules_of(chain))
    for (pos, f) in slots
        v = getfield(rules[pos], f)
        (v isa Number && !(v isa Reactant.RNumber)) || continue
        dev = memo_to_device(memo, (gi, Symbol(pos, ".", f)), elt(v); mesh)
        rules[pos] = _with_field(rules[pos], f, dev)
    end
    return chain isa Optimisers.OptimiserChain ? Optimisers.OptimiserChain(rules...) : only(rules)
end

_with_field(r, f::Symbol, v) = Base.typename(typeof(r)).wrapper(
    map(x -> x === f ? v : getfield(r, x), fieldnames(typeof(r)))...
)

"""
    ReactantNitro.schedule_log(
        sched, device_raw, opt_raw; groups, applied, fields, grad_raw, bind_raw
    ) -> NamedTuple

The schedule values the train line carries at one optimizer step, every key prefixed `schedule/`.
Constants and raw values that are not `Real` are left out; the binding report records the
constants.

- A `device` schedule logs as `schedule/<field>`, a `grad` one as `schedule/grad/<field>`, and a
  `bind` one as `schedule/bind/<name>`
- In the automatic loop, each group logs `schedule/<group>/<field>` for each of its
  [`schedule_bindings`](@ref) `fields`, as `applied` holds it: `eta` and `lambda` after the
  per-group scaling of [`effective_lr`](@ref)
- In manual mode (`groups === nothing`), each `opt` key in `opt_raw` logs its value as given, with
  the path joined by `/`
"""
function schedule_log(
        sched, device_raw::NamedTuple, opt_raw::NamedTuple = (;);
        groups = nothing, applied = (), fields = (), grad_raw = (;), bind_raw = (;)
    )
    out = Pair{Symbol, Any}[]
    # `Real`, not `Number`: a device scalar or a `Complex` is a `Number` that backends reject.
    logged(tbl, k, v) = !(tbl[k] isa Constant) && v isa Real
    key(name) = Symbol("schedule/", name)
    for (k, v) in pairs(device_raw)
        logged(sched.device, k, v) && push!(out, key(k) => v)
    end
    for (k, v) in pairs(grad_raw)
        logged(sched.grad, k, v) && push!(out, key(string("grad/", k)) => v)
    end
    for (k, v) in pairs(bind_raw)
        logged(sched.bind, k, v) && push!(out, key(string("bind/", k)) => v)
    end
    for (k, v) in pairs(opt_raw)
        logged(sched.opt, k, v) && push!(out, key(replace(string(k), "." => "/")) => v)
    end
    groups === nothing && return (; out...)
    for (gi, g) in enumerate(groups), f in fields[gi]
        push!(out, key(string(g, "/", f)) => applied[gi][f])
    end
    return (; out...)
end

# `:eta` -> ((), :eta); `Symbol("gen.eta")` -> ((:gen,), :eta). Rule field names cannot contain
# dots, so the last segment is always the field.
function _schedule_target(k::Symbol)
    parts = Symbol.(split(string(k), '.'))
    return (Tuple(parts[1:(end - 1)]), parts[end])
end

"""
    ReactantNitro.apply_schedule(opt_state, path, field, value) -> opt_state

Rebuild every rule under `opt_state`'s subtree `path` that has `field`, setting it to `value`
converted to the field's element type and moved to the device. Rules without the field are left
unchanged.
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

Set `field` to `value` in a new rule of the same type, keeping the leaf's state. The value is
converted to the field's element type so the rule's type does not change. Chains and single-field
rule wrappers are unwrapped and rewrapped, so a scheduled `eta` reaches an `Adam` inside either.
Returns `leaf` itself when no rule has `field`.
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

Collect the binding report's inputs from a [`Nitro`](@ref), for
[`binding_report_sections`](@ref) and [`binding_report_text`](@ref). It reads values the run already
computed and never fails. The report says where each configured value bound and from which source;
what the handle holds is the `state` band of `show(nitro)`, and what has been redefined since is
[`fixed_config_report`](@ref).

```
  data
  split  resolved
  train  46 batches; prefetch: 16 workers, 2 on device, 4 on host
  val    8 batches; prefetch: 1 producer, 2 on device, 4 on host

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
  group      binding                         values
  :backbone  present in the returned chain   eta: Momentum.eta, rho: Momentum.rho
```

The `level 2` block lists, per group, the rule fields each scheduled value landed in, from
[`schedule_bindings`](@ref). Each split's `samples`, `dropped` and `short_final` are left
`nothing`, so the `data` block shows the batch count and prefetch only.
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
                # Scheduled, so the accessor value above is not what the run uses.
                lambda_scheduled = nitro.schedules !== nothing && any(
                    k -> _binds(nitro.schedules, k, g) && last(_schedule_target(k)) === :lambda,
                    keys(nitro.schedules.opt)
                ),
                # The chain the run steps with, which at Level 2 is the factory's.
                rule = nitro.opt_state === nothing ? string(nameof(optimizer(nitro.e, Val(g)))) :
                    _chain_label(nitro.opt_state[gi].rule),
                params = lay.lengths[gi],
            )
                for (gi, g) in enumerate(lay.groups)
        ]
    end
    return (;
        name = string(nameof(typeof(nitro.e))), splits,
        clip = nitro.gradient_clip_norm,
        clip_source = clip_source(nitro.e, nitro.gradient_clip_norm),
        monitor = nitro.monitor,
        monitor_source = monitor_source(nitro),
        watchers = watchers(nitro),
        schedules = nitro.schedules, groups,
        manual = get(nitro.frozen, :manual, false),
        level2 = nitro.bindings.level2,
        landed = get(nitro.bindings, :landed, Dict()),
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
accessor returns, `:method` for a user accessor method, `:field` for a declared field, and
`:default` for the framework's own `0f0`. The keyword is detected by value, since by the time the
constructor body runs a passed keyword cannot be told apart from the accessor default; a keyword
equal to the accessor's value reports the accessor's source.
"""
clip_source(e, resolved) = accessor_source(gradient_clip_norm, e, resolved, :gradient_clip_norm)

# Which route supplied a run accessor's resolved value; see `clip_source`.
accessor_source(f, e, resolved, field::Symbol) =
    resolved != f(e) ? :keyword :
    !is_framework_default(f, Tuple{typeof(e)}) ? :method :
    hasfield(typeof(e), field) ? :field : :default

# The accessor's `nothing` resolved by setup reads as the framework's default.
monitor_source(nitro) =
    monitor(nitro.e) === nothing && nitro.monitor in (nothing, :loss => :min) ? :default :
    accessor_source(monitor, nitro.e, nitro.monitor, :monitor)

# What reads a validation metric, as `(who, metric => mode)` after setup filled in the monitor.
function watchers(nitro)
    out = Tuple{String, Pair{Symbol, Symbol}}[]
    ck = nitro.checkpointer
    ck isa Checkpointer && ck.metric !== nothing &&
        push!(out, ("checkpointer", ck.metric => ck.mode))
    es = nitro.early_stop
    es isa EarlyStopping && es.metric !== nothing &&
        push!(out, ("early stopping", es.metric => es.mode))
    for (k, s) in plateaus(nitro.schedules)
        push!(out, ("plateau `$k`", s.p.metric => s.p.mode))
    end
    return out
end

_commas(n::Integer) = replace(string(n), r"(?<=[0-9])(?=(?:[0-9]{3})+$)" => ",")
_g(x) = string(round(Float64(x); sigdigits = 3))

# `Adam + Decay` for a chain, `Adam` for a single rule.
_chain_label(r) = join((string(nameof(typeof(x))) for x in _rules_of(r)), " + ")

"""
    ReactantNitro.prefetch_report_entry(name, split) -> (; device_batches, host_batches, workers, ordered, path)

The prefetch setup a split resolved to, which may differ from what was requested (see
`prefetch_config`).
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
                                            schedules, groups, level2 = (), manual = false) -> Vector{TableSection}

The binding report's [`TableSection`](@ref)s, built from explicit pieces so they can be tested
without a run. They report where values bound and nothing else; the seed, accum, horizon, batch
size and preset belong to the `state` band, so no fact is printed twice.

  * `splits`: one `(; name, batches, batch_size, samples, dropped, short_final[, prefetch])` per
    split. `samples`, `dropped` and `short_final` are omitted when `nothing`; `prefetch` is
    optional; `batch_size` is accepted and not printed.
  * `clip_source`: `:keyword`, `:method`, `:field` or `:default`.
  * `groups`: one `(; name, base_eta, ratio, anchor, lambda, rule, params[, lambda_scheduled])` per
    parameter group.
  * `level2`: one `(; group, present, absent)` per Level 2 chain.
  * `manual`: describe `opt` keys as manual-mode bindings into `opt_state`.

`name` is accepted and not printed, since the table's title names the experiment.
"""
function binding_report_sections(;
        name = "", splits = (), clip = 0,
        clip_source::Symbol = :default, schedules = nothing, groups = (),
        level2 = (), manual = false, landed = Dict(), monitor = missing,
        monitor_source::Symbol = :default, watchers = ()
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

    if monitor === missing
        # Not given: a report built by hand, without a run.
    elseif monitor === nothing
        push!(rows, ["monitor", "none: `metrics` is defined and `eval_loss` is off; nothing ranks", "[" * _source_label(monitor_source) * "]"])
        push!(srcs, monitor_source)
    else
        follow = [w for (w, m) in watchers if m == monitor]
        own = ["$w on $(first(m)) ($(last(m)))" for (w, m) in watchers if m != monitor]
        what = "$(first(monitor)) ($(last(monitor)))" *
            (isempty(follow) ? "; nothing follows it" : "; followed by " * join(follow, ", ")) *
            (isempty(own) ? "" : "; own: " * join(own, ", "))
        push!(rows, ["monitor", what, "[" * _source_label(monitor_source) * "]"])
        push!(srcs, monitor_source)
    end

    if schedules !== nothing && !isempty(schedules)
        tables = (
            (:opt, schedules.opt), (:device, schedules.device), (:grad, schedules.grad),
            (:bind, schedules.bind),
        )
        for (ns, tbl) in tables
            for k in keys(tbl)
                disp = haskey(schedules.source, k) ? k : Symbol(ns, ".", k)
                what = ns === :opt ?
                    (
                        manual ? _manual_opt_desc(k, keys(tbl)) :
                        _auto_opt_desc(k, keys(tbl), Tuple(g.name for g in groups))
                    ) :
                    ns === :grad ? "gradient rule field, the ε program only" :
                    ns === :bind ? "bound by hand: " * join(get(landed, disp, String[]), ", ") :
                    "Device field   e.$k"
                note = (disp in schedules.constant ? "  (constant)" : "") * _plateau_desc(tbl[k])
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
                        get(g, :lambda_scheduled, false) ? "scheduled" : _g(g.lambda),
                        " toward ", g.anchor, ", ", g.rule
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

The style of each row's source cell in column `col`: `:accent` for a value a `train!` keyword set
and `:muted` for the framework's own default. Every other source is unstyled.
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

# Automatic-loop opt keys: a dotted `group.field` key binds that group's chain, and a bare key every
# group without one, with the per-group ratio still applied.
function _auto_opt_desc(k::Symbol, opt_keys = (), groups = ())
    parts = split(string(k), '.')
    field = Symbol(last(parts))
    scope = if length(parts) > 1
        "group :$(first(parts))"
    else
        rest = Tuple(g for g in groups if Symbol(g, ".", field) ∉ opt_keys)
        isempty(groups) || length(rest) == length(groups) ? "all groups" :
            isempty(rest) ? "no group, every group has a path key" :
            "groups " * join(repr.(rest), ", ")
    end
    return field === :eta ? "optimizer field, $scope, per-group ratio applied" :
        field === :lambda ? "optimizer field, $scope, times eta per group" :
        "optimizer field, $scope, absolute"
end

# Manual mode: the opt schedule keys are binding descriptors (a bare field name, or a dotted
# path into the user's opt_state). The report says what the key binds to.
function _manual_opt_desc(k::Symbol, opt_keys = ())
    parts = split(string(k), '.')
    length(parts) == 1 && return any(o -> endswith(string(o), "." * string(k)), opt_keys) ?
        "optimizer field, every rule outside a path key for it, absolute" :
        "optimizer field, every rule, absolute"
    return "optimizer field at opt_state path $(join(parts[1:(end - 1)], "."))"
end
