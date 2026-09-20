# Hooks.jl
#
# PROTOTYPE. The hook map: hooks supplied as a NamedTuple of functions rather than as methods.
#
# Why this exists at all is a reactive-notebook problem. A hook defined as a method mutates a
# global, unversioned method table, which no value names and no dependency graph can see, so a
# notebook cannot tell that a handle is downstream of it. A hook supplied as a VALUE is named, so
# it is downstream of exactly what it is built from, and `merge` gives a second hook set that
# coexists with the first instead of overwriting it.
#
# THE ONE RULE: a hook function may not capture. Non-capturing is not hygiene here, it is what
# makes the existing compile-cache key sound WITHOUT modification. The map travels into the
# programs the same way the `Router`s do, as a `Const` argument whose TYPE is read by
# `map(typeof, args)` in `cache_key`. A non-capturing closure is a singleton type, so its identity
# IS its type and the key separates two different hooks for free. A capturing closure is not: two
# closures over different values share a type, `map(typeof, args)` cannot tell them apart, and the
# cache would serve one program for two different computations. Cache.jl's invariant is "a hit
# must be provably the same program; when in doubt, MISS", and an unhashable capture cannot meet
# it, so the capture is refused at construction instead.
#
# It is not a restriction in practice: every hook already takes the experiment as its first
# argument, so a value a hook needs belongs on `e`, where `Device`/`Host`/`GraphConst` state its
# residency and `graphconst_field_hash` puts it in the key. That is the same discipline, reached
# from the other side.

"""
    ReactantNitro.hook_fn(hooks, name, fallback) -> Function

The hook map's lookup: the supplied function for `name`, or the dispatch-based `fallback`. One
funnel, so "a map entry shadows the method" is stated once rather than at every call site.
"""
hook_fn(hooks, name::Symbol, fallback) = get(hooks, name, fallback)

"""
    ReactantNitro.hook_fns(routing) -> NamedTuple

The hook map carried by a routing, or `(;)` when there is none.

Guarded rather than a bare `routing.fns`, because a routing is not always one this framework built:
the eval path can carry `nothing`, and the suite hand-builds routing tuples to exercise the cache
key directly. Both are legitimate, and both predate the map. Everything is type-level, so the
lookup folds at trace time.
"""
hook_fns(routing::NamedTuple) = haskey(routing, :fns) ? routing.fns : (;)
hook_fns(::Any) = (;)

"""
    ReactantNitro.check_hooks(hooks) -> hooks

Refuse a hook map that cannot be keyed: a non-function entry, or a closure that captures.

The capture check is the load-bearing one; see this file's header for why a captured value is
invisible to the compile-cache key and therefore a silent wrong-program hazard rather than a
recompile.
"""
function check_hooks(hooks)
    hooks isa NamedTuple || error(
        "ReactantNitro: `hooks` must be a NamedTuple of functions, as `@hooks` builds, and got \
         `$(typeof(hooks))`."
    )
    for (name, f) in pairs(hooks)
        f isa Function || error(
            "ReactantNitro: hook `$name` is a `$(typeof(f))` rather than a function."
        )
        # `Base.issingletontype`, NOT `isempty(fieldnames(...))`. A hook takes its batch fields as
        # keywords, and Julia lowers a keyword method to a wrapper holding the inner body
        # function, so every routed hook has exactly one field and the naive check refuses all of
        # them. That field is CODE identity, not runtime data: the inner function is itself a
        # singleton, the wrapper is still zero-size, and the type still determines the value. The
        # predicate that separates the two is "has exactly one instance", which is precisely the
        # property the cache key needs. `hook_predicate_available` is the assertion that it still
        # means that.
        Base.issingletontype(typeof(f)) || error(
            "ReactantNitro: hook `$name` CAPTURES $(join(("`$t`" for t in _captured(f)), ", ")). \
             A captured value is invisible to the compile-cache key, which reads the hook's TYPE \
             and cannot separate two closures over different values, so the cache would serve \
             one program for two different computations. Put the value on the experiment, where \
             `Device`/`Host`/`GraphConst` states its residency and it is hashed into the key, and \
             read it off the `e` the hook already receives."
        )
    end
    return hooks
end

# The captured field types, for the error message only. Descends through the keyword wrapper so
# the report names what the USER captured rather than the lowering artifact holding it.
function _captured(f)
    out = Any[]
    walk(T) = Base.issingletontype(T) ? nothing :
        isconcretetype(T) && isstructtype(T) && fieldcount(T) > 0 ?
        foreach(walk, fieldtypes(T)) : push!(out, T)
    walk(typeof(f))
    return isempty(out) ? Any["a value"] : unique(out)
end

"""
    ReactantNitro.hook_predicate_available() -> Bool

Whether `Base.issingletontype` still separates a keyword-lowered hook from a capturing one.

[`check_hooks`](@ref) rests entirely on that predicate, and it is a Base internal, so it gets the
same treatment as `primary_world_available`: a named check with a test asserting it still behaves
as documented. A `false` here means the capture guard has silently stopped guarding.
"""
function hook_predicate_available()
    kw = let
        g(e, x; y = 0) = x + y
        g
    end
    w = 2.0f0
    cap = let w = w
        (e, x; y = 0) -> x + y * w
    end
    return Base.issingletontype(typeof(kw)) && !Base.issingletontype(typeof(cap))
end

"""
    ReactantNitro.@hooks begin ... end -> NamedTuple

Build a hook map from ordinary function definitions.

```julia
base = @hooks begin
    forward(e, model, ps, st; x) = Lux.apply(model, x, ps, st)
    loss(e, pred; y) = sum(abs2, pred .- y) / size(y, 2)
end

tweaked = merge(base, @hooks begin
    loss(e, pred; y) = sum(abs, pred .- y) / size(y, 2)
end)
```

Each definition becomes a value rather than a method: the body is wrapped in a `let`, so the name
is local, nothing is added to any method table, and the result is a singleton function type. Two
maps built from the same source are still two distinct types, so a handle built from `base` and
one built from `tweaked` compile separately and neither invalidates the other.

A group is the unit of invalidation, so [`check_hook_grouping`](@ref) refuses one that mixes a
traced hook with a host hook: editing a free hook beside `forward` is not free. A group holding
`metrics` or `train_metrics` gets a warning naming the residency the grouping assumes, since
`metrics_residency` decides that at run time and the macro cannot see it.

Definitions here are ordinary Julia, so referencing a value defined elsewhere is a plain global
reference and behaves as it would in a method. What is refused is a CAPTURE, at
[`check_hooks`](@ref).
"""
# ── The grouping rules ──────────────────────────────────────────────────────────────
#
# A group is the unit of invalidation: re-evaluating one mints a fresh type for EVERY hook in it,
# so a group is only as cheap to edit as its most expensive member. That makes the layout a real
# cost decision rather than a style preference, and the macro sees the names, so it is checked
# here instead of documented and forgotten.
#
# Measured on a toy run: `build_data` edited in a host-only group costs 0 recompiles, and the same
# edit with `build_data` grouped beside `forward` and `loss` costs 3, the gradient among them. The
# rule exists for that number.

"Hooks that are ALWAYS part of a compiled program, so editing their group always recompiles."
const TRACED_HOOKS = (:forward, :loss)

"""
Hooks whose residency `metrics_residency` decides at RUN time, which the macro cannot see. Their
grouping is therefore read as a statement of intent and reported back, rather than checked.
"""
const RESIDENCY_HOOKS = (:metrics, :train_metrics)

"Hooks that are part of no compiled program, so editing their group costs no recompile."
const HOST_HOOKS = (:build_model, :build_data, :finalize_metrics)

const SUPPORTED_HOOKS = (TRACED_HOOKS..., RESIDENCY_HOOKS..., HOST_HOOKS...)

"""
    ReactantNitro.check_hook_grouping(names) -> nothing

Enforce the group layout, at macro expansion, from the hook names alone.

Refuses a group mixing a traced hook with a host one, and reports the residency a group ASSUMES
for `metrics` or `train_metrics`. Also refuses a name the map does not carry, which would
otherwise sit in the NamedTuple doing nothing while the method quietly served every call.
"""
function check_hook_grouping(names)
    unknown = filter(n -> !(n in SUPPORTED_HOOKS), names)
    isempty(unknown) || error(
        "ReactantNitro.@hooks: $(join(("`$n`" for n in unknown), ", ")) is not a hook the map \
         carries, so it would sit in the group unread while the method served every call. The \
         map carries $(join(("`$h`" for h in SUPPORTED_HOOKS), ", ")); define anything else as a \
         method."
    )
    traced = filter(n -> n in TRACED_HOOKS, names)
    host = filter(n -> n in HOST_HOOKS, names)
    isempty(traced) || isempty(host) || error(
        "ReactantNitro.@hooks: $(join(("`$n`" for n in traced), ", ")) and \
         $(join(("`$n`" for n in host), ", ")) are in one group. A group is the unit of \
         invalidation, so re-evaluating it mints a new type for every hook in it: editing \
         $(join(("`$n`" for n in host), ", ")), which on its own costs NO recompile, would \
         recompile the programs holding $(join(("`$n`" for n in traced), ", ")), the gradient \
         among them. Put them in separate `@hooks` groups and `merge` the results."
    )
    resid = filter(n -> n in RESIDENCY_HOOKS, names)
    isempty(resid) && return nothing
    # NOT an error: `metrics_residency` decides this at run time and the macro cannot see it. What
    # the macro CAN do is say which residency the grouping is consistent with, so a mismatch is
    # something the user was told about rather than something they measure later.
    if isempty(traced)
        @warn "ReactantNitro.@hooks: assuming HOST residency for \
            $(join(("`$n`" for n in resid), ", ")), since this group holds no traced hook. \
            Editing it should then cost no recompile. If `metrics_residency` returns `:device` \
            for one of them it IS in a program, and every edit to this group will invalidate it."
    else
        @warn "ReactantNitro.@hooks: assuming DEVICE residency for \
            $(join(("`$n`" for n in resid), ", ")), since this group also holds \
            $(join(("`$n`" for n in traced), ", ")). Editing it recompiles either way. If \
            `metrics_residency` returns `:host` for one of them, it is in no program and this \
            grouping is costing you nothing but is also buying you nothing: move it to a host \
            group to keep its edits free."
    end
    return nothing
end

macro hooks(block)
    Meta.isexpr(block, :block) || (block = Expr(:block, block))
    entries = Any[]
    for ex in block.args
        ex isa LineNumberNode && continue
        sig = _call_signature(ex)
        sig === nothing && error(
            "ReactantNitro.@hooks: every entry must be a function definition, as in \
             `loss(e, pred; y) = ...`, and got `$ex`."
        )
        name = _basename(sig.args[1])
        name === nothing && error(
            "ReactantNitro.@hooks: could not read a hook name out of `$(sig.args[1])`."
        )
        # `let` rather than an anonymous function: it keeps the keyword declaration, which routing
        # reads with `Base.kwarg_decl`, and it keeps the name local so nothing is defined globally.
        push!(entries, Expr(:(=), name, Expr(:let, Expr(:block), Expr(:block, ex, name))))
    end
    isempty(entries) && error("ReactantNitro.@hooks: the block defines no hooks.")
    check_hook_grouping([e.args[1] for e in entries])
    return esc(Expr(:call, check_hooks, Expr(:tuple, entries...)))
end
