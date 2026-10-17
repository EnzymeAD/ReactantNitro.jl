# Hooks.jl
#
# PROTOTYPE. Hooks supplied as a NamedTuple of functions rather than as methods, for reactive
# notebooks: a method mutates a global method table no dependency graph can see, while a value is
# named and downstream of what built it, and `merge` gives a second hook set that coexists with the
# first.
#
# The one rule: a hook function may not capture. The map travels into the programs as a `Const`
# argument whose type is read by `cache_key`; a non-capturing closure is a singleton type, so the
# key separates two hooks for free, while two capturing closures over different values share a
# type and the cache would serve one program for two computations. A value a hook needs belongs on
# `e`, where its marker states its residency and hashes it into the key.

"""
    ReactantNitro.hook_fn(hooks, name, fallback) -> Function

Look up hook `name`: the function in `hooks` if present, otherwise `fallback`, the generic
function that dispatches to methods. Every call site goes through this, so a map entry always takes
precedence over the method.
"""
hook_fn(hooks, name::Symbol, fallback) = get(hooks, name, fallback)

"""
    ReactantNitro.hook_fns(routing) -> NamedTuple

The hook map stored in a routing under `fns`, or `(;)` when there is none, as for a `nothing`
routing on the eval path or a routing tuple the test suite builds by hand.
"""
hook_fns(routing::NamedTuple) = haskey(routing, :fns) ? routing.fns : (;)
hook_fns(::Any) = (;)

"""
    ReactantNitro.check_hooks(hooks) -> hooks

Return `hooks` after rejecting a map the compile cache cannot key: one that is not a `NamedTuple`,
has a non-function entry, or has a closure that captures. A captured value does not enter the cache
key, so the cache could serve a program compiled for a different value.
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
        # `Base.issingletontype`, not `isempty(fieldnames(...))`: Julia lowers a keyword method to a
        # wrapper holding the body function, which is code identity rather than runtime data.
        # `hook_predicate_available` asserts the predicate still means that.
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

Whether `Base.issingletontype` still tells a non-capturing keyword hook from a capturing one. The
predicate relies on how Base lowers closures, so it has a named check with a test, as
`primary_world_available` does.
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

# ── The grouping rules ──────────────────────────────────────────────────────────────
#
# Re-evaluating a group mints a fresh type for every hook in it, so a group is only as cheap to
# edit as its most expensive member: `build_data` edited beside `forward` costs the gradient
# compile, and edited alone costs nothing.

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

Check one `@hooks` group's hook names at macro expansion.

Errors on a name the map does not support, which would sit in the map unused while the method
served every call, and on a group mixing a traced hook (`forward`, `loss`) with a host one
(`build_model`, `build_data`, `finalize_metrics`). For `metrics` or `train_metrics`, warns which
residency the grouping assumes, since `metrics_residency` is only known at run time.
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
    # Not an error: `metrics_residency` decides at run time; the macro can only say which residency
    # the grouping is consistent with.
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

Each definition becomes a value rather than a method: it is wrapped in a `let`, so nothing is
added to any method table and each hook has a singleton function type. Supported names are
`forward`, `loss`, `metrics`, `train_metrics`, `build_model`, `build_data` and `finalize_metrics`.
Re-evaluating a group gives every hook in it a new type, so [`check_hook_grouping`](@ref) refuses a
group mixing a traced hook with a host hook, and warns about the residency a group assumes for
`metrics` or `train_metrics`. A definition may reference globals; one that captures a local is
refused by [`check_hooks`](@ref).
"""
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
