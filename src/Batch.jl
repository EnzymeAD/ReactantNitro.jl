# Batch.jl
#
# Batch routing and validation, all internal. The framework does not splat the whole batch into
# every hook: at setup it resolves each hook's method for the concrete experiment type, reads its
# declared keywords, and passes exactly that subset.

"""
    ReactantNitro.Router{K}

A callable whose key set is a type parameter, built once at setup and captured as a trace-time
constant: `Router{(:img,)}()(batch) == (; img = batch.img)`. The selection resolves at trace time,
so the traced graph sees only the selected fields and passing the whole batch as one `Const` is
free. The routers are closure captures, which is correct because they are genuine trace-time
constants the cache key already covers.
"""
struct Router{K} end

@inline (::Router{K})(batch::NamedTuple) where {K} = NamedTuple{K}(batch)

Base.keys(::Router{K}) where {K} = K

"""
    ReactantNitro.KWARG_SINK

The literal `Symbol("kwargs...")` that `Base.kwarg_decl` returns as the **last** entry for a method
ending in a keyword sink. That is how routing rule 3 detects one; verified on Julia 1.12.6.
"""
const KWARG_SINK = Symbol("kwargs...")

"""
    ReactantNitro.declared(f, argtypes) -> Vector{Symbol}

`Base.kwarg_decl(which(f, argtypes))`: the keywords the resolved method declares, with
[`KWARG_SINK`](@ref) last for a method ending in `kwargs...`. An internal, which is one reason the
Julia floor is 1.12. It returns names only, so a required keyword and a defaulted one are
indistinguishable, which is why routing rule 2 is what it is. A hook with no method returns
`nothing` (checked with `hasmethod`, since `which` throws two different things), which is the
normal case for `metrics` and `train_metrics`.
"""
function declared(f, argtypes)
    hasmethod(f, argtypes) || return nothing
    return Base.kwarg_decl(which(f, argtypes))
end

"""
    ReactantNitro.has_sink(decl) -> Bool

Routing rule 3: a hook whose declaration ends in `kwargs...` receives the **whole batch**, and
nothing is checked for that hook, which is the trade for the flexibility.
"""
has_sink(decl::AbstractVector{Symbol}) = !isempty(decl) && last(decl) === KWARG_SINK
has_sink(::Nothing) = false

"""
    ReactantNitro.route_keys(decl, batch_keys, hook) -> Tuple{Vararg{Symbol}}

Routing rules 1 to 3 reduced to a key set. Rule 1: a hook receives the intersection of the fields
it declares and the fields the batch has; a field no hook declares reaches nobody and is not an
error. Rule 2: a declared keyword absent from the batch is left to Julia, so a required one raises
`UndefKeywordError` (re-raised with context by [`call_hook`](@ref)) and a defaulted one takes its
default, which is how optional batch fields work. Rule 3: a sink receives the whole batch. A typo
in a defaulted keyword silently takes the default, which is inherent to Julia.
"""
function route_keys(decl, batch_keys::Tuple{Vararg{Symbol}}, hook::Symbol)
    decl === nothing && return nothing
    has_sink(decl) && return batch_keys
    return Tuple(k for k in decl if k in batch_keys)
end

"""
    ReactantNitro.resolve_routing(ev, batch; model = Any, ps = Any, st = Any) -> NamedTuple

Resolve one [`Router`](@ref) per routed hook (`forward`, `loss`, `metrics`, `train_metrics`), once
per run at setup, always against `typeof(compile_view(e))`, the type the trace site calls with. A
hook with no method gets `nothing`, the signal the missing-`metrics` substitution keys off. Routing
is resolved from the first batch's schema, and [`check_batch_schema`](@ref) holds later batches to
it.
"""
function resolve_routing(ev, batch::NamedTuple; model = Any, ps = Any, st = Any, hooks = (;))
    E = ev isa Type ? ev : typeof(ev)
    bk = keys(batch)
    T(x) = x isa Type ? x : typeof(x)
    # PROTOTYPE: the function resolved is the map's when it supplies one, so routing reads the
    # keywords of the function that will be called.
    specs = (
        (:forward, hook_fn(hooks, :forward, forward), Tuple{E, T(model), T(ps), T(st)}),
        (:loss, hook_fn(hooks, :loss, loss), Tuple{E, Any}),
        (:metrics, hook_fn(hooks, :metrics, metrics), Tuple{E, Any}),
        (:train_metrics, hook_fn(hooks, :train_metrics, train_metrics), Tuple{E, Any}),
    )
    routers = map(specs) do (hook, f, argtypes)
        ks = route_keys(declared(f, argtypes), bk, hook)
        return hook => (ks === nothing ? nothing : Router{ks}())
    end
    # `fns` rides along the routers into the programs as part of one `Const` argument, so its type
    # is read by `cache_key` for free; hence a hook value must be a non-capturing singleton.
    fns = NamedTuple(map(sp -> sp[1] => sp[2], specs))
    return merge(NamedTuple(routers), (; fns))
end

"""
    ReactantNitro.call_hook(f, hook::Symbol, router, batch, args...)

Invoke a routed hook, turning rule 2's `UndefKeywordError` into a message naming the hook, the
keyword and the batch's fields.
"""
function call_hook(f, hook::Symbol, router::Router, batch::NamedTuple, args...)
    try
        return f(args...; router(batch)...)
    catch err
        err isa UndefKeywordError || rethrow()
        _missing_kw_error(hook, err, batch, router)
    end
end

"""
    ReactantNitro.call_host_hook(f, hook::Symbol, router, batch, e, outputs)

[`call_hook`](@ref) for a hook the framework promised host values to (`:host` residency). Both
sides of the call are converted, the outputs and the routed batch fields, since a loader may hand
over a device `y` beside a host `logits` and the hook would raise on a GPU and pass on CPU; then
[`assert_host`](@ref) runs on each side separately so the message says which. `e` is exempt, since
its `Device` fields are device-resident by design. The transfer is free for a loader already
yielding host arrays.
"""
function call_host_hook(f, hook::Symbol, router::Router, batch::NamedTuple, e, outputs)
    kw = host_tree(router(batch))
    assert_host(kw, "the batch fields routed to `$hook`")
    assert_host(outputs, "the outputs passed to `$hook`")
    try
        return f(e, outputs; kw...)
    catch err
        err isa UndefKeywordError || rethrow()
        _missing_kw_error(hook, err, batch, router)
    end
end

# Shared by both call paths. Julia's own error names the keyword and nothing else.
@noinline function _missing_kw_error(hook::Symbol, err, batch::NamedTuple, router::Router)
    return error(
        """
        ReactantNitro: `$hook` requires keyword `$(err.var)`, which the batch does not provide.
        The batch's fields are $(keys(batch)), and `$hook` was routed $(keys(router)).
        Either the loader should emit `$(err.var)`, or `$hook` should give it a default, which
        makes it an optional batch field."""
    )
end

# ── Batch validation ────────────────────────────────────────────────────────────────

"""
    ReactantNitro.validate_batch(batch, routing) -> nothing

Every routed field must be a concretely-typed array. A `Vector{NamedTuple}` field fails inside the
tracer with `FieldError: type UnionAll has no field parameters`, naming nothing, and a host scalar
in a batch bakes as a constant. Only routed fields are validated, which is what makes a
`Vector{String}` of case ids in the batch legal: it reaches nobody and is never transferred.
"""
function validate_batch(batch::NamedTuple, routing = nothing)
    checked = routing === nothing ? keys(batch) : routed_fields(routing)
    for k in checked
        v = getproperty(batch, k)
        if v isa AbstractArray
            isconcretetype(eltype(v)) && continue
            error(
                """
                ReactantNitro: batch field `$k` has abstract element type `$(eltype(v))`; batch
                fields must be concretely-typed arrays. Reaching the tracer, this
                fails with `FieldError: type UnionAll has no field parameters`, which names neither
                the field nor the batch.
                Build `$k` as a concrete array, or drop it from the batch: a field no hook declares
                reaches nobody and is not validated."""
            )
        else
            error(
                """
                ReactantNitro: batch field `$k` is a `$(typeof(v))`, not an array. Batch fields
                must be concretely-typed arrays; a host scalar riding in a batch bakes as a
                trace-time constant instead of being a traced input.
                If it is a per-run constant, make it a `Device` field on the experiment. If
                it is loader bookkeeping, leave it in the batch but declare it in no hook, and it
                will reach nobody and not be validated."""
            )
        end
    end
    return nothing
end

"""
    ReactantNitro.routed_fields(routing) -> Tuple{Vararg{Symbol}}

The union of every hook's routed keys: exactly the fields the framework transfers to device and
validates. Everything else in the batch belongs to the user's loader alone.
"""
function routed_fields(routing::NamedTuple)
    ks = Symbol[]
    for r in routing
        # The hook map rides inside this NamedTuple, so this one funnel skips the non-`Router`.
        r isa Router || continue
        for k in keys(r)
            k in ks || push!(ks, k)
        end
    end
    return Tuple(ks)
end

"""
    ReactantNitro.check_batch_schema(batch, expected::Tuple, split, idx) -> nothing

Routing rule 4. **Keep the batch schema uniform across batches**: a different field set is a
different `NamedTuple` type, hence a different cache key, hence a recompile, and it invalidates the
routing resolved at setup. The framework checks each batch's field names against the first and
errors on a change, naming what appeared and what went missing.
"""
function check_batch_schema(batch::NamedTuple, expected::Tuple, split::Symbol, idx::Integer)
    got = keys(batch)
    got === expected && return nothing
    added, missed = setdiff(got, expected), setdiff(expected, got)
    error(
        """
        ReactantNitro: the `$split` split changed its batch schema at batch $idx.
        Setup resolved routing against $(expected), and this batch has $(got).
        $(isempty(added) ? "" : "Appeared: $(Tuple(added)). ")$(isempty(missed) ? "" : "Missing: $(Tuple(missed)).")
        A different field set is a different NamedTuple type, so it is a different compile-cache
        key and a recompile, and it invalidates the routers resolved at setup. Emit the same
        fields every batch, using a defaulted keyword for one that is genuinely optional across
        DATASETS rather than across batches."""
    )
end
