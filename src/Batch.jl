# Batch.jl
#
# Batch routing and validation, all internal. The framework does not splat the whole batch into
# every hook: at setup it resolves each hook's method for the concrete experiment type, reads its
# declared keywords, and passes exactly that subset.

"""
    ReactantNitro.Router{K}

A callable that selects the batch fields in its type parameter `K`:
`Router{(:img,)}()(batch) == (; img = batch.img)`. Built once at setup. Because `K` is in the type,
the selection resolves at trace time, so the traced graph sees only the selected fields and passing
the whole batch as one `Const` costs nothing. Capturing routers in a closure is safe, since their
type is part of the cache key.
"""
struct Router{K} end

@inline (::Router{K})(batch::NamedTuple) where {K} = NamedTuple{K}(batch)

Base.keys(::Router{K}) where {K} = K

"""
    ReactantNitro.KWARG_SINK

The `Symbol("kwargs...")` that `Base.kwarg_decl` returns as the last entry for a method ending in a
keyword sink; routing rule 3 detects a sink by it. Verified on Julia 1.12.6.
"""
const KWARG_SINK = Symbol("kwargs...")

"""
    ReactantNitro.declared(f, argtypes) -> Union{Vector{Symbol}, Nothing}

The keywords declared by the method of `f` that `argtypes` resolves to, as
`Base.kwarg_decl(which(f, argtypes))`, with [`KWARG_SINK`](@ref) last for a method ending in
`kwargs...`. `kwarg_decl` is a Base internal, one reason the Julia floor is 1.12. It returns names
only, so a required keyword and a defaulted one look the same; routing rule 2 follows from that.
Returns `nothing` when `f` has no method for `argtypes`, the normal case for `metrics` and
`train_metrics`; checked with `hasmethod`, since `which` can throw two different errors.
"""
function declared(f, argtypes)
    hasmethod(f, argtypes) || return nothing
    return Base.kwarg_decl(which(f, argtypes))
end

"""
    ReactantNitro.has_sink(decl) -> Bool

Whether a declaration ends in `kwargs...`. Routing rule 3: such a hook receives the whole batch,
and none of its keywords are checked.
"""
has_sink(decl::AbstractVector{Symbol}) = !isempty(decl) && last(decl) === KWARG_SINK
has_sink(::Nothing) = false

"""
    ReactantNitro.route_keys(decl, batch_keys, hook) -> Union{Tuple{Vararg{Symbol}}, Nothing}

The batch fields a hook receives, by routing rules 1 to 3; `nothing` when `decl` is. Rule 1: a hook
receives the fields that both it declares and the batch has; a field no hook declares reaches no
hook and is not an error. Rule 2: a declared keyword missing from the batch is left to Julia, so a
required one raises `UndefKeywordError` (re-raised with context by [`call_hook`](@ref)) and a
defaulted one takes its default, which is how optional batch fields work. Rule 3: a sink receives
the whole batch. A misspelled defaulted keyword therefore takes its default without an error.
"""
function route_keys(decl, batch_keys::Tuple{Vararg{Symbol}}, hook::Symbol)
    decl === nothing && return nothing
    has_sink(decl) && return batch_keys
    return Tuple(k for k in decl if k in batch_keys)
end

"""
    ReactantNitro.resolve_routing(ev, batch; model = Any, ps = Any, st = Any,
                                  hooks = (;)) -> NamedTuple

Resolve one [`Router`](@ref) per routed hook (`forward`, `loss`, `metrics`, `train_metrics`). Called
once per run at setup, with `ev = compile_view(e)` (or its type), the type the traced call sites
use. A hook with no method gets `nothing`, which triggers the substitution for a missing `metrics`.
A function in `hooks` is resolved in place of the method, and the hook functions are returned
under `fns`. Routing is resolved from the first batch's schema, and [`check_batch_schema`](@ref)
holds later batches to it.
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

Call a hook with its routed batch fields as keywords, turning rule 2's `UndefKeywordError` into an
error naming the hook, the keyword and the batch's fields.
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

[`call_hook`](@ref) for a `:host` residency hook, which must receive host values. The caller converts
`outputs`; this converts the routed batch fields, since a loader may yield a device `y` beside host
`logits` and the hook would then fail on a GPU and pass on CPU. [`assert_host`](@ref) then checks
the batch fields and the outputs separately, so the error says which one held a device array. `e`
is exempt, since its `Device` fields are on the device by design. The conversion costs nothing for
a loader that already yields host arrays.
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

Require every routed field (every field when `routing` is `nothing`) to be an array with a
concrete element type. Otherwise a `Vector{NamedTuple}` field fails inside the tracer with
`FieldError: type UnionAll has no field parameters`, which names no field, and a host scalar is
baked in as a constant. Unrouted fields are not checked, so a `Vector{String}` of case ids can ride
in the batch; it reaches no hook and is never transferred.
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

The union of every hook's routed keys: the fields the framework validates and transfers to the
device. The framework ignores every other batch field.
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

Require batch `idx` of `split` to have the field names `expected`, those of the first batch
(routing rule 4). A different field set is a different `NamedTuple` type, so it would recompile and
would invalidate the routing resolved at setup. The error names the fields that appeared and the
ones that went missing.
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
