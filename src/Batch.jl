# Batch.jl
#
# Batch routing and validation. Nothing here is exported: the module's export list is the authority
# and routing is entirely internal.
#
# THE FRAMEWORK DOES NOT SPLAT THE WHOLE BATCH INTO EVERY HOOK. At setup it resolves each hook's
# method for the concrete experiment type, reads that method's declared keywords, and passes
# exactly that subset.

"""
    ReactantNitro.Router{K}

A callable whose key set is a **type parameter**, built once at setup and captured as a trace-time
constant by the objective wrapper:

```julia
R_FWD = Router{(:img,)}()
R_FWD(batch)   # -> (; img = batch.img)
```

Because `K` is a type parameter rather than a field, the field selection resolves at trace time and
the traced graph sees only the selected fields. Passing the whole batch as one `Const` and selecting
inside the wrapper is therefore free, which is why the framework does not build a different batch
object per hook.

The routers are closure captures rather than arguments, which is correct here and not a violation
of the rule against capturing a traced input: they are genuinely trace-time constants, and the
compile-cache key already covers them, since their key sets are derived from the batch schema (in
the key as argument types) and the hook signatures (in the key as `primary_world`).
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

`Base.kwarg_decl(which(f, argtypes))`: the keywords the resolved method declares, in order.

`which` selects the method for the experiment type (keywords do not participate in dispatch, so the
positional types are enough), and for a method ending in a sink the last entry is
[`KWARG_SINK`](@ref). **Verified on Julia 1.12.6**, and `Base.kwarg_decl` is an internal: the
version floor is pinned at 1.12 for this and for the cache's `primary_world`, and the documented
fallback is to pass the whole batch and require `kwargs...`.

Note what this **cannot** tell you: a required keyword and one with a default are indistinguishable,
because it returns names only. `forward(e, model, ps, st; img, mask = nothing)` reports
`[:img, :mask]`, exactly like a signature where both are required. That single fact is why routing
rule 2 is what it is.

A hook with no method for this experiment type returns `nothing` rather than raising, because two
hooks legitimately have none: `metrics`, where the framework substitutes its own behavior, and
`train_metrics`, which defaults to `(;)`.

That case is checked with `hasmethod` rather than by catching what `which` throws, because `which`
throws **two different things**: an `ArgumentError` when the function has methods but none match, and
a `MethodError` from `invoke` when the function has none at all. `metrics` and `train_metrics` are
declared as bare `function f end` precisely so that substitution can detect their absence, so the
second case is the normal one here rather than the exotic one.
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

Routing rules 1 to 3 reduced to a key set.

  * **Rule 1**, a hook receives **the intersection of the fields it declares and the fields the batch
    has**. Fields no hook declares reach nobody and are **not** an error: a batch may legitimately
    carry `case_id` for the user's loader alone.
  * **Rule 2**, a declared keyword absent from the batch is left alone, because the framework cannot
    decide it (see [`declared`](@ref)). It routes the intersection and lets Julia decide, which it
    does correctly and for free: omitting a **required** keyword raises `UndefKeywordError` at the
    first trace, which [`call_hook`](@ref) re-raises with the context Julia lacks, while omitting a
    **defaulted** one takes the default. **Optional batch fields are therefore supported**: a `mask`
    present in one dataset and absent in another is an ordinary defaulted keyword, not a reason to
    fork the experiment or to add a sink.
  * **Rule 3**, a sink receives the whole batch.

**Residual hole, documented rather than closed:** a typo in a *defaulted* keyword name silently takes
the default. That is inherent to defaulted keywords in Julia rather than something this design
introduces, and the schedule binding report is where it becomes visible.
"""
function route_keys(decl, batch_keys::Tuple{Vararg{Symbol}}, hook::Symbol)
    decl === nothing && return nothing
    has_sink(decl) && return batch_keys
    return Tuple(k for k in decl if k in batch_keys)
end

"""
    ReactantNitro.resolve_routing(ev, batch; model = Any, ps = Any, st = Any) -> NamedTuple

Resolve one [`Router`](@ref) per routed hook, once per run at setup, never per step.

Resolution is **always against `typeof(compile_view(e))`**, because that is the type the trace site
actually calls with, so resolving against any other risks selecting a different method. Pass `ev`,
never `e`.

Four hooks are routed: `forward`, `loss`, `metrics`, and `train_metrics`. `finalize_metrics` is
host-side and takes no batch. A hook with no method for this experiment type gets `nothing`, which
is the signal the missing-`metrics` substitution keys off.

**Routing is resolved from the first batch's schema**, and routing rule 4 requires the schema to
stay uniform: a different field set is a different `NamedTuple` type, hence a different cache key,
hence a recompile, and it invalidates the resolved routing. [`check_batch_schema`](@ref) enforces
that on every later batch.
"""
function resolve_routing(ev, batch::NamedTuple; model = Any, ps = Any, st = Any, hooks = (;))
    E = ev isa Type ? ev : typeof(ev)
    bk = keys(batch)
    T(x) = x isa Type ? x : typeof(x)
    # PROTOTYPE: the function resolved here is the MAP's when it supplies one, so routing reads the
    # keywords of the function that will actually be called. Resolving the method and calling the
    # closure would route by the wrong declaration and fail on the first batch.
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
    # `fns` rides ALONG the routers, into the programs, as part of one `Const` argument. Its type
    # is therefore read by `map(typeof, args)` in `cache_key` with no change to the key, which is
    # the whole reason a hook value must be a non-capturing singleton.
    fns = NamedTuple(map(sp -> sp[1] => sp[2], specs))
    return merge(NamedTuple(routers), (; fns))
end

"""
    ReactantNitro.call_hook(f, hook::Symbol, router, batch, args...)

Invoke a routed hook, turning routing rule 2's `UndefKeywordError` into a message that names the
hook, the keyword, and the batch's fields.

Julia's own error names the keyword and nothing else, which leaves a user guessing whether they
mistyped it, whether the loader stopped emitting it, or whether routing is broken. The framework
knows all three, so it says so.
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

[`call_hook`](@ref) for a hook the framework promised **host** values to: the `:host` residency of
`metrics` and `train_metrics`. It converts both sides of the call, asserts the conversion was total,
and only then invokes the hook.

**`e` is deliberately exempt from the assertion, and it has to be.** A `Device` field is
device-resident from setup onward, and every hook receives the experiment that way by design;
asserting over it would fire on every run of every experiment that has a `Device` field, which is a
mechanism that gets switched off within a day rather than a mechanism that works. The signature names
`e` and `outputs` separately rather than taking `args...` for exactly that reason: the exemption is
then a property of the contract, not a filter someone has to remember to apply.

**Both sides, which is the fix.** The framework used to convert only the arguments it owns, the
outputs, through `host_tree`, and hand `router(batch)...` through as keywords exactly as the loader
produced them. So one call could pass a host `logits` and a device `y`, and a hook doing
`argmax(logits)` and `argmax(y)` would work on one and raise "Scalar indexing is disallowed" on the
other. On CPU neither raises, so no test could see it. The framework's standing rule settles which
way to fix it: the package handles the device-to-host transfer, so the batch fields are converted
here rather than the hook being asked to remember.

**Then [`assert_host`](@ref), separately for each side, so the message says which.** A hook receiving
a device value after this is a gap in the conversion, and the assertion names the leaf.

The transfer itself is usually free: a loader already yielding host arrays hits `_host_value`'s
identity method on every leaf and the walk rebuilds a `NamedTuple` of the same references. It costs a
real device-to-host copy exactly when the loader was handing the hook something it could not have
used, which is the case this exists for.
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

# Shared by both, so the two call paths cannot drift into saying different things about the same
# mistake. Julia's own error names the keyword and nothing else.
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

**One concrete `NamedTuple` of device-able arrays.** `NamedTuple` being a `UnionAll` is not itself a
problem: a batch built from real arrays is concrete and traces fine. **Abstract contents are.** A
batch carrying a `Vector{NamedTuple}` field fails inside the tracer with

    FieldError: type UnionAll has no field `parameters`

which names neither the field nor the batch, so the framework validates before tracing and says what
is actually wrong. The same check catches host scalars riding in a batch.

**Only ROUTED fields are validated**, which follows from routing rule 1 rather than relaxing it: a
field no hook declares reaches nobody, is never transferred to device, and so cannot break a trace.
That is what makes the `case_id` rule 1 explicitly permits actually legal, since a `Vector{String}`
of case identifiers is neither concretely device-able nor ever looked at. Rule 1 and the validation
rule are both stated without saying which fields the validation covers; this is the only reading
under which both hold.
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
        # PROTOTYPE COST, stated where it bites: the hook map rides inside this NamedTuple, so the
        # one place that iterates it has to know that not every entry is a `Router`. That is the
        # leak from piggybacking rather than giving the map its own field, and this is the funnel
        # every other iteration goes through, so it is the only place that pays.
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
