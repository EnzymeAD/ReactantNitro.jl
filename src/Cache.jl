# Cache.jl
#
# The module-level compile cache: its key, its world guards, and the rule that decides a hit.
# Repeated `train!`, `validate` and `predict` in one session must not recompile for the same input.
# The invariant: a hit must be provably the same program, and when in doubt, miss. No cross-session
# persistence, no eviction.

"""
    ReactantNitro.CACHE

The module-level compile cache and its lock. It and the phase registry are the two process-global
structures, both lock-guarded so concurrent runs cannot corrupt each other's bookkeeping; one run
per process is the supported configuration.
"""
const CACHE = Dict{Any, Any}()
const CACHE_LOCK = ReentrantLock()

"""
    ReactantNitro.CACHE_CLOSURES

The world-closure guard: each entry's dependency closure, captured at compile time under the same
key as [`CACHE`](@ref). [`world_closure_staleness`](@ref) re-resolves it against live dispatch at
the entry points and poisons any entry whose methods moved, so a redefinition below the hooks (a
helper `forward` calls) becomes a miss for a new `Nitro`; an existing one keeps its programs and is
told so. The gradient program stores [`fwd_program`](@ref)'s closure, since the backward pass runs
inside Enzyme's interpreter, which inference cannot descend into. `LAST_WORLD_CHECKED` memoizes the
scan on the world counter.
"""
const CACHE_CLOSURES = Dict{Any, Any}()
const LAST_WORLD_CHECKED = Ref{UInt}(0)

# What the most recent entry-point staleness scan found, so `fixed_config_report` can tell a handle
# whether its OWN programs were among the poisoned. Set by `world_closure_staleness`.
const LAST_WORLD_STALENESS = Ref((; poisoned = Any[], drifted = Symbol[]))

"""
    ReactantNitro.cache_key(f, ev, args, baked) -> key

The compile cache key. The experiment-derived component is computed once per `Nitro`
(`nitro.frozen.graphconst_hash`); the argument types and shapes are read on every call, because
Reactant's `@generated` guard covers types but not shapes, which is why a ragged batch passes it and
fails inside XLA.

| Component | Why |
| --- | --- |
| Function identity | The obvious part |
| Argument types | Includes `st`'s train/eval mode, so the two paths separate for free |
| Argument shapes | Reactant's guard covers types but not shapes |
| A hash of the `GraphConst` experiment fields | These bake as trace-time constants |
| The `primary_world` of every resolved user hook | Method redefinition |
| `baked` | `train!` keywords that reach a trace as constants (`accum`) |

Method redefinition is in the key because function identity is stable across one: in a REPL with
Revise, editing `forward` and calling `train!` again would otherwise run the old program.
`primary_world` moves only when that method is redefined, where `Base.get_world_counter()` moves
on any definition. Hooks covered: `forward`, `loss`, `metrics`, `train_metrics`,
`finalize_metrics`, `param_group`, `optimizer`, plus every rule's `apply!` for the optimizer program
only. All of it is resolved at construction into `nitro.frozen`, so only a new `Nitro` observes a
redefinition (see [`frozen_dispatch`](@ref) and [`stale_hooks`](@ref)).

`accum` and `gradient_clip_norm` have no world entry: both are resolved once into the handle and
reach a trace through `baked` and through `Val`'s type parameter, so an accessor override cannot
change a program without moving a component already present, and a world entry could only fire
spuriously (it once re-bought a 500 s gradient compile for a byte-identical program). `accum`
invalidates the gradient program and the clip the optimizer program, so a clip sweep re-pays the
cheap compile only.

`Device` fields are skipped (traced inputs) and `Host` fields are skipped (not in the traced view);
the key hashes `compile_view(e)`'s `GraphConst` fields only, since hashing the view itself would
include device scalars that change every step. Every included field must hash by content, which
[`assert_graphconst_hashable`](@ref) enforces: identity hashing fails in both directions, a
harmless miss on two identical configurations and a silent hit on a value mutated in place.

Not covered: constants reached from method bodies, a `const` in the user's module, a global, a
literal changed on edit, and a custom ChainRules `rrule`. The closure guard catches a method
redefinition below the hooks; values are the one thing that still requires a REPL restart.
"""
const CACHE_HITS = Ref(0)
const CACHE_MISSES = Ref(0)

# Shapes are in the key because Reactant's `@generated` guard covers types but not shapes.
_shape(x::AbstractArray) = size(x)
_shape(x::Union{Tuple, NamedTuple}) = map(_shape, x)
_shape(x::Optimisers.Leaf) = (_shape(x.rule), _shape(x.state))
_shape(x) = nothing

"""
    ReactantNitro.graphconst_field_hash(ev) -> UInt

A hash of `compile_view(e)`'s `GraphConst` fields only. Both exclusions are needed: `compile_view`
strips `Host` and leaves `Device` in place, so hashing the view would hash device scalars that
change every step. Each field's instance is hashed, which is content-based only if its type has a
`hash` method; a struct holding a `Vector` falls through to `objectid`, and
[`assert_graphconst_hashable`](@ref) refuses that at setup.
"""
function graphconst_field_hash(ev)
    T = typeof(ev)
    df, hf = device_fields(T), host_fields(T)
    # Seeded on the type NAME: `Device` and `Host` fields carry their own type parameters, so
    # `typeof(e)` changes at every device conversion and `hash(T)` would make every step miss.
    h = hash(Base.typename(T))
    for f in fieldnames(T)
        (f in df || f in hf) && continue
        h = hash(getfield(ev, f), hash(f, h))
    end
    return h
end

"""
    ReactantNitro.assert_graphconst_hashable(e) -> nothing

Refuse at setup a `GraphConst` value that does not hash and compare by content, since the compile
cache and the resume check both rest on `hash` and `isequal` meaning "the same configuration". The
check: a value and its `deepcopy` must hash equal and be `isequal`. A `GraphConst{Vector{Int}}`
passes, since arrays hash by content; a struct containing a mutable field fails, since `Base.hash`
falls through to `objectid` for it. Left through, that costs a full recompile per handle, a spurious
resume refusal with an identical-looking diff, and, silently, reuse of a program compiled for a
value since mutated in place. Refused rather than fixed by a structural hash, which would change
every existing key and paper over a type whose author never decided what equality means.
"""
function assert_graphconst_hashable(e)
    for (f, x) in pairs(graphconst_fields(e))
        twin = try
            deepcopy(x)
        catch err
            error(
                """
                ReactantNitro: the `GraphConst` field `$f::$(typeof(x))` could not be `deepcopy`d, so \
                the framework cannot verify that it hashes by content:
                  $(sprint(showerror, err))
                A `GraphConst` value enters the compile cache key and the resume compatibility check, \
                so it must be an ordinary configuration value. Something holding a handle, a stream, \
                or a pointer belongs in `Host` or outside the experiment entirely."""
            )
        end
        ok_hash = hash(x) == hash(twin)
        ok_isequal = isequal(x, twin)
        (ok_hash && ok_isequal) && continue
        broke = ok_hash ? "`isequal` is false for" : ok_isequal ? "`hash` differs between" :
            "both `hash` and `isequal` disagree for"
        error(
            """
            ReactantNitro: the `GraphConst` field `$f::$(typeof(x))` does not hash and compare by \
            CONTENT, so it cannot key the compile cache or the resume compatibility check.

            Measured just now: $broke a value and its own `deepcopy`.

            WHY. `Base.hash` has no method for `$(typeof(x))`, so it falls through to \
            `hash(objectid(x), h)`, and `objectid` reaches a mutable field such as a `Vector` by \
            IDENTITY rather than descending into it. The vector's own content-based `hash` is never \
            called. A struct whose fields are all immutable does not have this problem, and a bare \
            `GraphConst{Vector{Int}}` field does not either: arrays hash by content, and using one \
            in a configuration is fine.

            WHAT IT WOULD COST if this were allowed through. Two identically built experiments would \
            key differently, so every `Nitro` would re-pay a full compile; a resume would be refused \
            with a diff whose two sides print the same; and, silently, mutating the value in place \
            would NOT move the key, so a program compiled for the old value would be served for the \
            new one.

            THE FIX, best first.

              1. FLATTEN THE STRUCT INTO PLAIN FIELDS ON THE EXPERIMENT, which sidesteps this \
            entirely and is the recommended answer. It also buys back what nesting costs: residency \
            is a per-FIELD property, so a nested struct takes ONE marker for its whole group and \
            nothing inside it can be `Device`, which is what makes a value sweepable and \
            schedulable without recompiling; and presets address top-level fields only, so a \
            preset has to supply the struct WHOLE and a partial one silently falls back to the \
            struct's own defaults rather than the experiment's.
              2. Make the fields immutable: `NTuple{N,Int}` instead of `Vector{Int}`, where the \
            length is fixed and small. A struct whose fields are ALL immutable needs nothing.
              3. Or write the three methods by hand, field-wise, which is the fastest way to keep \
            the struct as it is. All three, not one or two: the cache keys on `hash` and the resume \
            check compares with `isequal`, so fixing one leaves the other symptom live.

                     Base.hash(x::$(typeof(x)), h::UInt) =
                         $(join(["hash(x.$f, " for f in fieldnames(typeof(x))], ""))hash(:$(nameof(typeof(x))), h)$(")"^length(fieldnames(typeof(x))))
                     Base.:(==)(a::$(typeof(x)), b::$(typeof(x))) =
                         $(join(["a.$f == b.$f" for f in fieldnames(typeof(x))], " && "))
                     Base.isequal(a::$(typeof(x)), b::$(typeof(x))) =
                         $(join(["isequal(a.$f, b.$f)" for f in fieldnames(typeof(x))], " && "))

            Then assert it, or it comes back the next time someone adds a field: construct the \
            experiment twice and check `graphconst_field_hash` agrees, AND that an in-place mutation \
            makes it disagree. Both directions, for the reason above."""
        )
    end
    return nothing
end

"""
    ReactantNitro.method_world(f, argtypes) -> UInt

`which(f, argtypes).primary_world`, or `0` when the hook has no method. `primary_world` moves when
that method is redefined and is stable otherwise, where the global world counter moves on any
definition at the REPL. It is one of the load-bearing internals the Julia floor of 1.12 is pinned
for; `Base.get_world_counter()` is the over-firing fallback if it goes away, and
[`primary_world_available`](@ref) is the test that says so.
"""
function method_world(f, argtypes)
    hasmethod(f, argtypes) || return UInt(0)
    m = which(f, argtypes)
    return primary_world_available() ? getfield(m, :primary_world) : Base.get_world_counter()
end

"""
    ReactantNitro.primary_world_available() -> Bool

Whether `Method.primary_world` still exists. Each load-bearing internal needs a test asserting it
still behaves as documented, and this is the one for the compile cache: a `false` here means the
cache has silently degraded to the over-firing `get_world_counter` fallback.
"""
primary_world_available() = hasfield(Method, :primary_world)

"""
    ReactantNitro.hook_worlds(ev; model, ps, st, chains) -> Tuple

The `primary_world` of every resolved user hook, so that editing `forward` in a REPL and calling
`train!` again does not run the old program. Its only caller is [`frozen_dispatch`](@ref), at
construction: the chain set has to be resolved once, since `rebuild_rules` reconstructs chains
every step while only their values move, and the rules' `apply!` worlds go into `worlds_opt` only,
so a rule edit cannot re-buy the gradient compile. `accum` and `gradient_clip_norm` are absent by
design (see [`cache_key`](@ref)). A `const` redefined without touching a method is the one thing
that requires a REPL restart.
"""
function hook_worlds(ev; model = Any, ps = Any, st = Any, chains = ())
    # An instance, not a type: `metrics_residency` has to be evaluated to know whether a metric
    # hook is even in the key.
    ev isa Type && error("ReactantNitro: `hook_worlds` needs an experiment instance rather than the \
        type `$ev`. `metrics_residency` decides which metric hooks are in the compile key and must \
        be evaluated, which a type cannot dispatch.")
    E = typeof(ev)
    T(x) = x isa Type ? x : typeof(x)
    # A metric hook is in the key only when traced; a `:host` hook is part of no program.
    traced_metric(hook) = metrics_residency(ev, hook) === :device
    ws = UInt[
        method_world(forward, Tuple{E, T(model), T(ps), T(st)}),
        method_world(loss, Tuple{E, Any}),
        traced_metric(:metrics) ? method_world(metrics, Tuple{E, Any}) : UInt(0),
        traced_metric(:train_metrics) ? method_world(train_metrics, Tuple{E, Any}) : UInt(0),
        method_world(finalize_metrics, Tuple{E, Any, Any}),
        method_world(param_group, Tuple{E, Any}),
        method_world(optimizer, Tuple{E}),
    ]
    # No `accum` and no `gradient_clip_norm`: both reach a trace through components already in the
    # key, so a world entry could only fire spuriously (see `cache_key`).
    for chain in chains, r in _rules_of(chain)
        push!(ws, method_world(Optimisers.apply!, Tuple{typeof(r), Any, Any, Any}))
    end
    return Tuple(ws)
end

# `gc_hash` is supplied by a caller that already has it and recomputed only when it does not.
_gc_hash(ev, gc_hash) = gc_hash === nothing ? graphconst_field_hash(ev) : gc_hash

function cache_key(f, ev, args, baked = (); gc_hash = nothing)
    return (f, map(typeof, args), map(_shape, args), _gc_hash(ev, gc_hash), baked)
end

function cache_key(f, ev, args, baked, worlds; gc_hash = nothing)
    return (f, map(typeof, args), map(_shape, args), _gc_hash(ev, gc_hash), baked, worlds)
end

"""
    ReactantNitro.compile_cached(f, ev, args...; phase, baked) -> thunk

Look the program up and compile on a miss; the cache is populated lazily from real batches, not
declared shapes. When `nitro` is passed, the thunk is also memoized on the handle, so a handle that
has compiled a program keeps its thunk even after the module entry is poisoned: an existing `Nitro`
is a true fixed point and a stale program is served explicitly, with the report saying to rebuild.
A miss publishes the `Compiling` phase before the blocking call and restores the previous phase
after it; a hit publishes nothing. A compile that throws is wrapped by
[`compile_with_context`](@ref) and leaves the phase where it is.
"""
function compile_cached(
        f, ev, args...; phase = nothing, baked = (), worlds = nothing,
        model = Any, ps = Any, st = Any, chains = (), nitro = nothing,
        gc_hash = nitro === nothing ? nothing : nitro.frozen.graphconst_hash
    )
    w = worlds === nothing ? hook_worlds(ev; model, ps, st, chains) : worlds
    key = cache_key(f, ev, args, baked, w; gc_hash)

    # The handle-local memo: a handle that holds its programs never re-looks-up, so it can never
    # be made to recompile behind the user's back.
    programs = nitro === nothing ? nothing : nitro.programs
    if programs isa Dict && haskey(programs, key)
        CACHE_HITS[] += 1          # a memo hit IS a hit: the program exists, nothing recompiles
        return programs[key]
    end

    hit = lock(CACHE_LOCK) do
        get(CACHE, key, nothing)
    end
    if hit !== nothing
        CACHE_HITS[] += 1
        programs isa Dict && (programs[key] = hit)
        return hit
    end
    CACHE_MISSES[] += 1

    # Compiled outside the lock, so a minutes-long compile does not block every other lookup; two
    # concurrent runs may compile the same program once each, which is wasteful and correct. A miss
    # is a real compile, so the phase is published.
    resume_phase = nitro === nothing || phase === nothing ? nothing : nitro.phase
    resume_phase === nothing || set_phase!(nitro, phase)
    thunk = compile_with_context(f, args; phase)
    resume_phase === nothing || set_phase!(nitro, resume_phase)
    # Capture the dependency closure for the world-closure guard. A failed capture leaves the entry
    # unguarded rather than poisoning the run, since the hook worlds are still in the key.
    closure = try
        cf, at = _closure_target(f, args)
        world_closure(cf, at)
    catch err
        @warn "ReactantNitro: world-closure capture failed for `$(nameof(f))`; its entry will not \
            be guarded against downstream redefinitions" maxlog = 1 err
        nothing
    end
    thunk = lock(CACHE_LOCK) do
        get!(CACHE, key, thunk)
        get!(CACHE_CLOSURES, key, closure)
        thunk
    end
    programs isa Dict && (programs[key] = thunk)
    return thunk
end

"""
    ReactantNitro.compile_with_context(f, args; phase) -> thunk

An XLA compile failure surfaces as an enormous MLIR dump with no context, so every compile is
wrapped in a handler that **names the phase, the function, and the argument shapes** before
rethrowing, and truncates the dump to a bounded prefix with a pointer to the full text on disk.
"""
function compile_with_context(f, args; phase = nothing)
    try
        return Reactant.Compiler.compile(f, args)
    catch err
        io = IOBuffer()
        showerror(io, err)
        text = String(take!(io))
        path = tempname() * ".mlir.txt"
        try
            write(path, text)
        catch
            path = "(could not be written)"
        end
        head = first(text, min(length(text), MAX_DUMP_CHARS))
        error(
            """
            ReactantNitro: compiling `$(nameof(f))` failed$(phase === nothing ? "" : " during $phase").
            Argument shapes: $(map(_shape, args))
            Argument types:  $(map(typeof, args))
            Full compiler output: $path
            First $MAX_DUMP_CHARS characters follow.
            $head"""
        )
    end
end

"The bounded prefix an MLIR dump is truncated to."
const MAX_DUMP_CHARS = 2000

# ── World-closure guard ────────────────────────────────────────────────────────────

"""
    ReactantNitro.world_closure_available() -> Bool

Whether the internals the world-closure guard builds on (`Base.specialize_method`, `Base._which`,
`Core.CodeInstance.edges`) still exist. `false` means the guard has silently degraded to no
coverage, the one direction the cache invariant forbids.
"""
world_closure_available() =
    isdefined(Base, :specialize_method) && isdefined(Base, :_which) &&
    hasfield(Core.CodeInstance, :edges)

"""
    ReactantNitro.world_closure(f, argtypes) -> Vector{Tuple{Method, Type}}

The transitive closure of the methods a program depends on, from the compiler's own invalidation
edges: force inference of `f` at `argtypes`, then walk `CodeInstance.edges` from the root. Only
dispatch winners are kept: inference also records intersection edges to methods not selected, and
re-resolving those is a false positive, so each `(method, signature)` is verified with `_which` at
capture time.
"""
function world_closure(f, argtypes::Type)
    fullsig = Core.apply_type(Tuple, typeof(f), argtypes.parameters...)
    m = Base.which(f, argtypes)
    mi = Base.specialize_method(m, fullsig, Core.svec())
    Base.return_types(f, argtypes)
    seen = Base.IdSet{Core.MethodInstance}()
    stack = Core.MethodInstance[mi]
    while !isempty(stack)
        m_ = pop!(stack)
        m_ in seen && continue
        push!(seen, m_)
        isdefined(m_, :cache) || continue
        ci = getfield(m_, :cache)
        ci === nothing && continue
        for e in getfield(ci, :edges)
            e isa Core.CodeInstance || continue
            push!(stack, getfield(e, :def))
        end
    end
    world = Base.get_world_counter()
    keep = Tuple{Core.Method, Core.Type}[]
    seen_m = Base.IdSet{Core.Method}()
    for m_ in seen
        meth = m_.def::Core.Method
        meth in seen_m && continue
        push!(seen_m, meth)
        r = Base._which(m_.specTypes; world, raise = false)
        (r === nothing || r.method !== meth) && continue
        push!(keep, (meth, m_.specTypes))
    end
    return keep
end

"""
    ReactantNitro.closure_drift(closure) -> (; drift, drifted)

Re-resolve every `(method, signature)` pair against live dispatch (`Base._which` at the current
world). A pair whose unique match is no longer the captured method, or which no longer resolves
uniquely, is drifted: recompiling now would pick different dispatch. `drifted` names the moved
methods.
"""
function closure_drift(closure)
    world = Base.get_world_counter()
    drifted = Symbol[]
    for (m, sig) in closure
        r = Base._which(sig; world, raise = false)
        (r === nothing || r.method !== m) && push!(drifted, m.name)
    end
    return (; drift = !isempty(drifted), drifted)
end

"""
    ReactantNitro._closure_target(f, args) -> (f, argtypes)

Which function and argtypes an entry's closure is captured from. `grad_program` captures
[`fwd_program`](@ref)'s instead of its own: the backward pass runs inside Enzyme's interpreter,
which inference cannot descend into, so its own closure is glue with no `forward` or `loss` in it.
`fwd_program` rather than `forward` directly, because a hook takes its batch fields as keywords and
inference of the positional method stops at the kwcall shim.
"""
_closure_target(f, args) = f === grad_program ?
    (
        fwd_program,
        Core.apply_type(
            Tuple,
            map(typeof, args[1:4])...,
            typeof(args[5]),
            typeof(args[9].forward),
            typeof(hook_fn(hook_fns(args[9]), :forward, forward)),
        ),
    ) :
    (f, Core.apply_type(Tuple, map(typeof, args)...))

"""
    ReactantNitro.world_closure_staleness() -> (; poisoned, drifted)

The entry-point guard: re-resolve every cached entry's closure against live dispatch and poison
(delete) any entry whose closure moved, so a new `Nitro` recompiles against current dispatch while
an existing one keeps its thunk and is told it is stale. Memoized on the world counter, so an
unchanged counter skips the scan; `cache_reset!` resets the memo.
"""
function world_closure_staleness()
    w = Base.get_world_counter()
    w == LAST_WORLD_CHECKED[] && return (; poisoned = Any[], drifted = Symbol[])
    poisoned = Any[]
    drifted = Symbol[]
    lock(CACHE_LOCK) do
        for (key, closure) in CACHE_CLOSURES
            closure === nothing && continue
            d = closure_drift(closure)
            d.drift || continue
            append!(drifted, d.drifted)
            push!(poisoned, key)
            delete!(CACHE, key)
            delete!(CACHE_CLOSURES, key)
        end
    end
    LAST_WORLD_CHECKED[] = w
    LAST_WORLD_STALENESS[] = (; poisoned, drifted)
    if CONFIG_REPORT[] && !isempty(poisoned)
        print(
            stdout,
            "ReactantNitro: $(length(poisoned)) cached program(s) had dependency methods ",
            "redefined after compile ($(join(unique(drifted), ", "))). Entries poisoned: a new ",
            "`Nitro` recompiles against current dispatch, and an existing `Nitro` keeps its ",
            "compiled programs, now stale, and is told so in the report.\n",
        )
    end
    return (; poisoned, drifted)
end

"""
    ReactantNitro.cache_stats() -> (; hits, misses, entries)

The counter behind the acceptance check that a fixed-LR loop trains a two-layer MLP for two epochs
on CPU with **no recompile after step 1**.
"""
cache_stats() = (; hits = CACHE_HITS[], misses = CACHE_MISSES[], entries = length(CACHE))

"""
    ReactantNitro.cache_reset!() -> nothing

Empty the cache and its counters. Not part of the run path: it exists so a test can assert a miss,
and so a session that has changed something the key cannot see (a `const` in the user's module,
the cache's one documented hole) has an alternative to restarting the REPL.
"""
function cache_reset!()
    lock(CACHE_LOCK) do
        empty!(CACHE)
        empty!(CACHE_CLOSURES)
    end
    CACHE_HITS[] = 0
    CACHE_MISSES[] = 0
    LAST_WORLD_CHECKED[] = 0
    LAST_WORLD_STALENESS[] = (; poisoned = Any[], drifted = Symbol[])
    return nothing
end
