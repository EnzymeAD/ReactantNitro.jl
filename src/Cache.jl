# Cache.jl
#
# The module-level compile cache: its key, its world guards, and the rule that decides a hit.
#
# Requirement: repeated `train!`, `validate`, `predict` in the same session must not recompile for
# the same input. A module-level cache satisfies this with zero API surface: no handle to thread and
# no ownership question, which also defers the `CompiledModel` question until real use answers what a
# handle would add.
#
# THE INVARIANT: a hit must be provably the same program; when in doubt, MISS. An over-specific key
# costs a recompile; an under-specific key runs the wrong program.
#
# No cross-session persistence, no eviction, and no verification mode.

"""
    ReactantNitro.CACHE

The module-level compile cache and its lock. The cache and the module-level phase registry are the
two process-global pieces of bookkeeping, and both are lock-guarded so concurrent `train!` calls
cannot corrupt each other. **Nothing else about concurrent runs in one process is designed for**,
and the device memory of two simultaneous runs is the user's problem: one run per process is the
supported configuration.
"""
const CACHE = Dict{Any, Any}()
const CACHE_LOCK = ReentrantLock()

"""
    ReactantNitro.CACHE_CLOSURES

The world-closure guard: each cache entry's dependency closure, captured at compile time and
keyed by the same key as [`CACHE`](@ref). [`world_closure_staleness`](@ref) re-resolves it against
live dispatch at the entry points and poisons any entry whose methods moved, so a downstream
redefinition that the hook worlds cannot see (a helper `forward` calls, a dependency method)
becomes a cache miss for a NEW `Nitro` instead of a silent stale hit. An existing `Nitro` keeps
its compiled programs, now stale, and is told so in the report: the handle-local thunk store makes
the frozen contract literal (see [`compile_cached`](@ref) and `fixed_config_report`).

The gradient program stores [`fwd_program`](@ref)'s closure, not its own: the backward pass runs
inside Enzyme's interpreter, which plain inference cannot descend into, so `grad_program`'s own
closure captures nothing but glue (measured: 143 methods, missing `forward` and `loss` entirely). Its
behavior is `forward`'s behavior, so `fwd_program`'s closure is the guard that makes a downstream
edit miss for the expensive program too.

`LAST_WORLD_CHECKED` is the memo behind [`world_closure_staleness`](@ref)'s world check: a drift is
only possible when the world counter moved since the last scan, so an unchanged counter skips the
whole re-resolution.
"""
const CACHE_CLOSURES = Dict{Any, Any}()
const LAST_WORLD_CHECKED = Ref{UInt}(0)

# What the most recent entry-point staleness scan found, so `fixed_config_report` can tell a handle
# whether its OWN programs were among the poisoned. Set by `world_closure_staleness`.
const LAST_WORLD_STALENESS = Ref((; poisoned = Any[], drifted = Symbol[]))

"""
    ReactantNitro.cache_key(f, ev, args, baked) -> key

The compile cache key. Its experiment-derived components are computed **once per `Nitro`**, into
`nitro.frozen.graphconst_hash`, and passed in as `gc_hash`; its argument types and shapes are read
per compiled-program invocation.

**That split is a requirement, not an optimization, in one direction and the reverse in the other.**
The argument half MUST be recomputed every call: Reactant's `@generated` guard covers types but not
shapes, which is precisely why a ragged batch passes the guard and then fails inside XLA, so a cache
that skipped the shape check would reintroduce that. The experiment half CANNOT change within a run,
and that is guaranteed rather than assumed: the per-step rebuild reconstructs `ev` mid-run, and
`set_device!` errors if the rebuild moves `graphconst_field_hash`.

An earlier implementation recomputed the experiment half on every `compile_cached`, which runs per
micro-batch, while this docstring already claimed it was computed once. Measured, that cost about
0.04 ms per epoch and was never worth fixing for speed; it was fixed because the contract said one
thing and the code did another, and because it bounds a real footgun. A `GraphConst`
holding a 1e6-element array hashes in 624 us, so rehashing per micro-batch is about 343 ms per
epoch. Now it is hashed once per handle.

| Component | Why |
| --- | --- |
| Function identity | The obvious part |
| Argument types | Includes `st`'s train/eval mode, so the two paths separate for free |
| Argument **shapes** | Reactant's `@generated` guard covers types but not shapes, since shape is a runtime field; that gap is why a ragged batch passes the guard and then fails inside XLA |
| A hash of the **`GraphConst`** experiment fields | These bake as trace-time constants |
| The **`primary_world` of every resolved user hook** | Method redefinition |

**Method redefinition is in the key.** Function identity is stable across a redefinition, so in a
REPL with Revise, editing `forward` and calling `train!` again would hit the cache and silently run
the old program. Since this framework is REPL-first, this is the most likely failure in practice.
**Verified:** `which(f, argtypes).primary_world` moves when that method is redefined and is
**stable** when an unrelated function is defined, while `Base.get_world_counter()` moves on any
definition anywhere, which would invalidate the cache every time a user defines anything at the REPL.
Hooks covered: `forward`, `loss`, `metrics`, `train_metrics`, `finalize_metrics`, `param_group`,
`optimizer`, **plus every rule's `apply!`, for the optimizer program only**.

**All of it is resolved at CONSTRUCTION**, into `nitro.frozen`, so an existing handle is a fixed
point and only a new `Nitro` observes a redefinition. See
[`ReactantNitro.frozen_dispatch`](@ref) for the three tuples and why they are not one, and
[`ReactantNitro.stale_hooks`](@ref) for what tells a user their handle is stale.

**`accum` and `gradient_clip_norm` are deliberately NOT covered**, though they once were. Both are
resolved once, at `Nitro` construction, into `nitro.accum` and
`nitro.gradient_clip_norm`, and the value that reaches a trace is always the stored field: `accum`
through `baked = (; accum = nitro.accum)` and the clip through `Val(nitro.gradient_clip_norm)`, whose
value lives in the argument's **type**. So an accessor override cannot reach a trace without also
moving a key component that is already present, which makes the world entry unable to catch anything
and able to fire spuriously: revising `accum(::MyExp)` on a handle built before the edit used to
recompile **both** programs, 495.6 s plus 63.1 s, to rebuild the identical `inv_n = 1/nitro.accum`.
Worse than wasted, because the recompile the user sits through is their only evidence that Revise
"worked", and it is caused by the world entry rather than by any change in the program. See
[`fixed_config_report`](@ref) for what tells them instead.

**Skip `Device` fields** (traced inputs, so their values cannot affect the program) **and skip
`Host` fields** (not in the traced view at all). **Both exclusions are needed, and one does not
follow from `compile_view`**: that view strips `Host` and leaves `Device` in place, so keying on it
would hash device scalars that change every step and **every optimizer step would miss and
recompile**. The key is computed over `compile_view(e)`'s `GraphConst` fields only, that is
`setdiff(fieldnames(typeof(e)), device_fields(typeof(e)), host_fields(typeof(e)))`.

**`train!` keywords that bake must be in the key too.** `accum` reaches the gradient program as
`inv_n = 1/N`, a host constant, so two calls in one session at different `accum` would otherwise hit
the same program and silently train at the wrong micro-gradient scale. Today that set is `accum` and
`gradient_clip_norm`; `seed` is deliberately excluded because it reaches no trace. **The two differ
in which program they invalidate**: `accum` bakes into the **gradient** program (measured 495.6 s)
and the clip into the **optimizer** program (63.1 s), so the key is naturally per program rather
than per run, and a clip sweep re-pays the cheap compile and reuses the expensive one. Compiling no
fused program is what keeps that true.

**Hashing rule:** `hash` each included field, and **refuse at setup any field that does not hash by
content** (see [`assert_graphconst_hashable`](@ref)).

!!! warning "This rule was reversed, and the earlier version of it was wrong"
    This paragraph used to read: a field whose type has no value-based `hash` "falls
    back to object identity and therefore misses on every reconstruction of the experiment, which is
    a **recompile rather than a wrong answer, and is the safe direction**. Document it; do not try to
    detect it." **The safe-direction claim is false, and it was the reason nobody looked.** Identity
    hashing fails in BOTH directions: two identical configurations miss, which is the harmless half,
    and a value **mutated in place** still hashes the same, so the cache serves a program compiled
    for the old value and the resume check passes. That is a wrong answer, silently, and it is a
    second hole beside the method-body one below. Measured, and found by a port rather than by
    reasoning about the rule.

**What the key does not cover, and must not be assumed to:** constants reached from method *bodies*,
such as a `const` in the user's module, a global, or a literal that changed on edit. The
`primary_world` component catches the common case, the method itself being edited, but a `const`
redefined without touching the method is not visible. **That is the one thing requiring a REPL
restart.**

**What the entry-point guard adds, and what still requires a restart.** A method redefinition
anywhere in the trace call tree BELOW the hooks (a helper `forward` calls, a dependency method)
moves no hook world, so the key alone would serve the stale program silently.
[`world_closure_staleness`](@ref) closes that at the entry points: each entry carries the
transitive closure of the methods it was compiled against, and a moved closure poisons the entry,
so a NEW `Nitro` recompiles against current dispatch while an existing `Nitro` keeps its programs,
now stale, and is told so in the report (the handle-local thunk store is what makes "existing" and
"new" different: only a handle that never compiled a program can miss). What still
requires a REPL restart is the same class, values rather than methods: a `const` redefined, a
global, a literal, and a custom ChainRules `rrule` (a user method the backward pass calls that
plain inference cannot see). These are the cases a method world cannot express.
"""
const CACHE_HITS = Ref(0)
const CACHE_MISSES = Ref(0)

# Shapes are in the key because Reactant's `@generated` guard covers types but not shapes, since
# shape is a runtime field; that gap is why a ragged batch passes the guard and then fails inside
# XLA.
_shape(x::AbstractArray) = size(x)
_shape(x::Union{Tuple, NamedTuple}) = map(_shape, x)
_shape(x::Optimisers.Leaf) = (_shape(x.rule), _shape(x.state))
_shape(x) = nothing

"""
    ReactantNitro.graphconst_field_hash(ev) -> UInt

A hash of `compile_view(e)`'s **`GraphConst` fields only**, that is
`setdiff(fieldnames(typeof(e)), device_fields(typeof(e)), host_fields(typeof(e)))`.

**Both exclusions are needed and neither subsumes the other.** `compile_view` strips `Host` and
leaves `Device` in place, so hashing the view itself would hash device scalars that change every
step and **every optimizer step would miss the cache and recompile**.

**Hashing rule:** `hash` each included field. This hashes the **instance of `T`**, never the
`GraphConst{T}` marker, which is a zero-field type the `@experiment` macro strips: the declared field
type is `T` and the stored value is a `T`.

That is necessary and not sufficient. Hashing the instance is
content-based only if `T` **has** a `hash` method. A config struct holding a `Vector` has none, so
`Base.hash` falls through to `hash(objectid(x), h)` and reaches the vector by identity rather than
descending into it; the vector's own content-based `hash`, one level down, is never called.
[`assert_graphconst_hashable`](@ref) refuses that at setup rather than letting it reach a compile,
and carries the full argument for refusing over hashing structurally.
"""
function graphconst_field_hash(ev)
    T = typeof(ev)
    df, hf = device_fields(T), host_fields(T)
    # The seed is the type NAME, not the type. `Device` and `Host` fields each carry their own type
    # PARAMETER, so `typeof(e)` changes whenever a Device is converted to device residency, which
    # the framework does at setup and again on every optimizer step. Seeding with `hash(T)`
    # therefore makes every step miss and recompile, which is the exact failure keying on
    # `GraphConst` fields avoids, and which the cache tests cover. `typename` is stable across those
    # parameterizations and still distinguishes two experiment types that share field names.
    h = hash(Base.typename(T))
    for f in fieldnames(T)
        (f in df || f in hf) && continue
        h = hash(getfield(ev, f), hash(f, h))
    end
    return h
end

"""
    ReactantNitro.assert_graphconst_hashable(e) -> nothing

Refuse at setup a `GraphConst` value that does **not** hash and compare by CONTENT, because every
guarantee the compile cache and the resume check make about configuration rests on `hash` and
`isequal` answering "is this the same configuration" rather than "is this the same object".

**The check is one line of semantics:** a value and its `deepcopy` must hash equal and be `isequal`.
A `deepcopy` is structurally identical and a different object, so the two agree exactly when the
answer comes from contents.

**What this catches, and what it deliberately does not.** A `GraphConst{Vector{Int}}` passes:
`hash(::AbstractArray)` is content-based, and using a vector in a configuration is an ordinary thing
to do. What fails is a `GraphConst` whose value is a **struct containing** a mutable field, because
`Base.hash` has no method for that struct and falls through to `hash(objectid(x), h)`, and
`objectid` reaches a `Vector` field by identity rather than descending into it. The perfectly good
`hash` of the vector one level down is never called.

Three symptoms follow, and the third is why this is an error rather than a warning:

  * **A full recompile per `Nitro` handle.** Two identically constructed experiments produce
    different keys, and the gradient program is measured in hundreds of seconds.
  * **A spurious resume refusal.** [`check_config_compatible`](@ref) reports a field as changed and
    prints a diff whose two sides are identical.
  * **Silent reuse of the wrong program.** Mutating the vector in place does **not** move the key,
    so the framework serves a program compiled for the old value and the resume check passes. The
    cache's contract is that the framework refuses to silently reuse a program a change it can see
    would invalidate; this is a change it cannot see, and it is a second hole beside the documented
    one.

**Why refusing beats fixing it in the framework.** A structural hash that descends into any struct
without its own `hash` method would work, and was rejected: it would change the key of every
existing experiment carrying a struct-valued `GraphConst`, so every checkpoint for those models
would refuse to resume once, and it would silently paper over a type whose author never decided what
equality means. Refusing costs three lines in the model and makes the contract visible.
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

`which(f, argtypes).primary_world`, or `0` when the hook has no method.

**Verified:** `primary_world` moves when that method is redefined and is **stable** when an
unrelated function is defined, while `Base.get_world_counter()` moves on any definition anywhere.
So the key uses the per-hook method world; the global counter would invalidate the cache every time a
user defines anything at the REPL, which for a REPL-first design is every few seconds.

`Method.primary_world` is one of the two load-bearing internals the Julia floor of 1.12 is pinned
for. The documented fallback, `Base.get_world_counter()` with its over-firing accepted, is taken
automatically if the field ever goes away, and [`primary_world_available`](@ref) is the test that
fails loudly when it does.
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

The `primary_world` of every resolved user hook. Function identity is stable across a
redefinition, so without this, editing `forward` in a REPL with Revise and calling `train!` again
would hit the cache and silently run the old program. **Since this framework is REPL-first, this is
the most likely failure in practice.**

**The `chains` loop below WAS dead code, and is not any more.** Until it was fixed, all three
call sites omitted `chains`, so the `()` default always applied and `_rules_of` was never reached;
`compile_cached`'s fallback could not rescue it either, because every call site passed `worlds`
explicitly and the fallback only fires on `nothing`. A user with a custom `Optimisers.AbstractRule`
who revised its `apply!` got a stale optimizer program with no invalidation, which is the failure
class this function exists to prevent, and two docstrings asserted the opposite.

Its caller is now [`ReactantNitro.frozen_dispatch`](@ref), and it is the **only** one, because the
chain set has to be resolved **once, at construction**: `rebuild_rules` reconstructs chains every
optimizer step, so a per-call resolution has nothing correct to hand it, while only the rules' VALUES
move per step and their types, hence their `apply!` methods, are the ones built at setup. The result
goes into `worlds_opt` and **not** into `worlds_train`, so a rule edit cannot re-buy the 495.6 s
gradient compile: `opt_program` traces no user hook, and `grad_program` contains no `apply!`.

**`accum` and `gradient_clip_norm` are absent by design.** Both are snapshotted into the `Nitro` at
construction, so neither can reach a trace except through `baked` or through `Val`'s type parameter,
both already in the key. Including them could only ever fire spuriously; [`cache_key`](@ref) carries
the measurement.

**What this does not cover, and must not be assumed to:** constants reached from method *bodies*, a
`const` in the user's module, a global, or a literal that changed on edit. The `primary_world`
component catches the common case, the method itself being edited. A `const` redefined without
touching the method is **the one thing that requires a REPL restart**.
"""
function hook_worlds(ev; model = Any, ps = Any, st = Any, chains = ())
    # An INSTANCE, not a type. Every other hook here is only RESOLVED, so a type would do, but
    # `metrics_residency` has to be EVALUATED to know whether a metric hook is even in the key,
    # and evaluating it needs something to dispatch on. Handed a type, the call would fall through
    # to the defaults and silently key on the wrong set of hooks, so it is rejected instead.
    ev isa Type && error("ReactantNitro: `hook_worlds` needs an experiment instance rather than the \
        type `$ev`. `metrics_residency` decides which metric hooks are in the compile key and must \
        be evaluated, which a type cannot dispatch.")
    E = typeof(ev)
    T(x) = x isa Type ? x : typeof(x)
    # A metric hook is in the key only when it is TRACED. A `:host` hook is part of no program, so
    # including its world would recompile `forward` every time a user edited a metric that never
    # entered the graph, which is the friction the host default for metrics exists to remove.
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
    # NO `accum` and NO `gradient_clip_norm`. Both are resolved once at `Nitro`
    # construction and read from the stored field thereafter, so the value that reaches a trace is
    # already in the key: `accum` via `baked = (; accum = nitro.accum)` and the clip via
    # `Val(nitro.gradient_clip_norm)`, whose value IS its type parameter. An accessor override
    # therefore cannot change a program without moving a component that is already present, so a
    # world entry here catches nothing and can only fire spuriously. Adding one back re-buys the
    # 495.6 s gradient recompile for a byte-identical program; if you think you need it, you need
    # `baked`. `fixed_config_report` is what tells the user their handle is stale.
    for chain in chains, r in _rules_of(chain)
        push!(ws, method_world(Optimisers.apply!, Tuple{typeof(r), Any, Any, Any}))
    end
    return Tuple(ws)
end

# `gc_hash` is the experiment-derived component, supplied by a caller that already has it and
# recomputed only when it does not. See `frozen_dispatch`'s `graphconst_hash` for why passing it is
# safe for a whole run, and why the ARGUMENT half above it must still be recomputed every call.
_gc_hash(ev, gc_hash) = gc_hash === nothing ? graphconst_field_hash(ev) : gc_hash

function cache_key(f, ev, args, baked = (); gc_hash = nothing)
    return (f, map(typeof, args), map(_shape, args), _gc_hash(ev, gc_hash), baked)
end

function cache_key(f, ev, args, baked, worlds; gc_hash = nothing)
    return (f, map(typeof, args), map(_shape, args), _gc_hash(ev, gc_hash), baked, worlds)
end

"""
    ReactantNitro.compile_cached(f, ev, args...; phase, baked) -> thunk

Look the program up and compile on a miss. Compile follows data: the cache is created after
`build_data` and populated lazily on first use, not from declared shapes. There is no
`batch_shapes` declaration.

**When `nitro` is passed, the thunk is memoized on the handle as well as the module cache.**
A handle that has already compiled a program keeps its thunk even if the module entry is later
poisoned by [`world_closure_staleness`](@ref): the run paths consult the module cache at most once
per program per handle, so an existing `Nitro` is a true fixed point and a stale program is served
explicitly, not silently, with the report telling the user to rebuild.

An XLA compile failure surfaces as an enormous MLIR dump with no context, so **every compile is
wrapped in a handler that names the phase, the function, and the argument shapes before
rethrowing**, truncating the dump to a bounded prefix with a pointer to the full text on disk.

**`nitro` is how the `Compiling` phases become real transitions.** A cache HIT is not a compile
and publishes nothing; a MISS publishes the phase before the blocking call and the previous phase
after it. This is the one place that knows which of the two happened, which is why the firing lives
here rather than at the four call sites. On a compile that throws, the phase is left where it is and
the caller's handler moves the run to `Failed`, rather than announcing a return to `TrainStepping` that
never happened.
"""
function compile_cached(
        f, ev, args...; phase = nothing, baked = (), worlds = nothing,
        model = Any, ps = Any, st = Any, chains = (), nitro = nothing,
        gc_hash = nitro === nothing ? nothing : nitro.frozen.graphconst_hash
    )
    w = worlds === nothing ? hook_worlds(ev; model, ps, st, chains) : worlds
    key = cache_key(f, ev, args, baked, w; gc_hash)

    # The handle-local memo: a `Nitro` that has already compiled a program keeps
    # ITS thunk, even if the module entry is later poisoned. This is what makes "an existing handle
    # is a fixed point" literal: the entry-point guard poisons the module cache so NEW handles
    # recompile, and a handle that already holds its programs never re-looks-up, so it can never be
    # made to recompile behind the user's back.
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

    # Compiled OUTSIDE the lock. Holding it across a compile measured in hundreds of seconds would
    # block every other lookup in the process; the cost is that two concurrent runs can compile the
    # same program once each, which is wasteful and correct. One run per process is the supported
    # configuration anyway, and the lock exists so concurrent runs cannot CORRUPT each other's
    # bookkeeping, not to make them efficient.
    #
    # This is a MISS, so a compile really is about to happen and the phase is published. The
    # `Compiling` supertype exists so a watchdog can widen its timeout here rather than kill a run
    # mid-compile, and it is only useful if the transition actually fires.
    resume_phase = nitro === nothing || phase === nothing ? nothing : nitro.phase
    resume_phase === nothing || set_phase!(nitro, phase)
    thunk = compile_with_context(f, args; phase)
    resume_phase === nothing || set_phase!(nitro, resume_phase)
    # The world-closure guard: capture the dependency closure of the program just compiled, so
    # [`world_closure_staleness`](@ref) can detect a downstream redefinition at the next entry
    # point. A failed capture leaves the entry UNGUARDED rather than poisoning the run: the guard
    # is additive and the hook worlds are still in the key, so a capture failure is a return to
    # pre-guard behavior, not a new hole. `maxlog = 1` so a session that hits it hears it once.
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

Whether the internals the world-closure guard builds on still behave: `Base.specialize_method`,
`Base._which`, and `Core.CodeInstance.edges`. Each load-bearing internal needs a test asserting it
still behaves as documented, and this is the one for the closure guard: a `false` here means the
guard has silently degraded to no coverage, which is the one direction the cache invariant forbids
(an under-specific key runs the wrong program).
"""
world_closure_available() =
    isdefined(Base, :specialize_method) && isdefined(Base, :_which) &&
    hasfield(Core.CodeInstance, :edges)

"""
    ReactantNitro.world_closure(f, argtypes) -> Vector{Tuple{Method, Type}}

The transitive closure of the methods a program depends on, taken from the compiler's own
invalidation edges: force inference of `f` at `argtypes`, then walk `CodeInstance.edges` (each
compiled method's callees) from the root. Julia records exactly this graph for its own invalidation,
so this is the same data the runtime uses, exposed through the compiler.

**Only dispatch winners are kept.** Inference also records intersection edges to methods involved
in dispatch but not selected, and re-resolving those against `Base._which` is a false positive (the
`convert(::Type{T}, ...)` family, measured at ~0.25% of a real closure). So each candidate
`(method, signature)` is verified at capture time: `_which(sig)` must return that exact method, else
the pair is dropped. What remains is the set whose redefinition provably changes what dispatch
would pick, which is exactly the set a recompile would notice.
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

Which function and argtypes an entry's closure is captured from. Everything captures its own
closure; `grad_program` captures [`fwd_program`](@ref)'s instead, at the argument types derivable
from grad's own (`ev, model, ps, st, batch, router`): the backward pass runs inside Enzyme's
interpreter, which plain inference cannot descend into, so `grad_program`'s own closure captures
nothing but glue (measured: 143 methods, missing `forward` and `loss` entirely). The gradient
program's user-code behavior IS `forward`'s, so `fwd_program`'s closure is the guard that makes a
downstream edit a miss for the expensive program too.

`fwd_program` rather than `forward` directly, because a hook takes its batch fields as KEYWORDS:
plain inference of the positional `forward` method cannot see past the kwcall shim, so `img` is
unbound and the body's callees (the helpers that matter) are never reached. `fwd_program` calls
`forward` through the routing machinery with real kwargs, which is exactly the path the trace used.
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

The entry-point guard. Re-resolves every cached entry's closure against live dispatch and
POISONS (deletes from the module cache) any entry whose closure moved, so the next `compile_cached`
on that key misses and rebuilds against current dispatch for a NEW `Nitro`. An existing `Nitro`
that already compiled the program keeps its thunk (the handle-local memo), stays frozen, and is
told it is stale by `fixed_config_report`. This is what turns a downstream redefinition the hook
worlds cannot see (a helper `forward` calls, a dependency method) from a silent stale hit into a
real miss for new handles.

The world check is memoized: a drift is only possible if the world counter moved since the last
scan, so an unchanged counter skips the whole re-resolution. Per-entry-point cost is one counter
compare; the ~25 ms re-resolution is paid only when something was actually defined. `cache_reset!`
resets the memo, so a test that redefines and re-checks always gets a fresh scan.
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
