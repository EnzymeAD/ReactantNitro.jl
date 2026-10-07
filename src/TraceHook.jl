# TraceHook.jl
#
# Precise invalidation tracking for compiled Reactant programs, without patching Reactant.
#
# Every traced call goes through the generated function `Reactant.call_with_reactant`, whose
# generator is `Reactant.call_llvm_generator`. Instead of replacing that 515-line generator, this
# file wraps it: the wrapper calls the original, then rewrites ONE existing statement in the
# returned CodeInfo (the `getfield` that loads the callee) into a call that returns the same
# value and also records the specialization. Nothing is inserted, so SSA numbering, statement
# flags, SSA types and debug info stay untouched. Installing swaps a single reference
# (`Method.generator`), so no method is overwritten and nothing is invalidated.
#
# The wrapper is an OpaqueClosure, not a function. Julia 1.12 runs a generator in the world of the
# generated method's definition (here, the world Reactant was loaded in) and freezes it there:
# `invokelatest` and `invoke_in_world` are no-ops inside a generator. Every method this package
# defines is too new to be dispatched from that world. An OpaqueClosure is called through a builtin
# rather than dispatch and runs in the world it was created in, so creating it in `install!` makes
# this file's methods reachable.
#
# Relied on (Reactant 0.2.285, Julia 1.12), each checked at install or at expansion time:
#   - `Method.generator` is a mutable, non-atomic field
#   - the runtime calls a generator as `gen(world, source, sparams..., argtypes...)`
#   - both generator paths set `code_info.edges = Any[mi, ...]`
#   - both paths load the callee with `getfield(SlotNumber(2), k)` before calling anything
#   - the Reactant path starts with `Expr(:meta, :force_compile)`, the native path does not
# Any mismatch disables precise mode instead of producing a record that only looks complete.
#
# Known limit: wrappers expanded before `install!` (Reactant's precompile workload traces
# `sin` on a scalar and `sum` on a vector) keep their original code and do not record.

module TraceHook

import Reactant

const CC = Core.Compiler

# Recording

const _KEY = :_reactant_nitro_trace_recording
const Rec = Tuple{Core.MethodInstance, Bool}   # (specialization, compiled by the native interpreter)

_stack() = get!(() -> Set{Rec}[], task_local_storage(), _KEY)::Vector{Set{Rec}}

# Replaces the generated wrapper's `getfield(args, k)`: the same value, plus a record.
Base.@constprop :aggressive @inline function _record_getfield(
        mi::Core.MethodInstance, native::Bool, t, k::Int
    )
    stack = _stack()
    isempty(stack) || push!(last(stack), (mi, native))
    return getfield(t, k)
end

# The generator wrapper

const _ORIGINAL = Ref{Any}(nothing)
const _INSTALLED = Ref{Any}(nothing)   # the OpaqueClosure now in `Method.generator`
const _FAILED = Threads.Atomic{Bool}(false)
const _FAIL_REASON = Ref{String}("")

function _fail!(reason::String)
    _FAILED[] = true
    _FAIL_REASON[] = reason
    return nothing
end

function _nitro_generator(world::UInt, source, @nospecialize(xs...))
    code = _ORIGINAL[](world, source, xs...)
    code isa Core.CodeInfo || return code
    _FAILED[] || _instrument!(code)
    return code
end

# Created at install, so it runs in a world where `_nitro_generator` exists (see the header).
_make_generator() = Base.Experimental.@opaque (w, s, xs...) -> _nitro_generator(w, s, xs...)

_is_getfield(f) = f isa GlobalRef && f.mod === Core && f.name === :getfield

function _instrument!(code::Core.CodeInfo)
    edges = code.edges
    # A `MethodError` stub has no edges and calls nothing: nothing to record.
    (edges === nothing || isempty(edges)) && return nothing
    mi = first(edges)
    mi isa Core.MethodInstance ||
        return _fail!("the wrapper's first edge is a $(typeof(mi)), expected a MethodInstance")
    stmts = code.code
    first_stmt = isempty(stmts) ? nothing : stmts[1]
    native = !(first_stmt isa Expr && first_stmt.head === :meta && :force_compile in first_stmt.args)
    for i in eachindex(stmts)
        st = stmts[i]
        st isa Expr && st.head === :call && length(st.args) == 3 || continue
        _is_getfield(st.args[1]) && st.args[2] == Core.SlotNumber(2) && st.args[3] isa Int || continue
        stmts[i] = Expr(
            :call, _record_getfield, QuoteNode(mi), native, Core.SlotNumber(2), st.args[3]
        )
        return nothing
    end
    return _fail!("no `getfield(args, k)` statement found in a generated wrapper")
end

# Installation

function _refuse(reason::String)
    @warn "ReactantNitro: precise invalidation unavailable: $reason" maxlog = 1
    return false
end

_cwr_method() = only(methods(Reactant.call_with_reactant))

"""
    TraceHook.install!() -> Bool

Wrap `call_with_reactant`'s generator. Call from `__init__`, before anything is traced. Returns
`false`, with one warning, when Reactant's structure is not the one this file was written for.
"""
function install!()
    _ORIGINAL[] === nothing || return true
    isdefined(Reactant, :call_with_reactant) && isdefined(Reactant, :call_llvm_generator) ||
        return _refuse("Reactant no longer defines `call_with_reactant` and `call_llvm_generator`")
    ms = methods(Reactant.call_with_reactant)
    length(ms) == 1 ||
        return _refuse("expected one method of `call_with_reactant`, found $(length(ms))")
    m = first(ms)
    gen = getfield(m, :generator)
    gen === Reactant.call_llvm_generator ||
        return _refuse("`call_with_reactant`'s generator is a $(typeof(gen)), not `call_llvm_generator`")
    oc = _make_generator()
    _ORIGINAL[] = gen
    try
        setfield!(m, :generator, oc)
    catch err
        _ORIGINAL[] = nothing
        return _refuse("could not replace the generator: $(sprint(showerror, err))")
    end
    _INSTALLED[] = oc
    return true
end

"Restore the original generator. For tests; expansions made while installed keep recording."
function uninstall!()
    _ORIGINAL[] === nothing && return nothing
    m = _cwr_method()
    getfield(m, :generator) === _INSTALLED[] && setfield!(m, :generator, _ORIGINAL[])
    _ORIGINAL[] = nothing
    _INSTALLED[] = nothing
    return nothing
end

precise_available() = _ORIGINAL[] !== nothing && !_FAILED[]

"Why precise mode turned itself off this session, or `nothing`."
failure_reason() = _FAILED[] ? _FAIL_REASON[] : nothing

# Records

struct TracedSpec
    mi::Core.MethodInstance
    native::Bool
    # The CodeInstance compiled for this specialization, when one exists: Julia lowers its
    # `max_world` when anything it depends on changes, including callees inlined into it,
    # which never pass through a wrapper of their own.
    ci::Union{Nothing, Core.CodeInstance}
end

struct TraceRecord
    world::UInt
    specs::Vector{TracedSpec}
end

# The CodeInstance the trace used: the one valid at the world the trace ran in. If that world is
# already behind the latest one, its `max_world` is finite from the start, which `trace_drift`
# reports as drift: the program was compiled against definitions that have since changed.
function _valid_ci(mi::Core.MethodInstance, owner, world::UInt)
    isdefined(mi, :cache) || return nothing
    ci = getfield(mi, :cache)
    while ci isa Core.CodeInstance
        getfield(ci, :owner) === owner && getfield(ci, :min_world) <= world <= getfield(ci, :max_world) &&
            return ci
        ci = isdefined(ci, :next) ? getfield(ci, :next) : nothing
    end
    return nothing
end

"""
    TraceHook.with_trace_recording(f) -> (result, record)

Run `f()`, typically a Reactant compile, and return what the trace went through. `record` is
`nothing` when precise mode is unavailable, so the caller can fall back to its own closure.
"""
function with_trace_recording(f)
    precise_available() || return (f(), nothing)
    # The world the trace runs in, which is the calling task's, not the latest: code that `@eval`s
    # a definition and then compiles in the same function traces against the definitions before it.
    world = Base.tls_world_age()
    buf = Set{Rec}()
    push!(_stack(), buf)
    result = try
        f()
    finally
        pop!(_stack())
    end
    # An expansion during this very compile may have failed to instrument. The warning is issued
    # here rather than inside the generator, where logging is not safe to run.
    if !precise_available()
        @warn "ReactantNitro: precise invalidation disabled for this session: $(_FAIL_REASON[])" maxlog = 1
        return (result, nothing)
    end
    reactant_owner = CC.cache_owner(Reactant.ReactantInterpreter(; world))
    specs = TracedSpec[
        TracedSpec(mi, native, _valid_ci(mi, native ? nothing : reactant_owner, world))
            for (mi, native) in buf
    ]
    # Without its CodeInstance a specialization cannot vouch for the callees inlined into it, so
    # this compile falls back to the inference closure rather than storing a partial record.
    any(s -> s.ci === nothing, specs) && return (result, nothing)
    return (result, TraceRecord(world, specs))
end

# Invalidation

function _spec_invalidated(s::TracedSpec, trace_world::UInt, world::UInt, overlay)
    # Something this specialization depends on changed, including an inlined callee.
    s.ci !== nothing && getfield(s.ci, :max_world) != typemax(UInt) && return true
    def = s.mi.def
    def isa Method || return true
    # Abstract-signature frames are re-derived from their concrete callers.
    CC.isdispatchtuple(s.mi.specTypes) || return false
    # Redefined in place, as Revise does on 1.12.
    getfield(def, :primary_world) > trace_world && return true
    # Replaced, deleted, or shadowed by a more specific method (an extension loaded later).
    # Looked up in the same table the tracer used: the overlay table for the Reactant path.
    m = Base._which(
        s.mi.specTypes; method_table = s.native ? nothing : overlay, world, raise = false
    )
    return m === nothing || m.method !== def
end

_spec_name(s::TracedSpec) = (d = s.mi.def; d isa Method ? d.name : :toplevel)

# When drift was seen only through a specialization's CodeInstance, the method that changed is a
# callee inlined into it, possibly several calls down: each CodeInstance's edges name its direct
# callees only. Walk down through the callees whose own CodeInstance (same owner, the one valid at
# the trace world) was invalidated too, and report the methods that no longer resolve to the one
# compiled against. Names only: drift is already established. Empty when none is found; the caller
# then reports the specialization.
function _changed_callees(ci::Core.CodeInstance, trace_world::UInt, world::UInt)
    owner = getfield(ci, :owner)
    names = Symbol[]
    seen = Base.IdSet{Core.MethodInstance}()
    stack = Core.CodeInstance[ci]
    while !isempty(stack)
        for e in getfield(pop!(stack), :edges)
            mi = e isa Core.CodeInstance ? getfield(e, :def) : e
            mi isa Core.MethodInstance && !(mi in seen) || continue
            push!(seen, mi)
            def = mi.def
            def isa Method || continue
            moved = getfield(def, :primary_world) > trace_world
            if !moved && CC.isdispatchtuple(mi.specTypes)
                m = Base._which(mi.specTypes; world, raise = false)
                moved = m === nothing || m.method !== def
            end
            if moved
                push!(names, def.name)
                continue
            end
            callee = _valid_ci(mi, owner, trace_world)
            callee === nothing || getfield(callee, :max_world) == typemax(UInt) || push!(stack, callee)
        end
    end
    return unique!(names)
end

"""
    TraceHook.trace_drift(record) -> (; drift, drifted)

The same contract as `closure_drift`, so the cache's staleness scan can use either.
"""
function trace_drift(rec::TraceRecord)
    world = Base.get_world_counter()
    drifted = Symbol[]
    world == rec.world && return (; drift = false, drifted)
    overlay = CC.method_table(Reactant.ReactantInterpreter(; world))
    for s in rec.specs
        _spec_invalidated(s, rec.world, world, overlay) || continue
        callees = s.ci === nothing ? Symbol[] : _changed_callees(s.ci, rec.world, world)
        isempty(callees) ? push!(drifted, _spec_name(s)) : append!(drifted, callees)
    end
    return (; drift = !isempty(drifted), drifted)
end

end # module
