# Precise invalidation (`TraceHook`): the trace records what it went through, and a redefinition of
# any of it is drift. Each case below is a gap in the inference closure on concrete arguments.

@testitem "trace hook" begin
    using Test
    using ReactantNitro
    using ReactantNitro: TraceHook, closure_drift, world_closure, CACHE_CLOSURES, cache_reset!,
        cache_stats, compile_cached, compile_view, world_closure_staleness
    using Reactant

    @experiment struct TraceHookExp
        n::GraphConst{Int} = 1
    end

    @eval begin
        th_helper(x) = x .* 2                                  # generic, likely inlined into th_f
        th_traced_only(x::Reactant.TracedRArray) = x .+ 1      # exists only for traced arrays
        th_split(x) = x .* 1                                   # generic: what inference selects
        th_split(x::Reactant.TracedRArray) = x .+ 0            # traced specialization (LuxLib `mean_var`)
        th_dyn(x) = x .- 1                                     # reached through runtime dispatch
        const TH_DYNAMIC = Any[th_dyn]
        th_unrelated(x) = x
        th_f(x) = sum(th_split(x) .+ th_traced_only(th_helper(x)) .+ TH_DYNAMIC[1](x))
    end

    x = Reactant.to_rarray(rand(Float32, 8))

    # Pins the inference closure's blind spots, so the motivation stays documented and a future
    # change to the closure is visible. `th_split` is called on the concrete argument, so inference
    # selects the generic method; `th_dyn` leaves no inference edge at all.
    @testset "baseline: inference closure misses traced specializations and dynamic callees" begin
        for redefine in (
                :(th_split(x::Reactant.TracedRArray) = x .+ 3),
                :(th_dyn(x) = x .- 3),
            )
            closure = world_closure(th_f, Tuple{typeof(x)})
            @eval $redefine
            @test !closure_drift(closure).drift   # documents the gap
        end
    end

    @test TraceHook.install!()
    try
        function compile_rec()
            _, rec = TraceHook.with_trace_recording() do
                Reactant.Compiler.compile(th_f, (x,))
            end
            @test rec !== nothing
            @test rec !== nothing && !isempty(rec.specs)
            return rec
        end

        @testset "clean" begin
            rec = compile_rec()
            @test !TraceHook.trace_drift(rec).drift
            # Moves the world counter without touching anything the program used.
            @eval th_unrelated(x) = x .+ 0
            @test !TraceHook.trace_drift(rec).drift
            # The cache's staleness scan dispatches to the record.
            @test !closure_drift(rec).drift
        end

        @testset "redefinition: $label" for (label, redefine) in (
                "inlined generic helper" => :(th_helper(x) = x .* 3),
                "traced-only method" => :(th_traced_only(x::Reactant.TracedRArray) = x .+ 2),
                "traced specialization of a generic" => :(th_split(x::Reactant.TracedRArray) = x .+ 5),
                "runtime dispatch" => :(th_dyn(x) = x .- 2),
            )
            rec = compile_rec()
            @eval $redefine
            @test TraceHook.trace_drift(rec).drift
        end

        @testset "more specific method added after compile" begin
            rec = compile_rec()
            @eval th_helper(x::Reactant.TracedRArray{Float32, 1}) = x .* 4
            @test TraceHook.trace_drift(rec).drift
        end

        # Through the cache: the entry is guarded by the record, and a traced specialization the
        # inference closure cannot see poisons it.
        @testset "compile_cached stores the record and the scan poisons on it" begin
            @eval th_entry(x) = sum(th_split(x))
            cache_reset!()
            compile_cached(th_entry, compile_view(TraceHookExp()), x; gc_hash = UInt(0))
            @test only(values(CACHE_CLOSURES)) isa TraceHook.TraceRecord
            @eval th_unrelated(x) = x .+ 1
            @test isempty(world_closure_staleness().poisoned)
            @eval th_split(x::Reactant.TracedRArray) = x .+ 7
            @test length(world_closure_staleness().poisoned) == 1
            compile_cached(th_entry, compile_view(TraceHookExp()), x; gc_hash = UInt(0))
            @test cache_stats().misses == 2
            cache_reset!()
        end

        # The hook only rewrites a `getfield` into a call returning the same value: the lowered
        # program must be byte-identical to one traced without it.
        @testset "program unchanged" begin
            @eval th_g_on(x) = sum(th_split(x) .+ th_helper(x)) + sum(abs2, x)
            @eval th_g_off(x) = sum(th_split(x) .+ th_helper(x)) + sum(abs2, x)
            hlo_on = string(@code_hlo optimize = false th_g_on(x))
            r_on = Float32(@jit th_g_on(x))
            TraceHook.uninstall!()
            hlo_off = string(@code_hlo optimize = false th_g_off(x))
            r_off = Float32(@jit th_g_off(x))
            strip_names(s) = replace(s, r"loc\(.*?\)" => "", r"th_g_o(n|ff)" => "F")
            @test strip_names(hlo_on) == strip_names(hlo_off)
            @test r_on == r_off
        end

        @test TraceHook.failure_reason() === nothing
    finally
        TraceHook.uninstall!()
    end
end
