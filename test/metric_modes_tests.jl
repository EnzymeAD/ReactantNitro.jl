# Metric modes: how `value => mode` combines a metric's batches, under both residencies.

@testitem "metric modes" begin
    using Test
    using ReactantNitro
    using ReactantNitro: host_metrics, accumulate_metrics, reduce_metrics, cache_reset!
    using Lux, Random, Statistics

    const FV = Float32

    fold(batches; split = :val) = reduce_metrics(
        foldl(
            (acc, (m, n)) -> accumulate_metrics(acc, host_metrics(m, split, n), split),
            batches; init = nothing
        )
    )
    errmsg(f) =
    try
        f()
        ""
    catch ex
        ex isa ErrorException ? ex.msg : rethrow()
    end

    @testset "each mode, leaf by leaf" begin
        a, b = [1 2; 3 4], [10 20; 30 40]
        r = fold(
            [
                (
                    (;
                        s = a => :sum, hi = a => :max, lo = b => :min, m = 2.0 => 1,
                        nt = (; x = 1, v = [1.0, 2.0]) => :sum, c = [1 2; 3 4] => :concat,
                    ), 2,
                ),
                (
                    (;
                        s = b => :sum, hi = b => :max, lo = a => :min, m = 4.0 => 3,
                        nt = (; x = 2, v = [3.0, 4.0]) => :sum, c = reshape([5, 6], 2, 1) => :concat,
                    ), 1,
                ),
            ]
        )
        @test r.s == a .+ b
        @test r.hi == b && r.lo == a
        @test r.m == 6.0 / 4
        @test r.nt == (; x = 3, v = [4.0, 6.0])            # one mode covers a whole group
        @test r.c == [1 2 5; 3 4 6]                        # joined on the last axis
    end

    @testset "a group mixes modes, and keeps its shape" begin
        g(s, y) = (;
            rank = (; score = s => :concat, label = y => :concat, n_pos = sum(y) => :sum),
            pair = (s[1] => :max, length(s) => :sum),
        )
        r = fold([((; k = g([0.1, 0.2], [1, 0])), 2), ((; k = g([0.3], [1])), 1)])
        @test r.k.rank.score == [0.1, 0.2, 0.3] && r.k.rank.label == [1, 0, 1]   # aligned
        @test r.k.rank.n_pos == 2
        @test r.k.pair == (0.3, 3)
    end

    @testset "the older spellings mean the same" begin
        @test fold([((; k = (1, nothing)), 1), ((; k = 2 => :sum), 1)]).k == 3
        @test fold([((; k = (3.0, 2)), 1), ((; k = 1.0 => 2), 1)]).k == 1.0
    end

    @testset "refusals name the value" begin
        @test occursin("`k`", errmsg(() -> fold([((; k = 1 => :mean), 1)])))
        @test occursin(":concat", errmsg(() -> fold([((; k = 1 => :mean), 1)])))   # lists the modes
        @test occursin("same on every batch", errmsg(() -> fold([((; k = 1 => 2), 1), ((; k = 1 => :sum), 1)])))
        @test occursin("LAST axis", errmsg(() -> fold([((; k = 1.0 => :concat), 1)])))
        @test occursin("LAST axis", errmsg(() -> fold([((; k = zeros(3, 2) => :concat), 3)])))
        @test occursin(
            "Every axis but the last", errmsg(
                () -> fold([((; k = zeros(2, 1) => :concat), 1), ((; k = zeros(3, 1) => :concat), 1)])
            )
        )
        # A value with no mode, a value with two, and a group that changes shape.
        nomode = errmsg(() -> fold([((; k = (; a = 1 => :sum, b = 2)), 1)]))
        @test occursin("`k.b`", nomode) && occursin("(c ? a : b) => :sum", nomode)
        @test occursin("so does a value", errmsg(() -> fold([((; k = (; a = 1 => :sum) => :sum), 1)])))
        @test occursin("`k`", errmsg(() -> fold([((; k = 1), 1)])))
        @test occursin(
            "changed shape", errmsg(
                () -> fold([((; k = (; a = 1 => :sum)), 1), ((; k = 1 => :sum), 1)])
            )
        )
    end

    # ── end to end, both residencies, a short final batch ───────────────────────────────

    const RESIDENCY = Ref(:host)
    const PASS_THROUGH = Ref(false)

    @experiment struct ModesMLP
        width::GraphConst{Int} = 6
    end
    ReactantNitro.build_model(e::ModesMLP, rng) =
        (m = Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dense(e.width => 1)); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::ModesMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::ModesMLP, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.metrics_residency(::ModesMLP, ::Symbol) = RESIDENCY[]
    function ReactantNitro.metrics(::ModesMLP, ŷ; y)
        return (;
            r2 = (; n = length(y), sy = sum(y), syy = sum(abs2, y), sres = sum(abs2, y .- ŷ)) => :sum,
            worst = maximum(abs.(y .- ŷ)) => :max,
            best = minimum(abs.(y .- ŷ)) => :min,
            auroc = (; score = ŷ => :concat, label = y => :concat),
        )
    end
    function ReactantNitro.finalize_metrics(::ModesMLP, acc, split)
        PASS_THROUGH[] && return acc
        (; n, sy, syy, sres) = acc.r2
        (; score, label) = acc.auroc
        pos = vec(label) .> 0
        s = vec(score)
        auroc = mean(Float64(a > b) + 0.5 * (a == b) for a in s[pos], b in s[.!pos])
        return (; r2 = 1 - sres / (syy - sy^2 / n), acc.worst, acc.best, auroc, n_concat = length(s))
    end

    const BATCHES = let rng = Random.MersenneTwister(3)
        [(; x = randn(rng, FV, 4, n), y = randn(rng, FV, 1, n)) for n in (8, 8, 3)]
    end
    ReactantNitro.build_data(::ModesMLP, dist) = (; train = BATCHES, val = BATCHES)

    mk() = Nitro(ModesMLP(); checkpointer = nothing, run_dir = mktempdir())

    @testset "every mode under :host and :device agrees with a direct computation" begin
        PASS_THROUGH[] = false
        n = mk()
        ŷ = reduce(hcat, [predict(n, b) for b in BATCHES])
        y = reduce(hcat, [b.y for b in BATCHES])
        pos = vec(y) .> 0
        ref_auroc = mean(Float64(a > b) + 0.5 * (a == b) for a in vec(ŷ)[pos], b in vec(ŷ)[.!pos])
        ref_r2 = 1 - sum(abs2, y .- ŷ) / sum(abs2, y .- mean(y))

        for r in (:host, :device)
            RESIDENCY[] = r
            cache_reset!()
            v = mk()                                       # residency is read at construction
            v.ps, v.st = n.ps, n.st
            m = validate(v)
            @test m.n_concat == 19                         # 8 + 8 + 3: no padded entry joined
            @test m.auroc ≈ ref_auroc
            @test m.r2 ≈ ref_r2 rtol = 1.0e-4
            @test m.worst ≈ maximum(abs.(y .- ŷ)) rtol = 1.0e-5
            @test m.best ≈ minimum(abs.(y .- ŷ)) rtol = 1.0e-4
        end
    end

    @testset "a `:concat` array returned unreduced is refused" begin
        PASS_THROUGH[] = true
        for r in (:host, :device)
            RESIDENCY[] = r
            msg = errmsg(() -> validate(mk()))
            @test occursin("`auroc.score`", msg) && occursin("unreduced", msg)
        end
        PASS_THROUGH[] = false
    end
end
