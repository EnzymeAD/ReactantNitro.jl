# `loss` returning `(l, terms)`: the terms logged as `loss/<term>` on every split, and `eval_loss`.

@testitem "loss terms" begin
    using Test
    using ReactantNitro
    using Lux, Random, Statistics

    @experiment struct TermExp
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(e::TermExp, rng) = (m = Lux.Dense(2 => 1); (m, Lux.setup(rng, m)...))
    const D = [(; x = randn(Float32, 2, 4), y = randn(Float32, 1, 4)) for _ in 1:2]
    ReactantNitro.build_data(::TermExp, dist) = (; train = D, val = D)
    ReactantNitro.forward(::TermExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    function ReactantNitro.loss(::TermExp, ŷ; y)
        fit, size_ = mean(abs2, ŷ .- y), 0.1f0 * mean(abs2, ŷ)
        return fit + size_, (; fit, size_)
    end

    struct TLog
        lines::Vector{Any}
    end
    ReactantNitro.log_metrics!(l::TLog, m; context, kwargs...) = push!(l.lines, (; context, m))
    ReactantNitro.log_params!(::TLog, p) = nothing
    ReactantNitro.log_other!(::TLog, k, v) = nothing
    ReactantNitro.finish!(::TLog, s) = nothing
    ReactantNitro.run_id(::TLog) = nothing
    ReactantNitro.run_url(::TLog) = nothing

    mk(T = TermExp; kw...) = Nitro(T(); run_dir = mktempdir(), checkpointer = nothing, kw...)
    k(name) = Symbol("loss/", name)

    @testset "the train line carries each term, from the gradient program" begin
        lg = TLog([])
        train!(mk(; logger = lg))
        tl = [x.m for x in lg.lines if x.context == "train"]
        @test !isempty(tl)
        # Summed in Float32 on the device, compared in Float64: one Float32 ulp apart at most.
        @test all(m -> isapprox(m[k(:fit)] + m[k(:size_)], m.loss; rtol = 1.0e-6), tl)
    end

    @testset "with no `metrics`, validation reports the loss and its terms" begin
        n = train!(mk())
        @test keys(n.last_metrics) == (:loss, k(:fit), k(:size_))
        @test n.last_metrics[k(:fit)] + n.last_metrics[k(:size_)] ≈ n.last_metrics.loss
    end

    @experiment struct TermMetrics
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::TermMetrics, rng) = ReactantNitro.build_model(TermExp(), rng)
    ReactantNitro.build_data(::TermMetrics, dist) = ReactantNitro.build_data(TermExp(), dist)
    ReactantNitro.forward(::TermMetrics, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::TermMetrics, ŷ; y) = ReactantNitro.loss(TermExp(), ŷ; y)
    ReactantNitro.metrics(::TermMetrics, ŷ; y) = (; mae = sum(abs, ŷ .- y) => length(y))

    @testset "with `metrics`, `eval_loss` adds the loss beside them" begin
        off = train!(mk(TermMetrics; monitor = :mae => :min))
        @test keys(off.last_metrics) == (:mae,)
        on = train!(mk(TermMetrics; eval_loss = true))
        @test keys(on.last_metrics) == (:mae, :loss, k(:fit), k(:size_))
        @test on.monitor == (:loss => :min)
    end

    @testset "with `metrics` and no `eval_loss`, nothing ranks without a declared monitor" begin
        err = try
            Nitro(TermMetrics(); run_dir = mktempdir())
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("monitor", err.msg) && occursin("eval_loss", err.msg)
        @test mk(TermMetrics; checkpointer = Checkpointer(; keep = Last(1))).monitor === nothing
        @test Nitro(TermMetrics(); run_dir = mktempdir(), monitor = :mae => :min).monitor == (:mae => :min)
        @test_throws "monitor" mk(TermMetrics; early_stop = EarlyStopping())
    end

    @experiment struct TermDup
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::TermDup, rng) = ReactantNitro.build_model(TermExp(), rng)
    ReactantNitro.build_data(::TermDup, dist) = ReactantNitro.build_data(TermExp(), dist)
    ReactantNitro.forward(::TermDup, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::TermDup, ŷ; y) = ReactantNitro.loss(TermExp(), ŷ; y)
    ReactantNitro.metrics(::TermDup, ŷ; y) = (; loss = sum(abs2, ŷ .- y) => length(y))

    @testset "a key both sources report raises, naming both" begin
        @test_throws "reported by both" train!(mk(TermDup; eval_loss = true))
        # Off, the user's own `loss` is simply theirs.
        @test keys(train!(mk(TermDup; monitor = :loss => :min)).last_metrics) == (:loss,)
    end

    @testset "terms are flat scalars" begin
        @test ReactantNitro.split_loss(1.0f0) == (1.0f0, (;))
        @test ReactantNitro.split_loss((1.0f0, (; a = 2.0f0))) == (1.0f0, (; a = 2.0f0))
        @test_throws "scalar" ReactantNitro.split_loss((1.0f0, (; a = (; b = 1.0f0))))
        @test_throws "`(l, terms)`" ReactantNitro.split_loss((1.0f0, 2.0f0, 3.0f0))
    end
end
