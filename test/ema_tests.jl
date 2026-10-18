# The weights' exponential moving average: the update against a hand-computed trajectory, the two
# limits of beta, checkpoint and resume, and which weights evaluation and loading read.

@testitem "ema" begin
    using Test
    using ReactantNitro
    using ReactantNitro: cache_reset!, CACHE, fwd_program
    using Functors, Lux, Optimisers, Random, Statistics

    const RNG = Random.MersenneTwister(21)
    const DATA = [(; x = randn(RNG, Float32, 4, 8), y = randn(RNG, Float32, 3, 8)) for _ in 1:4]
    const VAL = [(; x = randn(RNG, Float32, 4, 8), y = randn(RNG, Float32, 3, 8)) for _ in 1:2]
    const ETA = 0.1f0

    flat(t) = reduce(vcat, vec.(Array.(Functors.fleaves(t))))

    @experiment struct LinEma
        beta::Host{Float64} = 0.9
        warmup::Host{Bool} = true
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::LinEma, rng) = (m = Dense(4 => 3); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::LinEma, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::LinEma, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.optimizer(::LinEma) = Optimisers.Descent
    ReactantNitro.learning_rate(::LinEma) = ETA
    ReactantNitro.ema(e::LinEma) = EMA(e.beta; warmup = e.warmup)

    function lin_grad(W, b, x, y)
        r = W * x .+ b .- y
        return 2 .* r * x' ./ length(r), 2 .* vec(sum(r; dims = 2)) ./ length(r)
    end

    handle(; beta, warmup = true) = Nitro(
        LinEma(; beta, warmup); data = (; train = DATA, val = VAL), run_dir = mktempdir(),
        checkpointer = nothing
    )

    @testset "the update matches a hand-computed trajectory, with warmup" begin
        n = handle(; beta = 0.9)
        W, b = Float64.(Array(parameters(n).weight)), Float64.(Array(parameters(n).bias))
        sW, sb = copy(W), copy(b)
        for (t, (; x, y)) in enumerate(DATA)
            gW, gb = lin_grad(W, b, Float64.(x), Float64.(y))
            W, b = W .- ETA .* gW, b .- ETA .* gb
            β = min(0.9, (1 + t) / (10 + t))
            sW, sb = β .* sW .+ (1 - β) .* W, β .* sb .+ (1 - β) .* b
        end
        train!(n)
        @test Array(n.ema_weights.weight) ≈ sW rtol = 1.0e-5
        @test Array(n.ema_weights.bias) ≈ sb rtol = 1.0e-5
        @test Array(parameters(n).weight) ≈ W rtol = 1.0e-5          # training is untouched
    end

    @testset "beta = 0 is the weights; beta = 1 is the starting weights" begin
        z = train!(handle(; beta = 0.0, warmup = false))
        @test flat(z.ema_weights) == flat(parameters(z))
        o = handle(; beta = 1.0, warmup = false)
        start = flat(parameters(o))
        train!(o)
        @test flat(o.ema_weights) == start
        @test flat(parameters(o)) != start
    end

    @testset "checkpointed and resumed: two epochs equal one plus one" begin
        straight = train!(
            Nitro(
                LinEma(; max_epochs = 2); data = (; train = DATA, val = VAL), run_dir = mktempdir()
            )
        )
        dir = mktempdir()
        train!(Nitro(LinEma(; max_epochs = 1); data = (; train = DATA, val = VAL), run_dir = dir))
        resumed = train!(
            Nitro(
                LinEma(; max_epochs = 2); data = (; train = DATA, val = VAL), run_dir = dir,
                resume = :auto
            )
        )
        @test flat(resumed.ema_weights) ≈ flat(straight.ema_weights) rtol = 1.0e-6
        @test checkpoint_info(ReactantNitro.find_latest(resumed.checkpointer, dir)).has_ema
    end

    @testset "evaluation reads the EMA by default, and `load_ema` loads it" begin
        cache_reset!()
        n = train!(handle(; beta = 0.9))
        on_ema = validate(n).val_loss
        raw = Nitro(
            LinEma(); data = (; val = VAL), run_dir = mktempdir(), weights = n,
            eval_weights = :raw
        )
        loaded = Nitro(
            LinEma(); data = (; val = VAL), run_dir = mktempdir(), weights = n,
            load_ema = true, eval_weights = :raw
        )
        @test validate(loaded).val_loss ≈ on_ema rtol = 1.0e-6
        @test !isapprox(validate(raw).val_loss, on_ema; rtol = 1.0e-6)
        # The EMA has the weights' types, so evaluating it reuses the one forward program.
        @test count(k -> k[1] === fwd_program, keys(CACHE)) == 1
    end

    @testset "refusals" begin
        @test_throws "must be in [0, 1]" EMA(1.5)
        @test_throws "has none" Nitro(
            LinEma(); data = (; val = VAL), run_dir = mktempdir(),
            ema = nothing, eval_weights = :ema
        )
        @test_throws "no `weights`" Nitro(
            LinEma(); data = (; val = VAL), run_dir = mktempdir(),
            load_ema = true
        )
    end
end
