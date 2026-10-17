# Gradient rules: SAM against a hand-computed reference, SAM(0) against the default rule, what
# compiles, what the train line carries, and the refusals. The two-device check is in
# `multi_device_tests.jl`.

@testitem "grad rules" begin
    using Test
    using ReactantNitro
    using ReactantNitro: cache_reset!, cache_stats, CACHE, grad_program, eps_program
    using Functors, Lux, Optimisers, Random, Statistics

    const RNG = Random.MersenneTwister(11)
    const DATA = [(; x = randn(RNG, Float32, 4, 8), y = randn(RNG, Float32, 3, 8)) for _ in 1:4]
    const ETA = 0.1f0

    flat(t) = reduce(vcat, vec.(Array.(Functors.fleaves(t))))

    mutable struct GapLog
        lines::Vector{Any}
    end
    ReactantNitro.log_metrics!(l::GapLog, m; context, kwargs...) =
        (context == "train" && push!(l.lines, m); nothing)
    ReactantNitro.log_params!(::GapLog, p) = nothing
    ReactantNitro.log_tags!(::GapLog, t) = nothing
    ReactantNitro.log_other!(::GapLog, k, v) = nothing
    ReactantNitro.log_confusion!(::GapLog, m, labels; kwargs...) = nothing
    ReactantNitro.finish!(::GapLog, status) = nothing
    ReactantNitro.run_id(::GapLog) = nothing
    ReactantNitro.run_url(::GapLog) = nothing
    ReactantNitro.reattach!(::GapLog, s) = nothing

    # ── A linear model, so the SAM step has a closed form ─────────────────────────────

    @experiment struct LinSam
        rho::Host{Float32} = 0.05f0
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::LinSam, rng) = (m = Dense(4 => 3); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::LinSam, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::LinSam, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.optimizer(::LinSam) = Optimisers.Descent
    ReactantNitro.learning_rate(::LinSam) = ETA
    ReactantNitro.grad_rule(e::LinSam) = SAM(e.rho)

    # ∂/∂(W, b) of mean(abs2, W x .+ b .- y).
    function lin_grad(W, b, x, y)
        r = W * x .+ b .- y
        return 2 .* r * x' ./ length(r), 2 .* vec(sum(r; dims = 2)) ./ length(r)
    end

    # SAM by hand: ε on the weight only (`no_decay` masks the bias), the step on the mean of the
    # micro-batches' perturbed gradients.
    function sam_reference(W, b, batches, rho, accum)
        for step in Iterators.partition(batches, accum)
            gW, gb = zero(W), zero(b)
            for (; x, y) in step
                g1W, _ = lin_grad(W, b, x, y)
                εW = rho .* g1W ./ sqrt(sum(abs2, g1W))
                g2W, g2b = lin_grad(W .+ εW, b, x, y)
                gW .+= g2W ./ accum
                gb .+= g2b ./ accum
            end
            W, b = W .- ETA .* gW, b .- ETA .* gb
        end
        return W, b
    end

    @testset "SAM matches a hand-computed reference" begin
        for accum in (1, 2)
            n = Nitro(
                LinSam(); data = (; train = DATA), accum, checkpointer = nothing,
                run_dir = mktempdir()
            )
            # Read before `train!`, which donates the parameter buffers.
            W0, b0 = Float64.(Array(parameters(n).weight)), Float64.(Array(parameters(n).bias))
            batches = [(; x = Float64.(d.x), y = Float64.(d.y)) for d in DATA]
            W, b = sam_reference(W0, b0, batches, 0.05, accum)
            train!(n)
            p = parameters(n)
            @test Array(p.weight) ≈ W rtol = 1.0e-5
            @test Array(p.bias) ≈ b rtol = 1.0e-5
            # Not vacuous: plain gradient descent lands elsewhere.
            Wp, _ = sam_reference(W0, b0, batches, 0.0, accum)
            @test !isapprox(Array(p.weight), Wp; rtol = 1.0e-5)
        end
    end

    # ── Dropout, RAdam, accumulation: SAM(0) must be the default rule exactly ─────────

    @experiment struct DropSam
        rho::Host{Float32} = 0.0f0
        sam::GraphConst{Bool} = true
        max_epochs::Host{Int} = 2
    end
    ReactantNitro.build_model(::DropSam, rng) = (
        m = Chain(Dense(4 => 16, tanh), Dropout(0.5f0), Dense(16 => 3));
        (m, Lux.setup(rng, m)...)
    )
    ReactantNitro.forward(::DropSam, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::DropSam, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.optimizer(::DropSam) = Optimisers.RAdam
    ReactantNitro.grad_rule(e::DropSam) = e.sam ? SAM(e.rho) : SingleGrad()

    drop_run(; logger = nothing, schedules = (;), kw...) = train!(
        Nitro(
            DropSam(; kw...); data = (; train = DATA), accum = 2, seed = 5, logger, schedules,
            checkpointer = nothing, run_dir = mktempdir()
        )
    )

    # ── Both binding forms: Momentum and SAM both have a field named `rho` ──────────────

    const WARM = _ -> t -> 0.05f0 * min(1.0f0, t / 2)

    @experiment struct MomSam
        bound::GraphConst{Bool} = false
        max_epochs::Host{Int} = 2
    end
    ReactantNitro.build_model(::MomSam, rng) = (
        m = Chain(Dense(4 => 16, tanh), Dropout(0.5f0), Dense(16 => 3));
        (m, Lux.setup(rng, m)...)
    )
    ReactantNitro.forward(::MomSam, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::MomSam, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.optimizer(e::MomSam, group::Symbol, hp) =
        Optimisers.Momentum(hp.eta, e.bound ? hp.bind.mom : hp.rho)
    ReactantNitro.grad_rule(e::MomSam, hp) = SAM(e.bound ? hp.bind.warm : hp.rho)

    mom_run(; logger = nothing, schedules = (;), kw...) = train!(
        Nitro(
            MomSam(; kw...); data = (; train = DATA), accum = 2, seed = 5, logger, schedules,
            checkpointer = nothing, run_dir = mktempdir()
        )
    )

    @testset "SAM(0) equals SingleGrad bitwise, so pass 2 reuses the dropout masks" begin
        a = drop_run(; sam = true, rho = 0.0f0)
        s = drop_run(; sam = false)
        @test flat(parameters(a)) == flat(parameters(s))
        @test Array(states(a).layer_2.rng.seed) == Array(states(s).layer_2.rng.seed)
        @test flat(parameters(drop_run(; sam = true, rho = 0.05f0))) != flat(parameters(s))
    end

    @testset "SAM compiles the gradient program once, and rho never recompiles" begin
        cache_reset!()
        drop_run(; sam = true, rho = 0.05f0)
        @test count(k -> k[1] === grad_program, keys(CACHE)) == 1
        @test count(k -> k[1] === eps_program, keys(CACHE)) == 1
        misses = cache_stats().misses
        drop_run(; sam = true, rho = 0.2f0)
        @test cache_stats().misses == misses
    end

    @testset "a `grad.rho` schedule reaches the perturbation, without recompiling" begin
        s = drop_run(; sam = false)
        k = drop_run(; sam = true, rho = 0.05f0)
        misses = cache_stats().misses
        # The rule says 0.05; a schedule holding it at 0 must give the default rule exactly.
        z = drop_run(; sam = true, rho = 0.05f0, schedules = (; grad = (; rho = _ -> t -> 0.0f0)))
        @test flat(parameters(z)) == flat(parameters(s))
        # And a schedule holding 0.05 must give the constant rule exactly.
        c = drop_run(; sam = true, rho = 0.0f0, schedules = (; grad = (; rho = _ -> t -> 0.05f0)))
        @test flat(parameters(c)) == flat(parameters(k))
        @test cache_stats().misses == misses
        lg = GapLog(Any[])
        drop_run(; sam = true, schedules = (; grad = (; rho = _ -> t -> 0.01f0 * t)), logger = lg)
        @test [m[Symbol("schedule/grad/rho")] for m in lg.lines] ≈ 0.01 .* (1:length(lg.lines))
    end

    @testset "an unqualified `rho` resolves to the rule when it is the only match" begin
        k = drop_run(; sam = true, rho = 0.05f0)
        u = drop_run(; sam = true, rho = 0.0f0, schedules = (; rho = _ -> t -> 0.05f0))
        @test flat(parameters(u)) == flat(parameters(k))
        @test_throws "names no schedulable field" drop_run(;
            schedules = (; grad = (; m = _ -> t -> 1))
        )
    end

    @testset "an ambiguous key is refused, naming qualification and `bind`" begin
        err = try
            mom_run(; schedules = (; rho = _ -> t -> 0.9f0))
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing
        @test occursin("ambiguous", err)
        @test occursin("opt = (; rho", err) && occursin("grad = (; rho", err)
        @test occursin("bind = (; my_rho", err)
    end

    @testset "`bind`: the user places the value, and it matches the named form" begin
        named = mom_run(; schedules = (; opt = (; rho = _ -> t -> 0.8f0), grad = (; rho = WARM)))
        bound = mom_run(;
            bound = true,
            schedules = (; bind = (; mom = _ -> t -> 0.8f0, warm = WARM)),
        )
        @test flat(parameters(bound)) == flat(parameters(named))
        lg = GapLog(Any[])
        mom_run(; bound = true, logger = lg, schedules = (; bind = (; mom = 0.8f0, warm = WARM)))
        @test all(m -> haskey(m, Symbol("schedule/bind/warm")), lg.lines)
        @test !any(m -> haskey(m, Symbol("schedule/bind/mom")), lg.lines)   # a constant
    end

    @testset "a `bind` key nothing reads is a setup error" begin
        @test_throws "nothing reads it" mom_run(;
            bound = true, schedules = (; bind = (; mom = 0.8f0, warm = WARM, unused = 1.0f0))
        )
    end

    @testset "the train line carries sam/gap under SAM only" begin
        lg = GapLog(Any[])
        drop_run(; sam = true, rho = 0.05f0, logger = lg)
        @test !isempty(lg.lines)
        @test all(m -> haskey(m, Symbol("sam/gap")) && isfinite(m[Symbol("sam/gap")]), lg.lines)
        lg = GapLog(Any[])
        drop_run(; sam = false, logger = lg)
        @test !any(m -> haskey(m, Symbol("sam/gap")), lg.lines)
    end

    # ── Refusals ──────────────────────────────────────────────────────────────────────

    @experiment struct BadRule
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::BadRule, rng) = (m = Dense(4 => 3); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::BadRule, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::BadRule, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.grad_rule(::BadRule) = :sam

    @experiment struct ManualSam
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::ManualSam, rng) = (m = Dense(4 => 3); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::ManualSam, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::ManualSam, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.train_step(::ManualSam, model, ps, opt_state, st; x, y) = error("unreached")
    ReactantNitro.grad_rule(::ManualSam) = SAM(0.05f0)

    @experiment struct DeviceRho
        rho::Device{Float32} = 0.05f0
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::DeviceRho, rng) = (m = Dense(4 => 3); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::DeviceRho, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::DeviceRho, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.grad_rule(e::DeviceRho) = SAM(e.rho)

    @testset "refusals" begin
        @test_throws "not supported yet" SAM(0.1f0; m = 4)
        @test_throws "must return a `GradRule`" Nitro(
            BadRule(); data = (; train = DATA), checkpointer = nothing, run_dir = mktempdir()
        )
        @test_throws "supports `grad_rule(e) = SingleGrad()`" Nitro(
            ManualSam(); data = (; train = DATA), checkpointer = nothing, run_dir = mktempdir()
        )
        # A `Device` field in the rule would bypass the `grad` namespace.
        @test_throws "holds a device value" Nitro(
            DeviceRho(); data = (; train = DATA), checkpointer = nothing, run_dir = mktempdir()
        )
    end
end
