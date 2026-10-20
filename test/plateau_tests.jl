# `Plateau`: schedules scaled down when a validation metric stops improving.

@testitem "plateau" begin
    using Test
    using ReactantNitro
    using ReactantNitro: cache_reset!, cache_stats, step_plateau!
    using Lux, Optimisers, Random, Statistics

    const F = Float32

    # The scale after each epoch's `step_plateau!`.
    # Driven by hand, so each `Plateau` below names its metric: setup is what fills in `monitor`.
    function scales(p::Plateau, vs)
        s = p(100)
        return [(step_plateau!(s, v); s.scale) for v in vs]
    end

    @testset "torch's ReduceLROnPlateau sequence" begin
        # Hand-traced through `torch.optim.lr_scheduler.ReduceLROnPlateau.step` (torch 2.x source)
        # with lr = 1, factor = 0.5, patience = 1, cooldown = 1, threshold = 0.1 (rel), min_lr = 0.2.
        vs = [1.0, 0.95, 0.85, 0.85, 0.8, 0.79, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7, 0.7]
        torch = [1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.2, 0.2, 0.2, 0.2]
        p = Plateau(1.0; metric = :m, mode = :min, factor = 0.5, patience = 1, cooldown = 1, threshold = 0.1, min_scale = 0.2)
        @test scales(p, vs) == torch
    end

    const m = (; metric = :m, mode = :min)

    @testset "patience, cooldown and the floor" begin
        # `patience = 2`: the third bad epoch reduces, as in torch.
        @test scales(Plateau(1.0; m..., patience = 2, factor = 0.5), [1, 1, 1, 1, 1]) == [1, 1, 1, 0.5, 0.5]
        # `cooldown = 2` holds the count at zero for two epochs after a reduction.
        @test scales(Plateau(1.0; m..., patience = 0, cooldown = 2, factor = 0.5), [1, 1, 1, 1, 1]) ==
            [1, 0.5, 0.5, 0.5, 0.25]
        @test scales(Plateau(1.0; m..., patience = 0, factor = 0.1, min_scale = 0.05), [1, 1, 1, 1]) ==
            [1, 0.1, 0.05, 0.05]
    end

    @testset "relative and absolute thresholds" begin
        vs = [0.5, 0.54, 0.56, 0.62, 0.6]
        kw = (; metric = :m, mode = :max, patience = 0, factor = 0.1, threshold = 0.05)
        # :abs improves past best + 0.05; :rel past best * 1.05.
        @test scales(Plateau(1.0; kw..., threshold_mode = :abs), vs) ≈ [1, 0.1, 0.1, 0.1, 0.01]
        @test scales(Plateau(1.0; kw..., threshold_mode = :rel), vs) ≈ [1, 1, 0.1, 0.1, 0.01]
    end

    @testset "the value is inner(step) * scale, in the inner's type" begin
        s = Plateau(total -> t -> F(t / total); m..., patience = 0, factor = 0.5)(10)
        @test s(4) === 0.4f0
        step_plateau!(s, 1.0)
        step_plateau!(s, 1.0)
        @test s(4) === 0.2f0
        @test_throws "monitor" step_plateau!(Plateau(1.0)(10), 1.0)
    end

    # ── in a run ────────────────────────────────────────────────────────────────────

    # `score` is whatever the next entry of `SCORES` says, so each epoch's metric is scripted.
    const SCORES = Float64[]
    const NEXT = Ref(1)
    script!(vs; from = 1) = (empty!(SCORES); append!(SCORES, vs); NEXT[] = from)

    @experiment struct PlatExp
        aux::Device{Float32} = 1.0f0
        max_epochs::Host{Int} = 3
    end
    plat_mlp() = Lux.Chain(;
        backbone = Lux.Dense(4 => 8, tanh), head = Lux.Dense(8 => 3)
    )
    plat_data(n; seed) = (
        rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, F, 4, 8), y = randn(rng, F, 3, 8)) for _ in 1:n]
    )
    const PLAT_TRAIN = plat_data(4; seed = 1)
    const PLAT_VAL = plat_data(1; seed = 2)
    ReactantNitro.param_group(::PlatExp, ks) = ks[1] === :backbone ? :backbone : :default
    ReactantNitro.learning_rate(::PlatExp) = 1.0f-3
    ReactantNitro.learning_rate(::PlatExp, ::Val{:backbone}) = 1.0f-4
    ReactantNitro.build_model(e::PlatExp, rng) = (m = plat_mlp(); (m, Lux.setup(rng, m)...))
    ReactantNitro.build_data(::PlatExp, dist) = (; train = PLAT_TRAIN, val = PLAT_VAL)
    ReactantNitro.forward(::PlatExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(e::PlatExp, ŷ; y) = e.aux * mean(abs2, ŷ .- y)
    ReactantNitro.metrics(::PlatExp, ŷ; y) = (; score = 0.0f0 => :sum)
    function ReactantNitro.finalize_metrics(::PlatExp, acc, split)
        v = SCORES[NEXT[]]
        NEXT[] += 1
        return (; score = v)
    end

    struct PlatLog
        lines::Vector{Any}
    end
    ReactantNitro.log_metrics!(l::PlatLog, m; step, epoch, context, kwargs...) =
        push!(l.lines, (; context, step, epoch, m))
    ReactantNitro.log_params!(::PlatLog, p) = nothing
    ReactantNitro.log_other!(::PlatLog, k, v) = nothing
    ReactantNitro.finish!(::PlatLog, s) = nothing
    ReactantNitro.run_id(::PlatLog) = nothing
    ReactantNitro.run_url(::PlatLog) = nothing

    ReactantNitro.monitor(::PlatExp) = :score => :min
    mk(; kw...) = Nitro(PlatExp(); checkpointer = nothing, run_dir = mktempdir(), kw...)
    sk(name) = Symbol("schedule/", name)
    reduce_every = (; patience = 0, factor = 0.5)

    @testset "a bare opt key: every step of the next epoch is scaled" begin
        script!([1, 2, 3])
        lg = PlatLog([])
        n = mk(; logger = lg, schedules = (; eta = Plateau(_ -> t -> 1.0f-2 * t; reduce_every...)))
        @test occursin("on plateau of score", sprint(show, MIME"text/plain"(), n))
        train!(n)
        tl = [x for x in lg.lines if x.context == "train"]
        @test [x.step for x in tl] == 1:12
        # Epoch 2 is a bad epoch, so epoch 3 runs at half; the backbone keeps its 0.1 ratio.
        scale(t) = t <= 8 ? 1 : 0.5
        @test all(x -> x.m[sk("default/eta")] ≈ 1.0f-2 * x.step * scale(x.step), tl)
        @test all(x -> x.m[sk("backbone/eta")] ≈ 1.0f-3 * x.step * scale(x.step), tl)
        vl = [x.m[sk("plateau/eta")] for x in lg.lines if haskey(x.m, sk("plateau/eta"))]
        @test vl == [1, 0.5, 0.25]
        @test n.schedules.plateaus[1].second.scale == 0.25
    end

    @testset "a group path key leaves the other groups alone" begin
        script!([1, 2, 3])
        n = mk(; schedules = (; opt = (; backbone = (; eta = Plateau(1.0f-3; reduce_every...)))))
        train!(n)
        @test Float32(n.opt_state[1].rule.eta) == 1.0f-3                 # :default, the accessor
        @test Float32(n.opt_state[2].rule.eta) ≈ 1.0f-3 * 0.1f0 * 0.5f0  # epoch 3's scale
    end

    @testset "a device key" begin
        script!([1, 2, 3])
        n = mk(; schedules = (; aux = Plateau(2.0f0; reduce_every...)))
        train!(n)
        @test device_value(n, :aux) == 1.0f0
        @test only(n.schedules.plateaus).first === Symbol("device.aux")
    end

    @testset "a reduction recompiles nothing" begin
        misses(vs, patience) = begin
            cache_reset!()
            script!(vs)
            train!(mk(; schedules = (; eta = Plateau(1.0f-3; patience))))
            cache_stats().misses
        end
        @test misses([1, 2, 3], 0) == misses([1, 2, 3], 100)
    end

    @testset "resume continues the sequence; weights starts fresh" begin
        p = Plateau(1.0f-3; patience = 1, factor = 0.5)
        vs = [1, 2, 3, 4, 5]
        script!(vs)
        full = train!(mk(; max_epochs = 5, schedules = (; eta = p)))
        whole = full.schedules.plateaus[1].second

        dir = mktempdir()
        script!(vs)
        ck = Checkpointer()
        first3 = train!(
            Nitro(PlatExp(); run_dir = dir, max_epochs = 3, checkpointer = ck, schedules = (; eta = p))
        )
        # A mid-run record holds the counters after its own epoch's update.
        e2 = only(e for e in read_manifest(dir) if e.epoch == 2)
        @test load_checkpoint(ck, joinpath(dir, e2.file)).schedule_state.eta.bad_epochs == 1
        script!(vs; from = 4)
        r = train!(
            Nitro(
                PlatExp(); run_dir = dir, max_epochs = 5, resume = :auto, checkpointer = ck,
                schedules = (; eta = p)
            )
        )
        s = r.schedules.plateaus[1].second
        @test (s.best, s.bad_epochs, s.cooldown_left, s.scale) ==
            (whole.best, whole.bad_epochs, whole.cooldown_left, whole.scale)
        @test Float32(r.opt_state[1].rule.eta) == Float32(full.opt_state[1].rule.eta)

        w = Nitro(
            PlatExp(); run_dir = dir, weights = checkpoint_run(first3) => :latest,
            checkpointer = ck, schedules = (; eta = p)
        )
        @test w.schedules.plateaus[1].second.scale == 1.0
        @test w.schedules.plateaus[1].second.best === nothing
    end

    @testset "everything follows the monitor unless it has its own" begin
        n = mk(;
            checkpointer = Checkpointer(), early_stop = EarlyStopping(),
            schedules = (; eta = Plateau(1.0f-3), aux = Plateau(1.0f0; metric = :score, mode = :max))
        )
        @test (n.checkpointer.metric, n.checkpointer.mode) == (:score, :min)
        @test (n.early_stop.metric, n.early_stop.mode) == (:score, :min)
        ps = Dict(n.schedules.plateaus)
        @test (ps[:eta].p.metric, ps[:eta].p.mode) == (:score, :min)
        @test (ps[Symbol("device.aux")].p.metric, ps[Symbol("device.aux")].p.mode) == (:score, :max)
        txt = sprint(show, MIME"text/plain"(), n)
        @test occursin("followed by checkpointer, early stopping, plateau `eta`", txt)
        @test occursin("own: plateau `device.aux` on score (max)", txt)
        # The keyword wins, and a reused checkpointer follows the newest handle.
        ck = Checkpointer()
        mk(; checkpointer = ck, monitor = :score => :max)
        @test ck.mode === :max
        @test_throws "metric => mode" mk(; monitor = :score)
        @test_throws "alone" Plateau(1.0f-3; metric = :score)
    end

    @testset "setup refusals" begin
        bad(; kw...) = Plateau(1.0f-3; kw...)
        @test_throws "mode" Plateau(1.0f-3; metric = :score, mode = :lower)
        @test_throws "factor" mk(; schedules = (; eta = bad(; factor = 1.0)))
        @test_throws "factor" mk(; schedules = (; eta = bad(; factor = 0)))
        @test_throws "threshold_mode" mk(; schedules = (; eta = bad(; threshold_mode = :pct)))
        @test_throws "val" mk(;
            data = (; train = PLAT_TRAIN), schedules = (; eta = bad())
        )
        # A custom metric missing from `metrics` raises at the first epoch.
        script!([1, 2, 3])
        n = mk(; schedules = (; eta = Plateau(1.0f-3; metric = :nope, mode = :min)))
        @test_throws "does not emit" train!(n)
    end
end

@testitem "plateau without metrics" begin
    using Test
    using ReactantNitro
    using Lux, Random, Statistics

    @experiment struct PlainExp
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(e::PlainExp, rng) = (m = Lux.Dense(2 => 1); (m, Lux.setup(rng, m)...))
    const D = [(; x = randn(Float32, 2, 4), y = randn(Float32, 1, 4)) for _ in 1:2]
    ReactantNitro.build_data(::PlainExp, dist) = (; train = D, val = D)
    ReactantNitro.forward(::PlainExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::PlainExp, ŷ; y) = mean(abs2, ŷ .- y)

    mk(p) = Nitro(PlainExp(); checkpointer = nothing, run_dir = mktempdir(), schedules = (; eta = p))
    # Only `:loss` exists without `metrics`, and setup knows it.
    @test_throws "loss" mk(Plateau(1.0f-3; metric = :acc, mode = :max))
    @test_throws "loss" Nitro(
        PlainExp(); checkpointer = nothing, run_dir = mktempdir(), monitor = :acc => :max
    )
    @test phase(train!(mk(Plateau(1.0f-3)))) isa Done
end
