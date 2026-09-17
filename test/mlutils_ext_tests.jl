# The ReactantNitroMLUtilsExt extension: the index-addressable trait for `MLUtils.DataLoader`.
#
# The point of the extension is that the loader everybody already writes fans out over every thread
# with nothing asked of the user, so the assertions that matter are: the trait is detected, the
# indexed pass is the SAME data as the sequential pass, the per-epoch reshuffle actually happens,
# and a fanned-out run reaches the same weights as a single-producer one.
@testitem "mlutils_ext" begin
    using Test
    using ReactantNitro
    using ReactantNitro: auto_prefetch, begin_epoch!, check_batch_at, check_source_options,
        fanout_capable, planned_batch_at, prefetch_config, prefetch_source
    using Functors, Lux, MLUtils, Random, Statistics

    const ML = Float32

    @experiment struct MLUtilsMLP
        max_epochs::Host{Int} = 1
    end
    ml_chain() = Lux.Chain(Lux.Dense(3 => 4, tanh), Lux.Dense(4 => 1))
    ReactantNitro.build_model(::MLUtilsMLP, rng) = (m = ml_chain(); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::MLUtilsMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::MLUtilsMLP, ŷ; y) = mean(abs2, ŷ .- y)

    # 40 observations, batched by four, so an epoch is ten batches and the batch dimension is last
    # in both fields, which is the framework's one shape requirement.
    const ML_X = randn(Random.MersenneTwister(5), ML, 3, 40)
    const ML_Y = randn(Random.MersenneTwister(6), ML, 1, 40)
    ml_loader(; kw...) = MLUtils.DataLoader((; x = ML_X, y = ML_Y); batchsize = 4, partial = false, kw...)

    @testset "the extension loads with MLUtils" begin
        @test Base.get_extension(ReactantNitro, :ReactantNitroMLUtilsExt) !== nothing
    end

    # ── the trait, and what it resolves to ──────────────────────────────────────────

    @testset "a plain DataLoader is index-addressable, with no user effort" begin
        dl = ml_loader(; shuffle = true)
        @test fanout_capable(dl)
        # The three-argument shape: `batch_at(dl, i)` alone does NOT exist, and the capability check
        # still finds it, which is the whole reason that shape was added.
        @test !applicable(ReactantNitro.batch_at, dl, 1)
        @test hasmethod(ReactantNitro.batch_at, Tuple{typeof(dl), Integer, Any})

        cfg = prefetch_config(auto_prefetch((; train = dl)).train)
        @test cfg.path === :fanout
        @test cfg.workers == max(1, Threads.nthreads(:default))
        @test cfg.ordered
    end

    # THE ONE THAT CATCHES THE DANGEROUS BUG. A batch index used as a sample offset produces
    # perfectly well-shaped batches and a loss curve that still falls; only comparing against the
    # sequential pass sees it.
    @testset "the indexed pass is the same data as the sequential pass" begin
        @test check_batch_at(ml_loader(); values = true) === nothing
    end

    # ── the half a `batch_at` alone cannot supply ───────────────────────────────────

    @testset "shuffle draws a new permutation every epoch, not just the first" begin
        dl = ml_loader(; shuffle = true, rng = Random.MersenneTwister(11))

        epoch(plan) = [vec(planned_batch_at(dl, i, plan).y) for i in 1:length(dl)]
        e1 = epoch(begin_epoch!(dl))
        e2 = epoch(begin_epoch!(dl))

        # Without `begin_epoch!` returning a fresh plan, every epoch would replay the first one's
        # permutation, silently, with a plausible loss curve.
        @test e1 != e2
        # And each epoch is still a permutation of the whole split: nothing dropped, nothing twice.
        for e in (e1, e2)
            @test sort(vcat(e...)) == sort(vec(ML_Y))
        end
    end

    @testset "an unshuffled loader plans the same epoch every time" begin
        dl = ml_loader()
        e1 = [vec(planned_batch_at(dl, i, begin_epoch!(dl)).y) for i in 1:length(dl)]
        e2 = [vec(planned_batch_at(dl, i, begin_epoch!(dl)).y) for i in 1:length(dl)]
        @test e1 == e2
    end

    # ── the two options a prefetched pipeline cannot honour ─────────────────────────

    # Both options live inside `DataLoader`'s `iterate`, so whether they matter depends entirely on
    # whether the resolved path iterates. The fan-out asks for batch `i` and never does.
    @testset "the option checks are judged against the resolved path" begin
        fanned = prefetch_config(PrefetchIterator(ml_loader(); workers = 4))
        single = prefetch_config(PrefetchIterator(ml_loader(); workers = 1))
        @test fanned.path === :fanout
        @test single.path === :single

        @testset "`buffer = true` is refused when one producer iterates the loader" begin
            err = try
                check_source_options(ml_loader(; buffer = true), :train, single)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("buffer = true", err.msg)
            @test occursin("NoPrefetch", err.msg)
        end

        # Refusing it here would reject a configuration in which nothing can go wrong: the buffered
        # `iterate` never runs, so every batch is freshly allocated. It is still said out loud,
        # because the user asked for something they are not getting.
        @testset "`buffer = true` is ignored, and said so, when the split fans out" begin
            @test_logs (:warn, r"NO EFFECT") check_source_options(
                ml_loader(; buffer = true), :train, fanned
            )
        end

        @testset "`parallel = true` warns only when it actually runs" begin
            @test_logs (:warn, r"break ordering guarantees") check_source_options(
                ml_loader(; parallel = true), :train, single
            )
            # Silent on the fan-out path: it is ignored, and the user gets this framework's
            # producers instead, in the source's order, which is what they were asking for.
            @test_logs check_source_options(ml_loader(; parallel = true), :train, fanned)
        end

        @testset "the default loader is silent on both paths" begin
            @test_logs check_source_options(ml_loader(), :train, single)
            @test_logs check_source_options(ml_loader(), :train, fanned)
        end
    end

    # ── end to end ──────────────────────────────────────────────────────────────────

    # The payoff, and the reason ordered delivery is the default: fanning a shuffled loader out over
    # four workers must reach the SAME weights as one producer over the same permutation. Each run
    # gets its own seeded RNG so both draw the identical epoch; what differs is only how many tasks
    # built it.
    @testset "a fanned-out DataLoader trains to the same weights as one producer" begin
        seeded() = ml_loader(; shuffle = true, rng = Random.MersenneTwister(99))
        val = ml_loader()

        one = train!(
            MLUtilsMLP(); run_dir = mktempdir(),
            data = (; train = PrefetchIterator(seeded(); workers = 1), val)
        )
        many = train!(
            MLUtilsMLP(); run_dir = mktempdir(),
            data = (; train = PrefetchIterator(seeded(); workers = 4), val)
        )

        @test prefetch_config(many.data.train).path === :fanout
        @test current_step(many) == current_step(one) == length(seeded())
        for (a, b) in zip(Functors.fleaves(parameters(one)), Functors.fleaves(parameters(many)))
            @test Array(a) == Array(b)
        end
        @test validate(many).val_loss == validate(one).val_loss
    end
end
