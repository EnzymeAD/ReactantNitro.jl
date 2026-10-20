# `monitor`: the validation metric a run is judged by, named without a split.

@testitem "monitor" begin
    using Test
    using ReactantNitro
    using ReactantNitro: write_manifest
    using Lux, Random, Statistics
    using Test: collect_test_logs

    @experiment struct MonExp
        max_epochs::Host{Int} = 2
    end
    ReactantNitro.build_model(e::MonExp, rng) = (m = Lux.Dense(2 => 1); (m, Lux.setup(rng, m)...))
    const D = [(; x = randn(Float32, 2, 4), y = randn(Float32, 1, 4)) for _ in 1:2]
    ReactantNitro.build_data(::MonExp, dist) = (; train = D, val = D, test = D)
    ReactantNitro.forward(::MonExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::MonExp, ŷ; y) = mean(abs2, ŷ .- y)

    mk(; kw...) = Nitro(MonExp(); run_dir = mktempdir(), logger = nothing, kw...)
    warned(f, pat) = any(l -> occursin(pat, string(l.message)), first(collect_test_logs(f)))

    @testset "the framework's metric is `loss` on every split" begin
        n = train!(mk(; checkpointer = nothing))
        @test keys(n.last_metrics) == (:loss,)
        @test keys(evaluate(n; split = :test)) == (:loss,)
        # The history holds both splits, so the train loss is the column named with its split.
        h = history(n)
        @test h.columns[1:4] == [:epoch, :step, :train_loss, :loss]
    end

    @testset "`:val_loss` is `:loss`, with a warning, until 1.0.0" begin
        @test warned(() -> ReactantNitro.check_monitor(:val_loss => :min), "is now `:loss`")
        @test ReactantNitro.check_monitor(:val_loss => :min) == (:loss => :min)
        es = EarlyStopping(; metric = :val_loss, mode = :min)
        @test es.metric === :loss
        @test mk(; monitor = :val_loss => :min, checkpointer = nothing).monitor == (:loss => :min)
    end

    @testset "a split-prefixed name warns at setup and in the error" begin
        @test warned(
            () -> ReactantNitro.warn_split_prefixes([("early stopping", :val_acc => :max)]),
            "carry no split"
        )
        @test !warned(() -> mk(; checkpointer = nothing), "carry no split")
        # No `metrics`, so the only key is `loss`: refused at setup, with the hint.
        @test_throws "carry no split" mk(; checkpointer = nothing, monitor = :val_acc => :max)
    end

    @testset "a manifest written before 0.2.0 ranks with the new entries" begin
        dir = mktempdir()
        n = train!(Nitro(MonExp(); run_dir = dir, logger = nothing))
        old = [merge(e, (; metric = :val_loss)) for e in read_manifest(dir)]
        write_manifest(dir, old)
        @test all(e.metric === :loss for e in read_manifest(dir))
        logs = first(
            collect_test_logs() do
                w = Nitro(MonExp(); run_dir = dir, logger = nothing, weights = :latest => :best)
                @test w.checkpoint_source == n.best_checkpoint.path
            end
        )
        @test !any(l -> occursin("other metrics", string(l.message)), logs)
    end
end
