# Defaults audit tests.
#
# The acceptance criterion this file exercises: "Defaults audit: every row of the default tables
# exercised together. Done when green: an experiment defining only the four required hooks
# trains, validates, and checkpoints."
#
# THE AUDIT IS THE POINT, AND THE BARE-RUN TEST IS ITS ACCEPTANCE. The failure this file exists to
# catch is not a wrong default, it is two right ones that cannot both be taken: `metrics`
# defaulting to nothing and a checkpointer defaulting to `metric = :val_loss` were each defensible
# and jointly unusable, which
# is how the `TopKCheckpointer(metric)` gap arose. So the row-by-row testsets below are the cheap
# half, and the last two testsets, which run the whole chain with NOTHING PASSED, are the point.
#
# EARLY STOPPING IS DELIBERATELY OUT OF SCOPE, and it has been got wrong once. An earlier version
# of the bare-run test required an early stop, which forced it to pass an `early_stop` and
# therefore to stop exercising the defaults alone, defeating its own purpose. The bare-run test
# asserts the run does NOT early-stop. Early stopping is covered by the phase system tests in
# `lifecycle_tests.jl`.

@testitem "defaults" begin
    using Test
    using ReactantNitro
    using ReactantNitro: check_checkpointer, clip_source, default_run_dir, is_framework_default,
        read_manifest
    using JSON3, Lux, Optimisers, Random, Reactant, Statistics

    # The checkpoint manifest's `file` is the only identity a checkpoint has, and its name
    # carries a metric VALUE, which belongs to this fixture's loss rather than to the design.
    # So a test that wants epoch 1's file asks the manifest for it.
    BARE_CKPT1() =
        joinpath(BARE_DIR, only(e.file for e in read_manifest(BARE_DIR) if e.epoch == 1))

    # THE BARE EXPERIMENT: no fields, and no hook beyond the four required ones. Both halves are
    # load-bearing. A field would be configuration this experiment is supposed not to have, and a fifth
    # hook would be one default not taken, which is exactly the row that would then go unexercised.
    @experiment struct BareMLP end

    bare_chain() = Lux.Chain(Lux.Dense(3 => 4, tanh), Lux.Dense(4 => 1))
    ReactantNitro.build_model(::BareMLP, rng) = (m = bare_chain(); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::BareMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::BareMLP, ŷ; y) = mean(abs2, ŷ .- y)

    bare_batches(n; seed) = (
        rng = Random.MersenneTwister(seed);
        [
            (; x = randn(rng, Float32, 3, 4), y = randn(rng, Float32, 1, 4))
                for _ in 1:n
        ]
    )
    const BARE_TRAIN = bare_batches(3; seed = 1)
    const BARE_VAL = bare_batches(2; seed = 2)
    ReactantNitro.build_data(::BareMLP, dist) = (; train = BARE_TRAIN, val = BARE_VAL)

    # By default, `run_dir` defaults to `joinpath("runs", string(nameof(typeof(e))))`, RELATIVE TO
    # THE WORKING DIRECTORY, and the default checkpointer writes there. A bare-run test that passed
    # a `run_dir` would no longer be a test of the defaults, so this file asserts on that path and
    # cleans it up instead. Cleaning
    # BEFORE as well as after is not tidiness: the resume testset below asks for `resume = :auto`,
    # so a directory left by an earlier suite run would change what it is measuring.
    const BARE_DIR = default_run_dir(BareMLP())
    clean_bare_dir() = (
        rm(BARE_DIR; recursive = true, force = true);
        isdir("runs") && isempty(readdir("runs")) && rm("runs"); nothing
    )
    clean_bare_dir()

    # ── The macro's fieldless case, which is what "bare" means ──────────────────────────

    @testset "an experiment declaring no fields is constructible" begin
        # Found during the defaults audit and fixed there: `@experiment` emitted its `@kwdef`-style
        # constructor unconditionally, and with no fields that expression is `BareMLP(; ) =
        # BareMLP()`, which REPLACES Julia's own zero-argument constructor with one that calls
        # itself. `BareMLP()` was a `StackOverflowError` at 80,000 frames naming nothing but
        # `BareMLP()`. Everything below this line is unreachable without the fix, and a fieldless
        # experiment is not an exotic case: it is what the bare-run test is about.
        @test BareMLP() isa BareMLP
        @test isempty(fieldnames(BareMLP))
        @test config_metadata(BareMLP) == NamedTuple()
        @test device_fields(BareMLP) == ()
        @test host_fields(BareMLP) == ()
        # There is nothing to strip, so the view is the experiment itself and the tracer sees one
        # object.
        @test compile_view(BareMLP()) === BareMLP()
    end

    # ── the user-hook default table ──────────────────────────────────────────────────────

    @testset "the hook default table: every row's default, on a bare experiment" begin
        e = BareMLP()

        @testset "the four required hooks are the four this experiment defines" begin
            # The hook table marks exactly these four required, and the file above defines exactly
            # these four. `metrics` having no method is what the metrics substitution keys on, so it
            # is asserted rather than assumed: a stray catch-all method anywhere in the suite would
            # silently move the run onto a different path and this testset is where that shows up.
            @test hasmethod(ReactantNitro.build_data, Tuple{BareMLP, Any})
            @test hasmethod(ReactantNitro.build_model, Tuple{BareMLP, Any})
            @test !hasmethod(ReactantNitro.metrics, Tuple{BareMLP, Any})
            @test !hasmethod(ReactantNitro.train_metrics, Tuple{BareMLP, Any})
        end

        @testset "metrics, finalize_metrics, derive, residency" begin
            @test ReactantNitro.finalize_metrics(e, (; err = 0.5), :val) == (; err = 0.5)
            @test ReactantNitro.derive(e, BARE_TRAIN) == (;)
            # Per hook, and the default follows the cadence rather than the residency.
            @test ReactantNitro.metrics_residency(e, :metrics) === :host
            @test ReactantNitro.metrics_residency(e, :train_metrics) === :device
        end

        @testset "the optimizer rows" begin
            @test ReactantNitro.param_group(e, (:layer_1, :weight)) === :default
            @test ReactantNitro.optimizer(e) === Optimisers.RAdam
            @test ReactantNitro.optimizer(e, Val(:default)) === Optimisers.RAdam
            @test ReactantNitro.learning_rate(e) === 1.0f-3
            @test ReactantNitro.learning_rate(e, Val(:default)) === ReactantNitro.learning_rate(e)
            # `0` everywhere by default is what keeps `Decay` out of the chain entirely.
            @test ReactantNitro.lambda(e) == 0
            @test ReactantNitro.lambda(e, Val(:backbone)) == ReactantNitro.lambda(e)
            @test ReactantNitro.decay_anchor(e, Val(:default)) === :zero
            # `0f0` means off, and the docs are explicit that a traced threshold could not keep
            # that meaning.
            @test ReactantNitro.gradient_clip_norm(e) === 0.0f0
            @test ReactantNitro.nonschedulable(Optimisers.Descent) == ()
        end

        @testset "the driver rows" begin
            # A first run finishing after one epoch reads as a bug, which is why the default table
            # states it and why the bare-run test asserts it below.
            @test ReactantNitro.max_epochs(e) == 1
            @test ReactantNitro.schedules(e) == (;)
            @test ReactantNitro.dispatch_variant((; kind = :bit_r50), :kind) === Val(:bit_r50)
        end

        @testset "each of these really is the FRAMEWORK's method, not one this file defined" begin
            # The rows above would pass identically against a method defined on `BareMLP`, so the
            # audit asserts provenance too: `is_framework_default` is what the binding report
            # already uses to tell a user method from the shipped one.
            for (f, argtypes) in (
                    (ReactantNitro.learning_rate, Tuple{BareMLP}),
                    (ReactantNitro.lambda, Tuple{BareMLP}),
                    (ReactantNitro.optimizer, Tuple{BareMLP}),
                    (ReactantNitro.max_epochs, Tuple{BareMLP}),
                    (ReactantNitro.schedules, Tuple{BareMLP}),
                    (ReactantNitro.gradient_clip_norm, Tuple{BareMLP}),
                    (ReactantNitro.derive, Tuple{BareMLP, Any}),
                    (ReactantNitro.finalize_metrics, Tuple{BareMLP, Any, Symbol}),
                )
                @test is_framework_default(f, argtypes)
            end
        end
    end

    # ── the `Nitro`-keyword default table ────────────────────────────────────────────────

    @testset "the keyword default table: `Nitro(e)` with nothing passed resolves every row" begin
        # Setup only, no training: the point of this testset is the RESOLUTION, and the bare-run
        # test below is the point about the run. The default logger writes its params line at step
        # 12, so construction
        # is not write-free; clean_bare_dir() below handles the file. Constructing in the suite's own
        # working directory is deliberate, since `run_dir`'s default is relative to it and that is a
        # row under audit.
        n = Nitro(BareMLP())

        @testset "the ten that default to an accessor of the same name" begin
            @test n.seed == 42
            @test run_dir(n) == BARE_DIR == joinpath("runs", "BareMLP")
            @test n.accum == 1
            @test n.max_epochs == 1
            @test n.gradient_clip_norm === 0.0f0
            # The shipped JSON default, already adopted to the default run dir, and `nothing`
            # is the documented opt-out rather than the default.
            @test n.logger isa JSONLogger
            @test n.logger.path == joinpath(BARE_DIR, "metrics.jsonl")
            @test n.early_stop === nothing
            @test isempty(n.schedules)                 # `schedules(e)` is `(;)`, so nothing varies
            @test n.checkpointer isa TopKCheckpointer
            @test (n.checkpointer.k, n.checkpointer.metric, n.checkpointer.mode) == (3, :val_loss, :min)
            # By default, `n_devs = 1` skips the mesh entirely rather than building a one-device
            # one.
            @test n.mesh === nothing
        end

        @testset "the four that stay keyword-only" begin
            # `data = nothing` means call `build_data`, which is what supplied these two splits.
            @test keys(n.data) == (:train, :val)
            # The TRAIN split arrives wrapped, because prefetch is a framework default now and setup
            # applies it (`auto_prefetch`). The identity assertion moves through the wrapper rather
            # than being dropped: what matters is that setup wrapped `build_data`'s own object and
            # did not substitute anything for it.
            @test n.data.train isa PrefetchIterator
            @test ReactantNitro.prefetch_source(n.data.train) === BARE_TRAIN
            @test ReactantNitro.prefetch_depth(n.data.train) == 1                      # the default depth
            @test ReactantNitro.prefetch_workers(n.data.train) == max(1, Threads.nthreads(:default))
            # And only `train`: `run_eval` iterates its split directly, so wrapping one would advertise a
            # worker count nothing uses.
            @test n.data.val === BARE_VAL
            # `checkpoint = nothing` and `resume = :auto` into an empty directory: a FRESH init, so the
            # trajectory starts at zero rather than being restored from something.
            @test current_step(n) == 0 && current_epoch(n) == 0
            @test phase(n) isa Starting
        end

        @testset "the defaults that only exist once a run is set up" begin
            # G == 1 is `param_group`'s default reaching the layout, and the horizon is
            # `max_epochs * div(steps_per_epoch, accum)` with all three at their defaults.
            @test length(n.layout.groups) == 1 && n.layout.groups[1] === :default
            @test n.total == length(BARE_TRAIN)
            # No `metrics` and no `train_metrics` method: both routers stay `nothing`, which is the
            # signal the metrics substitution and the optimizer fold-away both key on.
            @test n.routing.metrics === nothing
            @test n.routing.train_metrics === nothing
            # By default, `lambda` defaulting to 0 keeps `Decay` out of the chain, so the group's
            # rule is the bare `RAdam` rather than an `OptimiserChain`.
            @test n.opt_state[1].rule isa Optimisers.RAdam
        end

        @testset "the binding report names the framework default as its own source" begin
            # The defaults audit found this: both branches of `clip_source`'s accessor test answered
            # `:field`, so a bare experiment declaring no `gradient_clip_norm` field was reported as
            # taking one. A report naming a source that does not exist cannot be checked against the
            # source, and this line prints on every run of every experiment that leaves the clip alone.
            @test clip_source(BareMLP(), 0.0f0) === :default
            @test occursin("[framework default]", binding_report(n))
            @test !occursin("[field on e]", binding_report(n))
            # The other three routes still resolve, and the keyword is detected by value, since
            # defaulting a keyword to the experiment's accessor makes a passed keyword
            # indistinguishable from an omitted one by the time the body runs.
            @test clip_source(BareMLP(), 1.0f0) === :keyword
        end

        # RELEASE THE DEFAULT LOGGER'S APPEND HANDLE, in the testset that opened it. This `Nitro` is
        # setup-only, so no `train!` ever reached `finish!`, which is the only thing that closes it,
        # and `n` is local to this testset so nothing outside can. On a local filesystem leaving it
        # open is harmless: the bare-run test's `clean_bare_dir()` unlinks `metrics.jsonl` and the
        # directory entry is gone at once. On an NFS-BACKED CHECKOUT it is not, because unlinking a
        # file that some process still holds open is a SILLY-RENAME to `.nfs<hex>` rather than a
        # removal, so the `rmdir` that follows fails with ENOTEMPTY and the bare-run test dies
        # before it trains. That made the whole of the bare-run test and its resume testset error on
        # this machine while passing on a local disk, which is the worst shape a test failure can
        # take: environment-dependent and nothing to do with what the test is about.
        n.logger.io === nothing || (close(n.logger.io); n.logger.io = nothing)
    end

    # ── the bare-run test ─────────────────────────────────────────────────────────────────

    @testset "a bare experiment trains, validates, and checkpoints with NOTHING passed" begin
        clean_bare_dir()
        n = train!(BareMLP())

        @testset "it stops after one epoch, and does not early-stop" begin
            # The default table states both, and both are asserted here rather than inferred:
            # `max_epochs` defaults to 1, so a first run finishing after one epoch is correct and
            # reads as a bug; `early_stop`
            # defaults to `nothing`, so nothing truncated it.
            @test current_epoch(n) == 1
            @test current_step(n) == length(BARE_TRAIN)      # one epoch, accum = 1
            @test n.early_stop === nothing
            @test n.stop_reason === :completed               # NOT `:early_stop`
            @test phase(n) isa Done
        end

        @testset "it validated, through the metrics substitution, not a `metrics` method" begin
            m = validate(n)
            @test keys(m) == (:val_loss,)
            @test m.val_loss isa Real && isfinite(m.val_loss)
        end

        @testset "it checkpointed, into the default `run_dir`, relative to the working directory" begin
            @test isdir(BARE_DIR)
            @test isfile(joinpath(BARE_DIR, "manifest.jld2"))
            @test !any(endswith(f, ".tmp") for f in readdir(BARE_DIR))
            # The checkpoint filename, on the default path: epoch, step, and the selection metric
            # with its value. The step is `length(BARE_TRAIN)` after one epoch and the value
            # belongs to this fixture's loss, so the value itself is matched as a pattern rather
            # than a literal.
            @test read_manifest(BARE_DIR)[1].file ==
                only(filter(f -> endswith(f, ".jld2") && f != "manifest.jld2", readdir(BARE_DIR)))
            @test occursin(
                Regex("^epoch-0001-step-$(length(BARE_TRAIN))-val_loss=[-0-9.e+]+\\.jld2\$"),
                read_manifest(BARE_DIR)[1].file
            )
            @test isfile(BARE_CKPT1())

            rec = load_checkpoint(n.checkpointer, BARE_CKPT1())
            @test rec.epoch == 1 && rec.step == length(BARE_TRAIN)
            @test rec.seed == 42                             # the seeding default, in the record
            @test rec.config == NamedTuple()                 # no GraphConst fields to bake or compare
            @test rec.devices == NamedTuple()
            @test rec.stop_reason === :completed
            # THE ROW THIS TEST IS NAMED FOR: the default checkpointer selects on `:val_loss`, and
            # the only reason that key exists is the metrics substitution for an experiment
            # defining no `metrics`.
            # Two defaults, jointly usable, which is the whole claim this audit is checking.
            @test keys(rec.metrics) == (:val_loss,)
            @test read_manifest(BARE_DIR)[1].score isa Float64
        end

        @testset "it logged, into the same directory, through the shipped JSON default" begin
            path = joinpath(BARE_DIR, "metrics.jsonl")
            @test isfile(path)
            ls = [JSON3.read(ln) for ln in eachline(path)]
            @test first(ls)["type"] == "params"
            @test any(x -> x["type"] == "metrics" && x["context"] == "validate", ls)
            @test last(ls)["type"] == "finish" && last(ls)["status"] == "completed"
        end
    end

    @testset "second run: `resume = :auto` composes with `max_epochs`" begin
        # Designed behavior, and exactly the interaction a defaults audit exists to surface: a second
        # `train!` in the same directory with `resume = :auto` continues the first rather than
        # starting fresh, and since `max_epochs` is 1 and the first run reached it, this one returns
        # without running an epoch. It says so rather than exiting silently, which is what the
        # checkpoint layer built the `stop_reason` for. Resuming is opt in, so it is asked for here.
        before = load_checkpoint(TopKCheckpointer(), BARE_CKPT1())
        n2 = @test_logs (:warn, r"will return without running an epoch") match_mode = :any train!(
            BareMLP(); resume = :auto
        )

        @test current_epoch(n2) == 1                     # restored, not advanced
        @test current_step(n2) == length(BARE_TRAIN)
        @test n2.stop_reason === :completed

        @testset "and it does not erase what the first run recorded" begin
            # The defaults audit found this and fixed it in `train!`. The phase system's final
            # rewrite carries the outcome into the last epoch's record, and it was gated on
            # `nitro.epoch > 0`, which a RESUME satisfies without having written anything: the
            # rewrite fired with `last_metrics` still empty and
            # replaced the previous process's record with one carrying no metrics at all, so the manifest
            # entry's score became `nothing` and that epoch dropped out of the top-K ranking. Silent, on
            # the default path, and reachable by running `train!` twice.
            after = load_checkpoint(TopKCheckpointer(), BARE_CKPT1())
            @test keys(after.metrics) == (:val_loss,)
            @test after.metrics.val_loss == before.metrics.val_loss
            @test after.stop_reason === :completed
            @test read_manifest(BARE_DIR)[1].score isa Float64
        end
    end

    # ── The chain, link by link ─────────────────────────────────────────────────────────

    @testset "the defaults are jointly usable: a stronger claim than each being stated" begin
        @testset "the `TopKCheckpointer(metric)` gap, in both directions" begin
            collection = (; train = BARE_TRAIN, val = BARE_VAL)
            # A temp `run_dir` here, deliberately, and it is not a default going unexercised: this
            # `Nitro` exists only to obtain a resolved `routing`, and taking the default would
            # resume from the checkpoint the bare-run test just wrote, making this testset depend
            # on the order of the ones above it.
            routing = Nitro(BareMLP(); run_dir = mktempdir()).routing
            # The positive half: the default checkpointer's metric is exactly what the framework
            # substitutes, so setup accepts it. This is the link that was missing.
            @test check_checkpointer(TopKCheckpointer(), collection, routing) === nothing
            # The negative half is the load-bearing one: the positive half passes whether or not the
            # check exists. An experiment defining no `metrics` cannot select on anything else, and the
            # framework says so AT SETUP rather than an epoch later, naming the substitution.
            err = try
                check_checkpointer(TopKCheckpointer(; metric = :mae), collection, routing)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("val_loss", err.msg) && occursin("defines no", err.msg)
        end

        @testset "`checkpointer = nothing` is the documented opt-out, and nothing else changes" begin
            # The one default a bare experiment is most likely to override, and it must not take the run
            # with it: no directory, no manifest, and the same one epoch. `logger = nothing` keeps the
            # testset about the CHECKPOINTER alone, since the default logger's file would otherwise be
            # the thing in the directory.
            dir = mktempdir()
            n = train!(BareMLP(); run_dir = dir, checkpointer = nothing, logger = nothing)
            @test current_epoch(n) == 1 && n.stop_reason === :completed
            @test isempty(readdir(dir))
        end
    end

    clean_bare_dir()

end
