# Phase system tests: logging, the phase registry, and early stopping.
#
# The acceptance criterion this file exercises: "A ten-verb null logger and a two-monitor registry
# drive a run; `request_stop!` from a monitor exits through `Done`." (The registry's verbs were named
# `register_phase_callback!` when that criterion was written.)

@testitem "lifecycle" begin
    using Test
    using ReactantNitro
    using ReactantNitro: MONITORS, cache_reset!, check_control_readback, finite_only, set_phase!
    using Logging, Lux, Random, Reactant, Statistics

    const FL = Float32

    # The size budget every `show` assertion in this file checks against. It exists to catch ONE
    # regression: a display falling back to Julia's struct default and printing the weights, which
    # was measured at 51,635 characters on a toy experiment carrying three small buffers. The
    # number is therefore loose rather than tight, and it was raised from 2,000 when the handle
    # summary and the binding report became one table: a display carrying the data, clip, schedule
    # and parameter-group bands runs to about 5,500 characters on the handles below. What still
    # has to hold is that it does not grow with the MODEL, only with how many splits, groups and
    # metrics a run has.
    const DISPLAY_BUDGET = 8_000

    @experiment struct LifeMLP
        width::GraphConst{Int} = 6
    end
    life_chain(w) = Lux.Chain(Lux.Dense(4 => w, tanh), Lux.Dense(w => 2))
    ReactantNitro.build_model(e::LifeMLP, rng) = (m = life_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::LifeMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::LifeMLP, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::LifeMLP) = 1.0f-2

    life_batches(n; seed = 3) =
        (
        rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, FL, 4, 8), y = randn(rng, FL, 2, 8)) for _ in 1:n]
    )
    const LIFE_TRAIN = life_batches(4)
    const LIFE_VAL = life_batches(2; seed = 4)
    ReactantNitro.build_data(::LifeMLP, dist) = (; train = LIFE_TRAIN, val = LIFE_VAL)

    mk_life(; kw...) = Nitro(LifeMLP(); checkpointer = nothing, run_dir = mktempdir(), kw...)

    # ── the logger contract: a logger implementing ALL TEN verbs ────────────────────────
    #
    # A missing method is a loud MethodError by design, so a logger that drives a run is the test that
    # the driver calls only what the contract declares, with the argument shapes it declares.

    mutable struct RecLogger
        calls::Vector{Pair{Symbol, Any}}
        state::Any
    end
    RecLogger(state = nothing) = RecLogger(Pair{Symbol, Any}[], state)

    ReactantNitro.log_metrics!(l::RecLogger, m; step, epoch, context, kwargs...) =
        (push!(l.calls, :metrics => (; m, step, epoch, context)); nothing)
    ReactantNitro.log_params!(l::RecLogger, p) = (push!(l.calls, :params => p); nothing)
    ReactantNitro.log_tags!(l::RecLogger, t) = (push!(l.calls, :tags => t); nothing)
    ReactantNitro.log_other!(l::RecLogger, k, v) = (push!(l.calls, :other => k); nothing)
    ReactantNitro.log_confusion!(l::RecLogger, m, labels; kwargs...) =
        (push!(l.calls, :confusion => size(m)); nothing)
    ReactantNitro.finish!(l::RecLogger, status) = (push!(l.calls, :finish => status); nothing)
    ReactantNitro.run_id(l::RecLogger) = "rec-1"
    ReactantNitro.run_url(l::RecLogger) = "https://example.invalid/rec-1"
    # The hosted-logger convention: `logger_info` carrying the run id and URL under the same
    # names the informational accessors use, so a tool rendering the table shows the two canonical
    # identifiers.
    ReactantNitro.logger_info(l::RecLogger) = (; run_id = "rec-1", run_url = "https://example.invalid/rec-1")
    ReactantNitro.logger_state(l::RecLogger) = l.state
    ReactantNitro.reattach!(l::RecLogger, s) = (push!(l.calls, :reattach => s); nothing)

    kinds(l::RecLogger) = first.(l.calls)
    of(l::RecLogger, k::Symbol) = [v for (kk, v) in l.calls if kk === k]

    @testset "a ten-verb logger drives a run" begin
        lg = RecLogger()
        n = train!(mk_life(; logger = lg, max_epochs = 2))

        @testset "all ten verbs exist on it, which is what the contract asks of a backend" begin
            @test run_id(lg) == "rec-1"
            @test run_url(lg) == "https://example.invalid/rec-1"
            @test logger_info(lg) == (; run_id = "rec-1", run_url = "https://example.invalid/rec-1")
            @test logger_info(nothing) == (;)
            @test ReactantNitro.logger_state(lg) === nothing
            @test ReactantNitro.backend(lg) === lg                  # the identity default
            @test log_tags!(lg, (; a = "b")) === nothing
            @test log_confusion!(lg, [1 0; 0 1], ["a", "b"]; epoch = 1) === nothing
            @test ReactantNitro.reattach!(lg, nothing) === nothing
        end

        @testset "parameters are logged BEFORE the first compile" begin
            # A compile-time crash would otherwise lose them, which is the whole reason logging
            # happens where it does.
            @test first(kinds(lg)) === :params
            p = only(of(lg, :params))
            @test p.seed == 42 && p.max_epochs == 2 && p.width == 6
            @test :binding_report in of(lg, :other) || "binding_report" in of(lg, :other)
        end

        @testset "train metrics carry `step`, validation carries `epoch`, both carry both" begin
            train_lines = [v for v in of(lg, :metrics) if v.context == "train"]
            val_lines = [v for v in of(lg, :metrics) if v.context == "validate"]
            @test length(train_lines) == 8                          # once per OPTIMIZER step, accum = 1
            @test length(val_lines) == 2                            # once per epoch
            @test [v.step for v in train_lines] == 1:8
            @test all(v -> v.epoch in (1, 2), train_lines)
            @test [v.epoch for v in val_lines] == [1, 2]
            @test all(v -> v.step in (4, 8), val_lines)             # so the two overlay in a backend
            @test haskey(first(train_lines).m, :loss)
            @test all(v -> v.m.loss isa Real && isfinite(v.m.loss), train_lines)
            # "keep `context` values byte-identical to existing conventions"
            @test Set(v.context for v in of(lg, :metrics)) == Set(["train", "validate", "data"])

            @testset "`data_wait_frac` is its OWN context, so neither count above moves" begin
                # The two counts asserted above are the logger contract: one train line per
                # optimizer step, one validate line per epoch. A per-epoch data-path diagnostic
                # would falsify whichever of them it borrowed, which is exactly why it does not
                # borrow either.
                data_lines = [v for v in of(lg, :metrics) if v.context == "data"]
                @test length(data_lines) == 2                        # once per epoch
                @test [v.epoch for v in data_lines] == [1, 2]
                @test all(v -> keys(v.m) == (:data_wait_frac,), data_lines)
                @test all(v -> 0.0 <= v.m.data_wait_frac <= 1.0, data_lines)
            end
        end

        @testset "`finish!` is called once, with the run's outcome" begin
            @test of(lg, :finish) == [:completed]
            @test n.stop_reason === :completed
            @test phase(n) isa Done
        end
    end

    @testset "the framework drops non-finite values before calling" begin
        @test finite_only((; a = 1.0, b = NaN, c = Inf, d = -Inf, e = 2)) == (; a = 1.0, e = 2)
        @test finite_only((; m = [1 0; 0 1])) == (; m = [1 0; 0 1])   # not a Real: passed through
    end

    # ── the monitor registry ─────────────────────────────────────────────────────────────

    record!(sink) = (phase, step, epoch, info) -> push!(sink, (phase, step, epoch, info))

    @testset "a two-monitor registry drives a run, in registration order" begin
        cache_reset!()
        a, b = [], []
        ha = register_phase_monitor!(record!(a))                   # module-level: future runs
        order = Symbol[]
        hb = try
            n = mk_life(; max_epochs = 1)
            # Per-run, and effective immediately: registered after construction, before `train!`.
            hb = register_phase_monitor!(n, record!(b))
            register_phase_monitor!(n, (p, s, e, i) -> push!(order, :second))
            h_first = register_phase_monitor!((p, s, e, i) -> push!(order, :first))
            train!(n)
            unregister_phase_monitor!(h_first)
            hb
        finally
            unregister_phase_monitor!(ha)
        end
        unregister_phase_monitor!(hb)

        @testset "both fired, and the module-level one fires first" begin
            @test !isempty(a) && !isempty(b)
            # `a`'s registry was live before `mk_life`, so it caught the handle-free `Starting`
            # that `Nitro(e)` publishes for the construction itself (`build_data` can start and
            # compile a data server, which is minutes of work no run has declared yet). `b` was
            # registered after construction and cannot have seen it: exactly one event apart, and
            # the assertions below say WHICH event rather than leaving the count to stand alone.
            @test length(a) == length(b) + 1
            @test typeof(first(a)[1]) === Starting
            @test first(a)[4].nitro === nothing        # no handle existed to publish through
            @test [typeof(p) for (p, _, _, _) in a[2:end]] == [typeof(p) for (p, _, _, _) in b]
            # The documented order: the module-level registry is copied in and the run's own
            # monitors follow it. It holds from the entry `Starting` on, because `with_repl` adopts
            # before it publishes. When it did not, this read `[:second, :first]`, and a monitor
            # registered between `Nitro(e)` and `train!` missed the declaration altogether.
            @test order[1:2] == [:first, :second]
        end

        @testset "every phase a monitor may wait on actually arrives" begin
            seen = Set(typeof(p) for (p, _, _, _) in a)
            # Every leaf is something an implementer must emit, and a phase that never arrives is
            # a monitor that hangs. The three train-side compile phases fire on a MISS, which a
            # reset cache makes certain of. `ExportCompiling` is the fourth compile leaf and
            # fires per
            # `export_model` call rather than per miss, so it is asserted in test/export.jl where an
            # export actually happens.
            @test GradCompiling in seen && OptCompiling in seen && EvalCompiling in seen
            # Unqualified, and that is the point of the phase-naming rule: this file does
            # `using Lux` and
            # `using ReactantNitro`, which is what every experiment file does, and the leaf that used to
            # be `Training` collided with `Lux.Training` right here.
            @test TrainStepping in seen && EvalStepping in seen
            @test Checkpointing in seen && Done in seen
            @test !(Failed in seen)
            @test Repl in seen
        end

        @testset "`Repl` is published ONCE, by the outermost call, and last" begin
            # THE property that makes the idle phase safe, and the one whose absence is invisible.
            # `train!` runs `validate` once per epoch; if that inner return published `Repl`, every epoch
            # boundary would announce an idle session in the middle of a run. A monitor that widens its
            # patience while idle -- which is the whole reason the phase exists -- would then stop
            # enforcing the per-step budget for the rest of the run, so a genuinely wedged training step
            # would never be caught. Nothing else in the suite would notice.
            phases = [typeof(p) for (p, _, _, _) in a]
            @test count(==(Repl), phases) == 1
            @test last(phases) === Repl
            # It comes AFTER the terminal phase rather than instead of it. A monitor still gets `Done`
            # and its `stop_reason`; `Repl` then says the process is waiting rather than wedged in
            # teardown.
            @test phases[end - 1] === Done
            # That `Repl` is published WITHOUT being recorded is asserted where it matters, by the
            # `phase(n) isa Done` / `isa Failed` checks the surrounding testsets already make: erasing
            # the outcome would trade a fact callers read for one that was never about the run.
            #
            # The counter is balanced afterwards, which is what the `finally` buys: an unbalanced
            # increment would leave the process claiming work in flight for the rest of its life, so it
            # would never report itself idle and a supervised gate would never release its GPU.
            @test !ReactantNitro.work_in_flight()
        end

        @testset "the progress counter advances in BOTH loops" begin
            # A monitor needs "time since progress" because a phase deadline measures from the phase
            # transition, and a run stays in `TrainStepping` for a whole epoch. Without a progress signal
            # a budget meaning "how long may one step take" is applied to the entire stretch and kills
            # healthy runs; that is exactly what happened before this existed.
            #
            # Both loops, and the eval one is the easy half to forget: nothing on the handle moves during
            # evaluation, since the batch index is local to the loop and `step` does not advance, so a
            # monitor keying off `current_step` would see a long validation as motionless.
            before = progress_counter()
            n = mk_life(; max_epochs = 1)
            train!(n)
            after_train = progress_counter()
            @test after_train > before

            validate(n)
            @test progress_counter() > after_train
        end

        @testset "a nested entry point stays quiet, and a throwing one hands control back" begin
            n = mk_life(; max_epochs = 1)
            # A fresh handle reports `Starting`, not `Repl`: construction publishes no transitions, so
            # there is nothing for a monitor to miss and nothing to invent a phase for.
            @test phase(n) isa Starting
            seen = []
            h = register_phase_monitor!(n, (p, s, e, i) -> push!(seen, typeof(p)))
            try
                # `validate` on its own IS an outermost call, so it publishes exactly one.
                validate(n)
                @test count(==(Repl), seen) == 1
                # Published, not recorded: an eval on an untrained handle leaves it `Starting`.
                @test phase(n) isa Starting

                # Nested by hand: only the outer exit publishes. The body now runs on a worker thread
                # under `-t N,1`, so the count is computed inside the body and
                # asserted here, where the testset context lives: an `@test` inside the body would not
                # record from the worker.
                empty!(seen)
                inner = 0
                ReactantNitro.with_repl(n) do
                    validate(n)
                    inner = count(==(Repl), seen)
                end
                @test inner == 0
                @test count(==(Repl), seen) == 1

                # A raising entry point still releases the counter and still hands control back. Without
                # the `finally`, one failed call would wedge `work_in_flight` at true forever, and the
                # process would never report itself idle again.
                empty!(seen)
                @test_throws ErrorException ReactantNitro.with_repl(n) do
                    error("boom")
                end
                @test !ReactantNitro.work_in_flight()
                @test count(==(Repl), seen) == 1
            finally
                unregister_phase_monitor!(h)
            end
        end

        @testset "the payload, and `metrics` rides OUT of `EvalStepping`" begin
            # Every event carries the three fields, the handle-free construction publish included:
            # that one answers `nitro = nothing, logger = nothing, is_rank0 = true` deliberately,
            # because there is no handle to describe and a monitor reading the payload must not have
            # to guess which shape it got.
            for (p, s, e, info) in a
                @test info.is_rank0 === true
                @test haskey(info, :logger) && haskey(info, :nitro)
            end
            # The handle is there for everything published THROUGH one, which is every event but
            # the construction's own.
            for (p, s, e, info) in a[2:end]
                @test info.nitro isa Nitro
            end
            @test first(a)[4].nitro === nothing
            # Step and epoch are "nothing until an epoch has begun", so they are defined for
            # every event inside the loop and `nothing` for the two that precede it: the handle-free
            # construction publish, and the `Starting` `with_repl` declares on the way in.
            lead = a[1:2]
            @test all(typeof(p) === Starting for (p, _, _, _) in lead)
            @test all(s === nothing && e === nothing for (_, s, e, _) in lead)
            for (p, s, e, info) in a[3:end]
                @test s isa Int && e isa Int                        # inside the loop, both defined
            end
            # The transition after validation is the one that carries the numbers, which is what
            # replaces v3's phantom `on_validation_end` hook.
            withm = [(p, info) for (p, _, _, info) in a if haskey(info, :metrics)]
            @test length(withm) == 1
            @test first(withm)[1] isa TrainStepping                  # EvalStepping -> TrainStepping
            @test haskey(first(withm)[2].metrics, :val_loss)
        end

        @testset "a cache HIT publishes no compile phase" begin
            # The second run over the same experiment reuses every program, so a watchdog is not told to
            # widen its timeout for a compile that is not happening.
            c = []
            h = register_phase_monitor!(record!(c))
            try
                train!(mk_life(; max_epochs = 1))
            finally
                unregister_phase_monitor!(h)
            end
            @test !any(p isa Compiling for (p, _, _, _) in c)
        end
    end

    @testset "step and epoch are `nothing` before the first epoch" begin
        n = mk_life()
        seen = []
        h = register_phase_monitor!(n, record!(seen))
        validate(n)                                                 # a standalone call, no run at all
        unregister_phase_monitor!(h)
        @test !isempty(seen)
        @test all(s === nothing && e === nothing for (_, s, e, _) in seen)
        # `with_repl` declares `Starting` on the way in, so the verb's own first phase is the second
        # event. The entry declaration is what stops the pre-compile stretch being charged against
        # whatever budget was already counting down.
        @test typeof(first(seen)[1]) === Starting
        @test typeof(seen[2][1]) === EvalStepping
        # Restored to where it found it. Asserted on the HANDLE rather than on the last event, which is
        # what "restored" actually means: since the idle phase's `Repl`, the last event a
        # standalone call emits is
        # the published `Repl`, and `Repl` is deliberately never recorded, so the field still reads
        # `Starting` while the event stream ends one step later.
        @test phase(n) isa Starting
        @test typeof(seen[end - 1][1]) === Starting                 # the restoring transition itself
        @test typeof(last(seen)[1]) === Repl                        # published, not recorded
    end

    @testset "error isolation: a throwing monitor never kills a run, and warns ONCE" begin
        n = mk_life(; max_epochs = 2)
        register_phase_monitor!(n, (p, s, e, i) -> error("monitor is broken"))
        fine = []
        register_phase_monitor!(n, record!(fine))

        logger = Test.TestLogger()
        with_logger(logger) do
            train!(n)
        end
        @test phase(n) isa Done                                     # the run is unaffected
        @test !isempty(fine)                                        # and the other monitor still fired
        warns = [
            r for r in logger.logs if r.level == Logging.Warn &&
                occursin("phase monitor threw", r.message)
        ]
        @test length(warns) == 1                                    # once per monitor, NOT per event
    end

    @testset "`unregister_phase_monitor!` removes from the registry the handle came from" begin
        before = length(MONITORS)
        h = register_phase_monitor!((p, s, e, i) -> nothing)
        @test length(MONITORS) == before + 1
        unregister_phase_monitor!(h)
        @test length(MONITORS) == before
        @test unregister_phase_monitor!(h) === nothing              # removing twice is a no-op
    end

    # ── early stopping ───────────────────────────────────────────────────────────────────

    @testset "`should_stop` counts epochs without improvement, with an absolute `min_delta`" begin
        es = EarlyStopping(; metric = :val_loss, mode = :min, patience = 2, min_delta = 0.1)
        @test should_stop(es, 1, (; val_loss = 1.0)) === false      # first epoch sets the best
        @test es.best == 1.0
        @test should_stop(es, 2, (; val_loss = 0.95)) === false     # better, but not BY min_delta
        @test es.wait == 1
        @test should_stop(es, 3, (; val_loss = 0.5)) === false      # a real improvement resets patience
        @test es.wait == 0 && es.best == 0.5
        @test should_stop(es, 4, (; val_loss = 0.6)) === false
        @test should_stop(es, 5, (; val_loss = 0.6)) === true       # patience exhausted

        mx = EarlyStopping(; metric = :acc, mode = :max, patience = 1, min_delta = 0.0)
        @test should_stop(mx, 1, (; acc = 0.5)) === false
        @test should_stop(mx, 2, (; acc = 0.7)) === false
        @test should_stop(mx, 3, (; acc = 0.7)) === true            # equal is not an improvement

        @test should_stop(nothing, 1, (; val_loss = 1.0)) === false # the shipped no-op default
    end

    @testset "the metric must exist, and must survive its readback" begin
        es = EarlyStopping(; metric = :nope)
        err = try
            should_stop(es, 1, (; val_loss = 1.0))
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("(:val_loss,)", err.msg)                     # names what IS available

        # A scalar the framework BRANCHES on is validated, because a failed readback returns
        # garbage without raising on this stack.
        @test_throws ErrorException should_stop(EarlyStopping(), 1, (; val_loss = NaN))
        @test check_control_readback(1.5, "a test", :m) === 1.5
        @test_throws ErrorException check_control_readback([1, 2], "a test", :m)
    end

    @testset "setup rejects what it can see, before the run rather than an epoch in" begin
        @test_throws ErrorException mk_life(; early_stop = EarlyStopping(; mode = :lower))
        @test_throws ErrorException mk_life(; early_stop = EarlyStopping(; patience = 0))
        # No `val` split to evaluate the condition on: the run would silently never stop early.
        err = try
            Nitro(
                LifeMLP(); data = (; train = LIFE_TRAIN), checkpointer = nothing,
                run_dir = mktempdir(), early_stop = EarlyStopping()
            )
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("val", err.msg)
        # This experiment defines no `metrics`, so `:val_loss` is the only key that can ever appear, and
        # THAT much the framework can check at setup.
        err2 = try
            mk_life(; early_stop = EarlyStopping(; metric = :accuracy))
            nothing
        catch ex
            ex
        end
        @test err2 isa ErrorException
        @test occursin("accuracy", err2.msg) && occursin("val_loss", err2.msg)
    end

    @testset "a run that stops on patience exits through `Done`, gracefully" begin
        lg = RecLogger()
        # `min_delta` larger than any change the loss can make, so no epoch ever counts as an
        # improvement and patience is exhausted at epoch 2 deterministically.
        n = train!(
            mk_life(;
                logger = lg, max_epochs = 5,
                early_stop = EarlyStopping(; patience = 1, min_delta = 100.0)
            )
        )
        @test current_epoch(n) == 2
        @test phase(n) isa Done                                     # NOT Failed: an early stop finished
        @test n.stop_reason === :early_stop
        @test of(lg, :finish) == [:early_stop]
        # Graceful: the epoch finished, validation ran, and the metrics for the stopping epoch exist.
        @test length([v for v in of(lg, :metrics) if v.context == "validate"]) == 2
        @test length([v for v in of(lg, :metrics) if v.context == "train"]) == 8
    end

    @testset "phase system: `request_stop!` FROM A MONITOR exits through `Done`" begin
        lg = RecLogger()
        n = mk_life(; logger = lg, max_epochs = 5)
        register_phase_monitor!(n, (p, s, e, info) -> p isa EvalStepping && request_stop!(info.nitro))
        train!(n)

        @test phase(n) isa Done
        @test current_epoch(n) == 1                                 # stopped at the first check
        @test n.stop_reason === :requested                          # and says WHY, distinctly
        @test of(lg, :finish) == [:early_stop]
        @test length([v for v in of(lg, :metrics) if v.context == "validate"]) == 1
    end

    @testset "a run that throws exits through `Failed`, and finalizes the logger" begin
        lg = RecLogger()
        n = mk_life(; logger = lg, max_epochs = 1)
        seen = []
        register_phase_monitor!(n, record!(seen))
        # A batch schema change mid-epoch is the cheapest genuine mid-run failure.
        n.data = (; train = vcat(LIFE_TRAIN[1:2], [(; z = randn(FL, 4, 8))]), val = LIFE_VAL)
        @test_throws ErrorException train!(n)
        @test phase(n) isa Failed
        @test n.stop_reason === :error
        @test of(lg, :finish) == [:error]
        @test any(p isa Failed for (p, _, _, _) in seen)
    end

    # ── showing a handle, without showing the model ─────────────────────────────────────
    #
    # The same hazard `CheckpointRecord`'s `show` covers, one level up: six `Nitro` fields are
    # parameter trees and `data` is the loaded collection, so the DEFAULT struct `show` prints the
    # whole model for a bare `nitro` at the REPL. In an agent session the REPL's output is the
    # transcript, so that is context window spent to read an epoch number. Bounded output has to be
    # the default rendering or the trap stays one keystroke away.
    @testset "a handle shows its state and never its weights" begin
        n = mk_life(; max_epochs = 2)
        train!(n)

        for s in (sprint(show, n), sprint(show, MIME"text/plain"(), n))
            # The test that actually catches a regression: the printed form of an array of floats.
            # A `show` that fell back to the struct default prints these in the thousands.
            @test !occursin("Float32[", s)
            @test !occursin("Float64[", s)
            # Bounded, and bounded SMALL. This cannot be allowed to grow with the model, so the
            # limit is checked rather than the content.
            @test length(s) < DISPLAY_BUDGET
            @test occursin("LifeMLP", s)
        end

        # The weights ARE still in there: this is a summary, not a lighter handle.
        @test parameters(n) !== nothing

        short = sprint(show, n)
        @test occursin("epoch $(current_epoch(n))", short)
        @test occursin("step $(current_step(n))", short)

        long = sprint(show, MIME"text/plain"(), n)
        @test occursin("Done", long)                       # the phase, which is the usual question
        # Split sizes, not the splits themselves, and they live in the `data` band now that the
        # handle and its binding report are one table: a count leading the notes that band's row
        # already carried, rather than a `data` row of the summary saying the same thing twice.
        @test occursin(r"train\s+4 batches", long)
        @test occursin(r"val\s+2 batches", long)
        # What it withheld is said by the `weights` row and by the note naming `parameters`, so
        # the title is just the name; the assertions that no array reaches the output are above.
        @test occursin("Nitro for LifeMLP", long)
        @test occursin("binding_report", long)             # and names the readers, where it is asked
        # WHERE VALUES BOUND, in the same display: the report is no longer a second thing printed
        # beside the handle, and a reader who prints one handle sees both.
        @test occursin("gradient clip", long)
        @test occursin("[framework default]", long)
        @test occursin("parameter groups", long)

        # Counting parameters must not need the arrays: the count comes off the layout, which is
        # host metadata, so a handle displays without moving a byte of device memory.
        @test occursin(string(ReactantNitro._commas(sum(n.layout.lengths))), long)

        # A handle with no data and no training still displays, because a `show` that can throw is
        # a `show` nobody can use while debugging the thing that broke.
        bare = Nitro(LifeMLP(); data = (;), checkpointer = nothing, run_dir = mktempdir())
        @test occursin("none", sprint(show, MIME"text/plain"(), bare))
        @test length(sprint(show, bare)) < 2_000
    end

    # ── what the run produced, without a logger backend ──────────────────────────────────
    #
    # The shipped `JSONLogger` writes metrics to a file and a hosted backend sends them away, so a
    # REPL `train!(n)` used to return a handle that could say it had finished and not how it had
    # done. The numbers exist on the handle for exactly that reason.
    @testset "a finished handle reports what the run produced" begin
        n = mk_life(; max_epochs = 2)
        @test isempty(n.last_metrics)          # nothing has run yet, including after a restore
        @test n.elapsed === nothing

        untrained = sprint(show, MIME"text/plain"(), n)
        @test !occursin("metrics ", untrained)
        @test !occursin("elapsed", untrained)
        @test occursin("fresh from build_model", untrained)

        train!(n)

        # The last epoch's validation metrics, on the handle, as HOST values.
        @test !isempty(n.last_metrics)
        @test haskey(n.last_metrics, :val_loss)
        @test n.last_metrics.val_loss isa Real
        @test n.elapsed isa Real && n.elapsed > 0

        long = sprint(show, MIME"text/plain"(), n)
        @test occursin("metrics", long) && occursin("val_loss", long)
        @test occursin("elapsed", long)
        # `mk_life` passes `checkpointer = nothing`, so there is no selected checkpoint and the
        # row is absent rather than present and empty.
        @test n.best_checkpoint === nothing
        @test !occursin("checkpoint  ", long)
        # `checkpoint_source` is a CONSTRUCTION-time fact, so reporting it alone made a trained
        # handle claim its weights were "fresh from build_model": true of where they started and
        # wrong about what they are.
        @test occursin("trained here", long)
        @test !occursin("fresh from build_model", long)
        @test length(long) < DISPLAY_BUDGET

        # A metric may be an array (a confusion matrix is the standard case), and the summary
        # routes metrics through the same renderer as everything else. Set directly rather than
        # trained, so this pins the DISPLAY path and not a second experiment's numerics.
        n.last_metrics = (; acc = 0.5, confusion = zeros(Int, 10, 10))
        withmat = sprint(show, MIME"text/plain"(), n)
        @test occursin("size 10x10", withmat)
        @test !occursin("0, 0, 0", withmat)
        @test length(withmat) < DISPLAY_BUDGET
    end

    # ── metrics are their own section ───────────────────────────────────────────────────
    #
    # A band of their own rather than rows mixed into `state`, because they are the part of this
    # display that changes every epoch and the part a reader scans for. They used to be tiled
    # three `name value` pairs to a row, to keep a dozen metrics out of a dozen lines of their
    # own box; inside one shared table a metric is a label and a value like everything in the
    # `state` band above it, and a row each is what lines them up with it.
    @testset "the metrics section is named, epoch-stamped, and carries the note" begin
        eight = (; a = 1, b = 2, c = 3, d = 4, e = 5, f = 6, g = 7, h = 8)
        n = mk_life(; max_epochs = 1)
        train!(n)

        # No metrics, no section: an empty band is worse than an absent one.
        n.last_metrics = (;)
        @test !occursin("metrics", sprint(show, MIME"text/plain"(), n))

        n.last_metrics = eight
        long = sprint(show, MIME"text/plain"(), n)
        @test occursin("metrics", long)
        @test occursin("epoch $(current_epoch(n))", long)
        # Every name and every value, once each, one row apiece.
        for k in keys(eight)
            @test occursin(string(k), long)
        end
        # The trailing note rides under the whole table, so it stays at the bottom however many
        # sections there are.
        @test endswith(long, "`logger_info`")
        @test length(long) < DISPLAY_BUDGET
    end

    # ── progress reporting: begin, one per unit, end ────────────────────────────────────
    #
    # `progress_counter` was already correct and already invisible: one monotonic number with no
    # total and no output, which answers "is this process moving" for a watchdog and is not
    # something a person watches. A bar needs the two things it cannot supply, how long this
    # stretch is and when it starts and stops, and those are what this contract adds. The events
    # are tested rather than the drawing: a stub reporter pins the contract, and whether
    # ProgressMeter renders a bar correctly is ProgressMeter's business.
    @testset "the progress reporter sees every unit of work, bracketed" begin
        events = Tuple{Symbol, String, Int, Int, Int}[]
        prev = ReactantNitro.progress_reporter!(
            (v, l, t, e, m) -> (push!(events, (v, l, t, e, m)); nothing)
        )
        try
            n = mk_life(; max_epochs = 2)
            train!(n)
        finally
            ReactantNitro.progress_reporter!(prev)
        end

        begins = [e for e in events if e[1] === :begin]
        # Two epochs, each a training stretch, then a validation stretch over the `val` split, then
        # the metric reduction. That last one is UNCOUNTED (`total = 0`) and everything it covers
        # runs after the validation bar has closed: `reduce_metrics` folds the accumulator and
        # `finalize_metrics` is user code, so without a stretch of its own it is a silent stall
        # between the last batch and the checkpoint.
        @test [(e[2], e[3], e[4], e[5]) for e in begins] == [
            ("train", 4, 1, 2), ("val", 2, 1, 2), ("finalize metrics", 0, 1, 2),
            ("train", 4, 2, 2), ("val", 2, 2, 2), ("finalize metrics", 0, 2, 2),
        ]
        # Every stretch closes. An unbalanced pair is a bar left on someone's terminal.
        @test count(e -> e[1] === :end, events) == length(begins)
        # One `:step` per unit, and the same units the counter counts: 4 train steps and 2 val
        # batches, twice over.
        @test count(e -> e[1] === :step, events) == 2 * (4 + 2)
        # `:done` exactly once, at the very end of the entry point and not once per stretch.
        # It is what lets a reporter reuse one terminal line across every bar of a run and still
        # close that line when the run is over, and a second one would close it twice.
        @test count(e -> e[1] === :done, events) == 1
        # Nothing countable follows it. Not "it is the last event": `set_phase!(Done())` fires a
        # `:phase` after the run has closed its last bar, which is correct and says nothing about
        # work remaining.
        after = events[(findfirst(e -> e[1] === :done, events) + 1):end]
        @test all(e -> e[1] === :phase, after)
        # Ordering, not just counts: nothing may step outside a bracket.
        depth = 0
        for (v, _, _, _, _) in events
            v === :begin && (depth += 1)
            v === :step && (@test depth == 1)
            v === :end && (depth -= 1)
        end
        @test depth == 0

        # NO CHECKPOINT STRETCH HERE. `mk_life` passes `checkpointer = nothing`, and
        # `save_checkpoint!(::Nothing, ...)` is a no-op, so reporting one would open and close a
        # bar for a write that never happens.
        @test !any(e -> occursin("checkpoint", e[2]), events)
        # The stream teardown is a PHASE, not a stretch: it stalls a bar that is still open, with
        # the last batch already counted, so it decorates that bar rather than opening another.
        @test any(e -> e[1] === :phase && e[2] == "closing data stream", events)
    end

    # ── the checkpoint write, which emits no units and used to emit no bar ───────────────
    #
    # A write is not a countable stretch, so nothing on the progress path noticed it and an epoch
    # spent writing weights to a slow or remote filesystem looked exactly like an epoch that
    # finished and then hung. It is reported as an unknown-length stretch: no counting, just the
    # label and the epoch it belongs to.
    @testset "a checkpointed run reports its writes" begin
        events = Tuple{Symbol, String, Int, Int, Int}[]
        prev = ReactantNitro.progress_reporter!(
            (v, l, t, e, m) -> (push!(events, (v, l, t, e, m)); nothing)
        )
        try
            train!(Nitro(LifeMLP(); run_dir = mktempdir(), max_epochs = 2, resume = false))
        finally
            ReactantNitro.progress_reporter!(prev)
        end

        begins = [e for e in events if e[1] === :begin]
        # One per epoch, carrying the epoch it belongs to, plus the final rewrite. The rewrite is
        # LABELLED APART: it lands immediately after the last epoch's own write, and two identical
        # stretches back to back read as a stutter rather than as two different writes.
        @test [(e[2], e[4], e[5]) for e in begins if occursin("checkpoint", e[2])] ==
            [("checkpoint", 1, 2), ("checkpoint", 2, 2), ("final checkpoint", 2, 2)]
        # UNKNOWN LENGTH, never a count. A write emits no units, and claiming a total the stretch
        # will never reach leaves a bar stuck short of its own end.
        @test all(e -> e[3] == 0, [e for e in begins if occursin("checkpoint", e[2])])
        # Still balanced with the write in the brackets, which is what keeps a failed write from
        # leaving its bar on the terminal.
        @test count(e -> e[1] === :end, events) == length(begins)
        # And no units are attributed to it: every `:step` still belongs to a train or val stretch.
        open_label = ""
        for (v, l, _, _, _) in events
            v === :begin && (open_label = l)
            v === :step && (@test !occursin("checkpoint", open_label))
        end
    end

    # ── the bar's one blind spot: a compile produces no units ────────────────────────────
    #
    # An XLA compile emits nothing to count, so the bar sits at zero for what is most of a first
    # epoch's wall clock and reads as a hang. The phase tree already knows, and `p isa Compiling`
    # is the documented query for it, so the reporter is told instead of left to guess.
    @testset "compiling is reported, and named" begin
        ev = Tuple{Symbol, String}[]
        # The compile cache is MODULE-LEVEL, so the testsets above have already compiled every
        # program `LifeMLP` needs and this run would otherwise compile nothing and report nothing.
        # Resetting is what makes the compile actually happen here.
        cache_reset!()
        prev = ReactantNitro.progress_reporter!((v, l, t, e, m) -> (push!(ev, (v, l)); nothing))
        try
            train!(mk_life(; max_epochs = 1))
        finally
            ReactantNitro.progress_reporter!(prev)
        end

        labels = unique(l for (v, l) in ev if v === :phase && !isempty(l))
        # Which program, not merely "busy": the gradient compile is the long one and the eval
        # compile is the one that surprises people mid-run.
        @test "compiling gradient" in labels
        @test "compiling optimizer" in labels
        @test "compiling eval" in labels

        # Every compile is closed with an empty label, or the bar would wear it for the rest of
        # the stretch.
        opened = count(x -> x[1] === :phase && !isempty(x[2]), ev)
        closed = count(x -> x[1] === :phase && isempty(x[2]), ev)
        @test closed >= opened

        # The gradient compile happens INSIDE the training stretch, which is what puts the label
        # on a bar that is already on screen rather than nowhere.
        i = findfirst(x -> x == (:phase, "compiling gradient"), ev)
        j = findlast(x -> x[1] === :begin, ev[1:i])
        @test j !== nothing
    end

    # A reporter that throws is a display bug, and a training run is not the place to pay for one.
    @testset "a reporter that throws is switched off, not propagated" begin
        prev = ReactantNitro.progress_reporter!((args...) -> error("reporter is broken"))
        try
            n = mk_life(; max_epochs = 1)
            @test_logs (:warn,) match_mode = :any train!(n)
            @test phase(n) isa Done                       # the run finished regardless
            @test ReactantNitro._PROGRESS_REPORTER[] === nothing   # and said so once
        finally
            ReactantNitro.progress_reporter!(prev)
        end
    end

    @testset "the built-in reporter is installed and draws nothing off a terminal" begin
        # `__init__` installs it, so a fresh process has a reporter without anyone asking.
        @test ReactantNitro.progress_bar_reporter isa Function
        # Driving it directly must not throw, and must not draw here: the suite is not
        # interactive, so `_drawing_progress()` is false and the bar is created disabled.
        @test ReactantNitro._drawing_progress() == false
        @test ReactantNitro.progress_bar_reporter(:begin, "train", 3, 1, 2) === nothing
        @test ReactantNitro.progress_bar_reporter(:phase, "compiling gradient", 0, 0, 0) === nothing
        @test ReactantNitro.progress_bar_reporter(:phase, "", 0, 0, 0) === nothing
        @test ReactantNitro.progress_bar_reporter(:step, "", 0, 0, 0) === nothing
        @test ReactantNitro.progress_bar_reporter(:end, "", 0, 0, 0) === nothing
        @test ReactantNitro._BAR[] === nothing            # closed, not left open
        # Setup compiles before any stretch has begun, so a `:phase` with no live bar is ordinary
        # and must be a no-op rather than a crash inside a training run.
        @test ReactantNitro.progress_bar_reporter(:phase, "compiling gradient", 0, 0, 0) === nothing
        # An unknown length is an indeterminate bar rather than an error.
        @test ReactantNitro.progress_bar_reporter(:begin, "test", 0, 0, 0) === nothing
        ReactantNitro.progress_bar_reporter(:end, "", 0, 0, 0)
    end

    # ── the table renderer, which the PrettyTables extension swaps ───────────────────────
    @testset "a swapped table renderer receives the documented arguments" begin
        seen = Ref{Any}(nothing)
        prev = ReactantNitro.table_renderer!(
            (io, title, sections, note) -> begin
                seen[] = (; title, sections, note)
                print(io, "RENDERED BY THE STUB")
            end
        )
        try
            n = mk_life(; max_epochs = 1)
            out = sprint(show, MIME"text/plain"(), n)
            @test out == "RENDERED BY THE STUB"
            got = seen[]
            @test occursin("Nitro for LifeMLP", got.title)
            @test got.sections isa Vector{ReactantNitro.TableSection}
            @test all(s -> s.rows isa ReactantNitro.TableRows, got.sections)
            @test all(s -> all(r -> r isa Vector{String}, s.rows), got.sections)
            @test got.note isa AbstractString      # the handle summary carries one

            # The LEADING section is unlabelled and unheadered: the table title names it, and a
            # band labelled directly under the title is a heading printed twice.
            state = first(got.sections)
            @test state.title == ""
            @test state.header == String[]
            @test any(r -> r[1] == "phase", state.rows)

            # The binding report's bands are appended to it, which is the whole point of one
            # table: where each value bound is part of the handle's display, not a second one.
            @test any(s -> s.title == "data", got.sections)
            # The gradient clip shares the `bindings` band with the schedules rather than having
            # one of its own: a labelled heading and a column header around a single row read as a
            # section only because the value needed somewhere to live.
            @test any(s -> s.title == "bindings", got.sections)
            @test any(
                s -> s.title == "bindings" && any(r -> r[1] == "gradient clip", s.rows),
                got.sections
            )
            @test any(s -> startswith(s.title, "parameter groups"), got.sections)

            # The experiment show goes through the same path: one section, a real header, no note.
            sprint(show, MIME"text/plain"(), experiment(n))
            @test length(seen[].sections) == 1
            @test only(seen[].sections).header == ["field", "marker", "value"]
            @test seen[].note === nothing
        finally
            ReactantNitro.table_renderer!(prev)
        end
        # Restoring `nothing` puts the built-in aligned renderer back, which is what a user does
        # when something else in a session pulled PrettyTables in.
        @test ReactantNitro._TABLE_RENDERER[] === prev
    end

    # ── the selected checkpoint ─────────────────────────────────────────────────────────
    #
    # Which file the run would hand you is the BEST by the checkpointer's own metric and mode,
    # which is a different question from the newest, and it is answered from the manifest alone so
    # that no record is opened and `show` never touches the filesystem.
    @testset "a run with a checkpointer reports the one it selected" begin
        dir = mktempdir()
        n = Nitro(LifeMLP(); run_dir = dir, max_epochs = 3, resume = false)
        train!(n)

        bc = n.best_checkpoint
        @test bc !== nothing
        @test bc.metric === :val_loss
        @test bc.score isa Real
        @test isfile(bc.path)
        # The BEST, not the newest: `mode = :min` by default, so no retained entry scores lower.
        entries = [e for e in read_manifest(dir) if e.score !== nothing]
        @test bc.score == minimum(e.score for e in entries)

        long = sprint(show, MIME"text/plain"(), n)
        @test occursin("checkpoint", long)
        @test occursin("epoch $(bc.epoch)", long)
        @test occursin("val_loss", long)
        @test occursin(basename(bc.path), long)          # the file, pasteable
        @test length(long) < DISPLAY_BUDGET

        # `selected_checkpoint` is the whole mechanism, and it refuses rather than guesses.
        @test ReactantNitro.selected_checkpoint(nothing, dir) === nothing
        # A directory with no manifest is an answer, not an error. Note the checkpointer passed
        # here is a FRESH one: setup pins `ckpt.dir` to the run directory, so a bound checkpointer
        # ignores the `run_dir` argument entirely, which is the behaviour and not a bug.
        @test ReactantNitro.selected_checkpoint(
            TopKCheckpointer(; dir = mktempdir()), mktempdir()
        ) === nothing
    end

    # ── resuming is opt in ──────────────────────────────────────────────────────────────
    #
    # `run_dir` defaults to a name derived from the experiment type, so under the old
    # `resume = :auto` default a second `Nitro(MyExp())` in the same working directory silently
    # continued the previous run. A constructor picking up weights nobody named is the wrong
    # default whatever warning surrounds it, so the default is `false` and resuming is asked for.
    @testset "a fresh handle does not resume; `:auto` is what does" begin
        dir = mktempdir()
        n1 = Nitro(LifeMLP(); run_dir = dir, max_epochs = 1)
        train!(n1)
        @test current_epoch(n1) == 1

        # THE DEFAULT. Same directory, same experiment, checkpoints sitting right there.
        fresh = Nitro(LifeMLP(); run_dir = dir, max_epochs = 4)
        @test current_epoch(fresh) == 0
        @test current_step(fresh) == 0
        @test fresh.checkpoint_source === nothing
        @test occursin("fresh from build_model", sprint(show, MIME"text/plain"(), fresh))

        # And what resuming looks like when it is asked for.
        cont = Nitro(LifeMLP(); run_dir = dir, max_epochs = 4, resume = :auto)
        @test current_epoch(cont) == 1
        @test cont.checkpoint_source !== nothing
        @test occursin("restored from", sprint(show, MIME"text/plain"(), cont))

        # `:auto` in a directory with nothing in it is a fresh run, not an error: that is what
        # makes it usable as a standing setting in a harness that may be starting or recovering.
        empty_dir = Nitro(LifeMLP(); run_dir = mktempdir(), max_epochs = 4, resume = :auto)
        @test current_epoch(empty_dir) == 0
        @test empty_dir.checkpoint_source === nothing
    end

    @testset "elapsed reads at all three scales" begin
        el = ReactantNitro._nitro_elapsed
        @test el(nothing) === nothing
        @test el(1.25) == "1.2s"
        @test el(125.0) == "2m 05s"          # zero-padded, so a column of these lines up
        @test el(7_500.0) == "2h 05m"
    end

end
