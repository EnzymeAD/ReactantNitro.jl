# test/worker.jl
#
# `with_repl` moves a long entry point's body to a default-pool worker thread when the
# caller is on an interactive thread. CI runs the suite single-threaded, where `with_repl` runs
# inline, so the machinery is tested in two halves:
#
#   * the thread-independent parts, the failure unwrap and the interrupt-hook contract, run
#     in-process here; and
#   * the spawn itself is exercised by a `-t 4,1` subprocess that records which thread a
#     `with_repl` body actually ran on, that a failing body surfaces its own exception type, that
#     `Nitro(e)` itself runs its construction body off the interactive thread (`_off_interactive`),
#     and that a real `train!` composes with the spawn.

@testitem "worker" begin
    using Test
    using ReactantNitro

    @testset "with_repl: worker spawn, failure unwrap, interrupt hook" begin
        # This process is single-threaded, so the spawn guard is off: the suite runs inline, and the
        # entry points that `@test_throws` through `train!`/`validate` keep their inline semantics.
        @test !ReactantNitro._should_spawn()

        # A failing body surfaces its OWN exception type, not the `TaskFailedException` that `fetch`
        # wraps a failed task's result in. That is what an inline `@test_throws ErrorException
        # train!(...)` asserts and what the REPL shows for a failed run; exercised here through the
        # same fetch+unwrap path the spawn uses, on the only thread this process has.
        t = Threads.@spawn begin
            sleep(0.05)
            error("boom from the body")
        end
        @test_throws ErrorException ReactantNitro._repl_wait(t, nothing, nothing)

        # The interrupt-hook contract: `on_interrupt(task, nitro)` is called only when the wait is
        # interrupted; a completing task never touches it. (An InterruptException cannot be raised
        # into a parked task from this single thread, so the hook's end-to-end behavior, and the
        # spawn itself, is the subprocess test's job.)
        t2 = Threads.@spawn 42
        @test ReactantNitro._repl_wait(t2, nothing, (task, nitro) -> error("unreachable")) == 42
    end

    @testset "with_repl spawns onto a worker thread under -t N,1" begin
        # The suite runs single-threaded, so the spawn path never executes in-process. The honest
        # test is a subprocess started with a default pool: the body must run on a thread outside the
        # interactive pool, a failing body must surface its own exception type, and a real `train!`
        # must compose with the spawn (XLA compile and execute from a worker thread). The child
        # reports through a file, and its stderr is left attached so a failure names itself.
        out = joinpath(mktempdir(), "probe.txt")
        probe = raw"""
            record(k, v) = open(io -> println(io, k, "=", v), ENV["NITRO_PROBE_OUT"], "a")

            using ReactantNitro
            using ReactantNitro: @experiment
            using Lux

            record("interactive", Threads.nthreads(:interactive))
            record("default", Threads.nthreads(:default))

            @experiment struct MiniExp
                "Hidden width. Structural."
                width::GraphConst{Int} = 4

                "Epochs. Driver-only."
                max_epochs::Int = 1
            end

            ReactantNitro.build_model(e::MiniExp, rng) = begin
                # `build_model` runs inside `Nitro(e)`, so the thread it sees is the thread the
                # constructor body runs on: `_off_interactive` must have moved it off the
                # interactive pool.
                record("ctor_thread", Threads.threadid())
                record("ctor_pool", Threads.threadpool())
                model = Chain(Dense(4 => e.width, relu), Dense(e.width => 1))
                return (model, Lux.setup(rng, model)...)
            end

            # A constructor whose body fails must surface its own exception type through the
            # spawn, not `fetch`'s `TaskFailedException` wrapper.
            @experiment struct BrokenExp
                "Unused. Driver-only."
                max_epochs::Int = 1
            end
            ReactantNitro.build_model(e::BrokenExp, rng) = error("boom from build_model")
            ctor_failure = try
                Nitro(BrokenExp(); run_dir = joinpath(mktempdir(), "broken_run"), data = (;
                    train = [(; x = randn(Float32, 4, 8), y = randn(Float32, 1, 8))],
                ))
                "no-error"
            catch e
                string(typeof(e))
            end
            record("ctor_failure_type", ctor_failure)

            function ReactantNitro.forward(e::MiniExp, model, ps, st; x)
                y, st_new = Lux.apply(model, x, ps, st)
                return y, st_new
            end

            function ReactantNitro.loss(e::MiniExp, yhat; y)
                return sum(abs2, yhat .- y) / size(y, 2)
            end

            n = Nitro(MiniExp(); run_dir = joinpath(mktempdir(), "probe_run"), data = (;
                train = [(; x = randn(Float32, 4, 8), y = randn(Float32, 1, 8)) for _ in 1:4],
                val   = [(; x = randn(Float32, 4, 8), y = randn(Float32, 1, 8)) for _ in 1:2],
            ))

            # The spawn path: called from the interactive thread (tid 1 of `-t 4,1`), the body must
            # execute on a default-pool worker.
            body_tid = ReactantNitro.with_repl(n) do
                Threads.threadid()
            end
            record("body_thread", body_tid)

            # A failing body surfaces its OWN exception type through the spawn path, not a
            # `TaskFailedException` wrapper.
            got = try
                ReactantNitro.with_repl(n) do
                    error("boom from the spawned body")
                end
                "no-error"
            catch e
                string(typeof(e))
            end
            record("failure_type", got)

            # And a real `train!` composes with the spawn: the loop itself runs on a worker, and the
            # run completes normally. The phase monitor records the thread the training steps were
            # published from, which is the loop's own thread.
            seen = Int[]
            monitor = function (phase, step, epoch, info)
                phase isa ReactantNitro.Stepping && push!(seen, Threads.threadid())
                return nothing
            end
            h = ReactantNitro.register_phase_monitor!(n, monitor)
            try
                train!(n)
                record("train_threads", join(unique(seen), ","))
                record("train_stop_reason", string(n.stop_reason))
            finally
                ReactantNitro.unregister_phase_monitor!(h)
            end
        """
        cmd = addenv(
            `$(Base.julia_cmd()) -t 4,1 --project=$(Base.active_project()) --startup-file=no -e $probe`,
            "NITRO_PROBE_OUT" => out,
            "CUDA_VISIBLE_DEVICES" => "",
        )
        # The child's stderr is left attached so a failure names itself in the suite's output.
        @test success(pipeline(cmd, stdout = devnull))
        # `isfile` so a child that died before recording anything fails the assertions below rather
        # than erroring out of the testset and taking their diagnosis with it.
        got = Dict{String, String}()
        isfile(out) && for line in eachline(out)
            k, v = split(line, "=", limit = 2)
            got[k] = v
        end
        interactive = parse(Int, get(got, "interactive", "-1"))
        @test interactive == 1                      # the child really is `-t N,1`
        @test parse(Int, get(got, "default", "-1")) > 0
        # The body ran on a worker thread, not the interactive one.
        @test parse(Int, get(got, "body_thread", "-1")) > interactive
        # `Nitro(e)` itself ran its body on a default-pool worker (`_off_interactive`): the thread
        # `build_model` observed is outside the interactive pool, and its pool is `:default`.
        @test parse(Int, get(got, "ctor_thread", "-1")) > interactive
        @test get(got, "ctor_pool", "<missing>") == "default"
        # A failing construction surfaced its own exception type through the spawn.
        @test get(got, "ctor_failure_type", "<missing>") == "ErrorException"
        # A failing body surfaced its own exception type.
        @test get(got, "failure_type", "<missing>") == "ErrorException"
        # The training loop itself ran off the interactive thread and completed normally.
        tids = [parse(Int, s) for s in split(get(got, "train_threads", ""), ",") if !isempty(s)]
        @test !isempty(tids)
        @test all(>(interactive), tids)
        @test get(got, "train_stop_reason", "<missing>") == "completed"
    end

end
