# Prefetch tests: depth-N prefetch and its cleanup requirement.
#
# This is the one optional prefetch helper. The mechanism is a `Channel` of depth N with a
# producer task, and the part that needs testing is not the happy path: it is that A PRODUCER THAT
# THROWS SURFACES AT THE CONSUMER RATHER THAN HANGING IT, and that an early exit stops the producer
# and frees the device buffers it is holding. The framework's cleanup contract asks for exactly
# that, and says why: the path only runs when something has already gone wrong.
#
# A hang is the failure mode to fear here, so every test that could hang is written so that a hang
# fails the suite rather than stalling it.

@testitem "prefetch" begin
    using Test
    using ReactantNitro
    using ReactantNitro: PrefetchStream, auto_prefetch, batch_stream, check_batch_at,
        check_prefetch_delivery, close_stream!, default_prefetch_workers, epoch_token, fanout_capable,
        free_batch!, materialized_source, prefetch_config, prefetch_depth, prefetch_ordered,
        prefetch_source, prefetch_workers, routed_fields, to_device_batch
    using Functors, Lux, Optimisers, Random, Reactant, Statistics

    const PF = Float32

    @experiment struct PrefetchMLP
        max_epochs::Host{Int} = 1
    end

    pf_chain() = Lux.Chain(Lux.Dense(3 => 4, tanh), Lux.Dense(4 => 1))
    ReactantNitro.build_model(::PrefetchMLP, rng) = (m = pf_chain(); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::PrefetchMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::PrefetchMLP, ŷ; y) = mean(abs2, ŷ .- y)

    pf_batches(n; seed) = (
        rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, PF, 3, 4), y = randn(rng, PF, 1, 4)) for _ in 1:n]
    )
    const PF_TRAIN = pf_batches(4; seed = 11)
    const PF_VAL = pf_batches(2; seed = 12)

    # The drain is on the producer's clock, not the caller's: `close_stream!` closes the channels and
    # frees what it can, but a producer blocked in `put!` still counts toward `isready` (Julia's
    # `n_avail` adds the putter queue to the buffered data) until its raise clears the queue. An
    # immediate `!isready` assertion therefore races teardown and fails only under load. Poll, bounded,
    # in this file's contract: a hang fails the suite rather than stalling it.
    function pf_drained(f; timeout_s = 5.0)
        t0 = time()
        while time() - t0 < timeout_s
            f() && return true
            sleep(0.02)
        end
        return false
    end

    # A source that yields `k` batches and then throws, which is the "throw mid-epoch" case. It is a
    # source rather than a hook so the throw happens on the PRODUCER's task, which is the whole point.
    struct ThrowingSource
        batches::Vector{Any}
        fail_at::Int
    end
    Base.length(s::ThrowingSource) = length(s.batches)
    function Base.iterate(s::ThrowingSource, i = 1)
        i > length(s.batches) && return nothing
        i == s.fail_at && error("ReactantNitro test: the loader threw at batch $i")
        return (s.batches[i], i + 1)
    end

    # ── The wrapper itself ──────────────────────────────────────────────────────────────

    @testset "`PrefetchIterator` is a declaration, and a valid data source on its own" begin
        src = PF_TRAIN
        p = PrefetchIterator(src, 3)
        @test prefetch_source(p) === src
        @test prefetch_depth(p) == 3
        # Anything else is the no-prefetch case rather than a depth, which is why `0` is not a legal
        # depth to construct.
        @test prefetch_depth(src) == 0
        @test prefetch_source(src) === src
        @test PrefetchIterator(src).depth == 1          # the default depth
        @test PrefetchIterator(src).workers == default_prefetch_workers()
        @test default_prefetch_workers() == max(1, Threads.nthreads(:default))
        @test PrefetchIterator(src, 2; workers = 5).workers == 5
        @test prefetch_workers(src) == 0
        # Ordered delivery is the DEFAULT, and the opt-out is explicit. That default is what makes
        # adopting the index-addressable trait free: it changes a run's throughput and nothing else.
        @test PrefetchIterator(src).ordered
        @test prefetch_ordered(PrefetchIterator(src; ordered = false)) == false
        # Everything that is not a `PrefetchIterator` delivers in the source's order anyway.
        @test prefetch_ordered(src)
        @test prefetch_ordered(NoPrefetch(src))

        @testset "a depth of 0 RAISES and names the marker, rather than meaning `off`" begin
            # The whole point of the rule: running the host data path inline on the training task is
            # an intentionally suboptimal choice, so it must not be reachable by setting a number. The old
            # message advised "to turn prefetch off, do not wrap the source", which is wrong now that
            # setup wraps an unwrapped split itself.
            err = try
                PrefetchIterator(src, 0)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("at least 1", err.msg)
            @test occursin("NoPrefetch", err.msg)
            @test !occursin("do not wrap the source", err.msg)

            werr = try
                PrefetchIterator(src, 1; workers = 0)
            catch ex
                ex
            end
            @test werr isa ErrorException
            @test occursin("NoPrefetch", werr.msg)
        end

        @testset "`NoPrefetch` is the only opt-out, and is a passthrough source" begin
            n = NoPrefetch(src)
            @test prefetch_source(n) === src
            @test prefetch_depth(n) == 0                # the internal "does not stream" sentinel
            @test length(n) == length(src)
            @test collect(n) == collect(src)
            @test prefetch_config(n).path === :inline
        end

        @testset "`iterate` is a PASSTHROUGH, so it leaks no task when iterated by hand" begin
            # The transfer needs the routing and the device, both resolved during setup, and
            # `build_data` runs before that, so this wrapper cannot carry a producer. Its `iterate`
            # therefore yields the source's own batches: same length, same batches, same order, no
            # channel.
            @test length(p) == length(src)
            @test collect(p) == collect(src)
            @test first(p) === first(src)
            @test eltype(p) == eltype(src)
        end
    end

    # ── The stream, both paths, one loop body ───────────────────────────────────────────

    @testset "`batch_stream` yields (host, device) pairs on both paths" begin
        n = Nitro(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        routing = n.routing

        @testset "the inline path is a lazy generator and transfers per batch" begin
            s = batch_stream(PF_TRAIN, routing)
            @test !(s isa Channel)
            pairs = collect(s)
            @test length(pairs) == length(PF_TRAIN)
            host, dev = first(pairs)
            @test host === first(PF_TRAIN)               # the host batch is passed through untouched
            @test dev.x isa Reactant.AbstractConcreteArray
            # Only the routed fields are transferred, which is unchanged by prefetch.
            @test Set(keys(dev)) == Set(routed_fields(routing))
            @test close_stream!(s) === nothing           # a no-op, and it must not raise
        end

        @testset "the prefetch path is a channel of that depth, with the same contents" begin
            s = batch_stream(PrefetchIterator(PF_TRAIN, 2), routing)
            @test s isa Channel
            pairs = collect(s)
            @test length(pairs) == length(PF_TRAIN)
            @test [h for (h, _) in pairs] == collect(PF_TRAIN)
            @test all(d.x isa Reactant.AbstractConcreteArray for (_, d) in pairs)
        end
    end

    # ── the failure path, which is the reason this file exists ──────────────────────────

    @testset "a producer that throws surfaces at the consumer rather than hanging it" begin
        n = Nitro(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        src = ThrowingSource(collect(PF_TRAIN), 3)

        @testset "the inline path raises the loader's own error" begin
            s = batch_stream(src, n.routing)
            @test_throws ErrorException collect(s)
        end

        @testset "the prefetch path raises it too, on the CONSUMER, and does not hang" begin
            # A hang is the failure this asserts against, so it is run under a watchdog: the assertion
            # is that the consumer FINISHES, with the producer's exception, rather than blocking on a
            # channel nobody will feed again.
            s = batch_stream(PrefetchIterator(src, 2), n.routing)
            done = Threads.Atomic{Bool}(false)
            result = Ref{Any}(nothing)
            t = @async begin
                result[] = try
                    collect(s)
                    :no_error
                catch ex
                    ex
                end
                done[] = true
            end
            for _ in 1:200
                done[] && break
                sleep(0.05)
            end
            @test done[]                                  # false here means it hung
            @test result[] isa Exception
            wait(t)
        end
    end

    @testset "cleanup: an early exit stops the producer and frees the buffered batches" begin
        n = Nitro(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        s = batch_stream(PrefetchIterator(PF_TRAIN, 2), n.routing)

        # Take one and walk away, which is what `request_stop!` and a non-finite loss both do.
        _, first_dev = take!(s)
        # Let the producer fill the channel, so there is something buffered to leak.
        for _ in 1:100
            isready(s) && break
            sleep(0.02)
        end
        buffered = isready(s)
        close_stream!(s)

        @test !isopen(s)
        @test buffered                                    # the setup for the assertion below
        # Drained, not just closed, on the producer's clock: the blocked producer's `put!` still counts
        # as `isready` until its raise clears the wait queue, so wait for the drain rather than assert
        # it immediately. This raced under load and never on a quiet machine.
        @test pf_drained(() -> !isready(s))

        @testset "a batch the loop already consumed is NOT freed by the stream" begin
            # XLA execution is asynchronous and an executable holds its inputs, so the loop cannot
            # free at the point the Julia call returns. This batch is still readable after teardown,
            # and that is the designed behavior rather than a leak.
            @test Array(first_dev.x) isa Array
        end
    end

    @testset "`free_batch!` nulls the pointer, so a use-after-free is LOUD" begin
        n = Nitro(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        b = to_device_batch(first(PF_TRAIN), n.routing)
        @test Array(b.x) isa Array                        # live before
        free_batch!(b)

        # `XLA.free_buffer` is a no-op on a null pointer and a DOUBLE FREE on a live one, and the
        # finalizer will run again at GC time, so nulling is what makes one free safe. It also makes the
        # read below raise instead of returning whatever the allocator has since put there: this is the
        # `AssertionError: buffer.buffer !== C_NULL` the checkpoint layer's device-residency fix
        # spent a day on, arriving here on purpose.
        @test all(async.buffer.buffer == C_NULL for async in b.x.data)
        @test_throws Exception Array(b.x)
        @test free_batch!(b) === nothing                  # idempotent, which is what the null buys
    end

    # ── the worker fan-out, and the trait that makes it possible ────────────────────────
    #
    # The fan-out exists because ONE producer cost a 4.2x regression on a real model, and every test here
    # is aimed at a way it could silently deliver the wrong data rather than at the happy path. The order
    # it delivers in is NOT the source's, so every assertion below is written against multisets or against
    # the ledger, never against a sequence.

    # A source that implements BOTH halves of the trait, plus the counter `epoch_token` reads.
    mutable struct IndexedSource
        batches::Vector{Any}
        epochs::Int
        fail_at::Int                       # 0 for never
        off_by_one::Bool                   # simulate `batch_at` taking a sample offset
    end
    IndexedSource(b; fail_at = 0, off_by_one = false) =
        IndexedSource(collect(b), 0, fail_at, off_by_one)

    Base.length(s::IndexedSource) = length(s.batches)
    Base.iterate(s::IndexedSource) = (ReactantNitro.begin_epoch!(s); iterate(s, 1))
    function Base.iterate(s::IndexedSource, i::Int)
        i > length(s.batches) && return nothing
        return (s.batches[i], i + 1)
    end
    ReactantNitro.begin_epoch!(s::IndexedSource) = (s.epochs += 1; nothing)
    ReactantNitro.epoch_token(s::IndexedSource) = s.epochs
    function ReactantNitro.batch_at(s::IndexedSource, i::Integer)
        i == s.fail_at && error("ReactantNitro test: the loader threw at index $i")
        # The off-by-one is the failure mode this file warns about at length: a BATCH index used as
        # a SAMPLE offset. Here it just returns a neighbour, which is enough to be caught while
        # staying in bounds.
        j = s.off_by_one ? mod1(i + 1, length(s.batches)) : i
        return s.batches[j]
    end

    # Half the trait: `batch_at` with no `begin_epoch!`. This must NOT reach the fan-out, because its
    # epoch re-plan would still be in `iterate`'s init and the fan-out never calls `iterate`.
    struct HalfTraitSource
        batches::Vector{Any}
    end
    Base.length(s::HalfTraitSource) = length(s.batches)
    Base.iterate(s::HalfTraitSource, i::Int = 1) =
        i > length(s.batches) ? nothing : (s.batches[i], i + 1)
    ReactantNitro.batch_at(s::HalfTraitSource, i::Integer) = s.batches[i]

    # A source whose `begin_epoch!` forgets to advance its token, which is what the framework's assertion
    # is for: it stands in for a re-plan still living in `iterate`'s initialization.
    mutable struct StuckTokenSource
        batches::Vector{Any}
    end
    Base.length(s::StuckTokenSource) = length(s.batches)
    Base.iterate(s::StuckTokenSource, i::Int = 1) =
        i > length(s.batches) ? nothing : (s.batches[i], i + 1)
    ReactantNitro.begin_epoch!(::StuckTokenSource) = nothing
    ReactantNitro.epoch_token(::StuckTokenSource) = 0
    ReactantNitro.batch_at(s::StuckTokenSource, i::Integer) = s.batches[i]

    # A trait source whose `batch_at` is deliberately slowest on the FIRST index. That skew is what
    # makes the two delivery modes distinguishable: with several workers, batch 1 is still being built
    # while 2..N are finished, so an unordered stream emits those first and an ordered one may not.
    mutable struct SkewedSource
        batches::Vector{Any}
        epochs::Int
    end
    SkewedSource(b) = SkewedSource(collect(b), 0)
    Base.length(s::SkewedSource) = length(s.batches)
    Base.iterate(s::SkewedSource, i::Int = 1) =
        i > length(s.batches) ? nothing : (s.batches[i], i + 1)
    ReactantNitro.begin_epoch!(s::SkewedSource) = (s.epochs += 1; nothing)
    ReactantNitro.epoch_token(s::SkewedSource) = s.epochs
    function ReactantNitro.batch_at(s::SkewedSource, i::Integer)
        i == 1 && sleep(0.25)
        return s.batches[i]
    end

    const PF_MANY = pf_batches(16; seed = 21)

    # Every test that could hang is run under a watchdog, per this file's header: a hang must FAIL the
    # suite rather than stall it.
    function pf_await(f; timeout_s = 20.0)
        done = Threads.Atomic{Bool}(false)
        result = Ref{Any}(nothing)
        t = @async begin
            result[] = try
                f()
            catch ex
                ex
            end
            done[] = true
        end
        t0 = time()
        while !done[] && time() - t0 < timeout_s
            sleep(0.02)
        end
        return (done[], result[], t)
    end

    @testset "the trait is opted into by defining BOTH methods" begin
        @test fanout_capable(IndexedSource(PF_MANY))
        # THE LOAD-BEARING ONE. With `batch_at` alone the fan-out would bypass `iterate`, so an epoch
        # re-plan left in the iteration init would never run and every epoch after the first would train
        # on the first epoch's plan. Requiring both turns that into a loud fallback.
        @test !fanout_capable(HalfTraitSource(collect(PF_MANY)))
        @test !fanout_capable(PF_MANY)                       # a plain Vector opts into nothing
        @test !fanout_capable(Iterators.filter(_ -> true, PF_MANY))

        @testset "`prefetch_config` reports the RESOLVED path, not the requested one" begin
            @test prefetch_config(PrefetchIterator(IndexedSource(PF_MANY), 2; workers = 4)) ==
                (; depth = 2, workers = 4, ordered = true, path = :fanout)
            # The opt-out is a different resolved path, so the report and the warning can tell the
            # two apart without carrying the flag around separately.
            @test prefetch_config(
                PrefetchIterator(IndexedSource(PF_MANY), 2; workers = 4, ordered = false)
            ) == (; depth = 2, workers = 4, ordered = false, path = :fanout_unordered)
            # Asked for 4, got 1, and the label says which case it is so the setup warning can differ.
            @test prefetch_config(PrefetchIterator(HalfTraitSource(collect(PF_MANY)); workers = 4)) ==
                (; depth = 1, workers = 1, ordered = true, path = :single_no_trait)
            @test prefetch_config(PrefetchIterator(IndexedSource(PF_MANY); workers = 1)).path === :single
            @test prefetch_config(PF_MANY).path === :inline   # unwrapped: no stream at all
        end

        # The false positive this path exists to remove. A `Vector` of batches `build_data` already
        # built has no host work for N producers to spread, so resolving to one producer is the right
        # answer and saying so as a warning would teach people to ignore the real one.
        @testset "a materialized `Vector` source resolves to its own path, not to a complaint" begin
            @test prefetch_config(PrefetchIterator(PF_MANY; workers = 8)) ==
                (; depth = 1, workers = 1, ordered = true, path = :materialized)
            @test ReactantNitro.materialized_source(PF_MANY)
            # Narrow on purpose: a custom `AbstractVector` can compute in `getindex`, so it is not
            # covered and still hears the warning.
            @test !ReactantNitro.materialized_source(HalfTraitSource(collect(PF_MANY)))
            @test !ReactantNitro.materialized_source(IndexedSource(PF_MANY))
            # It decides the WARNING only. A `Vector` that implements the trait still fans out.
            @test prefetch_config(PrefetchIterator(IndexedSource(PF_MANY); workers = 4)).path === :fanout
        end
    end

    # ── delivery order, which is what makes the trait free to adopt ─────────────────────
    #
    # The reorder buffer is the whole point: before it, opting into `batch_at` changed a run's
    # trajectory, because reordering repartitions the epoch into different accumulation groups and a
    # fixed seed no longer reproduced bitwise. That made the one real throughput knob something a
    # user had to trade reproducibility for. Ordered delivery is now the default and the trade is
    # explicit.
    @testset "ordered fan-out delivers the SOURCE's order, whatever the workers do" begin
        n = Nitro(PrefetchMLP(); run_dir = mktempdir(), data = (; train = PF_TRAIN, val = PF_VAL))

        # The skew guarantees the workers FINISH out of order: index 1 sleeps while 2, 3 and 4 are
        # already done. An ordered stream must still emit 1 first. This is a one-directional
        # assertion and therefore not a timing test: ordered order is the source's order regardless
        # of who finished when.
        @testset "with a source whose first batch is the slowest" begin
            src = SkewedSource(PF_MANY)
            split = PrefetchIterator(src, 2; workers = 4)
            @test prefetch_config(split).path === :fanout
            ok, res, t = pf_await() do
                collect(batch_stream(split, n.routing))
            end
            @test ok                                        # false here means it hung
            @test res isa Vector
            @test [h for (h, _) in res] == collect(PF_MANY)
            wait(t)
        end

        # The credit window is `2 x workers`, so this exercises the branch where the whole epoch is
        # smaller than the window and the coordinator takes fewer credits than the channel holds.
        @testset "an epoch shorter than the credit window still completes" begin
            src = SkewedSource(PF_TRAIN)
            ok, res, t = pf_await() do
                collect(batch_stream(PrefetchIterator(src, 2; workers = 8), n.routing))
            end
            @test ok
            @test [h for (h, _) in res] == collect(PF_TRAIN)
            wait(t)
        end

        # The opt-out. Deliberately NOT asserting that the order differs: that would be a timing
        # assertion and the kind of flake this file is written to avoid. What must hold either way is
        # the invariant the ledger checks, that every batch is delivered exactly once.
        @testset "the unordered opt-out still delivers every batch exactly once" begin
            src = SkewedSource(PF_MANY)
            split = PrefetchIterator(src, 2; workers = 4, ordered = false)
            @test prefetch_config(split).path === :fanout_unordered
            ok, res, t = pf_await() do
                collect(batch_stream(split, n.routing))
            end
            @test ok
            @test length(res) == length(PF_MANY)
            @test sort([findfirst(==(h), collect(PF_MANY)) for (h, _) in res]) == collect(1:length(PF_MANY))
            wait(t)
        end
    end

    @testset "`auto_prefetch` wraps the TRAIN split, once, and only that split" begin
        c = auto_prefetch((; train = PF_TRAIN, val = PF_VAL))
        @test c.train isa PrefetchIterator
        @test prefetch_source(c.train) === PF_TRAIN
        # `run_eval` iterates its split directly and never enters `batch_stream`, so a wrapped eval split
        # would advertise a worker count nothing uses.
        @test c.val === PF_VAL

        @testset "an explicit wrap wins entirely, and is never double-wrapped" begin
            p = PrefetchIterator(PF_TRAIN, 7; workers = 3)
            c2 = auto_prefetch((; train = p, val = PF_VAL))
            @test c2.train === p
            @test prefetch_depth(c2.train) == 7 && prefetch_workers(c2.train) == 3
        end

        @testset "`NoPrefetch` survives the auto-wrap, which is the whole point of it" begin
            n = NoPrefetch(PF_TRAIN)
            @test auto_prefetch((; train = n)).train === n
        end

        @testset "a collection with no train split is returned untouched" begin
            c3 = (; val = PF_VAL)
            @test auto_prefetch(c3) === c3
        end
    end

    @testset "the fan-out delivers every batch EXACTLY ONCE, in some order" begin
        n = Nitro(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        src = IndexedSource(PF_MANY)
        s = batch_stream(PrefetchIterator(src, 2; workers = 4), n.routing, n.mesh)
        @test s isa PrefetchStream

        ok, pairs, t = pf_await(() -> collect(s))
        @test ok                                            # false here means it hung
        wait(t)
        @test pairs isa Vector
        @test length(pairs) == length(PF_MANY)
        hosts = [h for (h, _) in pairs]
        # A MULTISET comparison, deliberately: with N workers the order is not the source's, and asserting
        # a sequence here would be asserting something the design says is not true.
        @test Set(objectid.(hosts)) == Set(objectid.(PF_MANY))
        @test all(d.x isa Reactant.AbstractConcreteArray for (_, d) in pairs)
        # The exactly-once ledger, which catches the right COUNT delivered with one index twice and
        # another never -- the one thing `check_epoch_length` cannot see.
        @test check_prefetch_delivery(s) === nothing
        close_stream!(s)

        @testset "`begin_epoch!` fires exactly once per stream, BEFORE any job" begin
            @test epoch_token(src) == 1
            s2 = batch_stream(PrefetchIterator(src, 2; workers = 4), n.routing, n.mesh)
            @test epoch_token(src) == 2
            close_stream!(s2)
        end

        @testset "a token that does not advance is an ERROR naming the stale-plan failure" begin
            # This stands in for the mistake that would otherwise be silent: an epoch re-plan still living
            # in `Base.iterate`'s initialization, which the index path never calls.
            stuck = PrefetchIterator(StuckTokenSource(collect(PF_MANY)), 2; workers = 4)
            err = try
                batch_stream(stuck, n.routing, n.mesh)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("epoch_token", err.msg)
            @test occursin("NEVER CALLS", err.msg)
        end

        @testset "a source with only `batch_at` falls back to one producer, not to a stale plan" begin
            s3 = batch_stream(
                PrefetchIterator(HalfTraitSource(collect(PF_MANY)), 2; workers = 4),
                n.routing, n.mesh
            )
            @test s3 isa Channel && !(s3 isa PrefetchStream)
            ok3, pairs3, t3 = pf_await(() -> collect(s3))
            @test ok3
            wait(t3)
            # One producer preserves the source's order, which is what makes this fallback safe.
            @test [h for (h, _) in pairs3] == collect(PF_MANY)
            close_stream!(s3)
        end
    end

    @testset "a worker that throws surfaces at the consumer and does not hang" begin
        n = Nitro(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        # Index 5 of 16, so several workers are mid-flight and the coordinator still has jobs to place:
        # the case an earlier framework's comments record as the deadlock, where the coordinator
        # blocks in `put!`
        # on a full job channel that nobody will drain again.
        src = IndexedSource(PF_MANY; fail_at = 5)
        s = batch_stream(PrefetchIterator(src, 2; workers = 4), n.routing, n.mesh)

        ok, result, t = pf_await(() -> collect(s))
        @test ok                                            # false here means it hung
        wait(t)
        @test result isa Exception
        # Unwrapped, so the consumer sees the loader's own error rather than a `TaskFailedException`,
        # matching the inline path's semantics that this file asserts by type above.
        @test result isa ErrorException
        @test occursin("threw at index 5", result.msg)
        close_stream!(s)
    end

    @testset "cleanup: an early exit stops every task, drains the device channel, and returns" begin
        n = Nitro(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        s = batch_stream(PrefetchIterator(IndexedSource(PF_MANY), 2; workers = 4), n.routing, n.mesh)

        # Take one and walk away, which is what `request_stop!` and a non-finite loss both do.
        _, first_dev = take!(s.devch)
        for _ in 1:200
            isready(s.devch) && break
            sleep(0.02)
        end
        buffered = isready(s.devch)

        # Teardown must not block on a worker that is mid-flight, so it is timed rather than merely run.
        t0 = time()
        close_stream!(s)
        @test time() - t0 < 5.0

        @test buffered                                      # the setup for the assertion below
        @test !isopen(s.devch)
        # Drained, not just closed, on the transfer task's clock: its blocked `put!` still counts as
        # `isready` until the close raises it out of the wait queue, so wait for the drain rather than
        # assert it immediately (the same race the single-producer cleanup test had).
        @test pf_drained(() -> !isready(s.devch))
        @test !isopen(s.jobs) && !isopen(s.hostch)

        @testset "a batch the loop already consumed is NOT freed by the stream" begin
            # This narrowing, unchanged by the fan-out: XLA execution is asynchronous and an
            # executable holds its inputs, so the stream frees only what it still owns.
            @test Array(first_dev.x) isa Array
        end
    end

    @testset "`check_batch_at` catches the index-units mistake, which nothing else can" begin
        # The one failure mode with no runtime assertion available: a permuted or overlapping index set
        # produces perfectly well-shaped batches at every index, so only a comparison against the source's
        # own sequential pass can see it.
        @test check_batch_at(IndexedSource(PF_MANY); values = true) === nothing
        @test check_batch_at(IndexedSource(PF_MANY); values = false) === nothing

        err = try
            check_batch_at(IndexedSource(PF_MANY; off_by_one = true); values = true)
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("DIFFERS in value", err.msg)
        # The message wraps, so match either side of the line break rather than across it.
        @test occursin("SAMPLE", err.msg) && occursin("offset rather than a BATCH index", err.msg)

        @testset "shape-only checking cannot see it, and the docstring says so" begin
            # Same-shaped batches, so `values = false` passes on the broken source. This is asserted so
            # that nobody upgrades a model's test to the cheap form thinking it is equivalent.
            @test check_batch_at(IndexedSource(PF_MANY; off_by_one = true); values = false) === nothing
        end

        @testset "a source that opts into neither half is refused rather than silently passed" begin
            err2 = try
                check_batch_at(PF_MANY)
            catch ex
                ex
            end
            @test err2 isa ErrorException
            @test occursin("does not implement both", err2.msg) && occursin("halves", err2.msg)
        end
    end

    # ── End to end ──────────────────────────────────────────────────────────────────────

    @testset "a wrapped train split trains, and matches the unwrapped run exactly" begin
        # Prefetch changes WHEN the transfer happens and nothing about what is computed, so the two runs
        # must agree bitwise. Same seed, same data, same everything else.
        #
        # NOTE what this asserts after auto-wrap: `plain`'s split is wrapped by setup too, so both runs are
        # prefetched. They still agree bitwise because a `Vector` of batches implements neither half of the
        # index-addressable trait, so both resolve to ONE producer and one producer preserves order. That
        # is the backward-compatibility story in one assertion: nothing that exists today opts in, so
        # nothing that exists today changes its trajectory.
        plain = train!(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PF_TRAIN, val = PF_VAL)
        )
        pref = train!(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PrefetchIterator(PF_TRAIN, 2), val = PF_VAL)
        )

        @test current_epoch(pref) == current_epoch(plain) == 1
        @test current_step(pref) == current_step(plain) == length(PF_TRAIN)
        for (a, b) in zip(Functors.fleaves(parameters(plain)), Functors.fleaves(parameters(pref)))
            @test Array(a) == Array(b)
        end
        @test validate(pref).val_loss == validate(plain).val_loss

        # THE PAYOFF OF ORDERED DELIVERY, and the assertion the reorder buffer exists for. A source
        # that DOES implement the trait, fanned out over four workers, must reach the same weights
        # as the single-producer run. Before ordered delivery this was false by design: the fan-out
        # delivered as workers finished, which repartitions the epoch into different accumulation
        # groups, so opting into `batch_at` cost bitwise reproducibility. Adopting the trait now
        # changes throughput and nothing else.
        fanned = train!(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PrefetchIterator(SkewedSource(PF_TRAIN), 2; workers = 4), val = PF_VAL)
        )
        @test current_step(fanned) == current_step(plain)
        for (a, b) in zip(Functors.fleaves(parameters(plain)), Functors.fleaves(parameters(fanned)))
            @test Array(a) == Array(b)
        end
        @test validate(fanned).val_loss == validate(plain).val_loss

        @testset "and the data-source contract still names the source, through the wrapper" begin
            # `check_data_source` is given `prefetch_source`, so a bad loader is reported as itself
            # rather than as a `PrefetchIterator` the user did not write the guts of.
            # `Iterators.filter` rather than a generator: a generator over a `Vector` DOES have
            # `length`, so it would have sailed through the check this asserts.
            lengthless = Iterators.filter(_ -> true, PF_TRAIN)
            @test !applicable(length, lengthless)
            err = try
                Nitro(
                    PrefetchMLP(); run_dir = mktempdir(),
                    data = (; train = PrefetchIterator(lengthless, 2), val = PF_VAL)
                )
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("does not support `length`", err.msg)
        end
    end

    @testset "a fan-out train split trains end to end, and sees every batch once" begin
        # The fan-out's trajectory is NOT the sequential one, because reordering repartitions the epoch
        # into different accumulation groups, so this asserts the invariants that DO hold: the epoch is
        # complete, the step count is right, and the run finishes through the normal path.
        src = IndexedSource(PF_MANY)
        fan = train!(
            PrefetchMLP(); run_dir = mktempdir(),
            data = (; train = PrefetchIterator(src, 2; workers = 4), val = PF_VAL)
        )
        @test current_epoch(fan) == 1
        @test current_step(fan) == length(PF_MANY)          # accum 1, so one step per batch
        # TWO, not one, and that is pre-existing behavior rather than a double re-plan: setup draws
        # one batch through `first(source)` to learn the batch schema, which goes through
        # `Base.iterate` and therefore through `begin_epoch!`. It is why a resumed loader may carry
        # a `presynced` flag.
        # The framework's own assertion is about ONE stream advancing the token by exactly one, which
        # `check_epoch_advanced` checked on the way in.
        @test epoch_token(src) == 2
        @test validate(fan).val_loss isa Real
    end

end
