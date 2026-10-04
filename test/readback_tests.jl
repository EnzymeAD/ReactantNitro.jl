# Lagged loss readback tests: the automatic loop reads each micro-batch's loss back one
# micro-batch late, so the device runs the next one while the host waits, and frees each batch and
# emits each train line from that deferred half.
#
# Lagging is invisible when right and silently wrong in three ways: a line logged under the wrong
# step, a micro-batch missing from the epoch mean (the last one, which only the drain completes), or
# a batch never freed. The experiment here makes every logged number known in closed form: its loss
# does not depend on the parameters, and micro-batch k's loss is exactly k^2, so a line's value
# names the micro-batch it came from.

@testitem "readback" begin
    using Test
    using ReactantNitro
    using Lux, Random, Reactant, Statistics

    const FR = Float32
    const N_BATCHES = 6

    # A host matrix whose transfer is recorded, so the test can see every device batch the loop was
    # handed and check that each one was freed. Only this file's batches have the type, so the
    # `place_batch` method below reaches nothing else.
    struct TrackedMat <: AbstractMatrix{FR}
        m::Matrix{FR}
    end
    Base.size(t::TrackedMat) = size(t.m)
    Base.getindex(t::TrackedMat, i::Int...) = t.m[i...]
    const PLACED = Any[]
    const PLACED_LOCK = ReentrantLock()
    function ReactantNitro.place_batch(t::TrackedMat, ::Nothing)
        d = Reactant.to_rarray(t.m)
        lock(() -> push!(PLACED, d), PLACED_LOCK)
        return d
    end
    freed(d) = all(buf -> buf.buffer == C_NULL, ReactantNitro._device_buffers(d))

    # Integer-valued inputs, so `x - (x - k)` is exactly `k` in Float32 and every loss is exactly k^2.
    const RB_X = let rng = Random.MersenneTwister(11)
        [FR.(rand(rng, -3:3, 4, 8)) for _ in 1:N_BATCHES]
    end
    rb_data(; nan_at = 0, tracked = false) = [
        (;
            x = tracked ? TrackedMat(RB_X[k]) : RB_X[k],
            y = k == nan_at ? fill(NaN32, 2, 8) : RB_X[k][1:2, :] .- FR(k),
        )
            for k in 1:N_BATCHES
    ]

    # The same hooks under both train-metric residencies: `:host` is the path whose deferred half
    # transfers the outputs and calls the hook on the HOST batch the record kept.
    @experiment struct RbExp
        width::GraphConst{Int} = 2
    end
    @experiment struct RbHost
        width::GraphConst{Int} = 2
    end
    ReactantNitro.metrics_residency(::RbHost, ::Symbol) = :host
    # The field route of the run keyword's resolution chain.
    @experiment struct RbField
        width::GraphConst{Int} = 2
        check_divergence::Bool = false
    end
    for T in (:RbExp, :RbHost, :RbField)
        @eval begin
            ReactantNitro.build_model(e::$T, rng) =
                (m = Lux.Dense(4 => e.width); (m, Lux.setup(rng, m)...))
            ReactantNitro.forward(::$T, model, ps, st; x) = Lux.apply(model, x, ps, st)
            # Parameter-free in value (the model term is multiplied by zero), so no optimizer step
            # can move a later loss, and micro-batch k's loss is exactly k^2.
            ReactantNitro.loss(::$T, ŷ; x, y) = mean(abs2, x[1:2, :] .- y) + 0 * mean(ŷ)
            # Exactly k, so a line's stats also name the micro-batch they came from.
            ReactantNitro.train_metrics(::$T, ŷ; x, y) = (; k = mean(x[1:2, :] .- y))
            ReactantNitro.learning_rate(::$T) = 1.0f-2
        end
    end

    mutable struct RbLog
        lines::Vector{Any}
        params::Any
    end
    RbLog() = RbLog(Any[], nothing)
    ReactantNitro.log_metrics!(l::RbLog, m; step, epoch, context, kwargs...) =
        (context == "train" && push!(l.lines, (; m, step, epoch)); nothing)
    ReactantNitro.log_params!(l::RbLog, p) = (l.params = p; nothing)
    ReactantNitro.log_tags!(::RbLog, t) = nothing
    ReactantNitro.log_other!(::RbLog, k, v) = nothing
    ReactantNitro.log_confusion!(::RbLog, m, labels; kwargs...) = nothing
    ReactantNitro.finish!(::RbLog, status) = nothing
    ReactantNitro.run_id(::RbLog) = nothing
    ReactantNitro.run_url(::RbLog) = nothing
    ReactantNitro.reattach!(::RbLog, s) = nothing

    rb_nitro(E = RbExp; data = rb_data(), kw...) = Nitro(
        E(); data = (; train = data), logger = RbLog(), checkpointer = nothing,
        run_dir = mktempdir(), kw...
    )

    # What the synchronous loop logged: one line per optimizer step, carrying the step it closed
    # and its CLOSING micro-batch's loss and stats, with `step` counting across epochs.
    expected(accum, epochs) = [
        (; step = (e - 1) * (N_BATCHES ÷ accum) + s, epoch = e, k = s * accum)
            for e in 1:epochs for s in 1:(N_BATCHES ÷ accum)
    ]
    observed(lg) = [(; step = v.step, epoch = v.epoch, k = Int(v.m.k)) for v in lg.lines]

    # Two epochs throughout: the second's steps continue the first's count, and each epoch's last
    # step is the one only the drain emits.
    @testset "train lines: count, step, epoch and value unchanged ($E, accum = $accum)" for E in (RbExp, RbHost), accum in (1, 2, 3)
        epochs = 2
        n = train!(rb_nitro(E; accum, max_epochs = epochs))
        lg = n.logger
        @test observed(lg) == expected(accum, epochs)
        # The loss is the closing micro-batch's, exactly, including each epoch's last step, which
        # only the drain emits.
        @test [v.m.loss for v in lg.lines] == [Float64(x.k)^2 for x in expected(accum, epochs)]
        @test current_step(n) == epochs * (N_BATCHES ÷ accum)
        # The epoch mean sees every micro-batch, the drained last one included: mean(k^2, 1:6).
        @test [r.loss for r in n.history] ≈ fill(sum(abs2, 1:N_BATCHES) / N_BATCHES, epochs)
    end

    # A trainable control: the parameter-free loss above pins attribution; this pins that real,
    # moving losses reach both consumers the same way.
    @experiment struct RbTrain
        width::GraphConst{Int} = 2
    end
    ReactantNitro.build_model(e::RbTrain, rng) =
        (m = Lux.Dense(4 => e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::RbTrain, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::RbTrain, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::RbTrain) = 1.0f-1

    @testset "a trainable model's epoch mean is the mean of its lines (accum = 1)" begin
        n = train!(rb_nitro(RbTrain; max_epochs = 2))
        losses = [v.m.loss for v in n.logger.lines]
        @test length(losses) == 2N_BATCHES
        @test !allequal(losses)                  # the parameters really moved
        @test [r.loss for r in n.history] ≈
            [mean(losses[1:N_BATCHES]), mean(losses[(N_BATCHES + 1):end])]
    end

    @testset "`check_divergence = true` still stops on a non-finite loss, naming the same step" begin
        # Micro-batch 3's loss is NaN. The synchronous loop raised before micro-batch 3's optimizer
        # step, naming the step counter as it stood (2 at accum = 1, 1 at accum = 2); the lagged
        # loop raises one micro-batch later with the counter captured at dispatch, so the message
        # is the same.
        for (accum, step) in ((1, 2), (2, 1))
            n = rb_nitro(; data = rb_data(; nan_at = 3), accum)
            @test n.check_divergence === true          # the default
            err = try
                train!(n)
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("the training loss is NaN at step $step, epoch 1.", err.msg)
            @test phase(n) isa Failed
        end
    end

    @testset "`check_divergence = false` logs through a non-finite loss and trains on" begin
        n = train!(rb_nitro(; data = rb_data(; nan_at = 3), check_divergence = false))
        @test n.stop_reason === :completed
        @test phase(n) isa Done
        @test n.logger.params.check_divergence === false       # recorded with the run's params
        lines = n.logger.lines
        @test [v.step for v in lines] == 1:N_BATCHES
        # `finite_only` drops the NaN loss from step 3's line (and its stats, NaN too, with it);
        # every other line is untouched.
        @test isempty(lines[3].m)
        @test [lines[i].m.loss for i in (1, 2, 4, 5, 6)] == [1.0, 4.0, 16.0, 25.0, 36.0]
        @test isnan(only(n.history).loss)
    end

    @testset "`check_divergence` resolves keyword, then field, then the default" begin
        @test check_divergence(RbExp()) === true
        @test check_divergence(RbField()) === false
        @test rb_nitro(RbField).check_divergence === false
        @test rb_nitro(RbField; check_divergence = true).check_divergence === true
        @test rb_nitro(RbExp; check_divergence = false).check_divergence === false
    end

    @testset "every batch handed to the loop is freed (accum = $accum, epochs = $epochs)" for accum in (1, 2), epochs in (1, 2)
        n = rb_nitro(; data = rb_data(; tracked = true), accum, max_epochs = epochs)
        # Setup may transfer a first batch of its own; only the loop's are in question.
        lock(() -> empty!(PLACED), PLACED_LOCK)
        train!(n)
        placed = lock(() -> copy(PLACED), PLACED_LOCK)
        @test length(placed) == epochs * N_BATCHES
        # Not vacuous: each one really had a device buffer to free.
        @test all(d -> !isempty(ReactantNitro._device_buffers(d)), placed)
        @test all(freed, placed)
    end

    @testset "a requested stop drains the pending micro-batch" begin
        # The stop is requested before the first micro-batch, so the loop breaks right after
        # dispatching it, with its readback still pending: only the drain can log, count and free it.
        n = rb_nitro(; data = rb_data(; tracked = true), max_epochs = 2)
        register_phase_monitor!(n, (p, s, e, info) -> p isa TrainStepping && request_stop!(info.nitro))
        lock(() -> empty!(PLACED), PLACED_LOCK)
        train!(n)
        @test n.stop_reason === :requested
        @test observed(n.logger) == [(; step = 1, epoch = 1, k = 1)]
        @test only(n.history).loss == 1.0
        # The consumed batch, the first transferred on this ordered single-producer path, is freed
        # by the drain. Batches prefetched behind it are the stream close's business, and one still
        # inside the transfer task at the close is left to the GC with or without the lag (checked
        # against the synchronous loop), so they are not asserted here.
        placed = lock(() -> copy(PLACED), PLACED_LOCK)
        @test ReactantNitro.prefetch_config(n.data.train).ordered
        @test !isempty(placed)
        @test freed(first(placed))
    end
end
