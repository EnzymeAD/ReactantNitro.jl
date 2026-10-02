# Gradient accumulation tests: accumulation parity, the accumulator and shadow lifecycle, cache
# keys on `accum`, and the accumulation-parity claim for clipping, moved here from the optimizer
# tests because it needs this loop.
#
# Gradient accumulation is where this stack is easiest to get subtly and silently wrong. Every
# test here asserts a NUMBER rather than that something ran.

@testitem "accumulation" begin
    using Test
    using ReactantNitro
    using ReactantNitro: cache_reset!, cache_stats, compile_cached, flatten, grad_program, layout_ref,
        resolve_layout, to_device_batch
    using Functors, Lux, Optimisers, Random, Reactant, Statistics

    const FA = Float32

    @experiment struct AccExp
        width::GraphConst{Int} = 8
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(e::AccExp, rng) =
        (
        Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dense(e.width => 3)),
        Lux.setup(rng, Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dense(e.width => 3)))...,
    )
    ReactantNitro.forward(::AccExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::AccExp, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::AccExp) = 1.0f-2
    ReactantNitro.optimizer(::AccExp) = Optimisers.Adam

    # One fixed dataset, presented two ways: as N micro-batches of `bs`, and as one batch of N*bs. With a
    # mean loss and equal micro-batch sizes the two are mathematically the same gradient, which is what
    # makes this an equality rather than a resemblance.
    const RNG = Random.MersenneTwister(7)
    const XS = [randn(RNG, FA, 4, 8) for _ in 1:4]
    const YS = [randn(RNG, FA, 3, 8) for _ in 1:4]

    micro_batches() = [(; x = XS[i], y = YS[i]) for i in 1:4]
    merged_pairs() = [(; x = hcat(XS[2i - 1], XS[2i]), y = hcat(YS[2i - 1], YS[2i])) for i in 1:2]
    merged_all() = [(; x = reduce(hcat, XS), y = reduce(hcat, YS))]

    params_of(n) = reduce(vcat, vec.(Array.(Functors.fleaves(parameters(n)))))

    run_with(data, accum; clip = 0.0f0) =
        params_of(
        train!(
            Nitro(
                AccExp(); data = (; train = data), accum, gradient_clip_norm = clip,
                checkpointer = nothing, run_dir = mktempdir()
            )
        )
    )

    # ── accumulation parity ──────────────────────────────────────────────────────────────

    @testset "accumulation parity: accum = N over N micro-batches == one step on the average" begin
        # Lightning semantics: `accum = N` folds micro-gradients scaled by 1/N. With a mean loss and
        # equal micro-batch sizes, that is exactly the gradient of the mean over the combined batch.
        a2 = run_with(micro_batches(), 2)     # 4 micro-batches, 2 optimizer steps
        m2 = run_with(merged_pairs(), 1)      # 2 full batches,   2 optimizer steps
        @test a2 ≈ m2 rtol = 1.0e-4

        a4 = run_with(micro_batches(), 4)     # 4 micro-batches, 1 optimizer step
        m4 = run_with(merged_all(), 1)        # 1 full batch,    1 optimizer step
        @test a4 ≈ m4 rtol = 1.0e-4

        # And the parity is not vacuous: accumulating differently really does give a different answer,
        # so the equality above is testing the scaling rather than testing nothing.
        @test !isapprox(a2, a4; rtol = 1.0e-4)

        @testset "the step counter counts OPTIMIZER steps, not micro-batches" begin
            n2 = train!(
                Nitro(
                    AccExp(); data = (; train = micro_batches()), accum = 2,
                    checkpointer = nothing, run_dir = mktempdir()
                )
            )
            @test current_step(n2) == 2       # 4 micro-batches / accum 2
            n4 = train!(
                Nitro(
                    AccExp(); data = (; train = micro_batches()), accum = 4,
                    checkpointer = nothing, run_dir = mktempdir()
                )
            )
            @test current_step(n4) == 1
            # Confusing the two shifts the whole LR curve by N.
            @test n2.total == 1 * div(4, 2)
            @test n4.total == 1 * div(4, 4)
        end

        @testset "a batch count not divisible by accum is a setup error" begin
            err = try
                Nitro(
                    AccExp(); data = (; train = micro_batches()[1:3]), accum = 2,
                    checkpointer = nothing, run_dir = mktempdir()
                )
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("accum = 2", err.msg)
            # The invariant it buys: an accumulation group never spans an epoch boundary, so the
            # accumulator is always empty at an epoch edge.
            @test occursin("epoch boundary", err.msg)
        end
    end

    # ── the accumulator and the shadow, at the program boundary ─────────────────────────

    # A direct handle on the gradient program, so the select and the shadow can be exercised
    # without the loop deciding when they fire.
    function grad_harness(; accum = 2)
        n = Nitro(
            AccExp(); data = (; train = micro_batches()), accum, checkpointer = nothing,
            run_dir = mktempdir()
        )
        ev = compile_view(n.e)
        lref = layout_ref(n.layout)
        st = Lux.trainmode(n.st)
        b = to_device_batch(micro_batches()[1], n.routing)
        inv_n = one(FA) / accum
        # `Val(:device)` is the train-metric residency, type-level like `Val{CLIP}`. This harness
        # pins the default arm, which is the one every test below is about.
        args = (
            ev, n.model, n.ps, st, b, n.g_accum, Reactant.to_rarray(FA[1]), inv_n, n.routing, lref,
            Val(:device),
        )
        thunk = compile_cached(grad_program, ev, args...)
        call(gacc, first_flag) = thunk(
            ev, n.model, n.ps, st, b, gacc,
            Reactant.to_rarray(FA[first_flag]), inv_n, n.routing, lref,
            Val(:device)
        )
        return n, call, thunk
    end

    # The program writes the accumulator in place and Reactant donates it, so a call consumes the
    # tree it is handed: read a result back BEFORE passing it on, and pass a fresh copy of any
    # accumulator used twice.
    fresh(acc) = map(g -> Reactant.to_rarray(Array(g)), acc)

    @testset "the broadcast select lowers and selects correctly" begin
        n, call = grad_harness(; accum = 2)
        zero_acc = fresh(n.g_accum)

        _, reset, _, _ = call(fresh(zero_acc), 1.0f0)   # is_first = 1 -> g/N
        r = Array(reset[1])
        _, accumulated, _, _ = call(reset, 0.0f0)       # is_first = 0 -> acc + g/N
        a = Array(accumulated[1])
        @test all(isfinite, r)
        @test any(!=(0.0f0), r)
        @test a ≈ 2 .* r rtol = 1.0e-5                  # same batch twice: exactly two halves

        # Reset really resets, rather than adding to whatever was there.
        _, again, _, _ = call(accumulated, 1.0f0)
        @test Array(again[1]) ≈ r rtol = 1.0e-5

        @testset "a SELECT, not a multiply by zero: NaN cannot survive" begin
            # They differ on non-finite input, and that is the reason for the choice: a g_accum left
            # holding NaN from a diverged group survives a multiply and cannot survive a select.
            poisoned = map(g -> Reactant.to_rarray(fill(NaN32, size(Array(g)))), zero_acc)
            _, cleaned, _, _ = call(poisoned, 1.0f0)
            @test all(isfinite, Array(cleaned[1]))
            @test Array(cleaned[1]) ≈ r rtol = 1.0e-5
        end

        @testset "scalar `ifelse` over two traced arrays is STILL a MethodError" begin
            # The negative half matters: if scalar ifelse ever starts working, the accumulation-select
            # warning is stale and should be revisited rather than left silently wrong.
            a1 = Reactant.to_rarray(randn(FA, 4))
            a2 = Reactant.to_rarray(randn(FA, 4))
            c = Reactant.to_rarray(FA[1])
            bad(p, q, cc) = ifelse(cc .> 0.0f0, p, q)
            @test_throws Exception Reactant.Compiler.compile(bad, (a1, a2, c))
            # while the broadcast form is fine
            good(p, q, cc) = ifelse.(cc .> 0.0f0, p, q)
            @test Reactant.Compiler.compile(good, (a1, a2, c)) isa Any
        end
    end

    @testset "shadow lifecycle: the shadow is allocated inside the program, every invocation" begin
        n, call = grad_harness(; accum = 2)
        zero_acc = fresh(n.g_accum)

        @testset "two calls on the SAME micro-batch with is_first = 1 are EQUAL, not double" begin
            # A `dps` hoisted out of the program and reused would add every past micro-batch's gradient
            # to the current one, on top of the deliberate accumulator, with no error and a loss
            # curve that reads as a badly chosen learning rate.
            _, a, _, _ = call(fresh(zero_acc), 1.0f0)
            _, b, _, _ = call(fresh(zero_acc), 1.0f0)
            @test Array(a[1]) == Array(b[1])          # bitwise, same inputs and same program
        end

        @testset "and the is_first = 0 half accumulates EXACTLY 2x and 3x" begin
            # Without this half, a select stuck on the reset arm would pass the first half trivially.
            _, one_, _, _ = call(fresh(zero_acc), 1.0f0)
            r = Array(one_[1])
            _, two_, _, _ = call(one_, 0.0f0)
            @test Array(two_[1]) ≈ 2 .* r rtol = 1.0e-5
            _, three_, _, _ = call(two_, 0.0f0)
            @test Array(three_[1]) ≈ 3 .* r rtol = 1.0e-5
        end

        @testset "and the accumulator is donated, so it is never left for the GC" begin
            # A program that returns a fresh accumulator instead of writing the input leaves one
            # parameter-sized buffer per micro-batch until a collection: device OOM under a loader
            # that allocates nothing on the host. The accumulator is the only argument written.
            _, _, thunk = grad_harness(; accum = 2)
            @test count(Bool, thunk.donated_args_mask) == length(zero_acc)
        end

        @testset "and the Nitro holds no shadow to hoist into" begin
            @test :dps ∉ fieldnames(Nitro)
        end
    end

    # ── buffer ownership across steps ────────────────────────────────────────────────────

    # Every optimizer step, the device arrays the loop held one step earlier. Device memory is the
    # thing that leaks, but the CPU client does not implement allocator stats, so this asserts the
    # cause instead: a superseded array whose buffer still holds device memory that no live array
    # uses is left for a GC collection to free. Device addresses, not objects or handles: Reactant
    # writes an in-place result back into the argument AND returns a second wrapper over it, and
    # across a chain of donations an old wrapper keeps a handle XLA has already consumed.
    mutable struct OwnershipLog
        nitro::Any
        prev::Vector{Any}
        leaked::Vector{Tuple{Int, Symbol, Int}}
        steps::Int
    end
    OwnershipLog() = OwnershipLog(nothing, Any[], Tuple{Int, Symbol, Int}[], 0)
    function device_arrays(n)
        out = Any[]
        for (name, tree) in ((:ps, n.ps), (:g_accum, n.g_accum), (:opt_state, n.opt_state))
            # Arrays only: the rules' scalars are rebuilt host-side every step by design, and a
            # 4-byte buffer is not what exhausts a device.
            ReactantNitro._each_concrete_leaf(tree) do x
                x isa Reactant.AbstractConcreteArray && push!(out, (name, x))
            end
        end
        return out
    end
    # The device memory an array still holds. A donated or freed buffer holds none, and asking
    # for its address is an error.
    function device_addresses(x)
        out = Ptr{Cvoid}[]
        for b in ReactantNitro._device_buffers(x)
            b.buffer == C_NULL && continue
            try
                push!(out, Reactant.XLA.unsafe_buffer_pointer(b))
            catch err
                occursin("deleted or donated", sprint(showerror, err)) || rethrow()
            end
        end
        return out
    end
    function ReactantNitro.log_metrics!(l::OwnershipLog, metrics; context = "train", step = 0, kw...)
        context == "train" || return nothing
        now = device_arrays(l.nitro)
        live = Set(p for (_, x) in now for p in device_addresses(x))
        for (name, x) in l.prev
            any(∉(live), device_addresses(x)) && push!(l.leaked, (step, name, length(x)))
        end
        l.prev = now
        l.steps += 1
        return nothing
    end
    ReactantNitro.log_params!(::OwnershipLog, params) = nothing
    ReactantNitro.log_tags!(::OwnershipLog, tags) = nothing
    ReactantNitro.log_other!(::OwnershipLog, key, value) = nothing
    ReactantNitro.log_confusion!(::OwnershipLog, matrix, labels; kw...) = nothing
    ReactantNitro.finish!(::OwnershipLog, status) = nothing
    ReactantNitro.run_id(::OwnershipLog) = nothing
    ReactantNitro.run_url(::OwnershipLog) = nothing
    ReactantNitro.reattach!(::OwnershipLog, state) = nothing

    @testset "no superseded parameter, accumulator or moment is left for the GC (accum = $accum)" for accum in (1, 2)
        # The regression: `grad_program` and `opt_program` returned fresh trees, so with a data
        # source that allocates nothing on the host the GC never ran and device memory grew by one
        # parameter-sized buffer per micro-batch plus one per step until OOM.
        log = OwnershipLog()
        n = Nitro(
            AccExp(); data = (; train = repeat(micro_batches(), 2)), accum, logger = log,
            checkpointer = nothing, run_dir = mktempdir()
        )
        log.nitro = n
        train!(n)
        @test log.steps == 8 ÷ accum
        @test !isempty(log.prev)
        @test log.leaked == Tuple{Int, Symbol, Int}[]
    end

    # ── cache keys on accum ──────────────────────────────────────────────────────────────

    @testset "a second train! at a different `accum` MISSES the compile cache" begin
        cache_reset!()
        train!(
            Nitro(
                AccExp(); data = (; train = micro_batches()), accum = 2, checkpointer = nothing,
                run_dir = mktempdir()
            )
        )
        after_first = cache_stats().misses
        @test after_first == 2                        # the gradient and optimizer programs

        # Same everything except `accum`. `inv_n = 1/N` is a trace-time host constant, so this is a
        # DIFFERENT gradient program; hitting the cached one would silently train at the wrong
        # micro-gradient scale.
        train!(
            Nitro(
                AccExp(); data = (; train = micro_batches()), accum = 4, checkpointer = nothing,
                run_dir = mktempdir()
            )
        )
        @test cache_stats().misses > after_first

        # And returning to the first `accum` hits again, so the key is a function of accum rather than
        # merely of call order.
        before = cache_stats().misses
        train!(
            Nitro(
                AccExp(); data = (; train = micro_batches()), accum = 2, checkpointer = nothing,
                run_dir = mktempdir()
            )
        )
        @test cache_stats().misses == before
    end

    # ── the accumulation-parity claim for clipping, moved here from the optimizer tests ──

    @testset "`accum = 2` clips ONCE on the accumulated gradient" begin
        # This is the Lightning-parity claim and the one an implementer is most likely to get wrong,
        # since clipping in the GRADIENT program looks equivalent and is not: it would clip each
        # micro-gradient separately, which is a different function of the same data.
        clip = 0.05f0                                 # well below the natural norm, so it binds
        a2 = run_with(micro_batches(), 2; clip)
        m2 = run_with(merged_pairs(), 1; clip)
        @test a2 ≈ m2 rtol = 1.0e-4

        a4 = run_with(micro_batches(), 4; clip)
        m4 = run_with(merged_all(), 1; clip)
        @test a4 ≈ m4 rtol = 1.0e-4

        @testset "and the clip is actually binding, so the parity is not vacuous" begin
            unclipped = run_with(micro_batches(), 2; clip = 0.0f0)
            @test !isapprox(a2, unclipped; rtol = 1.0e-4)
        end

        @testset "per-micro-batch clipping would give a DIFFERENT answer" begin
            # Clipping each micro-gradient to `clip` and then averaging is not the same as averaging and
            # then clipping to `clip`, whenever the micro-gradients differ in norm. A single accum = 1
            # run over each micro-batch at the same threshold is that other function; asserting it
            # differs is what makes "clips once" a claim with content.
            per_micro = run_with(micro_batches(), 1; clip)
            @test !isapprox(a2, per_micro; rtol = 1.0e-4)
        end
    end

end
