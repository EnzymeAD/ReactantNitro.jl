# Eval tests: entry points and metric residency.
#
# The acceptance criterion this file exercises: all three entry points green on a `Nitro(e)` that
# never trained. The two metric residencies add to that, which is why the traced path and the host
# path are both exercised here and asserted to agree.

@testitem "eval" begin
    using Test
    using ReactantNitro
    using ReactantNitro: cache_reset!, cache_stats, free_device_buffers!, protected_buffers
    using Functors, Lux, Random, Reactant, Statistics

    const FV = Float32

    # ── The experiment under test, and one batch collection with a SHORT FINAL BATCH ────

    @experiment struct EvalMLP
        "Traced input, so the eval programs take the same stripped view the training ones do."
        scale::Device{Float32} = 1.0f0
        width::GraphConst{Int} = 6
    end

    eval_chain(width) = Lux.Chain(Lux.Dense(4 => width, tanh), Lux.Dense(width => 2))
    ReactantNitro.build_model(e::EvalMLP, rng) = (m = eval_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::EvalMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(e::EvalMLP, ŷ; y) = e.scale * mean(abs2, ŷ .- y)

    # Records the batch width every call sees, which is how this file asserts `metrics` is never
    # handed a padded row. It can do this because the default residency is `:host`: the hook is
    # ordinary Julia, called once per batch, rather than traced once per shape.
    const SEEN_WIDTHS = Int[]
    function ReactantNitro.metrics(::EvalMLP, ŷ; y)
        push!(SEEN_WIDTHS, size(ŷ, ndims(ŷ)))
        return (;
            mae = (sum(abs, ŷ .- y), length(y)),
            rows = (size(ŷ, ndims(ŷ)), nothing),
        )     # By default, `count === nothing` sums, never divides
    end

    eval_batches(ns; seed = 7) =
        (
        rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, FV, 4, n), y = randn(rng, FV, 2, n)) for n in ns]
    )

    const TRAIN_B = eval_batches([8, 8, 8, 8])
    const VAL_B = eval_batches([8, 8, 3]; seed = 11)      # THE SHORT FINAL BATCH is the point
    const TEST_B = eval_batches([8, 5]; seed = 13)
    ReactantNitro.build_data(::EvalMLP, dist) = (; train = TRAIN_B, val = VAL_B, test = TEST_B)

    mk(E = EvalMLP; kw...) = Nitro(E(); checkpointer = nothing, run_dir = mktempdir(), kw...)

    @testset "eval tests: all three entry points run on a `Nitro(e)` that never trained" begin
        cache_reset!()
        empty!(SEEN_WIDTHS)
        n = mk()
        @test current_epoch(n) == 0                       # nothing requires training

        v = validate(n)
        @test keys(v) == (:mae, :rows)
        @test v.mae isa Real && isfinite(v.mae)

        t = evaluate(n; split = :test)
        @test t.rows == 13                                # 8 + 5, summed rather than divided

        p = predict(n, VAL_B[1])
        @test size(p) == (2, 8)
        @test p isa Array{FV}                             # HOST arrays: the caller is leaving the run

        # The phase is restored, not left on `EvalStepping`: `validate` is a verb a user calls outside a
        # run too, and it has no business advancing a handle's phase past where the run left it.
        @test phase(n) isa Starting
    end

    @testset "pad-then-slice is EXACT, and `metrics` never sees a padded row" begin
        cache_reset!()
        n = mk()

        # The unpadded reference: the same weights, over a split holding only the short batch, so the
        # batch size is inferred as 3 and nothing is padded anywhere on this path.
        ref = Nitro(
            EvalMLP(); data = (; val = [VAL_B[3]]), checkpointer = nothing,
            run_dir = mktempdir()
        )
        ref.ps, ref.st = n.ps, n.st
        @test ref.batch_size == 3
        @test n.batch_size == 8

        ŷ_padded = predict(n, VAL_B[3])                   # padded to 8, run, sliced back to 3
        ŷ_plain = predict(ref, VAL_B[3])                  # run at 3, never padded
        @test size(ŷ_padded) == (2, 3)
        # The pad-and-slice claim: numerically exact to the precision of the two programs, and NOT
        # bitwise, on any backend including CPU. Measured on this very model: widths 5 and 7 are
        # bitwise, widths 1, 2, and 3 differ by up to 7.6e-7 relative, and repeating either program
        # is bitwise, so it is the shape and not nondeterminism. An exact tie here would be a test
        # that fails on a correct framework.
        @test ŷ_padded ≈ ŷ_plain rtol = 1.0e-5

        empty!(SEEN_WIDTHS)
        m = validate(n)
        # The whole basis for allowing the padding: `metrics` never sees a padded row, so there is no
        # mask for a user to forget to apply and no silent miscount to guard against.
        @test SEEN_WIDTHS == [8, 8, 3]

        ŷ1, ŷ2 = predict(n, VAL_B[1]), predict(n, VAL_B[2])
        sums = sum(abs, ŷ1 .- VAL_B[1].y) + sum(abs, ŷ2 .- VAL_B[2].y) + sum(abs, ŷ_plain .- VAL_B[3].y)
        counts = length(VAL_B[1].y) + length(VAL_B[2].y) + length(VAL_B[3].y)
        # By default, sums add, counts add, divide at the end. A tolerance for the same reason the
        # comparison above needs one, and no looser: the three per-batch sums are accumulated in
        # the same order the framework accumulates them, so only the short batch's last bits differ.
        @test m.mae ≈ sums / counts rtol = 1.0e-5
        @test m.rows == 19                                # 8 + 8 + 3, and NOT 24: no padded row counted
    end

    # ── wrong-axis refusal ───────────────────────────────────────────────────────────────

    @experiment struct EvalWrongAxis
        width::GraphConst{Int} = 6
    end
    ReactantNitro.build_model(e::EvalWrongAxis, rng) =
        (m = eval_chain(e.width); (m, Lux.setup(rng, m)...))
    # The batch dimension is FIRST here, which is exactly the shape the framework says must fail
    # loudly rather than have its feature dimension sliced.
    function ReactantNitro.forward(::EvalWrongAxis, model, ps, st; x)
        ŷ, st_new = Lux.apply(model, x, ps, st)
        return (permutedims(ŷ), st_new)
    end
    ReactantNitro.loss(::EvalWrongAxis, ŷ; y) = mean(abs2, ŷ .- permutedims(y))
    ReactantNitro.build_data(::EvalWrongAxis, dist) = (; train = TRAIN_B, val = VAL_B)

    @testset "`predict` slices to n_real, and refuses to slice the wrong axis" begin
        n = mk()
        @test size(predict(n, VAL_B[3])) == (2, 3)        # exactly `n_real` outputs, not `batch_size`
        @test size(predict(n, VAL_B[1])) == (2, 8)

        w = mk(EvalWrongAxis)
        # A full batch never slices, so it is not rejected: the assertion is made where it protects
        # something, which is the short final batch.
        @test size(predict(w, VAL_B[1])) == (8, 2)
        err = try
            predict(w, VAL_B[3])
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("last dimension", err.msg)
        @test occursin("(8, 2)", err.msg)                 # names the leaf's actual shape
    end

    # ── program sharing, and the two metric residencies ─────────────────────────────────

    @testset "inference, validation, and testing share ONE compiled `forward`" begin
        cache_reset!()
        n = mk()
        predict(n, VAL_B[1])
        @test cache_stats().misses == 1                   # the eval forward, and nothing else

        validate(n)                                       # `metrics` is host-resident: in NO program
        evaluate(n; split = :test)
        predict(n, VAL_B[3])                              # the short batch is padded to the same shape
        predict(n, (; x = VAL_B[1].x))                    # and a batch with no labels routes the same
        @test cache_stats().misses == 1
        @test cache_stats().hits > 0
    end

    @experiment struct EvalTraced
        width::GraphConst{Int} = 6
    end
    ReactantNitro.build_model(e::EvalTraced, rng) = (m = eval_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::EvalTraced, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::EvalTraced, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.metrics(::EvalTraced, ŷ; y) = (; mae = (sum(abs, ŷ .- y), length(y)))
    # Residency is the user's choice, per hook. This is the traced half.
    ReactantNitro.metrics_residency(::EvalTraced, ::Symbol) = :device
    ReactantNitro.build_data(::EvalTraced, dist) = (; train = TRAIN_B, val = VAL_B)

    @testset "the traced metric path agrees with the host one, and costs two shapes" begin
        cache_reset!()
        n = mk(EvalTraced)
        m = validate(n)
        st = cache_stats()
        # The cost paragraph: `forward` compiles ONCE, and `metrics` twice, at `batch_size` and at
        # the remainder, because slicing to `n_real` changes the shape reaching it.
        @test st.misses == 3

        validate(n)
        @test cache_stats().misses == 3                   # and never again

        h = mk()                                          # the same weights, `metrics` on the host
        ref = validate(h)
        # Not asserted bitwise: a device reduction and a host one may sum in different orders, which is
        # a different claim from the pad-and-slice exactness asserted above.
        @test m.mae ≈ ref.mae rtol = 1.0e-5
    end

    # ── the metrics substitution, for an experiment defining only the four required hooks ──

    @experiment struct EvalBare
        width::GraphConst{Int} = 6
    end
    ReactantNitro.build_model(e::EvalBare, rng) = (m = eval_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::EvalBare, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::EvalBare, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.build_data(::EvalBare, dist) = (; train = TRAIN_B, val = VAL_B)

    @testset "with no `metrics` method the framework reports the validation loss" begin
        n = mk(EvalBare)
        @test n.routing.metrics === nothing                # no method: the substitution keys off this
        m = validate(n)
        @test keys(m) == (:val_loss,)                      # the name two other defaults already use

        # `count` is 1, so this is the mean over BATCHES rather than over samples: a stated
        # inexactness, and the reference has to be computed the same way to be a reference at all.
        per_batch = [mean(abs2, predict(n, b) .- b.y) for b in VAL_B]
        @test m.val_loss ≈ mean(per_batch) rtol = 1.0e-5
    end

    # ── no-train variants ────────────────────────────────────────────────────────────────

    @testset "a `Nitro` with no data at all defers routing to the first `predict`" begin
        n = Nitro(EvalMLP(); data = (;), checkpointer = nothing, run_dir = mktempdir())
        @test n.routing === nothing                        # nothing to resolve it from at setup
        @test n.batch_size === nothing
        @test n.opt_state === nothing                      # step 9 skipped
        @test n.layout !== nothing                         # step 8 still runs
        @test_throws ErrorException validate(n)

        ŷ = predict(n, VAL_B[3])
        @test size(ŷ) == (2, 3)
        @test keys(n.routing.forward) == (:x,)
        # With no split, `batch_size` is the width of the batch `predict` was handed, and no
        # padding is needed because the caller supplied that batch whole.
        @test n.batch_size == 3
    end

    @testset "`evaluate` names the splits that exist, and `predict` names `forward`'s fields" begin
        n = mk()
        err = try
            evaluate(n; split = :nope)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("(:train, :val, :test)", err.msg)

        # A bare array is iterable, so without a method of its own it would reach the loader form and
        # fail somewhere with no mention of routing.
        err2 = try
            predict(n, VAL_B[1].x)
            nothing
        catch ex
            ex
        end
        @test err2 isa ErrorException
        @test occursin("x = ...", err2.msg)

        err3 = try
            predict(n, (; z = VAL_B[1].x))
            nothing
        catch ex
            ex
        end
        @test err3 isa ErrorException
        @test occursin("(:x,)", err3.msg)
    end

    @testset "`predict` over a loader is LAZY, one element per batch" begin
        n = mk()
        pulled = Ref(0)
        source = (
            begin
                pulled[] += 1
                b
            end for b in VAL_B
        )
        it = predict(n, source)
        @test pulled[] == 0                                # nothing has run yet
        outs = collect(it)
        @test pulled[] == 3
        @test length(outs) == 3
        @test size(outs[3]) == (2, 3)                      # sliced, like the single-batch form
    end

    # ── device-buffer freeing ────────────────────────────────────────────────────────────

    @testset "each batch's device buffers are freed eagerly, and live ones never are" begin
        n = mk()
        a = validate(n)
        b = validate(n)
        # If the eager free reached a parameter, this would read freed memory rather than repeat.
        @test a == b

        prot = protected_buffers(n)
        x = Reactant.to_rarray(randn(FV, 4, 2))
        free_device_buffers!(prot, (; x = x))
        @test all(d -> d.buffer.buffer == C_NULL, getfield(x, :data))

        live = first(Functors.fleaves(parameters(n)))
        free_device_buffers!(prot, (; live = live))
        @test all(d -> d.buffer.buffer != C_NULL, getfield(live, :data))
        @test Array(live) isa Array                        # and still readable
    end

    # ── residency's other half: `train_metrics` on the host ─────────────────────────────

    const TM_RESIDENCY = Ref(:device)
    const TM_SEEN = Ref(0)
    const TM_TYPES = DataType[]

    @experiment struct EvalTM
        width::GraphConst{Int} = 6
    end
    ReactantNitro.build_model(e::EvalTM, rng) = (m = eval_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::EvalTM, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::EvalTM, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::EvalTM) = 1.0f-2
    ReactantNitro.build_data(::EvalTM, dist) = (; train = TRAIN_B)
    ReactantNitro.metrics_residency(::EvalTM, hook::Symbol) =
        hook === :train_metrics ? TM_RESIDENCY[] : :host
    function ReactantNitro.train_metrics(::EvalTM, ŷ; y)
        TM_SEEN[] += 1
        push!(TM_TYPES, typeof(ŷ))
        return (; mean_abs = sum(abs, ŷ .- y) / length(y))
    end

    @testset "`train_metrics = :host` runs in Julia and does NOT change the gradient" begin
        TM_RESIDENCY[] = :device
        TM_SEEN[] = 0
        empty!(TM_TYPES)
        device_run = train!(mk(EvalTM))
        # Traced: the hook is called at TRACE time, once, and what it saw was a traced array.
        @test TM_SEEN[] == 1
        @test all(t -> t <: Reactant.TracedRArray, TM_TYPES)

        TM_RESIDENCY[] = :host
        TM_SEEN[] = 0
        empty!(TM_TYPES)
        host_run = train!(mk(EvalTM))
        # Host: once per MICRO-BATCH, on ordinary Julia arrays. That transfer is the cost of
        # choosing host residency, and the user chose by asking for it.
        @test TM_SEEN[] == 4
        @test all(t -> t <: Array, TM_TYPES)

        # The load-bearing half. Under `:host` the gradient program returns the PRIMAL as its fourth
        # result instead of the stats, which puts an `ignore_derivatives`-barriered array on the loss's
        # own path. If that barrier leaked, the seeds would reach `dps` and corrupt the gradient with no
        # error, exactly as the framework's state-barrier discipline describes for `st_new`.
        for (dl, hl) in zip(
                Functors.fleaves(parameters(device_run)),
                Functors.fleaves(parameters(host_run))
            )
            @test Array(dl) ≈ Array(hl) rtol = 1.0e-6
        end
    end

    # ── the metrics contract, checked rather than assumed ───────────────────────────────

    @experiment struct EvalBadMetric
        width::GraphConst{Int} = 6
    end
    ReactantNitro.build_model(e::EvalBadMetric, rng) =
        (m = eval_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::EvalBadMetric, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::EvalBadMetric, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.metrics(::EvalBadMetric, ŷ; y) = (; mae = sum(abs, ŷ .- y))   # a bare number, not a pair
    ReactantNitro.build_data(::EvalBadMetric, dist) = (; train = TRAIN_B, val = VAL_B)

    @testset "a metric that is not a `(sum, count)` pair is named, not accumulated" begin
        n = mk(EvalBadMetric)
        err = try
            validate(n)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("mae", err.msg)
        @test occursin("(sum, count)", err.msg)
    end

    # ── the host/device boundary assertion ──────────────────────────────────────────────
    #
    # The device-residency fix's finding, and the reason this section can exist at all: on a CPU
    # backend a "device" array IS host memory and scalar indexing into one is legal rather than
    # fatal, so the BEHAVIOUR these tests would like to check is unobservable here. What is
    # perfectly observable is the TYPE, and the mechanism keys on the type. So the assertion and
    # the conversion are CPU-testable even though every defect they exist to catch is not, which
    # is the one place a CPU test can speak about this boundary.

    @testset "`assert_host` names the path of a device leaf" begin
        ok = (; a = randn(FV, 2, 2), b = (; c = 1.0f0))
        @test ReactantNitro.assert_host(ok, "a test value") === ok    # host values pass through unchanged

        bad = (; a = randn(FV, 2, 2), b = (; c = Reactant.to_rarray(randn(FV, 2, 2))))
        err = try
            ReactantNitro.assert_host(bad, "a test value")
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("a test value", err.msg)       # names the CROSSING
        @test occursin(".b.c", err.msg)               # and the PATH, which is the whole point
    end

    @testset "`call_host_hook` converts BOTH sides of the call" begin
        # The regression this exists for. The framework used to convert only the outputs it owns and
        # hand the routed batch fields through exactly as the loader produced them, so one call could
        # pass a host `ŷ` and a device `y`. On a GPU the hook died in `argmax` on whichever one it
        # touched first; on CPU nothing raised, which is why no test caught it.
        router = ReactantNitro.Router{(:y,)}()
        batch = (; x = randn(FV, 4, 3), y = Reactant.to_rarray(randn(FV, 2, 3)))
        seen = Ref{Any}(nothing)
        hook = (_e, out; y) -> (seen[] = y; out)

        out = ReactantNitro.call_host_hook(hook, :metrics, router, batch, nothing, randn(FV, 2, 3))
        @test seen[] isa Array{FV}
        @test !(seen[] isa Reactant.AbstractConcreteArray)
        @test out isa Array{FV}

        # The experiment is EXEMPT and must be: a `Device` field is device-resident by design, so
        # asserting over `e` would fire on every run of every experiment that has one.
        e_dev = ReactantNitro.to_device_config(EvalMLP())
        @test ReactantNitro.device_paths(e_dev, "e") != []      # it really is device-resident
        @test ReactantNitro.call_host_hook(
            (_e, out; y) -> out, :metrics, router, batch, e_dev, randn(FV, 2, 3)
        ) isa Array{FV}
    end

    @experiment struct EvalLeakyMetric
        width::GraphConst{Int} = 6
    end
    ReactantNitro.build_model(e::EvalLeakyMetric, rng) =
        (m = eval_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::EvalLeakyMetric, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::EvalLeakyMetric, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.metrics(::EvalLeakyMetric, ŷ; y) = (; mae = (sum(abs, ŷ .- y), length(y)))
    # The mistake: a user hook handing a DEVICE value back to the framework. Nothing downstream can use
    # it, and every consumer of a finalized metric (the logger, the phase monitors, the checkpoint
    # metric) assumes host.
    ReactantNitro.finalize_metrics(::EvalLeakyMetric, acc, ::Symbol) =
        (; mae = Reactant.to_rarray(Float32[acc.mae]))
    ReactantNitro.build_data(::EvalLeakyMetric, dist) = (; val = VAL_B)

    @testset "a device value RETURNED by a hook is refused, naming it" begin
        n = mk(EvalLeakyMetric)
        err = try
            validate(n)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("finalize_metrics", err.msg)
        @test occursin("mae", err.msg)
        # And the message says which way to fix it, since a value the framework never saw is the one
        # case the framework cannot convert for you.
        @test occursin("hook RETURNED", err.msg)
    end

    # The gap `assert_host` surfaced: the framework had TWO walkers claiming to read a tree back to
    # host, and only one of them could see inside a struct. `to_host` was fixed by the
    # device-residency fix; `host_tree`, which feeds every `:host` metric hook and `predict`, was
    # `Functors.fmap` and treated an unregistered struct as a leaf. CPU-testable for the usual
    # reason: the TYPE moves even though the behaviour does not.
    struct BoxedOutput{A}
        inner::A
        tag::Symbol
    end

    @testset "`host_tree` reaches inside a struct, as `to_host` does" begin
        dev = Reactant.to_rarray(randn(FV, 2, 3))
        boxed = BoxedOutput(dev, :logits)
        @test ReactantNitro.device_paths(boxed, "out") != []     # the defect's precondition

        out = ReactantNitro.host_tree(boxed)
        @test out isa BoxedOutput
        @test out.inner isa Array{FV}
        @test !(out.inner isa Reactant.AbstractConcreteArray)
        @test out.tag === :logits                                # structure preserved, not flattened
        @test ReactantNitro.assert_host(out, "a test value") === out

        # Nested one level down, and inside a NamedTuple, which is how a real `outputs` arrives.
        nested = (; a = BoxedOutput((; b = dev), :inner), c = randn(FV, 2))
        @test ReactantNitro.assert_host(ReactantNitro.host_tree(nested), "a test value") isa NamedTuple

        # Identity preservation: nothing device-resident means the SAME object back, so the common case
        # allocates nothing and a struct that could not be rebuilt is never rebuilt.
        host_only = BoxedOutput(randn(FV, 2, 3), :already_host)
        @test ReactantNitro.host_tree(host_only) === host_only
        plain = (; a = randn(FV, 2), b = 3)
        @test ReactantNitro.host_tree(plain) === plain
        @test ReactantNitro.host_tree("a string") === "a string"
    end

    # The THIRD appearance of one shape: a wrapper AROUND a device array, rather than a device array.
    # `vec(::ConcretePJRTArray)` returning a `ReshapedArray` is what hid the `_concat_group` defect that
    # made the framework unable to build its flat parameter buffer on a GPU at all. Then `view` and
    # `reshape` hid a device array from BOTH the host converter and the assertion written to catch the
    # converter, so `assert_host` passed and the hook died in `argmax` one line later.
    #
    # CPU-testable for the usual reason: `SubArray{Float32,2,ConcretePJRTArray}` is not `Matrix{Float32}`
    # whatever the memory underneath, so the TYPE distinction is visible here even though the behaviour
    # is not.
    @testset "a device array behind a WRAPPER is found and converted" begin
        d = Reactant.to_rarray(rand(FV, 4, 3))
        host = rand(FV, 4, 3)

        @testset "device_paths follows the wrapper" begin
            for (name, w) in ("view" => view(d, :, 1:2), "reshape" => reshape(d, 3, 4), "vec" => vec(d))
                paths = ReactantNitro.device_paths(w, "x")
                @test !isempty(paths)                       # the regression: this was String[]
                @test occursin("parent", paths[1])          # and it says HOW it got there
            end
            # Nested, which is how one actually arrives: inside the outputs a hook is handed.
            @test !isempty(ReactantNitro.device_paths((; a = (; b = view(d, :, 1:2))), "out"))
            # A wrapper over HOST memory is not a finding.
            @test isempty(ReactantNitro.device_paths(view(host, :, 1:2), "x"))
        end

        @testset "host_tree converts through the wrapper, exactly" begin
            @test ReactantNitro.host_tree(view(d, :, 1:2)) == Array(d)[:, 1:2]
            @test ReactantNitro.host_tree(reshape(d, 3, 4)) == reshape(Array(d), 3, 4)
            @test ReactantNitro.host_tree(vec(d)) == vec(Array(d))
            @test ReactantNitro.host_tree(view(d, :, 1:2)) isa Matrix{FV}
            @test !(ReactantNitro.host_tree(vec(d)) isa SubArray)
        end

        @testset "and the assertion now catches what the converter would miss" begin
            err = try
                ReactantNitro.assert_host((; outputs = view(d, :, 1:2)), "a test value")
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("SubArray", err.msg)             # names the wrapper, not just "a device value"
        end

        @testset "host memory is left ALONE, identity included" begin
            # The conversion must not copy what is already host: `slice_last` hands views around on the
            # eval path and rebuilding them every batch would be pure waste.
            @test ReactantNitro.host_tree(host) === host
            v = view(host, :, 1:2)
            @test ReactantNitro.host_tree(v) === v
            @test ReactantNitro.host_tree((; a = host)) === (; a = host)
        end

        @testset "an unknown wrapper REFUSES rather than passing a device array through" begin
            struct OddWrap{T, N, P} <: AbstractArray{T, N}
                parent::P
            end
            Base.parent(w::OddWrap) = w.parent
            Base.size(w::OddWrap) = size(w.parent)
            w = OddWrap{FV, 2, typeof(d)}(d)
            @test !isempty(ReactantNitro.device_paths(w, "x"))    # still FOUND, which is the point
            err = try
                ReactantNitro.host_tree(w)
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("OddWrap", err.msg)
            @test occursin("_rewrap", err.msg)                    # and says how to add it
        end
    end

end
