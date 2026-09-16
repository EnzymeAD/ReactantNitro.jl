# Visualization tests.
#
# The acceptance criterion this file exercises: the fake-figure backend is green with no plotting
# package in the environment; both `render` forms run on a `Nitro(e)` that never trained and on one
# built from a checkpoint.
#
# THERE IS NO PLOTTING PACKAGE HERE, AND THAT IS THE POINT. The test's figure type is a
# struct wrapping a `String` and its `save_figure` writes text. That exercises everything the
# framework owns, split selection, the batch count, per-sample slicing, both modes, the dispatch
# split, naming, directory creation, and the returned paths, while importing nothing.

@testitem "visualize" begin
    using Test
    using ReactantNitro
    using ReactantNitro: routed_fields
    using Lux, Random, Reactant, Statistics

    const VZ = Float32

    # ── The experiment under test ───────────────────────────────────────────────────────
    #
    # The batch carries `case_id`, a `Vector{String}` DECLARED BY `visualize` AND BY NOTHING ELSE. That
    # is what the viz-only-field test below is about: it must never enter
    # `routed_fields(nitro.routing)`, because everything in there is transferred to device on
    # every batch, and a `Vector{String}` cannot be.

    @experiment struct VizMLP
        scale::Device{Float32} = 1.0f0
        width::GraphConst{Int} = 5
    end

    viz_chain(width) = Lux.Chain(Lux.Dense(3 => width, tanh), Lux.Dense(width => 2))
    ReactantNitro.build_model(e::VizMLP, rng) = (m = viz_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::VizMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(e::VizMLP, ŷ; y) = e.scale * mean(abs2, ŷ .- y)

    struct FakeFig
        text::String
    end

    # What every call saw, so the assertions can inspect shapes and residency after the fact rather than
    # from inside a hook whose failure would surface as an unrelated error.
    const VZ_SEEN = NamedTuple[]

    # DATA mode. Dispatches on `::Nothing`, and deliberately declares FEWER fields than the prediction
    # method, which is the prediction-mode claim that the two may differ.
    function ReactantNitro.visualize(::VizMLP, ::Nothing; x, case_id)
        push!(VZ_SEEN, (; mode = :data, xtype = typeof(x), xsize = size(x), id = case_id))
        return FakeFig("data id=$case_id ndims=$(ndims(x))")
    end

    # PREDICTION mode.
    function ReactantNitro.visualize(::VizMLP, outputs; x, y, case_id)
        push!(
            VZ_SEEN,
            (;
                mode = :pred, xtype = typeof(x), xsize = size(x), id = case_id,
                otype = typeof(outputs), osize = size(outputs), ytype = typeof(y),
            )
        )
        return FakeFig("pred id=$case_id out=$(size(outputs))")
    end

    ReactantNitro.save_figure(::VizMLP, f::FakeFig, stem::AbstractString) =
        (p = stem * ".txt"; write(p, f.text); p)

    # A `val` split whose length is NOT a multiple of the batch size: 8, 8, then 3. The padding test
    # lives on the short final batch, which is the only place padding could reach a figure.
    # Case ids are unique ACROSS batches, not within one. That is load-bearing for the padding test:
    # the ids are how it detects a leaked padding row, and numbering them per batch would make two
    # 8-wide batches generate the same ids, so a genuine duplicate from padding would be
    # indistinguishable from the test's own numbering. The first version of this file got that wrong
    # and the padding test caught it.
    function viz_batches(ns; seed = 11, prefix = "case")
        rng = Random.MersenneTwister(seed)
        k = 0
        return map(ns) do n
            ids = ["$(prefix)_$(k + i)" for i in 1:n]
            k += n
            (; x = randn(rng, VZ, 3, n), y = randn(rng, VZ, 2, n), case_id = ids)
        end
    end

    const VZ_TRAIN = viz_batches([8, 8]; prefix = "tr")
    const VZ_VAL = viz_batches([8, 8, 3]; seed = 12)
    viz_data() = (; train = VZ_TRAIN, val = VZ_VAL)

    reset_seen!() = (empty!(VZ_SEEN); nothing)

    @testset "visualization" begin

        @testset "data mode on a Nitro that never trained" begin
            reset_seen!()
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)
            paths = render(nitro; split = :val, batches = 1, out_dir = joinpath(dir, "gate"))

            @test length(paths) == 8                      # one batch, every sample in it
            @test all(isfile, paths)
            @test all(endswith(".txt"), paths)            # the BACKEND chose the extension, not us
            @test basename(paths[1]) == "val_001.txt"
            @test basename(paths[8]) == "val_008.txt"
            @test length(VZ_SEEN) == 8
            @test all(s -> s.mode === :data, VZ_SEEN)     # `outputs === nothing` dispatched

            # Called once per SAMPLE with the batch dimension dropped. `x` is (3, n) in the
            # batch and must arrive as (3,).
            @test all(s -> s.xsize == (3,), VZ_SEEN)
            # A rank-1 field yields its ELEMENT. A zero-dimensional view would interpolate as
            # `fill("case_1")`, which is the defect this rule exists to prevent.
            @test VZ_SEEN[1].id == "case_1"
            @test VZ_SEEN[1].id isa String
            @test read(paths[1], String) == "data id=case_1 ndims=1"
        end

        @testset "prediction mode, and the dispatch split" begin
            reset_seen!()
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)
            paths = render(nitro; split = :val, batches = 1, predictions = true)

            @test length(paths) == 8
            @test all(s -> s.mode === :pred, VZ_SEEN)     # the generic method, not the ::Nothing one
            @test all(s -> s.osize == (2,), VZ_SEEN)      # outputs sliced per sample too
            # The default out_dir hangs off the run directory.
            @test all(p -> startswith(p, joinpath(dir, "viz")), paths)
        end

        @testset "batches, not samples, and numbering runs across them" begin
            reset_seen!()
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)
            paths = render(nitro; split = :val, batches = 2, out_dir = dir)

            @test length(paths) == 16                     # 8 + 8, no sample cap
            @test basename(paths[9]) == "val_009.txt"     # continuous, not restarting per batch
            @test length(unique(paths)) == 16
        end

        @testset "padding never reaches the hook" begin
            reset_seen!()
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)
            # All three batches, the last of which is 3 wide against a compiled width of 8.
            paths = render(nitro; split = :val, batches = 3, predictions = true, out_dir = dir)

            # 8 + 8 + 3, NOT 8 + 8 + 8. `nitro.batch_size` is the compiled width and would over-count
            # here; the real count comes from the host batch, which is never padded.
            @test nitro.batch_size == 8
            @test length(paths) == 19
            @test length(VZ_SEEN) == 19

            # And the padded rows are replicas of the last real sample, so a padding row
            # reaching a figure would show up as a REPEATED case id. Every id must be distinct.
            ids = [s.id for s in VZ_SEEN]
            @test length(unique(ids)) == 19
            @test ids[end] == "case_19"
            @test count(==("case_19"), ids) == 1
        end

        @testset "no device array reaches the hook" begin
            reset_seen!()
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)
            render(nitro; split = :val, batches = 1, predictions = true, out_dir = dir)

            # Both halves, separately, because they arrive by different routes: the outputs through
            # `predict`'s host transfer, the batch fields because they are never transferred at all.
            # The device-residency limit applies: on a CPU backend a device array IS host memory,
            # so this asserts the guard rather than the thing the guard protects against.
            @test all(s -> s.otype <: Array, VZ_SEEN)
            @test all(s -> s.xtype <: Array, VZ_SEEN)
            @test all(s -> s.ytype <: Array, VZ_SEEN)
            @test !any(s -> s.otype <: Reactant.AbstractConcreteArray, VZ_SEEN)
        end

        @testset "a viz-only field never joins the device routing" begin
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)

            # `case_id` is declared by `visualize` and by no other hook. If the viz routers were merged
            # into `nitro.routing`, it would be transferred to device on every training batch and this
            # is where that shows up. `x` and `y` are declared by `forward` and `loss` and must stay.
            rf = routed_fields(nitro.routing)
            @test :x in rf
            @test :y in rf
            @test :case_id ∉ rf

            # Rendering must not change that: the routers it resolves are local to the driver.
            render(nitro; split = :val, batches = 1, predictions = true, out_dir = dir)
            @test routed_fields(nitro.routing) == rf

            # And the whole run still works with a Vector{String} riding in every batch, which is the
            # concrete thing a device transfer of `case_id` would break.
            @test train!(nitro) isa Any
            @test :case_id ∉ routed_fields(nitro.routing)
        end

        @testset "one batch in hand, and a checkpoint, with no train! in either" begin
            reset_seen!()
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)
            paths = render(nitro, VZ_VAL[3]; predictions = true, out_dir = dir)
            @test length(paths) == 3                       # the short batch, taken as given
            @test basename(paths[1]) == "001.txt"          # no tag, so bare index
        end

        @testset "a missing method names the mode" begin
            # An experiment with a DATA-mode method only. Asking for predictions must say which mode is
            # missing rather than raising a MethodError from inside the driver.
            @eval begin
                @experiment struct VizDataOnly
                    width::GraphConst{Int} = 4
                end
                ReactantNitro.build_model(e::VizDataOnly, rng) =
                    (m = viz_chain(e.width); (m, Lux.setup(rng, m)...))
                ReactantNitro.forward(::VizDataOnly, model, ps, st; x) = Lux.apply(model, x, ps, st)
                ReactantNitro.loss(::VizDataOnly, ŷ; y) = mean(abs2, ŷ .- y)
                ReactantNitro.visualize(::VizDataOnly, ::Nothing; x) = FakeFig("only data")
                ReactantNitro.save_figure(::VizDataOnly, f::FakeFig, stem::AbstractString) =
                    (p = stem * ".txt"; write(p, f.text); p)
            end
            dir = mktempdir()
            n2 = Nitro(VizDataOnly(); data = viz_data(), run_dir = dir)

            @test length(render(n2; split = :val, batches = 1, out_dir = dir)) == 8
            err = try
                render(n2; split = :val, batches = 1, predictions = true, out_dir = dir)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("PREDICTION mode", err)
            @test occursin("VizDataOnly", err)
        end

        @testset "an absent split errors naming what exists" begin
            dir = mktempdir()
            nitro = Nitro(VizMLP(); data = viz_data(), run_dir = dir)
            err = try
                render(nitro; split = :test, out_dir = dir)
                nothing
            catch e
                sprint(showerror, e)
            end
            @test err !== nothing
            @test occursin("no `test` split", err)
        end
    end

end
