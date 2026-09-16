# Export tests: the ReactantServerExport bundle boundary.
#
# THE BUNDLE ITSELF CANNOT BE TESTED HERE, and that limit is stated rather than papered over. Writing
# one needs the backend package, a real StableHLO trace, and a server to load the result; none of
# those belong in a CPU suite that runs in CI without an accelerator.
#
# What CAN be tested is everything on the framework's side of the backend seam, which is where
# defects would be: the hook validation, the wire-to-batch adapter, the output selection, the
# batch-last verification the backend's own axis derivation depends on, and the provenance. The fake
# backend below stands in for a real one by doing exactly what a tracer does with what it is handed,
# so a change that breaks a real export breaks this too.

@testitem "export" begin
    using Test
    using ReactantNitro
    using ReactantNitro: ExportBackend, check_export_batch_last, check_export_derived_only
    using ReactantNitro: device_paths, host_rngs, StrippedHost
    using Lux, Random
    using Reactant: ReactantRNG

    const XV = Float32

    # ── The fake backend ────────────────────────────────────────────────────────────────

    struct FakeBundle <: ExportBackend end

    # One recording per export, so a test reads what the framework decided rather than inferring it.
    const RECORDED = Ref{Any}(nothing)

    function ReactantNitro.write_export(
            ::FakeBundle, program, ps, st, example_inputs;
            dir, name, input_names, output_names, output_select,
            client_inputs, client_outputs, postprocess, batch_sizes, provenance
        )
        # Do what a tracer does: run the program on the example inputs and select. This is the part that
        # makes the fake worth having. A recording backend that only stored its keywords would pass while
        # the adapter and the selection were both broken.
        raw = first(program(length(example_inputs) == 1 ? example_inputs[1] : example_inputs, ps, st))
        selected = output_select(raw)
        RECORDED[] = (;
            dir, name, input_names, output_names, selected, raw, program,
            client_inputs, client_outputs, postprocess, batch_sizes, provenance, example_inputs, ps, st,
        )
        return joinpath(String(dir), String(name))
    end

    # ── The experiments, all at top level: `@experiment` declares a struct ──────────────

    @experiment struct ExportMLP
        width::GraphConst{Int} = 5
        num_classes::GraphConst{Int} = 3
    end

    export_chain(e::ExportMLP) = Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dense(e.width => e.num_classes))
    ReactantNitro.build_model(e::ExportMLP, rng) = (m = export_chain(e); (m, Lux.setup(rng, m)...))

    # `logits` is deliberately a VIEW. A head whose output arrives through a view chain serializes as its
    # producer's shape unless it is copied at the export boundary, and the boundary copy makes that
    # the framework's job; this is the fixture that proves it does it. `energy` is the training-only leaf that must NOT
    # ship, which is what `export_outputs` naming a subset is for.
    function ReactantNitro.forward(::ExportMLP, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = @view(y[:, :]), energy = sum(abs2, y; dims = 1)), st2
    end
    ReactantNitro.loss(::ExportMLP, o; y) = sum(abs2, o.logits .- y)

    ReactantNitro.export_inputs(::ExportMLP) = [ExportSpec("img", UInt8, [4, 1])]
    ReactantNitro.export_outputs(::ExportMLP) = [ExportSpec("logits")]
    ReactantNitro.export_preprocess(::ExportMLP, img) = (; x = XV.(img) ./ 255.0f0)

    # A model carrying a `Dropout`, whose `st` therefore holds a `Reactant.ReactantRNG` after setup
    # converts it. This is a REGRESSION FIXTURE: `export_model` host-ifies `st`, and `host_tree` had no
    # method for a value whose type forbids host contents, so the generic struct walk tried the
    # constructor that does not exist and every model with a dropout layer failed to export. Found by
    # porting a real model, which is the only way it could have been found.
    @experiment struct ExportDropout
        width::GraphConst{Int} = 5
    end
    function ReactantNitro.build_model(e::ExportDropout, rng)
        m = Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dropout(0.5f0), Lux.Dense(e.width => 3))
        return (m, Lux.setup(rng, m)...)
    end
    function ReactantNitro.forward(::ExportDropout, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportDropout, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportDropout) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportDropout) = [ExportSpec("logits")]

    # The rename-and-echo case, which is what `from` exists for. `forward` calls its leaf `g_pred`, the
    # wire contract calls that tensor `state_out`, and the postprocess needs the `mask` input echoed back
    # because a serve-time postprocess receives the program's OUTPUTS and never its inputs. Before `from`,
    # the only expressible port was to make `forward` return an aliased leaf plus the echo, which
    # recompiles the gradient program to serve a serving detail.
    @experiment struct ExportRenamed
        width::GraphConst{Int} = 5
    end
    ReactantNitro.build_model(::ExportRenamed, rng) = (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportRenamed, model, ps, st; x, mask)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; g_pred = y .* mask, energy = sum(abs2, y; dims = 1)), st2
    end
    ReactantNitro.loss(::ExportRenamed, o; y) = sum(abs2, o.g_pred)
    ReactantNitro.export_inputs(::ExportRenamed) =
        [ExportSpec("x", XV, [4, 1]), ExportSpec("mask", XV, [1, 1])]
    ReactantNitro.export_outputs(::ExportRenamed) = [
        ExportSpec("state_out"; from = :g_pred),
        ExportSpec("mask"; from = :mask),
    ]

    # A model whose output is batch-FIRST, which is the disagreement the batch-last verification
    # exists to catch.
    @experiment struct ExportBatchFirst
        width::GraphConst{Int} = 5
    end
    ReactantNitro.build_model(::ExportBatchFirst, rng) = (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportBatchFirst, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = permutedims(y)), st2
    end
    ReactantNitro.loss(::ExportBatchFirst, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportBatchFirst) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportBatchFirst) = [ExportSpec("logits")]

    # No `export_preprocess`: `forward` declares the wire tensor directly.
    @experiment struct ExportDirect
        width::GraphConst{Int} = 5
    end
    ReactantNitro.build_model(::ExportDirect, rng) = (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportDirect, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportDirect, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportDirect) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportDirect) = [ExportSpec("logits")]

    # `export_outputs` naming a leaf `forward` does not return.
    @experiment struct ExportBadName
        width::GraphConst{Int} = 5
    end
    ReactantNitro.build_model(::ExportBadName, rng) = (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportBadName, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportBadName, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportBadName) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportBadName) = [ExportSpec("probabilities")]

    # Client outputs with no postprocess to produce them: legal to write, rejected at serve.
    @experiment struct ExportUnpaired
        width::GraphConst{Int} = 5
    end
    ReactantNitro.build_model(::ExportUnpaired, rng) = (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportUnpaired, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportUnpaired, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportUnpaired) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportUnpaired) = [ExportSpec("logits")]
    ReactantNitro.export_client_outputs(::ExportUnpaired) =
        [ExportSpec("prob", XV, [3, 1]; batch_axis = 2)]

    # The same unpaired error, reached from the INPUT side.
    @experiment struct ExportUnpairedIn
        width::GraphConst{Int} = 5
    end
    ReactantNitro.build_model(::ExportUnpairedIn, rng) = (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportUnpairedIn, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportUnpairedIn, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportUnpairedIn) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportUnpairedIn) = [ExportSpec("logits")]
    ReactantNitro.export_client_inputs(::ExportUnpairedIn) =
        [ExportSpec("x", XV, [4, 1]; axis_letters = ['f'])]

    # A postprocess and the client outputs it produces, which is the shape a real model ships in.
    @experiment struct ExportPaired
        num_classes::GraphConst{Int} = 3
    end
    ReactantNitro.build_model(::ExportPaired, rng) = (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportPaired, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportPaired, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportPaired) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportPaired) = [ExportSpec("logits")]
    ReactantNitro.export_postprocess(::ExportPaired) = "register_model(basename(@__DIR__))\n"
    ReactantNitro.export_client_inputs(::ExportPaired) =
        [ExportSpec("x", XV, [4, 1]; axis_letters = ['f'])]
    ReactantNitro.export_client_outputs(e::ExportPaired) = [
        ExportSpec("prob", XV, [e.num_classes, 1]; batch_axis = 2, axis_letters = ['c']),
        ExportSpec("logits", XV, [e.num_classes, 1]; batch_axis = 2),
    ]

    # ── the export-view fixtures ─────────────────────────────────────────────────────────

    # `Device` fields only the LOSS reads, which is the shape the export view exists for. They are traced
    # inputs to a training step, and before the export view froze them they were lifted into the
    # exported module as arguments no bundle could name and no client could pass.
    @experiment struct ExportDeviceFields
        width::GraphConst{Int} = 5
        class_weights::Device{Vector{Float32}} = Float32[1.0, 2.0, 3.0]
        soft_targets::Device{Matrix{Float32}} = zeros(Float32, 3, 3)
        run_note::String = "unmarked, so Host"
    end
    ReactantNitro.build_model(::ExportDeviceFields, rng) =
        (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportDeviceFields, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    # The loss reads both fields and `forward` reads neither, which is what makes them loss-only.
    ReactantNitro.loss(e::ExportDeviceFields, o; y) =
        sum(abs2, (o.logits .- e.soft_targets * y) .* e.class_weights)
    ReactantNitro.export_inputs(::ExportDeviceFields) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportDeviceFields) = [ExportSpec("logits")]

    # A `Device` field `forward` DOES read. Freezing has to keep the VALUE, not merely remove the
    # argument: this is the fixture that would catch a "drop the Device fields" implementation, which
    # is the design the export view explains it did not take.
    @experiment struct ExportDeviceRead
        scale::Device{Float32} = 3.0f0
    end
    ReactantNitro.build_model(::ExportDeviceRead, rng) =
        (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(e::ExportDeviceRead, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y .+ e.scale), st2
    end
    ReactantNitro.loss(::ExportDeviceRead, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportDeviceRead) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportDeviceRead) = [ExportSpec("logits")]

    # THE UNFROZEN-EXPORT DEFECT, reproduced on purpose. Overriding `export_view` back to
    # `compile_view` is exactly what export did before the export view, so this fixture is
    # unservable-bundle-shaped and the residency check is what has to refuse it.
    @experiment struct ExportUnfrozen
        class_weights::Device{Vector{Float32}} = Float32[1.0, 2.0, 3.0]
    end
    ReactantNitro.build_model(::ExportUnfrozen, rng) =
        (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportUnfrozen, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportUnfrozen, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportUnfrozen) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportUnfrozen) = [ExportSpec("logits")]
    ReactantNitro.export_view(e::ExportUnfrozen) = compile_view(e)

    # ── the provenance fixtures: the four provenance layers ─────────────────────────────

    # A model that implements the model half. `checkpoint === nothing` is a DISCRIMINATOR here and not
    # just an absent value, which is the case a hand-merged dictionary got wrong: the fields that
    # describe what the weights were trained against have to read as unverifiable when there is no
    # checkpoint behind them.
    @experiment struct ExportProv
        variant::GraphConst{Symbol} = :variant_a
    end
    ReactantNitro.build_model(::ExportProv, rng) =
        (m = Lux.Dense(4 => 3); (m, Lux.setup(rng, m)...))
    function ReactantNitro.forward(::ExportProv, model, ps, st; x)
        y, st2 = Lux.apply(model, x, ps, st)
        return (; logits = y), st2
    end
    ReactantNitro.loss(::ExportProv, o; y) = sum(abs2, o.logits)
    ReactantNitro.export_inputs(::ExportProv) = [ExportSpec("x", XV, [4, 1])]
    ReactantNitro.export_outputs(::ExportProv) = [ExportSpec("logits")]
    function ReactantNitro.export_provenance_extra(e::ExportProv; checkpoint = nothing)
        prov = Dict{String, Any}(
            "model" => "ExportProv",
            "variant" => String(e.variant),
            "class_names" => ["a", "b", "c"],
            "trained_verified" => checkpoint !== nothing,
        )
        checkpoint === nothing || (prov["trained_from"] = String(checkpoint))
        return prov
    end

    # A backend that answers the site-provenance verb, standing in for a real one. The multi-line
    # value is
    # the whole point: it is the shape a working-tree patch has, and the shape a `name=value` list
    # cannot carry.
    struct SiteBundle <: ExportBackend end
    ReactantNitro.site_provenance(::SiteBundle, root) = Dict{String, Any}(
        "git_commit" => "abc123",
        "git_dirty" => true,
        "git_root_seen" => String(root),
        "git_diff" => "diff --git a/x b/x\n@@ -1 +1 @@\n-old, with a comma\n+new\n",
    )
    function ReactantNitro.write_export(
            ::SiteBundle, program, ps, st, example_inputs;
            dir, name, input_names, output_names, output_select,
            client_inputs, client_outputs, postprocess, batch_sizes, provenance
        )
        RECORDED[] = (; provenance, dir, name)
        return joinpath(String(dir), String(name))
    end

    mkexp(E = ExportMLP; kw...) =
        Nitro(E(); data = (;), checkpointer = nothing, run_dir = mktempdir(), kw...)

    # ── The tests ───────────────────────────────────────────────────────────────────────

    @testset "export" begin

        @testset "the happy path, end to end through a fake backend" begin
            RECORDED[] = nothing
            n = mkexp()
            out = export_model(n, FakeBundle(); dir = "someplace", name = "mlp_v1", batch_sizes = [2, 4])
            r = RECORDED[]

            @test out == joinpath("someplace", "mlp_v1")
            @test r.name == "mlp_v1"
            @test r.input_names == ["img"]
            @test r.output_names == ["logits"]
            @test r.batch_sizes == [2, 4]

            # The example array is built from the spec at the FIRST batch size, with the batch axis
            # substituted and everything else taken literally.
            @test length(r.example_inputs) == 1
            @test eltype(r.example_inputs[1]) === UInt8
            @test size(r.example_inputs[1]) == (4, 2)

            # `export_outputs` names a SUBSET. `energy` is training-only and must not ship.
            @test keys(r.raw) == (:logits, :energy)
            @test length(r.selected) == 1
            @test size(r.selected[1]) == (3, 2)
        end

        @testset "`ExportCompiling` is published around the backend call" begin
            # The one phase that fires per `export_model` call rather than per cache miss, which is
            # the reason test/lifecycle.jl's "every phase a monitor may wait on actually arrives"
            # asserts the other three compile leaves there and this one here.
            seen = Any[]
            n = mkexp()
            register_phase_monitor!(n, (p, s, e, i) -> push!(seen, p))
            export_model(n, FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = [2])

            @test any(p -> p isa ExportCompiling, seen)
            # Fired before the backend call and restored afterwards, BETWEEN the entry point's two
            # bookends: `with_repl` declares `Starting` on the way in (the stretch before a verb's
            # first compile is real work, and a supervisor is charged for it whether or not anyone
            # said so) and `Repl` on the way out. So this is the first phase the export itself
            # publishes, which is not the same as the first event.
            @test first(seen) isa Starting
            @test findfirst(p -> p isa ExportCompiling, seen) == 2
            @test last(seen) isa Repl
            # Restored, not left set: a finished handle must not read as still compiling, which is
            # the phase a monitor would hang on.
            @test !(phase(n) isa Compiling)
        end

        @testset "the boundary copy, which is the trap the framework absorbs" begin
            RECORDED[] = nothing
            export_model(mkexp(), FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = [2])
            r = RECORDED[]
            # `forward` returned a view; what ships is materialized. Without the copy the serialized
            # module carries the producer's shape and the bundle's declared shape is a lie nothing raises
            # about.
            @test r.raw.logits isa SubArray
            @test r.selected[1] isa Array
            @test r.selected[1] == r.raw.logits
        end

        @testset "the preprocess really is what feeds `forward`" begin
            RECORDED[] = nothing
            export_model(mkexp(), FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = [2])
            r = RECORDED[]
            # The wire array is all zeros, so `x` is all zeros and the logits are the network's response
            # to zeros. Recomputing that through the model directly is the independent check that the
            # adapter routed `img -> x` rather than handing `forward` the raw bytes, which would have
            # failed on the element type instead of quietly differing.
            want, _ = Lux.apply(export_chain(ExportMLP()), zeros(XV, 4, 2), r.ps, r.st)
            @test r.selected[1] ≈ want
        end

        @testset "provenance carries the recipe and not the repository" begin
            p = export_provenance(mkexp(; seed = 1234))
            @test p["framework"] == "ReactantNitro.jl"
            @test p["seed"] == 1234
            @test haskey(p, "reactantnitro_version")
            @test p["config"]["width"] == 5
            @test p["config"]["num_classes"] == 3
            # Repository state is site policy and the framework does not guess at it.
            for k in ("git_commit", "git_tree_sha1", "git_diff", "repo_remote")
                @test !haskey(p, k)
            end
            # No preset was claimed, so the key is absent rather than present and null: a backend showing
            # `preset = nothing` on every ordinary run is noise.
            @test !haskey(p, "preset")
        end

        @testset "the caller's provenance is merged on top" begin
            RECORDED[] = nothing
            export_model(
                mkexp(), FakeBundle(); dir = mktempdir(), name = "m",
                provenance = Dict("git_commit" => "abc123", "framework" => "overridden")
            )
            p = RECORDED[].provenance
            @test p["git_commit"] == "abc123"
            @test p["framework"] == "overridden"        # the caller wins, deliberately
            @test haskey(p, "config")                   # and the framework's own half survives
        end

        @testset "the batch-last verification the backend's derivation depends on" begin
            # This is the check that earns the batch-last verification its existence. A tracer
            # derives every tensor's batch axis as its last axis and never questions it; the same is true of `forward`'s
            # return; and the two agree only because this fires when they do not.
            @test_throws ErrorException export_model(
                mkexp(ExportBatchFirst), FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = [2]
            )
        end

        @testset "with no `export_preprocess`, the wire maps straight onto the batch" begin
            RECORDED[] = nothing
            export_model(
                mkexp(ExportDirect), FakeBundle(); dir = mktempdir(), name = "m",
                batch_sizes = [2]
            )
            @test size(RECORDED[].selected[1]) == (3, 2)
        end

        @testset "a name `forward` does not return is an error naming what it does" begin
            err = try
                export_model(mkexp(ExportBadName), FakeBundle(); dir = mktempdir(), name = "m")
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("probabilities", err.msg)
            @test occursin("logits", err.msg)         # it says what IS there, not only what is not
        end

        @testset "the postprocess pairing the server would otherwise catch at load" begin
            err = try
                export_model(mkexp(ExportUnpaired), FakeBundle(); dir = mktempdir(), name = "m")
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("export_postprocess", err.msg)
        end

        @testset "client INPUTS need a postprocess too, on the same rule" begin
            err = try
                export_model(mkexp(ExportUnpairedIn), FakeBundle(); dir = mktempdir(), name = "m")
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("export_client_inputs", err.msg)
        end

        @testset "a postprocess and its client outputs reach the backend together" begin
            RECORDED[] = nothing
            export_model(mkexp(ExportPaired), FakeBundle(); dir = mktempdir(), name = "m")
            r = RECORDED[]
            @test r.postprocess isa String && occursin("register_model", r.postprocess)
            @test length(r.client_outputs) == 2
            @test r.client_outputs[1].name == "prob"
            # The client side is where `ExportSpec`'s full expressiveness applies, because the framework
            # derives none of it.
            @test r.client_outputs[1].axis_letters == ['c']
            @test r.client_outputs[1].batch_axis == 2
            # The client INPUT spec exists for exactly one reason, which is the letters the
            # tracer cannot carry into a derived executable spec.
            @test length(r.client_inputs) == 1
            @test r.client_inputs[1].axis_letters == ['f']
        end

        @testset "what `ExportSpec` refuses" begin
            # A batch axis that is not an axis at all.
            @test_throws ErrorException ExportSpec("a", XV, [3, 4]; batch_axis = 7)
            # One letter per NON-batch axis, not per axis.
            @test_throws ErrorException ExportSpec("a", XV, [3, 4]; axis_letters = ['c', 'd'])
            @test ExportSpec("a", XV, [3, 4]; axis_letters = ['c']).axis_letters == ['c']
            # Duplicates would collide in the manifest's dims map.
            @test_throws ErrorException ExportSpec("a", XV, [3, 4, 5]; axis_letters = ['c', 'c'])
            # The default is batch-last, which is the framework's rule stated once more where an author
            # would otherwise have to remember it.
            @test ExportSpec("a", XV, [3, 4]).batch_axis == 2
            # Name-only, for an executable output whose shape is derived from the trace.
            s = ExportSpec("logits")
            @test s.dtype === nothing && isempty(s.shape) && s.batch_axis === nothing
        end

        @testset "the executable side declares nothing it cannot back up" begin
            @test_throws ErrorException check_export_derived_only(
                [ExportSpec("a", XV, [3, 4]; axis_letters = ['c'])], :export_inputs
            )
            @test_throws ErrorException check_export_derived_only(
                [ExportSpec("a", XV, [-1, 4])], :export_inputs
            )
            @test_throws ErrorException check_export_batch_last(
                [ExportSpec("a", XV, [3, 4]; batch_axis = 1)], :export_inputs
            )
            @test check_export_batch_last([ExportSpec("a", XV, [3, 4])], :export_inputs) === nothing
        end

        @testset "a backend with no method says which package to load" begin
            err = try
                export_model(mkexp(), ReactantServerBundle(); dir = mktempdir(), name = "m")
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("ReactantServerExport", err.msg)
        end

        @testset "a model with a `Dropout`: the RNG in `st` is REPLACED, not passed through" begin
            # This testset used to assert the opposite, and the reasoning it carried is the reason the
            # defect shipped: "eval mode makes leaving the RNG safe, because a testmode dropout never
            # draws from it". The first half is true, the second half is true, and the conclusion is
            # false, because the failure is not numerical. `st` is TRACED, and Reactant lifts a
            # device-resident seed into an MLIR argument whether or not the program reads it.
            RECORDED[] = nothing
            export_model(
                mkexp(ExportDropout), FakeBundle(); dir = mktempdir(), name = "m",
                batch_sizes = [2]
            )
            r = RECORDED[]
            @test size(r.selected[1]) == (3, 2)
            # Eval mode is still the framework's switch, and is still what makes the substitution safe.
            @test r.st.layer_2.training === Val(false)
            # The RNG is REPLACED and not removed: a testmode `Dropout` still dispatches on its type.
            @test r.st.layer_2.rng isa Random.AbstractRNG
            @test !(r.st.layer_2.rng isa ReactantRNG)
            # And the invariant that actually matters, which no count of weights can express.
            @test isempty(device_paths(r.st))
            @test isempty(device_paths(r.program))
        end

        @testset "loss-only `Device` fields never reach the traced program" begin
            # The unfrozen-export defect. `class_weights` and `soft_targets` are read by `loss` and
            # by nothing else, and they were arriving in the compiled module as two arguments the
            # bundle declared nowhere and the server could not supply.
            RECORDED[] = nothing
            export_model(
                mkexp(ExportDeviceFields), FakeBundle(); dir = mktempdir(), name = "m",
                batch_sizes = [2]
            )
            r = RECORDED[]
            @test size(r.selected[1]) == (3, 2)
            # Nothing anywhere in what the backend traces is device-resident. This is the whole
            # contract, and it is one assertion rather than an arithmetic identity over three counts.
            @test isempty(device_paths(r.program))
            @test isempty(device_paths(r.ps))
            @test isempty(device_paths(r.st))
            # The fields are still THERE and still hold their values; they are host values now.
            @test r.program.ev.class_weights == Float32[1.0, 2.0, 3.0]
            @test r.program.ev.class_weights isa Array{Float32, 1}
            @test r.program.ev.soft_targets isa Array{Float32, 2}
            # `Host` is still stripped, exactly as `compile_view` strips it: the export view adds a
            # rule, it does not replace the one the Host guard already had.
            @test r.program.ev.run_note isa StrippedHost
        end

        @testset "a `Device` field `forward` READS keeps its value" begin
            # Freezing rather than filtering, which is the half of the export view a "drop the loss
            # inputs"
            # implementation would get wrong: this field is a legitimate input to the exported graph
            # and has to bake with the value it held at export.
            RECORDED[] = nothing
            export_model(
                mkexp(ExportDeviceRead), FakeBundle(); dir = mktempdir(), name = "m",
                batch_sizes = [2]
            )
            r = RECORDED[]
            @test r.program.ev.scale == 3.0f0
            @test r.program.ev.scale isa Float32
            @test isempty(device_paths(r.program))
            # The value is really in the program and not merely on the struct: the wire input is
            # zeros, so `forward` returns the bias plus the frozen field, exactly. A "drop the
            # `Device` fields" implementation cannot produce this, and neither can a stale value.
            @test all(r.selected[1] .≈ (r.ps.bias .+ 3.0f0))
        end

        @testset "a device value reaching the trace is REFUSED, with its path" begin
            # `ExportUnfrozen` overrides `export_view` back to `compile_view`, which is what export did
            # before the export view. The bundle it would write is the unservable one, so the check
            # has to fire
            # before the trace rather than after the deploy.
            err = try
                export_model(
                    mkexp(ExportUnfrozen), FakeBundle(); dir = mktempdir(), name = "m",
                    batch_sizes = [2]
                )
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            # The path, which is the only useful thing to say, and the thing the serve-time error
            # cannot say at all.
            @test occursin("program.ev.class_weights", err.msg)
            @test occursin("still DEVICE-resident", err.msg)
        end

        @testset "`export_view` is `compile_view` plus a residency rule" begin
            e = ExportDeviceFields()
            nitro = mkexp(ExportDeviceFields)
            # Before setup nothing has been placed, so the two views agree on everything but identity.
            @test compile_view(e).run_note isa StrippedHost
            # After setup, `compile_view` leaves the `Device` fields device-resident: that is by
            # design, and
            # it is right for a training trace.
            ev_train = compile_view(experiment(nitro))
            @test !isempty(device_paths(ev_train))
            # `export_view` is the same view with that one rule reversed.
            ev_export = export_view(experiment(nitro))
            @test isempty(device_paths(ev_export))
            @test ev_export.run_note isa StrippedHost
            @test ev_export.width === 5                     # GraphConst still bakes, untouched
            @test ev_export.class_weights == Float32[1.0, 2.0, 3.0]
        end

        @testset "the model half arrives without a caller passing anything" begin
            RECORDED[] = nothing
            export_model(
                mkexp(ExportProv), FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = [2]
            )
            p = RECORDED[].provenance
            # The hook was merged, and nothing at the call site mentioned it. That is the whole point:
            # the argument used to be forgettable and omitting it produced a complete-looking bundle.
            @test p["model"] == "ExportProv"
            @test p["variant"] == "variant_a"
            @test p["class_names"] == ["a", "b", "c"]
            # Freshly initialized weights, so the hook's discriminator reads false rather than absent.
            @test p["trained_verified"] === false
            @test !haskey(p, "trained_from")
            # The framework's own layer is still underneath it.
            @test p["framework"] == "ReactantNitro.jl"
            @test haskey(p, "config")
            # No checkpoint behind this handle, so the framework omits the key rather than emptying it.
            @test !haskey(p, "checkpoint")
        end

        @testset "an experiment with no hook gets the default, which changes nothing" begin
            RECORDED[] = nothing
            export_model(
                mkexp(ExportMLP), FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = [2]
            )
            p = RECORDED[].provenance
            @test p["framework"] == "ReactantNitro.jl"
            @test !haskey(p, "model")
        end

        @testset "`provenance_root` carries what a `name=value` list cannot" begin
            RECORDED[] = nothing
            export_model(
                mkexp(ExportProv), SiteBundle(); dir = mktempdir(), name = "m",
                batch_sizes = [2], provenance_root = "/some/repo"
            )
            p = RECORDED[].provenance
            @test p["git_commit"] == "abc123"
            @test p["git_dirty"] === true
            @test p["git_root_seen"] == "/some/repo"
            # A multi-line patch containing commas, reaching the writer as itself. This is the value
            # whose shape is the reason the root exists.
            @test occursin("\n", p["git_diff"])
            @test occursin("old, with a comma", p["git_diff"])
            # All four layers coexist.
            @test p["framework"] == "ReactantNitro.jl"
            @test p["model"] == "ExportProv"
        end

        @testset "omitting the root omits repository state, and says so by omission" begin
            RECORDED[] = nothing
            export_model(
                mkexp(ExportProv), SiteBundle(); dir = mktempdir(), name = "m", batch_sizes = [2]
            )
            p = RECORDED[].provenance
            # The failure this surface is designed against: the export SUCCEEDS and the bundle is
            # untraceable. Asserted so that "looks complete" is a documented state rather than a
            # surprise, and so a future default cannot flip silently.
            @test !haskey(p, "git_commit")
            @test !haskey(p, "git_diff")
            @test p["framework"] == "ReactantNitro.jl"
        end

        @testset "the precedence: explicit beats model beats site beats framework" begin
            RECORDED[] = nothing
            export_model(
                mkexp(ExportProv), SiteBundle(); dir = mktempdir(), name = "m",
                batch_sizes = [2], provenance_root = "/some/repo",
                provenance = Dict("model" => "overridden", "git_commit" => "forced", :seed => 999)
            )
            p = RECORDED[].provenance
            # Explicit wins over the model hook and over the site collector.
            @test p["model"] == "overridden"
            @test p["git_commit"] == "forced"
            # And over the framework, including a key given as a Symbol: every layer is keyed by
            # string before merging, or an override would land beside the value instead of on it.
            @test p["seed"] == 999
        end

        @testset "a backend with no `site_provenance` refuses rather than stamping nothing" begin
            err = try
                export_model(
                    mkexp(ExportProv), FakeBundle(); dir = mktempdir(), name = "m",
                    batch_sizes = [2], provenance_root = "/some/repo"
                )
                nothing
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("site_provenance", err.msg)
            @test occursin("provenance_root", err.msg)
        end

        @testset "the framework stamps the checkpoint the handle restored from" begin
            # The path is not re-asserted by the caller anywhere below: it went to the CONSTRUCTOR,
            # and the handle carried it to both the framework's stamp and the model's hook.
            dir = mktempdir()
            # No `name`: setup binds the experiment's `checkpoint_filename` to the
            # checkpointer, so the handle's own checkpointer is the one that can write.
            writer = Nitro(
                ExportProv(); data = (;), run_dir = dir,
                checkpointer = TopKCheckpointer(; k = 1, metric = :val_loss, mode = :min, dir)
            )
            save_checkpoint!(
                writer.checkpointer, 3, (; val_loss = 0.25f0), ReactantNitro.snapshot(writer)
            )
            # The checkpoint layer's own manifest is a `.jld2` in the same directory, so it has to
            # be excluded or
            # the "one checkpoint" assertion counts two files and the path picked is whichever
            # `readdir` returned first.
            files = filter(readdir(dir; join = true)) do f
                endswith(f, ".jld2") && basename(f) != "manifest.jld2"
            end
            @test length(files) == 1
            n2 = Nitro(
                ExportProv(); data = (;), run_dir = mktempdir(),
                checkpointer = TopKCheckpointer(; dir), checkpoint = first(files)
            )
            # The handle retains the RESOLVED source, which is what both consumers below read.
            @test n2.checkpoint_source == first(files)
            RECORDED[] = nothing
            export_model(n2, FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = [2])
            p = RECORDED[].provenance
            @test p["checkpoint"] == first(files)
            # And the hook saw it, which is what removes the "pass the same path twice" failure mode.
            @test p["trained_verified"] === true
            @test p["trained_from"] == first(files)
        end

        @testset "`host_rngs` walks structs, which is how the seed hid" begin
            # A `NamedTuple`-only walker reports this tree as having no RNG in it at all, because the
            # RNG is a struct field two levels down. That is precisely why the seed went unnoticed
            # through a manifest check, a provenance check, a registration and a deploy.
            tree = (; layer_1 = (; w = zeros(Float32, 2, 2)), layer_2 = (; rng = Random.Xoshiro(7), training = Val(false)))
            out = host_rngs(tree)
            @test out.layer_2.rng isa Random.AbstractRNG
            # Identity preservation: a subtree with no RNG comes back the same object.
            @test out.layer_1 === tree.layer_1
        end

        @testset "`from` renames a leaf and echoes a wire input" begin
            RECORDED[] = nothing
            export_model(
                mkexp(ExportRenamed), FakeBundle(); dir = mktempdir(), name = "m",
                batch_sizes = [2]
            )
            r = RECORDED[]
            # The BUNDLE names, not the leaf names: this is the whole point.
            @test r.output_names == ["state_out", "mask"]
            @test length(r.selected) == 2
            @test size(r.selected[1]) == (3, 2)
            # The echo is the WIRE value, the tensor the client sent, and it really is a program output.
            @test r.selected[2] == r.example_inputs[2]
            # `energy` is still not shipped: naming a subset keeps working alongside `from`.
            @test keys(r.raw) == (:g_pred, :energy, :mask)
        end

        @testset "an echo must say so, rather than being inferred from a name" begin
            # `ExportSpec("mask")` with no `from`, where `mask` is also a wire input, is refused: turning a
            # client's own tensor into a program output on a name collision would be a guess.
            specs = [ExportSpec("mask")]
            @test_throws ErrorException ReactantNitro.check_export_sources(
                specs, ReactantNitro.spec_sources(specs), (:x, :mask)
            )
            ok = [ExportSpec("mask"; from = :mask)]
            @test ReactantNitro.check_export_sources(
                ok, ReactantNitro.spec_sources(ok), (:x, :mask)
            ) == (:mask,)
        end

        @testset "`from` is an output-side concept and is refused elsewhere" begin
            @test_throws ErrorException ReactantNitro.check_export_inputs(
                [ExportSpec("x", XV, [4, 1]; from = :y)]
            )
            @test_throws ErrorException ReactantNitro.check_client_specs(
                [ExportSpec("p", XV, [3, 1]; from = :y)], :export_client_outputs, "src"
            )
        end

        @testset "batch_sizes must name at least one size" begin
            @test_throws ErrorException export_model(
                mkexp(), FakeBundle(); dir = mktempdir(), name = "m", batch_sizes = Int[]
            )
        end
    end

end
