# The TFSavedModel backend, end to end: export, load in TensorFlow, run every signature, compare with
# the eager program. A separate project because it installs TensorFlow; CI runs it in its own job.
#
#     julia --project=test/savedmodel -e 'using Pkg; Pkg.instantiate(); include("test/savedmodel/runtests.jl")'

ENV["CUDA_VISIBLE_DEVICES"] = ""

using Test
using ReactantNitro
using Lux, Random
using PythonCall
using Reactant: Reactant
import JSON

const EXT = Base.get_extension(ReactantNitro, :ReactantNitroPythonCallExt)
const tf = pyimport("tensorflow")
const np = pyimport("numpy")

@experiment struct TFConv
    classes::GraphConst{Int} = 3
end
function ReactantNitro.build_model(e::TFConv, rng)
    m = Lux.Chain(
        Lux.Conv((3, 3), 1 => 4, relu), Lux.BatchNorm(4), Lux.MaxPool((2, 2)),
        Lux.FlattenLayer(), Lux.Dense(36 => e.classes),
    )
    return (m, Lux.setup(rng, m)...)
end
function ReactantNitro.forward(::TFConv, model, ps, st; x)
    y, st2 = Lux.apply(model, x, ps, st)
    return (; logits = y, prob = Lux.softmax(y)), st2
end
ReactantNitro.loss(::TFConv, o; y) = sum(abs2, o.logits)
ReactantNitro.export_inputs(::TFConv) = [ExportSpec("img", UInt8, [8, 8, 1, 1])]
ReactantNitro.export_outputs(::TFConv) = [ExportSpec("logits"), ExportSpec("prob")]
ReactantNitro.export_preprocess(::TFConv, img) = (; x = to_f32(img))
to_f32(x::AbstractArray) = Float32.(x) ./ 255.0f0
to_f32(x::Reactant.TracedRArray) = Reactant.Ops.convert(Reactant.TracedRArray{Float32, ndims(x)}, x) ./ 255.0f0

# Two wire inputs, a renamed output and an echo.
@experiment struct TFTwoIn
    width::GraphConst{Int} = 5
end
ReactantNitro.build_model(e::TFTwoIn, rng) = (m = Lux.Dense(4 => e.width); (m, Lux.setup(rng, m)...))
function ReactantNitro.forward(::TFTwoIn, model, ps, st; x, mask)
    y, st2 = Lux.apply(model, x, ps, st)
    return (; g_pred = y .* mask), st2
end
ReactantNitro.loss(::TFTwoIn, o; y) = sum(abs2, o.g_pred)
ReactantNitro.export_inputs(::TFTwoIn) = [ExportSpec("x", Float32, [4, 1]), ExportSpec("mask", Float32, [1, 1])]
ReactantNitro.export_outputs(::TFTwoIn) = [ExportSpec("state_out"; from = :g_pred), ExportSpec("mask"; from = :mask)]

@experiment struct TFPost
    width::GraphConst{Int} = 2
end
ReactantNitro.build_model(e::TFPost, rng) = (m = Lux.Dense(4 => e.width); (m, Lux.setup(rng, m)...))
function ReactantNitro.forward(::TFPost, model, ps, st; x)
    y, st2 = Lux.apply(model, x, ps, st)
    return (; logits = y), st2
end
ReactantNitro.loss(::TFPost, o; y) = sum(abs2, o.logits)
ReactantNitro.export_inputs(::TFPost) = [ExportSpec("x", Float32, [4, 1])]
ReactantNitro.export_outputs(::TFPost) = [ExportSpec("logits")]
ReactantNitro.export_postprocess(::TFPost) = "postprocess(outputs) = outputs"

mknitro(E) = Nitro(E(); data = (;), checkpointer = nothing, run_dir = mktempdir())

host(n) = (ReactantNitro.host_tree(n.ps), Lux.testmode(ReactantNitro.host_tree(n.st)))

# Julia batch-last to TF batch-first and back.
to_tf(x) = np.asarray(permutedims(x, ndims(x):-1:1))
from_tf(t) = (a = pyconvert(Array, t.numpy()); permutedims(a, ndims(a):-1:1))

const CPU = TFSavedModel(; platform = "CPU")

@testset "TFSavedModel" begin
    @testset "a conv classifier: signatures, layout, numerics, provenance" begin
        n = mknitro(TFConv)
        dir = mktempdir()
        path = export_model(
            n, CPU; dir, name = "conv", batch_sizes = [1, 3],
            provenance_root = pkgdir(ReactantNitro),
        )
        @test path == joinpath(dir, "conv")
        @test isfile(joinpath(path, "saved_model.pb"))
        @test isdir(joinpath(path, "variables"))

        prov = JSON.parsefile(joinpath(path, "assets.extra", "provenance.json"))
        @test prov["converter"] == "ReactantNitro.jl TFSavedModel"
        @test prov["platform"] == "CPU"
        @test prov["batch_sizes"] == [1, 3]
        @test haskey(prov, "git_commit")
        @test haskey(prov, "config")
        @test !haskey(prov, "git_diff")
        @test get(prov, "git_dirty", false) == isfile(joinpath(path, "assets.extra", "working_tree.patch"))

        m = tf.saved_model.load(path)
        sigs = m.signatures
        @test Set(pyconvert(Vector{String}, pylist(sigs.keys()))) == Set(["serving_default", "serving_b1", "serving_b3"])

        f3 = sigs["serving_b3"]
        spec = f3.structured_input_signature[1]["img"]
        @test pyconvert(Vector{Int}, spec.shape.as_list()) == [3, 1, 8, 8]
        @test pyconvert(String, spec.dtype.name) == "uint8"
        @test Set(pyconvert(Vector{String}, pylist(f3.structured_outputs.keys()))) == Set(["logits", "prob"])

        ps, st = host(n)
        for nb in (1, 3)
            img = rand(UInt8, 8, 8, 1, nb)
            out = sigs["serving_b$nb"](; img = to_tf(img))
            y, _ = Lux.apply(n.model, to_f32(img), ps, st)
            @test size(from_tf(out["logits"])) == (3, nb)
            @test from_tf(out["logits"]) ≈ y atol = 1.0f-5
            @test from_tf(out["prob"]) ≈ Lux.softmax(y) atol = 1.0f-5
        end
        img = rand(UInt8, 8, 8, 1, 1)
        @test from_tf(sigs["serving_default"](; img = to_tf(img))["logits"]) ≈
            first(Lux.apply(n.model, to_f32(img), ps, st)) atol = 1.0f-5

        @test_throws "not empty" export_model(n, CPU; dir, name = "conv")
    end

    @testset "two inputs, a rename and an echo" begin
        n = mknitro(TFTwoIn)
        path = export_model(n, CPU; dir = mktempdir(), name = "twoin", batch_sizes = [2])
        f = tf.saved_model.load(path).signatures["serving_default"]
        x, mask = rand(Float32, 4, 2), Float32[1 0]
        out = f(; x = to_tf(x), mask = to_tf(mask))
        ps, st = host(n)
        @test from_tf(out["state_out"]) ≈ first(Lux.apply(n.model, x, ps, st)) .* mask atol = 1.0f-5
        @test from_tf(out["mask"]) == mask
    end

    @testset "the default CUDA platform writes on a CPU host" begin
        path = export_model(mknitro(TFTwoIn), TFSavedModel(); dir = mktempdir(), name = "cuda")
        @test isfile(joinpath(path, "saved_model.pb"))
    end

    @testset "TF 2.15 settings" begin
        b = TFSavedModel(; platform = "CPU", stablehlo_version = v"0.14.0", call_module_version = 8)
        path = export_model(mknitro(TFConv), b; dir = mktempdir(), name = "old")
        img = rand(UInt8, 8, 8, 1, 1)
        @test size(from_tf(tf.saved_model.load(path).signatures["serving_default"](; img = to_tf(img))["prob"])) == (3, 1)
    end

    @testset "a postprocess is dropped with a warning" begin
        @test_logs (:warn, r"postprocess is dropped") match_mode = :any export_model(
            mknitro(TFPost), CPU; dir = mktempdir(), name = "post"
        )
    end

    @testset "refusals" begin
        @test_throws "cannot ship `model.jl`" ReactantNitro.write_export(
            CPU, nothing, (;), (;), (zeros(Float32, 1),);
            dir = mktempdir(), name = "x", input_names = ["x"], output_names = ["y"],
            output_select = identity, client_inputs = nothing,
            client_outputs = [ExportSpec("y", Float32, [1])], postprocess = "",
            batch_sizes = [1], provenance = Dict(),
        )
        @test EXT._custom_calls("%0 = stablehlo.custom_call @lapack_sgetrf(%a) : x\nfoo") == ["lapack_sgetrf"]
        @test EXT._custom_calls("\"stablehlo.custom_call\"(%a) {call_target_name = \"cu_threefry2x32\"}") == ["cu_threefry2x32"]
        @test isempty(EXT._custom_calls("stablehlo.add %a, %b"))
        @test_throws "takes 4 arguments" EXT._check_entry_arity((; in_tys = 1:4), 1, 2, 8)
        @test EXT._check_entry_arity((; in_tys = 1:3), 1, 2, 8) === nothing
    end
end
