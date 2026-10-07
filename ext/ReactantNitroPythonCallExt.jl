# ReactantNitroPythonCallExt.jl
#
# The TensorFlow SavedModel backend. Reactant lowers each batch size to StableHLO; TensorFlow, driven
# through PythonCall, wraps each module in one `XlaCallModule` op and writes the SavedModel.

module ReactantNitroPythonCallExt

import ReactantNitro
import Reactant
import JSON

using PythonCall: Py, pyimport, pylist, pydict, pyfunc, pybytes, pyconvert
using ReactantNitro: TFSavedModel

const MLIR = Reactant.MLIR
const Compiler = Reactant.Compiler

const TF_DTYPES = Dict{DataType, String}(
    Float16 => "float16", Float32 => "float32", Float64 => "float64",
    Int8 => "int8", Int16 => "int16", Int32 => "int32", Int64 => "int64",
    UInt8 => "uint8", UInt16 => "uint16", UInt32 => "uint32", UInt64 => "uint64",
    Bool => "bool",
)

function _tf_dtype(tf, T)
    haskey(TF_DTYPES, T) || error("ReactantNitro: `TFSavedModel` has no TensorFlow dtype for `$T`.")
    return getproperty(tf, Symbol(TF_DTYPES[T]))
end

function _tensorflow()
    tf = try
        pyimport("tensorflow")
    catch err
        error(
            """
            ReactantNitro: `TFSavedModel` needs the Python package `tensorflow` in PythonCall's
            environment, and importing it failed:
            $(first(split(sprint(showerror, err), '\n')))"""
        )
    end
    # Export is a CPU trace; TF must not claim the cards. Already set if TF was imported earlier.
    try
        tf.config.set_visible_devices(pylist(), "GPU")
    catch
    end
    return tf, pyimport("tensorflow.compiler.tf2xla.python.xla")
end

# ── Lowering, after ReactantServerExport's `export_bundle(:lux, ...)` ───────────────

function _named_leaves(x, prefix = "", out = Tuple{String, Any}[])
    if x isa AbstractArray{<:Number}
        push!(out, (isempty(prefix) ? "param" : prefix, x))
    elseif x isa NamedTuple
        for k in keys(x)
            _named_leaves(getfield(x, k), isempty(prefix) ? String(k) : "$prefix.$k", out)
        end
    elseif x isa Tuple || x isa AbstractVector
        for (i, v) in enumerate(x)
            _named_leaves(v, isempty(prefix) ? string(i) : "$prefix.$i", out)
        end
    end
    return out
end

function _rebuild(template, ws, idx = Ref(0))
    if template isa AbstractArray{<:Number}
        return ws[idx[] += 1]
    elseif template isa NamedTuple
        return NamedTuple{keys(template)}(map(k -> _rebuild(getfield(template, k), ws, idx), keys(template)))
    elseif template isa Tuple
        return map(v -> _rebuild(v, ws, idx), template)
    elseif template isa AbstractVector
        return [_rebuild(v, ws, idx) for v in template]
    end
    return template
end

function _with_batch(x::AbstractArray, s::Integer)
    sz = collect(size(x))
    sz[end] = Int(s)
    return zeros(eltype(x), sz...)
end

# StableHLO shapes are the Julia shape reversed, so batch-last becomes batch-first.
_tf_shape(sz) = pylist(reverse(collect(Int, sz)))
_np(np, x::AbstractArray) = np.asarray(ndims(x) <= 1 ? x : permutedims(x, ndims(x):-1:1))

function _check_entry_arity(fn_res, n_inputs, n_weights, s)
    got = nothing
    for f in (:in_tys, :linear_args)
        hasproperty(fn_res, f) && (got = length(getproperty(fn_res, f)); break)
    end
    (got === nothing || got == n_inputs + n_weights) && return nothing
    return error(
        """
        ReactantNitro: the program traced at batch size $s takes $got arguments, and $n_inputs inputs
        plus $n_weights weights were declared. A surplus is a device-resident value Reactant lifted
        into an argument, such as an RNG in layer state; the SavedModel would fail every request."""
    )
end

function _custom_calls(mod)
    txt = string(mod)
    targets = String[m.captures[1] for m in eachmatch(r"stablehlo\.custom_call\s+@([\w.$-]+)", txt)]
    append!(targets, String[m.captures[1] for m in eachmatch(r"call_target_name\s*=\s*\"([^\"]+)\"", txt)])
    return unique(targets)
end

function _serialize(mod, version::VersionNumber)
    cb = @cfunction(MLIR.IR.print_callback, Cvoid, (MLIR.API.MlirStringRef, Any))
    ref = Ref(IOBuffer())
    res = MLIR.API.stablehloSerializePortableArtifactFromModule(mod, string(version), cb, ref, true)
    MLIR.IR.isfailure(MLIR.IR.LogicalResult(res)) && error(
        """
        ReactantNitro: the module could not be serialized at StableHLO $version, usually because it
        uses an op newer than that version. Raise `stablehlo_version` on `TFSavedModel`, which
        narrows the TensorFlow releases that can load the result."""
    )
    return take!(ref[])
end

# ── The backend ─────────────────────────────────────────────────────────────────────

"""
    ReactantNitro.write_export(::TFSavedModel, program, ps, st, example_inputs; ...)

Write a TensorFlow SavedModel at `joinpath(dir, name)`: `saved_model.pb`, the weights under
`variables/`, and `assets.extra/provenance.json`, plus `assets.extra/working_tree.patch` when the
provenance carries a `git_diff`. Refuses a non-empty target directory.
"""
function ReactantNitro.write_export(
        b::TFSavedModel, program, ps, st, example_inputs;
        dir::AbstractString, name::AbstractString,
        input_names, output_names, output_select,
        client_inputs, client_outputs, postprocess, batch_sizes, provenance
    )
    client_outputs === nothing || error(
        """
        ReactantNitro: `export_client_outputs` declares a client contract produced by
        `export_postprocess`, and a SavedModel cannot ship `model.jl`. Export to
        `ReactantServerBundle`, or move the postprocess into `forward`."""
    )
    postprocess === nothing ||
        @warn "ReactantNitro: `TFSavedModel` cannot ship `model.jl`; the postprocess is dropped and clients receive the raw program outputs."
    allunique(batch_sizes) || error("ReactantNitro: `batch_sizes` has duplicates: $(batch_sizes).")

    path = joinpath(String(dir), String(name))
    (isdir(path) && !isempty(readdir(path))) &&
        error("ReactantNitro: `$path` already exists and is not empty; remove it or choose another `name`.")

    tf, xla = _tensorflow()
    np = pyimport("numpy")

    leaves = _named_leaves(ps)
    warrays = Any[w for (_, w) in leaves]
    nin = length(example_inputs)
    modelarg(t) = nin == 1 ? t[1] : t
    g = (a, ws...) -> output_select(first(program(a, _rebuild(ps, collect(ws)), st)))
    y0 = output_select(first(program(modelarg(example_inputs), ps, st)))

    vars = Py[tf.Variable(_np(np, w); trainable = false, name = n) for (n, w) in leaves]
    root = tf.Module()
    root.nitro_weights = pylist(vars)
    signatures = Dict{String, Py}()

    for s in batch_sizes
        xs = map(x -> _with_batch(x, s), example_inputs)
        ctx = Reactant.ReactantContext()
        args = (modelarg(map(Reactant.to_rarray, xs)), map(Reactant.to_rarray, warrays)...)
        mod, fn_res = Compiler.compile_mlir(ctx, g, args; drop_unsupported_attributes = true)
        _check_entry_arity(fn_res, nin, length(warrays), s)
        calls = _custom_calls(mod)
        isempty(calls) || error(
            """
            ReactantNitro: the program at batch size $s contains `custom_call` targets
            $(join(calls, ", ")), which `XlaCallModule` cannot run portably. Rewrite the op that
            lowers to it, or export to `ReactantServerBundle`."""
        )
        bytes = GC.@preserve ctx _serialize(mod, b.stablehlo_version)

        specs = pylist([tf.TensorSpec(_tf_shape(size(x)), _tf_dtype(tf, eltype(x)); name = input_names[i]) for (i, x) in enumerate(xs)])
        tout = pylist([_tf_dtype(tf, eltype(y)) for y in y0])
        sout = pylist([_tf_shape((size(y)[1:(end - 1)]..., s)) for y in y0])
        call = function (wire...)
            outs = xla.call_module(
                pylist([wire..., vars...]);
                version = b.call_module_version, var"module" = pybytes(bytes),
                Tout = tout, Sout = sout, platforms = pylist([b.platform]),
            )
            return pydict(Dict(output_names[i] => outs[i - 1] for i in eachindex(output_names)))
        end
        f = tf.function(pyfunc(call); input_signature = specs, autograph = false)
        key = "serving_b$s"
        setproperty!(root, Symbol(key), f)
        signatures[key] = f.get_concrete_function()
    end
    signatures["serving_default"] = signatures["serving_b$(first(batch_sizes))"]

    mkpath(path)
    tf.saved_model.save(root, path; signatures = pydict(signatures))

    prov = merge(
        Dict{String, Any}(
            "source_framework" => "reactant",
            "converter" => "ReactantNitro.jl TFSavedModel",
            "reactant_version" => string(pkgversion(Reactant)),
            "tensorflow_version" => pyconvert(String, tf.__version__),
            "stablehlo_version" => string(b.stablehlo_version),
            "xla_call_module_version" => b.call_module_version,
            "platform" => b.platform,
            "batch_sizes" => collect(Int, batch_sizes),
        ),
        Dict{String, Any}(provenance),
    )
    extra = mkpath(joinpath(path, "assets.extra"))
    diff = pop!(prov, "git_diff", nothing)
    diff === nothing || write(joinpath(extra, "working_tree.patch"), diff)
    open(io -> JSON.json(io, prov; pretty = true), joinpath(extra, "provenance.json"), "w")
    return path
end

end # module ReactantNitroPythonCallExt
