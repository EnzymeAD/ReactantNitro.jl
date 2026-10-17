# ReactantNitroReactantServerExportExt.jl
#
# The one shipped export backend, on a weak dependency. A translation and nothing more:
# ReactantServerExport is the bundle format's authority, and this file turns `ExportSpec` into
# `IOSpec`, calls `export_bundle`, and writes the `model.jl` the writer deliberately does not.

module ReactantNitroReactantServerExportExt

import ReactantNitro
import ReactantServerExport

using ReactantNitro: ExportSpec, ReactantServerBundle
using ReactantServerExport: IOSpec, export_bundle, collect_provenance

"""
    ReactantNitro.site_provenance(::ReactantServerBundle, root) -> Dict{String,Any}

Repository state at `root` for this backend, from `ReactantServerExport.collect_provenance`: the
commit, branch, dirty flag, tree hash, timestamp, remote, and `git_diff`, a full unified patch the
writer saves as `working_tree.patch`.
"""
ReactantNitro.site_provenance(::ReactantServerBundle, root) =
    collect_provenance(String(root))

# `ExportSpec` is 1-based Julia axes; `IOSpec` is 0-based network axes.
function _iospec(s::ExportSpec)
    return IOSpec(
        s.name, s.dtype, s.shape;
        batch_axis = s.batch_axis === nothing ? nothing : s.batch_axis - 1,
        letters = s.axis_letters,
    )
end

"""
    ReactantNitro.write_export(::ReactantServerBundle, program, ps, st, example_inputs; ...)

Write a ReactantServer StableHLO bundle: `manifest.yaml`, `weights.safetensors`, one
`model[.b{N}].mlir` per batch size, and the `model.jl` postprocess when the experiment ships one.
The bundle directory is `joinpath(dir, name)`, since the format requires the basename to equal the
name. No batch axis is passed for the executable tensors: `export_bundle` takes the last axis,
which `export_model` has already checked against the traced output. Client specs are passed when
the experiment declares them; otherwise the server uses the executable specs.
"""
function ReactantNitro.write_export(
        ::ReactantServerBundle, program, ps, st, example_inputs;
        dir::AbstractString, name::AbstractString,
        input_names, output_names, output_select,
        client_inputs, client_outputs, postprocess, batch_sizes, provenance
    )
    bundle_dir = joinpath(String(dir), String(name))
    _specs(v) = v === nothing ? nothing : IOSpec[_iospec(s) for s in v]

    written = export_bundle(
        :lux, program, ps, st, example_inputs;
        dir = bundle_dir, name = String(name),
        input_names = input_names,
        output_names = output_names,
        output_select = output_select,
        client_inputs = _specs(client_inputs),
        client_outputs = _specs(client_outputs),
        batch_sizes = batch_sizes,
        provenance = provenance,
    )

    # The writer makes shipping `model.jl` the caller's job.
    postprocess === nothing || write(joinpath(written, "model.jl"), postprocess)

    return written
end

end # module ReactantNitroReactantServerExportExt
