# ReactantNitroReactantServerExportExt.jl
#
# The one shipped export backend, on a weak dependency.
#
# This file is a TRANSLATION and must stay one. ReactantServerExport is the bundle format's
# authority; the moment something here starts deciding what a manifest contains, what a weights file
# is named, or how a StableHLO artifact is serialized, the framework has grown a second, worse copy
# of a format it does not own.
#
# What it therefore does is exactly three things: turn `ExportSpec` into `IOSpec`, call
# `export_bundle`, and write the `model.jl` the writer deliberately does not write.

module ReactantNitroReactantServerExportExt

import ReactantNitro
import ReactantServerExport

using ReactantNitro: ExportSpec, ReactantServerBundle
using ReactantServerExport: IOSpec, export_bundle, collect_provenance

"""
    ReactantNitro.site_provenance(::ReactantServerBundle, root) -> Dict{String,Any}

The site half of provenance for this backend, which is `ReactantServerExport.collect_provenance`
and nothing else. It returns the commit, the branch, the dirty flag, the tree hash, the timestamp,
the remote, and `git_diff`: a full unified patch that the writer materializes as
`working_tree.patch` beside the manifest.

**That patch is the reason this is reached through a root rather than through a dictionary.** On a
dirty tree it is the only thing tying a bundle to the code that produced it, and it is multi-line, so
it cannot ride any flat `name=value` surface. A caller naming a directory can carry it; a caller
spelling out key-value pairs cannot.

The translation is one call, which is the whole point of this file: the collector belongs to the
package that owns the artifact format, and the framework's part is to have asked for it.
"""
ReactantNitro.site_provenance(::ReactantServerBundle, root) =
    collect_provenance(String(root))

# `ExportSpec` is 1-based Julia axes, like every other axis statement in ReactantNitro; `IOSpec` is
# 0-based network axes. That single subtraction is most of what the extra framework type buys, and it
# is the reason the public hooks can name no backend.
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
`model[.b{N}].mlir` per compiled batch size, and the `model.jl` postprocess when the experiment
ships one.

**The bundle directory is `joinpath(dir, name)`**, because the format requires a bundle's directory
basename to equal its declared name. That rule is the format's, so it is applied here rather than
being pushed onto every caller of `export_model`.

**Batch axes are passed to nobody.** `export_bundle` derives each tensor's batch axis as its last
Julia axis, which is precisely the rule ReactantNitro asserts on every array leaf of a batch and of
`forward`'s return, and which `export_model` verified against the traced output before calling here.
Passing `input_batch_axes`/`output_batch_axes` would let a declaration override a derivation that is
already correct, so the keywords are deliberately absent rather than defaulted.

**Client specs are passed only when the experiment declares them.** The wire preprocess lives
inside the traced graph, so the executable inputs already ARE the wire inputs and most models need
no client input spec at all; the server falls back to the executable specs, which is the same
answer. A fallback cannot carry `axis_letters`, which is what `export_client_inputs` is for.
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

    # The writer does not do this, by design: its docstring makes shipping `model.jl` the caller's
    # job, because the bundle format allows one and the writer has no way to produce one.
    postprocess === nothing || write(joinpath(written, "model.jl"), postprocess)

    return written
end

end # module ReactantNitroReactantServerExportExt
