# Export: training and serving, Julia native

The same `forward` that trained becomes the servable program. Nothing leaves the Julia and Reactant
ecosystem between the first `train!` and the served bundle.

```julia
using ReactantServerExport                  # the extension that provides the backend

n = Nitro(e; checkpoint = "runs/x/best.jld2", data = (;))   # no training in this process
export_model(n, ReactantServerBundle(); dir = "export_out", name = "mnist_v1")
```

`export_model` takes a `Nitro`, not a path, so a model's export code never loads a checkpoint
itself. `data = (;)` skips `build_data` entirely, so an export process does not have to produce a
training split to get started.

## The wire contract

Export hooks declare what clients send and receive:

- `export_inputs` and `export_outputs` declare the wire shapes and dtypes.
- `export_preprocess` converts what clients send, usually `UInt8` pixels, into what the model trains
  on, usually normalized `Float32`. **This happens inside the traced graph**, so what ships is
  `export_preprocess ∘ forward` as one program and a client never has to reproduce your
  normalization.
- `export_postprocess` and `export_client_inputs` / `export_client_outputs` cover the other side of
  the seam when the served contract differs from the traced one.

Axes are the one place the two worlds disagree, and the framework owns the translation:
`ExportSpec` is 1-based Julia axes like every other axis statement in the package, and the backend's
`IOSpec` is 0-based network axes.

## What a bundle is

The artifact is a ReactantServer StableHLO bundle:

- `manifest.yaml`
- `weights.safetensors`
- one `model[.bN].mlir` per compiled batch size
- `model.jl`, when the model ships a postprocess

Export is a single-device CPU trace, so it needs no GPU and no accelerator lease. Provenance
(config, seed, preset, framework version, and on a dirty tree a full working-tree patch) lands in
the bundle automatically, so a served artifact can be traced back to the code that produced it.

## Caveats worth knowing up front

A view-shaped program output has to be `copy`ed before it leaves the traced graph, or the bundle
carries a wrapper rather than an array. This does not apply to parameters, which the flat layout
reconstructs through a `copy` already.

The backend lives behind a weak dependency, `ReactantServerExport`, so nothing in `src/` knows the
bundle format exists and a deployment that never exports pays nothing for it.
