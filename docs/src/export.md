# Export: training and serving, Julia native

The same `forward` that trained becomes the servable program. Nothing leaves the Julia and Reactant
ecosystem between the first `train!` and the served bundle.

```julia
using ReactantServerExport                  # the extension that provides the backend

n = Nitro(e; weights = "runs/x/best.jld2", data = (;))   # no training in this process
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

## TensorFlow SavedModel

`TFSavedModel` writes a SavedModel for TF Serving and the platforms built on it (SageMaker,
Vertex AI, KServe). Each traced program is one `XlaCallModule` op wrapping the StableHLO, the form
JAX's native serialization produces, so this is packaging and not a conversion.

```julia
using PythonCall                            # the extension; `tensorflow` must be importable

export_model(n, TFSavedModel(); dir = "export_out", name = "mnist_v1", batch_sizes = [1, 8])
```

- **One signature per batch size.** Every compiled program is static, so `batch_sizes = [1, 8]`
  gives `serving_b1` and `serving_b8`, sharing one copy of the weights; `serving_default` is the
  first. A client picks one with `signature_name`.
- **Batch-first tensors.** The Julia shape `(W, H, C, N)` is the TF shape `[N, C, H, W]`. Signature
  inputs and outputs are named by `export_inputs` and `export_outputs`.
- **One platform.** `platform = "CUDA"` by default; the module runs only there. Use `"CPU"` for a
  CPU server.
- **No postprocess.** A SavedModel cannot carry `model.jl`. A model declaring
  `export_client_outputs` is refused; any other postprocess is dropped with a warning.
- **Provenance** is written to `assets.extra/provenance.json`, with the working-tree patch beside it,
  which TF Serving ignores.

Verified TensorFlow releases, loading and running every signature on CPU:

| TF | Options |
| --- | --- |
| 2.17, 2.20 | the defaults, `stablehlo_version = v"1.5.0"`, `call_module_version = 9` |
| 2.15 | `stablehlo_version = v"0.14.0"`, `call_module_version = 8` |

An older `stablehlo_version` reaches older releases; a module using a newer op then fails to export
rather than at load. A program containing a `custom_call` is refused, since `XlaCallModule` cannot
run one portably.

## Caveats

A view-shaped program output has to be `copy`ed before it leaves the traced graph, or the bundle
carries a wrapper rather than an array. This does not apply to parameters, which the flat layout
reconstructs through a `copy` already.

Each backend lives behind a weak dependency, `ReactantServerExport` or `PythonCall`, so nothing in
`src/` knows an artifact format and a deployment that never exports pays nothing for it.
