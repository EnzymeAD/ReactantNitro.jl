# Export: training and serving, Julia native

The `forward` you trained is retraced into the served program, without leaving Julia and Reactant.

```julia
using ReactantServerExport                  # the extension that provides the backend

n = Nitro(e; weights = "runs/x/best.jld2", data = (;))   # no training in this process
export_model(n, ReactantServerBundle(); dir = "export_out", name = "mnist_v1")
```

`export_model` takes a `Nitro`, which already holds the restored weights, so a model's export code
never loads a checkpoint itself. `data = (;)` skips `build_data`, so an export process needs no
training data.

## The wire contract

Export hooks declare what clients send and receive:

- `export_inputs` and `export_outputs` declare the wire shapes and dtypes.
- `export_preprocess` converts what clients send, usually `UInt8` pixels, into what the model trains
  on, usually normalized `Float32`. **This happens inside the traced graph**, so what ships is
  `export_preprocess ∘ forward` as one program and a client never has to reproduce your
  normalization.
- `export_postprocess` and `export_client_inputs` / `export_client_outputs` cover the other side of
  the seam when the served contract differs from the traced one.

The framework translates axes: an `ExportSpec` uses 1-based Julia axes, like every other axis in
the package, and the backend's `IOSpec` uses 0-based axes.

## What a bundle is

The artifact is a ReactantServer StableHLO bundle:

- `manifest.yaml`
- `weights.safetensors`
- one `model[.bN].mlir` per compiled batch size
- `model.jl`, when the model ships a postprocess

Export is a single-device CPU trace, so it needs no GPU, and a handle with a mesh is refused; build
the export handle with `n_devs = 1`. The bundle always records the config, seed, preset, framework
version, run directory and checkpoint. Repository state (the commit, the tree hash, and on a dirty
tree the working-tree patch) is recorded only when you pass `provenance_root`, the repository root;
without it the export succeeds but cannot say which code produced it.

## TensorFlow SavedModel

`TFSavedModel` writes a SavedModel for TF Serving and the platforms built on it (SageMaker,
Vertex AI, KServe). Each traced program is one `XlaCallModule` op wrapping the StableHLO, the form
JAX's native serialization produces, so the program is packaged, not converted.

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

An older `stablehlo_version` reaches older releases; a module that uses an op newer than that
version then fails at export, not at load. A program containing a `custom_call` is refused, since `XlaCallModule` cannot
run one portably.

## Caveats

A view-shaped program output has to be `copy`ed before it leaves the traced graph, or the bundle
carries a wrapper rather than an array. This does not apply to parameters, which the flat layout
reconstructs through a `copy` already.

Each backend is an extension on a weak dependency, `ReactantServerExport` or `PythonCall`, so a
deployment that never exports does not load either.
