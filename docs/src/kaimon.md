# Driving runs from Kaimon

ReactantNitro ships a KaimonGate extension that registers a set of `nitro_*` tools with the
running Kaimon gate, so an agent can start and watch training, validation, evaluation,
prediction, and export from the session where the model code is loaded.

This page is about the tool interface, what the tools can and cannot do, and why they work the
way they do. The extension is a dev tool: nothing in `src/` knows it exists, it costs nothing
until KaimonGate is loaded, and the tools are a thin interface over entry points the framework
already ships.

## When the tools appear

The extension activates when KaimonGate and ReactantNitro are both loaded in one process, which
is exactly the shape of a Kaimon-hosted session. The one thing to know:

**`using ReactantNitro` is the first step of any session.** Kaimon's `start_session` boots the
gate but does not load your model code; the extension registers its tools the moment the model
package's `using ReactantNitro` runs. After that, the `nitro_*` tools appear in the session's
tool list, namespaced by the project (e.g. `reactantnitro.nitro_train`).

If the tools ever go missing, `Base.get_extension(ReactantNitro, :ReactantNitroKaimonGateExt).reinstall_kaimon_tools()`
registers them again; it also starts a gate if none is running.

## Why runs are background tasks

Kaimon's agent-side tool calls carry a hard deadline, and its eval path fails after ten minutes
without output. A training run lasts hours. So **no tool call ever blocks until the work
finishes**. Every launching tool does the same two things:

1. Starts the work on a background task in the session and returns a run id immediately.
2. Lets the agent poll `nitro_status` or `nitro_runs`, each a fast call, until the run
   completes.

A run id comes back in the launching tool's reply, and `nitro_runs()` lists every run the
session knows about:

```
a1b2c3d4  kind=train  status=completed  phase=Repl  epoch=40  step=1600  loss=0.021
```

Runs are process-local. A session restart starts with an empty registry, which is fine because
the framework's checkpoint machinery is the restart story: `resume = :auto` finds the latest
checkpoint in a run's directory and continues it, exactly as it does without the tools.

## The tools

| Tool | What it does |
| --- | --- |
| `nitro_setup` | Configure this session's accelerator (backend, `n_devs`) or report the current configuration |
| `nitro_train` | Start training in the background |
| `nitro_validate` | Run the `val` split over a `Nitro` |
| `nitro_evaluate` | Run one split (default `test`) over a `Nitro` |
| `nitro_predict` | Run `predict` on one batch |
| `nitro_export` | Export a trained model to a bundle backend |
| `nitro_runs` | List the session's runs |
| `nitro_status` | Report one run in detail |
| `nitro_logger` | Report a run's logger type and key parameters |
| `nitro_stop` | Request a graceful stop of one run |

### Configuring the accelerator first

`nitro_setup(; backend, n_devs)` configures the session's accelerator before any run, or reports
the current configuration with no arguments. It is a **thin wrapper over
[`ReactantNitro.setup_devices!`](@ref)**, the same function a REPL session calls, so a REPL
workflow and a Kaimon workflow configure identically, and calling it is optional: a session
that never does runs on Reactant's default backend with `n_devs` = every visible device.

`backend` names a Reactant backend: `"cpu"`, `"gpu"`, `"cuda"`, `"rocm"`, `"tpu"`. `n_devs`
pins how many visible devices runs shard over; the pin wins over an experiment's declared
`n_devs`, and an explicit `n_devs` keyword on `nitro_train` wins for that run. One Julia
process initializes XLA once, so to restrict which GPUs the session sees, set
`CUDA_VISIBLE_DEVICES` at session start.

```julia
nitro_setup(backend="cpu")            # run everything on CPU
nitro_setup(backend="cuda", n_devs=2) # two of the visible CUDA devices
nitro_setup()                         # report what is in effect
```

### Naming the experiment

Every launching tool takes the experiment as a module-qualified type string, resolved against
the session's loaded modules: `MyModels.MnistMLP`, or a bare name the model package exported
into `Main`. The model package must be loaded in the session first; that is the same `using`
that made the tools appear.

### Run knobs and experiment fields

The typed keywords (`max_epochs`, `run_dir`, `seed`, `n_devs`, `accum`, `gradient_clip_norm`,
`preset`, `resume`, `checkpoint`) are run knobs passed to `Nitro`, where a keyword beats the
experiment's own accessor for this run (the same rule the framework documents). Experiment
fields are per-model and cannot be in the schema, so they arrive through `overrides`, a
comma-separated `name=value` list of Julia literals:

```julia
nitro_train(experiment="MyModels.MnistMLP", max_epochs=40,
            run_dir="runs/mnist_v1", overrides="width=128, smoothing=0.05, labels=[1.0,2.0]")
```

An override key that is not a field of the experiment is a loud error, and a key passed both as
a run keyword and in `overrides` is an ambiguity error. `preset` selects a named configuration
from `presets(MyExp)`, and `overrides` may add field values on top of it.

### Reusing a trained run

`nitro_validate`, `nitro_evaluate`, `nitro_predict`, and `nitro_export` accept either `run_id`
(a completed train run in this session, whose trained `Nitro` is reused) or `experiment` plus
`checkpoint` (a checkpoint file, for work in a process that did not train):

```julia
nitro_validate(run_id="a1b2c3d4")
nitro_evaluate(experiment="MyModels.MnistMLP", split="test", checkpoint="runs/mnist_v1/epoch-0040.jld2")
nitro_predict(run_id="a1b2c3d4", inputs="x=[1.0 2.0 3.0; 4.0 5.0 6.0]")
nitro_export(run_id="a1b2c3d4", dir="export_out", name="mnist_v1")
```

`inputs` is the predict batch as `name=value` literals, one per field `forward` declares; the
batch axis is last, exactly as the framework asserts on every batch, and arrays are converted
to Float32. `nitro_export` requires `dir` and `name` (the bundle lands in `dir/name`),
defaults `batch_sizes` to `[1]`, and uses the `reactant_server` backend, loaded on demand; a
backend named for the session can be registered instead. Export is a single-device CPU trace,
so a fresh construction from `experiment` forces `n_devs = 1`.

**An `experiment` export does not build the training data.** Setup for an export reads weights
and traces a graph, and the data is part of neither, so the construction is the same
`Nitro(e; checkpoint = path, data = (;))` that [`export_model`](@ref) documents. It is what
makes a model whose exportable handle is a *different build* from its trainable one exportable
through the tool at all: name the flag in `overrides`, and the handle is built with it without
a `build_data` that refuses an inference configuration ever running.

```julia
nitro_export(experiment="MyModels.MyExperiment", preset="current",
             checkpoint="runs/v2/epoch-0040.jld2", overrides="export_inference=true",
             dir="runs/export_out", name="my_model", provenance_root="/path/to/model/repo")
```

`data="build"` calls `build_data` anyway, for the one case that needs it: [`derive`](@ref) is
skipped on a `checkpoint =` construction, but an export from an `experiment` with no checkpoint
recomputes the derived values, so a model whose `derive` reads its data has to build. With
`run_id` the keyword is refused, since that handle is already built.

### Status, stop, and the numbers

`nitro_status` reports the current phase, epoch and step while running, the latest train loss
and validation metrics, the run directory, the result summary on completion, or the error text
on failure. Every launch kind drives the phase monitor: an export run reports
`ExportCompiling` while the trace is in flight (the framework publishes it around the backend
call), a predict run reports `EvalCompiling` when its `forward` has to compile, and a completed
run's phase reads `Repl`. The loss and metrics come from a recording logger the tools install for each run,
which also tees to the experiment's own logger, so a hosted tracker keeps receiving what it
always received. A run that names no logger gets the framework's JSON default, which writes
`metrics.jsonl` into the run's directory, and `nitro_status` carries a one-line logger summary.

**`nitro_logger` reports the logger itself.** For any run, it names the backend's type and the
key identifying parameters its [`logger_info`](@ref) exposes, read live off the run's handle:
the metrics file path of the JSON default, or the URL, experiment key, and workspace of a hosted
tracker. The same table is on the `Nitro` itself, as `logger_info(nitro)`, so an agent driving a
session has a direct answer to "which experiment did this run attach to, and where is it".

`nitro_stop` requests a graceful stop: the epoch finishes, validation runs, the checkpoint is
written, and the run exits through the normal `Done` path with `stop_reason = requested`.

## What the tools do not do

They do not replace the framework. The tools are the agent's interface to `train!`, `validate`,
`evaluate`, `predict`, and `export_model`; anything those entry points cannot do, the tools
cannot do. They also do not manage checkpoints or resumes beyond passing the knobs through: a
fresh `run_dir` trains fresh, a reused one resumes by default, and `resume = "false"` starts
over, exactly as with `Nitro`.
