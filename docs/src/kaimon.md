# Driving runs from Kaimon

ReactantNitro ships a KaimonGate extension that registers a set of `nitro_*` tools with the
running Kaimon gate, so an agent can start and watch training, validation, evaluation,
prediction, and export from the session where the model code is loaded.

The extension is a development tool. Nothing in `src/` depends on it, it costs nothing until
KaimonGate is loaded, and each tool calls an entry point the framework already ships.

## When the tools appear

The extension activates when KaimonGate and ReactantNitro are both loaded in one process.

**Load your model package first in every session.** Kaimon's `start_session` boots the gate but does
not load your model code; the extension registers its tools when the model package's
`using ReactantNitro` runs. After that, the `nitro_*` tools appear in the session's
tool list, namespaced by the project (e.g. `reactantnitro.nitro_train`).

If the tools go missing, `Base.get_extension(ReactantNitro, :ReactantNitroKaimonGateExt).reinstall_kaimon_tools()`
registers them again; it also starts a gate if none is running.

## Why runs are background tasks

Kaimon's agent-side tool calls carry a hard deadline, and its eval path fails after ten minutes
without output, and a training run lasts hours. So **no tool call blocks until the work
finishes**. Every launching tool starts the work on a background task in the session and returns a
run id immediately; the agent then polls `nitro_status` or `nitro_runs`, both fast calls, until the
run completes. `nitro_runs()` lists every run the session knows about:

```
a1b2c3d4  kind=train  status=completed  phase=Repl  epoch=40  step=1600  loss=0.021
```

Runs are process-local, so a session restart starts with an empty registry. To continue a run
after a restart, launch it again with `resume = "auto"`, which finds the latest checkpoint in its
`run_dir` and continues it, as `resume = :auto` does without the tools.

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
the current configuration with no arguments. It calls [`ReactantNitro.setup_devices!`](@ref),
the same function a REPL session calls. Calling it is optional: a session that never does runs on
Reactant's default backend with `n_devs` equal to the number of visible devices.

`backend` names a Reactant backend: `"cpu"`, `"gpu"`, `"cuda"`, `"rocm"`, `"tpu"`. `n_devs`
pins how many visible devices runs shard over; the pin wins over an experiment's `n_devs` field,
and an explicit `n_devs` keyword on `nitro_train` wins for that run (see
[Devices and meshes](devices.md) for the full order). One Julia
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
`preset`, `resume`, `weights`, and on `nitro_train` the booleans `restore_optimizer` and
`restore_best`) are run knobs passed to `Nitro`, where a keyword overrides the experiment's own
accessor for this run. `weights` and `resume` take a
checkpoint source as a string: a path, or `"run => checkpoint"` with the run `latest`, `all` or a
run id and the checkpoint `best` or `latest`, spaces optional (`"latest => best"`,
`"a1b2c3d4 => latest"`, `"all => best"`); `resume` also takes `"auto"` and `"false"`. Bare
`"best"` and `"latest"` are refused, and the error names the pair to write. Experiment fields
differ per model and cannot be in the tool schema, so they are passed through `overrides`, a
comma-separated `name=value` list of Julia literals, each converted to the field's declared type:

```julia
nitro_train(experiment="MyModels.MnistMLP", max_epochs=40,
            run_dir="runs/mnist_v1", overrides="width=128, smoothing=0.05, labels=[1.0,2.0]")
```

An override key that is not a field of the experiment is an error, and so is a key passed both as
a run keyword and in `overrides`. `preset` selects a named configuration
from `presets(MyExp)`, and `overrides` may add field values on top of it.

### Reusing a trained run

`nitro_validate`, `nitro_evaluate`, `nitro_predict`, and `nitro_export` accept either `run_id`
(a completed train run in this session, whose trained `Nitro` is reused) or `experiment` plus
`weights` (a checkpoint file, or a source such as `"latest => best"` read in `run_dir`, for work
in a process that did not train; `checkpoint` is its deprecated spelling):

```julia
nitro_validate(run_id="a1b2c3d4")
nitro_evaluate(experiment="MyModels.MnistMLP", split="test", weights="runs/mnist_v1/epoch-0040.jld2")
nitro_predict(run_id="a1b2c3d4", inputs="x=[1.0 2.0 3.0; 4.0 5.0 6.0]")
nitro_export(run_id="a1b2c3d4", dir="export_out", name="mnist_v1")
```

`inputs` is the predict batch as `name=value` literals, one per field `forward` declares; the
batch axis is last, exactly as the framework asserts on every batch, and arrays are converted
to Float32. `nitro_export` requires `dir` and `name` (the bundle lands in `dir/name`),
defaults `batch_sizes` to `[1]`, and uses the `reactant_server` backend, loaded on demand; a
backend named for the session can be registered instead. Export is a single-device CPU trace,
so a fresh construction from `experiment` forces `n_devs = 1`.

**An `experiment` export does not build the training data.** An export reads weights and traces a
graph, and needs no data, so the construction is the `Nitro(e; weights = path, data = (;))` that
[`export_model`](@ref) documents. This also lets you export a model whose export build differs from
its training build: set the flag in `overrides`, and `build_data`, which may refuse that
configuration, never runs.

```julia
nitro_export(experiment="MyModels.MyExperiment", preset="current",
             weights="runs/v2/epoch-0040.jld2", overrides="export_inference=true",
             dir="runs/export_out", name="my_model", provenance_root="/path/to/model/repo")
```

`data="build"` calls `build_data` anyway. It is needed only when there are no `weights`:
[`derive`](@ref) is skipped when `weights` names a checkpoint record, whose derived values are
restored, but without one `derive` runs, and a `derive` that reads the data needs it built. With
`run_id` the keyword is refused, since that handle is already built.

### Status, stop, and the numbers

`nitro_status` reports the current phase, epoch and step while running, the latest train loss
and validation metrics, the run directory, the result summary on completion, or the error text
on failure. Every launch kind drives the phase monitor: an export run reports
`ExportCompiling` while the trace is in flight (the framework publishes it around the backend
call), a predict run reports `EvalCompiling` when its `forward` has to compile, and a completed
run's phase reads `Repl`. The loss and metrics come from a recording logger the tools install
for each run, which forwards every call to the experiment's own logger. A run that names no logger
gets the JSON default, which writes `metrics.jsonl` into the run's directory. `nitro_status` also
carries a one-line logger summary.

`nitro_logger` reports the logger itself: the backend's type and the identifying parameters its
[`logger_info`](@ref) returns, read from the run's live handle, such as the JSON default's metrics
file path or a hosted tracker's URL and experiment key. The same table is available in the REPL as
`logger_info(nitro)`.

`nitro_stop` requests a graceful stop: the epoch finishes, validation runs, the checkpoint is
written, and the run exits through the normal `Done` path with `stop_reason = :requested`.

## What the tools do not do

The tools are the agent's interface to `train!`, `validate`, `evaluate`, `predict`, and
`export_model`, and can do nothing those entry points cannot. They do not manage checkpoints or
resumes beyond passing the keywords through: a `run_dir` is trained fresh unless `resume = "auto"`
asks for its latest checkpoint, as with `Nitro`, whose default is `resume = false`.
