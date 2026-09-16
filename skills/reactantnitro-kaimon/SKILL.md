---
name: reactantnitro-kaimon
description: >
  Drive ReactantNitro runs from a Kaimon-hosted session: the `nitro_*` GateTools the
  KaimonGate extension registers (train, validate, evaluate, predict, export, runs,
  status, stop), when the tools appear, why every tool returns immediately and the run
  runs in the background, how to name an experiment and split run knobs from experiment
  fields, and how the tools map to the framework verbs `train!`, `validate`, `evaluate`,
  `predict`, and `export_model`. Invoke whenever a session is hosted by Kaimon, when a
  workflow says "train the model", or when writing a harness-specific skill that will
  assume the tool surface.
---

# Driving runs from a Kaimon-hosted session

The ReactantNitro package carries an extension that registers a set of `nitro_*` tools with the
Kaimon gate. This page is the general guidance on that surface, written so that any harness that
hosts a Kaimon session for this project can use it, and so that a harness-specific skill can say
"train the model" and rely on the agent knowing the mechanism below.

**This page is about a dev tool, not a framework verb.** Nothing in `src/` knows the extension
exists. The tools are a thin interface over the same entry points everything else uses
(`train!`, `validate`, `evaluate`, `predict`, `export_model`), and the framework behaves
identically whether or not the tools are present. If your harness does not host Kaimon, call the
verbs directly; the rest of the skills teach those. If it does, the tools are the driver.

## The one thing to know first

`using ReactantNitro` is the first step of any Kaimon-hosted session. Kaimon's session boot loads
the gate session library but not your model code, and it serves the gate LAST, so the model
package's `using ReactantNitro` normally fires the extension before there is a gate to register
with. The extension handles that ordering itself: it registers immediately if a gate is already
serving, and otherwise waits for one, checking once a second for 30 seconds. So the tools appear
when the gate binds, which in a normal session is a second or two after the model package loads
and before you can call anything. After that the `nitro_*` tools are in the session's tool list,
namespaced by the project (e.g. `reactantnitro.nitro_train`).

The one case that window does not cover is a gate served more than 30 seconds after the model
package loaded, which a hand-driven session can produce. If the tools are missing, this
re-registers them and also starts a gate if none is running:

```julia
Base.get_extension(ReactantNitro, :ReactantNitroKaimonGateExt).reinstall_kaimon_tools()
```

## Why every tool returns immediately

Kaimon's agent-side tool calls carry a hard deadline, and its eval path fails after ten minutes
without output. A training run lasts hours. So **no tool call blocks until the work finishes**.
Every launching tool does the same two things:

1. Starts the work on a background task in the session and returns a run id immediately.
2. Lets the agent poll `nitro_status` or `nitro_runs`, each a fast call, until the run
   completes.

A workflow that "waits for training" is a poll loop, not a blocking call in the session. The poll
is cheap; the run is unaffected by the caller. This is the whole of the timeout workaround. For CODE
that waits, `ReactantNitro.run_state(run_id)` is the same information as a NamedTuple (`status`,
`phase`, `epoch`, `step`, `loss`, `val_metrics`, ...), so a predicate reads
`run_state(id).status !== :running` rather than a regex over `nitro_status` text. A harness that can
poll on the agent's behalf should build its wait on that: a downstream harness whose wait reads
`run_state` gives the agent one turn back instead of one turn per poll.

**Do not "simplify" the extension's `@async`.** The gate dispatches every tool handler on a
default-pool thread, so the `@async` in `_launch_run!` already runs the run off the interactive
thread; the framework's `with_repl` spawn (which the entry points use) sees a worker-thread caller
and runs inline rather than double-spawning. The `@async` is also what lets the run registry set
`status = :completed` when the run finishes; removing it would break the tool contract that
`nitro_status` and `nitro_runs` are built on.

## The tool surface

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
| `nitro_stop` | Request a graceful stop of one run |

### Configuring the accelerator first

`nitro_setup(; backend, n_devs)` configures the session's accelerator before any run, or reports
the current configuration with no arguments. It is a **thin wrapper over
`ReactantNitro.setup_devices!`**, the same function a REPL session calls, so a REPL workflow
and a Kaimon workflow configure identically, and calling it is optional: a session that never
does runs on Reactant's default backend with `n_devs` = every visible device.

`backend` names a Reactant backend: `"cpu"`, `"gpu"`, `"cuda"`, `"rocm"`, `"tpu"`. `n_devs`
pins how many visible devices runs shard over; the pin wins over an experiment's declared
`n_devs`, and an explicit `n_devs` keyword on `nitro_train` wins for that run. One Julia
process initializes XLA once, so to restrict which GPUs the session sees, set
`CUDA_VISIBLE_DEVICES` at session start. The full model lives in `reactantnitro-accelerators`.

```julia
nitro_setup(backend="cpu")            # run everything on CPU
nitro_setup(backend="cuda", n_devs=2) # two of the visible CUDA devices
nitro_setup()                         # report what is in effect
```

### Naming the experiment

Every launching tool takes the experiment as a module-qualified type string, resolved against
the session's loaded modules: `MyModels.MnistMLP`, or a bare name the model package exported
into `Main`. The model package must be loaded in the session; that is the same `using` that made
the tools appear.

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
to Float32. `nitro_export` requires `dir` and `name` (the bundle lands in `dir/name`), defaults
`batch_sizes` to `[1]`, and uses the `reactant_server` backend, loaded on demand. Export is a
single-device CPU trace, so a fresh construction from `experiment` forces `n_devs = 1`.

### Status, stop, and the numbers

`nitro_status` reports the current phase, epoch and step while running, the latest train loss
and validation metrics, the run directory, the result summary on completion, or the error text
on failure. The loss and metrics come from a recording logger the tools install for each run,
which also tees to the experiment's own logger, so a hosted tracker keeps receiving what it always
received.

**Every launch kind drives the phase monitor.** Export runs report `ExportCompiling` while the
trace is in flight (the framework publishes it around the backend call), and a predict run
reports `EvalCompiling` when its `forward` has to compile; train, validate, and evaluate report
their phases as before. A run's recorded phase ends at `Repl` once it completes.

`nitro_stop` requests a graceful stop: the epoch finishes, validation runs, the checkpoint is
written, and the run exits through the normal `Done` path with `stop_reason = requested`.

**A REPL's ^C is the same stop, pressed by a different hand.** The entry points run on a worker
thread, and ^C is delivered to the parked caller, which converts it into `request_stop!`: the
same flag `nitro_stop` sets, with the identical wind-down (epoch, validation, checkpoint,
`stop_reason = :requested`). The only difference is who presses it: the human at the REPL uses
^C (see `reactantnitro-checkpoint-resume`), the agent driving a session uses `nitro_stop`. The
tool is the agent's way to stop a run it launched; it cannot stop a run the human started with
`train!` at the REPL, and ^C cannot stop a tool-launched run (that one stops with `nitro_stop`).

## The run model

Runs are process-local: a session restart starts with an empty registry. That is not a gap,
because the framework's checkpoint machinery is the restart story: `resume = :auto` finds the
latest checkpoint in a run's directory and continues it, exactly as it does without the tools.
`nitro_runs` lists every run the session knows about:

```
a1b2c3d4  kind=train  status=completed  phase=Repl  epoch=40  step=1600  loss=0.021
```

## The map to the framework verbs, and what downstream skills may assume

Each tool is the tool form of a framework verb, with the same semantics and the same knobs:

| Tool | Verb |
| --- | --- |
| `nitro_setup` | `setup_devices!` |
| `nitro_train` | `train!` |
| `nitro_validate` | `validate` |
| `nitro_evaluate` | `evaluate(nitro; split)` |
| `nitro_predict` | `predict` |
| `nitro_export` | `export_model` |
| `nitro_stop` | `request_stop!` (what a REPL's ^C calls) |

**A harness-specific skill may say "train the model" and assume this mechanism.** The agent
reads the run into a session, ensures `using ReactantNitro` has run, launches with
`nitro_train` (naming the experiment and passing run knobs and `overrides`), waits until the run
completes (a harness wait built on `run_state` where the host provides one, a `nitro_status` poll
about a minute apart where it does not), reads the result and the checkpoint path, and uses
`nitro_stop` when the workflow asks for an early stop. The same pattern covers validation,
evaluation, prediction, and export: launch, wait, read the result.

## What the tools do not do

They do not replace the framework. The tools are the interface to `train!`, `validate`,
`evaluate`, `predict`, and `export_model`; anything those entry points cannot do, the tools
cannot do. They also do not manage checkpoints or resumes beyond passing the knobs through: a
a `run_dir` trains fresh unless `resume = "auto"` asks for the latest checkpoint in it, and
`resume = "false"` is the default that starts
over, exactly as with `Nitro`.
