<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-dark.svg">
    <img src="docs/src/assets/logo.svg" alt="ReactantNitro.jl" width="140">
  </picture>
</p>

<h1 align="center">ReactantNitro.jl</h1>

<p align="center"><em>Reactant-first training for Lux models: the experiment is a struct whose fields decide what XLA compiles once and what it never recompiles.</em></p>

<p align="center">
  <a href="https://github.com/EnzymeAD/ReactantNitro.jl/actions/workflows/ci.yml"><img src="https://github.com/EnzymeAD/ReactantNitro.jl/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://enzymead.github.io/ReactantNitro.jl/"><img src="https://github.com/EnzymeAD/ReactantNitro.jl/actions/workflows/docs.yml/badge.svg" alt="Docs"></a>
  <img src="https://img.shields.io/badge/Julia-1.12-9558b2" alt="Julia 1.12">
  <a href="https://github.com/fredrikekre/Runic.jl"><img src="https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black" alt="code style: runic"></a>
</p>

## Overview

ReactantNitro is a training framework for Julia, built on [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
and [Lux.jl](https://github.com/LuxDL/Lux.jl). The model is an ordinary Lux model, every compiled
program is Reactant, and XLA runs it. You write the experiment as a struct plus a handful of hooks;
the framework supplies the compiled programs, the device transfers, the optimizer, the schedules,
checkpointing, and the run's lifecycle. Training and serving both stay in Julia, from the first
`train!` to the exported [bundle](#export-training-and-serving-julia-native).

The design follows PyTorch Lightning, pointed at Reactant. Lux has a training loop, but a
Reactant-first stack also needs gradient accumulation, a phase system, schedules, and control over
when XLA compiles, because with Reactant an unneeded compile is the easiest way to lose time. The
framework does not spend your time compiling when no compile is needed: the three field markers
below tell it which values shape the program and which only flow through it, so sweeping a
hyperparameter, scheduling a value, changing the learning rate, reseeding, or restarting a run
reuses the program it already has.

## The three field markers

Every field on an experiment carries one of three markers. The marker decides what the compiled
program sees and whether changing the value recompiles:

| Marker | Reaches traced code as | In the compile key? | Changing the value |
| --- | --- | --- | --- |
| `GraphConst{T}` | a baked literal | yes | recompiles, correctly: a different value is a different program |
| `Device{T}` | a device-resident traced input | no, by construction | never recompiles: sweep it, schedule it, rewrite it live |
| unmarked, i.e. `Host{T}` | not at all | no | never recompiles: driver-only, invisible to the tracer |

Unmarked means `Host` because that is the common case. In the first model ported to the framework,
83% of the fields were `Host`.

```julia
@experiment struct MyExp
    "Structural: changes the emitted graph, so it bakes and is part of the compile key."
    width::GraphConst{Int} = 128

    "A traced input: sweep it or schedule it without recompiling."
    smoothing::Device{Float32} = 0.05f0

    "Unmarked, therefore Host: driver-only and invisible to the tracer."
    max_epochs::Int = 20
end
```

Programs are keyed and stored once per process, not once per `Nitro`. The key holds the function,
its argument types and shapes, a hash of the `GraphConst` fields only, and the resolved method world
of every user hook, so a redefined function is never silently ignored. `Device` fields are excluded
because they are traced inputs rebuilt every step, and `Host` fields because they are not in the
traced view at all.

The entry points add a second guard on top of the key: each cached program also records the
transitive closure of the methods it was compiled against, and `train!`/`validate`/`predict`
re-resolve that closure against live dispatch, poisoning any entry whose methods moved. A
redefinition anywhere below the hooks, a helper `forward` calls or a dependency method, therefore
misses for a new `Nitro`, which recompiles against current dispatch; an existing `Nitro` keeps the
programs it was built with, now stale, and the entry-point report says so. The one class that still
needs a REPL restart is values, not methods: a `const` redefined, a global, or a literal edited in
place.

The `Host` marker also keeps compiles fast. Reactant and Enzyme traverse the whole experiment while
tracing, so a dataset-sized field reachable from it gets walked element by element on every
compile. The framework hands the trace a stripped view of the experiment, `compile_view(e)`, in
which every `Host` field is a sentinel carrying only the field's name. The real `e` runs everywhere
outside the trace: `build_data`, `derive`, metric finalization, checkpointing, and every driver
decision read it normally.

`ReactantNitro.cache_stats()` shows from the REPL whether a change recompiled, and
`ReactantNitro.cache_reset!()` clears the module-level cache. A `GraphConst` change misses the
cache; a `Device` change or a `Host` change hits.

## Quick start

Four hooks are required. Everything else has a default: the optimizer (RAdam at 1e-3), one
parameter group, no decay, no schedule, prefetching, validation, checkpointing, and a logging
contract that ships no backend.

```julia
using ReactantNitro, Lux, Random

@experiment struct MnistMLP
    width::GraphConst{Int} = 128
    smoothing::Device{Float32} = 0.05f0
    max_epochs::Int = 5
end

ReactantNitro.build_model(e::MnistMLP, rng) = begin
    model = Chain(Dense(784 => e.width, relu), Dense(e.width => 10))
    (model, Lux.setup(rng, model)...)
end

ReactantNitro.build_data(e::MnistMLP, dist) = (;
    train = [(; img = randn(Float32, 784, 32), label = rand(Float32, 10, 32)) for _ in 1:100],
    val   = [(; img = randn(Float32, 784, 32), label = rand(Float32, 10, 32)) for _ in 1:20],
)

function ReactantNitro.forward(e::MnistMLP, model, ps, st; img)
    logits, st_new = Lux.apply(model, img, ps, st)
    return logits ./ e.smoothing, st_new
end

function ReactantNitro.loss(e::MnistMLP, logits; label)
    logp = logits .- log.(sum(exp, logits; dims = 1))
    return -sum(label .* logp) / size(label, 2)
end

n = Nitro(MnistMLP())
train!(n)
```

`Nitro(e)` runs the setup sequence and nothing else, so validation, evaluation, and prediction never
depend on a `train!` having happened in the process. Each hook declares the batch fields it wants as
keywords, the framework routes those to the device, and a field no hook declares is never
transferred. The batch dimension is always last.

A follow-up run is a new handle and costs no compiles: the cache is module-level, `resume = :auto`
is the default, and `data = n.data` reuses the loaded collection instead of re-running `build_data`.

```julia
n2 = Nitro(MnistMLP(); data = n.data, max_epochs = 20)   # continues from the latest checkpoint
train!(n2)
ReactantNitro.cache_stats()                              # misses stays flat: every program reused
```

`validate`, `evaluate`, and `predict` share one compiled `forward` with each other, so moving
between them never recompiles. `predict` pads any batch to the compiled width and slices the outputs
back, so the one compiled program serves any batch size, including a single inference request.

## Built for the REPL and Revise

The interface is meant to be driven from a REPL with Revise loaded. Everything that decides which
compiled program a `Nitro` runs is resolved at construction, so an edit never changes a running
handle and a handle never recompiles underneath you. Edit a hook, build a new `Nitro`, and the
module-level cache recompiles only what the edit changed. Every entry point prints what it fixed at
construction and names any hook redefined since, so a stale handle is never silent.

Long runs leave the interactive thread. `train!`, `validate`, `evaluate`, and `predict` block while
XLA compiles and executes, and a compile or execute on the interactive thread starves every other
task on it: logger tasks, a Kaimon gate's message loop, the REPL itself. Whenever a worker pool
exists (`-t N,1`), the entry points run their loop on a default-pool worker and park your call, so
the REPL and your logger tasks keep running for the whole run. Ctrl+C then requests a graceful
stop: `train!` finishes the epoch, validates, checkpoints, and returns with
`stop_reason = :requested`, the same wind-down `nitro_stop` performs, and a run that fails surfaces
its own exception, not a task wrapper. Single-threaded Julia and the Kaimon tool path are unchanged:
with no worker pool the loop runs inline, and tool-driven runs were already executing on a worker
thread.

The one thing that can change on a live handle is a `Device` value, through `set_device!`, which
never recompiles. That is the inference-sweep loop:

```julia
n = Nitro(e; data = (;), checkpoint = "runs/x/best.jld2")   # no training split needed

for t in (0.5f0, 1f0, 1.5f0, 2f0)
    set_device!(n; temperature = t)
    logits = predict(n, batch)               # zero compiles, every iteration
end

device_value(n, :temperature)                # reads it back as a host value
```

`set_device!` refuses anything that would recompile, and the error says why: a `GraphConst` field is
baked into the program, a `Host` field reaches no trace, a scheduled field belongs to the schedule,
and a different element type or size changes the key.

## The automatic loop and the lifecycle

`train!` sequences the forwards, the backwards, the optimizer steps, and the metrics, and wraps the
run in a lifecycle with monitorable phases (`Starting`, `Compiling`, `Stepping`, `Checkpointing`,
`Terminal`, `Done`, `Failed`), `request_stop!` for graceful interruption, and a progress counter.
The `train` split is wrapped in a `PrefetchIterator` automatically, so the next batch's
host-to-device transfer overlaps the current step.

Metrics are `(sum, count)` pairs with a per-hook residency choice, `:host` or `:device`, detailed
in the next section; `finalize_metrics` reduces host-side when a macro-averaged recall is not the
mean of per-batch recalls. Checkpointing defaults to `TopKCheckpointer` and early stopping to
`EarlyStopping` when you opt in (the default is off); both are run accessors, and `resume = :auto`
continues a run from the latest checkpoint behind a config-compatibility check and an anchor
checksum. The logging contract is a set of verbs
(`log_metrics!`, `log_params!`, `log_other!`, `finish!`, ...) on your own type: the framework ships
no backend, `nothing` is the public "no logging" value, and your tracker's run handle goes in
directly.

## Metrics on the host or the device

A metric is a `(sum, count)` pair, so it carries its own denominator: a per-image accuracy, a
per-token loss, and a confusion matrix coexist in one call, and the framework never supplies a
sample count of its own, because there is no single right one. `count === nothing` means accumulate
by summation and never divide, which is what a confusion matrix wants. Metrics never see padding:
the framework pads a short final batch, runs the compiled program, slices the outputs back, and
only then calls you. `finalize_metrics` reduces host-side for anything that is not a mean of
per-batch values, such as a macro-averaged recall, and with no `metrics` method at all the
framework substitutes `val_loss`, so a bare experiment still validates.

Where a metric runs is a per-hook choice that trades compile cost, expressiveness, and transfer
cost against each other:

| | `:device` (traced) | `:host` (ordinary Julia) |
| --- | --- | --- |
| Performance | the model's raw outputs never cross the device boundary; only the metric scalars come back | every eval batch transfers the outputs to the host |
| Expressiveness | must be expressible as a traced program | anything Julia can do: matching, connected components, sorting with tie-breaking, any library |
| Editing the hook | edits the program, so it recompiles | part of no program, so it recompiles nothing |

The defaults follow the cadence. `train_metrics` runs once per micro-batch, so it is traced by
default; transferring a full output batch that often would cost far more than the few scalars it
produces. `metrics` runs once per eval batch, once per epoch, after a training epoch of thousands
of steps, so the same transfer is noise, and host residency lets evaluation code use anything Julia
can do.

Validation metrics for the `MnistMLP` from the Quick start, in host mode, the default for
`metrics`: ordinary Julia on transferred arrays, once per eval batch. `validate(nitro)` reports
these every epoch, and `evaluate(nitro; split = :test)` runs the same hooks on the test split.

```julia
function ReactantNitro.metrics(::MnistMLP, logits; label)
    pred = getindex.(argmax(logits; dims = 1), 1)    # (1, B), the predicted class
    truth = getindex.(argmax(label; dims = 1), 1)    # (1, B), the true class
    cm = zeros(Int, 10, 10)
    for (p, t) in zip(pred, truth)
        cm[p, t] += 1
    end
    return (;
        acc = (sum(pred .== truth), size(label, 2)),  # per image: divided at the end
        confusion = (cm, nothing),                    # summed across the split, never divided
    )
end

function ReactantNitro.finalize_metrics(::MnistMLP, acc, split)
    recall = [acc.confusion[c, c] / max(sum(acc.confusion[:, c]), 1) for c in 1:10]
    return (; acc.acc, macro_recall = sum(recall) / 10)
end
```

`acc` carries a denominator and `confusion` carries none. In `finalize_metrics`, counted keys arrive
already divided and `nothing`-counted keys as raw totals, so `acc.confusion` is the whole split's
matrix and `macro_recall` is computed once at the end rather than averaged per batch. Neither hook
is traced, so editing either recompiles nothing.

```julia
# The defaults, written out. You would not normally write this line.
ReactantNitro.metrics_residency(::MyExp, hook) =
    hook === :train_metrics ? :device : :host

# Trace the validation metric instead: it is cheap to express and the eval set is large.
ReactantNitro.metrics_residency(::MyExp, ::Symbol) = :device
```

Both hooks accept both values. The only difference for the hook author is what `outputs` is: device
arrays under `:device`, host arrays under `:host`. Keyword routing, the `(sum, count)` contract,
`count === nothing`, and the no-padding guarantee are the same either way. The cost to weigh is
compilation: editing a traced `train_metrics` repays the gradient compile, while a host metric can
be added mid-session without one.

## Manual mode

Manual mode exists for algorithms the automatic loop cannot express. Define `train_step` for your
experiment and you own the whole optimizer step; the framework keeps everything outside it: the
loop, prefetch, validation, checkpointing, early stopping, and the phase lifecycle.

The standard example is a GAN: one generator and one discriminator in a single parameter tree, each
with its own optimizer, two losses against the same batch, and a generator gradient that flows
through the discriminator's forward as a function of the fake data, never into the discriminator's
parameters. No explicit stop-gradient op is needed.

```julia
using Statistics: mean
using Lux, Optimisers   # or the same `using` lines as the quick start

ReactantNitro.setup_optimizers(e::ToyGAN, model, ps, st, mesh) = (;
    gen  = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.gen),
    disc = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.disc))

function ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z)
    fake, st_gen = Lux.apply(model.gen, z, ps.gen, st.gen)
    d_real, _ = Lux.apply(model.disc, x, ps.disc, st.disc)
    d_fake, _ = Lux.apply(model.disc, fake, ps.disc, st.disc)

    l_d, g_d = ReactantNitro.backward(ps.disc, x, fake, st) do ps_d, xc, fc, stc
        d1, _ = Lux.apply(model.disc, xc, ps_d, stc.disc)
        d2, _ = Lux.apply(model.disc, fc, ps_d, stc.disc)
        mean(abs2, d1 .- 1.0f0) + mean(abs2, d2)
    end

    l_g, g_g = ReactantNitro.backward(ps.gen, ps.disc, z, st) do ps_g, ps_d, zc, stc
        f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
        d3, _ = Lux.apply(model.disc, f2, ps_d, stc.disc)
        mean(abs2, d3 .- 1.0f0)
    end

    s_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
    s_d, ps_d = ReactantNitro.step_optimizer(opt_state.disc, ps.disc, g_d)

    return (; loss = l_d + l_g, ps = (; gen = ps_g, disc = ps_d),
              st = (; gen = st_gen, disc = st.disc),
              opt_state = (; gen = s_g, disc = s_d),
              stats = (; l_d, l_g))
end

train!(Nitro(ToyGAN()))
```

The closure is called once per optimizer step, in train mode, on one transferred batch, and it is
traced and compiled into one XLA program: nothing recompiles after step 1. `opt`-keyed schedules
work here too, path-bound to each network's optimizer state, so the generator and the discriminator
can carry different learning-rate curves.

## Optimization

The optimizer is built from `Optimisers.jl` rules. Declare nothing and you get RAdam at 1e-3 over
one parameter group. Every setting is an accessor on the experiment, so the recipe ships with the
model:

```julia
ReactantNitro.param_group(::MyExp, ks) = :backbone in ks ? :backbone : :default
ReactantNitro.learning_rate(::MyExp) = 1f-3
ReactantNitro.learning_rate(::MyExp, ::Val{:backbone}) = 1f-4   # a ratio against the base
ReactantNitro.lambda(::MyExp, ::Val{:backbone}) = 1f-4          # decoupled weight decay
ReactantNitro.decay_anchor(::MyExp, ::Val{:backbone}) = :w0     # L2-SP, toward the init
ReactantNitro.gradient_clip_norm(::MyExp) = 1f0
```

Per-group learning rates are ratios against the base, so a schedule moves every group together and
the ratio holds at every step. `Decay` anchors to zero (L2), to `:w0` (L2-SP), or to an explicit
array; `no_decay` excludes per leaf, and the ADD form extends `default_no_decay` (every 1-D
parameter: biases and norm affines) rather than replacing it, so a port does not silently change
what gets regularized. Rules live on a verified allowlist (RAdam, Adam, AdamW, Momentum, Nesterov,
and chains of them), `accum` folds N micro-batches into one optimizer step with the schedule
horizon to match, and gradient clipping is a global norm over the fully accumulated gradient,
applied at the top of the optimizer program, never a chain member.

## Schedules

A schedule varies one quantity per optimizer step; everything else stays fixed. Every entry is a
factory of the horizon: the framework calls `f(total)` once at setup with the total number of
optimizer steps, then calls what it returned once per step, with ordinary host-side Julia control
flow. Two namespaces:

- `opt` keys bind optimizer rule fields by their real names (`eta`, `rho`, ...), with path-bound
  keys for per-group curves;
- `device` keys write a `Device` field that traced code reads as `e.field`.

The `device` namespace is how a teacher-forcing ratio, a label-smoothing coefficient, or a loss
weight varies mid-run without recompiling:

```julia
@experiment struct Seq2SeqExp
    teacher_forcing::Device{Float32} = 1f0
    max_epochs::Int = 20
end

ReactantNitro.schedules(::Seq2SeqExp) = (;
    device = (; teacher_forcing = total -> t -> max(0f0, 1f0 - 2f0 * t / total)),
    opt    = (; eta = total -> OneCycle(total, 1f-3)),   # ParameterSchedulers.jl, no dependency
)
```

A scheduled value is coerced to the field's type and asserted, so a schedule that returns `Float64`
cannot silently move the compile key; a constant and a schedule share one device slot, so switching
between them costs no recompile; and the fields a rule may not schedule are declared by
`nonschedulable`. [ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl) is
recommended but not a dependency; a schedule is any callable of the step.

## Export: training and serving, Julia native

The same `forward` that trained becomes the servable program. Export hooks declare the wire contract
(`export_inputs`, `export_outputs`); the conversion from what clients send (usually `UInt8` pixels)
to what the model trains on (normalized `Float32`) happens inside the traced graph through
`export_preprocess`, so what ships is `export_preprocess ∘ forward`; and the artifact is a
ReactantServer StableHLO bundle: `manifest.yaml`, `weights.safetensors`, one `model[.bN].mlir` per
compiled batch size, and a `model.jl` postprocess when the model ships one. Nothing leaves the
Julia/Reactant ecosystem.

```julia
using ReactantServerExport                  # the extension that provides the backend

n = Nitro(e; checkpoint = "runs/x/best.jld2", data = (;))   # no training in this process
export_model(n, ReactantServerBundle(); dir = "export_out", name = "mnist_v1")
```

`export_model` takes a `Nitro`, not a path, so a model's export code never loads a checkpoint
itself. Export is a single-device CPU trace; provenance (config, seed, preset, framework version)
lands in the bundle automatically.

## Kaimon in the loop

A KaimonGate extension registers `nitro_*` tools with the running gate, so an agent drives the
framework from the session where the model code is loaded: `nitro_train`, `nitro_validate`,
`nitro_evaluate`, `nitro_predict`, and `nitro_export`, plus `nitro_runs`, `nitro_status`,
`nitro_logger`, and `nitro_stop`. Runs execute on a background task, so no tool call ever blocks
until the work finishes; the agent polls `nitro_status` until the run completes. A completed train
run's `Nitro` is reused by the eval and export tools, and `resume = :auto` continues a run after a
session restart, as it does without the tools.

```julia
nitro_train(experiment="MyModels.MnistMLP", max_epochs=40,
            run_dir="runs/mnist_v1", overrides="width=128, smoothing=0.05")
nitro_status(run_id="a1b2c3d4")
nitro_export(run_id="a1b2c3d4", dir="export_out", name="mnist_v1")
```

### Reading the run's logger

Every run has a logger. `nitro_logger` reports the backend's type and the key identifying
parameters its `logger_info` exposes, read live off the run's handle:

```julia
nitro_logger(run_id="a1b2c3d4")
# run a1b2c3d4  logger=JSONLogger
#   path: runs/mnist_v1/metrics.jsonl
```

A run that names no logger gets the shipped JSON default, so that path is the first place to look
for a run's numbers: `metrics.jsonl` is one JSON object per line, params and the binding report at
setup, one line per optimizer step, one per epoch of validation, and a `finish` line with the
outcome. An agent that wants a curve reads the file, parsing each line as JSON; an agent that
wants a dashboard reads the same table from a hosted logger, whose `logger_info` carries its URL,
experiment key, and workspace instead of a path:

```julia
# with a hosted experiment tracker passed as the experiment's logger
nitro_logger(run_id="a1b2c3d4")
# run a1b2c3d4  logger=HostedTrackerLogger
#   experiment_key: abc123
#   url: https://tracker.example.com/workspace/project/abc123
```

The table is also on the handle itself, as `logger_info(nitro)`, so a REPL session or a phase
monitor can answer "which experiment is this run attached to, and where is it" without the tools.

The extension is a dev tool: nothing in `src/` knows it exists, it costs nothing until KaimonGate is
loaded, and the tools are a thin interface over entry points the framework already ships.

## Agent skills

The repository ships agent skills, one per concern, each written about the surface it teaches so
they version with the framework:

| Skill | Use it when |
| --- | --- |
| `reactantnitro-experiment` | writing or porting an experiment: the four hooks, the three markers, the batch contract |
| `reactantnitro-metrics` | adding or debugging a metric: `(sum, count)`, residency, `finalize_metrics` |
| `reactantnitro-optimizer` | parameter groups, decay and L2-SP anchors, per-leaf exclusion, schedules |
| `reactantnitro-manual` | manual training mode: `train_step`, `setup_optimizers`, `backward`, `step_optimizer` |
| `reactantnitro-recompiles` | verifying no recompile, or why a REPL edit did nothing |
| `reactantnitro-checkpoint-resume` | checkpoints, resuming a run, early stopping |
| `reactantnitro-device-boundary` | code that passes on CPU and fails on a GPU, residency contracts |
| `reactantnitro-visualization` | the `visualize` / `save_figure` / `render` surface |
| `reactantnitro-export` | shipping a trained model: the export hooks, the wire seam |
| `reactantnitro-kaimon` | driving runs from a Kaimon session: the `nitro_*` tools |

Start at `reactantnitro-experiment`, which indexes the rest.
See [`skills/README.md`](skills/README.md).

## Installation

Requires Julia 1.12. `[compat]` floors Reactant at `0.2.264`, the release carrying the memory-leak
fix, and Optimisers at `0.4.8`, the first release whose Reactant extension can trace `RAdam`.

The package is not in the General registry yet, so install it from the repository:

```julia
julia> using Pkg; Pkg.add(url = "https://github.com/EnzymeAD/ReactantNitro.jl")
```

To hack on it instead, `Pkg.develop(url = ...)` clones a working copy into `~/.julia/dev`.

The test suite runs on CPU, needs no GPU, and is exercised in CI on every push.

## Documentation

The full documentation lives at <https://enzymead.github.io/ReactantNitro.jl/>:

- [Tutorial](https://enzymead.github.io/ReactantNitro.jl/dev/tutorial/): MNIST from configuration to prediction, end to end.
- [Experiments](https://enzymead.github.io/ReactantNitro.jl/dev/experiments/): the hook contract, the three markers, and the Revise workflow.
- [Recompilation](https://enzymead.github.io/ReactantNitro.jl/dev/recompilation/): the compile cache, and when a change costs a compile.
- [Optimization](https://enzymead.github.io/ReactantNitro.jl/dev/optimization/): parameter groups, decay, and clipping.
- [Schedules](https://enzymead.github.io/ReactantNitro.jl/dev/schedules/): what varies with the step.
- [Manual training](https://enzymead.github.io/ReactantNitro.jl/dev/manual/): owning the step, GANs and beyond.
- [Kaimon](https://enzymead.github.io/ReactantNitro.jl/dev/kaimon/): the `nitro_*` tools.
- [API reference](https://enzymead.github.io/ReactantNitro.jl/dev/api/): the docstrings, collected automatically.
