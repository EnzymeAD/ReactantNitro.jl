<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/src/assets/logo-dark.svg">
    <img src="docs/src/assets/logo.svg" alt="ReactantNitro.jl" width="140">
  </picture>
</p>

<h1 align="center">ReactantNitro.jl</h1>

<p align="center"><em>Reactant-first training for Lux models: declare the experiment, compile once, train without boilerplate.</em></p>

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
when XLA compiles.

## What it takes care of

Reactant and Enzyme are fast. These are the common ways that speed is lost, and what the framework
does about each.

**Your dataset is never traced.** Fields are `Host` unless marked otherwise, and the trace sees a
stripped view of the experiment. A dataset reachable from traced code is walked element by element
on every compile.

**Only a `GraphConst` change recompiles.** `Host` values never reach the compiled program, and a
`Device` value changes without a recompile unless its shape or element type changes. Sweeping a
hyperparameter, scheduling a value, changing the learning rate and reseeding are all free.

**Device transfers happen at known times.** A scheduled `Device` value uploads once per step; a
constant one uploads once, at the start of training.

**Parameters are flattened into one buffer per parameter group.** The gradient accumulator and the
optimizer state cross the program boundary as `NTuple{G}`, so the optimizer program emits G updates
rather than one per parameter array. An unflattened tree makes that program grow with the model's
array count, and the compile with it.

**Device buffers are freed per batch.** Host GC pressure does not track device memory, so the host
can stay comfortable while the device fills and the run dies out of memory. Each batch's buffers are
released explicitly once the step that used them has read back.

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

Programs are keyed and stored once per process, not once per `Nitro`, and
`ReactantNitro.cache_stats()` is the acceptance check that a change did not recompile.

[Recompilation](https://enzymead.github.io/ReactantNitro.jl/dev/recompilation/)
has what is in the key, the guard that catches a redefinition below the hooks, and the two holes.

## Quick start

### Installation

The package is not in the General registry yet, so install it from the repository:

```julia
julia> using Pkg; Pkg.add(url = "https://github.com/EnzymeAD/ReactantNitro.jl")
```

### MNIST

Four hooks are required. Everything else has a default: the optimizer (RAdam at 1e-3), one
parameter group, no decay, no schedule, prefetching, validation, checkpointing, and a logging
contract whose shipped default writes JSON Lines.

Real MNIST, so the numbers at the end mean something. Neither `MLDatasets` nor `MLUtils` is a
dependency of this package; `] add MLDatasets MLUtils` and the data downloads on first use.
Batching and shuffling are `MLUtils`' job, deliberately: the framework ships neither.

```julia
using ReactantNitro, Lux, Random
using MLDatasets: MNIST
using MLUtils: DataLoader

# Explicit CPU, so the quick start runs anywhere. It has to come BEFORE the first `Nitro`: that
# is where the XLA client initializes, and the backend is fixed for the process from then on.
setup_devices!(backend = "cpu")

@experiment struct MnistMLP
    width::GraphConst{Int} = 128
    smoothing::Device{Float32} = 0.05f0
    max_epochs::Int = 5
end

ReactantNitro.build_model(e::MnistMLP, rng) = begin
    model = Chain(Dense(784 => e.width, relu), Dense(e.width => 10))
    (model, Lux.setup(rng, model)...)
end

function ReactantNitro.build_data(::MnistMLP, dist)
    function fields(d)
        x = reshape(d.features, 28 * 28, :)              # Float32, already in [0, 1]
        y = zeros(Float32, 10, length(d.targets))
        for (i, t) in pairs(d.targets)
            y[t + 1, i] = 1f0                            # targets are 0..9
        end
        return (; img = x, label = y)                    # batch dimension LAST, always
    end
    # `shuffle = true` reshuffles every epoch, and the framework drives this loader by index rather
    # than iterating it, so one producer per thread fills the pipeline.
    return (;
        train = DataLoader(fields(MNIST(split = :train)); batchsize = 32, shuffle = true, partial = false),
        val = DataLoader(fields(MNIST(split = :test)); batchsize = 32),
    )
end

ReactantNitro.forward(::MnistMLP, model, ps, st; img) = Lux.apply(model, img, ps, st)

function ReactantNitro.loss(e::MnistMLP, logits; label)
    smoothed = (1f0 - e.smoothing) .* label .+ e.smoothing / 10f0
    return -sum(smoothed .* logsoftmax(logits; dims = 1)) / size(label, 2)
end

function ReactantNitro.metrics(::MnistMLP, logits; label)
    return (; acc = (sum(argmax(logits; dims = 1) .== argmax(label; dims = 1)), size(label, 2)))
end

n = Nitro(MnistMLP(); checkpointer = TopKCheckpointer(; metric = :acc, mode = :max))
train!(n)
```

One epoch on CPU reaches about 94% on the test split, reported as `acc 0.9421073717948718`: 9406
correct out of 9984, not an average of 312 per-batch fractions.

`examples/mnist_tutorial.jl` is the same task with everything turned on, as one runnable file; see the [Tutorial](https://enzymead.github.io/ReactantNitro.jl/dev/tutorial/)

## Built for the REPL and Revise

Everything that decides which compiled program a `Nitro` runs is resolved at construction, so an
edit never changes a running handle and a handle never recompiles underneath you. Edit a hook,
build a new `Nitro`, and the module-level cache recompiles only what the edit changed; every entry
point prints what it fixed and names any hook redefined since, so a stale handle is never silent.
The one thing that changes on a live handle is a `Device` value, through `set_device!`, which
provably recompiles nothing and is what makes an inference sweep one compile for the whole sweep.

Long runs leave the interactive thread. With a worker pool (`-t N,1`) the entry points run the loop
on a worker and park your call, so the REPL and your logger tasks keep running; Ctrl+C then requests
a graceful stop rather than tearing the run down.

[Experiments](https://enzymead.github.io/ReactantNitro.jl/dev/experiments/) has the Revise loop.
[Recompilation](https://enzymead.github.io/ReactantNitro.jl/dev/recompilation/)
has the rules for what a change costs.

## The automatic loop and the lifecycle

`train!` sequences the forwards, the backwards, the optimizer steps, and the metrics, and wraps the
run in a lifecycle with monitorable phases (`Starting`, `Compiling`, `Stepping`, `Checkpointing`,
`Terminal`, `Done`, `Failed`), `request_stop!` for graceful interruption, and a progress counter.
Every split is wrapped in a `PrefetchIterator` automatically, so the next batch's host-to-device
transfer overlaps the current step, and each stream lives only for the phase that uses it.

Metrics are `(sum, count)` pairs with a per-hook residency choice, `:host` or `:device`, detailed
in the next section; `finalize_metrics` reduces host-side when a macro-averaged recall is not the
mean of per-batch recalls. Checkpointing defaults to `TopKCheckpointer` and early stopping to
`EarlyStopping` when you opt in (the default is off); both are run accessors, and `resume = :auto`,
also opt in, continues a run from the latest checkpoint behind a config-compatibility check and an
anchor checksum. The logging contract is a set of verbs
(`log_metrics!`, `log_params!`, `log_other!`, `finish!`, ...) on your own type: your tracker's run
handle goes in directly, and `nothing` is the public "no logging" value.

## Metrics on the host or the device

A metric is a `(sum, count)` pair, so it carries its own denominator: a per-image accuracy, a
per-token loss, and a confusion matrix coexist in one call, and the framework never supplies a
sample count of its own, because there is no single right one. `count === nothing` means accumulate
by summation and never divide. Metrics never see padding, and `finalize_metrics` reduces host-side
for anything that is not a mean of per-batch values.

Where a metric runs is a per-hook choice. `:device` keeps the model's raw outputs off the host wire
and costs a recompile when you edit the hook; `:host` transfers the outputs and lets evaluation code
use anything Julia can do, recompiling nothing. `train_metrics` is traced by default and `metrics`
is not, because their cadences differ by thousands of steps.

[Metrics](https://enzymead.github.io/ReactantNitro.jl/dev/metrics/)
has the residency table and a worked pair of hooks.

## Manual mode

Manual mode exists for algorithms the automatic loop cannot express. Define `train_step` for your
experiment and you own the whole optimizer step; the framework keeps everything outside it: the
loop, prefetch, validation, checkpointing, early stopping, and the phase lifecycle. The closure is
called once per optimizer step, in train mode, on one transferred batch, and it is traced and
compiled into one XLA program, so nothing recompiles after step 1.

The standard example is a GAN: one generator and one discriminator in a single parameter tree, each
with its own optimizer, two losses against the same batch, and no explicit stop-gradient op needed.
`opt`-keyed schedules work there too, path-bound into your own optimizer state, so the two networks
can carry different learning-rate curves.

[Manual training](https://enzymead.github.io/ReactantNitro.jl/dev/manual/)
has the worked GAN and the hook contracts.

## Optimization

The optimizer is built from `Optimisers.jl` rules. Declare nothing and you get RAdam at 1e-3 over
one parameter group. Every setting is an accessor on the experiment, so the recipe ships with the
model: `param_group` splits the parameters, `learning_rate` gives per-group rates as ratios against
the base so a schedule moves every group together, `lambda` and `decay_anchor` give decoupled decay
toward zero, toward the init (L2-SP), or toward an array you supply, `no_decay` excludes per leaf,
and `gradient_clip_norm` is a global norm over the fully accumulated gradient.

[Optimization](https://enzymead.github.io/ReactantNitro.jl/dev/optimization/)
has the three levels of optimizer and a complete recipe.

## Schedules

A schedule varies one quantity per optimizer step; everything else stays fixed. Every entry is a
factory of the horizon: the framework calls `f(total)` once at setup with the total number of
optimizer steps, then calls what it returned once per step, with ordinary host-side Julia control
flow. `opt` keys bind optimizer rule fields by their real names (`eta`, `rho`, ...), with path-bound
keys for per-group curves; `device` keys write a `Device` field that traced code reads as `e.field`,
which is how a teacher-forcing ratio, a label-smoothing coefficient, or a loss weight varies mid-run
without recompiling.
[ParameterSchedulers.jl](https://github.com/FluxML/ParameterSchedulers.jl) is recommended but not a
dependency; a schedule is any callable of the step.

[Schedules](https://enzymead.github.io/ReactantNitro.jl/dev/schedules/)
has both namespaces, the coercion rules, and `nonschedulable`.

## Export: training and serving, Julia native

The same `forward` that trained becomes the servable program. Export hooks declare the wire
contract; the conversion from what clients send (usually `UInt8` pixels) to what the model trains on
happens inside the traced graph, so what ships is `export_preprocess ∘ forward`; and the artifact is
a ReactantServer StableHLO bundle. `export_model` takes a `Nitro` rather than a path, so a model's
export code never loads a checkpoint itself, the trace is single-device CPU, and provenance lands in
the bundle automatically. Nothing leaves the Julia/Reactant ecosystem.

[Export](https://enzymead.github.io/ReactantNitro.jl/dev/export/)
has the hooks, the bundle layout, and the caveats.

## Logging backends

The logging contract is ten verbs on your own type, duck-typed: no supertype, no registration, and
`nothing` is the public "no logging" value. A missing method is a loud `MethodError` rather than a
silent no-op, so a backend opts into silence per verb. A run that names no logger gets the shipped
`JSONLogger` and leaves a machine-readable `metrics.jsonl` in its run directory.

There are three legal ways to get a backend and no one of them blocks on another: your own logger
defines the verbs in your own code with no extension, a common public logger gets an extension here
(TensorBoard ships today), and a backend's own package may define them itself, taking ReactantNitro
as a weak dependency so nobody installing it pays for Reactant and Enzyme.

[Logging](https://enzymead.github.io/ReactantNitro.jl/dev/logging/)
has the contract, the JSON default, and the TensorBoard extension.

## Kaimon in the loop

A KaimonGate extension registers `nitro_*` tools with the running gate, so an agent drives the
framework from the session where the model code is loaded: `nitro_train`, `nitro_validate`,
`nitro_evaluate`, `nitro_predict`, and `nitro_export`, plus `nitro_runs`, `nitro_status`,
`nitro_logger`, and `nitro_stop`. Runs execute on a background task, so no tool call ever blocks
until the work finishes; the agent polls `nitro_status` until the run completes.

The extension is a dev tool: nothing in `src/` knows it exists, it costs nothing until KaimonGate is
loaded, and the tools are a thin interface over entry points the framework already ships.

[Kaimon](https://enzymead.github.io/ReactantNitro.jl/dev/kaimon/)
has the tools, their arguments, and what they deliberately do not do.

## Agent skills

The repository ships agent skills under `skills/`, one per concern, each written about the surface
it teaches so they version with the framework: experiments, metrics, the optimizer, manual mode,
recompiles, checkpoint and resume, the device boundary, visualization, export, and Kaimon. Start at
`reactantnitro-experiment`, which indexes the rest. See [`skills/README.md`](skills/README.md).

## Documentation

The full documentation lives at <https://enzymead.github.io/ReactantNitro.jl/>:

- [Tutorial](https://enzymead.github.io/ReactantNitro.jl/dev/tutorial/): MNIST from configuration to prediction, end to end.
- [Experiments](https://enzymead.github.io/ReactantNitro.jl/dev/experiments/): the hook contract, the three markers, and the Revise workflow.
- [Recompilation](https://enzymead.github.io/ReactantNitro.jl/dev/recompilation/): the compile cache, and when a change costs a compile.
- [Optimization](https://enzymead.github.io/ReactantNitro.jl/dev/optimization/): parameter groups, decay, and clipping.
- [Schedules](https://enzymead.github.io/ReactantNitro.jl/dev/schedules/): what varies with the step.
- [Metrics](https://enzymead.github.io/ReactantNitro.jl/dev/metrics/): `(sum, count)`, residency, and `finalize_metrics`.
- [Logging](https://enzymead.github.io/ReactantNitro.jl/dev/logging/): the ten verbs, the JSON default, and the TensorBoard extension.
- [Export](https://enzymead.github.io/ReactantNitro.jl/dev/export/): the wire contract and the ReactantServer bundle.
- [Manual training](https://enzymead.github.io/ReactantNitro.jl/dev/manual/): owning the step, GANs and beyond.
- [Kaimon](https://enzymead.github.io/ReactantNitro.jl/dev/kaimon/): the `nitro_*` tools.
- [API reference](https://enzymead.github.io/ReactantNitro.jl/dev/api/): the docstrings, collected automatically.
