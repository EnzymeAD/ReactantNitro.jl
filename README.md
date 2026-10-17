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
  <a href="https://enzymead.github.io/ReactantNitro.jl/dev/"><img src="https://img.shields.io/badge/docs-dev-blue.svg" alt="Documentation"></a>
  <a href="https://codecov.io/gh/EnzymeAD/ReactantNitro.jl"><img src="https://codecov.io/gh/EnzymeAD/ReactantNitro.jl/branch/main/graph/badge.svg" alt="Coverage"></a>
  <img src="https://img.shields.io/badge/Julia-1.12-9558b2" alt="Julia 1.12">
  <a href="https://github.com/fredrikekre/Runic.jl"><img src="https://img.shields.io/badge/code_style-%E1%9A%B1%E1%9A%A2%E1%9A%BE%E1%9B%81%E1%9A%B2-black" alt="code style: runic"></a>
</p>

## Overview

ReactantNitro is a training framework for Julia, built on [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl)
and [Lux.jl](https://github.com/LuxDL/Lux.jl). The model is an ordinary Lux model, every compiled
program is Reactant, and XLA runs it. You write the experiment as a struct plus a handful of hooks;
the framework supplies the compiled programs, the device transfers, the optimizer, the schedules,
checkpointing, and the run's lifecycle. Training and serving both stay in Julia, from the first
`train!` to the exported [bundle](https://enzymead.github.io/ReactantNitro.jl/dev/export/).

The design follows PyTorch Lightning, adapted to Reactant. Beyond a training loop, a Reactant-first
stack needs control over when XLA compiles and what the tracer walks, device memory freed after each
batch instead of when the host GC runs, and binding by name between loaders, hooks, schedules and
optimizer rules. On top of those it supplies what Lightning does: gradient accumulation, phases,
schedules, checkpoint and resume, logging, and export, plus sharpness-aware minimization (SAM).

A Reactant-first stack has [pitfalls](https://enzymead.github.io/ReactantNitro.jl/dev/pitfalls/) that are easy to
hit and hard to diagnose: a compile that takes far longer than it should, an edit that reuses a stale
program, a run that runs out of device memory hours in. None of them raises an error on its own; the
framework guards against each one.

## The three field markers

Every field on an experiment carries one of three markers. The marker decides what the compiled
program sees and whether changing the value recompiles:

| Marker | Reaches traced code as | In the compile key? | Changing the value |
| --- | --- | --- | --- |
| `GraphConst{T}` | a baked literal | yes | recompiles, correctly: a different value is a different program |
| `Device{T}` | a device-resident traced input | no, by construction | never recompiles: sweep it, schedule it, rewrite it live |
| unmarked, i.e. `Host{T}` | not at all | no | never recompiles: driver-only, invisible to the tracer |

Unmarked means `Host` because most fields are.

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

Programs are keyed and stored once per process, not once per `Nitro`.

[Recompilation](https://enzymead.github.io/ReactantNitro.jl/dev/recompilation/) explains when a
program is reused and what the compile key cannot see.

## Quick start

### Installation

```julia
using Pkg
Pkg.add("ReactantNitro")
```

### MNIST

Four hooks are required. Everything else has a default: the optimizer (RAdam at 1e-3), one
parameter group, no decay, no schedule, prefetching, validation, checkpointing, and a JSON Lines
logger that writes the loss, metrics and schedule values to `metrics.jsonl` in the run directory.

```julia
using ReactantNitro, Lux, Random
using MLDatasets: MNIST
using MLUtils: DataLoader

# Explicit CPU, so the quick start runs anywhere. It must come before the first `Nitro`, where the
# XLA client initializes; the backend is fixed for the process from then on.
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
    d = MNIST(split = :train)
    x = reshape(d.features, 28 * 28, :)                  # Float32, already in [0, 1]
    y = zeros(Float32, 10, length(d.targets))
    for (i, t) in pairs(d.targets)
        y[t + 1, i] = 1f0                                # targets are 0..9
    end
    part(idx) = (; img = x[:, idx], label = y[:, idx])   # batch dimension LAST, always
    return (;
        train = DataLoader(part(1:55_000); batchsize = 100, shuffle = true, partial = false),
        val = DataLoader(part(55_001:60_000); batchsize = 100),
    )
end

ReactantNitro.forward(::MnistMLP, model, ps, st; img) = Lux.apply(model, img, ps, st)

function ReactantNitro.loss(e::MnistMLP, logits; label)
    smoothed = (1f0 - e.smoothing) .* label .+ e.smoothing / 10f0
    return -sum(smoothed .* logsoftmax(logits; dims = 1)) / size(label, 2)
end

function ReactantNitro.metrics(::MnistMLP, logits; label)
    return (; acc = sum(argmax(logits; dims = 1) .== argmax(label; dims = 1)) => size(label, 2))
end

n = Nitro(MnistMLP(); checkpointer = Checkpointer(; metric = :acc, mode = :max))
train!(n)
```

See the [tutorial](https://enzymead.github.io/ReactantNitro.jl/dev/tutorial/) for a longer walkthrough that
covers more of the framework and how it works.
