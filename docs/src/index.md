```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: ReactantNitro.jl
  text: A training framework for Reactant-first machine learning
  tagline: Declare an experiment. The framework owns the compiles, the device transfers, the optimizer, the schedules, and the loop.
  actions:
    - theme: brand
      text: Tutorial
      link: /tutorial/
    - theme: alt
      text: API Reference
      link: /api/
    - theme: alt
      text: View on GitHub
      link: https://github.com/EnzymeAD/ReactantNitro.jl
  image:
    src: /logo.svg
    alt: ReactantNitro.jl
    width: 320
    height: 320

features:
  - icon: ⚡
    title: Reactant-first
    details: XLA is the accelerator under the hood. Every compiled program is Reactant, and the framework exists to make Reactant workflows fast, correct, and boring.
    link: /experiments/
  - icon: 🚀
    title: Compile once, train forever
    details: A module-level compile cache hashes the GraphConst fields on your experiment, so only a change that really moves the graph recompiles. Device values never do.
    link: /recompilation/
  - icon: 🔁
    title: REPL-native
    details: "Revise, rebuild, repeat: the workflow is designed to flow naturally in a REPL session, perfect for testing new ideas on a whim."
    link: /experiments/
  - icon: 🧩
    title: Lux.jl models
    details: Your model is an ordinary Lux model. ReactantNitro builds on Lux the way Lightning builds on PyTorch.
    link: /tutorial/
  - icon: 🎛️
    title: Parameters and optimization
    details: Parameter groups, decoupled weight decay with per-leaf exclusion, L2-SP anchors, and gradient accumulation.
    link: /optimization/
  - icon: 🤖
    title: Kaimon in the loop
    details: "The KaimonGate extension registers tools with your session: an agent drives training, validation, evaluation, prediction, and export, with runs executing in the background so no tool call ever hits a deadline."
    link: /kaimon/
---
```

## What it is

ReactantNitro.jl is a training framework for Julia, built on [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and [Lux.jl](https://github.com/LuxDL/Lux.jl). You declare an experiment as a struct plus a handful of hooks; the framework owns the compiled programs, the device transfers, the optimizer, the schedules, and the run's lifecycle.

The central object is a [`Nitro`](@ref). `Nitro(e)` runs the setup sequence and nothing else, so evaluation and serving never depend on a [`train!`](@ref) having happened in the process; a run is what happens when you `train!` one.

Every field on the experiment carries one of three markers, and the marker is the point. A [`GraphConst`](@ref) field bakes into the compiled graph as a constant; a [`Device`](@ref) field becomes a traced input; an unmarked field is [`Host`](@ref), driver-only and invisible to the tracer.

## Why ReactantNitro?

Lux.jl already has a training loop, so why a framework on top of it? The plain answer is that this framework is **Reactant-first**, and Lux currently does not support certain things a Reactant-first training stack needs, such as gradient accumulation and the sort of phase system that frameworks like PyTorch Lightning have. ReactantNitro builds on top of Lux.jl the way Lightning builds on PyTorch, with Reactant/XLA as the accelerator under the hood.

## Philosophy

This is a framework for the typical deep learning workflow: train, validate, test, predict, with dataloaders, and with Reactant compiling to an accelerator so the loop is fast. Within that context the framework tries not to make decisions that reduce what a user can express; it is not trying to be general beyond it. What it always provides is the workflow itself: the training loop, checkpointing, and a Lux.jl based model, usually a neural network.

The direct inspiration is frameworks like PyTorch Lightning. The focus, above everything else, is being the best machine learning framework for Reactant specifically.

## The REPL and the Revise workflow

The package is meant to be driven from a REPL with Revise loaded. The rules are short.

- A `Nitro` is a fixed point. Revise a hook in the REPL and build a new handle to pick it up; the rebuild is cheap because the compile cache is module-level. The loop is on the [Experiments](experiments.md) page.
- A [`Device`](@ref) value is the one thing that changes on a live handle, via [`set_device!`](@ref), provably without recompiling.
- The fixed-config report names the hooks you have redefined since the handle was built, so a stale handle is never silent.
- `ReactantNitro.cache_stats()` is the acceptance check that a change did not recompile; it is one of the tools the [Recompilation](recompilation.md) page builds on.

## Start here

- The [Tutorial](tutorial.md): an MNIST run from configuration to prediction, end to end.
- [Experiments](experiments.md): the hook contract, the three markers, and the Revise workflow.
- [Recompilation](recompilation.md): the compile cache and when a change costs a compile.
- [Optimization](optimization.md): parameter groups, decay, and clipping.
- [Schedules](schedules.md): what varies with the step.
- [Kaimon](kaimon.md): driving runs from a Kaimon-hosted session, the `nitro_*` tools.
- The [API](api.md): the docstrings, collected automatically.
