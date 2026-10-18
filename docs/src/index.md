```@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: ReactantNitro.jl
  text: A training framework for Reactant-first machine learning
  tagline: Declare an experiment as a struct plus hooks. The framework decides when to compile, what the tracer sees, and when device memory is freed.
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
    title: Reactant-first, Lux models
    details: Your model is an ordinary Lux model. Every compiled program is Reactant, and XLA runs it. The framework builds on Lux the way Lightning builds on PyTorch.
    link: /experiments/
  - icon: 🚀
    title: Compile only when the graph changes
    details: Programs are keyed on argument shapes, the GraphConst fields and the hook code. Sweeps of Device values, schedules, seeds and longer runs reuse the programs. A width change or a code edit recompiles only the programs it affects.
    link: /recompilation/
  - icon: 🧹
    title: Device memory freed per batch
    details: The host GC cannot see device pressure. Each batch's buffers are released as soon as the step that used them has read back, so device residency is bounded by batches in flight.
    link: /pitfalls/
  - icon: 🔗
    title: Bound by name, not wired by hand
    details: Batch fields route to hooks by keyword. Schedule keys bind to optimizer rule fields or Device fields. Run keywords bind to accessors. The binding report shows where every value landed.
    link: /tutorial/
  - icon: 🎛️
    title: Optimization
    details: Parameter groups with ratio learning rates, decoupled decay with per-leaf exclusion, L2-SP anchors, global-norm clipping, and gradient accumulation.
    link: /optimization/
  - icon: 🤖
    title: Kaimon in the loop
    details: The KaimonGate extension registers nitro_* tools with your session. An agent starts training, validation, evaluation, prediction and export as background runs, so no tool call waits for a run to finish.
    link: /kaimon/
---
```

## What it is

ReactantNitro.jl is a training framework for Julia, built on [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and [Lux.jl](https://github.com/LuxDL/Lux.jl). An experiment is a struct plus four hooks: [`build_model`](@ref), [`build_data`](@ref), [`forward`](@ref), and [`loss`](@ref). Everything else has a default. The framework owns the compiled programs, the device transfers, the optimizer, the schedules, checkpointing, and the run's lifecycle.

[`Nitro`](@ref)`(e)` runs setup and nothing else; [`train!`](@ref) runs the loop. Evaluation, prediction and export work on a `Nitro` that has never trained.

Every field on the experiment is one of three kinds. A [`GraphConst`](@ref) field bakes into the compiled graph as a constant. A [`Device`](@ref) field becomes a traced input. An unmarked field is [`Host`](@ref): the driver reads it and the tracer never sees it.

## Why ReactantNitro?

Lux provides the training step. A Reactant-first stack needs four more things.

**Control over when to compile.** A gradient program can take minutes to compile, and an unintended recompile shows up as slow training, not as an error. Besides argument types and shapes, every program is keyed on the `GraphConst` fields and on the hook methods and the methods they call. Sweeping or scheduling a `Device` value, changing the seed or training longer compiles nothing. Changing a layer width or editing `forward` recompiles the programs that changed and reuses the rest. `ReactantNitro.cache_stats()` shows the hit and miss counts. See [Recompilation](recompilation.md).

**Control over what gets traced.** Reactant and Enzyme walk every argument of a compiled program, and the experiment is one of them. A dataset held on the experiment is walked element by element on every compile, on one thread, so compile time grows with the data. The trace receives a stripped view of the experiment in which every `Host` field is a sentinel, `Device` fields cross as single buffers, and only `GraphConst` fields bake. The marker is chosen per field. See [Experiments](experiments.md).

**Device memory freed per batch.** The host GC runs on host memory pressure and cannot see device memory, so a run that leaves batch buffers to the finalizer can run out of device memory hours in. The loop frees each batch's device buffers as soon as the loss readback confirms the step has finished, and the prefetch pipeline frees what it staged when a phase ends. See [Pitfalls](pitfalls.md).

**Binding instead of wiring.** A hook declares the batch fields it wants as keywords, and the framework routes exactly those from whatever the loader yields. A schedule key names an optimizer rule field such as `eta` or a `Device` field on the experiment, and resolves to one of them. Fourteen run keywords default to an accessor of the same name, and every accessor except `schedules` falls back to a field of the same name. An `MLUtils.DataLoader` gets multi-threaded prefetching with nothing declared. The binding report prints where every value came from. See the [Binding cheat sheet](binding.md).

It also supplies the loop around the step: gradient accumulation, phases, checkpoint and resume, early stopping, schedules, logging, and export to a servable bundle.

## Working from the REPL

The package is meant to be driven from a REPL with Revise loaded. A `Nitro` freezes its configuration and hook code at construction: revise a hook, build a new handle, and the module-level compile cache recompiles only the programs the edit touched. A stale handle says so when you train, validate or evaluate with it. The one value that changes on a live handle is a `Device` field, through [`set_device!`](@ref), which never recompiles. `Nitro(e; weights = n)` starts a new run from a trained handle's weights. [`history`](@ref)`(n)` returns the handle's per-epoch metrics as a table indexed by epoch and metric; `plot(n)` draws it when a Makie backend is loaded.

## Start here

**Getting started**

- [Tutorial](tutorial.md): MNIST from configuration to prediction, with each feature explained where it appears.
- [Binding cheat sheet](binding.md): how everything connects by name, on one page.
- [Pitfalls](pitfalls.md): failures of a Reactant-first stack that raise no error, and what the framework does about each.

**Writing an experiment**

- [Experiments](experiments.md): the hook contract, the three markers, and the Revise workflow.
- [Metrics](metrics.md): `value => mode` pairs, host or device residency, and `finalize_metrics`.
- [Optimization](optimization.md): parameter groups, decay, and clipping.
- [Schedules](schedules.md): what varies with the step, and where a schedule key binds.
- [Manual training](manual.md): owning the optimizer step, for GANs and other multi-optimizer algorithms.

**Running**

- [Devices and meshes](devices.md): the backend, `n_devs`, and what a data-parallel mesh computes.
- [Logging](logging.md): the ten verbs, the JSON default, and the TensorBoard extension.
- [Kaimon](kaimon.md): the `nitro_*` tools for driving runs from an agent session.

**Compilation**

- [Recompilation](recompilation.md): the compile cache, what is in the key, and the acceptance check.
- [Export](export.md): the wire contract and the ReactantServer bundle.

**Reference**

- [API](api.md): every docstring, collected automatically.
