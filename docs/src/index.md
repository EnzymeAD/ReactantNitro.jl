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
    details: Programs are keyed on the GraphConst fields and the hook code, nothing else. Sweeps, schedules, seeds and longer runs reuse the programs. A width change or a code edit recompiles exactly what moved.
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
    details: The KaimonGate extension registers nitro_* tools with your session. An agent starts training, validation, evaluation, prediction and export as background runs, so no tool call blocks on a deadline.
    link: /kaimon/
---
```

## What it is

ReactantNitro.jl is a training framework for Julia, built on [Reactant.jl](https://github.com/EnzymeAD/Reactant.jl) and [Lux.jl](https://github.com/LuxDL/Lux.jl). An experiment is a struct plus four hooks: [`build_model`](@ref), [`build_data`](@ref), [`forward`](@ref), and [`loss`](@ref). Everything else has a default. The framework owns the compiled programs, the device transfers, the optimizer, the schedules, checkpointing, and the run's lifecycle.

[`Nitro`](@ref)`(e)` runs setup and nothing else; [`train!`](@ref) runs the loop. Evaluation, prediction and export work on a `Nitro` that has never trained.

Every field on the experiment is one of three kinds. A [`GraphConst`](@ref) field bakes into the compiled graph as a constant. A [`Device`](@ref) field becomes a traced input. An unmarked field is [`Host`](@ref): the driver reads it and the tracer never sees it.

## Why ReactantNitro?

Lux has a training loop. A Reactant-first stack needs four things a training loop does not provide.

**Precision over when to compile.** A gradient program has been measured at eight minutes to compile, and an unintended recompile presents as slow training, never as an error. Every program is keyed on the `GraphConst` fields and the method world of each hook, and on nothing else. Sweeping a hyperparameter, scheduling a value, changing the seed or training longer compiles nothing. Changing a layer width or editing `forward` compiles the programs that changed and reuses the rest. `ReactantNitro.cache_stats()` is the check. See [Recompilation](recompilation.md).

**Control over what gets traced.** Reactant and Enzyme walk every argument of a compiled program, and the experiment is one of them. A dataset held on the experiment is walked element by element on every compile, on one thread, and the symptom is compile time that grows with the data. The trace receives a stripped view of the experiment in which every `Host` field is a sentinel. `Device` fields cross as single buffers. Only `GraphConst` fields bake. You pick, per field. See [Experiments](experiments.md).

**Device memory freed per batch.** The host GC runs on host pressure and cannot see the accelerator filling, so a run that leaves batch buffers to the finalizer can die out of memory hours in while the host looks idle. The loop frees each batch's device buffers as soon as the loss readback confirms the step has finished, and the prefetch pipeline frees what it staged when a phase ends. See [Pitfalls](pitfalls.md).

**Binding instead of wiring.** A hook declares the batch fields it wants as keywords, and the framework routes exactly those from whatever the loader yields. A schedule key names an optimizer rule field such as `eta` or a `Device` field on the experiment, and resolves to one of them. Ten run keywords default to an accessor of the same name, and most accessors fall back to a field of the same name. An `MLUtils.DataLoader` gets multi-threaded prefetching with nothing declared. The binding report prints where every value came from. See the [Tutorial](tutorial.md#Binding).

On top of those it is the loop Lux does not ship: gradient accumulation, phases, checkpoint and resume, early stopping, schedules, logging, and export to a servable bundle.

## Working from the REPL

The package is meant to be driven from a REPL with Revise loaded. A `Nitro` is a fixed point: revise a hook, build a new handle, and the module-level compile cache recompiles only the programs the edit touched. A stale handle says so on every entry point. The one value that changes on a live handle is a `Device` field, through [`set_device!`](@ref), which never recompiles. `Nitro(e; weights = n)` starts a new run from a trained handle's weights, and [`history`](@ref)`(n)` is the finished run as a table that indexes by epoch and metric.

## Start here

- [Tutorial](tutorial.md): MNIST from configuration to prediction, with each feature explained where it appears.
- [Pitfalls](pitfalls.md): what a Reactant-first stack gets wrong silently, and what the framework does about each.
- [Experiments](experiments.md): the hook contract, the three markers, and the Revise workflow.
- [Recompilation](recompilation.md): the compile cache, what is in the key, and the acceptance check.
- [Optimization](optimization.md): parameter groups, decay, and clipping.
- [Schedules](schedules.md): what varies with the step, and where a schedule key binds.
- [Metrics](metrics.md): `(sum, count)` pairs, host or device residency, and `finalize_metrics`.
- [Manual training](manual.md): owning the optimizer step, for GANs and other multi-optimizer algorithms.
- [Logging](logging.md): the ten verbs, the JSON default, and the TensorBoard extension.
- [Export](export.md): the wire contract and the ReactantServer bundle.
- [Kaimon](kaimon.md): the `nitro_*` tools for driving runs from an agent session.
- [API](api.md): every docstring, collected automatically.
