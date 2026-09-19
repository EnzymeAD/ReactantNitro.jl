# Pitfalls: what the framework takes care of

A Reactant-first stack has snags that are easy to hit and hard to diagnose: a compile that
balloons for no visible reason, an edit that reuses a stale program, a run that dies out of memory
hours in. None of them raise; each presents as slow training or as a process that vanishes
overnight. Each section below says what the framework does about one and points at the page that
owns the mechanism.

## Your dataset is never traced

Fields are [`Host`](@ref) unless marked otherwise, and the trace sees a stripped view of the
experiment. A dataset reachable from traced code is walked element by element on every compile.

The three markers, and how to tell which one a field wants, are on the
[Experiments](experiments.md) page.

## Beyond code changes, only a `GraphConst` change recompiles

[`Host`](@ref) values never reach the compiled program, and a [`Device`](@ref) value changes without
a recompile unless its shape or element type changes. Sweeping a hyperparameter, scheduling a value,
changing the learning rate and reseeding are all free.

[Recompilation](recompilation.md) is the page for this, including
`ReactantNitro.cache_stats()`, the acceptance check that a change did not recompile.

## Revising a method invalidates the programs built on it

The compile key carries the resolved world age of every hook, and each cached program also records
the transitive closure of the methods it was traced against, so editing a helper `forward` calls
several levels down is caught too. A new [`Nitro`](@ref) recompiles against the current code; an
existing handle keeps the programs it was built with and says so.

The REPL and Revise workflow this exists for is on the [Experiments](experiments.md) page.

## Device transfers happen at known times

A scheduled [`Device`](@ref) value uploads once per step; a constant one uploads once, at the start
of training.

What may vary with the step, and where each schedule binds, is on the [Schedules](schedules.md)
page.

## Parameters are flattened into one buffer per parameter group

The gradient accumulator and the optimizer state cross the program boundary as `NTuple{G}`, so the
optimizer program emits G updates rather than one per parameter array. An unflattened tree makes
that program grow with the model's array count, and the compile with it.

Parameter groups, decay and clipping are on the [Optimization](optimization.md) page.

## Device buffers are freed per batch

The host GC runs on host pressure and cannot see the device filling, so a run that leaves batch
buffers to the finalizer can die out of memory while the host looks idle. The loop frees each
batch's buffers as soon as the loss readback confirms the step has finished, and the prefetch
pipeline frees what it staged when a phase ends. Freeing nulls the buffer pointer, so a
use-after-free raises on readback instead of returning stale memory.

The host/device boundary, and what a model author may and may not do at it, is the subject of the
`reactantnitro-device-boundary` skill.
