# Pitfalls: what the framework takes care of

Reactant and Enzyme are fast, but a Reactant-first stack has snags that are easy to hit and hard to
diagnose: a compile that balloons for no visible reason, an edit that silently reuses a stale
program, a run that dies out of memory hours in. None of them raise. Each one presents as something
else, usually as "training is mysteriously slow" or as a process that vanishes overnight, which is
why they are collected here rather than left to be met one at a time.

These are the ones the framework handles for you. Each section says what the framework does and
points at the page that owns the mechanism.

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

Host GC pressure does not track device memory, so the host can stay comfortable while the device
fills and the run dies out of memory. Each batch's buffers are released explicitly once the step
that used them has read back.

The host/device boundary, and what a model author may and may not do at it, is the subject of the
`reactantnitro-device-boundary` skill.
