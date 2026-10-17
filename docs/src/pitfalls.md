# Pitfalls: what the framework takes care of

A Reactant-first stack has failures that raise no error: a compile that grows for no visible
reason, an edit that reuses a stale program, device memory exhausted hours into a run. They show
up as slow training or as a process that dies. Each section below says what the framework does
about one and links the page that covers it.

## Host fields are never traced

A dataset reachable from traced code is walked element by element on every compile. Fields are
[`Host`](@ref) unless marked otherwise, and the trace sees a view of the experiment in which every
`Host` field is replaced by a sentinel.

The three markers, and how to choose one, are on the [Experiments](experiments.md) page.

## What recompiles besides a code edit

[`Host`](@ref) values never reach the compiled program, and a new [`Device`](@ref) value of the
same element type and size reuses it. Sweeping or scheduling a `Device` value, changing the learning
rate and reseeding compile nothing. The run keywords `accum` and `gradient_clip_norm` are also in
the key: changing `accum` recompiles the gradient program, and changing the clip recompiles the
optimizer program.

[Recompilation](recompilation.md) covers the key, and `ReactantNitro.cache_stats()` to check that a
change did not recompile.

## Revising a method invalidates the programs built on it

The compile key carries the world age of every hook, and each cached program also records the
methods it was traced through, so an edit to a helper that `forward` calls several levels down is
caught too. A new [`Nitro`](@ref) recompiles against the current code; an existing handle keeps the
programs it was built with and says so. A changed `const` or global value is not caught: restart
the REPL.

The Revise workflow is on the [Experiments](experiments.md) page.

## Device transfers happen at known times

A scheduled [`Device`](@ref) field uploads once per optimizer step; an unscheduled one uploads
once, at setup.

What may vary with the step, and where each schedule binds, is on the [Schedules](schedules.md)
page.

## Parameters are flattened into one buffer per parameter group

The gradient accumulator and the optimizer state cross the program boundary as `NTuple{G}`, so the
optimizer program emits G updates rather than one per parameter array. An unflattened tree makes
that program grow with the model's array count, and the compile with it.

Parameter groups, decay and clipping are on the [Optimization](optimization.md) page.

## Device buffers are freed per batch

The host GC runs on host memory pressure and cannot see device memory, so a run that leaves batch
buffers to the finalizer can run out of device memory while host memory use stays low. The loop
frees each batch's buffers as soon as the loss readback confirms the step has finished, and the
prefetch pipeline frees what it staged when a phase ends. Freeing nulls the buffer pointer, so a
use-after-free raises on readback instead of returning stale memory.

The host/device boundary, and what a model author may do at it, is covered by the
`reactantnitro-device-boundary` skill in the repository's `skills/` directory.
