# Recompilation: when a change costs a compile

## The cost

Compiles are the dominant cost on this stack: a real gradient program has been measured at
495.6 seconds, and its optimizer program at 63.1. An unintended recompile presents as slow training,
not as an error: nothing raises, the graphs are identical, and the only symptom is time. The
acceptance check below is the first thing to run on a new experiment.

## The compile cache is module-level

Programs are keyed and stored once per process, not once per [`Nitro`](@ref). A new handle on the
same experiment reuses every program whose key did not move, so rebuilding after a REPL edit pays
setup and device conversion, not compilation. The cache is populated lazily, from the arguments a
program is first called with; there is no declaration of batch shapes anywhere.

## The acceptance check

"It trains" is not the criterion. "No recompile after step 1" is.

```julia
ReactantNitro.cache_reset!()
n = Nitro(e; max_epochs = 1)
train!(n)
ReactantNitro.cache_stats()      # (; hits, misses, entries)
```

Both functions are unexported and documented on the [API](api.md) page. The criterion is
`misses == entries` and `misses` not growing with steps. Both counters are cumulative, so a miss
that is not also a new entry is a program compiled twice.

The absolute number depends on what you ran, so assert the shape of the answer, not a literal. A
train-only run compiles 2: the gradient program and the optimizer program, with deliberately no
fused program even at `accum = 1`. Validation adds 1 for the evaluation `forward`, so a run that
trains and validates reports 3. Add one more per `:device`-residency metric hook; a host metric
adds nothing, because it is part of no program at all; and a split whose length is not a
multiple of the batch size builds the `:device` metric program twice, once at the padded width
and again at the remainder. If `misses` grows with steps, something in
the key is moving per step; the usual causes are a `GraphConst` field rebuilt with a new value, a
schedule returning a type the field does not hold, or a batch whose shapes vary.

## What is in the key

Every program's key is a tuple, and each component earns its place:

| In the key | Why |
| --- | --- |
| the function, its argument types, and their shapes | shape is a runtime field, so Reactant's own guard covers types but not shapes |
| a hash of the `GraphConst` fields only | they bake into the graph as trace-time constants |
| the resolved method world of every user hook and every rule's `apply!` | so a redefinition is not silently ignored |

Shape is a runtime field, so a ragged batch passes Reactant's generated type guard and fails
inside XLA; a key that skipped shapes would serve it a program compiled for a different width.
Function identity is stable across a redefinition, so without the world component a redefined hook
would hit the cache and run the old program.

Two exclusions. [`Device`](@ref) fields are traced inputs rebuilt every step, so hashing them would
miss on every step. [`Host`](@ref) fields are not in the traced view at all. Neither exclusion
implies the other: [`compile_view`](@ref) strips `Host` and leaves `Device` in place, so keying on
the view itself would hash device scalars that change every step.

The practical rule: changing a [`GraphConst`](@ref) field misses, changing a `Device` field hits,
and changing a `Host` field hits.

## The second guard: the dependency closure

The key covers the hooks. It does not cover what the hooks *call*, and a redefined helper two
levels below [`forward`](@ref) leaves every hook's world age untouched, so on the key alone it
would hit the cache and silently run the old program.

The entry points close that. Each cached program also records the transitive closure of the
methods it was compiled against, and [`train!`](@ref), [`validate`](@ref), [`evaluate`](@ref) and
[`predict`](@ref) re-resolve that closure against live dispatch on entry, **poisoning** any entry
whose methods moved. A redefinition anywhere below the hooks therefore misses for a new
[`Nitro`](@ref), which recompiles against current dispatch.

The scan is memoized on the world counter, so an entry point that follows no redefinition pays one
counter comparison and skips the re-resolution entirely.

An existing handle is a fixed point and keeps the programs it was built with, now stale; the
entry-point report says so. There are two messages:

- **The cache-level line** names how many entries were poisoned and which methods drifted. It is
  informational and fires once per redefinition, whichever handle happens to run next. Seeing it
  while running a freshly built handle is the *good* case: the poisoning is what makes that handle
  recompile against your edit.
- **The per-handle line**, `! this handle's compiled programs had dependency methods redefined`,
  fires only when the handle actually running holds one of the poisoned programs. That is the one
  that means "rebuild the `Nitro`".

What neither guard can see is values rather than methods: a `const` redefined, a global, or a
literal edited in place. Those are the first of [the two holes](#Two-holes-in-the-key)
below, and `ReactantNitro.cache_reset!()` is the escape hatch.

## Hashing the GraphConsts

`ReactantNitro.graphconst_field_hash` hashes [`compile_view`](@ref)`(e)`'s `GraphConst` fields
only, that is
`setdiff(fieldnames(typeof(e)), device_fields(typeof(e)), host_fields(typeof(e)))`. The seed is the
type name, not the type: a `Device` field carries its own type parameter, so `typeof(e)` changes
whenever one is converted to device residency, which happens at setup and again on every optimizer
step, and seeding with `hash(T)` would make every step miss and recompile.

`ReactantNitro.assert_graphconst_hashable` refuses at setup a `GraphConst` value that does not
hash and compare by content. The check is one line of semantics: a value and its `deepcopy` must hash equal and be
`isequal`, and the two agree exactly when the answer comes from contents. What fails is a struct
containing a mutable field, because `Base.hash` has no method for that struct and falls through to
`hash(objectid(x), h)`, and `objectid` reaches the mutable field by identity rather than descending
into it; the field's own content-based hash is never called. A bare `GraphConst` `Vector` field is
fine, because arrays hash by content.

Three symptoms follow if one gets through, and the third is why this is an error rather than a
warning: a full recompile per `Nitro` handle, a spurious resume refusal whose diff prints
identically on both sides, and silent reuse of the wrong program when the value is mutated in place.

## Two holes in the key

Two things the key cannot see, both of which present as an edit that "did nothing".

**Constants reached from a method body are not in the key.** A `const` in your module, a global, an
edited literal: the key covers the method's identity and world age, not what its body closed over.
`ReactantNitro.cache_reset!()` is the escape hatch.

**A value your model closed over is not in the key either, and this one is the common case.** A
model written as a `@compact` block over a config object emits the same type whatever the object
holds, so two experiments differing only in a captured value produce the same type, the same
argument shapes, and the same key, and the second silently reuses the first's program. Nothing
raises, and the run looks fine.

The rule that closes it is broader than "the traced code reads it from `e`": if a value determines
the emitted graph, it must be reachable from `e` as a [`GraphConst`](@ref), whether or not the
traced code reads it from there. A config object is a legal `GraphConst`, and marking it is what
puts its contents in the key.

## Nitro is a fixed point

The programs a handle uses are determined at construction. A revised hook needs a new handle, and
the rebuild recompiles only what the edit changed. A running [`train!`](@ref) is pinned to the
world age it started in, so mid-run edits do nothing. [Experiments](experiments.md) owns the Revise
workflow; the shape of it:

```julia
n = Nitro(e)
train!(n)          # ... edit `loss` in your editor ...
train!(n)          # still the OLD program, deliberately
n2 = Nitro(e)      # this one sees the edit
train!(n2)
```

## When a compile is genuinely slow

When the cache is healthy and a compile is still slow, the cost is what the tracer walks and how
many operations the emitted graph contains. The usual cause is dataset-sized state reachable from
the experiment, which Enzyme traverses element by element on one thread; leave such fields unmarked
or keep them off the experiment ([Experiments](experiments.md)). The optimizer program's size is
governed by the flat parameter layout, one update per parameter group rather than per parameter
array ([Optimization](optimization.md)).
