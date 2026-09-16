---
name: reactantnitro-recompiles
description: >
  Verify a ReactantNitro run is not recompiling, and work out why an edit did not take
  effect. Covers `cache_stats()` as the acceptance check, what is and is not in the
  compile key, why a changed GraphConst misses and a changed Device hits, the rule that a
  `Nitro` is a fixed point (so a revised hook needs a NEW handle), `set_device!` as the
  sanctioned way to sweep without rebuilding, and the two holes the key cannot see. Invoke
  when a run is unexpectedly slow, when a REPL edit appears to be ignored, when sweeping
  a value, or when accepting that a new experiment trains correctly.
---

# Recompiles, the compile cache, and why your edit did nothing

Compiles are the dominant cost on this stack: a real gradient program has been measured at 495.6 s
and its optimizer program at 63.1 s. An unintended recompile per step is not a rounding error, and it
presents as "training is mysteriously slow" rather than as an error.

## The acceptance check: `cache_stats()` after two steps

**"It trains" is not the criterion. "No recompile after step 1" is.**

```julia
using ReactantNitro: cache_stats, cache_reset!

cache_reset!()
n = Nitro(e; max_epochs = 1)
train!(n)
cache_stats()      # (; hits, misses, entries)
```

**The criterion is `misses == entries`, and `misses` not growing with steps.** Both counters are
cumulative, so a miss that is not also a new entry is a program compiled twice, which is exactly what
this check exists to catch. Every later step should be a hit.

**The absolute number depends on what you ran, so do not assert a literal.** Training compiles two
programs, the gradient program and the optimizer program, and there is deliberately **no fused
program** even at `accum = 1`, so 2 is right for a run that never evaluates and 1 would be
surprising. Add one for the evaluation `forward`, which any `validate`, `evaluate` or `predict`
compiles and which an experiment with a `metrics` hook therefore compiles every epoch. Add one more
per `:device` metric hook, and one more again if a split's length is not a multiple of the batch
size, since the metric program is built at the padded width and again at the remainder. **A host
metric adds nothing**, because it is part of no program at all.

So 2 is a subtotal, not the expectation. A perfectly healthy run that trains and validates reports
3, and asserting the 2 in a test suite is a way to fail on a correct run.

Run this on any new experiment before you spend GPU hours on it. It has caught, in this framework's
own history, several defects that nothing else would have: a schedule silently recompiling per step,
a widened element type, a value baking that should have crossed as an input.

If `misses` grows with steps, something in the key is moving per step. The usual causes are a
`GraphConst` field being rebuilt with a new value, a schedule returning a different type than the
field holds, or a batch whose shapes vary.

## What is in the key

| In the key | Why |
| --- | --- |
| the function, its argument **types**, and their **shapes** | shape is a runtime field, so Reactant's own guard covers types but not shapes |
| a hash of the **`GraphConst`** fields | they bake as trace-time constants |
| the resolved method world of every user hook, and every rule's `apply!` | so a redefinition is not silently ignored |

**`Device` fields are skipped by construction.** They are traced inputs, so their values cannot affect
the program, and hashing them would miss on every step since they are rebuilt per step.

**`Host` fields are skipped too**, and for a different reason: they are not in the traced view at all.
Both exclusions are needed and neither implies the other.

So: **changing a `GraphConst` field misses, changing a `Device` field hits, changing a `Host` field
hits.** That is the marker scheme's practical payoff and it is worth asserting in your own tests.

## A `Nitro` is a fixed point

**The compiled programs a handle uses are determined at construction, and nothing it does later
changes them.** If you revise `forward`, `loss`, `metrics`, a rule's `apply!`, or an accessor in the
REPL, the handle you already have keeps the programs it was built with.

```julia
# Revise picks up your edit. The handle does not.
n = Nitro(e)
train!(n)          # ... edit `loss` in your editor ...
train!(n)          # STILL the old program, deliberately
n2 = Nitro(e)      # this one sees the edit
train!(n2)
```

**The Kaimon tools follow the same rule** (`reactantnitro-kaimon`): a run launched with
`nitro_train` holds the `Nitro` it constructed, so an edit to a hook mid-run does not change the
running programs, and a NEW `nitro_train` after the edit picks the edit up exactly like `n2`
above.

This is not a bug to work around; it is what makes a handle's behaviour predictable. The framework
tells you rather than leaving it silent: the fixed-config report names any hook that has been
redefined since the handle was built, so "I edited it and nothing happened" is visible in the output
rather than something you have to deduce.

**The cache is module-level, so rebuilding is cheap.** A new `Nitro` reuses every program whose key
did not move. Rebuilding after an edit is the workflow, not a penalty.

## Sweeping without rebuilding

```julia
device_value(nitro, :temperature)          # read it back as a host value
set_device!(nitro, :temperature, 2.0f0)    # write it, provably no recompile
set_device!(nitro; temperature = 2.0f0, smoothing = 0.1f0)
```

A `Device` field is excluded from the key by construction, so this is the supported way to sweep a
value or change an inference threshold on a live handle. It refuses a `GraphConst` field, a `Host`
field, a scheduled field (the schedule owns it and would overwrite you next step), and a mismatched
type or size, each with an error naming the rebuild. The type check matters because a widened element
type would move the key silently; the size check matters because size is **not** in the key, which is
worse, since the key would match and the failure would land inside XLA.

## The two holes, stated plainly

**Constants reached from a method body are not in the key**: a `const` in your module, a global, a
literal you edited. The key covers the method's identity and world age, not what its body closed
over. If you have changed something like that and are getting stale behaviour, `cache_reset!()` is
the escape hatch and exists for exactly this rather than restarting the REPL.

**A value your MODEL closed over is not in the key either, and this one is the common case.** The key
covers the model's *type*, not what its closure captured, so a model written as a `@compact` block
over a config object emits the same type whatever that object holds. Two experiments differing only
in a captured layer count or step count therefore produce the same type, the same argument shapes,
and the same key, and the second silently reuses the first's program. Nothing raises and the run
looks fine.

The rule that closes it is broader than "the traced code reads it from `e`": **if a value determines
the emitted graph, it must be reachable from `e` as a `GraphConst`, whether or not the traced code
reads it from there.** A config object is a legal `GraphConst`, and marking it is what puts its
contents in the key. Assert it in your own tests, which is one line: the key hash moves when the
config changes and does not move when a `Host` field does.

## When a compile is genuinely slow

Compile cost is dominated by what the tracer has to walk. The usual cause of an unexpectedly long
compile is dataset-sized state reachable from the experiment, which Enzyme traverses element by
element on one thread every time. Leave such fields unmarked (`Host` is the default) or keep them off
the experiment entirely. See `reactantnitro-experiment`.
