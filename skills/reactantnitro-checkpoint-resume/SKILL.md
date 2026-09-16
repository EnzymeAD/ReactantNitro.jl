---
name: reactantnitro-checkpoint-resume
description: >
  Checkpoint and resume a ReactantNitro run: `TopKCheckpointer` and the latest-plus-top-K
  retention rule, what a checkpoint record holds and why every value in it is a host
  value, opt-in `resume = :auto` and the four refusals behind it, and what
  is recomputed rather than restored. Invoke when configuring checkpointing, resuming or
  continuing a run, debugging a refused or failed resume, or deciding which artifact to
  load for evaluation or export, or configuring an early stop.
---

# Checkpointing and resume

**`resume = false` is the default: a fresh `Nitro` does not resume.** Resuming is opt in.
`resume = :auto` looks for `latest` in the run directory and continues from it, and an explicit
path names one directly. The default used to be `:auto`, and it was changed because `run_dir`
defaults to a name derived from the experiment type, so a second `Nitro(MyExp())` in the same
working directory silently continued the previous run. Picking up weights nobody named is not
something a constructor should do on its own.

**When driving from a Kaimon session, the tools pass this through unchanged** (`reactantnitro-kaimon`):
`nitro_train(..., run_dir = ..., resume = ...)` accepts `"auto"`, `"false"`, or a checkpoint path,
and the same default applies: a reused `run_dir` does NOT resume unless asked. The tools' run
registry is process-local, so a session restart loses in-flight runs and `resume = "auto"` is the
recovery path, not a convenience. Ask for it explicitly after a restart.

## Configuring it

```julia
TopKCheckpointer(; k = 3, metric = :val_loss, mode = :min, dir = nothing)
```

`metric` names a key your `finalize_metrics` returns (`:val_loss` is what the default `metrics`
emits, not a magic name). A metric nothing emits is a setup error naming the ones available.
`checkpointer = nothing` writes nothing.

**It must also be FINITE**, and that is checked, because the checkpointer compares it against the
retained set and a `NaN` compares false against everything. The way this arrives is not exotic: a
metric whose count was zero on a split reduces to `0 / 0`. See `reactantnitro-metrics` for the scrub.

**Under `resume = :auto`, a second `train!` into a directory that already holds checkpoints
continues rather than starting over.** That is the point of it, and it has one trap worth knowing
before you opt in: a seed sweep that varies the seed and not the run directory resumes into itself
and runs one seed N times. Vary `run_dir` too.

**Retention is top-K plus a `latest` rule: never rotate out the newest, whatever it scored.** On disk
that is K+1 files when the newest is not among the best and exactly K when it is. Resuming from the
*best* checkpoint is not resuming from where you were, which is why both exist.

`latest` is a retained file, **not a symlink** into the top-K set: top-K rotation deletes its target
during entirely normal operation, and checkpoint directories get copied between paths where tools
handle symlinks inconsistently.

A small manifest sits alongside with file, epoch, metric, and stop reason, so resume can find the
newest without reading every file.

## Early stopping

```julia
train!(e; early_stop = EarlyStopping(; metric = :val_loss, mode = :min,
                                       patience = 5, min_delta = 1f-4))
```

`early_stop` defaults to `nothing`, so a bare experiment does not early-stop. `EarlyStopping` is a
small dedicated component, independent of the checkpointer even though both track improvement: the
metrics can legitimately differ (checkpoint on `mae`, stop on `val_loss`), and the jobs differ
(retention versus control flow). They share only the convention of a `metric` name and a
`:min`/`:max` mode.

`metric` names a key your `finalize_metrics` returns, the same rule as the checkpointer, and a
metric nothing emits is an error naming the ones available, checked before training when the
framework can know the key set. `patience` counts **epochs without improvement** and `min_delta`
is **absolute**, both matching Keras and Lightning: an epoch improves when it beats the best seen
by more than `min_delta`, and anything else, including an equal or slightly better value, counts
against patience.

**Stopping is graceful and recorded.** The epoch finishes, validation runs, the checkpoint is
written, the logger finalizes, and the run exits through the normal `Done` path. `stop_reason` in
the checkpoint record says `:early_stop` rather than `:completed`, which is what makes resume
comprehensible: a run that early-stopped and is resumed with `:auto` would immediately re-satisfy
the condition, and with the reason stored it says so instead of exiting silently.

`request_stop!` sets the same flag imperatively, from a REPL or a phase monitor. Both are checked
once per epoch after validation. Custom policies implement `should_stop(es, epoch, metrics)` for
their own type; a `::Nothing` method returns `false`, which is how "no early stopping" stays the
default with no branch in the driver.

**^C in the REPL is the same stop.** The entry points run on a worker thread, and ^C is delivered
to the parked caller, which converts it into `request_stop!`: the epoch finishes, the checkpoint is
written, and the record's `stop_reason` reads `:requested`, so a ^C'd run resumes cleanly, exactly
like one that stopped on patience.

## Checkpoint versus weights

**A "checkpoint" is full training state; "weights" are parameters alone.** Optimizer state is saved
on every checkpoint including top-K ones, not only on a dedicated resume checkpoint, because an API
that says checkpoint and hands back something unresumable has picked the wrong word. Anything
producing parameters alone for export or serving is named for weights.

## Every value in a record is a HOST value

This is a correctness requirement, not tidiness, and it is the single most expensive class of bug
this framework has had.

A device array survives serialization: it is written field by field, raw pointer included, and read
back without complaint. The pointer means nothing in the reading process, so the failure surfaces much
later, in a **different process**, as an assertion from inside an unrelated read-back that names
neither the field nor serialization.

The framework converts on the way out and **asserts at write time**, naming the path of any leaf that
survived. If you see that error, it is telling you a walker has a gap, not that you did something
wrong. A type that genuinely cannot hold a host value needs a surrogate plus a method to turn it back;
that is how the layer-state RNG is handled.

The flow is uniform and has exactly one normalization point per path: **write host, read host,
normalize on the way in.**

## What resume restores, and what it does not

Restored: parameters, layer state, optimizer state including its moments, the step and epoch counters,
the seed, and derived-value bookkeeping. The logger reattaches so a resumed run extends one history
rather than opening a second.

**Recomputed, not restored:** derived values. `derive` runs again, so resuming against changed data
picks up new values. They are recorded in the record so a change is visible after the fact.

**The seed is restored from the record and overrides a `seed` keyword, with a warning when they
differ.** Restoring is what makes the initialization reproducible, so the override is right; only its
silence was wrong. Watch for this if you are sweeping seeds: vary the run directory too, or the sweep
resumes into itself and runs the same seed N times.

## The four refusals

`resume = :auto` is safe to reach for only because the compatibility check lives in the framework
rather than in your harness. It refuses, with a diff, when:

1. **Config changed.** The flattened config is in the record. Identical continues, changed errors,
   neither silently does the wrong thing. `Host` fields are excluded, since raising `max_epochs` on
   resume is the normal case. A changed derived `GraphConst` field is an error; a changed derived
   `Device` value is fine and is logged.
2. **The parameter permutation moved.** A silently different flat permutation scrambles optimizer
   state against parameters.
3. **A decay anchor does not match its checksum.** For any group anchored to `:w0`, the anchor is
   re-captured from a fresh initialization and checked against the record. A match proves you are
   anchored to the same tensor the original run used.
4. **The logger backend differs** from the one whose state is in the record.

## Two operational traps

**Do not resume into a run directory containing a wedged temporary file.** A checkpoint write killed
mid-flight can leave a partial `.tmp`; resuming in place walks the new run into the same path and
wedges it identically. Move or delete the fragment first.

**A long run's log may be empty and the run perfectly healthy.** Julia's stderr is block-buffered when
redirected to a file, so nothing appears until the process exits, and a hard kill discards the buffer
entirely. **Use the checkpoint files as the progress signal**, not the log. If you need live progress,
write it yourself with an explicit open, append, and close per epoch from a phase monitor
(`register_phase_monitor!`).
