---
name: reactantnitro-checkpoint-resume
description: >
  Checkpoint and resume a ReactantNitro run: `TopKCheckpointer` and the latest-plus-top-K
  retention rule (`keep_latest`, `scope = :run | :dir`, current-metric-only selection),
  the `run => checkpoint` source grammar shared by `weights` and `resume`, `resume = :auto`,
  `restore_optimizer`, `restore_best`, finding run ids with `runs()` and `checkpoint_run()`,
  what a checkpoint record holds and why every value in it is a host value, the four
  refusals behind a resume, and what is recomputed rather than restored. Invoke when
  configuring checkpointing, resuming or continuing a run, debugging a refused or failed
  resume, or deciding which artifact to load for evaluation or export, or configuring an
  early stop.
---

# Checkpointing and resume

**`resume = false` is the default: a fresh `Nitro` does not resume.** Resuming is opt in.
`resume = :auto` is exactly `resume = :latest => :latest`, the newest record of the most recent run
in the run directory, and logs what it resolved to (`resume = :auto: continuing :latest => :latest,
run <id>, epoch <n>`). The one difference: in a directory with nothing to resume it warns and
starts a fresh run, where `:latest => :latest` raises. The default used to be `:auto`, and it was
changed because `run_dir` defaults to a name derived from the experiment type, so a second
`Nitro(MyExp())` in the same working directory silently continued the previous run. Picking up
weights nobody named is not something a constructor should do on its own.

**When driving from a Kaimon session, the tools pass this through unchanged** (`reactantnitro-kaimon`):
`nitro_train(..., run_dir = ..., resume = ...)` accepts `"auto"`, `"false"`, a checkpoint path, or
a source string such as `"latest => latest"` or `"<run id> => best"`, and the same default applies:
a reused `run_dir` does NOT resume unless asked. The tools' run registry is process-local, so a
session restart loses in-flight runs and `resume = "auto"` is the recovery path, not a convenience.
Ask for it explicitly after a restart.

## Configuring it

```julia
TopKCheckpointer(; k = 3, metric = :val_loss, mode = :min, dir = nothing,
                   keep_latest = true, scope = :run)
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
*best* checkpoint is not resuming from where you were, which is why both exist. That is the default,
`TopKCheckpointer(; keep_latest = true)`.

`keep_latest = false` is the opt-in escape hatch when a record is large and the filesystem slow:
the run keeps exactly its top K, and an epoch that cannot enter them is never written at all
(it must beat the K-th best strictly), so there is no per-epoch write-then-delete. The final
`stop_reason` rewrite follows the same rule. What it costs: the state training ended in is saved
only if the last epoch was a top-K epoch, `weights = :latest => :latest` and `resume = :auto`
resolve to the run's newest *retained* record (its most recent improvement, not where it stopped), a resume from
it continues the same run id and retrains the epochs after it, and a crash loses every epoch since
that improvement. Setup warns once about this for a handle that trains, and refuses the flag for a
training run with no `val` split, since no epoch would have a score to be kept by.

`latest` is a retained file, **not a symlink** into the top-K set: top-K rotation deletes its target
during entirely normal operation, and checkpoint directories get copied between paths where tools
handle symlinks inconsistently.

A small manifest sits alongside with file, epoch, score, stop reason, and the run, metric and mode
that wrote each entry, so resume can find the newest without reading every file.

**Retention is per run, and one directory can hold several.** The default `run_dir` is the
experiment's type name, so two fresh runs routinely share one. Setup gives every run an id
(`n.checkpointer.run`, and `run` on each `read_manifest` entry), and the rule above applies to a
run's own entries only: a second run never ranks, rotates, deletes or overwrites the first run's
checkpoints, and a name that would land on another run's file (same seed and config, same epoch,
step and score) is written as `...-run-<id>.jld2` instead. Within a run, top-K ranks only entries
scored under the checkpointer's current `metric` and `mode`. A manifest written before runs were
recorded reads as one unnamed run (`run === nothing`, listed by `runs()` as `"legacy"`), the
oldest, and a new run leaves it alone. A fresh run into an occupied directory says so at setup.

**`scope = :dir` is the space-saving alternative.** The whole directory keeps only its top K by the
current metric across every run, plus this run's newest (by write time, and only under
`keep_latest`); every other entry scored the same way is deleted, another run's and legacy ones
included. Ranking is by score alone. With `keep_latest = false` as well, at most K such files
remain in the directory. Names still never collide: a file another run owns gets the `-run-<id>`
suffix rather than being overwritten. Setup of a training handle warns how many checkpoints of
other runs the directory already holds and that they may be deleted as this run beats them.

**Selection only ever uses the CURRENT metric and mode, and never prunes the others.** Every
selection (retention under either scope and either `keep_latest`, `:best`, `:all => :best`) ranks
the entries scored under the checkpointer's `metric` and `mode` and nothing else. Entries scored
under another pair, written by an earlier run or by this run before a resume changed the selection,
are disjoint: never ranked against current ones and never deleted. When the entries being
considered are mixed, setup of a training handle (over its retention scope) and every `:best`
lookup `@warn`, naming the other metrics and modes with counts and saying they are kept. A run with
no entry under the current metric cannot answer `run => :best`: it raises, listing the metrics the
run does have; set the checkpointer's `metric` and `mode` to match, or pass the path. Legacy entries
with no recorded metric count as matching. **The manifest records a metric's name and mode, not its
definition**: a metric whose name stays the same while what it computes changes (a new
`finalize_metrics`, a different split) cannot be detected, and old and new scores will be ranked
together as if comparable. Rename the metric when its definition changes.

**Resuming is always exact continuation, of a run or as a branch from it.** A resume restores
parameters, layer state, optimizer state, step, epoch and schedule position, under the refusals
below. Resuming a run's newest record (`:auto`, `:latest => :latest`, `"<id>" => :latest`,
`other => :latest`, or that record's path) adopts that run's id, so its later checkpoints join the
same run. Anything else, an earlier record (`=> :best`, `:all => :best`) or a file this manifest
does not list, starts a new run and logs that it branched, and from which run and epoch. That is
what keeps a resume from epoch 5 of a run that reached epoch 12 from rewriting epochs 6 to 12 and
rotating the originals away. The branch's entries carry the source path as `parent`.
`resume = other` and `other => :current` are refused as not supported yet; use `weights = other`
for the parameters.

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

**Checkpoint sources: one grammar for `weights` and `resume`.** Either keyword takes a path, or
`run => checkpoint`:

| Run (left) | Means |
| --- | --- |
| `:latest` | the most recent run in `run_dir` (the run of the entry written last; legacy entries count as oldest) |
| `"<run id>"` | that run; `"legacy"` is the unnamed run of a manifest written before runs were recorded |
| `:all` | every run in `run_dir`; only `:all => :best` |
| `other::Nitro` | that handle's own run, read in its own run directory through its own checkpointer |

| Checkpoint (right) | Means |
| --- | --- |
| `:best` | the run's best record by the CURRENT checkpointer's `metric` and `mode` |
| `:latest` | the run's newest retained record |
| `:current` | a live handle's in-memory parameters; only `weights = other => :current` |

`weights = other` (a bare handle) is the same as `other => :current`. Bare `:best` and `:latest` are
refused with the pair to use (`weights = :latest => :best`), `:all => :latest` is refused in favour
of `:latest => :latest`, and `:all => :current` is invalid. A source that cannot be found raises,
naming the directory, the run ids it holds, and `runs(run_dir)`; nothing asked for is silently
replaced by fresh weights. The pair is resolved to the actual file at setup, so
`checkpoint_source` and an exported bundle's provenance name the file, not the words.

```julia
n = Nitro(MyExp, :baseline; run_dir = "runs/my_run", weights = :latest => :best, logger = nothing)
n = Nitro(MyExp, :baseline; run_dir = "runs/my_run", weights = "a1b2c3d4" => :best)  # a named run
n = Nitro(MyExp, :baseline; run_dir = "runs/my_run", weights = :all => :best)        # best of all
```

That is the construction for evaluation, mining and export. With the handle that trained still in
hand, `weights = trained => :best` names the record of THAT handle's run, whatever else has written
to its directory since. The gate tools take the same sources as strings (`weights = "latest =>
best"`).

**Finding run ids.** `runs(run_dir)` lists the runs a directory's manifest holds, most recent
first: id, checkpoints kept and their epochs, best score with its metric and mode, last write, and
the parent of a branch. `checkpoint_run(nitro)` is a handle's own id (not `run_id`, which is the
logger's), and `show(nitro)` prints it beside `run_dir` and on the selected checkpoint.
`read_manifest(dir)` is the per-entry view.

**`restore_optimizer = true`: a record's parameters and its optimizer state, under a new schedule.**
With `weights` naming a record (a path or any `run => :best | :latest`), the record's optimizer
state is restored as well, normalized to device residency exactly as a resume normalizes it and
checked against the optimizer this run builds: a different rule or group layout refuses. Step,
epoch and schedule start fresh: it is a new run with its own `max_epochs`. It is refused with no
record to take the state from (no `weights`, a bare handle, `other => :current`; restoring a live
handle's optimizer is not supported yet), with `resume` (which always restores it), and on a handle
with no `train` split.

| You want | Write |
| --- | --- |
| the exact training state of a record (a branch unless it is the newest) | `resume = run => :best` |
| a record's parameters and optimizer state, fresh step and schedule | `weights = run => :best, restore_optimizer = true` |
| a record's parameters, fresh optimizer | `weights = run => :best` |

**`restore_best = true`: end `train!` holding the run's best weights.** When training completes,
stops early or is stopped on request (not on error), the handle loads `ps` and `st` from its OWN
run's best checkpoint under the current metric. Step, epoch, `history` and the last metrics stay as
training ended; `checkpoint_source` becomes the best file, so an export names it, and `show` says
the handle holds that checkpoint's weights and which epoch. A run that saved no scored checkpoint
warns and keeps its final weights. A handle restored this way refuses a later `train!`: the best
weights beside the optimizer state and step training ended in are not a state any run was in. The
error prints the three real ones, filled in with the handle's experiment, preset, run directory
and run id (the table above, with `"<id>" => :best`). `train!(e; restore_best = true)` and the
`nitro_train` tool's `restore_best` argument reach the same keyword.

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
