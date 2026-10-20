# Checkpoints

A run writes checkpoints as it trains; later handles read them back to resume, to evaluate, to
export, or to average. This page covers what a checkpoint holds, which ones a run keeps, how to name
one, and how to average several. The docstrings of [`Checkpointer`](@ref), [`Average`](@ref) and
[`Nitro`](@ref) are the reference.

## What a checkpoint is

**A checkpoint is full training state; weights are parameters alone.** Every checkpoint holds the
parameters, layer state and optimizer state, plus the step, epoch, seed, config and validation
metrics, so any of them can be resumed. A run with an [`EMA`](@ref) also keeps the average in the
record's `ema` field. Anything that produces parameters alone, for export or
serving, is named for weights.

A [`CheckpointRecord`](@ref) is a JLD2 file, and every value in it is a host value. A device array
would serialize without complaint and fail later, in another process, so the framework converts on
the way out and asserts at write time, naming the path of any leaf that is still on a device.

Beside the records sits a manifest, `manifest.jld2`, listing each kept file with its epoch, score,
`stop_reason`, and the run, metric and mode that wrote it. Lookups and resume read the manifest and
never open a record to find one. [`read_manifest`](@ref) lists it, and [`checkpoint_info`](@ref)
reads one record's metadata without its weights.

## Which checkpoints a run keeps

```julia
ReactantNitro.checkpointer(::MyExp) = Checkpointer(; keep = (Best(3), Last(1)))
```

`keep` is a set of selectors, and the run keeps their **union**:

| Selector | Keeps |
| --- | --- |
| [`Best`](@ref)`(k)` | the `k` best by the run's [`monitor`](@ref) |
| [`Last`](@ref)`(n)` | this run's latest `n`, whatever they scored |
| [`Epochs`](@ref)`(r)` | this run's entries whose epoch is in `r` |
| [`Every`](@ref)`(n)` | this run's entries whose epoch is a multiple of `n` |

The default, `(Best(3), Last(1))`, keeps the best three plus the latest. Keeping both is
deliberate: resuming from the best checkpoint is not resuming from where training stopped. Some
other choices:

```julia
Checkpointer(; keep = (Best(3), Last(10)))   # a window of the latest 10, for averaging
Checkpointer(; keep = Last(5))               # a rolling window
Checkpointer(; keep = (Best(1), Every(10)))  # the best, and every tenth epoch for later analysis
Checkpointer(; keep = Best(3))               # the best three only
```

Checkpoints are ranked by the run's [`monitor`](@ref), `metric => mode`, which
[`EarlyStopping`](@ref) and [`Plateau`](@ref) schedules also watch, so it is set once:

```julia
ReactantNitro.monitor(::MyExp) = :acc => :max
```

Without `metrics`, `monitor` defaults to `:loss => :min`. With `metrics`, declare it, or turn on
[`eval_loss`](@ref) to report the validation loss and keep that default; setup refuses a `Best`
selector with nothing to rank by, and `Checkpointer(; keep = Last(1))` needs no metric at all.
`Checkpointer(; metric, mode)` ranks by something else; give both. The metric must be a key
`finalize_metrics` returns, not the name a logging backend shows (`:acc`, not `:validate_acc`), and
finite: a `NaN` compares false against everything, so it is refused when read. Names carry no
split: `:loss` is the validation loss the framework supplies when the experiment defines no
`metrics`, and an accuracy is `:acc`, never `:val_acc`. With a custom
`metrics`, a name cannot be checked until validation first runs, so a misspelled one raises at the
end of the first epoch. `checkpointer = nothing` writes nothing.

**An epoch is written only if some selector would keep it.** With a [`Last`](@ref), every epoch is
written. Without one, as in `keep = Best(3)`, an epoch that cannot enter the top three is never
written at all, which saves a write and a delete per epoch when records are large. The cost: the
state training ended in is saved only if a selector keeps the last epoch, `resume = :auto` continues
from the latest kept epoch rather than from where training stopped, and a crash loses every epoch
since. Setup warns about this, and refuses a `keep` of only `Best` selectors for a run with no `val`
split, since no epoch would have a score.

### Several runs in one directory

The default `run_dir` is the experiment's type name, so runs routinely share a directory. Setup
gives each run an id, and under the default `scope = :run` retention applies to the run's own
entries: a run never ranks, rotates, deletes or overwrites another run's checkpoints. A filename
that would land on another run's file gets `-run-<id>` inserted instead. [`runs`](@ref)`(run_dir)`
lists the runs a directory holds, and [`checkpoint_run`](@ref)`(nitro)` is a handle's own id.

`scope = :dir` saves space: [`Best`](@ref) ranks the whole directory across every run, and a write
deletes every other entry scored the same way, other runs' included. Setup warns a training handle
when the directory already holds checkpoints this may delete.

### Only the current metric is ranked

Every selection ranks entries scored under the checkpointer's current `metric` and `mode`, and
nothing else. Entries scored under another pair, by an earlier run or before a resume changed the
selection, are never ranked against current ones and never deleted; setup and `:best` lookups warn
when they are mixed. The manifest records a metric's name and mode, not its definition, so rename a
metric when what it computes changes, or old and new scores will be ranked together.

## Naming a checkpoint

`weights` and `resume` take the same sources: a path, or `run => checkpoint`.

| Run (left) | Means |
| --- | --- |
| `:latest` | the most recent run in `run_dir` |
| `"<run id>"` | that run; [`runs`](@ref)`(run_dir)` lists the ids |
| `:all` | every run in `run_dir`, only as `:all => :best` |
| `other::Nitro` | that handle's own run, in its own run directory |

| Checkpoint (right) | Means |
| --- | --- |
| `:best`, or `Best(1)` | the run's best by the current checkpointer's metric and mode |
| `:latest`, or `Last(1)` | the run's latest kept record |
| `Average(sel)` | the average of the checkpoints `sel` picks, for `weights` only (see below) |
| `:current` | a live handle's in-memory parameters, only as `weights = other => :current` |

```julia
n = Nitro(MyExp(); run_dir = "runs/my_run", weights = :latest => :best)
n = Nitro(MyExp(); run_dir = "runs/my_run", weights = "a1b2c3d4" => :latest)
n = Nitro(MyExp(); run_dir = "runs/my_run", weights = :all => :best)
```

Bare `:best` and `:latest` are refused with the pair to write, and a source that cannot be found
raises, naming the directory and the runs it holds; nothing asked for is replaced by fresh weights.
The pair is resolved to its file at setup, so `checkpoint_source` and an export's provenance name
the file.

## Resume versus weights

**`resume` continues training exactly**: parameters, layer state, optimizer state, step, epoch and
schedule position. `resume = false` is the default, and `resume = :auto` is `:latest => :latest`
except that a directory with nothing to resume starts fresh with a warning. Resuming a run's latest
record continues that run under its id. Resuming anything else, an earlier record or a file the
manifest does not list, branches into a new run and says so, so the original run's later epochs are
never overwritten.

**`weights` takes parameters only**: a new run with a fresh optimizer, step and schedule.
`restore_optimizer = true` also takes the record's optimizer state, under a fresh schedule.

| You want | Write |
| --- | --- |
| the exact training state of a record | `resume = run => :best` |
| a record's parameters and optimizer state, fresh step and schedule | `weights = run => :best, restore_optimizer = true` |
| a record's parameters, fresh optimizer | `weights = run => :best` |
| another handle's parameters, in memory | `weights = other` |
| a record's EMA as the parameters, fresh optimizer | `weights = run => :best, load_ema = true` |

A resume restores the seed from the record, overriding a `seed` keyword with a warning, and
recomputes derived values by running `derive` again. It refuses, with a diff, when:

1. the `GraphConst` config changed (`Host` fields may change, which is how `max_epochs` is raised;
   a field added since the record is compatible at its declared default)
2. the flat parameter permutation moved, which would scramble the optimizer state against the
   parameters
3. a `:w0` decay anchor does not match its recorded checksum
4. the logger backend differs from the one whose state is in the record

Every restore refuses a record whose parameter tree does not fit the model, naming the leaf and
both sizes. A `weights` load only warns about a changed `GraphConst` config: it builds the program
from the current experiment, so evaluating old weights under a different graph is allowed.

A seed sweep that varies the seed and not `run_dir` resumes into itself under `resume = :auto`; vary
the directory too.

**`restore_best = true`** ends `train!` holding the run's own best checkpoint rather than its final
weights, and its EMA with them when the run keeps one. Step, epoch and history stay as training ended, and the handle then refuses a further
`train!`, since those weights beside that optimizer state are not a state any run was in; the error
shows the constructions that continue from the best checkpoint.

## Averaging checkpoints

Averaging the weights of several checkpoints from one run often generalizes better than any single
one. [`Average`](@ref) takes a selector and averages the parameters of what it picks:

```julia
Checkpointer(; keep = (Best(3), Last(10)))                      # while training

n = Nitro(MyExp(); run_dir = "runs/my_run", weights = :latest => Average(Last(10)))
n = Nitro(MyExp(); run_dir = "runs/my_run",
          weights = :latest => Average(Best(5; among = Last(20))))
```

- **Selection.** Any selector works. `Best(k; among = sel)` ranks only what `sel` picks, so
  `Best(5; among = Last(20))` is the best five of the latest twenty. `among` is for loading only.
- **Weighting.** Uniform, the only weighting so far. Records are streamed one at a time into a
  `Float64` accumulator, so memory holds one record and the sum.
- **Layer state is not averaged.** It comes from the best picked checkpoint when the selector ranks
  (`Best`) and the latest otherwise, or as `state = :best | :latest` says. That is exact for a model
  without running statistics and an approximation for one with them, such as BatchNorm.
- **The run must still hold them.** When it does not, the error lists the epochs it kept and the
  `keep` that would have held them. `other => Average(...)` checks `other`'s `keep` first.
- **EMA.** `load_ema = true` averages the picked records' EMAs instead, and refuses when one
  keeps none. `restore_best = Average(...)` on a run with an EMA averages both, and warns and keeps
  the final EMA when a picked record has none.
- **Refused:** averaging across runs (`:all => Average(...)`, planned), `Average` in `resume`, and
  `restore_optimizer = true` with an average, which carries no optimizer state.

[`average_checkpoints`](@ref) builds the same averaged record, and with `save` writes it to a file:

```julia
average_checkpoints("runs/my_run", :latest => Average(Last(10)); save = "runs/my_run/avg10.jld2")
n = Nitro(MyExp(); weights = "runs/my_run/avg10.jld2")
```

The file is an ordinary record outside the manifest, so retention never deletes it, and it refuses
to be overwritten. Its `averaged_from` field names the spec, run, directory, files, epochs and
weights it averages: [`checkpoint_info`](@ref) shows it, a handle loaded from it carries it as
`nitro.averaged_from`, and an export records it in its provenance.

`restore_best = Average(Last(10))` ends `train!` holding the run's own average instead of its best
checkpoint. Setup refuses a `keep` that cannot hold the selection, and a run that ends before it has
enough, after an early stop say, warns and keeps its final weights.

## From the gate tools

The [Kaimon](kaimon.md) tools take sources as strings, in the same grammar written as Julia:

```julia
nitro_validate(experiment="MyModels.MyExp", run_dir="runs/my_run", weights="latest => best")
nitro_validate(experiment="MyModels.MyExp", run_dir="runs/my_run",
               weights="latest => Average(Last(10))")
nitro_train(experiment="MyModels.MyExp", run_dir="runs/my_run", resume="auto",
            restore_best="Average(Last(10))")
```

The string is parsed against a whitelist of the selector constructors and integer, range, vector
and symbol literals, and is never evaluated, so it can mean nothing the keyword cannot. `resume`
also takes `"auto"` and `"false"`, and `restore_best` takes `true` and `false`.

## Migrating from `TopKCheckpointer`

[`TopKCheckpointer`](@ref) is deprecated and supported until at least 1.0.0. It warns once per
session and builds the equivalent [`Checkpointer`](@ref), which writes the same files:

| Before | After |
| --- | --- |
| `TopKCheckpointer(; k = 3)` | `Checkpointer(; keep = (Best(3), Last(1)))`, the default |
| `TopKCheckpointer(; k = 3, keep_latest = false)` | `Checkpointer(; keep = Best(3))` |

`metric`, `mode`, `dir`, `name` and `scope` carry over unchanged, and `ckpt.k` and
`ckpt.keep_latest` remain readable.

## Two operational notes

**Do not resume into a directory holding a partial `.tmp` file.** A write killed mid-flight can
leave one, and resuming in place walks the new run into the same path. Move or delete it first.

**Use the checkpoint files, not the log, as the progress signal of a long run.** Julia's stderr is
block-buffered when redirected to a file, so a healthy run's log can stay empty until it exits.
