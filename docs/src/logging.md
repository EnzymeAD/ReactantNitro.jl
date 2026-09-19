# Logging: the contract and its backends

The logging contract is ten verbs on your own type:

`log_metrics!`, `log_params!`, `log_tags!`, `log_other!`, `log_confusion!`, `finish!`, `run_id`,
`run_url`, `logger_state`, `reattach!`

It is **duck-typed**: no supertype, no registration, and `nothing` is the public "no logging" value
with a no-op method for every verb. A missing method is a `MethodError` rather than a silent no-op,
because optional-no-op contracts make wrappers quietly lossy; a backend opts into silence per verb.

The step counter lives in the driver, not the logger, which is what makes the contract stateless.
Train metrics carry `step`, validation metrics carry `epoch` plus the current step so the two
overlay, and a third context, `"data"`, carries the host data path's per-epoch `data_wait_frac`.

## Three ways to get a backend

None of them blocks on another:

1. **Your own logger** defines the verbs in your own code, with no extension and no supertype.
2. **A common public logger** gets an extension here, with the logger as a weak dependency.
3. **A backend's own package** may equally define them itself, taking ReactantNitro as a weak
   dependency so that nobody installing it pays for Reactant and Enzyme.

None of these is type piracy. If both sides ship glue for the same backend, the methods have
identical signatures and the last one loaded wins silently, so the convention is that a backend's
own implementation is authoritative and the extension here is retired once one exists.

## The shipped default

A run that names no logger gets `JSONLogger`, which writes JSON Lines to `metrics.jsonl` in the
run's directory. That file is the first place to look for a run's numbers.

```json
{"type":"params","seed":42,"max_epochs":40,"width":128}
{"type":"metrics","context":"train","epoch":1,"step":4,"loss":0.31}
{"type":"metrics","context":"validate","epoch":1,"step":160,"val_loss":0.29}
{"type":"metrics","context":"data","epoch":1,"step":160,"data_wait_frac":0.012}
{"type":"other","key":"binding_report","value":"..."}
{"type":"finish","status":"completed"}
```

Every line carries a `type` discriminator so the file is self-describing and filterable, and every
write flushes, so a crash loses at most the line being written. The handle opens in append mode, so
a resumed run extends the same file rather than truncating it. That is also why `logger_state`
returns `nothing` for it: a file logger has nothing to reattach, and a backend that defines neither
`logger_state` nor `reattach!` keeps working.

`logger = nothing` is the documented opt-out, as a `Nitro` keyword or as a `logger = nothing` field.

## Supplying a logger

`logger(e)` is called **exactly once**, at the end of setup, which is what makes it safe for the
accessor to have a side effect: opening a file here, or registering a run with a hosted tracker.

```julia
# a field on the experiment
@experiment struct MyExp
    logger::Host{Any} = MyTracker.start_run("mnist")
end

# the accessor, which is the usual choice
ReactantNitro.logger(::MyExp) = MyTracker.start_run("mnist")

# the keyword, for a one-off
train!(Nitro(MyExp(); logger = nothing))
```

## TensorBoard

TensorBoardLogger arrives as an extension, active as soon as the package is loaded.

```julia
using ReactantNitro
using TensorBoardLogger        # activates ReactantNitroTensorBoardLoggerExt

ReactantNitro.logger(::MnistMLP) = TBLogger("tb/mnist_v1", tb_append)
```

```
tensorboard --logdir tb
```

Metrics become scalar summaries tagged by context, `train/loss`, `validate/val_loss`, `data/…`,
with `step` as the x axis in every context and the epoch alongside as `<context>/epoch`.

**The driver's step is the only one in the file.** Nothing goes through the stdlib `AbstractLogger`
path, so `TBLogger`'s own global counter never moves and cannot disagree with the step recorded in
the checkpoint. If you also use the logger for ordinary `@info` under `with_logger`, that traffic
still increments it, which is fine and does not interfere.

**Parameters are written twice, deliberately.** They land immediately as a text summary, so a crash
during the first compile still leaves them in the file, and again through `write_hparams!` at
`finish!`, which is the first moment the metric tags the HParams comparison table needs actually
exist.

**Pass `tb_append`.** `TBLogger(dir)` defaults to `tb_increment`, which silently writes to `dir_1`
when `dir` already exists, and on a resume it always exists. `logger_state` stores the directory
and `reattach!` refuses a resume that would write somewhere else, naming `tb_append` as the fix, so
the mistake is an error rather than a run whose curves are split across two entries in TensorBoard's
run list.

### Runs, and comparing experiments

TensorBoard has no run-name field. A **run is a subdirectory** of `--logdir`, and the run name is
the relative path from that root. To compare experiments on one board, give each its own
subdirectory under a shared root:

```julia
ReactantNitro.logger(e) = TBLogger(joinpath("tb", string(nameof(typeof(e)))), tb_append)
```

Two experiments must not share a subdirectory: TensorBoard would merge them into one run and
interleave the curves.

### Past the ten verbs

`backend(lgr)` returns the `TBLogger` itself, so everything TensorBoardLogger can do stays
callable. A phase monitor is the natural place, since `info.logger` hands you the run's logger:

```julia
register_phase_monitor!(n) do phase, step, epoch, info
    phase isa EvalStepping && return                # fires on entry and on exit
    lg = ReactantNitro.backend(info.logger)
    lg isa TBLogger || return
    TensorBoardLogger.log_histogram(lg, "weights/dense1", vec(Array(n.ps.layer_1.weight)); step)
end
```

That is also where `log_confusion!` belongs. **The framework never calls it**, because only the
experiment knows both that a matrix exists and what its class labels mean: accumulate it in
`metrics` with a `nothing` count, stash it from `finalize_metrics`, and emit it on the transition
out of `EvalStepping`, which is exactly when a fresh one exists.

One caveat if you read event files back from Julia rather than from TensorBoard:
`TensorBoardLogger.map_summaries` raises a `FieldError` on any text summary, because its
deserializer reaches for a field its generated protobuf type does not have. That is a reader bug
and not a writer bug, so TensorBoard itself reads these files without complaint.

## Resumption state

`logger_state` is machine-readable resumption state, opaque to the framework and backend-specific:
a hosted experiment key, a run id plus project and entity, a tracking URI, a directory. It must
return plain serializable data, never the live backend object, because the checkpoint record goes
through JLD2 and outlives the process.

The pair is self-describing. If `logger_state` returns anything but `nothing`, `reattach!` is
**required** and a missing method is the usual `MethodError`. Resuming into a different logger type
is refused by name, since the record stores the logger's type alongside its state.

`run_id`, `run_url` and `logger_info` are informational, for humans and for tools.
`logger_info(nitro)` is what the Kaimon `nitro_logger` tool renders; see [Kaimon](kaimon.md).
