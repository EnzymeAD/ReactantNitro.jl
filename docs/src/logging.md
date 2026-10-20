# Logging: the contract and its backends

The logging contract is ten verbs on your own type:

`log_metrics!`, `log_params!`, `log_tags!`, `log_other!`, `log_confusion!`, `finish!`, `run_id`,
`run_url`, `logger_state`, `reattach!`

It is **duck-typed**: no supertype and no registration. `nothing` is the "no logging" value and has
a no-op method for every verb. A verb your type does not define raises a `MethodError`, so a
wrapper cannot drop data unnoticed; a backend that wants a verb to do nothing defines it as a no-op.
The exceptions are `logger_state`, `run_id` and `run_url`, which carry no data and default to
`nothing`.

The driver owns the step counter and passes `step` and `epoch` to every `log_metrics!` call, so a
logger needs no step counter of its own. Each call names one of three contexts:

- `"train"`: one line per logged optimizer step, carrying `loss`, the [`train_metrics`](@ref)
  values, the schedule values, `sam/gap` under a [`PerturbRule`](@ref), and `ema/beta` with an
  [`EMA`](@ref). With `accum > 1` these are the values of the step's last
  micro-batch.
- `"validate"`: one line per validation pass, carrying the validation metrics at the current step
  and epoch, so it overlays the train curve.
- `"data"`: one line per epoch, carrying the host data path's `data_wait_frac`.

Every varying scalar schedule is logged on the train line with no code from you: a `Device` field as
`schedule/<field>`, an optimizer rule value as `schedule/<group>/<field>`, after the per-group
scaling (see [Schedules](schedules.md)). Constant schedules are left out, since the binding report
records them. In manual mode each optimizer schedule key is logged as given, its path joined by
`/`. Pass `autolog_schedules = false` (see [`autolog_schedules`](@ref)) to turn this off.

[`log_every_n_steps`](@ref) (default `1`) writes the train line only on steps that are a multiple
of `n`, so a fast run does not flood a hosted backend. It thins the writing only: the loss is still
read back and `train_metrics` still runs every micro-batch, and the validate and data lines are
unaffected.

Before the logger is called, the framework drops every non-finite value from the train and validate
lines, and every schedule value that is not a `Real`. Other non-`Real` metric values reach the
backend as they are.

## Three ways to get a backend

1. **Your own logger** defines the verbs in your own code, with no extension and no supertype.
2. **A common public logger** gets an extension here, with the logger as a weak dependency.
3. **A backend's own package** can define them itself, taking ReactantNitro as a weak
   dependency so that nobody installing it pays for Reactant and Enzyme.

None of these is type piracy. If both an extension here and the backend's own package define the
verbs, the signatures are identical and whichever loads last wins. The backend's own implementation
is authoritative, and the extension here is retired once one exists.

## The shipped default

A run that names no logger gets `JSONLogger`, which writes JSON Lines to `metrics.jsonl` in the
run's directory.

```json
{"type":"params","seed":42,"max_epochs":40,"width":128}
{"type":"metrics","context":"train","epoch":1,"step":4,"loss":0.31}
{"type":"metrics","context":"validate","epoch":1,"step":160,"loss":0.29}
{"type":"metrics","context":"data","epoch":1,"step":160,"data_wait_frac":0.012}
{"type":"other","key":"binding_report","value":"..."}
{"type":"finish","status":"completed"}
```

Every line carries a `type` field to filter on, and every write flushes, so a crash loses at most
the line being written. The file opens in append mode, so a resumed run extends it. For the same
reason `logger_state` returns `nothing`: there is nothing to reattach (see
[Resumption state](#Resumption-state)).

To turn logging off, pass `logger = nothing` as a `Nitro` keyword or declare a `logger` field with
default `nothing`.

## Supplying a logger

`logger(e)` is called once per `Nitro`, at the start of construction, before `build_data` and the
first compile, and not at all when the `logger` keyword is given. A side effect in the accessor,
such as opening a file or registering a run with a hosted tracker, therefore happens once per run.
A field default runs when the experiment is constructed instead.

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

Each metric becomes a scalar summary tagged `<context>/<name>`, such as `train/loss`,
`train/schedule/backbone/eta` or `validate/loss`, with `step` as the x axis in every context and
the epoch logged as `<context>/epoch`. A metric value that is not a `Real` is an error naming the
key; write it through `backend` instead (see [below](#Past-the-ten-verbs)).

Every scalar is written at the driver's step. Nothing goes through the stdlib `AbstractLogger`
path, so `TBLogger`'s own step counter does not move and cannot disagree with the step in the
checkpoint. Ordinary `@info` under `with_logger` still increments it, which does not affect the
scalars.

Parameters are written twice: as a text summary immediately, so a crash during the first compile
still leaves them in the file, and through `write_hparams!` at `finish!`, because the HParams table
also needs the metric tags, which exist only after metrics have been logged.

**Pass `tb_append`.** `TBLogger(dir)` defaults to `tb_increment`, which writes to `dir_1` when `dir`
already exists, and on a resume it always exists. `logger_state` stores the directory, and
`reattach!` refuses a resume into a different one and names `tb_append` as the fix, instead of
splitting the run's curves across two TensorBoard runs.

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

`backend(lgr)` returns the `TBLogger` itself, so every TensorBoardLogger function stays callable.
A phase monitor can reach it through `info.logger`, the run's logger:

```julia
register_phase_monitor!(n) do phase, step, epoch, info
    haskey(info, :metrics) || return                # only the transition out of EvalStepping
    lg = ReactantNitro.backend(info.logger)
    lg isa TBLogger || return
    TensorBoardLogger.log_histogram(lg, "weights/dense1", vec(Array(n.ps.layer_1.weight)); step)
end
```

The same monitor is where `log_confusion!` belongs. **The framework never calls it**, because only
the experiment knows that a matrix exists and what its class labels mean. Accumulate the matrix in
`metrics` with `:sum`, stash it from `finalize_metrics`, and emit it on the transition out of
`EvalStepping`, when a new one has just been computed.

If you read event files back from Julia, `TensorBoardLogger.map_summaries` raises a `FieldError` on
any text summary, because its deserializer reads a field its generated protobuf type does not have.
The bug is in the reader; TensorBoard reads the files correctly.

## Resumption state

`logger_state` is machine-readable resumption state, opaque to the framework and backend-specific:
a hosted experiment key, a run id plus project and entity, a tracking URI, a directory. It must
return plain serializable data, never the live backend object, because the checkpoint record goes
through JLD2 and outlives the process.

`logger_state` defaults to `nothing`, which stores nothing and attempts no reattachment, so a
backend that defines neither verb keeps working. If `logger_state` returns anything else,
`reattach!` is **required**, and a missing method is a `MethodError`. The record stores the
logger's type with its state, and a resume into a different logger type is refused with both names.

`run_id`, `run_url` and `logger_info` are informational, for humans and for tools.
`logger_info(nitro)` is what the Kaimon `nitro_logger` tool renders; see [Kaimon](kaimon.md).
