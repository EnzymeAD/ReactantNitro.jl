# Logging.jl
#
# The logging contract: ten verbs, their `::Nothing` methods, and the two defaulted accessors.
#
# Duck-typed, with no mandatory supertype; `nothing` is the public "no logging" value. A missing
# method is a loud `MethodError` by design, since optional-no-op contracts make wrappers silently
# lossy. The framework ships one backend, the JSON default in JSONLog.jl. A user's logger defines
# the methods in its own code; a common public logger gets an extension here with the logger as a
# weakdep, defining methods only, never types.
#
# Details each of which has bitten something: `kwargs...` on `log_metrics!` so a future axis does
# not break every backend; the framework drops non-finite values before calling; `log_confusion!`
# takes a plain `Matrix{Int}`; `context` values stay byte-identical ("train", "validate", "data");
# parameter keys stay flat; parameters are logged before the first compile; and `Base.close` must
# not map to `finish!(:completed)`, since in a `finally` it would stamp every failed run completed.
# The step counter lives in the driver, which is what makes the contract stateless.

"""
    log_metrics!(lgr, metrics; step, epoch, context, kwargs...) -> nothing

Log a `NamedTuple` of metrics. `context` is one of:

  * `"train"`: one line per logged optimizer step (every [`log_every_n_steps`](@ref)th step),
    carrying the loss, the [`train_metrics`](@ref) values and, under
    [`autolog_schedules`](@ref), the schedule values under `schedule/` keys
  * `"validate"`: one line per epoch with the `val` split's metrics
  * `"data"`: one line per epoch with `data_wait_frac`, the fraction of the epoch the host spent
    waiting on data; a separate context so it does not mix into the other two series

Values arrive as host values. The framework drops non-finite `Real` values before calling and
passes other values (arrays, tuples) through unchanged. A method must accept `kwargs...` so a
keyword added later does not break it.
"""
function log_metrics! end
log_metrics!(::Nothing, metrics; kwargs...) = nothing

"""
    log_params!(lgr, params) -> nothing

Log the run's hyperparameters. Setup calls this before the first compile, so a crash during
compilation still leaves them logged. `params` is a flat `NamedTuple` of the experiment's
configuration and the run's settings (`seed`, `accum`, `max_epochs`, the resolved prefetch
settings), plus `preset` when the run names one. Keep keys flat: dotted keys split comparison
tables in hosted trackers.
"""
function log_params! end
log_params!(::Nothing, params) = nothing

"""
    log_tags!(lgr, tags) -> nothing

Free-form tags for the run; the backend decides what a tag is.
"""
function log_tags! end
log_tags!(::Nothing, tags) = nothing

"""
    log_other!(lgr, key, value) -> nothing

Log one free-form value under `key`. Setup logs the schedule binding report this way, as
`log_other!(lgr, "binding_report", str)`.
"""
function log_other! end
log_other!(::Nothing, key, value) = nothing

"""
    log_confusion!(lgr, matrix, labels; epoch) -> nothing

Log a confusion matrix, given as a plain `Matrix{Int}` that the backend adapts. The framework
never calls this, since only the experiment knows whether it has a confusion matrix and what its
labels mean. Accumulate it in [`metrics`](@ref) with `:sum`, stash it from
[`finalize_metrics`](@ref), and emit it from a phase monitor on the transition out of
`EvalStepping`:

```julia
metrics(e, out; y)            = (; confusion = confusion_matrix(out, y) => :sum)
finalize_metrics(e, acc, spl) = (e.rt.last_confusion = acc.confusion; derived_scalars(acc))
# then, in a phase monitor:
log_confusion!(info.logger, e.rt.last_confusion, e.rt.class_names; epoch)
```
"""
function log_confusion! end
log_confusion!(::Nothing, matrix, labels; kwargs...) = nothing

"""
    finish!(lgr, status) -> nothing

Close the run on the backend. `status` is `:completed`, `:early_stop`, `:requested` (a
[`request_stop!`](@ref)), or `:error`, matching the `stop_reason` recorded in the checkpoint.

**Do not map `Base.close` to `finish!(:completed)`**: in a `finally` it runs after the driver's
`finish!(:error)` and would mark every failed run completed.
"""
function finish! end
finish!(::Nothing, status) = nothing

"""
    run_id(lgr) -> Any

The backend's run identifier, for people and reports, or `nothing` if it has none (the default).
Stored in the checkpoint record so a checkpoint traces back to its logged run. Unrelated to
[`checkpoint_run`](@ref).
"""
function run_id end
run_id(lgr) = nothing

"""
    run_url(lgr) -> Any

The backend's URL for the run, or `nothing` if it has none (the default). Informational like
[`run_id`](@ref), and stored in the checkpoint record with it.
"""
function run_url end
run_url(lgr) = nothing

"""
    logger_state(lgr) -> Any

The state a resumed run needs to continue the same logged run, such as a hosted experiment key, a
run id plus project, or a tracking URI. Opaque to the framework and stored in the checkpoint
record. Default `nothing`: no resumable state, so nothing is stored and no reattachment is
attempted. Return plain serializable data (`String`, `NamedTuple`, `Dict`), never the live backend
object, since the record is written with JLD2 and outlives the process. A logger that returns
anything else must define [`reattach!`](@ref). [`run_id`](@ref) is separate and informational
only.
"""
logger_state(lgr) = nothing
logger_state(::Nothing) = nothing

"""
    reattach!(lgr, state) -> nothing

Point a logger back at the logged run that `state` (from [`logger_state`](@ref)) identifies.
Setup calls it on resume, before any metric is logged. There is no default method; a logger needs
one exactly when its `logger_state` can return something other than `nothing`. Resuming into a
logger of a different type than the one that wrote the record is refused, naming both.
"""
function reattach! end
reattach!(::Nothing, state) = nothing

"""
    backend(lgr) -> Any

The native backend object, for calls the contract does not cover (images, artifacts, curves).
The default is identity, since a logger is usually the backend object itself; a wrapper returns
what it wraps. Phase monitors reach the logger as `info.logger`.

```julia
backend(w::MyWrapper) = w.exp
```
"""
backend(lgr) = lgr
backend(::Nothing) = nothing

"""
    logger_info(lgr) -> NamedTuple

The parameters that identify where the backend writes, as a plain `NamedTuple`: an experiment
key and URL, a run id and project, or for [`JSONLogger`](@ref) its file path. Informational, like
[`run_id`](@ref); the `nitro_logger` tool renders it. Default `(;)`. Return plain data, never the
live object, since the value is passed to tools and reports. Not stored in the checkpoint record.
"""
logger_info(lgr) = (;)
logger_info(::Nothing) = (;)

"""
    ReactantNitro._adopt_logger!(lgr, dir) -> lgr

Called by setup to give the logger the run's resolved `run_dir`. A pathless
[`JSONLogger`](@ref) sets its path to `metrics.jsonl` there, as the default `Checkpointer`
adopts `run_dir`. A wrapper forwards the call to the logger it wraps; the default returns any
other logger unchanged.
"""
_adopt_logger!(lgr, dir) = lgr
