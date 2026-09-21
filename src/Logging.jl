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

The metric channel. `context` is `"train"` (one line per optimizer step), `"validate"` (one per
epoch of metrics), or `"data"` (the per-epoch `data_wait_frac`, a third context so it dilutes
neither count). The framework drops non-finite values before calling.
"""
function log_metrics! end
log_metrics!(::Nothing, metrics; kwargs...) = nothing

"""
    log_params!(lgr, params) -> nothing

Hyperparameters, logged **before the first compile**, since a compile-time crash would otherwise
lose them. Keep the key style flat: dotted keys split comparison tables under one-project-per-model.
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

Free-form single values. The schedule binding report goes through here as
`log_other!(lgr, "binding_report", str)` so it lands in the run's record.
"""
function log_other! end
log_other!(::Nothing, key, value) = nothing

"""
    log_confusion!(lgr, matrix, labels; epoch) -> nothing

Takes a plain `Matrix{Int}`; the backend adapts it. The framework never calls this: a confusion
matrix is a user metric accumulated by the `count === nothing` rule, and only the experiment knows
both that one exists and what its labels mean. The shape that works is to accumulate it in
[`metrics`](@ref), stash it from [`finalize_metrics`](@ref), and emit it from a phase monitor on
the transition out of `EvalStepping`:

```julia
metrics(e, out; y)            = (; confusion = (confusion_matrix(out, y), nothing))
finalize_metrics(e, acc, spl) = (e.rt.last_confusion = acc.confusion; derived_scalars(acc))
# then, in a phase monitor:
log_confusion!(info.logger, e.rt.last_confusion, e.rt.class_names; epoch)
```
"""
function log_confusion! end
log_confusion!(::Nothing, matrix, labels; kwargs...) = nothing

"""
    finish!(lgr, status) -> nothing

Finalize. `status` is `:completed`, `:early_stop`, or `:error`, matching the `stop_reason` recorded
in the checkpoint.

**Do not map `Base.close` to `finish!(:completed)`**: in a `finally` it runs *after* the driver's
`finish!(:error)` and would stamp every failed run completed.
"""
function finish! end
finish!(::Nothing, status) = nothing

"""
    run_id(lgr) -> Any

Informational, for humans and reports. Stored in the checkpoint record so a checkpoint traces back
to its experiment.
"""
function run_id end
run_id(::Nothing) = nothing

"""
    run_url(lgr) -> Any

Informational, as [`run_id`](@ref).
"""
function run_url end
run_url(::Nothing) = nothing

"""
    logger_state(lgr) -> Any

Machine-readable resumption state, opaque to the framework and backend-specific: a hosted
experiment key, a W&B run id plus project, a tracking URI. Default `nothing`, meaning no resumable
state, in which case nothing is stored and no reattachment is attempted. It must return plain
serializable data (`String`, `NamedTuple`, `Dict`), never the live backend object, since the record
goes through JLD2 and outlives the process. If this returns anything but `nothing`,
[`reattach!`](@ref) is required. Separate from [`run_id`](@ref), which is informational only.
"""
logger_state(lgr) = nothing
logger_state(::Nothing) = nothing

"""
    reattach!(lgr, state) -> nothing

Restore a logger onto its previous experiment, before the run starts and before any metric is
logged. Required if and only if [`logger_state`](@ref) returns non-`nothing`, with no default
method. Resuming into a different logger type refuses with both names.
"""
function reattach! end
reattach!(::Nothing, state) = nothing

"""
    backend(lgr) -> Any

Reach the native backend object. Default identity: the logger a user passes is their backend
object, and every other client function in that package stays callable on it. This escape hatch is
why the contract has no `log_image!`, `log_artifact!` or `log_curve!`; monitors reach the logger
through `info.logger`.

```julia
backend(w::MyWrapper) = w.exp
```
"""
backend(lgr) = lgr
backend(::Nothing) = nothing

"""
    logger_info(lgr) -> NamedTuple

The backend's key identifying parameters, as one plain `NamedTuple`. Informational, like
[`run_id`](@ref): an experiment key and URL, a run id and project, the shipped
[`JSONLogger`](@ref)'s file path, which the `nitro_logger` tool renders. Default `(;)`. Plain
serializable data only, never the live object, since the table crosses tool and report
boundaries. Not stored in the checkpoint record.
"""
logger_info(lgr) = (;)
logger_info(::Nothing) = (;)

"""
    ReactantNitro._adopt_logger!(lgr, dir) -> lgr

Internal: the logger-side half of setup's default wiring. The default [`JSONLogger`](@ref) is
constructed without a path and adopts the run's resolved `run_dir` here, as the default
`TopKCheckpointer` does; a wrapper over the user's logger forwards this, and the identity default
leaves every other logger untouched.
"""
_adopt_logger!(lgr, dir) = lgr
