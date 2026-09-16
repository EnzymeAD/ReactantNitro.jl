# Logging.jl
#
# The logging contract: ten verbs, their `::Nothing` methods, and the two defaulted accessors.
#
# No community standard exists. Backends subtype the stdlib `AbstractLogger` and interoperate only
# via `LoggingExtras.TeeLogger`, each reinventing its own step counter. That pattern fits badly here:
# the driver's genuine `@info` traffic would share the metric channel, and metric names are dynamic.
#
# DUCK-TYPED, NO MANDATORY SUPERTYPE, and `nothing` is the public "no logging" value with a
# `::Nothing` method for all ten verbs. A MISSING METHOD IS A LOUD `MethodError`, BY DESIGN:
# optional-no-op contracts make wrappers silently lossy, so silence must be opted into explicitly,
# per logger, per verb.
#
# THE FRAMEWORK SHIPS ONE BACKEND, the JSON default in JSONLog.jl, so a run that names no logger
# still leaves a machine-readable record. Everything else about the contract stands: a user's own
# logger defines the methods in their own code, with no extension and no supertype; common public
# loggers get an extension here with the logger as a weakdep; a backend's own package may
# equally define them itself. Both directions are legal and neither is type piracy. DEFINE METHODS
# ONLY, NO TYPES, IN EXTENSIONS: a type defined inside an extension cannot be referenced by the
# parent package and forces users through `Base.get_extension`.
#
# Contract details, each of which has bitten something:
#   * `kwargs...` on `log_metrics!`, so a future axis does not break every backend
#   * the framework drops non-finite values before calling
#   * `log_confusion!` takes a plain `Matrix{Int}`; the backend adapts it
#   * keep `context` values byte-identical to existing conventions ("train", "validate", "data")
#   * keep flat parameter key style; dotted keys split comparison tables
#   * log parameters BEFORE the first compile, since a compile-time crash would
#     otherwise lose them
#   * do NOT map `Base.close` to `finish!(:completed)`: in a `finally` it runs after the driver's
#     `finish!(:error)` and would stamp every failed run completed
#
# Train metrics carry `step`; validation metrics carry `epoch` plus the current step so the two
# overlay. The step counter lives in the driver, not the logger, which is what makes the contract
# stateless and the whole duck-typed design possible.

"""
    log_metrics!(lgr, metrics; step, epoch, context, kwargs...) -> nothing

The metric channel. `context` is `"train"`, `"validate"`, or `"data"`, byte-identical to existing
conventions. The framework drops non-finite values before calling.

**`"data"` is the third one and it is deliberately not either of the first two.** It carries the
data path's per-epoch `data_wait_frac`, and the two established contexts are each pinned to a
cadence that a diagnostic would falsify: a `"train"` line is exactly one per **optimizer step** and
a `"validate"` line is exactly one per **epoch** of metrics. A per-epoch fact about the host data
path is a third thing, so it says so instead of diluting either count. A backend that passes the
string through, which is all the contract asks, needs no change for it.
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

Takes a plain `Matrix{Int}`; the backend adapts it.

**THE FRAMEWORK NEVER CALLS THIS. It is for your code.** Every other verb in the contract fires
from the driver; this one cannot, because a confusion matrix is not something the framework has. It
is a user metric, accumulated by the `count === nothing` rule, and what reaches a phase monitor
is the *finalized* scalars rather than the raw accumulator, so only the experiment knows both that a
matrix exists and what its class labels mean.

The shape that works, and the one the first port uses: accumulate it in
[`metrics`](@ref) with a `nothing` count, stash it from [`finalize_metrics`](@ref), and emit it from
a [`register_phase_monitor!`](@ref) on the transition out of `EvalStepping`, which is exactly when a
fresh one exists.

```julia
metrics(e, out; y)            = (; confusion = (confusion_matrix(out, y), nothing))
finalize_metrics(e, acc, spl) = (e.rt.last_confusion = acc.confusion; derived_scalars(acc))
# then, in a phase monitor:
log_confusion!(info.logger, e.rt.last_confusion, e.rt.class_names; epoch)
```

The verb is declared here, with the `::Nothing` no-op, so that a backend implements it once and every
experiment can reach it. That is its whole job.
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

**Informational**, for humans and for reports. It stays in the checkpoint record so a checkpoint can
be traced back to its experiment without deserializing anything, which is why the `::Nothing` method
cannot be omitted.
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

**Machine-readable resumption state**, opaque to the framework and backend-specific. Default
`nothing`, meaning "I have no resumable state", in which case nothing is stored and no reattachment
is attempted.

Most backends can continue an existing experiment and each needs different information to do it: a
hosted experiment key, a W&B run id plus project and entity, an MLFlow run id plus a tracking URI,
a directory for a file logger. The framework must not know the shape of that, so it round-trips it
opaquely: `logger_state` on checkpoint, [`reattach!`](@ref) on resume.

**It must return plain serializable data**, a `String`, `NamedTuple`, or `Dict`, **never the live
backend object**. The record goes through JLD2 and outlives the process, so a stored handle would be
either unserializable or dead on arrival. This is the one contract detail a backend author is likely
to get wrong, so the error names it.

**The pair is self-describing**, which is how it stays loud without burdening simple loggers: if this
returns anything but `nothing`, [`reattach!`](@ref) is **required** and a missing method is the usual
`MethodError`. A ten-line file logger defines neither and keeps working.

**This is separate from [`run_id`](@ref) / [`run_url`](@ref), deliberately.** Those are
informational and displayed; this is for machines and never is. An earlier draft used `run_id` for
both jobs, which works only for backends whose entire resumption state happens to be one identifier.
"""
logger_state(lgr) = nothing
logger_state(::Nothing) = nothing

"""
    reattach!(lgr, state) -> nothing

Restore a logger onto its previous experiment, called after the user constructs their logger and
**before the run starts and before any metric is logged**.

**Required if and only if [`logger_state`](@ref) returns non-`nothing`**; there is deliberately no
default method, because a logger that claims resumable state and cannot restore it is a bug rather
than a configuration.

**Type mismatch is caught**: the record stores the logger's type name alongside its state, and
resuming into a different logger type refuses with both names rather than handing one backend's
state to another's logger.
"""
function reattach! end
reattach!(::Nothing, state) = nothing

"""
    backend(lgr) -> Any

Reach the native backend object. Default is the identity, because **the default is that no
unwrapping is needed: the logger a user passes IS their backend object.** Pass an experiment
tracker's own run handle, the extension supplies `log_metrics!` for it, and every other client
function in that package stays callable on the same handle.

```julia
backend(w::MyWrapper) = w.exp
```

**This escape hatch is what makes a small interface defensible rather than limiting**, so the two
decisions stand or fall together: `log_image!`, `log_artifact!`, `log_curve!`, and `log_text!` are
dropped precisely because anyone needing them reaches the backend directly. Monitors reach the
logger through `info.logger`, so logging something custom at validation time needs no
dedicated hook.
"""
backend(lgr) = lgr
backend(::Nothing) = nothing

"""
    logger_info(lgr) -> NamedTuple

**Informational**, for humans and for tools, like [`run_id`](@ref) / [`run_url`](@ref): the
backend's key identifying parameters, as one plain `NamedTuple`. A hosted-tracker logger returns
its experiment key, URL, workspace, and project; a W&B logger its run id, URL, project, and entity;
the shipped [`JSONLogger`](@ref) its file path. The ReactantNitroKaimonGateExt `nitro_logger` tool
renders exactly this table for a running experiment.

The default is `(;)`, and that is the whole contract: a backend that has nothing to say says
nothing, and one that does implements this one small method. The docstring convention is to include
the [`run_id`](@ref) / [`run_url`](@ref) values under those names when the backend has them, so a
tool rendering the table shows the same two canonical identifiers every hosted logger reports.

**It must return plain serializable data**, a `String`, `Number`, `Bool`, `nothing`, or
arrays/`NamedTuple`s of those, **never the live backend object**. This is the same rule as
[`logger_state`](@ref), for the same reason: the table is meant to cross boundaries (a tool reply,
a report), and a live handle is useless there. Unlike `logger_state` this is **not** stored in the
checkpoint record: it is live information about the running experiment, so the record keeps only
the two identifiers that existed before this verb.
"""
logger_info(lgr) = (;)
logger_info(::Nothing) = (;)

"""
    ReactantNitro._adopt_logger!(lgr, dir) -> lgr

Internal, and the logger-side half of setup's default wiring. The framework's own default logger,
[`JSONLogger`](@ref), is constructed without a path and adopts the run's resolved `run_dir` at
setup, exactly as the default `TopKCheckpointer` adopts it (its `dir` field defaults to `nothing`
and is pinned the same way). A wrapper that tee- or records over the user's logger must forward
this so the wrapped default still lands in the run's directory; the identity default here is what
keeps every other logger untouched.
"""
_adopt_logger!(lgr, dir) = lgr
