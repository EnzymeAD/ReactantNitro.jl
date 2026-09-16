# JSONLog.jl
#
# The shipped half of the logging layer: the JSON default logger. The contract in Logging.jl stays
# duck-typed and backend-free; this file is where the framework's OWN backend lives, exactly as
# `TopKCheckpointer` is the shipped default checkpointer and RAdam the shipped default optimizer.
# A user's backend for a hosted experiment tracker defines the same ten verbs on its own handle
# and nothing here changes for it.
#
# ── Format: JSON Lines, one object per line ─────────────────────────────────────────
#
# Each line carries a `"type"` discriminator so the file is self-describing and filterable, and
# every write flushes, so a crash loses at most the line being written. The handle opens in
# append mode, which is what makes a resumed run extend the same file rather than truncate it,
# which is why `logger_state` returns `nothing`: a file logger has nothing to reattach, the
# "defines neither `logger_state` nor `reattach!` and keeps working" pattern.
#
# The lazy handle is load-bearing, not an optimization. The default accessor constructs a
# PATHLESS `JSONLogger()` and setup pins its path to the run's resolved `run_dir`, exactly as
# the default checkpointer's `dir = nothing` is pinned; opening at construction would
# force the accessor to open a file it cannot yet name. `_jsonl_open` is where the error lives
# for a pathless logger that reaches a write without ever being adopted.

"""
    JSONLogger(path = nothing) -> JSONLogger

The framework's default logger: writes **JSON Lines** (one JSON object per line) to `path`,
defaulting to the run's `metrics.jsonl`. It is the value [`logger`](@ref)`(e)` returns when the
experiment declares no logger, and setup pins a pathless one to the run's resolved `run_dir`, so
a bare `Nitro(e)` writes `runs/<Exp>/metrics.jsonl` beside its checkpoints.

Every line carries a `"type"` discriminator:

```json
{"type":"params","seed":42,"max_epochs":40,"width":128}
{"type":"metrics","context":"train","epoch":1,"step":4,"loss":0.31}
{"type":"metrics","context":"validate","epoch":1,"step":160,"val_loss":0.29}
{"type":"metrics","context":"data","epoch":1,"step":160,"data_wait_frac":0.012}
{"type":"other","key":"binding_report","value":"..."}
{"type":"finish","status":"completed"}
```

**Append mode, always**, so a resumed run extends the same file rather than truncating it; that
is also why `logger_state` returns `nothing` and no `reattach!` is defined: a ten-line file logger
defines neither and keeps working.

**`logger = nothing` is the documented opt-out**, explicit per run or as an experiment field, and
stays the public "no logging" value with its `::Nothing` no-op methods.
"""
mutable struct JSONLogger
    path::Union{String, Nothing}
    io::Union{IO, Nothing}
end

JSONLogger(path::AbstractString) = JSONLogger(String(path), nothing)
JSONLogger() = JSONLogger(nothing, nothing)

# Setup's adoption step, the JSONLogger half (the identity half lives in Logging.jl). A pathless
# default pins itself to the run's resolved `run_dir`; an explicit path is a user choice and is
# never overwritten, exactly like `TopKCheckpointer.dir`.
function _adopt_logger!(lgr::JSONLogger, dir)
    lgr.path === nothing && (lgr.path = joinpath(String(dir), METRICS_FILE))
    return lgr
end

"""
    ReactantNitro.METRICS_FILE

The default logger's file name inside the run's directory: `metrics.jsonl`.
"""
const METRICS_FILE = "metrics.jsonl"

"""
    ReactantNitro._jsonl_open(lgr::JSONLogger) -> IO

Open (or return) the append handle, and name the one way a pathless logger reaches a write
without being adopted: constructed directly as `JSONLogger()` and logged through without ever
passing through a `Nitro`. Loud on purpose, since silently dropping the line would be the
lossy no-op the logging contract forbids.
"""
function _jsonl_open(lgr::JSONLogger)
    lgr.path === nothing && error(
        "ReactantNitro: this `JSONLogger` has no path to write to. Construct one with an \
         explicit path, `JSONLogger(\"runs/x/metrics.jsonl\")`, or pass a pathless one as the \
         run's logger and let setup adopt the run's `run_dir`, which is the default."
    )
    # The logger may be the FIRST writer into the run's directory: the checkpointer creates it
    # lazily at save time, and an evaluation handle never saves. `mkpath` on a bare filename's
    # "." parent is a no-op, so an explicit file path works the same either way.
    lgr.io === nothing && (mkpath(dirname(lgr.path)); lgr.io = open(lgr.path, "a"))
    return lgr.io
end

"""
    ReactantNitro._json_value(x) -> JSON-safe x

Coerce a value to the closed set JSON3 serializes without complaint: scalars pass through, arrays
and tables are walked, and anything else becomes its string. Non-finite floats become `null`,
which is the honest JSON for "not a number" and the one way a stray `NaN` param must not crash
the run; the framework already drops non-finite METRICS before calling (finite_only), so this
only ever sees them from a params table or a user's own `log_other!`.
"""
_json_value(x::Union{AbstractString, Bool, Nothing, Integer}) = x
_json_value(x::AbstractFloat) = isfinite(x) ? x : nothing
_json_value(x::Symbol) = string(x)
# A matrix becomes row-major nested arrays, the standard JSON shape for a confusion matrix;
# the generic array method would otherwise leave the nesting to the writer's column-major walk.
_json_value(x::AbstractMatrix) = [map(_json_value, row) for row in eachrow(x)]
_json_value(x::AbstractArray) = map(_json_value, x)
_json_value(x::AbstractDict) = Dict(k => _json_value(v) for (k, v) in x)
_json_value(x::NamedTuple) = map(_json_value, x)
_json_value(x) = string(x)

"""
    ReactantNitro._jsonl_write(lgr::JSONLogger, obj) -> nothing

Serialize `obj` as one JSON line and flush. `obj` is a `NamedTuple`; the `type` discriminator is
the caller's responsibility so each verb names its own kind.
"""
function _jsonl_write(lgr::JSONLogger, obj)
    io = _jsonl_open(lgr)
    JSON3.write(io, _json_value(obj))
    write(io, '\n')
    flush(io)
    return nothing
end

# ── The ten verbs ────────────────────────────────────────────────────────────────────
#
# The carrier fields (`context`, `epoch`, `step`) are merged LAST so they win over a metric that
# happens to share a name, and the `type` discriminator is part of that carrier. A user metric
# named `step` would otherwise rewrite the axis the line is filed under, which is exactly the
# silent corruption a schema should refuse.

function log_metrics!(lgr::JSONLogger, metrics; context = "train", epoch, step, kwargs...)
    _jsonl_write(lgr, merge(metrics, (; type = "metrics", context, epoch, step)))
    return nothing
end

function log_params!(lgr::JSONLogger, params)
    _jsonl_write(lgr, merge(params, (; type = "params")))
    return nothing
end

function log_tags!(lgr::JSONLogger, tags)
    _jsonl_write(lgr, merge(tags, (; type = "tags")))
    return nothing
end

function log_other!(lgr::JSONLogger, key, value)
    _jsonl_write(lgr, (; type = "other", key = key, value = _json_value(value)))
    return nothing
end

function log_confusion!(lgr::JSONLogger, matrix, labels; epoch, kwargs...)
    _jsonl_write(lgr, (; type = "confusion", epoch = epoch, matrix = matrix, labels = labels))
    return nothing
end

function finish!(lgr::JSONLogger, status)
    _jsonl_write(lgr, (; type = "finish", status = status))
    lgr.io === nothing || (close(lgr.io); lgr.io = nothing)
    return nothing
end

# A local file logger has neither a hosted run id nor a URL; [`logger_info`](@ref) is where its
# one identifying parameter lives.
run_id(lgr::JSONLogger) = nothing
run_url(lgr::JSONLogger) = nothing

# Append mode needs no reattachment: the path is already right and the file already extends. The
# pair is self-describing, so returning `nothing` here is what keeps `reattach!` undefined.
logger_state(lgr::JSONLogger) = nothing

"""
    logger_info(lgr::JSONLogger) -> NamedTuple

The one identifying parameter of a file logger: where the lines land. `path` is the resolved
`metrics.jsonl` path, or `nothing` for a pathless default that setup has not adopted yet.
"""
logger_info(lgr::JSONLogger) = (; path = lgr.path)
