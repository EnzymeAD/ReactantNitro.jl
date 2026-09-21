# JSONLog.jl
#
# The shipped JSON default logger, the framework's own backend, exactly as `TopKCheckpointer` is
# the shipped checkpointer. JSON Lines, one object per line with a `"type"` discriminator, every
# write flushed, opened in append mode so a resumed run extends the same file (which is why
# `logger_state` is `nothing`). The lazy handle is load-bearing: the default accessor constructs a
# pathless `JSONLogger()` and setup pins its path to the resolved `run_dir`.

"""
    JSONLogger(path = nothing) -> JSONLogger

The framework's default logger: JSON Lines to `path`, defaulting to the run's `metrics.jsonl`. It
is what [`logger`](@ref)`(e)` returns when the experiment declares none, so a bare `Nitro(e)` writes
`runs/<Exp>/metrics.jsonl` beside its checkpoints. Every line carries a `"type"` discriminator:

```json
{"type":"params","seed":42,"max_epochs":40,"width":128}
{"type":"metrics","context":"train","epoch":1,"step":4,"loss":0.31}
{"type":"metrics","context":"validate","epoch":1,"step":160,"val_loss":0.29}
{"type":"metrics","context":"data","epoch":1,"step":160,"data_wait_frac":0.012}
{"type":"other","key":"binding_report","value":"..."}
{"type":"finish","status":"completed"}
```

Append mode, so a resumed run extends the same file. `logger = nothing` is the opt-out.
"""
mutable struct JSONLogger
    path::Union{String, Nothing}
    io::Union{IO, Nothing}
end

JSONLogger(path::AbstractString) = JSONLogger(String(path), nothing)
JSONLogger() = JSONLogger(nothing, nothing)

# Setup's adoption step: a pathless default pins itself to `run_dir`; an explicit path is never
# overwritten.
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

Open (or return) the append handle, and error for a pathless logger that reaches a write without
having been adopted by a `Nitro`, since silently dropping the line is the lossy no-op the contract
forbids.
"""
function _jsonl_open(lgr::JSONLogger)
    lgr.path === nothing && error(
        "ReactantNitro: this `JSONLogger` has no path to write to. Construct one with an \
         explicit path, `JSONLogger(\"runs/x/metrics.jsonl\")`, or pass a pathless one as the \
         run's logger and let setup adopt the run's `run_dir`, which is the default."
    )
    # The logger may be the first writer into the run's directory.
    lgr.io === nothing && (mkpath(dirname(lgr.path)); lgr.io = open(lgr.path, "a"))
    return lgr.io
end

"""
    ReactantNitro._json_value(x) -> JSON-safe x

Coerce a value to what JSON3 serializes: scalars pass through, arrays and tables are walked,
anything else becomes its string. Non-finite floats become `null`; the framework already drops
non-finite metrics, so this only sees them from a params table or a user's `log_other!`.
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

# The carrier fields (`context`, `epoch`, `step`) and `type` are merged last so they win over a
# metric that happens to share a name.

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

# A local file logger has neither a hosted run id nor a URL.
run_id(lgr::JSONLogger) = nothing
run_url(lgr::JSONLogger) = nothing

# Append mode needs no reattachment.
logger_state(lgr::JSONLogger) = nothing

"""
    logger_info(lgr::JSONLogger) -> NamedTuple

The one identifying parameter of a file logger: where the lines land. `path` is the resolved
`metrics.jsonl` path, or `nothing` for a pathless default that setup has not adopted yet.
"""
logger_info(lgr::JSONLogger) = (; path = lgr.path)
