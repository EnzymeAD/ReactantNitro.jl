# ReactantNitroTensorBoardLoggerExt.jl
#
# The TensorBoard backend for the logging contract: the ten verbs on a `TBLogger`.
#
# TensorBoardLogger keeps its own global step, incremented by every `@info` that reaches it.
# Nothing here goes through `handle_message`: `log_value` and `log_text` take an explicit `step`,
# so the driver's step is the only one in the file. Keys are prefixed with the contract's
# `context`, so `train/loss` and `validate/val_loss` are separate tags; `step` is the x axis for
# every context, and the epoch rides along as `<context>/epoch`.
#
# TensorBoard's HParams dashboard wants the hyperparameters and the metric tags in one call, and the
# contract logs parameters before any metric name exists. So `log_params!` writes the parameters
# immediately as a text summary and stashes them, `log_metrics!` records its tags, and `finish!`
# calls `write_hparams!` with both. The stash is a `WeakKeyDict` keyed on the logger.

module ReactantNitroTensorBoardLoggerExt

import ReactantNitro
import TensorBoardLogger

using TensorBoardLogger: TBLogger, log_text, log_value, write_hparams!

# Per-logger state: the stashed hyperparameters, the tags seen, and the last step logged at, so
# the verbs carrying no step of their own land where the run was.
const _PENDING = WeakKeyDict{
    TBLogger,
    @NamedTuple{hparams::Dict{String, Any}, tags::Set{String}, step::Base.RefValue{Int}}
}()

_pending(lg::TBLogger) = get!(
    () -> (; hparams = Dict{String, Any}(), tags = Set{String}(), step = Ref(0)),
    _PENDING, lg
)

"""
    ReactantNitro.log_metrics!(lgr::TBLogger, metrics; step, epoch, context, kwargs...) -> nothing

One scalar summary per metric, tagged `<context>/<name>` at `step`, plus `<context>/epoch`. A
non-`Real` value is an error naming the keys, since quietly dropping a metric is the lossy no-op
the contract forbids.
"""
function ReactantNitro.log_metrics!(lgr::TBLogger, metrics; step, epoch, context = "train", kwargs...)
    st = _pending(lgr)
    st.step[] = Int(step)
    rejected = String[]
    for (k, v) in pairs(metrics)
        if v isa Real
            _tagged(lgr, st, "$context/$(String(k))", v, Int(step))
        else
            push!(rejected, "$(String(k))::$(typeof(v))")
        end
    end
    isempty(rejected) || error(
        """
        ReactantNitro: the TensorBoard backend can only record `Real` metric values, and this \
        `log_metrics!` carried $(join(rejected, ", ")) in context "$context".
        A TensorBoard scalar summary is a number against an integer step. Anything else has its
        own TensorBoardLogger verb: `backend(lgr)` is the `TBLogger` itself, so `log_image`,
        `log_histogram` and `log_embeddings` are all directly callable on it."""
    )
    _tagged(lgr, st, "$context/epoch", epoch, Int(step))
    return nothing
end

# Every scalar goes through here so the tag is recorded for `write_hparams!` as it is written.
function _tagged(lg::TBLogger, st, name::AbstractString, value::Real, step::Int)
    push!(st.tags, String(name))
    log_value(lg, String(name), value; step = step)
    return nothing
end

"""
    ReactantNitro.log_params!(lgr::TBLogger, params) -> nothing

Writes the parameters twice: a text summary at step 0 immediately, so a compile-time crash still
leaves them in the event file, and a stashed HParams record that [`finish!`](@ref) writes once the
metric tags exist. Values are coerced to the `String`, `Bool` and `Real` the plugin accepts.
"""
function ReactantNitro.log_params!(lgr::TBLogger, params)
    st = _pending(lgr)
    for (k, v) in pairs(params)
        st.hparams[String(k)] = _hparam(v)
    end
    log_text(lgr, "params", _Markdown(_markdown_pairs(params)); step = 0)
    return nothing
end

# `write_hparams!` asserts on anything outside `String`, `Bool` and `Real`, so a `Symbol` preset
# name or a `nothing` becomes its string here rather than an `AssertionError` at finish time.
_hparam(v::Union{AbstractString, Bool, Real}) = v isa AbstractString ? String(v) : v
_hparam(v) = string(v)

"""
    ReactantNitro.log_tags!(lgr::TBLogger, tags) -> nothing

TensorBoard has no tag concept, so tags become a text summary under `tags`.
"""
function ReactantNitro.log_tags!(lgr::TBLogger, tags::Union{NamedTuple, AbstractDict})
    log_text(lgr, "tags", _Markdown(_markdown_pairs(tags)); step = _pending(lgr).step[])
    return nothing
end

function ReactantNitro.log_tags!(lgr::TBLogger, tags::AbstractString)
    log_text(lgr, "tags", _Markdown(tags); step = _pending(lgr).step[])
    return nothing
end

function ReactantNitro.log_tags!(lgr::TBLogger, tags)
    log_text(lgr, "tags", _Markdown(join(("- $(t)" for t in tags), "\n")); step = _pending(lgr).step[])
    return nothing
end

"""
    ReactantNitro.log_other!(lgr::TBLogger, key, value) -> nothing

A scalar summary under `other/<key>` for a number, a text summary otherwise; the binding report
renders as markdown.
"""
function ReactantNitro.log_other!(lgr::TBLogger, key, value)
    st = _pending(lgr)
    if value isa Real
        _tagged(lgr, st, "other/$(String(key))", value, st.step[])
    else
        log_text(lgr, "other/$(String(key))", _Markdown(string(value)); step = st.step[])
    end
    return nothing
end

"""
    ReactantNitro.log_confusion!(lgr::TBLogger, matrix, labels; epoch, step, kwargs...) -> nothing

A markdown table as a text summary under `confusion`, rows true and columns predicted. `step`
defaults to the last step a metric was logged at. Built as a string here, since
TensorBoardLogger's own 2-D path serializes a non-square matrix with its shape and contents
disagreeing.
"""
function ReactantNitro.log_confusion!(
        lgr::TBLogger, matrix, labels; epoch, step = nothing, kwargs...
    )
    st = _pending(lgr)
    at = step === nothing ? st.step[] : Int(step)
    log_text(lgr, "confusion", _Markdown(_markdown_matrix(matrix, labels, epoch)); step = at)
    return nothing
end

"""
    ReactantNitro.finish!(lgr::TBLogger, status) -> nothing

Writes the stashed HParams record with the metric tags the run produced, then the status under
`run/status`. It does not close the logger: a `TBLogger` flushes on every event, and a second
`train!` on the same handle is legitimate.
"""
function ReactantNitro.finish!(lgr::TBLogger, status)
    st = _pending(lgr)
    # `write_hparams!` mutates the dictionary it is given, casting every `Real` to `Float64`, so
    # it gets a copy and the stash survives a second `train!` on this logger.
    isempty(st.hparams) || write_hparams!(lgr, copy(st.hparams), sort!(collect(st.tags)))
    log_text(lgr, "run/status", _Markdown(string(status)); step = st.step[])
    return nothing
end

# ── The informational and resumption verbs ───────────────────────────────────────────

# A local event-file backend has neither a hosted run id nor a URL.
ReactantNitro.run_id(lgr::TBLogger) = nothing
ReactantNitro.run_url(lgr::TBLogger) = nothing

"""
    ReactantNitro.logger_state(lgr::TBLogger) -> NamedTuple

The log directory. Not the "nothing to reattach" case of the shipped `JSONLogger`: `TBLogger`'s
default init policy is `tb_increment`, which silently writes to `<logdir>_1` over an existing
directory, splitting one run's curves across two entries on a resume. Storing the directory is
what lets [`reattach!`](@ref) catch that.
"""
ReactantNitro.logger_state(lgr::TBLogger) = (; logdir = TensorBoardLogger.logdir(lgr))

"""
    ReactantNitro.reattach!(lgr::TBLogger, state) -> nothing

Checks that this run writes where the checkpoint's run wrote, and refuses when it does not. It
checks rather than relocates, since moving a live `TBLogger` means reopening event files behind its
back; the supported way to continue a run is `TBLogger(dir, tb_append)`.
"""
function ReactantNitro.reattach!(lgr::TBLogger, state)
    here = TensorBoardLogger.logdir(lgr)
    here == state.logdir && return nothing
    error(
        """
        ReactantNitro: this run's `TBLogger` writes to
            $(here)
        but the checkpoint it resumes was written by one at
            $(state.logdir)
        so the resumed run would appear in TensorBoard as a second, separate run.

        The usual cause is the default init policy. `TBLogger(dir)` is `tb_increment`, which
        quietly becomes `dir_1` when `dir` already exists, and a resume always finds it existing.
        Construct the logger with `TBLogger("$(state.logdir)", tb_append)` to extend the run, or
        pass `resume = nothing` if a separate run is what you meant."""
    )
end

"""
    ReactantNitro.logger_info(lgr::TBLogger) -> NamedTuple

The event directory, plus the logger's own step counter, which stays put while nothing logs
through the stdlib `AbstractLogger` path.
"""
ReactantNitro.logger_info(lgr::TBLogger) = (;
    logdir = TensorBoardLogger.logdir(lgr),
    step = TensorBoardLogger.step(lgr),
)

# ── Markdown, which is how TensorBoard renders a text summary ────────────────────────
#
# For a `String`, TensorBoardLogger renders `repr(MIME"text/plain"(), s)`, which quotes and
# escapes it, so a multi-line report would arrive as one line with literal `\n`. `_Markdown` says
# the string is already markdown. A type inside an extension, which the contract otherwise rules
# out, and harmless because nothing outside this file ever names or receives one.
struct _Markdown
    text::String
end

Base.show(io::IO, ::MIME"text/markdown", m::_Markdown) = print(io, m.text)

_markdown_pairs(kv) = join(("- **$(String(k))**: $(v)" for (k, v) in pairs(kv)), "\n")

# A markdown table needs one label per row and per column, so a mismatch is unbuildable.
function _markdown_matrix(matrix, labels, epoch)
    names = [string(l) for l in labels]
    size(matrix, 1) == size(matrix, 2) == length(names) || error(
        """
        ReactantNitro: `log_confusion!` got a $(size(matrix, 1))x$(size(matrix, 2)) matrix and \
        $(length(names)) labels, which cannot be rendered as a table. A confusion matrix is
        square and carries one label per class."""
    )
    io = IOBuffer()
    println(io, "**epoch $(epoch)**, rows are true and columns are predicted\n")
    println(io, "| | ", join(names, " | "), " |")
    println(io, "|", repeat(" --- |", length(names) + 1))
    for (i, row) in enumerate(eachrow(matrix))
        println(io, "| **", names[i], "** | ", join(row, " | "), " |")
    end
    return String(take!(io))
end

end # module
