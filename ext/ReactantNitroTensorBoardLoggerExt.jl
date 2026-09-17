# ReactantNitroTensorBoardLoggerExt.jl
#
# The TensorBoard backend for the logging contract: the ten verbs on a `TBLogger`.
#
# ── The step counter stays where the contract put it ─────────────────────────────────
#
# TensorBoardLogger is an `AbstractLogger` with its own global step, incremented by every
# `@info` that reaches it. That is the pattern the contract in Logging.jl was written against
# and away from: the driver owns the step, and a backend that counts its own would disagree with
# the checkpoint record the moment anything else logged.
#
# Nothing here goes through `handle_message`. `log_value` and `log_text` take an explicit
# `step`, and only `handle_message` touches `lg.global_step`, so a logger driven purely through
# this file never moves its own counter and the driver's step is the only one in the file. A user
# who ALSO uses the logger as a stdlib `AbstractLogger` gets both, which is fine and is exactly
# what `backend(lgr)` returning the logger is for.
#
# ── Names and the axis ───────────────────────────────────────────────────────────────
#
# Keys are prefixed with the contract's `context`, so `train/loss` and `validate/val_loss` are
# separate tags that TensorBoard groups under collapsible sections. Overlaying the two in ONE
# chart is a TensorBoard "runs" question, not a tag question: point two `TBLogger`s at
# `<dir>/train` and `<dir>/validate` if that is what you want.
#
# `step` is the x axis for all three contexts, which is what the contract means by train and
# validation metrics each carrying the other's counter. The epoch rides along as
# `<context>/epoch` so a curve can be read back against epoch without the run directory.
#
# ── Why this file keeps a little state ───────────────────────────────────────────────
#
# TensorBoard's HParams dashboard is the reason. `write_hparams!` takes the hyperparameters AND
# the list of metric tags the comparison table should carry columns for, and it wants them in one
# call. The contract logs parameters BEFORE the first compile, which is before any metric name
# exists, so the two halves are simply not available at the same moment.
#
# So `log_params!` writes the parameters immediately as a text summary, which is what makes a
# compile-time crash still leave them in the event file, and stashes them; `log_metrics!` records
# the tags it emits; `finish!` calls `write_hparams!` once with both halves. The stash is a
# `WeakKeyDict` keyed on the logger, so two concurrent runs never share one and a dropped logger
# does not hold its entry alive.

module ReactantNitroTensorBoardLoggerExt

import ReactantNitro
import TensorBoardLogger

using TensorBoardLogger: TBLogger, log_text, log_value, write_hparams!

# Per-logger state: the stashed hyperparameters, the metric tags seen so far, and the last step
# logged at. The last step exists so the verbs that carry no step of their own (`finish!`, and
# `log_confusion!`, whose signature has only an epoch) land where the run actually was rather
# than at the logger's untouched zero.
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

One scalar summary per metric, tagged `<context>/<name>` at `step`, plus `<context>/epoch`.

Non-`Real` values are a loud error naming the keys. A TensorBoard scalar is a `Float32` against
an integer step, and quietly dropping a metric the experiment asked to record is the lossy no-op
the contract forbids. The framework has already dropped non-finite values before this is reached.
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

# Every scalar goes through here so the tag is recorded for `write_hparams!` at the same moment
# it is written, and the two can never drift apart.
function _tagged(lg::TBLogger, st, name::AbstractString, value::Real, step::Int)
    push!(st.tags, String(name))
    log_value(lg, String(name), value; step = step)
    return nothing
end

"""
    ReactantNitro.log_params!(lgr::TBLogger, params) -> nothing

Writes the parameters twice, and both writes matter.

The **text summary** `params` goes in immediately, at step 0, which is what leaves them in the
event file when a compile-time crash follows. The contract puts this call before the first
compile for exactly that reason, so the backend must not defer the only copy.

The **HParams record** is stashed and written by [`finish!`](@ref), because `write_hparams!`
wants the hyperparameters and the metric tags in one call and no metric tag exists yet. Values
are coerced to the `String`, `Bool` and `Real` that plugin accepts.
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

A scalar summary under `other/<key>` when the value is a number, a text summary otherwise. The
schedule binding report arrives here and is multi-line; TensorBoard renders a text summary as
markdown, so it stays readable.
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

A markdown table as a text summary under `confusion`, rows indexed by true label and columns by
predicted. `step` defaults to the last step a metric was logged at, which for the intended call
site (a phase monitor leaving `EvalStepping`) is the step that epoch's validation just wrote.

**The table is built here as a string rather than handed to `log_text` as a matrix.**
TensorBoardLogger's own 2-D path computes a transposed copy and then serializes the original,
so a non-square matrix comes back with its shape and its contents disagreeing.
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

Writes the stashed HParams record with the metric tags the run actually produced, then the
status as a text summary under `run/status`.

**It does not close the logger**, deliberately, and the contract's own warning is the near miss
here: a `TBLogger` flushes on every event, so closing would protect nothing, while a second
`train!` on the same handle is a legitimate thing to do and would land on a closed stream. The
file handle belongs to whoever built the logger, and `TBLogger` finalizes itself.
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

# A local event-file backend has neither a hosted run id nor a URL, the same position the shipped
# `JSONLogger` is in; `logger_info` is where its one identifying parameter lives.
ReactantNitro.run_id(lgr::TBLogger) = nothing
ReactantNitro.run_url(lgr::TBLogger) = nothing

"""
    ReactantNitro.logger_state(lgr::TBLogger) -> NamedTuple

The log directory, which is all a TensorBoard run is.

This is **not** the "file logger has nothing to reattach" case that lets the shipped `JSONLogger`
return `nothing`. `TBLogger`'s default init policy is `tb_increment`: constructing one over an
existing directory silently writes to `<logdir>_1` instead, which on a resume splits one run's
curves across two entries in TensorBoard's run list with no error anywhere. Storing the directory
is what lets [`reattach!`](@ref) catch that.
"""
ReactantNitro.logger_state(lgr::TBLogger) = (; logdir = TensorBoardLogger.logdir(lgr))

"""
    ReactantNitro.reattach!(lgr::TBLogger, state) -> nothing

Checks that this run is writing where the checkpoint's run wrote, and refuses loudly when it is
not.

**It checks rather than relocates.** Moving a live `TBLogger` to another directory means closing
its handles and reopening event files behind its back, through internals it does not expose; the
supported way to continue a run is `TBLogger(dir, tb_append)`, which is the user's call to make.
The one thing this can do without reaching into the backend is refuse to let a resume scatter its
history silently, which is the failure that actually happens.
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

The one identifying parameter of an event-file backend: the directory the events land in, plus
the logger's own step counter, which stays at its initial value for as long as nothing logs
through the stdlib `AbstractLogger` path.
"""
ReactantNitro.logger_info(lgr::TBLogger) = (;
    logdir = TensorBoardLogger.logdir(lgr),
    step = TensorBoardLogger.step(lgr),
)

# ── Markdown, which is how TensorBoard renders a text summary ────────────────────────
#
# TensorBoardLogger renders a text summary by asking the value for its best markdown, HTML or
# plain representation, and for a `String` the winner is `repr(MIME"text/plain"(), s)`, which
# QUOTES AND ESCAPES it. A multi-line binding report would arrive in TensorBoard as one line with
# literal `\n` in it, and a markdown table would stop being a table. `_Markdown` is the smallest thing
# that says "this string is already markdown".
#
# This is a type inside an extension, which the contract otherwise rules out, and the reason the
# rule does not bite is what the rule is FOR: a type here cannot be named from ReactantNitro and
# would push users through `Base.get_extension`. Nothing outside this file ever constructs, names
# or receives one. It goes into a `log_text` call and dies there.
struct _Markdown
    text::String
end

Base.show(io::IO, ::MIME"text/markdown", m::_Markdown) = print(io, m.text)

_markdown_pairs(kv) = join(("- **$(String(k))**: $(v)" for (k, v) in pairs(kv)), "\n")

# A markdown table needs one label per row AND one per column, so a mismatch is unbuildable
# rather than merely ugly. The shipped JSON logger writes the matrix and the labels side by side
# and lets a reader notice; a renderer has to say so.
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
