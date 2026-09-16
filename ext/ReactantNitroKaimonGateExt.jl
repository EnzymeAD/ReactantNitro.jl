# ReactantNitroKaimonGateExt.jl
#
# The KaimonGate extension: registers `nitro_*` GateTools with the running Kaimon gate so an
# agent can drive training, validation, evaluation, prediction, and export from the very session
# where the model code is loaded. It is a dev tool, not a framework surface: nothing in `src/`
# knows it exists, and the tools it registers are the agent's interface to entry points the
# framework already ships (`train!`, `validate`, `evaluate`, `predict`, `export_model`).
#
# ── Why runs are background tasks ────────────────────────────────────────────────────
#
# Kaimon's agent-side tool calls carry a HARD deadline (`_call_session_tool_async` in Kaimon's
# gate client defaults to 5 minutes, and `tool_progress` messages do not extend it), and its eval
# path fails after 10 minutes without output. A training run lasts hours, so no tool call may
# block until the work finishes. Every tool here therefore does the same two things: start the
# work on a background task and return a run id, and let the agent poll `nitro_status` or
# `nitro_runs`, each a fast call, until the run completes.
#
# What the registry cannot survive is a session restart, and it does not pretend to: the
# framework's own checkpoint machinery is the restart story (`resume = :auto`), and the docs page
# says so. A restarted session simply has no in-flight runs, exactly as it has no in-flight evals.
#
# ── Registration ──────────────────────────────────────────────────────────────────────
#
# KaimonGate exposes one tool-registration verb, `serve(tools = [...])`, and a running gate's
# tool list is REPLACED by that call; Kaimon's health checker re-imports the tools from the next
# pong and sends `tools/list_changed`. The extension therefore registers on load, then keeps
# retrying for a short window to cover the race with Kaimon's own `@async KaimonGate.serve(...)`
# at session boot. `ReactantNitroKaimonGateExt.reinstall_kaimon_tools()` is the manual hammer: it registers
# unconditionally, starting a gate if none is running, exactly as a bare `serve()` would.
#
# Outside a gate the extension is a quiet no-op: the automatic path probes the gate's running
# flag and never starts a gate itself, and `serve()` skips non-interactive processes, so a test
# process that loads the extension pays nothing.

module ReactantNitroKaimonGateExt

import ReactantNitro
import KaimonGate

using ReactantNitro: Nitro, ExportBackend, ReactantServerBundle

# ── The run registry ──────────────────────────────────────────────────────────────────
#
# One entry per tool-launched unit of work. Process-local and lock-guarded, because the
# training task writes it from one thread while the status tools read it from another. Finished
# runs are trimmed oldest-first past a small cap so a long-lived session cannot accumulate them
# forever; a RUNNING run is never trimmed, and neither is a failed run the agent has not yet
# read, which is what the cap's "oldest finished first" ordering gives for free.

mutable struct RunState
    id::String
    kind::Symbol            # :train, :validate, :evaluate, :predict, :export
    experiment::String
    status::Symbol          # :running, :completed, :failed
    phase::String           # latest phase name, e.g. "TrainStepping"; "" before the first
    step::Union{Int, Nothing}
    epoch::Union{Int, Nothing}
    loss::Union{Float64, Nothing}
    train_metrics::Any      # latest "train"-context metrics NamedTuple, or nothing
    val_metrics::Any        # latest "validate"-context metrics NamedTuple, or nothing
    run_dir::Union{String, Nothing}
    error::Union{String, Nothing}
    result::Union{String, Nothing}
    nitro::Union{Nitro, Nothing}
    monitor::Any            # the phase-monitor handle, for unregistration on exit
    started_at::Float64
    finished_at::Union{Float64, Nothing}
end

function RunState(id::String, kind::Symbol, experiment::String)
    return RunState(
        id, kind, experiment, :running, "", nothing, nothing, nothing,
        nothing, nothing, nothing, nothing, nothing, nothing, nothing, time(), nothing,
    )
end

const RUNS = Dict{String, RunState}()
const RUNS_LOCK = ReentrantLock()
const MAX_FINISHED_RUNS = 20

_next_run_id() = bytes2hex(rand(UInt8, 4))

function _new_run(kind::Symbol, experiment::String)
    state = RunState(_next_run_id(), kind, experiment)
    lock(RUNS_LOCK) do
        RUNS[state.id] = state
        _trim_registry_locked!()
    end
    return state
end

function _trim_registry_locked!()
    finished = [s for s in values(RUNS) if s.status !== :running]
    n = length(finished) - MAX_FINISHED_RUNS
    n <= 0 && return nothing
    sort!(finished; by = s -> s.started_at)
    for s in finished[1:n]
        delete!(RUNS, s.id)
    end
    return nothing
end

function _get_run(run_id::AbstractString)
    return lock(RUNS_LOCK) do
        get(RUNS, run_id, nothing)
    end
end

# ── The recording logger ──────────────────────────────────────────────────────────────
#
# The logging contract is duck-typed and `nothing` is the "no logging" value, but `nothing`
# cannot hold state, and the status tools need the latest metrics. This tiny backend keeps them
# in the run state and TEES to the experiment's own logger when the run has one, so a user's
# hosted-tracker backend keeps receiving what it always received. `logger_state` returns
# `nothing`, which keeps `reattach!` optional and the checkpoint record clean.
#
# All ten verbs get a method. The contract's "a MISSING METHOD IS A LOUD MethodError" rule is
# per-verb on purpose, and a future driver call is not a reason to break a running tool.

struct RecordingLogger
    state::RunState
    inner::Any
end

function ReactantNitro.log_metrics!(lgr::RecordingLogger, metrics; context = "train", kwargs...)
    lock(RUNS_LOCK) do
        if context == "train"
            lgr.state.train_metrics = metrics
            hasproperty(metrics, :loss) && (lgr.state.loss = _to_float(metrics.loss))
        elseif context == "validate"
            lgr.state.val_metrics = metrics
        end
    end
    lgr.inner === nothing || ReactantNitro.log_metrics!(lgr.inner, metrics; context, kwargs...)
    return nothing
end

function ReactantNitro.log_params!(lgr::RecordingLogger, params)
    lgr.inner === nothing || ReactantNitro.log_params!(lgr.inner, params)
    return nothing
end

function ReactantNitro.log_tags!(lgr::RecordingLogger, tags)
    lgr.inner === nothing || ReactantNitro.log_tags!(lgr.inner, tags)
    return nothing
end

function ReactantNitro.log_other!(lgr::RecordingLogger, key, value)
    lgr.inner === nothing || ReactantNitro.log_other!(lgr.inner, key, value)
    return nothing
end

function ReactantNitro.log_confusion!(lgr::RecordingLogger, matrix, labels; kwargs...)
    lgr.inner === nothing ||
        ReactantNitro.log_confusion!(lgr.inner, matrix, labels; kwargs...)
    return nothing
end

function ReactantNitro.finish!(lgr::RecordingLogger, reason)
    lgr.inner === nothing || ReactantNitro.finish!(lgr.inner, reason)
    return nothing
end

ReactantNitro.logger_state(lgr::RecordingLogger) = nothing
# Forwarded, not returned empty: the checkpoint record for a tool-launched run should carry the
# real backend's trace-back ids, and the status tools should report them. `applicable` guards the
# two informational accessors because they have no generic default (a ten-line file logger
# defines neither), while `logger_info` needs no guard since its default is `(;)`.
ReactantNitro.run_id(lgr::RecordingLogger) =
    lgr.inner === nothing || !applicable(ReactantNitro.run_id, lgr.inner) ? nothing :
    ReactantNitro.run_id(lgr.inner)
ReactantNitro.run_url(lgr::RecordingLogger) =
    lgr.inner === nothing || !applicable(ReactantNitro.run_url, lgr.inner) ? nothing :
    ReactantNitro.run_url(lgr.inner)
ReactantNitro.logger_info(lgr::RecordingLogger) =
    lgr.inner === nothing ? (;) : ReactantNitro.logger_info(lgr.inner)
ReactantNitro.reattach!(lgr::RecordingLogger, state) = nothing
ReactantNitro.backend(lgr::RecordingLogger) = :recording
# Setup's adoption step, forwarded: the tools may wrap the framework's own default `JSONLogger` (a
# pathless one), and setup pins its path through this wrapper exactly as it pins an unwrapped one.
ReactantNitro._adopt_logger!(lgr::RecordingLogger, dir) =
    (lgr.inner === nothing || ReactantNitro._adopt_logger!(lgr.inner, dir); lgr)

_to_float(x::Real) = Float64(x)
_to_float(x::AbstractArray) = isempty(x) ? NaN : Float64(first(x))
_to_float(::Any) = NaN

# ── Experiment resolution ─────────────────────────────────────────────────────────────
#
# The tools name an experiment as a module-qualified type string, e.g. `MyModels.MnistMLP`, or
# as a bare type name the model package exported into Main. Resolution walks loaded bindings
# only; the experiment code is already in this session, which is the whole point of running the
# tools where the model is loaded rather than in a separate process.

function _resolve_experiment(spec::AbstractString)
    parts = split(spec, '.')
    isempty(parts) && error(
        "ReactantNitro ReactantNitroKaimonGateExt: `experiment` must name an experiment type, e.g. \
         `MyModels.MnistMLP`; got an empty string."
    )
    head = Symbol(parts[1])
    mod = if isdefined(Main, head)
        getfield(Main, head)
    else
        found = nothing
        for m in values(Base.loaded_modules)
            if nameof(m) == head
                found = m
                break
            end
        end
        found
    end
    mod === nothing && error(
        "ReactantNitro ReactantNitroKaimonGateExt: cannot resolve `$(spec)`: no module named `$(parts[1])` is \
         bound in Main or loaded as a package. Load the model package first (`using MyModels`), \
         then pass `MyModels.MyExperiment`."
    )
    for part in parts[2:end]
        sym = Symbol(part)
        isdefined(mod, sym) || error(
            "ReactantNitro ReactantNitroKaimonGateExt: `$(spec)` resolves to module `$(nameof(mod))`, which has \
             no binding `$(part)`."
        )
        mod = getfield(mod, sym)
    end
    mod isa Type || error(
        "ReactantNitro ReactantNitroKaimonGateExt: `$(spec)` resolved to `$(mod)`, which is not a type. Pass the \
         name of an `@experiment` struct, e.g. `MyModels.MyExperiment`."
    )
    return mod
end

# ── The overrides parser ──────────────────────────────────────────────────────────────
#
# `overrides` carries experiment-FIELD values, which cannot be declared in the tool schema
# because they are per-experiment. The format is a comma-separated `name=value` list where each
# value is a Julia literal: `width=128`, `smoothing=0.05`, `labels=[1.0,2.0]`, `name="x"`. A
# tiny hand-rolled parser keeps this dependency-free (JSON is no longer a stdlib) and validates
# that every value is a literal, so a typo is a loud error rather than an `eval` of arbitrary
# text. The same parser serves `nitro_predict`'s batch inputs.

function _split_top_level(s::AbstractString, c::Char)
    chars = collect(s)
    parts = String[]
    start = 1
    depth = 0
    in_quote = false
    quote_char = '\0'
    for (i, ch) in enumerate(chars)
        if in_quote
            ch == quote_char && (in_quote = false)
        elseif ch == '"' || ch == '\''
            in_quote = true
            quote_char = ch
        elseif ch == '[' || ch == '(' || ch == '{'
            depth += 1
        elseif ch == ']' || ch == ')' || ch == '}'
            depth -= 1
        elseif ch == c && depth == 0
            push!(parts, join(chars[start:(i - 1)]))
            start = i + 1
        end
    end
    push!(parts, join(chars[start:end]))
    return parts
end

function _first_top_level(s::AbstractString, c::Char)
    depth = 0
    in_quote = false
    quote_char = '\0'
    for (i, ch) in enumerate(collect(s))
        if in_quote
            ch == quote_char && (in_quote = false)
        elseif ch == '"' || ch == '\''
            in_quote = true
            quote_char = ch
        elseif ch == '[' || ch == '(' || ch == '{'
            depth += 1
        elseif ch == ']' || ch == ')' || ch == '}'
            depth -= 1
        elseif ch == c && depth == 0
            return i
        end
    end
    return nothing
end

function _unquote(s::AbstractString)
    inner = s[2:prevind(s, lastindex(s))]
    out = IOBuffer()
    i = firstindex(inner)
    while i <= lastindex(inner)
        ch = inner[i]
        if ch == '\\' && i < lastindex(inner)
            nxt = inner[nextind(inner, i)]
            if nxt == 'n'
                print(out, '\n')
            elseif nxt == 't'
                print(out, '\t')
            elseif nxt == 'r'
                print(out, '\r')
            else
                print(out, nxt)
            end
            i = nextind(inner, nextind(inner, i))
        else
            print(out, ch)
            i = nextind(inner, i)
        end
    end
    return String(take!(out))
end

# A value must be a literal: numbers, strings, booleans, symbols, `nothing`, or a
# vector/tuple/matrix of literals. Anything else (a call, a variable, an assignment) is
# rejected, which is what makes the parser safe to hand arbitrary agent text.

# Pure check, no eval, so a `:row` sub-expression (the space-separated rows of a matrix
# literal) is examined but never evaluated outside its `:vcat`/`:hcat` parent, where it is
# the only place `:row` is legal syntax.
function _is_literal(ex)
    ex isa Union{Number, String, Bool, Char, Symbol, QuoteNode} && return true
    ex === nothing && return true
    ex isa Expr && ex.head in (:vect, :vcat, :hcat, :row, :tuple) &&
        return all(_is_literal, ex.args)
    return false
end

function _literal_value(ex, raw::AbstractString)
    if ex isa QuoteNode
        return ex.value
    elseif ex isa Union{Number, String, Bool, Char, Symbol} || ex === nothing
        return ex
    elseif ex isa Expr && ex.head in (:vect, :vcat, :hcat, :row, :tuple)
        _is_literal(ex) || error(
            "ReactantNitro ReactantNitroKaimonGateExt: value `$(strip(raw))` is not a literal. Use numbers, \
             strings in quotes, booleans, symbols, or vector/tuple/matrix literals, e.g. \
             `width=128`, `smoothing=0.05`, `labels=[1.0,2.0]`."
        )
        return eval(ex)
    end
    error(
        "ReactantNitro ReactantNitroKaimonGateExt: value `$(strip(raw))` is not a literal. Use numbers, \
         strings in quotes, booleans, symbols, or vector/tuple/matrix literals, e.g. \
         `width=128`, `smoothing=0.05`, `labels=[1.0,2.0]`."
    )
end

function _parse_literal(s::AbstractString)
    str = strip(s)
    isempty(str) && error("ReactantNitro ReactantNitroKaimonGateExt: empty value; write `name=value`.")
    (startswith(str, '"') || startswith(str, '\'')) && return _unquote(str)
    ex = Meta.parse(str; raise = true)
    return _literal_value(ex, str)
end

function _parse_overrides(s::Union{AbstractString, Nothing})
    s === nothing && return Dict{Symbol, Any}()
    str = strip(s)
    isempty(str) && return Dict{Symbol, Any}()
    out = Dict{Symbol, Any}()
    for part in _split_top_level(str, ',')
        part = strip(part)
        isempty(part) && continue
        eq = _first_top_level(part, '=')
        eq === nothing && error(
            "ReactantNitro ReactantNitroKaimonGateExt: overrides entry `$(part)` has no `=`; write \
             `name=value` pairs separated by commas."
        )
        name = strip(part[1:(eq - 1)])
        raw = strip(part[(eq + 1):end])
        (isempty(name) || isempty(raw)) && error(
            "ReactantNitro ReactantNitroKaimonGateExt: overrides entry `$(part)` needs both a name and a value."
        )
        out[Symbol(name)] = _parse_literal(raw)
    end
    return out
end

# ── Nitro construction ────────────────────────────────────────────────────────────────
#
# The tools split their keywords into two worlds, mirroring the framework's own split. The typed
# run knobs (max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm, preset, resume,
# checkpoint) go to `Nitro`, where a keyword beats the experiment's own accessor
# for one run. `overrides` carries experiment-FIELD values and goes to the constructor, exactly
# like `from_preset`'s overrides, validated the same way: a key that is not a field is a loud
# error, and a key passed both as a run knob and in `overrides` is an ambiguity error rather
# than a silent winner.

const TOOL_NITRO_KWARGS = (
    :max_epochs, :run_dir, :seed, :n_devs, :accum, :gradient_clip_norm, :preset, :resume,
    :checkpoint,
)

function _check_overrides(::Type{T}, overrides::Dict{Symbol}, kwargs) where {T}
    meta = ReactantNitro.config_metadata(T)
    for (k, v) in overrides
        hasfield(T, k) || error(
            "ReactantNitro ReactantNitroKaimonGateExt: override `$(k)` is not a field of `$(nameof(T))`. Its \
             fields are: $(join(map(string, fieldnames(T)), ", "))."
        )
        any(first(p) == k for p in kwargs) && error(
            "ReactantNitro ReactantNitroKaimonGateExt: `$(k)` was passed both as a run keyword and in \
             `overrides`. Put it in exactly one place."
        )
        declared = hasproperty(meta, k) ? meta[k].type : fieldtype(T, k)
        overrides[k] = convert(declared, v)
    end
    return overrides
end

function _collect_run_kwargs(;
        max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm, preset, resume
    )
    kwargs = Pair{Symbol, Any}[]
    max_epochs === nothing || push!(kwargs, :max_epochs => max_epochs)
    run_dir === nothing || push!(kwargs, :run_dir => run_dir)
    seed === nothing || push!(kwargs, :seed => seed)
    n_devs === nothing || push!(kwargs, :n_devs => n_devs)
    accum === nothing || push!(kwargs, :accum => accum)
    gradient_clip_norm === nothing || push!(kwargs, :gradient_clip_norm => gradient_clip_norm)
    preset === nothing || push!(kwargs, :preset => Symbol(preset))
    resume === nothing || push!(kwargs, :resume => _parse_resume(resume))
    return kwargs
end

# `resume` mirrors the constructor's own three-way split: the symbol `:auto` (the default, find
# the latest checkpoint in run_dir), `false` (start over), or a checkpoint path.
_parse_resume(s::AbstractString) = s == "auto" ? :auto : s == "false" ? false : String(s)

# `nitro_export`'s `data`, which is the one invocation keyword that lets an export SKIP a setup
# step, and the last one this tool learned to express.
#
# `export_model`'s own docstring writes the export construction as
# `Nitro(e; checkpoint = path, data = (;))` and says in as many words that `data = (;)` is not a
# workaround: setup reads weights and traces a graph, and the training data is not part of either.
# This tool built `Nitro(e; kwargs...)` with no `data` at all, so `build_data` always ran, and a
# model whose exportable handle is a DIFFERENT BUILD from its trainable one could not be exported
# through the tool in either direction. The worked case is a model with an `export_inference` flag:
# with the flag set, `build_data` refuses the handle because an inference model cannot train; with
# the flag clear, the first export hook refuses it because the program is teacher-forced and wants
# ground truth on the wire. A pincer like that pushes exports off the tool and onto hand-written
# `export_model` calls, including for the models that never had the problem, which is why the
# keyword exists rather than being left to the caller.
#
# NOT BUILDING IS THE DEFAULT on the `experiment` path, which is the change of behavior here and is
# deliberate. It is also what makes the export stop starting a data server it only has to tear down
# again (`_finish_export_nitro!` exists for that). `data = "build"` is the escape hatch, for the one
# case that needs it: `derive` is skipped on a `checkpoint =` construction (setup takes the
# derived `Device` values from the record instead), but an export from an experiment with NO
# checkpoint recomputes them, so a model that defines `derive` and reads its data has to build.
#
# The `run_id` path takes no answer at all. That handle exists already, with whatever data the run
# built, and there is nothing left for this keyword to decide; accepting it there would be a knob
# that reads as if it did something.
function _push_export_data!(
        kwargs::Vector{Pair{Symbol, Any}}, data::Union{AbstractString, Nothing},
        run_id::Union{AbstractString, Nothing}, experiment::Union{AbstractString, Nothing}
    )
    if run_id !== nothing
        data === nothing || error(
            "ReactantNitro ReactantNitroKaimonGateExt: `nitro_export` takes `data` only with \
             `experiment`. Run `$(run_id)`'s `Nitro` is already built, so there is no setup left \
             for it to decide."
        )
        return kwargs
    end
    experiment === nothing && return kwargs        # neither: `_target_nitro` says so, and better.
    d = data === nothing ? "none" : String(data)
    d in ("none", "build") || error(
        "ReactantNitro ReactantNitroKaimonGateExt: `nitro_export`'s `data` is `\"none\"` (the \
         default: skip `build_data`, since an export reads weights and traces a graph) or \
         `\"build\"` (call `build_data`, for a model whose `derive` reads its data on an export \
         with no checkpoint). Got `$(repr(d))`."
    )
    d == "none" && push!(kwargs, :data => (;))
    return kwargs
end

function _build_nitro(
        state::RunState, spec::AbstractString, overrides::Union{AbstractString, Nothing},
        kwargs::Vector{Pair{Symbol, Any}}, preset::Union{AbstractString, Nothing}
    )
    T = _resolve_experiment(spec)
    o = _check_overrides(T, _parse_overrides(overrides), kwargs)
    e = preset === nothing ? T(; o...) : ReactantNitro.from_preset(T, Symbol(preset); o...)
    nitro = Nitro(e; kwargs..., logger = RecordingLogger(state, ReactantNitro.logger(e)))
    lock(RUNS_LOCK) do
        state.experiment = spec
        state.run_dir = ReactantNitro.run_dir(nitro)
        # Kept so a later `nitro_validate`/`nitro_evaluate`/`nitro_predict`/`nitro_export` with
        # this run's id can reuse the trained handle, and so `nitro_stop` can reach it.
        state.nitro = nitro
    end
    return nitro
end

# ── The launcher and the phase monitor ────────────────────────────────────────────────

function _attach_monitor!(state::RunState, nitro::Nitro)
    f = function (phase, step, epoch, info)
        lock(RUNS_LOCK) do
            state.phase = string(nameof(typeof(phase)))
            step === nothing || (state.step = step)
            epoch === nothing || (state.epoch = epoch)
            hasproperty(info, :metrics) && info.metrics !== nothing &&
                (state.val_metrics = info.metrics)
        end
        return nothing
    end
    state.monitor = ReactantNitro.register_phase_monitor!(nitro, f)
    return nothing
end

# A run body executes at the LATEST WORLD, and that is a guarantee rather than belt-and-braces: a
# method added after the task STARTED is invisible to it, and a run task raises the world counter
# while it is already running. `_reactant_server_backend` is the case that found this. It loads
# ReactantServerExport with `Base.require` when an export names that backend, the extension's
# `site_provenance` and `write_export` methods only exist after that load, and the export dispatched
# to the erroring generic instead: the FIRST `nitro_export` of a session failed on provenance while
# an identical relaunch succeeded, because by then the session had the method.
# The hazard belongs to every tool here and not only to export, since a session defines methods in
# the REPL (`ex`, a Revise reload) and launches a run in the same turn.
#
# A task's world age is taken when it FIRST RUNS rather than when it is constructed, so ordering
# alone very nearly hides this: a load landing before the task's first slice IS seen. That is what
# makes it intermittent rather than a hard failure, and it is why the guarantee is here rather than
# in the ordering of the caller. One dynamic call per RUN, against work measured in minutes.
function _launch_run!(f::Function, state::RunState)
    @async begin
        try
            result = Base.invokelatest(f)
            lock(RUNS_LOCK) do
                state.status = :completed
                state.result = result
                state.finished_at = time()
            end
        catch e
            lock(RUNS_LOCK) do
                state.status = :failed
                state.error = sprint(showerror, e, catch_backtrace())
                state.finished_at = time()
            end
        finally
            m = state.monitor
            if m !== nothing
                try
                    ReactantNitro.unregister_phase_monitor!(m)
                catch
                end
            end
            lock(RUNS_LOCK) do
                _trim_registry_locked!()
            end
        end
    end
    return state.id
end

# The four task bodies. Each returns the summary string stored in `state.result`.

function _train_task(
        state::RunState, spec::AbstractString, overrides::Union{AbstractString, Nothing},
        kwargs::Vector{Pair{Symbol, Any}}, preset::Union{AbstractString, Nothing}
    )
    nitro = _build_nitro(state, spec, overrides, kwargs, preset)
    _attach_monitor!(state, nitro)
    trained = ReactantNitro.train!(nitro)
    return "completed: epoch=$(ReactantNitro.current_epoch(trained)) \
            step=$(ReactantNitro.current_step(trained)) stop_reason=$(trained.stop_reason) \
            run_dir=$(trained.run_dir)"
end

function _eval_task(state::RunState, nitro::Nitro, split::Symbol)
    _attach_monitor!(state, nitro)
    metrics = split === :val ? ReactantNitro.validate(nitro) : ReactantNitro.evaluate(nitro; split)
    return "completed: split=$split metrics=$metrics"
end

function _predict_task(state::RunState, nitro::Nitro, inputs_spec::AbstractString)
    _attach_monitor!(state, nitro)
    inputs = _parse_inputs(inputs_spec)
    out = ReactantNitro.predict(nitro, inputs)
    return _summarize_outputs(out)
end

function _export_task(
        state::RunState, nitro::Nitro, backend::ExportBackend,
        dir::AbstractString, name::AbstractString,
        batch_sizes::Vector{Int}, provenance::Dict{String, Any},
        provenance_root::Union{AbstractString, Nothing}
    )
    # Without the monitor, an export run's status never moves (the trace publishes no other
    # phases) and `nitro_status` would show "starting" for the whole minutes-long compile. The
    # framework now publishes `ExportCompiling` around `write_export`; attaching the same monitor
    # the train and eval tasks attach is what reports it.
    _attach_monitor!(state, nitro)
    written = ReactantNitro.export_model(
        nitro, backend; dir = String(dir), name = String(name),
        batch_sizes = batch_sizes, provenance_root = provenance_root,
        provenance = provenance,
    )
    return "exported to $written"
end

# ── Target resolution: a completed run's Nitro, or a fresh one ───────────────────────

function _target_nitro(
        state::RunState, run_id::Union{AbstractString, Nothing},
        spec::Union{AbstractString, Nothing}, checkpoint::Union{AbstractString, Nothing},
        overrides::Union{AbstractString, Nothing}, kwargs::Vector{Pair{Symbol, Any}},
        preset::Union{AbstractString, Nothing}
    )
    (run_id === nothing && spec === nothing) && error(
        "ReactantNitro ReactantNitroKaimonGateExt: pass `run_id` (a completed run in this session) or \
         `experiment` (a type string), not neither."
    )
    (run_id !== nothing && spec !== nothing) && error(
        "ReactantNitro ReactantNitroKaimonGateExt: pass `run_id` OR `experiment`, not both."
    )
    if run_id !== nothing
        src = _get_run(run_id)
        src === nothing && error(
            "ReactantNitro ReactantNitroKaimonGateExt: no run `$(run_id)` in this session's registry; \
             `nitro_runs()` lists the ones there are."
        )
        src.nitro === nothing && error(
            "ReactantNitro ReactantNitroKaimonGateExt: run `$(run_id)` (kind $(src.kind)) holds no `Nitro` to \
             operate on. Pass `experiment` (with `checkpoint` for trained weights) instead."
        )
        return src.nitro
    end
    kw = copy(kwargs)
    checkpoint === nothing || push!(kw, :checkpoint => checkpoint)
    return _build_nitro(state, spec, overrides, kw, preset)
end

# ── Batch inputs for predict ──────────────────────────────────────────────────────────
#
# The batch is the same `name=value` literal format as `overrides`, with each value an array.
# Arrays are converted to Float32, the framework's storage standard, since a batch field
# is host data crossing into a traced program; the batch axis is last, exactly as the framework
# asserts everywhere else.

function _parse_inputs(s::AbstractString)
    d = _parse_overrides(s)
    for (k, v) in d
        d[k] = _batch_value(v)
    end
    ks = Tuple(keys(d))
    return NamedTuple{ks}(Tuple(values(d)))
end

_batch_value(v) = v isa AbstractArray && eltype(v) <: Union{AbstractFloat, Integer} ?
    convert(Array{Float32}, v) : v

function _summarize_outputs(out)
    if out isa NamedTuple
        return join(("$k = $(_leaf_summary(v))" for (k, v) in pairs(out)), "\n")
    elseif out isa Tuple
        return join((_leaf_summary(v) for v in out), "\n")
    else
        return _leaf_summary(out)
    end
end

function _leaf_summary(x)
    if x isa AbstractArray
        flat = vec(x)
        n = min(length(flat), 5)
        preview = isempty(flat) ? "" : "  first: " * join(string.(flat[1:n]), ", ")
        return "array($(eltype(x)), size $(join(size(x), "x")))$preview"
    end
    return string(x)
end

# ── The export backend registry ───────────────────────────────────────────────────────
#
# The tool names a backend with a string. `reactant_server` is the shipped one, resolved by
# loading the ReactantServerExport weakdep on demand so the tool costs nothing until an export
# actually happens. A small registry lets a test inject a recording backend instead of tracing a
# real StableHLO bundle, which the CPU suite deliberately does not do.
#
# RESOLUTION HAPPENS IN THE CALLER, not in the run task: `nitro_export` calls this before it
# launches. A factory that loads a package raises the world counter, and `_launch_run!` says what
# that did to an export that resolved its backend from inside the task it was already running in.

const _BACKENDS = Dict{String, Function}()

function _register_backend!(name::AbstractString, factory::Function)
    _BACKENDS[String(name)] = factory
    return nothing
end

function _resolve_backend(name::AbstractString)
    name == "reactant_server" && return _reactant_server_backend()
    haskey(_BACKENDS, name) || error(
        "ReactantNitro ReactantNitroKaimonGateExt: unknown export backend `$(name)`. Known backends: \
         reactant_server$(isempty(_BACKENDS) ? "" : ", " * join(keys(_BACKENDS), ", "))."
    )
    return _BACKENDS[name]()
end

function _reactant_server_backend()
    try
        Base.require(
            Base.PkgId(Base.UUID("90fa0446-f028-4638-8833-e223884d9b4d"), "ReactantServerExport")
        )
    catch
        error(
            "ReactantNitro ReactantNitroKaimonGateExt: the `reactant_server` export backend needs the \
             ReactantServerExport package loadable from this project. `Pkg.add` it, or pass a \
             backend registered for this session."
        )
    end
    return ReactantServerBundle()
end

# ── Status rendering ──────────────────────────────────────────────────────────────────

# One line naming the run's logger and its key parameters, read live off the handle. The
# `RecordingLogger` the tools install wraps the real logger, so the summary unwraps to name the
# backend the user actually chose and forwards `logger_info` for its identifying parameters
# (the URL and experiment key of a hosted tracker, the file path of the shipped JSON default).
function _logger_summary(state::RunState)
    lgr = state.nitro.logger
    inner = lgr isa RecordingLogger ? lgr.inner : lgr
    kind = inner === nothing ? "<none>" : string(nameof(typeof(inner)))
    info = ReactantNitro.logger_info(lgr)
    bits = [string(k) * "=" * string(v) for (k, v) in pairs(info) if v !== nothing]
    return isempty(bits) ? kind : kind * " (" * join(bits, ", ") * ")"
end

function _status_text(state::RunState)
    io = IOBuffer()
    println(
        io, "run $(state.id)  kind=$(state.kind)  experiment=$(state.experiment)  ",
        "status=$(state.status)",
    )
    if state.status === :running
        println(io, "  phase: $(isempty(state.phase) ? "starting" : state.phase)")
        println(io, "  epoch: $(state.epoch === nothing ? "n/a" : state.epoch)")
        println(io, "  step:  $(state.step === nothing ? "n/a" : state.step)")
    end
    state.loss === nothing || println(io, "  latest train loss: $(state.loss)")
    state.val_metrics === nothing || println(io, "  val metrics: $(state.val_metrics)")
    state.run_dir === nothing || println(io, "  run_dir: $(state.run_dir)")
    state.nitro === nothing || println(io, "  logger: $(_logger_summary(state))")
    state.status === :completed && state.result !== nothing &&
        println(io, "  result: $(state.result)")
    state.status === :failed && println(io, "  error: $(state.error)")
    return String(take!(io))
end

# ── The tools ─────────────────────────────────────────────────────────────────────────
#
# Each handler is a module-level named function with a docstring; KaimonGate reflects the
# signature into the MCP schema and the docstring into the tool description, so the agent sees
# exactly the contract below. Handlers return immediately; the work runs on a background task
# and the agent polls `nitro_status`.

"""
    nitro_train(experiment; max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm,
                preset, resume, checkpoint, overrides) -> String

Start training an experiment in the background and return the run id immediately.

`experiment` is a module-qualified type string for an `@experiment` struct already loaded in
this session, e.g. `MyModels.MnistMLP`, or a bare name the model package exported into Main.

The typed keywords are run knobs passed to `Nitro` (a keyword beats the experiment's own
accessor for this run): `max_epochs`, `run_dir`, `seed`, `n_devs`, `accum`,
`gradient_clip_norm`, `preset` (a name from `presets(MyExp)`), `resume` (`"auto"`, `"false"`,
or a checkpoint path), and `checkpoint` (a checkpoint path to load).

`overrides` carries experiment-FIELD values that cannot be in the schema because they are
per-experiment: a comma-separated `name=value` list of Julia literals, e.g.
`overrides="width=128, smoothing=0.05, labels=[1.0,2.0]"`. A key that is not a field of the
experiment is an error, and a key passed both as a run keyword and in `overrides` is an
ambiguity error.

The call returns before training starts. Poll `nitro_status(run_id=...)` until the run
completes; `nitro_runs()` lists all runs. Training is graceful under `nitro_stop(run_id=...)`:
the epoch finishes, validation runs, the checkpoint is written, and the run exits `Done`.

```julia
nitro_train(experiment="MyModels.MnistMLP", max_epochs=40, run_dir="runs/mnist_v1")
```
"""
function nitro_train(
        experiment::String;
        max_epochs::Union{Int, Nothing} = nothing,
        run_dir::Union{String, Nothing} = nothing,
        seed::Union{Int, Nothing} = nothing,
        n_devs::Union{Int, Nothing} = nothing,
        accum::Union{Int, Nothing} = nothing,
        gradient_clip_norm::Union{Float64, Nothing} = nothing,
        preset::Union{String, Nothing} = nothing,
        resume::Union{String, Nothing} = nothing,
        checkpoint::Union{String, Nothing} = nothing,
        overrides::Union{String, Nothing} = nothing,
    )
    kwargs = _collect_run_kwargs(;
        max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm, preset, resume,
    )
    checkpoint === nothing || push!(kwargs, :checkpoint => checkpoint)
    state = _new_run(:train, experiment)
    _launch_run!(state) do
        _train_task(state, experiment, overrides, kwargs, preset)
    end
    return "started run $(state.id) (kind=train, experiment=$(experiment)). Poll \
            `nitro_status(run_id=\"$(state.id)\")`; stop with \
            `nitro_stop(run_id=\"$(state.id)\")`."
end

"""
    nitro_validate(; run_id, experiment, checkpoint, overrides, ...) -> String

Run the `val` split over a `Nitro` in the background and return the run id immediately.

Pass exactly one of `run_id` (a completed train run in this session; its trained `Nitro` is
reused) or `experiment` (a type string; `checkpoint` loads trained weights, otherwise the
weights are the freshly built ones). The remaining typed keywords are run knobs used only when
constructing from `experiment`; `overrides` is experiment-field values, same format as
`nitro_train`.

Poll `nitro_status(run_id=...)` for the finalized metrics.
"""
function nitro_validate(
        ;
        run_id::Union{String, Nothing} = nothing,
        experiment::Union{String, Nothing} = nothing,
        checkpoint::Union{String, Nothing} = nothing,
        overrides::Union{String, Nothing} = nothing,
        max_epochs::Union{Int, Nothing} = nothing,
        run_dir::Union{String, Nothing} = nothing,
        seed::Union{Int, Nothing} = nothing,
        n_devs::Union{Int, Nothing} = nothing,
        accum::Union{Int, Nothing} = nothing,
        gradient_clip_norm::Union{Float64, Nothing} = nothing,
        preset::Union{String, Nothing} = nothing,
        resume::Union{String, Nothing} = nothing,
    )
    kwargs = _collect_run_kwargs(;
        max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm, preset, resume,
    )
    label = something(experiment, run_id, "")
    state = _new_run(:validate, label)
    _launch_run!(state) do
        nitro = _target_nitro(state, run_id, experiment, checkpoint, overrides, kwargs, preset)
        _eval_task(state, nitro, :val)
    end
    return "started run $(state.id) (kind=validate). Poll `nitro_status(run_id=\"$(state.id)\")`."
end

"""
    nitro_evaluate(; run_id, experiment, split="test", checkpoint, overrides, ...) -> String

Run one split (default `"test"`) over a `Nitro` in the background and return the run id
immediately. Otherwise identical to `nitro_validate`: pass `run_id` or `experiment` (plus
`checkpoint` for trained weights), and poll `nitro_status(run_id=...)` for the metrics.

```julia
nitro_evaluate(run_id="a1b2c3d4")
nitro_evaluate(experiment="MyModels.MnistMLP", split="val", checkpoint="runs/mnist_v1/latest")
```
"""
function nitro_evaluate(
        ;
        run_id::Union{String, Nothing} = nothing,
        experiment::Union{String, Nothing} = nothing,
        split::String = "test",
        checkpoint::Union{String, Nothing} = nothing,
        overrides::Union{String, Nothing} = nothing,
        max_epochs::Union{Int, Nothing} = nothing,
        run_dir::Union{String, Nothing} = nothing,
        seed::Union{Int, Nothing} = nothing,
        n_devs::Union{Int, Nothing} = nothing,
        accum::Union{Int, Nothing} = nothing,
        gradient_clip_norm::Union{Float64, Nothing} = nothing,
        preset::Union{String, Nothing} = nothing,
        resume::Union{String, Nothing} = nothing,
    )
    kwargs = _collect_run_kwargs(;
        max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm, preset, resume,
    )
    label = something(experiment, run_id, "")
    state = _new_run(:evaluate, label)
    _launch_run!(state) do
        nitro = _target_nitro(state, run_id, experiment, checkpoint, overrides, kwargs, preset)
        _eval_task(state, nitro, Symbol(split))
    end
    return "started run $(state.id) (kind=evaluate, split=$split). Poll \
            `nitro_status(run_id=\"$(state.id)\")`."
end

"""
    nitro_predict(; run_id, experiment, inputs, checkpoint, overrides, ...) -> String

Run `predict` on one batch in the background and return the run id immediately.

`inputs` is the batch as `name=value` literals, one per field `forward` declares, e.g.
`inputs="x=[1.0,2.0,3.0]"` or `inputs="x=[1.0 2.0 3.0; 4.0 5.0 6.0]"`. The batch axis is
LAST, exactly as the framework asserts on every batch. Arrays are converted to Float32. The
result carries each output's shape and the first few values, never the full arrays.

Pass `run_id` (a completed train run) or `experiment` plus `checkpoint` for trained weights.
"""
function nitro_predict(
        ;
        run_id::Union{String, Nothing} = nothing,
        experiment::Union{String, Nothing} = nothing,
        inputs::String = "",
        checkpoint::Union{String, Nothing} = nothing,
        overrides::Union{String, Nothing} = nothing,
        max_epochs::Union{Int, Nothing} = nothing,
        run_dir::Union{String, Nothing} = nothing,
        seed::Union{Int, Nothing} = nothing,
        n_devs::Union{Int, Nothing} = nothing,
        accum::Union{Int, Nothing} = nothing,
        gradient_clip_norm::Union{Float64, Nothing} = nothing,
        preset::Union{String, Nothing} = nothing,
        resume::Union{String, Nothing} = nothing,
    )
    kwargs = _collect_run_kwargs(;
        max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm, preset, resume,
    )
    label = something(experiment, run_id, "")
    state = _new_run(:predict, label)
    _launch_run!(state) do
        nitro = _target_nitro(state, run_id, experiment, checkpoint, overrides, kwargs, preset)
        _predict_task(state, nitro, inputs)
    end
    return "started run $(state.id) (kind=predict). Poll `nitro_status(run_id=\"$(state.id)\")`."
end

"""
    nitro_export(; run_id, experiment, dir, name, backend="reactant_server",
                 batch_sizes="[1]", provenance_root, provenance, checkpoint,
                 overrides, ...) -> String

Export a trained model in the background and return the run id immediately.

`dir` and `name` are required: the artifact lands in `dir/name` (the bundle format requires
the directory basename to equal `name`). `backend` is `"reactant_server"` (the shipped
StableHLO bundle backend, loaded on demand) or a backend registered for this session.
`batch_sizes` is a vector literal of the batch sizes to compile, e.g. `"[1, 8]"`.

**`provenance_root` is the repository root to collect repository state from**, and passing it
is how a bundle exported through this tool gets a git commit, a tree hash and, on a dirty
tree, the `working_tree.patch` that is the only thing tying the artifact to the code that
produced it. The backend collects it, through its `site_provenance` method, so the patch never
has to pass through this tool as text.

**WHAT YOU GET IF YOU OMIT IT**, stated because it is the failure worth designing against:
the export succeeds, every check passes, and the manifest carries the framework's stamps (the
flat config, the preset, the seed, the run directory, the framework version, and the
checkpoint the weights were restored from) plus whatever the experiment's
`export_provenance_extra` hook adds, and **no repository state at all**. The bundle looks
complete and cannot answer which code built it. That is a legitimate choice for a throwaway
export and the wrong one for anything that gets registered.

`provenance` is an optional `name=value` list of Julia literals, merged LAST so it overrides
every other layer. It is for one-off scalars and strings; a model's own facts belong in the
`export_provenance_extra` hook, where neither this tool nor a hand-written `export_model`
call can forget them. `export_model`'s own docstring gives the full precedence.

Pass `run_id` (a completed train run; its trained `Nitro` is exported) or `experiment` plus
`checkpoint` for trained weights. Export asserts a single-device handle, so a fresh
construction uses `n_devs = 1`.

**An `experiment` export does not build the training data**, because setup for an export reads
weights and traces a graph and the data is part of neither; it is the same
`Nitro(e; checkpoint = path, data = (;))` construction `export_model`'s own docstring
prescribes. Two things follow. A model whose exportable handle is a DIFFERENT BUILD from its
trainable one is exportable through this tool: put the flag in `overrides` and the handle is
built with it, where a `build_data` that refuses an inference configuration never runs.
And nothing starts a data server that the export then has to tear down.

`data = "build"` calls `build_data` anyway. The one case that needs it: `derive` is skipped on
a `checkpoint =` construction, but an export from an `experiment` with NO checkpoint
recomputes the derived values, so a model whose `derive` reads its data has to build. With
`run_id` the keyword is refused, since that handle is already built.

```julia
nitro_export(run_id="a1b2c3d4", dir="export_out", name="mnist_v1",
             provenance_root="/path/to/model/repo")

# a handle whose export build differs from its train build
nitro_export(experiment="MyModels.MyExperiment", preset="my_preset",
             checkpoint="runs/v2/epoch-0040.jld2", overrides="export_inference=true",
             dir="runs/export_out", name="my_model", provenance_root="/path/to/model/repo")
```
"""
function nitro_export(
        ;
        run_id::Union{String, Nothing} = nothing,
        experiment::Union{String, Nothing} = nothing,
        dir::String = "",
        name::String = "",
        backend::String = "reactant_server",
        batch_sizes::String = "[1]",
        provenance_root::Union{String, Nothing} = nothing,
        provenance::Union{String, Nothing} = nothing,
        checkpoint::Union{String, Nothing} = nothing,
        overrides::Union{String, Nothing} = nothing,
        max_epochs::Union{Int, Nothing} = nothing,
        run_dir::Union{String, Nothing} = nothing,
        seed::Union{Int, Nothing} = nothing,
        n_devs::Union{Int, Nothing} = nothing,
        accum::Union{Int, Nothing} = nothing,
        gradient_clip_norm::Union{Float64, Nothing} = nothing,
        preset::Union{String, Nothing} = nothing,
        resume::Union{String, Nothing} = nothing,
        data::Union{String, Nothing} = nothing,
    )
    isempty(dir) && error("ReactantNitro ReactantNitroKaimonGateExt: `nitro_export` needs `dir` (where the \
                           artifact goes) and `name` (the bundle's name).")
    isempty(name) && error("ReactantNitro ReactantNitroKaimonGateExt: `nitro_export` needs `name` (the \
                            bundle's name); the artifact lands in `dir/name`.")
    kwargs = _collect_run_kwargs(;
        max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm, preset, resume,
    )
    # Export is a CPU trace asserting one device: a fresh construction must not inherit
    # a multi-device mesh from the run knobs. An explicit `n_devs = 1` is fine and is not pushed
    # twice; anything else is refused.
    if experiment !== nothing
        any(p -> first(p) === :n_devs && last(p) != 1, kwargs) && error(
            "ReactantNitro ReactantNitroKaimonGateExt: `nitro_export` from `experiment` requires \
             `n_devs = 1`; export is a single-device CPU trace."
        )
        all(p -> first(p) !== :n_devs, kwargs) && push!(kwargs, :n_devs => 1)
    end
    _push_export_data!(kwargs, data, run_id, experiment)
    # Resolved HERE rather than inside the run body, for two reasons that are not the same reason.
    # A name this session has no backend for is the caller's mistake and belongs in the caller's
    # answer, not in a background run that has to be polled to discover it. And `reactant_server`
    # loads a package to answer, which is work that has no business happening inside a run task:
    # `_launch_run!` guarantees the latest world for the methods that arrive with it, and this
    # keeps the load out of the traced stretch entirely.
    resolved = _resolve_backend(backend)
    label = something(experiment, run_id, "")
    state = _new_run(:export, label)
    _launch_run!(state) do
        nitro = _target_nitro(state, run_id, experiment, checkpoint, overrides, kwargs, preset)
        bs = _parse_batch_sizes(batch_sizes)
        prov = _parse_provenance(provenance)
        # A `Nitro` built for this export (the `experiment` path) has no run to end it, so nothing
        # would ever publish its `Terminal`, and whatever `build_data` started (a data server, a
        # client) outlives the export until a person notices. Publishing `Done` when the export is
        # over, success or failure, is what lets the experiment's own Terminal monitors tear down.
        # NOT for a run's `Nitro` (the `run_id` path): that one belongs to the run, which may still
        # evaluate or export again, and ending it here would take the run's data server with it.
        #
        # Since `data` defaults to not building, the usual export now starts nothing for this to
        # tear down. It stays because `data = "build"` still can, and because the teardown is what
        # closes an open logger either way.
        finish = experiment !== nothing
        try
            _export_task(state, nitro, resolved, dir, name, bs, prov, provenance_root)
        finally
            finish && _finish_export_nitro!(nitro)
        end
    end
    return "started run $(state.id) (kind=export). Poll `nitro_status(run_id=\"$(state.id)\")`."
end

# Publish `Done` on a `Nitro` that exists only for an export, so its experiment's Terminal monitors
# run (teardown of a data server, a client, an open logger). Never throws: the export's own result
# is already decided by the time this runs, and a teardown that raised would replace it.
function _finish_export_nitro!(nitro::Nitro)
    try
        # Idempotent, and cheap: a monitor registered during `build_data` is adopted at the end of
        # setup already, but a monitor registered later (a session that loaded a package mid-round)
        # is not, and a teardown that runs no monitors tears nothing down.
        ReactantNitro.adopt_monitors!(nitro)
        ReactantNitro.publish_phase(nitro, ReactantNitro.Done())
    catch err
        @warn "ReactantNitro ReactantNitroKaimonGateExt: teardown after export raised; the bundle is \
               unaffected." exception = (err, catch_backtrace())
    end
    return nothing
end

function _parse_batch_sizes(s::String)
    v = _parse_literal(s)
    v isa AbstractVector{<:Integer} || error(
        "ReactantNitro ReactantNitroKaimonGateExt: `batch_sizes` must be a vector literal of integers, e.g. \
         `\"[1, 8]\"`; got `$(repr(v))`."
    )
    isempty(v) && error("ReactantNitro ReactantNitroKaimonGateExt: `batch_sizes` must name at least one size.")
    all(>(0), v) || error("ReactantNitro ReactantNitroKaimonGateExt: every batch size must be positive.")
    return Int.(v)
end

function _parse_provenance(s::Union{AbstractString, Nothing})
    d = _parse_overrides(s)
    return Dict{String, Any}(string(k) => v for (k, v) in d)
end

"""
    nitro_runs() -> String

List every run this session's registry knows about: id, kind, status, phase, epoch, step,
latest loss, and run_dir. Runs are process-local, so a session restart starts with an empty
list; checkpoints are the restart story.
"""
function nitro_runs()
    return lock(RUNS_LOCK) do
        isempty(RUNS) && return "No runs yet. Start one with `nitro_train`."
        rows = sort!(collect(values(RUNS)); by = s -> s.started_at, rev = true)
        io = IOBuffer()
        for s in rows
            println(
                io, "$(s.id)  kind=$(s.kind)  status=$(s.status)  ",
                "phase=$(isempty(s.phase) ? "-" : s.phase)  ",
                "epoch=$(s.epoch === nothing ? "-" : s.epoch)  ",
                "step=$(s.step === nothing ? "-" : s.step)  ",
                "loss=$(s.loss === nothing ? "-" : s.loss)",
            )
        end
        return String(take!(io))
    end
end

"""
    nitro_status(run_id) -> String

Report one run: status, current phase, epoch and step (while running), the latest train loss
and validation metrics, run_dir, the result summary on completion, or the error text on
failure. A run id comes from `nitro_train`/`nitro_validate`/`nitro_evaluate`/`nitro_predict`/
`nitro_export` or from `nitro_runs()`.
"""
function nitro_status(run_id::AbstractString)
    state = _get_run(run_id)
    state === nothing && error(
        "ReactantNitro ReactantNitroKaimonGateExt: no run `$(run_id)` in this session's registry; \
         `nitro_runs()` lists the ones there are."
    )
    return _status_text(state)
end

"""
    ReactantNitro.run_state(run_id) -> NamedTuple

The registry's answer for `ReactantNitro.run_state`: the same facts `nitro_status` renders, as data.
Documented on the stub in `src/Runs.jl`; this is the method.
"""
function ReactantNitro.run_state(run_id::AbstractString)
    state = _get_run(run_id)
    state === nothing && error(
        "ReactantNitro.run_state: no run `$(run_id)` in this session's registry; `nitro_runs()` lists \
         the ones there are."
    )
    finished = state.finished_at
    return (;
        id = state.id, kind = state.kind, experiment = state.experiment, status = state.status,
        phase = state.phase, epoch = state.epoch, step = state.step, loss = state.loss,
        val_metrics = state.val_metrics, run_dir = state.run_dir, error = state.error,
        result = state.result,
        elapsed_s = (finished === nothing ? time() : finished) - state.started_at,
    )
end

"""
    nitro_logger(run_id) -> String

Report the run's logger: the backend's type and the key identifying parameters its
`logger_info` exposes, e.g. a hosted tracker's experiment key and URL, a W&B run id and project,
or the shipped JSON logger's metrics file path. Reads the live logger off the run's `Nitro`, so
the answer is current even while the run trains.

```julia
nitro_logger(run_id="a1b2c3d4")
```
"""
function nitro_logger(run_id::AbstractString)
    state = _get_run(run_id)
    state === nothing && error(
        "ReactantNitro ReactantNitroKaimonGateExt: no run `$(run_id)` in this session's registry; \
         `nitro_runs()` lists the ones there are."
    )
    state.nitro === nothing && error(
        "ReactantNitro ReactantNitroKaimonGateExt: run `$(run_id)` holds no `Nitro`, so there is no logger to \
         report. Start one with `nitro_train`."
    )
    io = IOBuffer()
    lgr = state.nitro.logger
    inner = lgr isa RecordingLogger ? lgr.inner : lgr
    if inner === nothing
        println(io, "run $(state.id)  logger=<none>")
        println(io, "  this run was launched with `logger = nothing`, the documented opt-out.")
    else
        println(io, "run $(state.id)  logger=$(string(nameof(typeof(inner))))")
        info = ReactantNitro.logger_info(lgr)
        if isempty(info)
            println(
                io, "  no logger info: the backend defines no `logger_info` method (the \
                logging contract's default is `(;)`)."
            )
        else
            for (k, v) in pairs(info)
                println(io, "  $k: $(v === nothing ? "n/a" : v)")
            end
        end
    end
    return String(take!(io))
end

"""
    nitro_stop(run_id) -> String

Request a graceful stop of a running train or eval run: the epoch finishes, validation runs,
the checkpoint is written, and the run exits through the normal `Done` path. Poll
`nitro_status(run_id=...)` to watch it wind down.
"""
function nitro_stop(run_id::AbstractString)
    state = _get_run(run_id)
    state === nothing && error(
        "ReactantNitro ReactantNitroKaimonGateExt: no run `$(run_id)` in this session's registry; \
         `nitro_runs()` lists the ones there are."
    )
    state.status !== :running && return "run $(run_id) is already $(state.status); nothing to stop."
    state.nitro === nothing && error(
        "ReactantNitro ReactantNitroKaimonGateExt: run `$(run_id)` has no live `Nitro` to stop."
    )
    ReactantNitro.request_stop!(state.nitro)
    return "stop requested for run $(run_id); it will finish the current epoch, validate, and \
            checkpoint before exiting. Poll `nitro_status(run_id=\"$(run_id)\")`."
end

"""
    nitro_setup(; backend, n_devs) -> String

Configure this session's accelerator for every subsequent run, or report the current
configuration with no arguments. **Thin tool form of `ReactantNitro.setup_devices!`**, the same
function a REPL session calls, so a REPL workflow and a Kaimon workflow configure identically.

`backend` selects the Reactant backend for the process: `"cpu"`, `"gpu"` (whichever of
CUDA/ROCm is available), `"cuda"`, `"rocm"`, `"tpu"`. Omit it to leave Reactant's default (on a
GPU machine that is the GPU; with none visible, CPU).

`n_devs` pins how many VISIBLE devices subsequent runs shard the batch over. The default, when
neither this call nor an experiment declares a count, is `length(Reactant.devices())`, meaning
every visible device. On a GPU host, restrict the set with `CUDA_VISIBLE_DEVICES` at session start:
one Julia process initializes XLA once, so device visibility is fixed for the process, and the
pin can only select within what is visible. The pin wins over an experiment's declared
`n_devs`; an explicit `n_devs` keyword on `nitro_train` still wins for that run.

```julia
nitro_setup(backend="cpu")            # run everything on CPU
nitro_setup(backend="cuda", n_devs=2) # two of the visible CUDA devices
nitro_setup()                         # report what is in effect
```
"""
function nitro_setup(;
        backend::Union{AbstractString, Nothing} = nothing,
        n_devs::Union{Int, Nothing} = nothing,
    )
    cfg = ReactantNitro.setup_devices!(; backend = backend, n_devs = n_devs)
    p = cfg.pinned ? "pinned via nitro_setup" : "default (every visible device)"
    return "accelerator: backend=$(cfg.backend), n_devs=$(cfg.n_devs) of " *
        "$(cfg.visible) visible device(s) ($p). Subsequent runs in this session shard the " *
        "batch over $(cfg.n_devs) device(s)."
end

# ── Registration ──────────────────────────────────────────────────────────────────────

function _build_tools()
    return KaimonGate.GateTool[
        KaimonGate.GateTool("nitro_setup", nitro_setup),
        KaimonGate.GateTool("nitro_train", nitro_train),
        KaimonGate.GateTool("nitro_validate", nitro_validate),
        KaimonGate.GateTool("nitro_evaluate", nitro_evaluate),
        KaimonGate.GateTool("nitro_predict", nitro_predict),
        KaimonGate.GateTool("nitro_export", nitro_export),
        KaimonGate.GateTool("nitro_runs", nitro_runs),
        KaimonGate.GateTool("nitro_status", nitro_status),
        KaimonGate.GateTool("nitro_logger", nitro_logger),
        KaimonGate.GateTool("nitro_stop", nitro_stop),
    ]
end

# Is a Kaimon gate bound in this process right now?
#
# KaimonGate's signal for that is PRIVATE and has already moved once: this extension was written
# against `_RUNNING::Ref{Bool}`, and since the GateSession refactor it is the accessor `_running()`
# (`gate_session.jl`: `_running() = (s = _SESSION[]; s === nothing ? false : s.running)`). So probe
# the known spellings in turn rather than naming one.
#
# WHEN NEITHER EXISTS, SAY SO ONCE AND LOUDLY. The previous version answered `false` for an
# unrecognized gate and called that graceful degradation. It is not: it silently switches the whole
# `nitro_*` tool surface OFF. Against KaimonGate 1.4 `_install_with_retry`'s loop asked "is a gate
# running" thirty times, heard "no" every time from a LIVE gate, and registered nothing; no
# exception was thrown, so `_install_tools`' own `@warn` never fired either, and an agent session
# simply had no `nitro_train` with nothing in the log to say why (measured). A rename upstream
# must cost a warning, never a feature.
#
# The real fix is upstream and is the same one the retry loop below asks for: a PUBLIC predicate
# (and registration hook) in KaimonGate, so an extension depends on API instead of on the name of
# a `Ref`. Until then, this is the compat shim, and `[compat] KaimonGate` cannot catch a private
# rename inside an admitted version range.
const _GATE_PROBE_WARNED = Ref(false)

function _warn_gate_probe_once(what, err)
    _GATE_PROBE_WARNED[] && return nothing
    _GATE_PROBE_WARNED[] = true
    msg = "ReactantNitro ReactantNitroKaimonGateExt: cannot tell whether a Kaimon gate is " *
        "running, so the nitro_* tools will NOT register. " * what * ". KaimonGate's " *
        "running-state signal is private and has moved before (`_RUNNING[]` -> `_running()`)."
    err === nothing ? (@warn msg) : (@warn msg exception = err)
    return nothing
end

function _gate_running()
    for probe in (:_running, :_RUNNING)
        isdefined(KaimonGate, probe) || continue
        try
            v = getproperty(KaimonGate, probe)
            return Bool(probe === :_running ? v() : v[])
        catch err
            _warn_gate_probe_once("probing `KaimonGate.$(probe)` threw", err)
            return false
        end
    end
    _warn_gate_probe_once(
        "KaimonGate exposes neither `_running()` nor `_RUNNING[]`", nothing
    )
    return false
end

function _install_tools()
    try
        # `force = true` keeps registration working in non-interactive processes (the test suite
        # runs headless), where `serve` would otherwise skip on the interactivity guard before
        # reaching its replace-tools branch. Safe here because the automatic path calls this only
        # when a gate is already running, and the manual hammer documents starting one.
        KaimonGate.serve(force = true, tools = _build_tools())
        return true
    catch e
        @warn "ReactantNitro ReactantNitroKaimonGateExt: registering the nitro_* tools failed" exception = e
        return false
    end
end

function _install_with_retry()
    _gate_running() && _install_tools() && return true
    # No gate yet, which in a real session means one is about to arrive: the host's preamble runs
    # `using KaimonGate`, then loads the model package (firing this `__init__`), and calls
    # `KaimonGate.serve(...)` last. So the extension normally loads BEFORE the gate binds, and the
    # loop below is the path that actually registers the tools. Retry for a bounded window rather
    # than registering once and hoping.
    #
    # There is deliberately NO `isinteractive()` guard here. A gate host launches as
    # `julia -t 16,1 --project=... --startup-file=no -e '<preamble>'`, where `isinteractive()` is
    # false, so guarding on it disarmed this loop in exactly the case it exists for: a live gate
    # registered nothing, silently, because no exception was thrown on the way out. Note that
    # `_install_tools` was written for headless hosts (that is what its `force = true` is for),
    # so the guard also contradicted its only caller. Cost of dropping it: a process that loads
    # KaimonGate and never serves a gate pays one idle background task that wakes once a second
    # for 30 seconds and then exits.
    #
    # This tolerates the ordering race rather than removing it. The real fix is upstream, an
    # additive registration hook in KaimonGate so an extension can register before `serve` binds
    # and have the tools picked up when it does. That would also stop two extensions'
    # `serve(tools = ...)` calls from overwriting each other's tool list, which nothing on this
    # side can.
    @async begin
        for _ in 1:30
            sleep(1.0)
            _gate_running() || continue
            _install_tools() && break
        end
    end
    return false
end

"""
    reinstall_kaimon_tools() -> Bool

Register the `nitro_*` GateTools with the Kaimon gate. Called automatically when the extension
loads; call it again after a manual `KaimonGate.stop()`/`serve()` cycle, or when another
extension's `serve(tools=...)` replaced this session's tools. Unlike the automatic path this
registers unconditionally: if no gate is running it starts one, exactly as `KaimonGate.serve()`
would.

Reach it from a session as `Base.get_extension(ReactantNitro, :ReactantNitroKaimonGateExt).reinstall_kaimon_tools()`.
It lives in the extension module rather than in ReactantNitro because a precompiled module
cannot create a new binding in another module, only add methods to existing ones.
"""
function reinstall_kaimon_tools()
    return _install_tools()
end

# The auto-registration lives in `__init__`, not at module top level: the extension body runs
# once, at precompile time (where no gate exists and the call is a no-op), while `__init__`
# runs on every runtime load, which is when the session's gate is (or is about to be) there.
function __init__()
    _install_with_retry()
    return nothing
end

end # module ReactantNitroKaimonGateExt
