# ReactantNitroKaimonGateExt.jl
#
# The KaimonGate extension: registers `nitro_*` GateTools with the running Kaimon gate so an agent
# can drive training, validation, evaluation, prediction and export from the session where the
# model code is loaded. A dev tool over entry points the framework already ships; nothing in
# `src/` knows it exists.
#
# Runs are background tasks because Kaimon's tool calls carry a hard deadline of minutes and a
# training run lasts hours: every tool starts the work and returns a run id, and the agent polls
# `nitro_status`. The registry does not survive a session restart; checkpoints are the restart
# story. KaimonGate's `serve(tools = [...])` REPLACES a gate's tool list, so the extension
# registers on load and retries for a short window to cover the race with the host's own `serve`
# at session boot; `reinstall_kaimon_tools()` is the manual hammer. Outside a gate the extension
# is a no-op.

module ReactantNitroKaimonGateExt

import ReactantNitro
import KaimonGate

using ReactantNitro: Nitro, ExportBackend, ReactantServerBundle

# ── The run registry ──────────────────────────────────────────────────────────────────
#
# One entry per tool-launched unit of work, process-local and lock-guarded. Finished runs are
# trimmed oldest-first past a small cap; a running run is never trimmed.

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
# The status tools need the latest metrics, so this backend keeps them in the run state and tees
# to the experiment's own logger. All ten verbs get a method, since a missing one is a MethodError.

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
# Forwarded, so the checkpoint record and the status tools carry the real backend's ids.
# `applicable` guards the two accessors with no generic default.
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
# Setup's adoption step, forwarded, so a wrapped default `JSONLogger` still lands in the run dir.
ReactantNitro._adopt_logger!(lgr::RecordingLogger, dir) =
    (lgr.inner === nothing || ReactantNitro._adopt_logger!(lgr.inner, dir); lgr)

_to_float(x::Real) = Float64(x)
_to_float(x::AbstractArray) = isempty(x) ? NaN : Float64(first(x))
_to_float(::Any) = NaN

# ── Experiment resolution ─────────────────────────────────────────────────────────────
#
# A module-qualified type string (`MyModels.MnistMLP`) or a bare name exported into Main, resolved
# against loaded bindings only.

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
# `overrides` carries experiment-field values, which cannot be in the tool schema: a comma-separated
# `name=value` list of Julia literals. A hand-rolled, dependency-free parser that accepts literals
# only, so a typo is an error rather than an `eval` of arbitrary text. Also serves `nitro_predict`.

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

# A value must be a literal: numbers, strings, booleans, symbols, `nothing`, or a vector, tuple or
# matrix of literals. A `:row` (a matrix literal's rows) is examined but only ever evaluated inside
# its `:vcat`/`:hcat` parent.
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
# The typed run knobs go to `Nitro`, where a keyword beats the experiment's accessor for one run;
# `overrides` carries experiment-field values and goes to the constructor. A key in both places is
# an ambiguity error.

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

# `resume` mirrors the constructor's own three-way split: the symbol `:auto` (find the latest
# checkpoint in run_dir), `false` (the default, start over), or a checkpoint path.
_parse_resume(s::AbstractString) = s == "auto" ? :auto : s == "false" ? false : String(s)

# `nitro_export`'s `data`: `"none"` (the default) builds no data, since an export reads weights and
# traces a graph, and is what lets a model whose exportable handle is a different build from its
# trainable one be exported through the tool; `"build"` calls `build_data`, for a model whose
# `derive` reads its data on an export with no checkpoint. The `run_id` path takes no answer, since
# that handle is already built.
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

# The run body executes at the latest world: a method added after the task started (the
# ReactantServerExport extension loaded on demand by `_reactant_server_backend`) is otherwise
# invisible to it, and the first `nitro_export` of a session failed on provenance while an
# identical relaunch succeeded. A task's world age is taken when it first runs, which is what made
# it intermittent.
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
    # Without the monitor an export run's status never moves; the framework publishes
    # `ExportCompiling` around `write_export` and this is what reports it.
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
# The same `name=value` literal format as `overrides`, each value an array converted to Float32,
# batch axis last.

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
# `reactant_server` is the shipped backend, resolved by loading the ReactantServerExport weakdep on
# demand; a small registry lets a test inject a recording backend. Resolution happens in the
# caller, not the run task, since loading a package raises the world counter (see `_launch_run!`).

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

# One line naming the run's logger and its key parameters, unwrapping the `RecordingLogger` to
# name the backend the user chose.
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
# Each handler is a named function with a docstring; KaimonGate reflects the signature into the MCP
# schema and the docstring into the tool description. Handlers return immediately.

"""
    nitro_train(experiment; max_epochs, run_dir, seed, n_devs, accum, gradient_clip_norm,
                preset, resume, checkpoint, overrides) -> String

Start training an experiment in the background and return the run id immediately.

`experiment` is a module-qualified type string for an `@experiment` struct loaded in this session,
e.g. `MyModels.MnistMLP`. The typed keywords are run knobs passed to `Nitro`, beating the
experiment's own accessor for this run: `max_epochs`, `run_dir`, `seed`, `n_devs`, `accum`,
`gradient_clip_norm`, `preset` (a name from `presets(MyExp)`), `resume` (`"auto"`, `"false"`, or a
checkpoint path), and `checkpoint`. `overrides` carries experiment-field values as a
comma-separated `name=value` list of Julia literals, e.g. `overrides="width=128, labels=[1.0,2.0]"`;
an unknown field or a key given in both places is an error.

Poll `nitro_status(run_id=...)` until the run completes; `nitro_runs()` lists all runs.
`nitro_stop(run_id=...)` stops gracefully after the current epoch's validation and checkpoint.

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

Run the `val` split over a `Nitro` in the background and return the run id immediately. Pass
exactly one of `run_id` (a completed train run in this session, whose trained `Nitro` is reused)
or `experiment` (a type string; `checkpoint` loads trained weights). The remaining typed keywords
are run knobs for the `experiment` construction; `overrides` has `nitro_train`'s format. Poll
`nitro_status(run_id=...)` for the finalized metrics.
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

`dir` and `name` are required; the artifact lands in `dir/name`. `backend` is
`"reactant_server"` (the shipped StableHLO bundle backend, loaded on demand) or a backend
registered for this session. `batch_sizes` is a vector literal, e.g. `"[1, 8]"`. Pass `run_id` (a
completed train run) or `experiment` plus `checkpoint`; export asserts a single-device handle.

`provenance_root` is the repository root to collect repository state from: the git commit, tree
hash and, on a dirty tree, the `working_tree.patch` that ties the artifact to the code that produced
it. Omitted, the export succeeds, every check passes, and the bundle cannot say which code built it;
fine for a throwaway export and wrong for anything that gets registered. `provenance` is a
`name=value` list merged last, for one-off scalars; a model's own facts belong in its
`export_provenance_extra` hook.

An `experiment` export does not build the training data (the `Nitro(e; checkpoint, data = (;))`
construction `export_model` prescribes), so a model whose exportable handle differs from its
trainable one is exportable by putting the flag in `overrides`. `data = "build"` calls `build_data`
anyway, for a model whose `derive` reads its data on an export with no checkpoint.

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
    # Export is a single-device CPU trace: a fresh construction must not inherit a mesh.
    if experiment !== nothing
        any(p -> first(p) === :n_devs && last(p) != 1, kwargs) && error(
            "ReactantNitro ReactantNitroKaimonGateExt: `nitro_export` from `experiment` requires \
             `n_devs = 1`; export is a single-device CPU trace."
        )
        all(p -> first(p) !== :n_devs, kwargs) && push!(kwargs, :n_devs => 1)
    end
    _push_export_data!(kwargs, data, run_id, experiment)
    # Resolved here rather than in the run body: an unknown backend is the caller's mistake, and
    # loading a package has no business inside a run task (see `_launch_run!`).
    resolved = _resolve_backend(backend)
    label = something(experiment, run_id, "")
    state = _new_run(:export, label)
    _launch_run!(state) do
        nitro = _target_nitro(state, run_id, experiment, checkpoint, overrides, kwargs, preset)
        bs = _parse_batch_sizes(batch_sizes)
        prov = _parse_provenance(provenance)
        # A `Nitro` built for this export has no run to end it, so `Done` is published when the
        # export is over to let the experiment's Terminal monitors tear down a data server or
        # close a logger. Not for a run's `Nitro`, which the run may still evaluate or export.
        finish = experiment !== nothing
        try
            _export_task(state, nitro, resolved, dir, name, bs, prov, provenance_root)
        finally
            finish && _finish_export_nitro!(nitro)
        end
    end
    return "started run $(state.id) (kind=export). Poll `nitro_status(run_id=\"$(state.id)\")`."
end

# Publish `Done` on a `Nitro` that exists only for an export, so its Terminal monitors run. Never
# throws: the export's result is already decided.
function _finish_export_nitro!(nitro::Nitro)
    try
        # Idempotent; a monitor registered after setup would otherwise be missed.
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
configuration with no arguments. The tool form of `ReactantNitro.setup_devices!`. `backend` is
`"cpu"`, `"gpu"`, `"cuda"`, `"rocm"` or `"tpu"`; omit it for Reactant's default. `n_devs` pins how
many visible devices runs shard over; visibility is fixed for the process, so restrict GPUs with
`CUDA_VISIBLE_DEVICES` at session start. The pin beats an experiment's `n_devs`; an explicit
keyword on `nitro_train` still wins for that run.

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

# Is a Kaimon gate bound in this process? KaimonGate's signal is private and has moved once
# (`_RUNNING[]` to `_running()`), so both spellings are probed. When neither exists this warns once
# rather than answering `false`: a silent `false` switched the whole `nitro_*` surface off against
# a live gate with nothing in the log to say why. The real fix is a public predicate upstream.
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
        # `force = true` keeps registration working in headless processes, where `serve` would
        # otherwise skip on its interactivity guard.
        KaimonGate.serve(force = true, tools = _build_tools())
        return true
    catch e
        @warn "ReactantNitro ReactantNitroKaimonGateExt: registering the nitro_* tools failed" exception = e
        return false
    end
end

function _install_with_retry()
    _gate_running() && _install_tools() && return true
    # No gate yet normally means one is about to arrive: the host loads the model package (firing
    # this `__init__`) before it calls `serve`, so this loop is the path that actually registers.
    # No `isinteractive()` guard, since a gate host runs `julia -e '<preamble>'` where that is
    # false; the cost is one idle task that wakes once a second for 30 s in a process that never
    # serves a gate. The real fix is an additive registration hook upstream.
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

Register the `nitro_*` GateTools with the Kaimon gate, unconditionally, starting a gate if none is
running. Called automatically on load; call it again after a manual `KaimonGate.stop()`/`serve()`
cycle or when another extension's `serve(tools=...)` replaced this session's tools, as
`Base.get_extension(ReactantNitro, :ReactantNitroKaimonGateExt).reinstall_kaimon_tools()`.
"""
function reinstall_kaimon_tools()
    return _install_tools()
end

# In `__init__` rather than at top level: the module body runs once at precompile time, where no
# gate exists.
function __init__()
    _install_with_retry()
    return nothing
end

end # module ReactantNitroKaimonGateExt
