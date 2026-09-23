# Phases.jl
#
# The phase tree, the monitor registry (module-level plus per-run), the `Nitro` handle and its
# display, and progress reporting.

# ── The phase tree ─────────────────────────────────────────────────────────────────
#
# Three groupings, each with more than one child and a real query behind it (`p isa Compiling`).
# No single-child parents. Users may extend it: `struct Preprocessing <: Phase end`.

"""
    Phase

Root of the run-phase tree. The framework publishes phase transitions through
[`register_phase_monitor!`](@ref) so an external heartbeat, watchdog or dashboard can be written
without the framework shipping one. Phases differ in duration by orders of magnitude (a compile
takes minutes, a step milliseconds), so a watchdog needs the signal; what counts as too long is
site policy and no timeout table ships here.
"""
abstract type Phase end

"""
    Repl <: Phase

The caller has control and no framework work is in flight: a REPL prompt between calls, or the
moment after any public entry point returns. Without it a monitor cannot tell a finished run from
a process wedged in teardown.

Published but never recorded: `Repl` is a property of the process, not of the run, so
[`phase`](@ref) keeps answering how the run ended (`Done` or `Failed`). Published only by the
outermost entry point, through a process-level depth counter ([`work_in_flight`](@ref)), so
`train!` calling `validate` each epoch does not announce an idle session mid-run.
"""
struct Repl <: Phase end

"Setup is running: the setup sequence, before the first compile."
struct Starting <: Phase end

"""
    Compiling <: Phase

Supertype of the four compile phases, which are **slow by design**. Subscribe here rather than to
a leaf so that adding a leaf stays non-breaking.
"""
abstract type Compiling <: Phase end

"Tracing and compiling the gradient program."
struct GradCompiling <: Compiling end

"Tracing and compiling the optimizer program."
struct OptCompiling <: Compiling end

"Tracing and compiling the eval-mode `forward` / `metrics` programs."
struct EvalCompiling <: Compiling end

"""
    ExportCompiling <: Compiling

Tracing and compiling the export program, published once per `export_model` call around the
backend's `write_export`, since export never touches the compile cache. The previous phase is
restored when the bundle is written or the call fails.
"""
struct ExportCompiling <: Compiling end

"""
    Stepping <: Phase

Supertype of the two per-batch phases. A run alternates between them.
"""
abstract type Stepping <: Phase end

"Running training steps."
struct TrainStepping <: Stepping end

"""
    EvalStepping <: Stepping

Running the validation, testing, or inference loop.
"""
struct EvalStepping <: Stepping end

"Writing a checkpoint."
struct Checkpointing <: Phase end

"""
    Terminal <: Phase

Supertype of the two end states. `p isa Terminal` is the one query a monitor needs in order to
close itself.
"""
abstract type Terminal <: Phase end

"The run finished, whether by reaching `max_epochs` or by stopping early."
struct Done <: Terminal end

"The run raised."
struct Failed <: Terminal end

# ── The `Nitro` handle ─────────────────────────────────────────────────────────────

"""
    Nitro(e; kwargs...) -> Nitro

The run's materialized state. `Nitro(e)` executes the setup sequence and nothing else, so
[`validate`](@ref), [`evaluate`](@ref) and [`predict`](@ref) work with no training anywhere in the
process; `train!(e)` is sugar for `train!(Nitro(e))`, and every keyword belongs here.

```julia
nitro = Nitro(e)                                  # fresh weights from build_model
nitro = Nitro(e; checkpoint = "runs/x/latest")    # trained weights, no training this process
nitro = Nitro(e; data = (; test = loader))        # supply data directly, skip build_data
```

Opaque, with accessors: [`experiment`](@ref), [`parameters`](@ref), [`states`](@ref),
[`run_dir`](@ref), [`current_step`](@ref), [`current_epoch`](@ref), [`phase`](@ref),
[`history`](@ref), and [`request_stop!`](@ref) to write. The object is a `Nitro`; a run is what
happens when you `train!` one, which is why `run_dir`, `run_id` and the run phases keep the word.
It deliberately holds no Enzyme shadow `dps`; that lives inside the gradient program.
"""
mutable struct Nitro
    # Setup products, in the order the setup sequence produces them.
    e::Any                    # step 5, the POST-conversion experiment; every hook call sees this
    model::Any                # step 6
    ps::Any                   # step 6, then restored or trained; a Lux tree
    st::Any                   # step 6
    w0::Any                   # step 7, the anchor for `decay_anchor = :w0`
    layout::Any               # step 8, the flat permutation and the per-group ranges
    opt_state::Any            # step 9, TRAINING ONLY; NTuple{G}
    data::Any                 # step 3, the named data collection
    routing::Any              # step 10, the resolved `Router`s
    schema::Any               # step 10, the first batch's field names; routing rule 4 checks it
    mesh::Any                 # step 4.5, the Sharding.Mesh; `nothing` at n_devs == 1
    programs::Any             # the handle-local thunk memo; keys ARE the compile-cache keys
    schedules::Any            # step 11, each factory already called with the horizon
    total::Any                # step 11, the schedule horizon; `nothing` with no train split
    batch_size::Any           # inferred from the first batch, never read from config
    logger::Any               # step 12
    sections::Any             # the binding report as `TableSection`s, appended to `show`
    anchor_checksum::Any      # the per-group decay-anchor checksum; `nothing` if unanchored
    preset::Any               # the named configuration this run claimed; `nothing` if none
    # The resolved checkpoint the restore read (the file `resume = :auto` found, not the symbol),
    # or `nothing` for fresh weights. Export stamps it into the bundle.
    checkpoint_source::Any

    # The run that trained these weights, from the restored record. A `checkpoint = path`
    # construction gets a fresh logger, so this handle's own `run_id` would name the exporting
    # process rather than the training run. `nothing` for fresh weights.
    trained_run_id::Any
    trained_run_url::Any

    # Configuration the handle carries, one field per constructor keyword.
    run_dir::String
    seed::Int
    accum::Int
    max_epochs::Int
    gradient_clip_norm::Any
    checkpointer::Any
    early_stop::Any
    # What the pure scalar run accessors returned at construction, for `fixed_config_report`. The
    # stored field cannot be compared against the accessor: a passed keyword is indistinguishable
    # from the accessor default by the time the body runs.
    accessors_at_setup::Any
    # Every live-dispatch component of the compile-cache key, resolved once; see `frozen_dispatch`.
    # With these frozen, the programs a `Nitro` uses are fixed at construction.
    frozen::Any
    # `(; masks, anchors)`, resolved at construction so a revised `no_decay` cannot change an
    # existing handle's numerics.
    decay::Any

    # Mutated as it runs.
    g_accum::Any              # NTuple{G}, first written by the gradient program
    monitors::Any             # the per-run copy of the module-level registry
    step::Int
    epoch::Int
    phase::Phase
    stop_requested::Bool
    stop_reason::Any          # also a record field: how the run ended, `nothing` while it runs
    # The last validation metrics, so a handle can say how it did without a logger backend. Host
    # values, the last epoch's only; the series is `history`.
    last_metrics::Any         # the last validation metrics; `(;)` until a run produces some
    # Wall seconds of the most recent `train!` on this handle; `nothing` before one.
    elapsed::Any
    # `(; path, epoch, metric, score)`, resolved once at the end of `train!` so `show` does no I/O.
    best_checkpoint::Any
    # `(; experiment, epoch, step, run_dir)` for a handle built with `weights = other_nitro`, else
    # `nothing`. Distinct from `checkpoint_source`, which names a file.
    weights_source::Any
    # One row per validated epoch this handle produced: `(; epoch, step, loss, metrics...)`, with
    # `loss` the epoch's mean train loss. Fresh per handle; a resumed run's earlier epochs belong to
    # the process that trained them.
    history::Vector{NamedTuple}
    # Whether `release!` has run on this handle's splits; set by the first `Terminal` or an explicit
    # `release!(nitro)`, so the sources are released once however many times either happens.
    sources_released::Bool
end

# ── The accessors, which are reads and which everything downstream is specified in terms of.


"""
    experiment(nitro) -> e

The **post-conversion** experiment, which is what every hook except `build_data` and
`derive` sees. Its `Device` fields hold device values.
"""
experiment(nitro::Nitro) = nitro.e

"""
    parameters(nitro) -> ps

The parameters, as the Lux tree `build_model` returned. After `train!` these are the
trained values.
"""
parameters(nitro::Nitro) = nitro.ps

"""
    states(nitro) -> st

The layer state, including running statistics. Plural because `state` reads as the handle's
own state, which is the whole `Nitro`.
"""
states(nitro::Nitro) = nitro.st

"""
    run_dir(nitro) -> String

The run's output directory: checkpoints, the manifest, and `resume = :auto` all resolve against
it. It is the one path concept in the framework, and defaults to
`joinpath("runs", string(nameof(typeof(e))))`.
"""
run_dir(nitro::Nitro) = nitro.run_dir

"""
    current_step(nitro) -> Int

The **optimizer** step, not the micro-batch. With `accum = N` the two differ by a factor of
N, and confusing them shifts the whole learning-rate curve by N.
"""
current_step(nitro::Nitro) = nitro.step

"""
    current_epoch(nitro) -> Int

The current epoch, `0` before the first one begins.
"""
current_epoch(nitro::Nitro) = nitro.epoch

"""
    phase(nitro) -> Phase

The run's current phase. Phase *transitions* also reach the monitor registry; this reads
the latest one.
"""
phase(nitro::Nitro) = nitro.phase

"""
    request_stop!(nitro) -> nothing

Ask the run to stop, from a REPL or from a phase monitor. It sets the same flag
[`EarlyStopping`](@ref) sets, checked once per epoch after validation.

**Stopping is graceful:** the run finishes the epoch, validates, checkpoints, finalizes the logger,
and exits through the normal [`Done`](@ref) path. Aborting mid-epoch would skip exactly the steps
that make the run useful.
"""
function request_stop!(nitro::Nitro)
    nitro.stop_requested = true
    return nothing
end

"""
    logger_info(nitro) -> NamedTuple

The run's logger's key identifying parameters, read straight off a running experiment:
[`logger_info`](@ref)`(nitro.logger)`. The shipped default [`JSONLogger`](@ref) reports its
`path`; a hosted backend reports its URL, experiment key, workspace, and whatever else it
extends the verb with. The ReactantNitroKaimonGateExt `nitro_logger` tool renders exactly this table.
"""
logger_info(nitro::Nitro) = logger_info(nitro.logger)

# ── Showing a `Nitro` never shows the weights ───────────────────────────────────────
#
# Six fields are parameter trees and `data` is the whole collection, so the default struct `show`
# would print every weight; in an agent session that is context window spent to learn which epoch
# a run is on. The summary is an allowlist, and every read in it is a host read: the parameter
# count comes from `layout.lengths`, never from the arrays.

# ── One table shape, several renderers ──────────────────────────────────────────────
#
# Every long `show` produces a title and a list of sections, each with an optional column header
# and rows of strings; rendering is a separate decision that depends on where Julia is running.
# The handle summary and the binding report are one table, and a section is a labelled band inside
# one frame. Columns are global, so the widest cell anywhere sets a column for every section; that
# is why the frame draws no vertical rules and why `parameter groups` reads as prose rather than
# seven columns. A `Ref` because the renderer is process-wide; Render.jl installs the default.
const _TABLE_RENDERER = Ref{Any}(nothing)

"""
    ReactantNitro.TableRows

The row type every renderer receives, `Vector{Vector{String}}`: one inner vector per row, already
rendered to strings. Spelled as an alias so a renderer states the same type.
"""
const TableRows = Vector{Vector{String}}

"""
    ReactantNitro.CellStyles

A section's per-cell styling, `Dict{Tuple{Int, Int}, Symbol}` from `(row, column)` into `rows` to
one of the roles [`TableSection`](@ref) documents. Empty for a section that wants none.
"""
const CellStyles = Dict{Tuple{Int, Int}, Symbol}

"""
    ReactantNitro.TableSection(title, header, rows[, styles])

One labelled band of a table. `title` is drawn full-width above the section, or `""` for none
(the leading section, whose label would repeat the table title). `header` is the column header; an
all-empty vector means a label-and-value band. `styles` maps a `(row, column)` to a ROLE (`:good`,
`:busy`, `:warn`, `:bad`, `:muted`, `:accent`), never a colour, so the same rows can be rendered
into the logged report's plain string and a destination that cannot colour ignores the map. Short
rows are read as having empty cells.
"""
struct TableSection
    title::String
    header::Vector{String}
    rows::TableRows
    styles::CellStyles
end

TableSection(title, header, rows) = TableSection(title, header, rows, CellStyles())

"""
    ReactantNitro.table_renderer!(f) -> previous

Set the renderer every long `show` goes through, and return the previous one. `f` is called as
`f(io::IO, mime::MIME, title::AbstractString, sections::Vector{TableSection},
note::Union{AbstractString, Nothing})`, with `mime` either `MIME"text/plain"` or
`MIME"text/html"`; write those types out, since it is reached through a `Ref{Any}`. `nothing`
switches the table off, leaving the title, the note, and one line saying so. The framed
PrettyTables renderer is installed at definition and is the only one shipped.
"""
function table_renderer!(f)
    prev = _TABLE_RENDERER[]
    _TABLE_RENDERER[] = f
    return prev
end

# No second renderer: a plain fallback was a second display path to keep in step.
function _render_sections(
        io::IO, mime::MIME, title::AbstractString, sections::Vector{TableSection};
        note::Union{AbstractString, Nothing} = nothing
    )
    r = _TABLE_RENDERER[]
    r === nothing || return r(io, mime, title, sections, note)
    print(io, title)
    print(io, "\n  (table display is off; `ReactantNitro.table_renderer!` reinstalls one)")
    note === nothing || print(io, "\n  ", note)
    return nothing
end

# The one-section call, which is what a display with nothing to divide up wants: the experiment
# table, and a caller assembling a single band by hand.
_render_table(
    io::IO, mime::MIME, title::AbstractString, header::Vector{String}, rows::TableRows;
    note::Union{AbstractString, Nothing} = nothing
) = _render_sections(io, mime, title, [TableSection("", header, rows)]; note)

# Every long display answers both MIME types: a terminal asks for text, a notebook for HTML and
# falls back to text. ONE METHOD PER MIME, never a `Union`: `show(io, ::Union{...}, ::T)` is
# ambiguous with Base's own `show(io, ::MIME"text/plain", x)`.

# `nothing` rather than a guess for a handle whose layout is not a `FlatLayout`: a display reports
# what it can read and invents nothing.
function _nitro_params(nitro::Nitro)
    lay = nitro.layout
    lay === nothing && return nothing
    return try
        (sum(lay.lengths), length(lay.groups))
    catch
        nothing
    end
end

function _nitro_devices(nitro::Nitro)
    nitro.mesh === nothing && return 1
    return try
        length(nitro.mesh.device_ids)
    catch
        nothing
    end
end

# A merely built handle reports `Starting` forever, which reads as "in progress". Display only:
# `phase(nitro)` still returns `Starting()`, because the phase tree is the monitor contract.
function _nitro_phase_label(nitro::Nitro)
    p = nitro.phase
    p isa Starting && nitro.step == 0 && nitro.elapsed === nothing &&
        return "Created  (nothing has run on this handle)"
    return string(nameof(typeof(p))) *
        (nitro.stop_reason === nothing ? "" : " (" * string(nitro.stop_reason) * ")")
end

# ── What each phase means, as one of the table's roles ──────────────────────────────
#
# The phase is the cell a reader checks against an expectation, so it earns colour. Mapped by what
# the phase says about the run, not by its place in the tree. `Repl` is muted, not good: idle is
# neither progress nor a problem.
_phase_role(p::Phase) = :busy
_phase_role(::Terminal) = :good
_phase_role(::Failed) = :bad
_phase_role(::Repl) = :muted
_phase_role(::Starting) = :muted

# The stop reason overrides the phase, since both ways a run can end badly end on `Done`.
# `:completed` is the ordinary success and must not divert; `:error` is bad; an early exit is
# worth seeing, not alarming about; an unrecognized reason warns.
function _phase_role(p::Phase, stop_reason)
    (stop_reason === nothing || stop_reason === :completed) && return _phase_role(p)
    stop_reason === :error && return :bad
    return :warn
end

function _nitro_elapsed(sec)
    sec === nothing && return nothing
    sec < 60 && return string(round(sec; digits = 1)) * "s"
    m, sc = divrem(round(Int, sec), 60)
    m < 60 && return string(m) * "m " * lpad(string(sc), 2, '0') * "s"
    h, mm = divrem(m, 60)
    return string(h) * "h " * lpad(string(mm), 2, '0') * "m"
end

# Where these weights came from and whether this handle trained them are two questions. The four
# wordings share a shape: an origin, and `trained here` in front of it when this handle trained.
function _nitro_weights(nitro::Nitro)
    src = nitro.checkpoint_source
    ws = nitro.weights_source
    origin = ws !== nothing ?
        "from Nitro(" * string(ws.experiment) * ") epoch " * string(ws.epoch) * ", " * ws.run_dir :
        src === nothing ? "from build_model" : "restored from " * string(src)
    nitro.elapsed === nothing && return src === nothing && ws === nothing ?
        "fresh " * origin : origin
    return "trained here, " * (src === nothing ? origin : "resumed " * origin)
end

# A split's size without iterating it. A loader that promises no length is "streaming" rather
# than counted, since `length` on one may consume the split the next epoch was going to read.
function _nitro_split(v)
    return try
        Base.IteratorSize(typeof(v)) isa Union{Base.HasLength, Base.HasShape} ?
            string(length(v)) : "streaming"
    catch
        "?"
    end
end

function Base.show(io::IO, nitro::Nitro)
    pg = _nitro_params(nitro)
    print(
        io, "Nitro(", nameof(typeof(nitro.e)),
        ", ", nameof(typeof(nitro.phase)),
        ", epoch ", nitro.epoch, "/", nitro.max_epochs,
        ", step ", nitro.step,
        pg === nothing ? "" : ", " * _commas(pg[1]) * " params",
        ", ", repr(nitro.run_dir), ")",
    )
    return nothing
end

# One table: the binding report is part of the handle's display rather than a second box beside
# it. A value appears in exactly one section: the state band holds what the handle carries, the
# binding bands hold where each configured value came from.
Base.show(io::IO, ::MIME"text/plain", nitro::Nitro) = _show_nitro(io, MIME"text/plain"(), nitro)
Base.show(io::IO, ::MIME"text/html", nitro::Nitro) = _show_nitro(io, MIME"text/html"(), nitro)

function _show_nitro(io::IO, mime::MIME, nitro::Nitro)
    pg = _nitro_params(nitro)
    devs = _nitro_devices(nitro)
    state = TableRows(
        [
            ["phase", _nitro_phase_label(nitro)],
            ["epoch", string(nitro.epoch, " / ", nitro.max_epochs)],
            [
                "step",
                string(nitro.step) *
                    (nitro.total === nothing ? "" : " / " * string(nitro.total)),
            ],
            [
                "params",
                pg === nothing ? "not built" :
                    _commas(pg[1]) * " in " * string(pg[2]) * (pg[2] == 1 ? " group" : " groups"),
            ],
            ["batch_size", string(something(nitro.batch_size, "pending"))],
            [
                "devices",
                (devs === nothing ? "sharded" : string(devs)) *
                    (nitro.mesh === nothing ? "  (no mesh)" : "  (mesh :data)"),
            ],
        ]
    )
    el = _nitro_elapsed(nitro.elapsed)
    el === nothing || push!(state, ["elapsed", el * "  (this `train!` call)"])
    push!(state, ["weights", _nitro_weights(nitro)])
    push!(state, ["run_dir", nitro.run_dir])
    push!(state, ["seed", string(nitro.seed, "   accum ", nitro.accum)])
    nitro.preset === nothing || push!(state, ["preset", string(nitro.preset)])

    # The phase is checked against an expectation and `weights` has silently been wrong before, so
    # those two earn colour; a number is read, not checked, and stays plain.
    styles = CellStyles(
        (1, 2) => _phase_role(nitro.phase, nitro.stop_reason),
        (findfirst(r -> r[1] == "weights", state), 2) =>
            nitro.elapsed === nothing ? :muted : :good,
    )
    sections = [TableSection("", String[], state, styles)]

    # The selected checkpoint, with the path on a row of its own: it is the longest cell in the
    # table, and the one a reader wants to copy. `relpath` touches no filesystem.
    bc = nitro.best_checkpoint
    bc === nothing || push!(
        sections, TableSection(
            "checkpoint", String[], TableRows(
                [
                    [
                        "selected",
                        string("epoch ", bc.epoch, ", ", bc.metric, " ", _shown(bc.score)),
                    ],
                    ["path", _nitro_relpath(bc.path)],
                ]
            ),
            # The path is what a reader is here to copy, so it is the accent; the epoch and score
            # above it are the justification for that path and read as ordinary text.
            CellStyles((2, 2) => :accent)
        )
    )

    # A row per metric, like everything else in the frame.
    isempty(nitro.last_metrics) || push!(
        sections, TableSection(
            "metrics  (validation, epoch " * string(nitro.epoch) * ")", String[],
            TableRows([[string(k), _shown(v)] for (k, v) in pairs(nitro.last_metrics)])
        )
    )

    # The binding sections built at setup. `nothing` on a hand-assembled handle in a test, and a
    # `show` that can throw is a `show` nobody can use while debugging.
    nitro.sections === nothing || append!(sections, nitro.sections)

    # Named where the question is asked; the summary is not any of these.
    note = "ask it for more with `history`, `experiment`, `parameters`, `states`, `logger_info`"
    title = "Nitro for " * string(nameof(typeof(nitro.e)))
    _render_sections(io, mime, title, sections; note)
    return nothing
end

# Relative to the working directory when it helps and absolute when it does not: a path that
# climbed out through a pile of `..` is worse than the one it replaced.
function _nitro_relpath(path)
    return try
        rel = relpath(path, pwd())
        startswith(rel, "..") ? path : rel
    catch
        path
    end
end


# ── Monitor registry, not dispatch ──────────────────────────────────────────────────
#
# Dispatch would allow one monitor; two independent observers need a registry, at module and run
# level. They are monitors rather than callbacks because every real use watches a run and never
# steers it: a heartbeat writer, a stall watchdog, a progress bar, a dashboard feed.

"""
    ReactantNitro.RegisteredMonitor

One entry in a phase registry: the function, an id, and whether it has already been warned about,
which is what makes error isolation once per monitor rather than once per event.
"""
mutable struct RegisteredMonitor
    id::Int
    f::Any
    warned::Bool
end

"""
    ReactantNitro.MonitorHandle

What [`register_phase_monitor!`](@ref) returns: the id and the registry it was added to, so
[`unregister_phase_monitor!`](@ref) needs no second argument.
"""
struct MonitorHandle
    id::Int
    registry::Vector{RegisteredMonitor}
end

# The module-level registry and its lock. It and the compile cache are the two process-global
# structures, both lock-guarded so concurrent runs cannot corrupt each other's bookkeeping.
const MONITORS = RegisteredMonitor[]
const MONITOR_LOCK = ReentrantLock()
const MONITOR_ID = Ref(0)

next_monitor_id() = lock(MONITOR_LOCK) do
    MONITOR_ID[] += 1
end

"""
    register_phase_monitor!(f) -> handle
    register_phase_monitor!(nitro, f) -> handle

Register `f` as an observer of phase transitions. The module-level form applies to future runs; the
`nitro` form targets the live per-run copy and is effective immediately. This is the hook a
heartbeat or a watchdog is written against; the framework ships neither, since what counts as "too
long" is site policy.

The signature is `f(phase, step, epoch, info)` with `info::NamedTuple`, not keyword arguments,
because `do`-block functions cannot take keywords:

```julia
register_phase_monitor!() do phase, step, epoch, info
    phase isa Compiling && @info "slow phase, do not kill me" phase
    phase isa Terminal  && close(my_heartbeat)
end
```

`step` and `epoch` are `nothing` until the first epoch begins. `info` carries at least
`(; nitro, logger, is_rank0)`, plus `metrics` on the transition out of [`EvalStepping`](@ref). A
throwing monitor never kills a run: it is caught and warned about once per monitor. Module-level
monitors fire in registration order, then the run's own. The per-run copy is a snapshot taken at
`train!`, so unregistering a module-level handle mid-run takes effect on the next `train!`.

The framework fires on transition only. A compile is a blocking foreign call, so nothing can emit
from inside it; a monitor that must prove liveness needs its own task on the `:interactive` pool,
and [`Compiling`](@ref) is published so it can widen its patience.
"""
function register_phase_monitor!(f)
    m = RegisteredMonitor(next_monitor_id(), f, false)
    lock(MONITOR_LOCK) do
        push!(MONITORS, m)
    end
    return MonitorHandle(m.id, MONITORS)
end

function register_phase_monitor!(nitro::Nitro, f)
    m = RegisteredMonitor(next_monitor_id(), f, false)
    lock(MONITOR_LOCK) do
        push!(nitro.monitors, m)
    end
    return MonitorHandle(m.id, nitro.monitors)
end

"""
    unregister_phase_monitor!(handle) -> nothing

Remove a monitor registered by [`register_phase_monitor!`](@ref), by the handle it returned. The
handle carries the registry it was added to, so one verb removes from either without the caller
having to say which, and removing an already-removed monitor is a no-op rather than an error.
"""
function unregister_phase_monitor!(handle::MonitorHandle)
    lock(MONITOR_LOCK) do
        i = findfirst(m -> m.id == handle.id, handle.registry)
        i === nothing || deleteat!(handle.registry, i)
    end
    return nothing
end

"""
    ReactantNitro.adopt_monitors!(nitro) -> nothing

The per-run copy of the module-level registry, taken on the way into every entry point and again
inside `train!`. Entries are copied, so "warn once per monitor" means once per run. Monitors
registered directly on the `Nitro` are kept and fire after the module-level ones; re-adopting is
idempotent.
"""
function adopt_monitors!(nitro::Nitro)
    lock(MONITOR_LOCK) do
        own = nitro.monitors::Vector{RegisteredMonitor}
        adopted = [
            RegisteredMonitor(m.id, m.f, false)
                for m in MONITORS if !any(x -> x.id == m.id, own)
        ]
        prepend!(own, adopted)
    end
    return nothing
end

"""
    ReactantNitro.set_phase!(nitro, phase; info...) -> nothing

Move the run to `phase` and publish the transition, as one operation and from the only writer.
Fires on transition only, so re-entering the current phase is a no-op. Extra keywords are merged
into `info`, which is how the transition out of [`EvalStepping`](@ref) carries `metrics`.
"""
function set_phase!(nitro::Nitro, phase::Phase; info...)
    nitro.phase === phase && return nothing
    nitro.phase = phase
    # A compile produces no units of work, so the bar would sit at zero and read as a hang; the
    # reporter is told the phase instead.
    progress_phase!(phase isa Compiling ? _compiling_label(phase) : "")
    fire_monitors(nitro, phase; info...)
    # After the monitors, so one that reads a metric off the data source still can.
    phase isa Terminal && release_sources!(nitro)
    return nothing
end

# Which program is compiling, in words: the gradient compile is the long one and the eval compile
# is the one that surprises people mid-run.
_compiling_label(::GradCompiling) = "compiling gradient"
_compiling_label(::OptCompiling) = "compiling optimizer"
_compiling_label(::EvalCompiling) = "compiling eval"
_compiling_label(::ExportCompiling) = "compiling export"
_compiling_label(p::Compiling) = "compiling " * string(nameof(typeof(p)))

"""
    ReactantNitro.publish_phase(nitro, phase; info...) -> nothing

Fire the monitor registry for `phase` without recording it on the handle. Exists for
[`Repl`](@ref), which is a property of the process rather than of the run: writing it into
`nitro.phase` would erase how the run ended, while a heartbeat still needs the event. No transition
guard, since there is no stored phase to compare against; the depth counter keeps this to one
event per outermost call.
"""
function publish_phase(nitro::Nitro, phase::Phase; info...)
    fire_monitors(nitro, phase; info...)
    phase isa Terminal && release_sources!(nitro)
    return nothing
end

"""
    ReactantNitro.release!(nitro) -> nothing

Release every split of the handle's data collection through the source trait's
[`release!`](@ref) method, once per handle. `train!` does this itself at `Done` or `Failed`; call
it for a handle that only validated, predicted or exported, whose sources would otherwise stay
open. A split that raises is warned about and never takes the others, or the run's result, with
it. After this the handle cannot iterate a released source again.
"""
release!(nitro::Nitro) = release_sources!(nitro)

function release_sources!(nitro::Nitro)
    nitro.sources_released && return nothing
    nitro.sources_released = true
    for name in keys(nitro.data)
        src = prefetch_source(getproperty(nitro.data, name))
        try
            release!(src)
        catch err
            @warn "ReactantNitro: `release!` raised on the `$name` split; the run's result is \
                   unaffected, but whatever the source held may still be open." source = typeof(src) exception = (err, catch_backtrace())
        end
    end
    return nothing
end

"""
    ReactantNitro.publish_phase(phase::Phase; info...) -> nothing

Publish through the module-level monitors, for the window in which there is no handle: `Nitro(e)`
itself, where `build_data` may start and compile a data server. A monitor adopted into the run
later is the same object, so the events arrive in order.
"""
function publish_phase(phase::Phase; info...)
    ms = lock(() -> copy(MONITORS), MONITOR_LOCK)
    isempty(ms) && return nothing
    payload = merge((; nitro = nothing, logger = nothing, is_rank0 = true), NamedTuple(info))
    for m in ms
        try
            m.f(phase, nothing, nothing, payload)
        catch err
            m.warned && continue
            m.warned = true
            @warn """
            ReactantNitro: a phase monitor threw on a handle-free publish. Construction is
            unaffected, because monitors observe and never control. This is the
            ONLY warning it gets.
            """ phase exception = (err, catch_backtrace())
        end
    end
    return nothing
end

# The shared delivery half of both verbs.
function fire_monitors(nitro::Nitro, phase::Phase; info...)
    ms = nitro.monitors
    isempty(ms) && return nothing
    # `nothing` until an epoch has begun, rather than a zero indistinguishable from a real one.
    started = nitro.epoch > 0
    payload = merge(
        (; nitro, logger = nitro.logger, is_rank0 = rank(nitro.mesh) == 0),
        NamedTuple(info)
    )
    for m in copy(ms)
        try
            m.f(phase, started ? nitro.step : nothing, started ? nitro.epoch : nothing, payload)
        catch err
            # Error isolation: a throwing monitor never kills a run, and is warned about once.
            m.warned && continue
            m.warned = true
            @warn """
            ReactantNitro: a phase monitor threw. The run is unaffected, because monitors
            observe and never control. It stays registered and will keep
            being called, since a monitor that throws on one phase may be fine on the
            others, but this is the ONLY warning it gets in this run.
            """ phase exception = (err, catch_backtrace())
        end
    end
    return nothing
end

# ── The `Repl` phase's depth counter ────────────────────────────────────────────────
#
# Process-level, not per-`Nitro`: it has to cover `Nitro(e)` itself, and the question a monitor asks
# is whether any work is in flight in the process. Guarded by `MONITOR_LOCK`.

const REPL_DEPTH = Ref(0)

"""
    ReactantNitro.work_in_flight() -> Bool

Whether any public entry point is currently executing in this process; `false` is what
[`Repl`](@ref) announces. A monitor needs this as well as the transition, because an entry point
that throws before it has a `Nitro` leaves the last published phase in place, and an interval
monitor can reconcile against this predicate.
"""
work_in_flight() = lock(MONITOR_LOCK) do
    REPL_DEPTH[] > 0
end

# ── The progress counter ────────────────────────────────────────────────────────────
#
# One monotonic count of completed units of work: per optimizer step in training, per batch in
# evaluation. Atomic and process-level, like `REPL_DEPTH`.

const PROGRESS = Threads.Atomic{Int}(0)

"""
    ReactantNitro.progress_counter() -> Int

A monotonic count of units of work this process has completed: one per optimizer step in
training, one per batch in evaluation. Only its change is meaningful. A stall watchdog needs this
because a phase deadline measures from the transition, and a run stays in `TrainStepping` for a
whole epoch; comparing this across two observations turns the same budget into "time since the
last completed unit". It covers evaluation because nothing on the handle advances during an eval
loop.
"""
progress_counter() = PROGRESS[]

# ── The progress reporter, the counter's display half ────────────────────────────────
#
# A bar needs what the counter cannot supply: how many units a stretch will take, and when it
# starts and stops. Those are `progress_begin!`/`progress_end!`; the per-unit advance is
# `note_progress!`. A `Ref` with a default installed at load.
const _PROGRESS_REPORTER = Ref{Any}(nothing)

"""
    ReactantNitro.progress_reporter!(f) -> previous

Set the progress reporter and return the previous one. `nothing` disables progress output; the
default is [`default_progress_reporter`](@ref). `f` is called as
`f(verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int)`, with `verb` one of:

  * `:begin`, once per stretch of work, with its `label` (`"train"`, or the split name), its
    `total` units (`0` when unknown), and the epoch position.
  * `:step`, once per completed unit. The other arguments are placeholders.
  * `:end`, once when the stretch finishes, however it finishes.
  * `:done`, once when the last stretch of an entry point is over, so a reporter reusing one
    terminal line can close it. `label` is the closing text, or `""` for the default.
  * `:phase`, when the current stretch starts or stops doing something that produces no units;
    `label` names it (`"compiling gradient"`) or is `""` when ordinary work resumes.
  * `:note`, when the note user code set with [`progress_note!`](@ref) or
    [`with_progress_note`](@ref) changes; `label` is the note, `""` when there is none. A new
    `:phase` or `:begin` clears it.

A reporter that throws is switched off rather than propagated: losing GPU hours to a broken
progress bar is not a trade this framework makes.
"""
function progress_reporter!(f)
    prev = _PROGRESS_REPORTER[]
    _PROGRESS_REPORTER[] = f
    return prev
end

# Held for every call: `progress_note!` may come from a hook's worker threads while the driver
# reports steps and phases.
const _PROGRESS_LOCK = ReentrantLock()

function _progress_report(verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int)
    r = _PROGRESS_REPORTER[]
    r === nothing && return nothing
    lock(_PROGRESS_LOCK)
    try
        r(verb, label, total, epoch, max_epochs)
    catch err
        # Off, and said once; leaving it installed would repeat the failure every step.
        _PROGRESS_REPORTER[] = nothing
        @warn "ReactantNitro: the progress reporter threw and has been switched off for this \
               process. The run is unaffected. Reinstall one with `progress_reporter!`." exception =
            (err, catch_backtrace())
    finally
        unlock(_PROGRESS_LOCK)
    end
    return nothing
end

"""
    ReactantNitro.progress_begin!(label, total, epoch, max_epochs) -> nothing
    ReactantNitro.progress_end!() -> nothing

Open and close one stretch of reported work. Both are no-ops with no reporter installed, which is
the default. `total = 0` means the length is not known ahead of time.
"""
progress_begin!(label::AbstractString, total::Integer, epoch::Integer, max_epochs::Integer) =
    lock(_PROGRESS_LOCK) do
    _clear_notes!("")
    _progress_report(:begin, String(label), Int(total), Int(epoch), Int(max_epochs))
end

progress_end!() = _progress_report(:end, "", 0, 0, 0)

"""
    ReactantNitro.with_progress_stretch(f, label, total, epoch, max_epochs)

Run `f` as one reported stretch: [`progress_begin!`](@ref) around it and [`progress_end!`](@ref)
in a `finally`, so a stretch that threw does not leave its bar on the terminal.
"""
function with_progress_stretch(
        f, label::AbstractString, total::Integer, epoch::Integer, max_epochs::Integer
    )
    progress_begin!(label, total, epoch, max_epochs)
    return try
        f()
    finally
        progress_end!()
    end
end

"""
    ReactantNitro.progress_done!(label = "") -> nothing

Tell the reporter that the last stretch of an entry point is over, closing with `label` when it is
not empty. Separate from [`progress_end!`](@ref) because only the entry point knows whether
another stretch is coming.
"""
progress_done!(label::AbstractString = "") = _progress_report(:done, String(label), 0, 0, 0)

"""
    ReactantNitro.progress_phase!(label) -> nothing

Tell the reporter what the current stretch is doing while its counter is not moving, or `""` when
it is back to ordinary work. A compile emits no units, so without this a bar stalls at zero for the
length of an XLA compile with nothing to say why.
"""
progress_phase!(label::AbstractString) = lock(_PROGRESS_LOCK) do
    label == _NOTE_PHASE[] || _clear_notes!(String(label))
    _progress_report(:phase, String(label), 0, 0, 0)
end

# ── Notes, a stack user code pushes onto ─────────────────────────────────────────────
#
# The top entry is what the reporter draws. Entries carry an id so a pop after a phase has cleared
# the stack is a no-op. Guarded by `_PROGRESS_LOCK`, and cleared where the reporter clears its
# note: a new stretch, or a changed phase.
const _NOTES = Pair{Int, String}[]
const _NOTE_ID = Ref(0)
const _NOTE_PHASE = Ref("")
const _NOTE_LAST = Ref(0.0)
# `progress_note!` may be called per item, so its frames are throttled to this. Pushes and pops
# always draw, or a finished block could leave its note on screen.
const _NOTE_INTERVAL = 0.1

function _clear_notes!(phase::String)
    empty!(_NOTES)
    _NOTE_PHASE[] = phase
    _NOTE_LAST[] = 0.0
    return nothing
end

function _report_note!()
    _NOTE_LAST[] = time()
    return _progress_report(:note, isempty(_NOTES) ? "" : last(_NOTES).second, 0, 0, 0)
end

"""
    progress_note!(msg) -> nothing

Show `msg` beside the current phase of the progress display, as
`setup [building data] (decoding 1200/5000)`. Inside [`with_progress_note`](@ref) it replaces
that block's note; outside one it sets a note that lasts until the next phase or stretch. Cheap
enough to call per item: it redraws at most ten times a second.
"""
function progress_note!(msg::AbstractString)
    lock(_PROGRESS_LOCK) do
        if isempty(_NOTES)
            push!(_NOTES, (_NOTE_ID[] += 1) => String(msg))
        else
            _NOTES[end] = first(_NOTES[end]) => String(msg)
        end
        time() - _NOTE_LAST[] >= _NOTE_INTERVAL && _report_note!()
    end
    return nothing
end

"""
    with_progress_note(f, msg)

Run `f` with `msg` shown beside the current phase, and restore the enclosing note when it returns
or throws. Nested blocks stack, and the innermost one is shown:

```julia
with_progress_note("loading index") do
    for (i, f) in enumerate(files)
        with_progress_note(() -> decode(f), "decoding \$i/\$(length(files))")
    end
end
```

A new phase or stretch clears every note, including those of blocks still open.
"""
function with_progress_note(f, msg::AbstractString)
    id = lock(_PROGRESS_LOCK) do
        push!(_NOTES, (_NOTE_ID[] += 1) => String(msg))
        _report_note!()
        _NOTE_ID[]
    end
    return try
        f()
    finally
        lock(_PROGRESS_LOCK) do
            i = findlast(n -> first(n) == id, _NOTES)
            i === nothing || (deleteat!(_NOTES, i); _report_note!())
        end
    end
end

# On the hot path: an atomic add, then one `Ref` load and a branch. Here rather than at a second
# call site so the bar and the watchdog cannot disagree about what a unit of work is.
function note_progress!()
    Threads.atomic_add!(PROGRESS, 1)
    _PROGRESS_REPORTER[] === nothing || _progress_report(:step, "", 0, 0, 0)
    return nothing
end

# ── Run progress, the model both reporters draw ──────────────────────────────────────
#
# One bar per entry point: the name is the current stretch and phase, the fraction is epochs
# completed with the running stretch interpolated, never moving backwards. A run without an epoch
# budget reports each stretch's own fraction, or none for a stretch with no units.

mutable struct _RunProgress
    label::String
    phase::String
    note::String
    epoch::Int
    max_epochs::Int
    total::Int
    counter::Int
    floor::Float64
end
_RunProgress() = _RunProgress("", "", "", 0, 0, 0, 0, 0.0)

function _begin_stretch!(r::_RunProgress, label::String, total::Int, epoch::Int, max_epochs::Int)
    r.label, r.phase, r.note, r.epoch, r.max_epochs = label, "", "", epoch, max_epochs
    r.total, r.counter = total, 0
    return r
end

function _run_fraction(r::_RunProgress)
    within = r.total > 0 ? min(1.0, r.counter / r.total) : 0.0
    r.max_epochs > 0 || return r.total > 0 ? within : nothing
    r.floor = max(r.floor, min(1.0, (r.epoch - 1 + within) / r.max_epochs))
    return r.floor
end

function _run_name(r::_RunProgress)
    name = r.max_epochs > 0 ? "epoch $(r.epoch)/$(r.max_epochs): $(r.label)" : r.label
    isempty(r.phase) || (name *= " [" * r.phase * "]")
    return isempty(r.note) ? name : name * " (" * r.note * ")"
end

_done_name(r::_RunProgress, label::String = "") =
    !isempty(label) ? label : r.max_epochs > 0 ? "done: $(r.epoch)/$(r.max_epochs) epochs" : "done"

# ── The terminal bar ─────────────────────────────────────────────────────────────────

# The one live bar and its run. ONE LINE IS REUSED for the whole run through ProgressMeter's
# `keep = false`, and `:done` prints the final frame with the newline nobody else prints.
const _BAR = Ref{Any}(nothing)
const _BAR_RUN = Ref{Union{Nothing, _RunProgress}}(nothing)
const _BAR_DIRTY = Ref(false)
# The bar's counter runs to this, so a run fraction maps onto a fixed resolution.
const _BAR_RES = 1000

# One colour for both draws, or a name-only line beside a green bar reads as a different display.
const _BAR_COLOR = :green

# Drawn only where a person is watching: in a captured transcript a bar is thousands of carriage
# returns. Read at each `:begin`, since a session can gain or lose a terminal.
_drawing_progress() = isinteractive() && (stderr isa Base.TTY)

# A frame. A run without a fraction draws its name alone, since `ProgressUnknown` reads as hung.
# `force = true` because an unforced frame within `dt` of the last is thrown away.
function _bar_frame!(r::_RunProgress)
    f = _run_fraction(r)
    name = _run_name(r)
    if f === nothing
        _BAR[] = nothing
        ProgressMeter.printover(stderr, name, _BAR_COLOR)
        return nothing
    end
    p = _BAR[]
    if p === nothing
        p = ProgressMeter.Progress(_BAR_RES; desc = name * " ", output = stderr, color = _BAR_COLOR)
        _BAR[] = p
    end
    p.core.desc = name * " "
    ProgressMeter.update!(p, round(Int, f * _BAR_RES); keep = false, force = true)
    return nothing
end

"""
    ReactantNitro.progress_bar_reporter(verb, label, total, epoch, max_epochs) -> nothing

The built-in terminal reporter: one bar per entry point on one terminal line, filled by the run's
fraction so the ETA is the run's, described as `epoch 3/40: train [compiling gradient]` and ending
as `done: 40/40 epochs`. Draws only when the session is interactive and `stderr` is a terminal.
"""
function progress_bar_reporter(
        verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int
    )
    if verb === :begin
        r = _BAR_RUN[]
        if r === nothing
            _drawing_progress() || return nothing
            r = _RunProgress()
            _BAR_RUN[] = r
            _BAR_DIRTY[] = true
        end
        _begin_stretch!(r, label, total, epoch, max_epochs)
        _bar_frame!(r)
    elseif verb === :phase
        r = _BAR_RUN[]
        r === nothing && return nothing
        r.phase == label && return nothing
        r.phase, r.note = label, ""
        _bar_frame!(r)
    elseif verb === :note
        r = _BAR_RUN[]
        (r === nothing || r.note == label) && return nothing
        r.note = label
        _bar_frame!(r)
    elseif verb === :step
        r = _BAR_RUN[]
        (r === nothing || r.total <= 0) && return nothing
        r.counter += 1
        p = _BAR[]
        f = _run_fraction(r)
        # ProgressMeter throttles these by its own `dt`; only the announcing frames are forced.
        p === nothing || f === nothing ||
            ProgressMeter.update!(p, round(Int, f * _BAR_RES); keep = false)
    elseif verb === :done
        r = _BAR_RUN[]
        _BAR_RUN[] = nothing
        p = _BAR[]
        _BAR[] = nothing
        r === nothing && return nothing
        _BAR_DIRTY[] = false
        if p === nothing
            # A name-only run: the line is still open, so the final word and the newline.
            ProgressMeter.printover(stderr, _done_name(r, label), _BAR_COLOR)
            println(stderr)
        else
            # Not `finish!`, which is a no-op once the counter has reached the total and would
            # leave the last stretch's name on the line. `keep = true` prints the newline.
            p.core.desc = _done_name(r, label) * " "
            ProgressMeter.update!(p, _BAR_RES; keep = true, force = true)
        end
    end
    return nothing
end

# ── The log reporter, for a notebook ─────────────────────────────────────────────────
#
# Emits ProgressLogging records, which Pluto, VS Code and TerminalLoggers render and other loggers
# drop, from the same run model, under one id per entry point. Steps are throttled to ten frames a
# second; the opening and closing frames always go out.

mutable struct _ProgressLog
    const id::UUIDs.UUID
    const run::_RunProgress
    last_emit::Float64
end

const _PLOG = Ref{Union{Nothing, _ProgressLog}}(nothing)
const _PLOG_INTERVAL = 0.1

# The record `@logprogress` emits, both halves: the `progress` keyword is the old API Pluto and
# VS Code read, the `ProgressString` message is the new one TerminalLoggers reads.
function _plog_emit!(st::_ProgressLog; done::Bool = false, label::String = "")
    fraction = _run_fraction(st.run)
    name = done ? _done_name(st.run, label) : _run_name(st.run)
    msg = ProgressLogging.ProgressString(
        ProgressLogging.Progress(st.id, fraction; name, done)
    )
    Logging.@logmsg ProgressLogging.ProgressLevel msg progress = (done ? "done" : fraction) _id =
        st.id
    st.last_emit = time()
    return nothing
end

# `ProgressLevel` is below `Info`, so the stdlib `ConsoleLogger` drops the records and a CI log
# stays clean. Read at each `:begin` because `with_logger` changes the answer per call.
_logging_progress() =
    Logging.min_enabled_level(Logging.current_logger()) <= ProgressLogging.ProgressLevel

"""
    ReactantNitro.progress_log_reporter(verb, label, total, epoch, max_epochs) -> nothing

The reporter for an environment that renders log records: Pluto, VS Code, a REPL with
TerminalLoggers. Emits `ProgressLogging.Progress` records under one id per entry point, opened at
the first `:begin` and closed with `done = true` at `:done`, with the same name and fraction
[`progress_bar_reporter`](@ref) draws. Emits regardless of whether anything is listening; the
logger drops what it does not accept.
"""
function progress_log_reporter(
        verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int
    )
    if verb === :begin
        st = _PLOG[]
        if st === nothing
            st = _ProgressLog(UUIDs.uuid4(), _RunProgress(), 0.0)
            _PLOG[] = st
        end
        _begin_stretch!(st.run, label, total, epoch, max_epochs)
        _plog_emit!(st)
    elseif verb === :phase
        st = _PLOG[]
        st === nothing && return nothing
        st.run.phase == label && return nothing
        st.run.phase, st.run.note = label, ""
        _plog_emit!(st)
    elseif verb === :note
        st = _PLOG[]
        (st === nothing || st.run.note == label) && return nothing
        st.run.note = label
        _plog_emit!(st)
    elseif verb === :step
        st = _PLOG[]
        (st === nothing || st.run.total <= 0) && return nothing
        st.run.counter += 1
        if st.run.counter >= st.run.total || time() - st.last_emit >= _PLOG_INTERVAL
            _plog_emit!(st)
        end
    elseif verb === :done
        st = _PLOG[]
        _PLOG[] = nothing
        st === nothing || _plog_emit!(st; done = true, label)
    end
    return nothing
end

# ── The default: a terminal first, a logger second, silence otherwise ────────────────

# Chosen at `:begin` and held to `:end`, so one stretch is never half-drawn and half-logged.
const _PROGRESS_ROUTE = Ref{Symbol}(:none)

"""
    ReactantNitro.default_progress_reporter(verb, label, total, epoch, max_epochs) -> nothing

The progress reporter installed at load. At each stretch's `:begin` it picks, in order:
[`progress_bar_reporter`](@ref) when the session is interactive and `stderr` is a terminal;
[`progress_log_reporter`](@ref) when the current logger accepts ProgressLogging's level, which is
a notebook or a REPL with TerminalLoggers; nothing otherwise, which is a CI log or a captured
transcript. The choice holds for the stretch. Install either reporter directly with
[`progress_reporter!`](@ref) to force one.
"""
function default_progress_reporter(
        verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int
    )
    if verb === :begin
        _PROGRESS_ROUTE[] = _drawing_progress() ? :bar : _logging_progress() ? :log : :none
    end
    route = _PROGRESS_ROUTE[]
    if verb === :done
        # Both may hold state to close: the bar its line, the log an open stretch.
        progress_bar_reporter(verb, label, total, epoch, max_epochs)
        progress_log_reporter(verb, label, total, epoch, max_epochs)
        _PROGRESS_ROUTE[] = :none
    elseif route === :bar
        progress_bar_reporter(verb, label, total, epoch, max_epochs)
    elseif route === :log
        progress_log_reporter(verb, label, total, epoch, max_epochs)
    end
    return nothing
end

# Returns whether this entry was the OUTERMOST one, mirroring `repl_exit!`, so an entry point can
# declare `Starting` exactly once however deeply the verbs nest.
repl_enter!() = lock(MONITOR_LOCK) do
    REPL_DEPTH[] += 1
    return REPL_DEPTH[] == 1
end

# Clamped at zero: a negative counter would read as "work in flight" forever.
repl_exit!() = lock(MONITOR_LOCK) do
    REPL_DEPTH[] > 0 && (REPL_DEPTH[] -= 1)
    return REPL_DEPTH[] == 0
end

"""
    ReactantNitro.with_repl(f, nitro; spawn = true, on_interrupt = nothing) -> f()

Run `f`, then publish [`Repl`](@ref) through `nitro` if this was the outermost entry point. The
wrapper every public entry point goes through.

Long loops leave the interactive thread: a compile or execute there starves every other task on
it, including logger tasks and a Kaimon gate's message loop. When the caller is on an interactive
thread and a default pool exists, `f` runs on a default-pool worker and the caller parks in
`fetch`; single-threaded Julia and worker-thread callers run inline. `spawn = false` forces inline,
for rendering hooks that need the main thread.

Ctrl+C interrupts the parked wait, not the run, and `on_interrupt` decides what it means; the
default rethrows. A failed run surfaces its own exception, unwrapped from `TaskFailedException`.
The depth decrement and the `Repl` publish happen in a `finally`, so a failed run still hands
control back and a supervised session does not stay alive on a stale counter; `Failed` is published
first.
"""
function with_repl(f, nitro; spawn::Bool = true, on_interrupt = nothing)
    # `Starting` on the way in, outermost only, so the stretch before a verb's first compile is
    # declared rather than charged against whatever budget was already running.
    if repl_enter!()
        # Adopt before publishing, so a monitor registered between `Nitro(e)` and the verb sees
        # this event. Idempotent.
        adopt_monitors!(nitro)
        publish_phase(nitro, Starting())
    end
    try
        if spawn && _should_spawn()
            t = Threads.@spawn f()
            return _repl_wait(t, nitro, on_interrupt)
        end
        return f()
    finally
        repl_exit!() && publish_phase(nitro, Repl())
    end
end

# Spawn only when the caller is on an interactive thread and there is a thread outside that pool
# to move to.
function _should_spawn()
    return Threads.nthreads() > Threads.nthreads(:interactive) &&
        Threads.threadid() <= Threads.nthreads(:interactive)
end

"""
    ReactantNitro._off_interactive(f) -> f()

Run `f` off the interactive thread under [`with_repl`](@ref)'s spawn policy, for `Nitro(e)`, which
`with_repl` cannot wrap because there is no handle until the constructor returns. Construction
blocks the calling thread for a minute or more (PJRT init, `build_model`, the device conversion),
and from the main task under `julia -t N,1` that thread is the interactive one, so an external
supervisor's heartbeat stopped and the session was reclaimed before the first step. No
`on_interrupt`: a ^C during construction has nothing graceful to stop.
"""
function _off_interactive(f)
    _should_spawn() || return f()
    t = Threads.@spawn f()
    try
        return fetch(t)
    catch e
        throw(unwrap_task_exc(e))
    end
end

function _repl_wait(t::Task, nitro, on_interrupt)
    try
        return fetch(t)
    catch e
        # SIGINT lands in the parked caller, never in the worker; hand it to the entry point.
        e isa InterruptException || throw(unwrap_task_exc(e))
        on_interrupt === nothing && rethrow()
        return on_interrupt(t, nitro)
    end
end

"""
    ReactantNitro.with_repl_result(f) -> f()

The `Repl` wrapper for an entry point that has no `Nitro` until it returns one. Nothing is
published when `f` throws, since there is no handle; the counter is still released, and
[`work_in_flight`](@ref) is how an interval monitor corrects.
"""
function with_repl_result(f)
    repl_enter!()
    local result
    try
        result = f()
    catch
        repl_exit!()
        rethrow()
    end
    repl_exit!() && result isa Nitro && publish_phase(result, Repl())
    return result
end
