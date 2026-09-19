# Phases.jl
#
# The phase tree, the monitor registry (module-level plus per-run), and the `Nitro` handle. There
# is NO `CompilingFused` leaf: the framework compiles no fused program, and a phase that never
# arrives is a monitor that hangs.

# ── The phase tree ─────────────────────────────────────────────────────────────────
#
# Three groupings, each with more than one child and a real query behind it (`p isa Compiling`).
# No single-child parents. Users may extend it: `struct Preprocessing <: Phase end`.

"""
    Phase

Root of the run-phase tree. The framework publishes phase transitions through
[`register_phase_monitor!`](@ref) so an external heartbeat, watchdog, progress display, or dashboard
can be written without the framework shipping one. The registry is named for what gets built on it:
a **monitor** is anything that watches a run and never steers it.

This is framework-level because phases differ in duration by four orders of magnitude: a compile
takes hundreds of seconds while a training step takes milliseconds, so a watchdog with a fixed
timeout kills runs mid-compile. The [`Compiling`](@ref) supertype is the general part of that
signal; expected durations are site policy and no timeout table ships here.
"""
abstract type Phase end

"""
    Repl <: Phase

**The caller has control and no framework work is in flight**: a REPL prompt between
calls, or the moment after any public entry point returns. It is the phase a
long-lived process spends most of its wall clock in, and the only one the framework
publishes on the way *out* of its own code rather than on the way in.

It exists because a monitor cannot otherwise tell "finished, waiting for me" from
"still working". Without it the last thing a monitor sees after a run is
[`Terminal`](@ref), which is indistinguishable from a process wedged during teardown,
so a watchdog either kills a healthy idle session or waits forever on a dead one. What
"too long to sit idle" means is site policy, exactly as for every other phase, so no
budget ships here.

**It is PUBLISHED but never RECORDED**, the one place the framework separates those.
`Repl` is a property of the PROCESS rather than of the run, so [`phase`](@ref) keeps
answering how the run ended: `phase(nitro) isa Done` after a successful [`train!`](@ref)
and `isa Failed` after one that raised. Writing `Repl` into that field would erase the
outcome in order to record something that was never about the run, and a freshly
constructed handle reports [`Starting`](@ref) rather than `Repl` for the same reason.
[`publish_phase`](@ref) is the verb, and is worth reading for where the split is drawn.

**Published only by the OUTERMOST entry point.** [`train!`](@ref) calls
[`validate`](@ref) once per epoch and [`render`](@ref) calls [`predict`](@ref) per
batch; if an inner return published `Repl`, every epoch boundary would announce an idle
session in the middle of a run, and a monitor that widens its patience while idle would
stop enforcing the per-step budget for the rest of the run. A process-level depth
counter is what prevents that; [`work_in_flight`](@ref) is the same counter, readable.
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

Tracing and compiling the export program, published once per `export_model` call rather than per
compile-cache miss. Export retraces by design and never touches the compile cache, so there is no
cache miss to key a transition on: the phase wraps the backend's `write_export`, which is where
the one CPU compile per batch size happens, and the previous phase is restored when the bundle is
written (or the call fails, since export publishes no [`Failed`](@ref) of its own).
"""
struct ExportCompiling <: Compiling end

"""
    Stepping <: Phase

Supertype of the two per-batch phases. A run alternates between them.
"""
abstract type Stepping <: Phase end

"Running training steps."
struct TrainStepping <: Stepping end

"Running the validation, testing, or inference loop."
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

The run's **materialized state**, and the public constructor for it: `Nitro(e)` executes the setup
sequence and nothing else, so [`validate`](@ref), [`evaluate`](@ref), and [`predict`](@ref) all
work with no training anywhere in the process. `train!(e)` is sugar for `train!(Nitro(e))`.

**Every keyword belongs here and `train!(nitro)` has none.** This constructor is the single
authority for the keyword defaults; `train!(e; kwargs...)` forwards everything to it.

Three constructions cover the cases with no training in them:

```julia
nitro = Nitro(e)                                  # fresh weights from build_model
nitro = Nitro(e; checkpoint = "runs/x/latest")    # trained weights, no training this process
nitro = Nitro(e; data = (; test = loader))        # supply data directly, skip build_data
```

**Opaque, with accessors.** Users never construct or mutate one field by field; read it through
[`experiment`](@ref), [`parameters`](@ref), [`states`](@ref), [`run_dir`](@ref),
[`current_step`](@ref), [`current_epoch`](@ref), [`phase`](@ref), and [`history`](@ref), and
write to it through [`request_stop!`](@ref).

**On the name.** The object is a `Nitro`; a *run* is what happens when you `train!` one. That is why
`run_dir`, `run_id`, `run_url`, `run_ref`, and the "run phases" all keep `run`: every one of them
names the event. `Trainer` would be the conventional choice and is the wrong one, since this object
is constructed for evaluation and serving with no training anywhere in the process.

**What it deliberately does not hold is the Enzyme shadow `dps`**, which is allocated inside the
gradient program on every invocation and never crosses a boundary. A `dps` field here is
the natural way to write that bug.

!!! note "Field types are deliberately loose"
    The field list is fixed; the types are `Any` wherever tightening them would have meant
    depending on a piece that did not exist yet. The accessors below are the surface everything
    else should go through.
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
    # WHERE THESE WEIGHTS CAME FROM: the resolved checkpoint the restore actually read, or `nothing`
    # for freshly initialized weights. Resolved rather than echoed, which is the point: it is the
    # file `resume = :auto` DISCOVERED, not the symbol that asked for it, so a provenance stamp
    # naming it names something a reader can open. Setup already computed this to run the
    # compatibility checks against; retaining it is what lets export stamp the checkpoint without
    # asking a caller to hand back a path it already passed to this constructor.
    checkpoint_source::Any

    # WHICH RUN TRAINED THESE WEIGHTS, from the restored record rather than from this handle's own
    # logger, and that distinction is the whole reason these exist. A `checkpoint = path`
    # construction gets a FRESH logger, so `run_id(nitro)` names the process doing the exporting and
    # not the run that produced the weights: stamping that into a bundle would confidently point a
    # reader at an experiment holding an export trace and no training metrics at all. The record
    # holds the training run's id and url (`CheckpointRecord.run_id`, `.run_url`), setup already
    # reads it for the compatibility check, and retaining these two scalars is what lets a
    # manifest STATE the experiment instead of implying it through a run directory's name.
    # `nothing` for freshly initialized weights, and for a record written by a run with no logger.
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
    # What the PURE scalar run accessors returned at construction, for `fixed_config_report`'s
    # divergence check. Comparing the stored field against the accessor cannot work:
    # a `Nitro` keyword defaults to the accessor call, so by the time the constructor body runs,
    # a keyword that was passed is indistinguishable from one that was not, and `run_dir =
    # mktempdir()` would report as a drifted accessor on every single run. `clip_source` documents the
    # same trap. Comparing accessor-THEN against accessor-NOW separates the two exactly: only a real
    # redefinition moves it.
    accessors_at_setup::Any
    # Every LIVE-DISPATCH component of the compile-cache key, resolved once here. A `NamedTuple`
    # of `(; worlds_train, worlds_opt, worlds_eval, tm_residency, metrics_residency)` rather than
    # five fields, because this struct is already wide and a positional constructor is how it is
    # built. With these frozen, EVERY component of the key derives from stored state, which is what
    # makes "the programs a `Nitro` uses are fixed at construction" an invariant rather than an
    # observation. See `frozen_dispatch` for what each entry covers and why the three world tuples
    # are not one.
    frozen::Any
    # The two per-run decay buffers, `(; masks, anchors)`, resolved at construction. `masks` is the
    # per-leaf exclusion `no_decay` produced, `nothing` without a train split; `anchors` is the
    # per-group `:w0` slice or `nothing`. Both were once rebuilt at every `train!` entry, which was
    # harmless while both were pure functions of frozen state and is not once a USER HOOK feeds
    # one: recomputing would let a revised `no_decay` change an existing handle's numerics, which
    # is exactly what freezing the dispatch state exists to forbid.
    decay::Any

    # Mutated as it runs.
    g_accum::Any              # NTuple{G}, first written by the gradient program
    monitors::Any             # the per-run copy of the module-level registry
    step::Int
    epoch::Int
    phase::Phase
    stop_requested::Bool
    stop_reason::Any          # also a record field: how the run ended, `nothing` while it runs
    # WHAT THE RUN ACTUALLY PRODUCED, retained so that a handle can answer it without a logger
    # backend. The shipped `JSONLogger` writes these to a file and a hosted backend sends them
    # away, so before this the numbers a run computed were reachable only by opening something
    # else, and a REPL `train!(n)` returned a handle that could say it had finished and not how it
    # had done. Host values, asserted so by `run_eval`, and the LAST epoch's rather than a series:
    # a handle is not a metrics store, and the series is what a logger is for.
    last_metrics::Any         # the last validation metrics; `(;)` until a run produces some
    # Wall seconds of the most recent `train!` ON THIS HANDLE, and `nothing` before one. Not the
    # run's total across a resume: this call is the only thing this handle timed, and adding a
    # restored duration to it would report a number no clock ever measured.
    elapsed::Any
    # The checkpoint the run would hand you: `(; path, epoch, metric, score)`, resolved ONCE at the
    # end of `train!` rather than on demand. A `show` must not do I/O, and this is the one piece of
    # the summary that lives on disk instead of in the handle.
    best_checkpoint::Any
    # WHERE TRANSFERRED WEIGHTS CAME FROM: `(; experiment, epoch, step, run_dir)` when the handle was
    # built with `weights = other_nitro`, else `nothing`. Distinct from `checkpoint_source`, which
    # names a FILE a restore read; this names a handle in the same process, which has no path. The
    # `weights` row of `show` and the export provenance both read it.
    weights_source::Any
    # ONE ROW PER VALIDATED EPOCH this handle's `train!` calls produced: `(; epoch, step, loss,
    # metrics...)`, where `loss` is the epoch's mean train loss over its micro-batches and the rest
    # are that epoch's finalized validation metrics as host values. `last_metrics` above is the last
    # of these; the series exists so that `history(nitro)` can answer "how did it go" without a
    # logger backend. Fresh per handle: a resumed run's earlier epochs belong to the process that
    # trained them, and `history` says so.
    history::Vector{NamedTuple}
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

# ── Showing a `Nitro` NEVER shows the weights ───────────────────────────────────────
#
# The failure `CheckpointRecord`'s `show` exists to prevent, one level up and bigger. A `Nitro` is
# a mutable struct of thirty-odd `Any` fields, and six of them (`model`, `ps`, `st`, `w0`,
# `opt_state`, `g_accum`) are parameter trees, while `data` is the whole loaded collection. The
# default struct `show` walks all of it, so evaluating a bare `nitro`, the most ordinary thing
# anyone does with a handle, prints every weight in the model. In a REPL that is a lost screen; in
# an agent session the REPL's output IS the transcript, so it is context window, and it gets spent
# to learn which epoch a run is on.
#
# So the summary is the show, and it is an ALLOWLIST rather than the record's denylist. That is
# deliberate: nearly every field here is either bulk or an internal whose printed form tells a
# reader nothing, so the short list of what is worth showing is the one that stays correct as
# fields are added. A new field is invisible here until someone decides it belongs, which is the
# right default for a display.
#
# EVERY READ BELOW IS A HOST READ, and that is a requirement rather than an accident. `show` runs
# on every REPL expression, and a display that moved device memory would make looking at a handle
# cost transfers. The parameter count comes from `layout.lengths`, host metadata computed once at
# setup, and never from the arrays it describes.

# ── One table shape, several renderers ──────────────────────────────────────────────
#
# Every long `show` in this package produces the same thing: a title and a list of SECTIONS, each
# with an optional column header and rows of strings. Rendering that is a separate decision from
# deciding WHAT to show, and it is the decision that depends on where Julia is running: a log file
# wants aligned columns, and a session with PrettyTables loaded can have one framed table.
# Keeping the two apart means a new destination is a renderer rather than another copy of every
# `show` in the package.
#
# ── Why sections, and why they share their columns ──────────────────────────────────
#
# The handle summary and the binding report used to be separate displays, each drawing its own
# box, so looking at a run meant reading five boxes of three different widths with their titles
# floating between them. They are one table now, and a section is the unit that makes that
# possible: a labelled band inside one frame, carrying its own column header.
#
# The columns are GLOBAL, shared by every section, and that is a constraint rather than a
# preference. A framed table has one column structure, so a section cannot set its own widths, and
# the widest cell anywhere in a column sets that column for all of them. The consequence to design
# around is that a short value in one section sits in a narrow strip with air to its right, which
# is why the frame draws no vertical rules: unruled air is invisible, and a rule through it is
# what would make the table look broken. The other consequence is the reason `parameter groups`
# reads as prose rather than seven columns; see `binding_report_sections`.
#
# A `Ref` rather than dispatch, because the renderer is a process-wide setting with no argument to
# dispatch on, and because an extension setting it in `__init__` is one assignment that cannot
# invalidate anything already compiled.
const _TABLE_RENDERER = Ref{Any}(nothing)

"""
    ReactantNitro.TableRows

The row type every renderer receives, `Vector{Vector{String}}`: one inner vector per row, already
rendered to strings. Spelled as an alias so the core and an extension state the same type rather
than two that happen to agree today.
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

One labelled band of a table: a section title, a column header, its [`TableRows`](@ref), and an
optional [`CellStyles`](@ref).

  * `title` is drawn as a full-width label above the section. **The empty string means no label**,
    which is what the leading section uses: the table's own title already names it, and a band
    labelled immediately under the title reads as a heading printed twice.
  * `header` is the section's column header. An all-empty vector means the section has no column
    names, which is the shape of a label-and-value band like the handle's `state`.
  * `styles` maps a `(row, column)` of `rows` to a ROLE, never to a colour. The roles are
    `:good`, `:busy`, `:warn`, `:bad`, `:muted` and `:accent`.

Rows within a section may be short; a renderer reads missing cells as empty.

**A role rather than a colour, and a lookup rather than an escape in the string.** Two things
follow from it. The text stays plain, which matters because the same rows are rendered into
the logged binding report's string and handed to a logger, where an escape sequence is corruption rather
than styling. And which colour a role gets is the renderer's decision, so a destination that
cannot colour ignores the map entirely rather than having to strip anything out of the cells.
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

Set the renderer every long `show` in this package goes through, and return the previous one.
`f` is called as `f(io::IO, title::AbstractString, sections::Vector{`[`TableSection`](@ref)`},
note::Union{AbstractString, Nothing})`, and `nothing` restores the built-in aligned-column
renderer. **Write those argument types out in the renderer**: it is reached through a `Ref{Any}`,
so nothing checks them for you, and four untyped arguments under a generic name is the signature
that muddles a stack trace and looks applicable to calls that are not this one.

The framed table is the only long display there is: `Reactant` depends on `PrettyTables`, so every
session that loads this package loads it too and this package's extension installs the framed
renderer in its `__init__`. There is no built-in fallback renderer to maintain beside it. With
`nothing` installed, a long `show` prints the table's title, its trailing note, and one line saying
the renderer is missing, which is what a process that loaded neither PrettyTables nor its extension
sees.

**The contract takes sections rather than one header and one set of rows**, which it did until the
handle summary and the binding report became one table. The old five-argument form
`f(io, title, header, rows, note)` is gone, and a renderer still written against it raises a
`MethodError` on the first display rather than being quietly skipped.
"""
function table_renderer!(f)
    prev = _TABLE_RENDERER[]
    _TABLE_RENDERER[] = f
    return prev
end

# CONCRETELY TYPED, all four arguments, and not because this is hot: it is called once per
# display. A renderer is reached through a `Ref{Any}`, so the call is already dynamic, and a
# generically named `render(io, title, sections, note)` with four untyped arguments is the shape
# that makes a stack trace ambiguous and invites a method from somewhere else to look applicable.
# The types are the contract `table_renderer!` documents; an extension writes the same ones.
#
# NO SECOND RENDERER. There was a plain aligned-column fallback here, and it was a second display
# path to keep in step with the framed one for a process that never exists in practice, since
# Reactant loads PrettyTables. With no renderer installed the display says so and prints what it
# can without one: the title and the note, which between them name the handle and the accessors
# that answer the same questions as data.
function _render_sections(
        io::IO, title::AbstractString, sections::Vector{TableSection};
        note::Union{AbstractString, Nothing} = nothing
    )
    r = _TABLE_RENDERER[]
    r === nothing || return r(io, title, sections, note)
    print(io, title)
    print(io, "\n  (no table renderer is installed: `using PrettyTables` renders this display)")
    note === nothing || print(io, "\n  ", note)
    return nothing
end

# The one-section call, which is what a display with nothing to divide up wants: the experiment
# table, and a caller assembling a single band by hand.
_render_table(
    io::IO, title::AbstractString, header::Vector{String}, rows::TableRows;
    note::Union{AbstractString, Nothing} = nothing
) = _render_sections(io, title, [TableSection("", header, rows)]; note)

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

# `Starting` is published at the top of `Nitro` construction and nothing moves it until an entry
# point runs, so a handle that was merely built reports `Starting` forever. That is accurate about
# the phase and misleading as a display: it reads as "in progress" for something that has not begun
# and may never. `Created` is what a constructed, unrun handle is.
#
# DISPLAY ONLY. `phase(nitro)` still returns `Starting()`, because the phase tree is the monitor
# contract and a new type in it would be a change to that contract rather than to this line.
function _nitro_phase_label(nitro::Nitro)
    p = nitro.phase
    p isa Starting && nitro.step == 0 && nitro.elapsed === nothing &&
        return "Created  (nothing has run on this handle)"
    return string(nameof(typeof(p))) *
        (nitro.stop_reason === nothing ? "" : " (" * string(nitro.stop_reason) * ")")
end

# ── What each phase MEANS, as one of the table's roles ───────────────────────────────
#
# The phase is the cell a reader looks at first and the only one whose value they are checking
# against an expectation rather than reading, so it is the cell that earns colour. The mapping is
# by what the phase says about the run and not by the type's place in the tree: `Checkpointing` and
# `Stepping` are unrelated types that both mean "this is moving", and a reader scanning a wall of
# handles wants those to look the same.
#
# `Repl` is MUTED rather than good, deliberately. It means the framework has handed control back
# and is waiting on a person, which is neither progress nor a problem, and a green idle handle in
# a list of running ones is the kind of thing that gets misread at a glance.
_phase_role(p::Phase) = :busy
_phase_role(::Terminal) = :good
_phase_role(::Failed) = :bad
_phase_role(::Repl) = :muted
_phase_role(::Starting) = :muted

# The stop reason overrides the phase, because both ways a run can end badly end on `Done`.
#
# `:completed` IS THE ORDINARY SUCCESSFUL RUN and must not divert, which is the whole reason this
# names the reasons rather than treating "has a reason at all" as the signal: `train!` sets
# `:completed` on every normal finish, so a catch-all painted the success case as a warning. The
# reasons that do divert are `:error`, a failure whatever phase it was recorded on, and an early
# exit (patience, a `request_stop!`), which is a run that ended on purpose before its last epoch:
# worth seeing, not worth alarming about. An unrecognized reason warns, since a run that ended for
# a reason this display cannot name is not one to call green.
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

# Where these weights came from AND whether this handle put training into them, which are two
# different questions that one field cannot answer. `checkpoint_source` is a construction-time
# fact, so reporting it alone made a trained handle claim its weights were "fresh from
# build_model": true of where they started and wrong about what they are.
#
# THE FOUR WORDINGS SHARE A SHAPE on purpose: an origin (`build_model` or a path), and, when this
# handle trained them, `trained here` in front of it. The trained-from-scratch case used to read
# "trained here, from fresh init", which named the origin a fourth way (`init`) that appears
# nowhere else in this display, so the one case a reader most wants to tell from `fresh from
# build_model` was the one phrased least like it.
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

# A split's size WITHOUT iterating it, as the cell under the data band's `batches` column. A
# loader that promises no length is reported as streaming rather than counted: `length` on one is
# wrong at best, and at worst consumes the split that the next epoch was going to read. The word
# rather than a number is also why this is a string: there is no count to give, and a zero would
# be a count that is wrong.
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

# ONE TABLE, and the binding report is part of it rather than a second display printed beside it.
# Both used to render themselves: the handle drew a box of its state and the binding report drew
# one box per section, so a run opened with five frames of three widths and their titles floating
# between them. They answer one question between them (what is this run) and they now share one
# frame, which is also what stops the two from disagreeing about a fact they both print.
#
# The rule that kept them apart is still enforced, and it is about CONTENT rather than layout: a
# value appears in exactly one section. The state band holds what the handle carries, the binding
# bands hold where each configured value came from, and neither repeats the other.
function Base.show(io::IO, ::MIME"text/plain", nitro::Nitro)
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

    # WHAT EARNS COLOUR IN THIS BAND, and the rest deliberately does not. The phase is the cell a
    # reader checks against an expectation rather than reads, and `weights` is the one that has
    # silently been wrong before: a trained handle used to claim `fresh from build_model`, and
    # muting the untrained wording is what makes the two tell apart at a glance. A number is read,
    # not checked, so `epoch`, `step` and `params` stay plain; colouring them would spend the
    # signal on cells that do not carry one.
    styles = CellStyles(
        (1, 2) => _phase_role(nitro.phase, nitro.stop_reason),
        (findfirst(r -> r[1] == "weights", state), 2) =>
            nitro.elapsed === nothing ? :muted : :good,
    )
    sections = [TableSection("", String[], state, styles)]

    # The selected checkpoint: which epoch won, on which metric, and a path short enough to paste.
    # `relpath` is string arithmetic and touches no filesystem, which is what keeps it legal here.
    #
    # A SECTION, and the path on a row of its own, because it used to share a cell with the epoch
    # and the score. That made it the longest cell in the table, and in a frame whose columns are
    # global the longest cell is the one that sets the width every other section is padded to. A
    # path also happens to be the cell a reader wants to select and paste, which a line holding
    # nothing else makes easy.
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

    # Values, not tiles. The grid existed to keep a dozen metrics off a dozen lines of their own
    # box; inside a shared frame a metric is a label and a value like everything else in the
    # state band, and a row each is what lines them up with it.
    isempty(nitro.last_metrics) || push!(
        sections, TableSection(
            "metrics  (validation, epoch " * string(nitro.epoch) * ")", String[],
            TableRows([[string(k), _shown(v)] for (k, v) in pairs(nitro.last_metrics)])
        )
    )

    # WHERE EACH VALUE BOUND, from the report the handle already built at setup. `nothing` on a
    # handle assembled by hand in a test, which is not an error: the bands are omitted and the
    # state band still displays, because a `show` that can throw is a `show` nobody can use while
    # debugging the thing that broke.
    nitro.sections === nothing || append!(sections, nitro.sections)

    # Named where the question is asked, exactly as the record's `show` names `checkpoint_info`:
    # whoever printed this handle wanted one of these and the summary is not it. It rides under
    # the frame, so it stays at the bottom.
    note = "ask it for more with `history`, `experiment`, `parameters`, `states`, `logger_info`"
    # JUST THE NAME. The title used to carry "(the run handle; no weights are shown)", which was
    # a disclaimer for a display that has a `weights` row saying where they came from and a note
    # naming `parameters` as the way to get the arrays themselves. Saying it a third time in the
    # title told a reader nothing the table was not already telling them.
    title = "Nitro for " * string(nameof(typeof(nitro.e)))
    _render_sections(io, title, sections; note)
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
# Dispatch does not scale here: only one monitor could be passed, so two independent observers
# collide. The registry is two levels, module and run, and the types below come before the verbs
# because a method signature is evaluated where it is written.
#
# ON THE NAME. These are `monitor`s rather than `callback`s because "callback" says only how the
# function is invoked, which is the least interesting thing about it, while every real use of this
# registry is a MONITOR: a heartbeat writer, a stall watchdog, a progress bar, a dashboard feed.
# The framework ships none of them and publishes the signal precisely so that they can exist
# outside it, so the registry's name is the only place a reader can learn what it is for.

"""
    ReactantNitro.RegisteredMonitor

One entry in a phase registry: the function, an id the handle refers to, and **whether it has
already been warned about**. The flag is what makes the registry's error isolation "once per
monitor" rather than once per event, which matters because a monitor that throws on a
`TrainStepping` transition would otherwise warn once per epoch for the length of the run.
"""
mutable struct RegisteredMonitor
    id::Int
    f::Any
    warned::Bool
end

"""
    ReactantNitro.MonitorHandle

What [`register_phase_monitor!`](@ref) returns. It holds the registry it was added to as well as
the id, so [`unregister_phase_monitor!`](@ref) needs no second argument naming which registry, and
so a per-run handle cannot silently remove a module-level monitor with the same id.
"""
struct MonitorHandle
    id::Int
    registry::Vector{RegisteredMonitor}
end

# The module-level registry, and the lock concurrent runs require: it and the compile cache are
# the two process-global structures, and both are lock-guarded so concurrent runs cannot corrupt
# each other's bookkeeping. That is the whole of what concurrent runs in one process are for.
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
`nitro` form targets the live per-run copy and is effective immediately (an earlier design's
registry was module-level only, so a monitor registered during a run never fired in it).

**This is the hook a heartbeat or a watchdog is written against.** The framework ships neither, and
that is the point of publishing the signal: what counts as "too long" is site policy, and a
timeout table belongs where the hardware and the operational rules are known rather than in a
general framework. What ships here is the transition, on time and in full, so that whatever
consumes it can live entirely outside this package.

**The signature is `f(phase, step, epoch, info)`** with `info::NamedTuple`, **not** keyword
arguments. `do`-block anonymous functions cannot accept keyword arguments at all, and in `do` syntax
a semicolon separates the argument list from the body, so the keyword form could not be written the
documented way.

```julia
register_phase_monitor!() do phase, step, epoch, info
    phase isa Compiling && @info "slow phase, do not kill me" phase
    phase isa Terminal  && close(my_heartbeat)
end
```

`step` and `epoch` are `Union{Int,Nothing}`, because neither is always defined: **both are `nothing`
until the first epoch begins**, which is what a monitor observing a standalone
[`validate`](@ref) or a transition before the loop sees. `info` carries at least
`(; nitro, logger, is_rank0)`, plus `metrics` on the transition **out of** [`EvalStepping`](@ref),
which is where the finalized numbers first exist, and may grow.

Registry rules: **error isolation** (a throwing monitor never kills a run; it is caught and warned
about once per monitor rather than per event), **per-run scope** (the module-level registry is
copied into the run at `train!`), and **registration order**, documented: module-level monitors
fire in the order they were registered, then the run's own.

**The per-run copy is a snapshot**, which is what "applies to future runs" means for the
module-level form: unregistering a module-level handle mid-run leaves the live run's copy firing,
and takes effect on the next `train!`. Use the `nitro` form to reach a run in flight.

**The framework fires on transition; sustained liveness is the monitor's job.** This is forced by
XLA: a compile is a blocking foreign call, so nothing can emit from inside it, and that is exactly
the window where a watchdog most needs evidence of life. A monitor that writes only when this fires
will look hung during a normal compile, so one that has to prove liveness needs its own task, on
the `:interactive` threadpool, since the `:default` pool is the one blocked in the foreign call.
[`Compiling`](@ref) is published so that such a monitor can widen its patience rather than guess.
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

The per-run copy of the registry, taken on the way into every entry point ([`with_repl`](@ref),
before the `Starting` it publishes) and again inside `train!`'s loop. The entries are **copied**,
not shared, so a run's error isolation is its own: a monitor that threw in a previous run gets one
more chance in this one, which is what makes "warn once per monitor" mean once per monitor per run
rather than once per process.

Monitors already registered directly on this `Nitro` are kept and fire **after** the module-level
ones, and re-adopting is idempotent, so `train!` on the same handle twice does not double every
module-level monitor.
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

Move the run to `phase` and **publish the transition**. Setting the field and firing the
registry are one operation on purpose: a phase recorded but not published is a monitor that misses
it, and this is the only writer.

**Fires on transition only.** The phase leaves are singletons, so re-entering the phase you are
already in is a no-op rather than an event per batch.

Any extra keywords are merged into `info`, which is how the transition out of [`EvalStepping`](@ref)
carries `metrics`.
"""
function set_phase!(nitro::Nitro, phase::Phase; info...)
    nitro.phase === phase && return nothing
    nitro.phase = phase
    # The bar's one blind spot, closed here. A compile produces no units of work, so the bar sits
    # at zero for however long XLA takes, which on a first epoch is most of the wall clock and
    # reads as a hang. The phase tree already knows, and `p isa Compiling` is the documented query
    # for exactly this, so the reporter is told rather than left to guess from a stalled counter.
    progress_phase!(phase isa Compiling ? _compiling_label(phase) : "")
    return fire_monitors(nitro, phase; info...)
end

# Which program is compiling, in words. Worth the mapping rather than printing the type name: the
# gradient compile is the long one and the eval compile is the one that surprises people mid-run,
# so naming them is the difference between "it is busy" and "it is busy with the expected thing".
_compiling_label(::GradCompiling) = "compiling gradient"
_compiling_label(::OptCompiling) = "compiling optimizer"
_compiling_label(::EvalCompiling) = "compiling eval"
_compiling_label(::ExportCompiling) = "compiling export"
_compiling_label(p::Compiling) = "compiling " * string(nameof(typeof(p)))

"""
    ReactantNitro.publish_phase(nitro, phase; info...) -> nothing

Fire the monitor registry for `phase` **without recording it on the handle**. The one case
[`set_phase!`](@ref)'s "record and publish are one operation" rule does not cover.

It exists for [`Repl`](@ref), and the reason is a distinction worth naming: **`Repl` is a property of
the PROCESS, not of the run.** Every other phase answers "what is this run doing", and
[`phase`](@ref) reads it back, which is why `phase(nitro) isa Done` after a successful `train!` and
`isa Failed` after one that raised. That is the handle's record of the outcome and callers depend on
it. `Repl` answers a different question, "does anyone have work in flight in this process", so writing
it into `nitro.phase` would erase how the run ended in order to say something that was never about the
run. A monitor still needs the event, because a heartbeat cannot otherwise tell a finished run from a
process wedged in teardown.

So the two are split: the run's phase is recorded and published, and `Repl` is published only.
`phase(nitro)` keeps naming the outcome; the monitor stream carries both.

No transition guard, unlike `set_phase!`: there is no stored phase to compare against. The depth
counter is what keeps this to one event per outermost call.
"""
function publish_phase(nitro::Nitro, phase::Phase; info...)
    return fire_monitors(nitro, phase; info...)
end

"""
    ReactantNitro.publish_phase(phase::Phase; info...) -> nothing

Publish through the MODULE-LEVEL monitors, for the window in which there is no handle to publish
through: `Nitro(e)` itself. `build_data` runs there, and on this stack that can mean starting and
compiling a data server, so the construction is minutes of real work that no run has declared yet.
Without this a supervisor is still being told whatever was declared before the call, which under a
session that budgets idle time is a budget already counting down.

A monitor adopted into a run later (`adopt_monitors!`) is the same object, so a `Starting` published
here and a `Compiling` published through the handle afterwards reach the same observer in order.
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
    # `nothing` until an epoch has begun: a monitor watching a standalone `validate`, or a
    # transition before the loop, is told there is no step rather than shown a zero it cannot
    # distinguish from a real one.
    started = nitro.epoch > 0
    payload = merge(
        (; nitro, logger = nitro.logger, is_rank0 = rank(nitro.mesh) == 0),
        NamedTuple(info)
    )
    for m in copy(ms)
        try
            m.f(phase, started ? nitro.step : nothing, started ? nitro.epoch : nothing, payload)
        catch err
            # ERROR ISOLATION: a throwing monitor never kills a run. Warned once per monitor, not
            # once per event, or a monitor throwing on every `TrainStepping` transition would emit
            # one warning per epoch for the length of the run.
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
# PROCESS-LEVEL, not per-`Nitro`, for two reasons. It has to cover `Nitro(e)` itself,
# where no handle exists at entry and the setup compile is the longest thing in the
# call; and the question a monitor is actually asking is "is any framework work in
# flight in this PROCESS", which is what makes it right for concurrent runs in one
# process: with two runs live, `Repl` follows the second one finishing, not the
# first. Guarded by `MONITOR_LOCK`, the lock the registry already uses.

const REPL_DEPTH = Ref(0)

"""
    ReactantNitro.work_in_flight() -> Bool

Whether any public entry point is currently executing in this process. `false` means
the caller has control, which is what [`Repl`](@ref) announces.

**A monitor needs this as well as the transition**, because a transition can be missed:
an entry point that throws before it has a `Nitro` to publish through, most obviously a
failing `Nitro(e)`, leaves the last published phase in place. A monitor that re-stamps
on an interval can reconcile against this predicate and correct itself, which is the
difference between "briefly wrong" and "wrong until the process exits".
"""
work_in_flight() = lock(MONITOR_LOCK) do
    REPL_DEPTH[] > 0
end

# ── The progress counter ────────────────────────────────────────────────────────────
#
# One monotonic count of completed units of work, bumped by the training loop per
# optimizer step and by the eval loop per batch. Atomic and process-level, for the same
# reason `REPL_DEPTH` is: the question it answers is about the process.

const PROGRESS = Threads.Atomic{Int}(0)

"""
    ReactantNitro.progress_counter() -> Int

A monotonic count of units of work this process has completed: one per optimizer step in
the training loop, one per batch in the eval loop. Only ever increases, and only its
CHANGE is meaningful.

**A stall watchdog needs this, and the phase alone cannot give it.** A phase deadline
measures from the phase transition, and a run stays in [`TrainStepping`](@ref) for a whole
epoch, so a budget meant as "how long may one step take" is silently applied to the entire
stretch: an epoch longer than that budget is then killed while perfectly healthy. Comparing
this counter across two observations turns the same budget into "time since the last
completed unit of work", which is what such a budget is nearly always meant to express.

It covers evaluation as well as training deliberately. Nothing on the handle advances
during an eval loop, since the batch index is local to it, so a monitor keying off
[`current_step`](@ref) would leave a long validation or test pass looking motionless.
"""
progress_counter() = PROGRESS[]

# ── The progress REPORTER, which is the counter's display half ───────────────────────
#
# `progress_counter` is a watchdog primitive: one monotonic number, no total, no output. It answers
# "is this process still moving" and it is not, and was never, something a person watches. A bar
# needs two more things the counter cannot supply: how many units this stretch of work will take,
# and when the stretch starts and stops. Those are `progress_begin!` and `progress_end!`, and the
# per-unit advance is `note_progress!`, which the loops already call in exactly the right places.
#
# A `Ref` for the same reason `_TABLE_RENDERER` is one, but unlike the table renderer this one has
# a DEFAULT, `progress_bar_reporter`, installed at load. ProgressMeter is an ordinary dependency
# here: its only dependency is Printf, which this package already has, so making it optional would
# have bought conditionality over nothing and left the common case with no progress output.
const _PROGRESS_REPORTER = Ref{Any}(nothing)

"""
    ReactantNitro.progress_reporter!(f) -> previous

Set the progress reporter and return the previous one. `nothing` disables progress output
entirely; the default is [`ReactantNitro.default_progress_reporter`](@ref), which draws a
ProgressMeter bar when a terminal is watching, emits ProgressLogging records when the current
logger accepts them, and does nothing otherwise.

`f` is called as `f(verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int)`, with
`verb` one of:

  * `:begin`, once per stretch of work, carrying that stretch's `label` (`"train"`, or the split
    name for an evaluation), its `total` units, and the epoch position. `total = 0` means the
    length is not known.
  * `:step`, once per completed unit. The other arguments are placeholders.
  * `:end`, once when the stretch finishes, however it finishes.
  * `:done`, once when the LAST stretch of an entry point is over, so a reporter that has been
    reusing one terminal line can close it. Every argument is a placeholder.
  * `:phase`, whenever the current stretch starts or stops doing something that produces no units
    of work. `label` names it (`"compiling gradient"`) or is `""` when ordinary work resumes.


**A reporter that throws is switched off rather than propagated.** It is display, the caller is a
training run, and losing GPU hours to a broken progress bar is not a trade this framework makes.
"""
function progress_reporter!(f)
    prev = _PROGRESS_REPORTER[]
    _PROGRESS_REPORTER[] = f
    return prev
end

function _progress_report(verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int)
    r = _PROGRESS_REPORTER[]
    r === nothing && return nothing
    try
        r(verb, label, total, epoch, max_epochs)
    catch err
        # Off, and said once. Leaving it installed would repeat the failure every step, which turns
        # a cosmetic bug into a flooded log on top of a missing bar.
        _PROGRESS_REPORTER[] = nothing
        @warn "ReactantNitro: the progress reporter threw and has been switched off for this \
               process. The run is unaffected. Reinstall one with `progress_reporter!`." exception =
            (err, catch_backtrace())
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
    _progress_report(:begin, String(label), Int(total), Int(epoch), Int(max_epochs))

progress_end!() = _progress_report(:end, "", 0, 0, 0)

"""
    ReactantNitro.with_progress_stretch(f, label, total, epoch, max_epochs)

Run `f` as one reported stretch of work: [`progress_begin!`](@ref) around it and
[`progress_end!`](@ref) in a `finally`, returning whatever `f` returns.

**The `finally` is the point.** A stretch that threw and never closed leaves its bar on the
terminal for whatever the run prints next to land on top of, and the call sites that need this
most are the ones doing I/O, which is where the throwing happens.
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
    ReactantNitro.progress_done!() -> nothing

Tell the reporter that the last stretch of an entry point is over. Separate from
[`progress_end!`](@ref) because a reporter that redraws one line cannot know, at the end of a
stretch, whether another is coming: an epoch is followed by a validation pass and then by the next
epoch, and only the entry point knows which one was the last.
"""
progress_done!() = _progress_report(:done, "", 0, 0, 0)

"""
    ReactantNitro.progress_phase!(label) -> nothing

Tell the reporter what the current stretch is doing while its counter is not moving, or `""` when
it is back to ordinary work. A compile emits no units, so without this a bar stalls at zero for the
length of an XLA compile with nothing to say why.
"""
progress_phase!(label::AbstractString) = _progress_report(:phase, String(label), 0, 0, 0)

# Bumped on the hot path, so the counter half is an atomic add and nothing else. The reporter half
# is one `Ref` load and a branch, which is nothing against an optimizer step, and it is here rather
# than at a second call site so that the bar and the watchdog can never disagree about what a unit
# of work is. Not exported: the framework's own loops are the only callers.
function note_progress!()
    Threads.atomic_add!(PROGRESS, 1)
    _PROGRESS_REPORTER[] === nothing || _progress_report(:step, "", 0, 0, 0)
    return nothing
end

# The one live bar. There is exactly one stretch of work in flight per process: the loops do not
# nest, and a validation pass inside a training epoch happens after that epoch's bar has closed.
const _BAR = Ref{Any}(nothing)

# Whether a bar has written to the terminal line and not yet closed it. ONE LINE IS REUSED across
# every stretch of a run, which is the difference between a forty-epoch run leaving eighty lines
# of finished bars and leaving one that updates. The mechanism is ProgressMeter's `keep = false`:
# it skips the newline after the final frame, `printover` opens the next frame with a carriage
# return and clears to end of line, and the next bar therefore lands on the same line. What that
# costs is the newline nobody prints at the end, which is what `:done` is for.
const _BAR_DIRTY = Ref(false)

# The live bar's description without any phase suffix, so a `:phase` can be taken back off again.
const _BAR_DESC = Ref("")

# ONE COLOUR, PASSED TO BOTH DRAWS. A counted stretch is drawn by ProgressMeter and a unit-less one
# by `printover`, and their defaults do not agree: a meter defaults to `:green` and `printover` to
# `:color_normal`, so `checkpoint` came out in the terminal's plain text beside a green `train`.
# ProgressMeter does not export its default, so naming it here and handing it to both is what keeps
# the two halves of one display the same colour rather than the same by coincidence.
const _BAR_COLOR = :green

# Drawn only where a person is watching. A bar is a cursor animation: in a CI log, a `nohup` file,
# or a captured gate transcript it renders as thousands of carriage returns and escape codes,
# which is worse than nothing and is precisely what a long unattended training run produces. Read
# at each `:begin` rather than cached at load, because a session can gain or lose a terminal.
_drawing_progress() = isinteractive() && (stderr isa Base.TTY)

function _close_bar!()
    p = _BAR[]
    _BAR[] = nothing
    p === nothing || ProgressMeter.finish!(p; keep = false)
    return nothing
end

"""
    ReactantNitro.progress_bar_reporter(verb, label, total, epoch, max_epochs) -> nothing

The built-in progress reporter, installed by default, and the reference implementation of the
contract [`progress_reporter!`](@ref) documents.

**One bar per stretch of work**, an epoch of training or one pass over an evaluation split,
counting the steps left in THAT stretch. A stretch with `total = 0` emits no units, so it gets its
NAME on the line and no meter: a counter stuck at zero beside a clock that never advances is what
`ProgressUnknown` renders for one, and it reads as hung rather than busy. The epoch position rides in the bar's description as
text rather than as a second bar, because "epoch 3/40" is a fact to read and not a thing to watch
fill. No metrics on the bar: the contract carries a label and counts, and widening it is a change
to the contract rather than to this function.

Draws only when the session is interactive and `stderr` is a terminal. Everything else, including
the per-epoch metrics a run logs, is unaffected either way.
"""
function progress_bar_reporter(
        verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int
    )
    if verb === :begin
        # Defensive, not expected: a stretch that never closed would otherwise leave its bar on
        # the terminal while the next one draws over it.
        _close_bar!()
        desc = max_epochs > 0 ? "$label epoch $epoch/$max_epochs " : "$label "
        _BAR_DESC[] = desc
        on = _drawing_progress()
        on && (_BAR_DIRTY[] = true)
        # A STRETCH WITH NO UNITS GETS NO METER, just its own name on the line.
        #
        # `ProgressUnknown` was the obvious choice and is the wrong one. It formats as
        # `"<desc> <counter>    Time: <elapsed>"`, and a stretch that emits nothing has no counter
        # to show and nothing to drive a redraw, so it renders once as `checkpoint epoch 1/5  0
        # Time: 0:00:00` and then sits there. The zero counter is noise; the frozen clock is worse
        # than noise, because a seven-second write showing `Time: 0:00:00` reads as hung at exactly
        # the moment the label exists to say the opposite.
        #
        # `printover` is the same call the meter itself draws through, so the line is reused and
        # cleared identically and the next stretch's bar lands on top of it as usual.
        if total <= 0
            _BAR[] = nothing
            on && ProgressMeter.printover(stderr, rstrip(desc), _BAR_COLOR)
            return nothing
        end
        p = ProgressMeter.Progress(
            total; desc, output = stderr, enabled = on, color = _BAR_COLOR
        )
        _BAR[] = p
        # AN OPENING FRAME, so the bar is up before the stretch's first unit rather than after it,
        # and `force = true` IS WHAT MAKES IT ONE. The constructor draws nothing, and `update!`
        # alone draws nothing either: `_updateProgress!` returns early unless
        # `force || t > p.tlast + p.dt`, and on a bar constructed microseconds ago `t` is not past
        # `tlast + dt`, so an unforced frame here is silently throttled away. That omission is what
        # made a compile's first seconds show the PREVIOUS frame.
        ProgressMeter.update!(p, p.counter; keep = false, force = true)
    elseif verb === :phase
        p = _BAR[]
        p === nothing && return nothing
        want = isempty(label) ? _BAR_DESC[] : _BAR_DESC[] * "[" * label * "] "
        p.core.desc == want && return nothing
        p.core.desc = want
        # FORCED REDRAW, and `force = true` is load-bearing for the same reason it is on `:begin`.
        # Nothing else will draw this: a compile emits no `next!`, so without a frame here the new
        # description would first appear on the step AFTER the compile everyone was waiting on, and
        # without `force` the frame is thrown away whenever the phase changes within `dt` of the
        # last one, which is precisely the case of a phase entered immediately after a step.
        ProgressMeter.update!(p, p.counter; keep = false, force = true)
    elseif verb === :step
        p = _BAR[]
        # `keep = false` on EVERY update, not only the last. `finish!` is a no-op once the counter
        # has reached the total, so a bar that completed through `next!` never sees the close, and
        # its final frame is the one that would otherwise have printed the newline.
        p === nothing || ProgressMeter.next!(p; keep = false)
    elseif verb === :end
        _close_bar!()
    elseif verb === :done
        _close_bar!()
        # The newline nobody else printed. Without it the run's last bar is still on the cursor's
        # line and whatever prints next, a returned handle's summary or the prompt, lands on top
        # of it.
        _BAR_DIRTY[] || return nothing
        _BAR_DIRTY[] = false
        println(stderr)
    end
    return nothing
end

# ── The log reporter, for a notebook ─────────────────────────────────────────────────
#
# A notebook has no cursor for a bar but does have a logger. This emits ProgressLogging records,
# which Pluto, VS Code and TerminalLoggers render and other loggers drop. Steps are throttled to a
# frame per tenth of a second; the first and last frame of a stretch always go out.

mutable struct _ProgressLog
    const id::UUIDs.UUID
    const name::String
    phase::String
    const total::Int
    counter::Int
    last_emit::Float64
end

const _PLOG = Ref{Union{Nothing, _ProgressLog}}(nothing)
const _PLOG_INTERVAL = 0.1

# The record `@logprogress` emits, both halves: the `progress` keyword is the old API Pluto and
# VS Code read, the `ProgressString` message is the new one TerminalLoggers reads.
function _plog_emit!(st::_ProgressLog; done::Bool = false)
    fraction = st.total > 0 ? min(1.0, st.counter / st.total) : nothing
    name = isempty(st.phase) ? st.name : st.name * " [" * st.phase * "]"
    msg = ProgressLogging.ProgressString(
        ProgressLogging.Progress(st.id, fraction; name, done)
    )
    Logging.@logmsg ProgressLogging.ProgressLevel msg progress = (done ? "done" : fraction) _id =
        st.id
    st.last_emit = time()
    return nothing
end

function _plog_close!()
    st = _PLOG[]
    _PLOG[] = nothing
    st === nothing || _plog_emit!(st; done = true)
    return nothing
end

# `ProgressLevel` is below `Info`, so the stdlib `ConsoleLogger` drops the records and a CI log
# stays clean. Read at each `:begin` because `with_logger` changes the answer per call.
_logging_progress() =
    Logging.min_enabled_level(Logging.current_logger()) <= ProgressLogging.ProgressLevel

"""
    ReactantNitro.progress_log_reporter(verb, label, total, epoch, max_epochs) -> nothing

The progress reporter for an environment that renders log records rather than a terminal: Pluto,
VS Code, a REPL with TerminalLoggers, and any notebook whose logger accepts ProgressLogging's
`ProgressLevel`. Implements the contract [`progress_reporter!`](@ref) documents by emitting one
[`ProgressLogging.Progress`](https://github.com/JuliaLogging/ProgressLogging.jl) record per frame,
under one id per stretch of work: a fraction of zero at `:begin`, the fraction done on `:step` at
most every tenth of a second, the phase appended to the name on `:phase`, and `done = true` at
`:end`. A stretch with `total = 0` carries no fraction, which the renderers draw as busy.

Emits regardless of whether anything is listening; the logger drops what it does not accept. The
default reporter, [`default_progress_reporter`](@ref), picks this one when no terminal is watching
and the current logger accepts the level.
"""
function progress_log_reporter(
        verb::Symbol, label::String, total::Int, epoch::Int, max_epochs::Int
    )
    if verb === :begin
        _plog_close!()
        name = max_epochs > 0 ? "$label epoch $epoch/$max_epochs" : label
        st = _ProgressLog(UUIDs.uuid4(), name, "", total, 0, 0.0)
        _PLOG[] = st
        _plog_emit!(st)
    elseif verb === :phase
        st = _PLOG[]
        st === nothing && return nothing
        st.phase == label && return nothing
        st.phase = label
        _plog_emit!(st)
    elseif verb === :step
        st = _PLOG[]
        # A unit-less stretch has no fraction to advance, so a stray step draws nothing.
        (st === nothing || st.total <= 0) && return nothing
        st.counter += 1
        if st.counter >= st.total || time() - st.last_emit >= _PLOG_INTERVAL
            _plog_emit!(st)
        end
    elseif verb === :end || verb === :done
        _plog_close!()
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

# Returns whether this exit was the outermost one, so the caller knows whether to
# publish. Clamped at zero: an unbalanced decrement must not make the counter negative,
# because a negative counter reads as "work in flight" forever afterwards.
repl_exit!() = lock(MONITOR_LOCK) do
    REPL_DEPTH[] > 0 && (REPL_DEPTH[] -= 1)
    return REPL_DEPTH[] == 0
end

"""
    ReactantNitro.with_repl(f, nitro; spawn = true, on_interrupt = nothing) -> f()

Run `f`, then publish [`Repl`](@ref) through `nitro` if this was the outermost entry
point. The wrapper every public entry point goes through.

**Long loops leave the interactive thread.** [`train!`](@ref), [`validate`](@ref),
[`evaluate`](@ref), and [`predict`](@ref) block while XLA compiles and executes, and a single
compile or execute on the interactive thread starves every other task on it: logger tasks and a
Kaimon gate's message loop stop answering for the duration. So when the caller is on an
interactive thread and a default pool exists, `with_repl` runs `f` on a default-pool worker via
`Threads.@spawn` and PARKS the caller's task in `fetch`. Parking frees the interactive thread's
scheduler, which keeps running the logger and gate tasks while the loop executes on the worker;
single-threaded Julia (`-t 1`) and worker-thread callers (the Kaimon tool path, agent evals) run
inline, byte for byte as before. `spawn = false` forces inline, which the visualization entry
points use: rendering hooks may own a display backend that needs the main thread.

**Ctrl+C interrupts the parked wait, not the run.** SIGINT is delivered only into the
interactive thread's current task, so `on_interrupt` decides what a ^C means; the default
`nothing` rethrows, which is what an inline abort would have done. [`train!`](@ref) passes a
handler that requests the run's graceful stop ([`request_stop!`](@ref)) and waits for the epoch,
validation, and checkpoint to wind down; the eval entry points stop the split at its next batch
boundary and rethrow. A second ^C while winding down abandons the wait; the run still stops and
checkpoints on its own.

**A failed run surfaces its own exception.** `fetch` wraps a failed task's result in
`TaskFailedException`; `with_repl` strips it via [`unwrap_task_exc`](@ref) so callers, and the
test suite's `@test_throws`, see the exception the run actually raised, not a task wrapper.
The worker's internal frames fold into the call-site stack; the exception type and message are
preserved.

**The decrement is in a `finally` and the publish happens even when `f` throws.** A
`train!` that raised has still handed control back, and a counter left above zero would
mean the process never reports itself idle again, so one failed run would keep a
supervised session alive until something else killed it. The phase on the way out of a
failure is therefore `Repl` rather than [`Failed`](@ref); `Failed` is still published
first, and a monitor that needs to keep the reason should carry it forward rather than
expect it to be the final word.
"""
function with_repl(f, nitro; spawn::Bool = true, on_interrupt = nothing)
    # `Starting` on the way in, symmetric with `Repl` on the way out, and for the same reason: the
    # stretch before a verb's first compile is real work (hooks, the graph rebuild, an eval handle's
    # data) and until it is declared it is charged against whatever budget was already running.
    # Outermost only, so `train!` -> `validate` does not re-declare over the training phase.
    if repl_enter!()
        # ADOPT BEFORE PUBLISHING, not after. `_train!` takes the per-run copy of the
        # module-level registry from INSIDE this bracket, so until this line a monitor registered
        # between `Nitro(e)` and the verb missed the one event it most needed, and arrived out of
        # the documented order for it: the run's own monitors fired on `Starting` and the
        # module-level ones only from the next transition on. A supervisor registering in that
        # window is exactly the observer this declaration exists for, which makes the miss more
        # than cosmetic. Idempotent, so the loop's own `adopt_monitors!` still picks up anything
        # registered after this point and doubles nothing.
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

# Spawn only when it means something: the caller is on an interactive thread AND there is a
# thread outside the interactive pool to move to. Single-threaded Julia (`-t 1`, CI) and
# worker-thread callers (the Kaimon tool path, agent evals) run inline, byte for byte as before.
function _should_spawn()
    return Threads.nthreads() > Threads.nthreads(:interactive) &&
        Threads.threadid() <= Threads.nthreads(:interactive)
end

"""
    ReactantNitro._off_interactive(f) -> f()

Run `f` off the interactive thread under the same policy as [`with_repl`](@ref): when
[`_should_spawn`](@ref) holds, `f` runs on a default-pool worker and the caller parks in `fetch`;
otherwise `f` runs inline. A failing `f` surfaces its own exception, unwrapped from the
`TaskFailedException` that `fetch` wraps it in.

**This exists for `Nitro(e)`, which `with_repl` cannot wrap**: there is no handle to publish
through until the constructor returns, and the constructor publishes no `Repl` by design (it is
`with_repl_result`'s job when `train!(e)` calls it). What it does share with the entry points is the
problem `with_repl` solves. Construction blocks the calling thread for 45 to 95 seconds measured
(PJRT and cuDNN initialization in `setup_devices`, `build_model`, the `to_rarray` conversion), and
a headless script under `julia -t N,1` calls it from the main task, which lives on the interactive
thread. Every `Threads.@spawn :interactive` task in the process is then starved for the duration:
under an external supervisor, a heartbeat on a 10 s interval against a 30 s time-to-live stopped,
the supervisor reclaimed the session, and its watchdog killed the process before the first training
step. Measured identically on Reactant 0.2.264 and 0.2.284; moving the construction to a
default-pool task restored heartbeats every 5 s throughout. The Kaimon tool path never saw it
because tool handlers already run on worker threads, which is exactly what `_should_spawn`
detects and leaves inline.

No `on_interrupt` handler: a ^C during construction has nothing graceful to stop, so it
propagates as an inline abort would have.
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
        # SIGINT lands in the parked caller, never in the worker: deliver it to the entry
        # point's handler so `train!` can request the graceful stop instead of abandoning the
        # wait with the run continuing invisibly.
        e isa InterruptException || throw(unwrap_task_exc(e))
        on_interrupt === nothing && rethrow()
        return on_interrupt(t, nitro)
    end
end

"""
    ReactantNitro.with_repl_result(f) -> f()

The `Repl` wrapper for an entry point that has no `Nitro` until it returns one, which is
[`Nitro`](@ref)'s own constructor. Publishes through the RESULT.

Nothing is published when `f` throws, because there is no handle to publish through: a
half-constructed run has no monitor registry. The counter is still released, and
[`work_in_flight`](@ref) is how an interval monitor notices and corrects.
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
