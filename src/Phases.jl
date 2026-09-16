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
[`current_step`](@ref), [`current_epoch`](@ref), [`phase`](@ref), and [`binding_report`](@ref), and
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
    report::Any               # the schedule binding report, as text
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
    binding_report(nitro) -> String

The binding report: where every configured value actually bound. It is a **diagnostic, not a
check**; it computes nothing the run does not already compute and it never fails. The framework
prints it once at the end of setup and hands the same text to
`log_other!(lgr, "binding_report", str)`.

It exists because the rules that resolve a learning rate, a schedule key, and a per-group
accessor are individually simple and jointly hard to hold in your head, and because a wrong
binding is otherwise invisible until a curve looks strange three hours in.
"""
binding_report(nitro::Nitro) = nitro.report

"""
    logger_info(nitro) -> NamedTuple

The run's logger's key identifying parameters, read straight off a running experiment:
[`logger_info`](@ref)`(nitro.logger)`. The shipped default [`JSONLogger`](@ref) reports its
`path`; a hosted backend reports its URL, experiment key, workspace, and whatever else it
extends the verb with. The ReactantNitroKaimonGateExt `nitro_logger` tool renders exactly this table.
"""
logger_info(nitro::Nitro) = logger_info(nitro.logger)

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
    return fire_monitors(nitro, phase; info...)
end

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

# Bumped on the hot path, so it is an atomic add and nothing else. Not exported: the
# framework's own loops are the only callers.
note_progress!() = (Threads.atomic_add!(PROGRESS, 1); nothing)

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
