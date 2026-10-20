# Checkpoint.jl
#
# Checkpointing: the record, the top-K checkpointer, the manifest, and the resume compatibility
# checks. A "checkpoint" is full training state and "weights" are parameters alone, so optimizer
# state is saved on every checkpoint, including the top-K ones.

"""
    CheckpointRecord

Everything a checkpoint holds: the `snapshot` passed to [`save_checkpoint!`](@ref), plus
`format_version` and `framework_version`, which the framework stamps.

| Field | Required | Why it is in the record |
| --- | --- | --- |
| `format_version` | yes | JLD2 files outlive framework versions |
| `framework_version` | yes | Diagnosing a restore that misbehaves |
| `ps` | yes | The parameters, as a tree |
| `st` | yes | Layer state, including running statistics |
| `opt_state` | yes | Opaque; carries moments and bias correction. Stored as host values |
| `flat_permutation` | yes | A different permutation scrambles `opt_state` against `ps` |
| `step` | yes | Optimizer steps; restores schedules exactly |
| `epoch` | yes | Loop position and reporting |
| `seed` | yes | A rebuild must reproduce the same initialization |
| `config` | yes | Flattened, for the resume compatibility check |
| `devices` | yes | Derived and adaptive `Device` values, so a change is visible after the fact |
| `metrics` | yes | The epoch's validation metrics, for top-K bookkeeping |
| `run_id`, `run_url` | optional | Traces a checkpoint back to its experiment |
| `logger_state` | optional | Opaque, backend-specific resumption state |
| `logger_type` | optional | So resuming into a different backend refuses |
| `anchor_checksum` | optional | Per anchored group; the arrays themselves are not stored |
| `stop_reason` | optional | `:completed`, `:early_stop`, `:requested`, or `:error` |
| `preset` | optional | The preset the run was built from, `nothing` if none |
| `derived` | optional | The keys `derive` returned, so a weights-only load restores them |
| `early_stop` | optional | The early-stopping counters, so a resume continues its patience |
| `averaged_from` | optional | For an averaged record, what it averages: the spec, run, directory, files, epochs, weights and the state's file |
| `ema` | optional | The weights' exponential moving average ([`EMA`](@ref)), a tree like `ps`; `nothing` without one |
| `schedule_state` | optional | Each [`Plateau`](@ref) schedule's counters, by key, so a resume continues its sequence |

Every value is a host value; setup moves them to the device when it loads a record.
"""
struct CheckpointRecord
    format_version::Int
    framework_version::String
    ps::Any
    st::Any
    opt_state::Any
    flat_permutation::Any
    step::Int
    epoch::Int
    seed::Int
    config::Any
    devices::Any
    metrics::Any
    run_id::Any
    run_url::Any
    logger_state::Any
    logger_type::Any
    anchor_checksum::Any
    stop_reason::Any
    preset::Any
    # The keys `derive` returned, so a weights-only load restores them; `nothing` in older records.
    derived::Any
    # `EarlyStopping`'s counters, so a resume continues its patience; `nothing` in older records.
    early_stop::Any
    # What an averaged record averages (`average_checkpoints`); `nothing` for a trained one.
    averaged_from::Any
    # The weights' moving average, when the run kept one; `nothing` otherwise and in older records.
    ema::Any
    # Each `Plateau` schedule's counters by key; `nothing` without one and in older records.
    schedule_state::Any
end

# ── Showing a record never shows the weights ─────────────────────────────────────────
#
# Four fields are the whole parameter tree, so the default struct `show` prints tens of millions of
# characters, which in an agent session is the context window. The summary is the show, the arrays
# are named rather than printed, and `checkpoint_info` is the supported way to ask. `_shown` is also
# the experiment renderer (Config.jl), since a `Device{Tuple}` of buffers has the same shape.
_shown(x::Union{Nothing, Symbol, AbstractString, Real}) = repr(x)
# A device scalar reads back as a host number: four bytes, and the value is the point.
_shown(x::Reactant.RNumber) = repr(Reactant.to_number(x))
_shown(x::NamedTuple) = isempty(x) ? "(;)" :
    "(" * join(("$k = " * _shown(v) for (k, v) in pairs(x)), ", ") * ")"
# Tuples recurse so a `Device{Tuple{Matrix, Matrix}}` renders as two shapes.
_shown(x::Tuple) = isempty(x) ? "()" : "(" * join(map(_shown, x), ", ") * ")"
_shown(x::AbstractArray) = "<" * string(eltype(x)) * " array, size " * join(size(x), "x") * ">"
_shown(x) = "<" * string(nameof(typeof(x))) * ">"

# The four weight-bearing fields, named once so the two `show` methods cannot disagree about which
# ones are unsafe to print.
const _RECORD_WEIGHTS = (:ps, :st, :opt_state, :flat_permutation)

function Base.show(io::IO, r::CheckpointRecord)
    print(
        io, "CheckpointRecord(epoch ", r.epoch, ", step ", r.step,
        ", preset ", _shown(r.preset), ", ", length(_RECORD_WEIGHTS), " weight fields not shown)",
    )
    return nothing
end

function Base.show(io::IO, ::MIME"text/plain", r::CheckpointRecord)
    println(io, "CheckpointRecord (framework ", r.framework_version, ", format ", r.format_version, ")")
    for f in fieldnames(CheckpointRecord)
        f in (:format_version, :framework_version) && continue
        if f in _RECORD_WEIGHTS
            println(io, "  ", rpad(string(f), 17), "<not shown: this is the parameter tree>")
        else
            println(io, "  ", rpad(string(f), 17), _shown(getfield(r, f)))
        end
    end
    print(io, "  the weights reach a run through `Nitro(e; weights = path)`; for the metadata \
               alone use `checkpoint_info(path)`")
    return nothing
end

# ── Selectors: which checkpoints ────────────────────────────────────────────────────

"""
    Selector

Which entries of a run's manifest a rule picks. [`Checkpointer`](@ref)`(; keep)` retains the union
of its selectors; a `weights` source picks with one. [`Best`](@ref), [`Last`](@ref),
[`Epochs`](@ref) and [`Every`](@ref) are built in. A custom selector defines [`select`](@ref) and,
optionally, [`admits`](@ref).
"""
abstract type Selector end

"""
    Best(k = 1; among = nothing)

The `k` best entries by the checkpointer's `metric` and `mode`. Only entries scored under that pair
are ranked; under `scope = :dir`, every run's entries compete. `among`, a selector, ranks only what
it picks: `Best(5; among = Last(20))` is the best 5 of the latest 20. `among` is for loading, as in
[`Average`](@ref); `keep` refuses it.
"""
struct Best{A} <: Selector
    k::Int
    among::A
end
Best(k::Integer = 1; among = nothing) = Best(Int(k), check_among(among))

check_among(a) = a === nothing || a isa Selector ? a : error(
        "ReactantNitro: `Best(k; among)` takes a selector for `among`, such as `Last(20)`; got a \
     `$(typeof(a))`."
    )

"""
    Last(n = 1)

This run's latest `n` entries: by epoch under `scope = :run`, by write time under `scope = :dir`.
"""
struct Last <: Selector
    n::Int
end
Last() = Last(1)

"""
    Epochs(epochs)

This run's entries whose epoch is in `epochs`, a range or a collection.
"""
struct Epochs{R} <: Selector
    epochs::R
end

"""
    Every(n)

This run's entries whose epoch is a multiple of `n`.
"""
struct Every <: Selector
    n::Int
end

Base.show(io::IO, s::Best) =
    s.among === nothing ? print(io, "Best(", s.k, ")") :
    print(io, "Best(", s.k, "; among = ", s.among, ")")
Base.show(io::IO, s::Last) = print(io, "Last(", s.n, ")")
Base.show(io::IO, s::Epochs) = print(io, "Epochs(", s.epochs, ")")
# By value: `epochs` may be a vector, which the default `==` compares by identity.
Base.:(==)(a::Epochs, b::Epochs) = a.epochs == b.epochs
Base.:(==)(a::Best, b::Best) = a.k == b.k && a.among == b.among
Base.show(io::IO, s::Every) = print(io, "Every(", s.n, ")")

"""
    select(sel::Selector, entries, ckpt) -> Vector{ManifestEntry}

The entries of `entries` that `sel` picks. `entries` is what retention decides over: this run's
under `scope = :run`, the directory's under `scope = :dir`.
"""
function select end

_own(entries, ckpt) = [e for e in entries if isequal(e.run, ckpt.run)]
_latest_order(ckpt) = ckpt.scope === :dir ? (e -> e.written) : (e -> e.epoch)

function select(sel::Best, entries, ckpt)
    pool = sel.among === nothing ? entries : select(sel.among, entries, ckpt)
    scored = [e for e in pool if e.score !== nothing && same_selection(e, ckpt)]
    return first(sort(scored; by = e -> e.score, rev = ckpt.mode === :max), max(sel.k, 0))
end
select(sel::Last, entries, ckpt) =
    first(sort(_own(entries, ckpt); by = _latest_order(ckpt), rev = true), max(sel.n, 0))
select(sel::Epochs, entries, ckpt) = [e for e in _own(entries, ckpt) if e.epoch in sel.epochs]
select(sel::Every, entries, ckpt) = [e for e in _own(entries, ckpt) if e.epoch % sel.n == 0]

"""
    admits(sel::Selector, pool, ckpt, epoch, score) -> Bool

Whether `sel` would retain a write of `epoch` scoring `score`, decided before the record is built.
An epoch no selector admits is not written. The default is `true`, which always writes and leaves
the decision to [`select`](@ref).
"""
admits(::Selector, pool, ckpt, epoch, score) = true
admits(sel::Best, pool, ckpt, epoch, score) =
    score !== nothing && enters_top_k(pool, ckpt, epoch, score, sel.k)
admits(sel::Last, pool, ckpt, epoch, score) = sel.n >= 1
admits(sel::Epochs, pool, ckpt, epoch, score) = epoch in sel.epochs
admits(sel::Every, pool, ckpt, epoch, score) = epoch % sel.n == 0

# ── The checkpointer ────────────────────────────────────────────────────────────────

"""
    Checkpointer(; keep = (Best(3), Last(1)), metric = nothing, mode = nothing, dir = nothing,
                   name = nothing, scope = :run)

The default checkpointer. After each epoch it writes a full [`CheckpointRecord`](@ref) into
[`run_dir`](@ref) if some selector in `keep` would retain it, then keeps the union of what the
selectors pick and deletes the rest. The default, `(Best(3), Last(1))`, keeps the best 3 by
`metric` (lower is better under `mode = :min`, higher under `:max`) plus the latest, whatever it
scored. `checkpointer = nothing` disables checkpointing.

```julia
Checkpointer(; keep = (Best(3), Last(10)))   # the best 3 and the latest 10, for averaging
Checkpointer(; keep = Last(5))               # a rolling window
Checkpointer(; keep = Best(3))               # the best 3 only
```

`metric` and `mode` default to `nothing`, which follows the run's [`monitor`](@ref); give both to
rank by something else. Setup fills a following checkpointer in, every time it builds a `Nitro`. A
checkpointer used without setup (in [`average_checkpoints`](@ref), say) ranks by the metric and
mode the run's newest scored entry was written under. `dir` and `name` left as `nothing` are filled in
at setup: `dir` with `run_dir`, and `name`, the filename function
`name(; epoch, step, metric, score)`, with the experiment's [`checkpoint_filename`](@ref).

A manifest, `manifest.jld2`, lists each retained file with its epoch, score, `stop_reason`, and
the run, metric and mode that wrote it, so lookups and resume never open a record; see
[`read_manifest`](@ref). The latest checkpoint is found through it rather than a `latest`
symlink, which would dangle once rotation deleted its target.

**Without a [`Last`](@ref)**, an epoch no selector admits is not written at all, rather than
written and deleted. The final `stop_reason` rewrite follows the same rule, so the state training
ended in is saved only if a selector keeps the last epoch, and `weights = :latest => :latest` and
`resume = :auto` resolve to the latest kept epoch, not to where training stopped. A crash loses
every epoch since then. Setup warns about this for a handle that trains, and refuses `keep` with
only [`Best`](@ref) selectors for a training run with no `val` split, whose epochs have no score.

**`scope = :run`**, the default, applies retention to this run's checkpoints only. A directory
often holds several runs (the default `run_dir` is the experiment's type name), so setup gives
each run an id, the field `run`, and never ranks, rotates, deletes or overwrites another run's
checkpoints. A fresh run gets a new id. A resume from a run's latest record continues that run's
id; a resume from any other record starts a new id and records its source path in `parent`. Setup
sets both fields for every `Nitro` it builds, so a checkpointer reused across handles belongs to
the most recent one. A checkpointer used without setup has `run = nothing`, the same unnamed run
as manifest entries written before runs were recorded.

**`scope = :dir`** saves space: [`Best`](@ref) ranks the whole directory, across every run, and a
write deletes every other entry scored the same way, including other runs'; [`Last`](@ref) and the
other selectors keep this run's own entries. Setup warns a training handle when the directory
already holds checkpoints this may delete.

**Only entries scored under the current `metric` and `mode` are ranked.** Entries written under
another pair, by an earlier run or before a resume changed the selection, are never ranked or
deleted, under either scope. Setup and `:best` lookups warn when the entries they consider are
mixed this way. An entry with no recorded metric predates the field and counts as matching. Under
either scope, a filename that would overwrite a file this run does not own gets `-run-<id>`
inserted before `.jld2`.

`ckpt.k` (the largest [`Best`](@ref)'s `k`, else 0) and `ckpt.keep_latest` (whether `keep` has a
[`Last`](@ref)) remain readable for code written against [`TopKCheckpointer`](@ref), and are
deprecated.
"""
mutable struct Checkpointer
    keep::Tuple
    # The checkpointer's own selection; `nothing` follows `monitor` (read through `getproperty`).
    own::Union{Pair{Symbol, Symbol}, Nothing}
    # The run's `monitor`, assigned at setup.
    monitor::Union{Pair{Symbol, Symbol}, Nothing}
    dir::Union{String, Nothing}
    name::Any
    manifest::Any
    # This run's id, and the checkpoint a branched run was resumed from. Assigned at setup.
    run::Union{String, Nothing}
    parent::Union{String, Nothing}
    # Which entries retention ranks and prunes: `:run` (this run's) or `:dir` (the directory's).
    scope::Symbol
end

function Checkpointer(;
        keep = (Best(3), Last(1)), metric::Union{Symbol, Nothing} = nothing,
        mode::Union{Symbol, Nothing} = nothing, dir::Union{String, Nothing} = nothing,
        name = nothing, scope::Symbol = :run
    )
    check_scope(scope)
    keep = check_keep(keep isa Selector ? (keep,) : Tuple(keep))
    own = _own_monitor(metric, mode, "the checkpointer")
    return Checkpointer(keep, own, nothing, dir, name, nothing, nothing, nothing, scope)
end

# The selection in force: the checkpointer's own, else the run's `monitor`; `nothing` before setup.
_selection(c::Checkpointer) = something(getfield(c, :own), getfield(c, :monitor), Some(nothing))

# A copy that ranks by `sel`, for a checkpointer used without setup.
_with_monitor(c::Checkpointer, sel) = Checkpointer(
    c.keep, getfield(c, :own), sel, c.dir, c.name, c.manifest, c.run, c.parent, c.scope
)

"""
    ReactantNitro.bind_selection(ckpt, entries) -> ckpt

`ckpt` itself when it has a selection, else a copy ranking by the metric and mode the newest
scored entry of `entries` was written under. `ckpt` unchanged when no entry is scored.
"""
bind_selection(ckpt, entries) = ckpt
function bind_selection(ckpt::Checkpointer, entries)
    _selection(ckpt) === nothing || return ckpt
    scored = [e for e in entries if e.score !== nothing && e.metric !== nothing]
    isempty(scored) && return ckpt
    e = scored[argmax([e.written for e in scored])]
    return _with_monitor(ckpt, e.metric => e.mode)
end

function check_keep(keep::Tuple)
    isempty(keep) && error(
        "ReactantNitro: the checkpointer's `keep` is empty, so it would keep nothing. Use \
         `checkpointer = nothing` to disable checkpointing."
    )
    for sel in keep
        sel isa Selector || error(
            "ReactantNitro: the checkpointer's `keep` holds a `$(typeof(sel))`; every entry must \
             be a selector such as `Best(3)` or `Last(10)`."
        )
        sel isa Best && sel.among !== nothing && error(
            "ReactantNitro: `$(sel)` in the checkpointer's `keep`: `among` is for loading, as in \
             `Average`. Keep the selector it narrows (`$(sel.among)`) instead."
        )
        n = sel isa Best ? sel.k : sel isa Union{Last, Every} ? sel.n : 1
        n >= 1 || error("ReactantNitro: `$(sel)` in the checkpointer's `keep` must count at least 1.")
    end
    return keep
end

# The compatibility properties, derived from `keep`.
function Base.getproperty(c::Checkpointer, s::Symbol)
    s === :k && return maximum((sel.k for sel in getfield(c, :keep) if sel isa Best); init = 0)
    s === :keep_latest && return any(sel -> sel isa Last, getfield(c, :keep))
    s === :metric && return (sel = _selection(c); sel === nothing ? nothing : first(sel))
    s === :mode && return (sel = _selection(c); sel === nothing ? nothing : last(sel))
    return getfield(c, s)
end
Base.propertynames(::Checkpointer) =
    (fieldnames(Checkpointer)..., :metric, :mode, :k, :keep_latest)

"""
    TopKCheckpointer(; k = 3, metric = nothing, mode = nothing, dir = nothing, name = nothing,
                       keep_latest = true, scope = :run) -> Checkpointer

Deprecated: use [`Checkpointer`](@ref)`(; keep = (Best(k), Last(1)))`, or `keep = Best(k)` for
`keep_latest = false`. Builds exactly that `Checkpointer`, so the files and the manifest are the
same as before. Warns once per session; supported until at least 1.0.0. `metric` and `mode` follow
[`monitor`](@ref) as `Checkpointer`'s do.
"""
function TopKCheckpointer(;
        k::Int = 3, metric::Union{Symbol, Nothing} = nothing,
        mode::Union{Symbol, Nothing} = nothing,
        dir::Union{String, Nothing} = nothing, name = nothing, keep_latest::Bool = true,
        scope::Symbol = :run
    )
    @warn "ReactantNitro: `TopKCheckpointer` is deprecated and supported until at least 1.0.0. \
           Use `Checkpointer(; keep = $(keep_latest ? "(Best($k), Last(1))" : "Best($k)"))`, \
           which writes the same files." maxlog = 1
    k >= 1 || error("ReactantNitro: the checkpointer's `k` is $(k); it must be at least 1. Pass \
        `checkpointer = nothing` to disable checkpointing, which is the documented opt-out.")
    keep = keep_latest ? (Best(k), Last(1)) : (Best(k),)
    return Checkpointer(; keep, metric, mode, dir, name, scope)
end

check_scope(scope) = scope in (:run, :dir) || error(
    "ReactantNitro: the checkpointer's `scope` is $(repr(scope)); it must be `:run` (retention \
     within this run's own checkpoints, the default) or `:dir` (the best across every run in the \
     directory)."
)

# A run id: eight characters from the OS entropy source. Not the global rng, which setup has
# seeded, so two runs with one seed would draw the same id and be one run to the manifest.
new_run_id() = Random.randstring(Random.RandomDevice(), 8)

"""
    save_checkpoint!(ckpt, epoch, metrics, snapshot) -> Bool or nothing

Write a checkpoint of `epoch`, if the checkpointer decides the epoch warrants one. `metrics` is the
epoch's validation metrics, empty for a run with no `val` split, and `snapshot` holds the record's
other fields as host values. Returns `false` when the checkpointer declined; any other value,
`nothing` included, means it wrote. `train!` uses this to decide whether the final `stop_reason`
rewrite has a record to rewrite.

[`Checkpointer`](@ref) returns `true` or `false`, declining an epoch no selector in `keep`
[`admits`](@ref) (with a [`Last`](@ref), every epoch is admitted). It writes through [`with_io_retry`](@ref) to a temporary
path renamed into place, then updates the manifest, and only then deletes the files rotation
displaced. The experiment is not an argument: the filename comes from the checkpointer's `name`,
so the object holding device values never reaches the code that serializes.
"""
function save_checkpoint! end
save_checkpoint!(::Nothing, epoch, metrics, snapshot) = nothing

function save_checkpoint!(ckpt::Checkpointer, epoch, metrics, snapshot)
    # Without a selection only unranked selectors work: setup refuses a `Best` with no monitor.
    ckpt.metric === nothing && any(s -> s isa Best, ckpt.keep) && error(
        "ReactantNitro: this `Checkpointer` ranks with `Best` and has no metric: it follows the \
         run's `monitor`, which `Nitro` setup fills in. Give it `metric` and `mode` to save with \
         it by hand."
    )
    dir = checkpoint_dir(ckpt)
    mkpath(dir)
    # The selection metric drives retention, which is control flow, so the readback is validated.
    # A metric that is only logged is not, and that asymmetry is deliberate.
    score = ckpt.metric !== nothing && haskey(metrics, ckpt.metric) ?
        check_control_readback(
            getproperty(metrics, ckpt.metric), "the checkpointer",
            ckpt.metric
        ) : nothing
    (score === nothing && ckpt.metric !== nothing && !isempty(metrics)) && error(
        """
        ReactantNitro: the checkpointer selects on `$(ckpt.metric)`, which `metrics` does not emit.
        The available metrics are $(keys(metrics)).
        `:loss` is what the framework supplies when an experiment defines no `metrics`; an
        experiment defining its own names one of its own keys.$(split_prefix_hint(ckpt.metric))"""
    )
    (score === nothing && all(sel -> sel isa Best, ckpt.keep)) && error(
        """
        ReactantNitro: this checkpointer keeps $(join(map(repr, ckpt.keep), ", ")) only (what
        `keep_latest = false` builds), and epoch $(epoch) has no `$(ckpt.metric)` score (the run
        has no `val` split), so it could never be kept. `Best` retains by score. Add `Last(1)` to `keep` (the default, `(Best(3), Last(1))`,
        keeps the latest record whatever it scored), or add a `val` split."""
    )

    # This run's entries and everyone else's. The one-entry-per-epoch rule and the displaced file
    # are applied to this run's alone. Under `scope = :run` so is retention, and another run's
    # entries go back into the manifest exactly as read, their files never touched; under
    # `scope = :dir` retention ranks the whole directory and may prune another run's entries.
    prior = read_manifest(dir)
    own = ManifestEntry[e for e in prior if isequal(e.run, ckpt.run)]
    others = ManifestEntry[e for e in prior if !isequal(e.run, ckpt.run)]
    pool_others = ckpt.scope === :dir ? others : ManifestEntry[]
    # An epoch no selector admits would be written only for the rotation below to delete it.
    # Decided before the record is built or the name hook is called.
    pool0 = vcat(pool_others, own)
    any(sel -> admits(sel, pool0, ckpt, Int(epoch), score), ckpt.keep) || return false

    record = CheckpointRecord(
        FORMAT_VERSION, framework_version(),
        snapshot.ps, snapshot.st, snapshot.opt_state,
        snapshot.flat_permutation, snapshot.step, snapshot.epoch,
        snapshot.seed, snapshot.config, snapshot.devices, metrics,
        snapshot.run_id, snapshot.run_url,
        snapshot.logger_state, snapshot.logger_type,
        snapshot.anchor_checksum, snapshot.stop_reason,
        # `nothing` for a run that named no preset, which is most of them.
        hasproperty(snapshot, :preset) ? snapshot.preset : nothing,
        hasproperty(snapshot, :derived) ? snapshot.derived : nothing,
        hasproperty(snapshot, :early_stop) ? snapshot.early_stop : nothing,
        nothing,
        hasproperty(snapshot, :ema) ? snapshot.ema : nothing,
        hasproperty(snapshot, :schedule_state) ? snapshot.schedule_state : nothing
    )

    name = owned_name(
        checkpoint_name(ckpt, epoch, snapshot.step, score), dir, ckpt.run,
        Set(e.file for e in own), Set(e.file for e in others)
    )
    path = joinpath(dir, name)
    write_record(path, record)

    entry = ManifestEntry(
        (
            name, Int(epoch), score, snapshot.stop_reason, ckpt.run, ckpt.metric, ckpt.mode,
            time(), ckpt.parent,
        )
    )
    # One entry per epoch and one per file: `file` is a checkpoint's only identity, and a `name`
    # that omits the epoch ("best only") would otherwise grow an entry per epoch naming one file.
    entries = ManifestEntry[
        e for e in own
            if e.epoch != Int(epoch) && e.file != entry.file
    ]
    push!(entries, entry)
    # A write whose name differs from the previous write of that epoch would leave a file no entry
    # names, invisible to every rotation; `name` is a hook and Revise can change it mid-run.
    displaced = String[
        e.file for e in own
            if e.epoch == Int(epoch) && e.file != entry.file
    ]
    # The entries retention decides over, and the ones it leaves exactly as they were.
    pool = vcat(pool_others, entries)
    untouched = ckpt.scope === :dir ? ManifestEntry[] : others
    keep = retained(pool, ckpt)
    # The manifest is written before the rotation deletes anything, so a failure leaves at worst
    # an unlisted file rather than fewer checkpoints than the policy promises.
    write_manifest(dir, vcat(untouched, [e for e in pool if e.file in keep]))
    # A file an entry still listed names is never deleted, whatever the rotation says.
    listed = union(Set(e.file for e in untouched), keep)
    for f in Iterators.flatten((displaced, (e.file for e in pool if !(e.file in keep))))
        f in listed && continue
        p = joinpath(dir, f)
        isfile(p) && with_io_retry(() -> rm(p))
    end
    return true
end

"""
    ReactantNitro.enters_top_k(pool, ckpt, epoch, score, k = ckpt.k) -> Bool

Whether a write of `epoch` scoring `score` would be among the top `k`: what [`admits`](@ref) asks
of a [`Best`](@ref). `pool` is the entries retention decides over (the run's own under
`scope = :run`, the directory's under `scope = :dir`). Rivals are the scored entries under this
checkpointer's `metric` and `mode`, except this run's entry for `epoch`, which the write replaces;
so the final `stop_reason` rewrite of a top-K epoch passes, and that of an unwritten epoch does
not. With fewer than K rivals the answer is yes. Otherwise the score must strictly beat the K-th
best, since a tie sorts after the entries already held and would be rotated out.
"""
function enters_top_k(pool, ckpt::Checkpointer, epoch::Int, score, k::Int = ckpt.k)
    rivals = [
        e.score for e in pool
            if !(isequal(e.run, ckpt.run) && e.epoch == epoch) && e.score !== nothing &&
            same_selection(e, ckpt)
    ]
    length(rivals) < k && return true
    kth = sort(rivals; rev = ckpt.mode === :max)[k]
    return ckpt.mode === :max ? score > kth : score < kth
end

"""
    ReactantNitro.owned_name(name, dir, run, own, theirs) -> String

The filename this run writes under: `name` when that file is absent or already this run's, and
otherwise `name` with `-run-<id>` inserted before `.jld2`, plus a counter if that is taken too.
Two runs with the same seed and config produce the same default names, so without this one would
overwrite the other's records. A file the manifest does not list is not this run's either.
"""
function owned_name(name, dir, run, own, theirs)
    free(n) = n in own || (!(n in theirs) && !isfile(joinpath(dir, n)))
    free(name) && return name
    stem = name[1:(end - length(".jld2"))] * "-run-" * something(run, "unnamed")
    candidate = stem * ".jld2"
    i = 2
    while !free(candidate)
        candidate = stem * "-" * string(i) * ".jld2"
        i += 1
    end
    return candidate
end

"""
    ReactantNitro.retained(pool, ckpt) -> Set{String}

The files retention keeps out of `pool` (one run's entries under `scope = :run`, the directory's
under `scope = :dir`): the union of what each selector in `keep` [`select`](@ref)s. An unscored
entry (a run with no `val` split) is kept only by a selector that does not rank, such as
[`Last`](@ref), which keeps a train-only run resumable. Only entries scored under the
checkpointer's `metric` and `mode` are ranked; every entry written under another pair is kept,
under either scope. An entry with no recorded metric predates the field and is ranked.
"""
function retained(pool, ckpt::Checkpointer)
    picked = String[e.file for sel in ckpt.keep for e in select(sel, pool, ckpt)]
    other = String[e.file for e in pool if !same_selection(e, ckpt)]
    return Set(vcat(picked, other))
end

# Whether an entry's score was produced by this checkpointer's selection. An entry with no recorded
# metric was written before the manifest carried one and is taken to match.
same_selection(e, ckpt) =
    e.metric === nothing || (e.metric === ckpt.metric && e.mode === ckpt.mode)

"""
    ReactantNitro.write_record(path, record) -> nothing

Write `record` to `path` through [`with_io_retry`](@ref): to a temporary file, then renamed into
place. A reader, including `resume = :auto` in another process, sees the old file or the whole new
one, never a partial record.
"""
function write_record(path::AbstractString, record::CheckpointRecord)
    tmp = path * ".tmp"
    with_io_retry() do
        JLD2.jldsave(tmp; record)
        mv(tmp, path; force = true)
    end
    return nothing
end

# The manifest: file, epoch, score and `stop_reason`, so top-K bookkeeping and `resume = :auto`
# work without reading every record, and the run, metric and mode that wrote each entry, so one
# directory can hold several runs without one rotating another's files away. `written` is the
# wall-clock time of the write and orders runs, nothing else; `parent` is the checkpoint a branched
# run was resumed from. The entry type is explicit, with `Union` fields: a manifest built from bare
# NamedTuples typed itself off its first row, and the final rewrite carrying a `stop_reason` could
# not be pushed into it.
const ManifestEntry = @NamedTuple{
    file::String, epoch::Int, score::Union{Float64, Nothing},
    stop_reason::Union{Symbol, Nothing},
    run::Union{String, Nothing}, metric::Union{Symbol, Nothing}, mode::Union{Symbol, Nothing},
    written::Float64, parent::Union{String, Nothing},
}

manifest_path(dir) = joinpath(dir, "manifest.jld2")

"""
    read_manifest(dir) -> Vector

Which checkpoints a run directory holds, without opening one. Each entry carries `file` (a
basename), `epoch`, `score` (the retention metric, or `nothing` for a run with no `val` split),
`stop_reason` (`nothing` until the run records how it ended), `run` (the id of the run that wrote
it), `metric` and `mode` (what `score` measures, and which way is better), `written` (the
wall-clock `time()` of the write) and `parent` (for a run that branched from an earlier record, that
record's path). A manifest written before runs were recorded reads with `run`, `metric`, `mode`
and `parent` as `nothing` and `written` as `0.0`: one unnamed run, older than every named one.
Returns an empty vector for a directory no run has written to. Use [`checkpoint_info`](@ref) only
for what the manifest does not carry.

```julia
for e in sort(read_manifest("runs/MyExp"); by = e -> (e.written, e.epoch))
    println(e.run, "  ", e.epoch, "  ", e.score, "  ", e.file)
end
```
"""
function read_manifest(dir)
    p = manifest_path(dir)
    isfile(p) || return ManifestEntry[]
    raw = with_io_retry() do
        JLD2.load(p, "entries")
    end
    field(e, f, default) = hasproperty(e, f) ? getproperty(e, f) : default
    return ManifestEntry[
        ManifestEntry(
            (
                String(e.file), Int(e.epoch), e.score, e.stop_reason,
                field(e, :run, nothing), _migrate_metric(field(e, :metric, nothing)),
                field(e, :mode, nothing),
                Float64(field(e, :written, 0.0)), field(e, :parent, nothing),
            )
        )
            for e in raw
    ]
end

# Entries scored before 0.2.0 name the validation loss `:val_loss`; it is `:loss` now, so they keep
# ranking with the entries written since. Silent, since nothing is wrong with the run. Until 1.0.0.
_migrate_metric(m) = m === :val_loss ? :loss : m

# The runs a manifest holds, most recent first: ordered by each run's latest `written`, so the
# unnamed legacy run, written at `0.0`, is the oldest.
function manifest_runs(entries)
    latest = Dict{Union{String, Nothing}, Float64}()
    for e in entries
        latest[e.run] = max(get(latest, e.run, -Inf), e.written)
    end
    return sort!(collect(keys(latest)); by = r -> latest[r], rev = true)
end

# How a run is named in a message.
run_label(run) = run === nothing ? "the unnamed run (written before runs were recorded)" :
    "run `$(run)`"

function write_manifest(dir, entries)
    p = manifest_path(dir)
    tmp = p * ".tmp"
    with_io_retry() do
        JLD2.jldsave(tmp; entries)
        mv(tmp, p; force = true)
    end
    return nothing
end

"""
    checkpoint_info(path) -> NamedTuple

Read a checkpoint's metadata without its weights. The fields are `format_version`,
`framework_version`, `epoch`, `step`, `seed`, `preset`, the logger's `run_id`, `run_url` and
`logger_type`, `stop_reason`, `anchor_checksum`, and `metrics`, the validation metrics of the epoch
it was written at.

```julia
i = checkpoint_info("runs/MyExp/epoch-0012-step-116520-mae=0.0253034.jld2")
i.epoch, i.step, i.preset, i.metrics.mae, i.run_id
```

Use this instead of opening the JLD2 file by hand: a record is one object whose four
weight-bearing fields load with it, and printing it carelessly prints the whole parameter tree.
Opened in a process without ReactantNitro loaded, JLD2 reconstructs a look-alike type. The
checkpoint's own run id (see [`checkpoint_run`](@ref)) is in the manifest, not here. For a
whole directory, use [`read_manifest`](@ref) instead of calling this in a loop.
"""
function checkpoint_info(path::AbstractString)
    isfile(path) || error("ReactantNitro: there is no checkpoint at `$path`.")
    r = _migrate_record(
        with_io_retry() do
            _load_record_quietly(path)
        end, path
    )
    return (;
        format_version = r.format_version, framework_version = r.framework_version,
        epoch = r.epoch, step = r.step, seed = r.seed, preset = r.preset,
        run_id = r.run_id, run_url = r.run_url, logger_type = r.logger_type,
        stop_reason = r.stop_reason, anchor_checksum = r.anchor_checksum,
        metrics = r.metrics, averaged_from = r.averaged_from, has_ema = r.ema !== nothing,
    )
end

# ── The filename ────────────────────────────────────────────────────────────────────

"""
    checkpoint_filename(e; epoch, step, metric, score) -> String

The filename of a checkpoint; a hook a model package may extend.

    epoch-0006-step-4500-loss=0.0416831.jld2
    epoch-0006-step-4500.jld2                          # a run with no `val` split

The epoch comes first, zero-padded, so lexicographic order is training order; the step is
unpadded, and `score === nothing` omits the score segment. The score is printed with `%.6g` and
needs no sanitizing, since [`check_control_readback`](@ref) made it a finite `Float64`; the metric
name goes through `ReactantNitro.sanitize_metric_name`. A method must accept `kwargs...` so a
keyword added later is not a `MethodError`:

```julia
ReactantNitro.checkpoint_filename(e::MyExp; epoch, score, kwargs...) =
    "e\$(lpad(epoch, 4, '0'))_\$(round(something(score, 0.0); digits = 4)).jld2"
```

[`Checkpointer`](@ref)'s `name`, bound at setup, calls this on every write, so a redefined
method takes effect at the next checkpoint of a live run.
"""
function checkpoint_filename end

function checkpoint_filename(e; epoch, step, metric, score)
    return string(
        "epoch-", lpad(epoch, 4, '0'), "-step-", step,
        score === nothing ? "" :
            string("-", sanitize_metric_name(metric), "=", Printf.@sprintf("%.6g", score)),
        ".jld2"
    )
end

"""
    ReactantNitro.sanitize_metric_name(metric) -> String

A metric name made safe for a filename: every character outside `[A-Za-z0-9_.]` becomes `_`, the
result is truncated to 40 characters, and an empty result becomes `metric`.
"""
function sanitize_metric_name(metric)
    s = replace(String(metric), r"[^A-Za-z0-9_.]" => "_")
    length(s) > 40 && (s = s[1:nextind(s, 0, 40)])
    return isempty(s) ? "metric" : s
end

"""
    ReactantNitro.checkpoint_name(ckpt, epoch, step, score) -> String

Call [`Checkpointer`](@ref)'s `name` for one write and check the result: a non-empty `.jld2`
basename with no path separator and no `..`. A name that omits the epoch ("best only") is allowed;
it repeats across epochs, and each write deletes the file it displaced.
"""
function checkpoint_name(ckpt::Checkpointer, epoch, step, score)
    ckpt.name === nothing && error("ReactantNitro: this `Checkpointer` has no `name`. Setup \
        binds the experiment's `checkpoint_filename` to it; construct it \
        with `name = ...` to name checkpoints yourself, or reach it through a `Nitro` rather than \
        calling `save_checkpoint!` by hand.")
    s = ckpt.name(; epoch = Int(epoch), step = Int(step), metric = ckpt.metric, score)
    (s isa AbstractString && !isempty(s) && endswith(s, ".jld2")) || error(
        """
        ReactantNitro: the checkpointer's `name` returned $(repr(s)) for epoch $(epoch), which is
        not a checkpoint filename.
        It must return a non-empty `String` ending in `.jld2`; the framework writes it into the
        run directory, so it is a basename and not a path."""
    )
    (occursin('/', s) || occursin('\\', s) || occursin("..", s)) && error(
        """
        ReactantNitro: the checkpointer's `name` returned $(repr(s)) for epoch $(epoch), which
        contains a path separator or `..`.
        A checkpoint name is a BASENAME inside `run_dir`, which is the one path concept in the
        framework. Set the checkpointer's `dir` to write somewhere else."""
    )
    return String(s)
end

"""
    ReactantNitro.checkpoint_dir(ckpt) -> String

Where a [`Checkpointer`](@ref) writes: its `dir`, which setup sets to [`run_dir`](@ref) unless
the checkpointer was constructed with one. Errors when `dir` is still `nothing`.
"""
function checkpoint_dir(ckpt::Checkpointer)
    ckpt.dir === nothing && error("ReactantNitro: this `Checkpointer` has no directory. Setup \
        assigns `run_dir` to it; construct it with `dir = ...` to write \
        somewhere else, or reach it through a `Nitro` rather than calling `save_checkpoint!` by \
        hand.")
    return ckpt.dir
end

# The two framework-stamped fields. The version is read from the package's own Project.toml so
# it cannot drift from the release it shipped in. Located from this file, not `pkgdir`, which is
# `nothing` when the module was evaluated from source rather than loaded as a package.
const FORMAT_VERSION = 1

function framework_version()
    p = joinpath(dirname(@__DIR__), "Project.toml")
    m = isfile(p) ? match(r"(?m)^version\s*=\s*\"([^\"]+)\"", read(p, String)) : nothing
    return m === nothing ? "unknown" : m.captures[1]
end

"""
    load_checkpoint(ckpt, path) -> CheckpointRecord

Load the record at `path` through the checkpointer that wrote it. It dispatches on the
checkpointer rather than the path so that a checkpointer with a non-file backend can implement it.
The [`Checkpointer`](@ref) method migrates records written before a field was added and refuses
a record of another `format_version`. With `checkpointer = nothing` it returns `nothing`.
"""
function load_checkpoint end
load_checkpoint(::Nothing, path) = nothing

# JLD2 warns "saved type CheckpointRecord is missing field ... reconstructing" on every record that
# predates a field addition, which is every mid-training checkpoint of a long run. The migration
# below is explicit and tested, so the warning is suppressed narrowly, matching both the phrase and
# the type name.
struct _QuietReconstruct{L <: Base.CoreLogging.AbstractLogger} <: Base.CoreLogging.AbstractLogger
    parent::L
end
Base.CoreLogging.min_enabled_level(l::_QuietReconstruct) =
    Base.CoreLogging.min_enabled_level(l.parent)
Base.CoreLogging.shouldlog(l::_QuietReconstruct, args...) =
    Base.CoreLogging.shouldlog(l.parent, args...)
Base.CoreLogging.catch_exceptions(l::_QuietReconstruct) =
    Base.CoreLogging.catch_exceptions(l.parent)
function Base.CoreLogging.handle_message(
        l::_QuietReconstruct, level, message, _mod, group, id, file, line; kwargs...
    )
    txt = string(message)
    if level == Base.CoreLogging.Warn &&
            occursin("missing field", txt) && occursin("CheckpointRecord", txt)
        return nothing
    end
    return Base.CoreLogging.handle_message(
        l.parent, level, message, _mod, group, id, file, line; kwargs...
    )
end

_load_record_quietly(path) = Base.CoreLogging.with_logger(
    () -> JLD2.load(path, "record"), _QuietReconstruct(Base.CoreLogging.current_logger())
)

# Field-shape migrations: the `tunables` to `devices` rename, and the additions of `preset`,
# `derived`, `early_stop`, `averaged_from`, `ema` and `schedule_state`.
# JLD2 reconstructs an old record as a ReconstructedMutable with the on-disk field names, and the
# rename is done here. A function because `checkpoint_info` reads the same records. FORMAT_VERSION
# stays 1, since the semantics did not change.
function _migrate_record(raw, path = "<record>")
    raw isa CheckpointRecord && return raw
    if hasproperty(raw, :tunables) || !hasproperty(raw, :preset) || !hasproperty(raw, :derived) ||
            !hasproperty(raw, :early_stop) || !hasproperty(raw, :averaged_from) ||
            !hasproperty(raw, :ema) || !hasproperty(raw, :schedule_state)
        return CheckpointRecord(
            raw.format_version, raw.framework_version,
            raw.ps, raw.st, raw.opt_state, raw.flat_permutation,
            raw.step, raw.epoch, raw.seed, raw.config,
            hasproperty(raw, :devices) ? raw.devices : raw.tunables,
            raw.metrics, raw.run_id, raw.run_url, raw.logger_state,
            raw.logger_type, raw.anchor_checksum, raw.stop_reason,
            hasproperty(raw, :preset) ? raw.preset : nothing,
            hasproperty(raw, :derived) ? raw.derived : nothing,
            hasproperty(raw, :early_stop) ? raw.early_stop : nothing,
            hasproperty(raw, :averaged_from) ? raw.averaged_from : nothing,
            hasproperty(raw, :ema) ? raw.ema : nothing,
            hasproperty(raw, :schedule_state) ? raw.schedule_state : nothing
        )
    end
    return error("ReactantNitro: `$path` does not hold a `CheckpointRecord`; it holds a \
        `$(typeof(raw))`.")
end

function load_checkpoint(ckpt::Checkpointer, path)
    isfile(path) || error("ReactantNitro: there is no checkpoint at `$path`.")
    raw = with_io_retry() do
        _load_record_quietly(path)
    end
    record = _migrate_record(raw, path)
    record.format_version == FORMAT_VERSION || error(
        """
        ReactantNitro: the checkpoint at `$path` is format version $(record.format_version) and this
        framework writes and reads version $FORMAT_VERSION.
        The record was written by framework version $(record.framework_version).
        JLD2 files outlive framework versions, which is why the version is stamped rather than
        assumed; there is no migration path across format versions."""
    )
    return record
end

"""
    ReactantNitro.check_config_compatible(record, e, path; weights_only = false) -> nothing

Compare a record's `GraphConst` fields with `e`'s. These are the fields the compile cache is keyed
on, so for a resume a change means the run would train a different compiled program, and it
refuses with a diff. A weights-only load rebuilds the program from `e` and reuses the parameters
alone, so it warns with the same diff; the parameter tree is checked separately.
A field the record predates is compatible while it holds its declared default, which by convention
reproduces the behaviour from before the field existed.
`Device` fields are recorded separately in `devices`, and `Host` fields may change on resume.
Runs after `derive` has merged, so a derived field is compared under the kind it was declared as.
"""
function check_config_compatible(record, e, path; weights_only::Bool = false)
    now = graphconst_fields(e)
    was = record.config
    was isa NamedTuple || error("ReactantNitro: the checkpoint at `$path` has no usable `config` \
        field; it holds a `$(typeof(was))`.")
    added = [
        k for k in setdiff(keys(now), keys(was))
            if !isequal(getproperty(now, k), _declared_default(typeof(e), k))
    ]
    removed = setdiff(keys(was), keys(now))
    changed = [
        k for k in intersect(keys(now), keys(was))
            if !isequal(getproperty(now, k), getproperty(was, k))
    ]
    (isempty(added) && isempty(removed) && isempty(changed)) && return nothing
    diff = String[]
    for k in changed
        push!(diff, "  $k: $(_diff_repr(getproperty(was, k))) -> $(_diff_repr(getproperty(now, k)))")
    end
    for k in added
        push!(diff, "  $k: (absent in the checkpoint) -> $(_diff_repr(getproperty(now, k)))")
    end
    for k in removed
        push!(diff, "  $k: $(_diff_repr(getproperty(was, k))) -> (absent now)")
    end
    if weights_only
        @warn """
        ReactantNitro: the configuration differs from the one the weights at `$path` were trained
        with:
        $(join(diff, "\n"))
        The program is built from the current experiment and only the parameters are loaded."""
        return nothing
    end
    error(
        """
        ReactantNitro: the configuration changed since the checkpoint at `$path`, so the run
        cannot be resumed:
        $(join(diff, "\n"))
        These are `GraphConst` fields: they bake as trace-time constants and enter the compile cache
        key, so continuing would silently train a different program than the one being resumed.
        `Device` fields are not compared, since they are traced inputs and cannot change the graph,
        and `Host` fields are not compared, since raising `max_epochs` on resume is the normal case.
        A field added since the checkpoint is compatible at its declared default.
        Start a fresh run, or pass `resume = false` to train from scratch in this `run_dir`."""
    )
end

# The declared default of field `k`, converted to the declared type as the constructor converts a
# `GraphConst`; `NO_DEFAULT` when there is none or it cannot be evaluated outside the constructor.
function _declared_default(::Type{E}, k::Symbol) where {E}
    meta = try
        config_metadata(E)
    catch
        return NO_DEFAULT
    end
    haskey(meta, k) || return NO_DEFAULT
    m = getproperty(meta, k)
    m.default === NO_DEFAULT && return NO_DEFAULT
    return try
        convert(m.type, m.default)
    catch
        m.default
    end
end

"""
    ReactantNitro.check_record_tree_compatible(record, layout::FlatLayout, path) -> nothing

Refuse a record whose parameter tree does not fit the model this experiment builds, naming the
first mismatching leaves by keypath with both sizes. `layout` is the fresh model's, built before
the record's parameters replace it. Keyed by keypath, so a changed `param_group` is not a mismatch.
"""
function check_record_tree_compatible(record, layout::FlatLayout, path)
    was = Dict(r.keypath => r.size for r in record.flat_permutation)
    now = Dict(r.keypath => r.size for r in layout.permutation)
    diff = String[]
    for r in layout.permutation
        haskey(was, r.keypath) || push!(diff, "  $(r.keypath): absent in the checkpoint, size $(r.size) now")
        haskey(was, r.keypath) && was[r.keypath] != r.size &&
            push!(diff, "  $(r.keypath): size $(was[r.keypath]) in the checkpoint, $(r.size) now")
    end
    for r in record.flat_permutation
        haskey(now, r.keypath) || push!(diff, "  $(r.keypath): size $(r.size) in the checkpoint, absent now")
    end
    isempty(diff) && return nothing
    length(diff) > 6 && (diff = push!(diff[1:6], "  ..."))
    error(
        """
        ReactantNitro: the checkpoint at `$path` holds a parameter tree that does not match the model
        this experiment builds:
        $(join(diff, "\n"))
        This usually means the architecture fields differ from the ones the checkpoint was trained with."""
    )
end

# A long value, such as program text, is cut so the diff stays readable in a size-limited console.
function _diff_repr(x; limit = 200)
    s = repr(x)
    length(s) <= limit && return s
    return first(s, 60) * "... ($(length(s)) characters)"
end

"""
    ReactantNitro.graphconst_fields(e) -> NamedTuple

The fields of `e` that are neither `Device` nor `Host`: the fields the compile cache is keyed on
and the resume check compares. Both use this one definition.
"""
function graphconst_fields(e)
    T = typeof(e)
    df, hf = device_fields(T), host_fields(T)
    ks = Tuple(f for f in fieldnames(T) if !(f in df) && !(f in hf))
    return NamedTuple{ks}(map(f -> getfield(e, f), ks))
end

"""
    ReactantNitro.device_values(e) -> NamedTuple

The experiment's `Device` field values, read back to host, as stored in the record's `devices`
field. Recording them keeps a change to a derived or adaptive `Device` value visible after the
fact, although the resume check does not compare them.
"""
function device_values(e)
    T = typeof(e)
    df = device_fields(T)
    return NamedTuple{df}(map(f -> to_host(getfield(e, f)), df))
end

"""
    ReactantNitro.check_permutation_compatible(record, layout, path) -> nothing

Refuse a record whose flat parameter permutation differs from `layout`'s, since the restored
`opt_state` would no longer line up with `ps`. Each [`LeafRow`](@ref) carries keypath, group,
offset, length and size, so shapes are compared as well as counts, and the error names the leaves
that moved.
"""
function check_permutation_compatible(record, layout::FlatLayout, path)
    was, now = record.flat_permutation, layout.permutation
    was == now && return nothing
    diff = String[]
    if length(was) != length(now)
        push!(diff, "  the checkpoint has $(length(was)) parameter leaves and this model has \
                     $(length(now))")
    end
    for (a, b) in zip(was, now)
        a == b && continue
        push!(diff, "  $(a.keypath): group $(repr(a.group)) size $(a.size) offset $(a.offset) \
                     -> group $(repr(b.group)) size $(b.size) offset $(b.offset)")
        length(diff) >= 6 && (push!(diff, "  ..."); break)
    end
    error(
        """
        ReactantNitro: the flat parameter permutation changed since the checkpoint at `$path`:
        $(join(diff, "\n"))
        A different permutation scrambles the restored `opt_state` against `ps`, so every moment
        would be applied to the wrong parameter, with no error and a loss curve that reads as a
        bad learning rate. This usually means `build_model` or `param_group` changed."""
    )
end

"""
    ReactantNitro.check_optimizer_compatible(record, fresh, path) -> nothing

Check, for `weights = path` with `restore_optimizer = true`, that the record's `opt_state` fits the
optimizer state `fresh` that setup just built. Compared per group: the rule (its type name, and
each member's for an `OptimiserChain`) and the shape of every state leaf. A mismatch refuses,
naming the group and both sides. A resume skips this check, since it restores state the same run
wrote under the same config; a weights-only start is a new run whose `optimizer` hook may have
changed. The permutation check has already run, so the groups hold the same leaves.
"""
function check_optimizer_compatible(record, fresh, path)
    was, now = to_host(record.opt_state), to_host(fresh)
    diff = String[]
    if was isa Tuple && now isa Tuple && length(was) != length(now)
        push!(diff, "  the checkpoint has $(length(was)) optimizer groups and this run has $(length(now))")
    elseif was isa Tuple && now isa Tuple
        for gi in eachindex(now)
            a, b = was[gi], now[gi]
            ra = a isa Optimisers.Leaf ? _rule_name(a.rule) : string(nameof(typeof(a)))
            rb = b isa Optimisers.Leaf ? _rule_name(b.rule) : string(nameof(typeof(b)))
            ra == rb || push!(diff, "  group $(gi): rule $(ra) in the checkpoint, $(rb) now")
            sa = [_leaf_shape(x) for x in Functors.fleaves(a isa Optimisers.Leaf ? a.state : a)]
            sb = [_leaf_shape(x) for x in Functors.fleaves(b isa Optimisers.Leaf ? b.state : b)]
            sa == sb || push!(diff, "  group $(gi): state leaf shapes $(sa) in the checkpoint, $(sb) now")
        end
    else
        sa = [_leaf_shape(x) for x in Functors.fleaves(was)]
        sb = [_leaf_shape(x) for x in Functors.fleaves(now)]
        sa == sb || push!(diff, "  optimizer state leaf shapes $(sa) in the checkpoint, $(sb) now")
    end
    isempty(diff) && return nothing
    return error(
        """
        ReactantNitro: `restore_optimizer = true` takes the optimizer state of the checkpoint at
        `$path`, and it does not fit the optimizer this run builds:
        $(join(first(diff, 6), "\n"))
        The state is restored as the moments of THIS run's rules, so a different rule or group
        layout would apply them to the wrong thing. Build the run with the optimizer that wrote the
        checkpoint, or leave `restore_optimizer` off for a fresh optimizer."""
    )
end

_rule_name(r::Optimisers.OptimiserChain) = "OptimiserChain(" * join(map(_rule_name, r.opts), ", ") * ")"
_rule_name(r) = string(nameof(typeof(r)))
_leaf_shape(x) = x isa AbstractArray ? size(x) : ()

"""
    ReactantNitro.anchor_checksums(anchors) -> NTuple or nothing

The decay-anchor checksum of each group, `nothing` for a `:zero` group: SHA256 of the anchor's
host bytes, which unlike `Base.hash` is stable across processes and Julia versions.
"""
anchor_checksums(::Nothing) = nothing
anchor_checksums(anchors::Tuple) = map(anchor_checksum, anchors)

anchor_checksum(::Nothing) = nothing
function anchor_checksum(a)
    host = a isa AbstractArray ? Array(a) : a
    return bytes2hex(SHA.sha256(reinterpret(UInt8, vec(host))))
end

"""
    ReactantNitro.check_anchors_compatible(record, layout, anchors, path) -> nothing

Refuse a record whose decay-anchor checksums differ from those of the freshly captured `w0`,
naming the group and both checksums. Storing a checksum instead of the anchor arrays is sound only
because of this check: continuing past a mismatch would regularize toward a different point.
"""
function check_anchors_compatible(record, layout::FlatLayout, anchors, path)
    fresh = anchor_checksums(anchors)
    was = record.anchor_checksum
    (fresh === nothing && was === nothing) && return nothing
    (fresh === nothing || was === nothing || length(fresh) != length(was)) && error(
        """
        ReactantNitro: the checkpoint at `$path` was written with
        $(was === nothing ? "no" : string(count(!isnothing, was))) anchored parameter groups and this
        run has $(fresh === nothing ? "none" : string(count(!isnothing, fresh))).
        A `decay_anchor` changed between the two runs, so the decay is not the same regularizer
        and the run cannot be resumed into it."""
    )
    for gi in eachindex(fresh)
        isequal(fresh[gi], was[gi]) && continue
        error(
            """
            ReactantNitro: the decay anchor for group $(repr(layout.groups[gi])) does not match the
            checkpoint at `$path`:
              checkpoint: $(repr(was[gi]))
              rebuilt:    $(repr(fresh[gi]))
            The record stores a CHECKSUM of the anchor rather than the arrays, which keeps a
            checkpoint from roughly doubling in size for a `:w0`-anchored backbone, and that is
            sound only because this check refuses on mismatch.
            The likely cause is that `build_model` is not reproducible for this group under the
            restored seed: a pretrained load that is not deterministic, or an initializer reading
            a different rng. Continuing is not offered, because it would silently regularize
            toward a different point than the run being resumed, with a plausible loss curve and
            no error.
            Make the source deterministic, or pass `decay_anchor` an explicit array you persist
            yourself."""
        )
    end
    return nothing
end

"""
    ReactantNitro.check_logger_compatible(record, logger, path; weights_only = false) -> nothing

Refuse to hand a record's `logger_state` to a logger of another type, naming both types. A record
with no `logger_state` passes. A record with one and a run with no logger is refused unless
`weights_only = true`: a weights-only restore continues no logged run, so a checkpoint written with
a logger can be evaluated or exported without one.
"""
function check_logger_compatible(record, logger, path; weights_only::Bool = false)
    record.logger_state === nothing && return nothing
    logger = inner_logger(logger)
    logger === nothing && !weights_only && error(
        """
        ReactantNitro: the checkpoint at `$path` carries resumption state for a
        `$(record.logger_type)` logger, and this run has no logger.
        Pass the same logger to continue its experiment, or drop the state deliberately by resuming
        with `resume = false`."""
    )
    logger === nothing && return nothing
    now = string(nameof(typeof(logger)))
    now == record.logger_type && return nothing
    error(
        """
        ReactantNitro: the checkpoint at `$path` carries resumption state written by a
        `$(record.logger_type)` logger, and this run's logger is a `$now`.
        `logger_state` is opaque to the framework and backend-specific, so handing it to a
        different backend is a bug the framework can see and refuses rather than passing on."""
    )
end

"""
    ReactantNitro.restored_devices(record, e) -> NamedTuple

The `Device` values a weights-only restore takes from the record instead of recomputing them,
since an evaluation process may not have the training data `derive` reads. Only fields the
experiment still declares as `Device` are taken.
"""
function restored_devices(record, e)
    record.devices isa NamedTuple || return (;)
    df = device_fields(typeof(e))
    ks = Tuple(k for k in keys(record.devices) if k in df)
    return NamedTuple{ks}(map(k -> getproperty(record.devices, k), ks))
end

"""
    ReactantNitro.restored_graphconsts(record, e) -> NamedTuple

The derived `GraphConst` values a weights-only restore takes from `record.config`, for the keys in
`record.derived`. A key the experiment no longer declares as `GraphConst` is skipped and left for
the config check to report.
"""
function restored_graphconsts(record, e)
    gc = keys(graphconst_fields(e))
    ks = Tuple(k for k in record.derived if k in gc && haskey(record.config, k))
    return NamedTuple{ks}(map(k -> getproperty(record.config, k), ks))
end

"""
    ReactantNitro.check_checkpointer(ckpt, collection, routing) -> nothing

The checkpointer's setup-time checks: `k`, `scope`, and the selection metric when the
experiment defines no `metrics`, so that the only key is `:loss`. With a `metrics` method the
keys exist only once it has run, and [`save_checkpoint!`](@ref) raises at the first checkpoint
instead. A run with no `val` split passes, since the newest checkpoint is kept regardless, except
under `keep_latest = false`, which keeps nothing unscored: a handle with a `train` split and no
`val` split is then refused.
"""
check_checkpointer(::Nothing, collection, routing) = nothing

function check_checkpointer(ckpt::Checkpointer, collection, routing)
    # `mode` was checked at construction (`check_monitor`).
    check_scope(ckpt.scope)
    check_keep(ckpt.keep)
    (
        haskey(collection, :val) && routing !== nothing && routing.metrics === nothing &&
            ckpt.metric !== nothing && ckpt.metric !== :loss
    ) && error(
        """
        ReactantNitro: the checkpointer selects on `$(ckpt.metric)`, and this experiment defines no
        `metrics` method, so the only metric the framework will emit is `:loss`, the validation loss.
        Either select on `:loss`, or define `metrics` emitting `$(ckpt.metric)`.$(split_prefix_hint(ckpt.metric))"""
    )
    (all(sel -> sel isa Best, ckpt.keep) && haskey(collection, :train) && !haskey(collection, :val)) && error(
        """
        ReactantNitro: the checkpointer keeps only `Best` selectors (what `keep_latest = false`
        builds), and this run has a `train` split and no `val` split, so no epoch gets a
        `$(ckpt.metric)` score. `Best` retains by score, so nothing this run trained would ever be
        saved. Add `Last(1)` to `keep` (the default keeps the latest record whatever it scored),
        or add a `val` split."""
    )
    return nothing
end

"""
    ReactantNitro.warn_keep_latest(ckpt, resume) -> nothing

Warn once, at setup of a handle that trains with no [`Last`](@ref) in `keep` (what
`keep_latest = false` builds), what the run gives up by not keeping its latest record: the final state, and a resume from where training stopped. Under
`resume = :auto` the warning adds that this handle continued from the newest retained record and
will retrain the epochs after it.
"""
warn_keep_latest(ckpt, resume) = nothing
function warn_keep_latest(ckpt::Checkpointer, resume)
    ckpt.keep_latest && return nothing
    kept = join(map(repr, ckpt.keep), ", ")
    auto = resume === :auto ? " This handle was built with `resume = :auto`, so where the \
        directory held a record it continues from the run's latest RETAINED record and retrains \
        every epoch after it; without `Last` (`keep_latest = false`) that is not where the \
        earlier process stopped." : ""
    @warn "ReactantNitro: the checkpointer keeps $(kept) and no `Last` (what `keep_latest = false` \
           builds), so this run does not write an epoch none of them keeps. The state training \
           ends in is not saved unless the last epoch is kept. \
           `resume = :auto` and `weights = :latest => :latest` resolve to the most recent \
           improvement, not to where training stopped, and a crash loses every epoch since that \
           improvement.$(auto)"
    return nothing
end

"""
    ReactantNitro.warn_retention_scope(ckpt) -> nothing

Warn at setup of a handle that trains, about the entries its retention decides over (this run's
under `scope = :run`, the directory's under `scope = :dir`): when they are scored under mixed
metrics ([`warn_mixed_metrics`](@ref)), and, under `scope = :dir`, when the directory already holds
other runs' checkpoints that this run may delete as it beats them.
"""
warn_retention_scope(ckpt) = nothing
function warn_retention_scope(ckpt::Checkpointer)
    dir = checkpoint_dir(ckpt)
    entries = isdir(dir) ? read_manifest(dir) : ManifestEntry[]
    pool = ckpt.scope === :dir ? entries : ManifestEntry[e for e in entries if isequal(e.run, ckpt.run)]
    warn_mixed_metrics(pool, ckpt, "this run's retention")
    ckpt.scope === :dir || return nothing
    theirs = [e for e in entries if !isequal(e.run, ckpt.run) && same_selection(e, ckpt)]
    isempty(theirs) && return nothing
    n = length(unique(e.run for e in theirs))
    @warn "ReactantNitro: the checkpointer has `scope = :dir`, and `$(dir)` already holds \
           $(length(theirs)) checkpoint(s) of $(n) other run(s) (legacy entries included) scored \
           under $(_metric_label(ckpt.metric, ckpt.mode)). The directory keeps only its best \
           $(ckpt.k) across every run$(ckpt.keep_latest ? " plus this run's latest" : ""), so they \
           may be deleted as this run beats them. Checkpoints under other metrics are kept. \
           `runs(\"$(dir)\")` lists what is there."
    return nothing
end

# The directory a lookup reads: the checkpointer's pinned `dir`, else the `run_dir` it was given.
lookup_dir(ckpt::Checkpointer, run_dir) = ckpt.dir === nothing ? String(run_dir) : ckpt.dir

# One run's entries. `run = :recent` is the most recent run in the directory, which is what a
# lookup that names no run means; `nruns` is how many the directory holds, for the announcement.
function run_scope(dir, run)
    entries = isdir(dir) ? read_manifest(dir) : ManifestEntry[]
    ids = manifest_runs(entries)
    run === :recent && (run = isempty(ids) ? nothing : first(ids))
    return (; run, entries = ManifestEntry[e for e in entries if isequal(e.run, run)], nruns = length(ids))
end

"""
    ReactantNitro.find_latest(ckpt, run_dir; run = :recent) -> path or nothing

The path of one run's highest-epoch retained record in `run_dir`, read from the manifest: what
`:latest => :latest` and `resume = :auto` resolve to. Under `keep_latest = false` that is the run's
most recent improvement, not where training stopped. `run` is a run id, `nothing` for the unnamed
run of a manifest written before runs were recorded, or `:recent` (the default), the run that wrote
the directory's most recent entry. Returns `nothing` when there is no such record or its file is
gone; [`resolve_checkpoint`](@ref) is the form that raises and says what the directory holds.
"""
find_latest(::Nothing, run_dir; run = :recent) = nothing

function find_latest(ckpt::Checkpointer, run_dir; run = :recent)
    dir = lookup_dir(ckpt, run_dir)
    entries = run_scope(dir, run).entries
    isempty(entries) && return nothing
    newest = entries[argmax([e.epoch for e in entries])]
    path = joinpath(dir, newest.file)
    return isfile(path) ? path : nothing
end

"""
    ReactantNitro.selected_checkpoint(ckpt, run_dir; run = :recent) -> NamedTuple or `nothing`

The best of one run's checkpoints by the checkpointer's current `metric` and `mode`, as
`(; path, epoch, metric, score, run)`, read from the manifest alone. `run` is as for
[`find_latest`](@ref), the most recent run by default; `train!` passes its own. Only entries scored
under the current metric and mode are ranked; an entry with no recorded metric predates the field
and is ranked. Returns `nothing` with no checkpointer, no directory, no scored entry, or a missing
winning file. `show` and `train!` use this quiet form; `:latest => :best` goes through
[`resolve_checkpoint`](@ref), which raises and warns about mixed metrics.
"""
selected_checkpoint(::Nothing, run_dir; run = :recent) = nothing

function selected_checkpoint(ckpt::Checkpointer, run_dir; run = :recent)
    dir = lookup_dir(ckpt, run_dir)
    scope = run_scope(dir, run)
    ckpt = bind_selection(ckpt, scope.entries)
    scored = [e for e in scope.entries if e.score !== nothing && same_selection(e, ckpt)]
    isempty(scored) && return nothing
    pick = ckpt.mode === :max ? argmax : argmin
    best = scored[pick([e.score for e in scored])]
    path = joinpath(dir, best.file)
    isfile(path) || return nothing
    return (; path, epoch = best.epoch, metric = ckpt.metric, score = best.score, run = scope.run)
end

# What `train!` reports as the handle's selection: within ITS run, not the directory's most recent,
# which a second process writing into the same directory could make someone else's.
own_selected_checkpoint(ckpt::Checkpointer, run_dir) =
    selected_checkpoint(ckpt, run_dir; run = ckpt.run)
own_selected_checkpoint(ckpt, run_dir) = selected_checkpoint(ckpt, run_dir)

# The announcement for a lookup that named no run, in a directory holding more than one: which
# run it read, since the answer is otherwise indistinguishable from the other runs' files.
announce_recent_run(ckpt, run_dir, what) = nothing
function announce_recent_run(ckpt::Checkpointer, run_dir, what)
    dir = lookup_dir(ckpt, run_dir)
    scope = run_scope(dir, :recent)
    scope.nruns > 1 && @info "ReactantNitro: `$(dir)` holds $(scope.nruns) runs; $(what) reads the \
        most recent, $(run_label(scope.run)). `runs(\"$(dir)\")` lists them."
    return scope.run
end

# ── Which runs a directory holds ────────────────────────────────────────────────────

# The id a run is named by in `runs` and in a `"<id>" => :best` source. The unnamed run of a
# manifest written before runs were recorded is `"legacy"`, which no generated id can be: those are
# eight characters.
const LEGACY_RUN = "legacy"
run_key(run) = run === nothing ? LEGACY_RUN : run
run_from_key(id::AbstractString) = id == LEGACY_RUN ? nothing : String(id)

"""
    ReactantNitro.RunSummary

One run of a run directory, as [`runs`](@ref) reports it: `id` (the run id, or `"legacy"` for
the entries of a manifest written before runs were recorded), `checkpoints` (how many the manifest
keeps), `epochs` (theirs, ascending), `metric` and `mode` (the selection of the run's most recently
written scored entry, `nothing` for a run with no score), `best` (the best score under that
selection), `metrics` (every `(metric, mode)` pair the run's entries were scored under, with
counts), `written` (the wall-clock `time()` of its latest write, `0.0` for legacy) and `parent`
(the checkpoint a branched run was resumed from).
"""
struct RunSummary
    id::String
    checkpoints::Int
    epochs::Vector{Int}
    metric::Union{Symbol, Nothing}
    mode::Union{Symbol, Nothing}
    best::Union{Float64, Nothing}
    metrics::Vector{Pair{Tuple{Union{Symbol, Nothing}, Union{Symbol, Nothing}}, Int}}
    written::Float64
    parent::Union{String, Nothing}
end

function Base.show(io::IO, r::RunSummary)
    n = r.checkpoints
    print(io, "run ", r.id, ": ", n, n == 1 ? " checkpoint" : " checkpoints")
    isempty(r.epochs) || print(io, ", epochs ", join(r.epochs, ", "))
    r.best === nothing || print(
        io, ", best ", something(r.metric, "score"),
        r.mode === nothing ? "" : " (" * string(r.mode) * ")", " ", _shown(r.best)
    )
    length(r.metrics) > 1 && print(
        io, ", mixed metrics: ",
        join((_metric_label(m...) * " x" * string(c) for (m, c) in r.metrics), ", ")
    )
    print(
        io, ", last written ",
        r.written == 0.0 ? "before runs were recorded" : Libc.strftime("%Y-%m-%d %H:%M:%S", r.written)
    )
    r.parent === nothing || print(io, ", branched from ", r.parent)
    return nothing
end

_metric_label(metric, mode) = metric === nothing ? "unrecorded metric" :
    "`" * string(metric) * "`" * (mode === nothing ? "" : " (" * string(mode) * ")")

"""
    runs(run_dir) -> Vector{RunSummary}

The runs a run directory's manifest holds, most recent first, one [`RunSummary`](@ref) each: id,
how many checkpoints it keeps and their epochs, its best score and the metric and mode that score
is under, when it last wrote, and the checkpoint it branched from. The entries of a manifest
written before runs were recorded are one run, `"legacy"`, listed last. An id names the run in a
checkpoint source, `weights = "<id>" => :best`, and `checkpoint_run(nitro)` is a handle's own.
Answered from the manifest; no record is opened. Empty for a directory no run has written to.

```julia
runs("runs/MyExp")
n = Nitro(MyExp(); run_dir = "runs/MyExp", weights = first(runs("runs/MyExp")).id => :best)
```
"""
function runs(run_dir::AbstractString)
    entries = isdir(run_dir) ? read_manifest(run_dir) : ManifestEntry[]
    out = RunSummary[]
    for run in manifest_runs(entries)
        mine = sort!([e for e in entries if isequal(e.run, run)]; by = e -> e.epoch)
        scored = [e for e in mine if e.score !== nothing]
        counts = Dict{Tuple{Union{Symbol, Nothing}, Union{Symbol, Nothing}}, Int}()
        for e in scored
            counts[(e.metric, e.mode)] = get(counts, (e.metric, e.mode), 0) + 1
        end
        metric, mode, best = nothing, nothing, nothing
        if !isempty(scored)
            recent = scored[argmax([e.written for e in scored])]
            metric, mode = recent.metric, recent.mode
            same = [e.score for e in scored if e.metric === metric && e.mode === mode]
            best = mode === :max ? maximum(same) : minimum(same)
        end
        push!(
            out, RunSummary(
                run_key(run), length(mine), [e.epoch for e in mine], metric, mode, best,
                sort!(collect(counts); by = p -> -last(p)),
                maximum(e.written for e in mine), mine[end].parent
            )
        )
    end
    return out
end

"""
    checkpoint_run(nitro) -> String or nothing

The id this handle's checkpoints are written under, the `run` of their manifest entries, and what
names the run in a checkpoint source: `weights = "<id>" => :best`. `nothing` with no
`Checkpointer`, and for a run continuing the unnamed run of a manifest written before runs were
recorded, which [`runs`](@ref) lists as `"legacy"`. Not [`run_id`](@ref), which is the logger's.
"""
checkpoint_run(nitro::Nitro) =
    nitro.checkpointer isa Checkpointer ? nitro.checkpointer.run : nothing

# ── Mixed metrics ────────────────────────────────────────────────────────────────────

# The entries scored under a (metric, mode) other than the checkpointer's, grouped with counts.
# A legacy entry, with no recorded metric, is taken to match and is never "mixed".
function other_selections(entries, ckpt::Checkpointer)
    counts = Dict{Tuple{Symbol, Union{Symbol, Nothing}}, Int}()
    for e in entries
        (e.score === nothing || same_selection(e, ckpt)) && continue
        counts[(e.metric, e.mode)] = get(counts, (e.metric, e.mode), 0) + 1
    end
    return sort!(collect(counts); by = p -> -last(p))
end

"""
    ReactantNitro.warn_mixed_metrics(entries, ckpt, what) -> nothing

Warn when some of `entries` were scored under a metric or mode other than the checkpointer's,
listing each pair with its count and saying they are kept. Selection ranks only the current pair,
so those checkpoints are invisible to `:best` and to retention. Called at setup of a training
handle, over the entries its retention decides over, and by every `:best` lookup, over the entries
it considered. The manifest records a metric's name only, so a metric whose definition changed
under the same name is not detected.
"""
function warn_mixed_metrics(entries, ckpt::Checkpointer, what)
    others = other_selections(entries, ckpt)
    isempty(others) && return nothing
    list = join((_metric_label(m...) * " x" * string(c) for (m, c) in others), ", ")
    @warn "ReactantNitro: $(what) considers checkpoints scored under other metrics: $(list). \
           Selection ranks only the current $(_metric_label(ckpt.metric, ckpt.mode)), so those \
           checkpoints are neither ranked against it nor ever deleted by retention; they are kept."
    return nothing
end

# ── Resolving a checkpoint source ───────────────────────────────────────────────────

# What a directory holds, for a message: its run ids, and the reader that lists them.
function holds_text(dir, entries)
    ids = [run_key(r) for r in manifest_runs(entries)]
    body = isempty(ids) ? "`$(dir)` holds no checkpoints" :
        "`$(dir)` holds $(length(ids)) run(s): $(join(("`$(i)`" for i in ids), ", "))"
    return body * "; `runs(\"$(dir)\")` lists them"
end

# ── Averaging checkpoints ───────────────────────────────────────────────────────────

"""
    Average(sel::Selector; weight = :uniform, state = nothing)

The parameters of the checkpoints `sel` picks from one run, averaged, as a `weights` source:

```julia
Nitro(e; weights = :latest => Average(Last(10)))
Nitro(e; weights = "a1b2c3" => Average(Best(5; among = Last(20))))
```

`weight` is `:uniform`, the only weighting so far. Lux state is not averaged: it comes from one
picked checkpoint, the best when `sel` ranks (a [`Best`](@ref)) and the latest otherwise, or as
`state = :best | :latest` says. That is exact for a model without running statistics and an
approximation for one with them, such as BatchNorm. The optimizer starts fresh, so
`restore_optimizer = true` and `resume` refuse an `Average`. Averaging across runs (`:all`) is
refused for now. The run must still hold every checkpoint `sel` names; keep them while training
with [`Checkpointer`](@ref)`(; keep)`, for example `keep = (Best(3), Last(10))`.
"""
struct Average{S <: Selector}
    sel::S
    weight::Symbol
    state::Union{Nothing, Symbol}
end

function Average(sel::Selector; weight::Symbol = :uniform, state = nothing)
    weight === :uniform || error(
        "ReactantNitro: `Average(...; weight = $(repr(weight)))` is not supported yet; `:uniform` \
         is the only weighting."
    )
    state in (nothing, :latest, :best) || error(
        "ReactantNitro: `Average(...; state = $(repr(state)))`: `state` is `:latest`, `:best`, or \
         left out to follow the selection."
    )
    return Average(sel, weight, state)
end

Base.:(==)(a::Average, b::Average) = a.sel == b.sel && a.weight === b.weight && a.state === b.state
Base.show(io::IO, a::Average) = print(
    io, "Average(", a.sel, a.state === nothing ? "" : "; state = " * repr(a.state), ")"
)

# How many entries `sel` names when the run holds them all; `nothing` when it is open-ended.
expected_count(sel::Best) = sel.k
expected_count(sel::Last) = sel.n
expected_count(sel::Epochs) = length(sel.epochs)
expected_count(::Selector) = nothing

# Whether `sel` ranks by score, which decides where an `Average` takes its Lux state from.
ranks(::Best) = true
ranks(::Selector) = false

"""
    ReactantNitro.covers(keep, sel) -> Bool or nothing

Whether a checkpointer retaining `keep` is sure to hold everything `sel` picks: `true`, `false`,
or `nothing` when that cannot be decided without the manifest. `Last(n)` is covered by a
`Last(m)` with `m >= n`; `Best(k)` by a `Best(j)` with `j >= k`; `Best(k; among)` by whatever
covers `among`.
"""
function covers(keep, sel::Last)
    return any(s -> s isa Last && s.n >= sel.n, keep)
end
function covers(keep, sel::Best)
    sel.among === nothing || return covers(keep, sel.among)
    return any(s -> s isa Best && s.among === nothing && s.k >= sel.k, keep)
end
covers(keep, sel::Selector) = any(==(sel), keep) ? true : nothing

# The entries a lookup decides over: one run's (`:latest`, a run id) or the directory's (`:all`),
# with a label for messages, the directory, and the run id (`nothing` for `:all`).
function run_pool(ckpt, run_dir, run; what)
    dir = lookup_dir(ckpt, run_dir)
    entries = isdir(dir) ? read_manifest(dir) : ManifestEntry[]
    pool, label, r = if run === :all
        entries, "every run in `$(dir)`", nothing
    elseif run === :latest
        isempty(entries) && error("ReactantNitro: $(what) found no checkpoint: $(holds_text(dir, entries)).")
        r = announce_recent_run(ckpt, dir, what)
        ManifestEntry[e for e in entries if isequal(e.run, r)], run_label(r), r
    else
        r = run_from_key(run)
        mine = ManifestEntry[e for e in entries if isequal(e.run, r)]
        isempty(mine) && error(
            "ReactantNitro: $(what) names run `$(run)`, which has no checkpoints here: \
             $(holds_text(dir, entries))."
        )
        mine, run_label(r), r
    end
    isempty(pool) && error("ReactantNitro: $(what) found no checkpoint: $(holds_text(dir, entries)).")
    return pool, label, dir, r
end

"""
    ReactantNitro.AverageSource

An [`Average`](@ref) resolved against a run's manifest: the files to average, in epoch order, the
file the Lux state and metadata come from, and a description for messages and provenance.
"""
struct AverageSource
    spec::Average
    dir::String
    run::Union{String, Nothing}
    files::Vector{String}
    epochs::Vector{Int}
    state_file::String
end

describe(a::AverageSource) = string(
    a.spec, " of ", run_label(a.run), " in `", a.dir, "`, epochs ", join(a.epochs, ", ")
)

"""
    ReactantNitro.resolve_average(ckpt, run_dir, run, avg; what) -> AverageSource

Pick `avg`'s checkpoints from one run's manifest. Refuses when the run no longer holds all of them,
naming the `keep` that would have.
"""
function resolve_average(ckpt, run_dir, run, avg::Average; what)
    run === :all && error(
        "ReactantNitro: $(what) averages across every run in the directory. Averaging across runs \
         (model soups) is planned and not supported yet: it needs a shared initialization and a \
         layout check across runs. Name one run: `:latest => $(avg)` or `\"<run id>\" => $(avg)`."
    )
    pool, label, dir, r = run_pool(ckpt, run_dir, run; what)
    ckpt = bind_selection(ckpt, pool)
    view = (; run = r, metric = ckpt.metric, mode = ckpt.mode, scope = :run)
    ranks(avg.sel) && warn_mixed_metrics(pool, ckpt, what)
    picked = sort(select(avg.sel, pool, view); by = e -> e.epoch)
    n = expected_count(avg.sel)
    held = join(sort([e.epoch for e in pool]), ", ")
    want = n === nothing ? "at least one checkpoint" : "$(n) checkpoints"
    suggest = avg.sel isa Last ? avg.sel : Last(something(n, 10))
    (isempty(picked) || (n !== nothing && length(picked) < n)) && error(
        """
        ReactantNitro: $(what) needs $(want)
        and $(label) holds $(length(picked)) that `$(avg.sel)` picks. Its retained epochs are
        $(isempty(held) ? "(none)" : held). Keep them while training, for example
        `Checkpointer(; keep = (Best(3), $(suggest)))`."""
    )
    state = something(avg.state, ranks(avg.sel) ? :best : :latest)
    src = if state === :latest
        picked[argmax([e.epoch for e in picked])]
    else
        scored = [e for e in picked if e.score !== nothing && same_selection(e, view)]
        isempty(scored) && error(
            "ReactantNitro: $(what) takes its Lux state from the best picked checkpoint, and none \
             is scored under $(_metric_label(ckpt.metric, ckpt.mode)). Pass `state = :latest`."
        )
        scored[(ckpt.mode === :max ? argmax : argmin)([e.score for e in scored])]
    end
    return AverageSource(avg, dir, r, [e.file for e in picked], [e.epoch for e in picked], src.file)
end

"""
    ReactantNitro.average_records(ckpt, src::AverageSource) -> CheckpointRecord

The averaged record: `src.state_file`'s record with `ps` replaced by the uniform mean of every
picked record's `ps`, and `averaged_from` saying what was averaged. Records are read one at a time into a `Float64` accumulator, so memory holds
one record and the sum, not all of them. Refuses records whose parameter layouts differ.
"""
function average_records(ckpt, src::AverageSource; field::Symbol = :ps)
    # Any `Checkpointer` reads the framework's format; this handle's may be `nothing`.
    ckpt isa Checkpointer || (ckpt = Checkpointer())
    base = load_checkpoint(ckpt, joinpath(src.dir, src.state_file))
    zero64(x) = x isa AbstractArray{<:AbstractFloat} ? zeros(Float64, size(x)) : x
    getfield(base, field) === nothing && error(
        "ReactantNitro: averaging the records' `$(field)`, and `$(src.state_file)` keeps none (its \
         run kept no EMA)."
    )
    acc = Functors.fmap(zero64, getfield(base, field))
    for f in src.files
        rec = load_checkpoint(ckpt, joinpath(src.dir, f))
        getfield(rec, field) === nothing && error(
            "ReactantNitro: averaging the records' `$(field)`, and `$(f)` keeps none."
        )
        rec.flat_permutation == base.flat_permutation || error(
            "ReactantNitro: `$(f)` and `$(src.state_file)` have different parameter layouts, so \
             they cannot be averaged leaf for leaf. Average checkpoints of one architecture."
        )
        Functors.fmap(acc, getfield(rec, field)) do a, x
            a isa AbstractArray{Float64} && (a .+= x)
            return a
        end
    end
    n = length(src.files)
    ps = Functors.fmap(acc, getfield(base, field)) do a, x
        a isa AbstractArray{Float64} ? convert(typeof(x), a ./ n) : x
    end
    from = (;
        spec = string(src.spec), run = src.run, dir = src.dir, files = copy(src.files),
        epochs = copy(src.epochs), weights = fill(1 / n, n), state_file = src.state_file,
        field,
    )
    # The average is the record's weights; its own EMA, if any, does not describe them.
    swap(f) = f === :ps ? ps : f === :averaged_from ? from : f === :ema ? nothing : getfield(base, f)
    return CheckpointRecord(map(swap, fieldnames(CheckpointRecord))...)
end

"""
    average_checkpoints(run_dir, run => Average(sel); checkpointer = Checkpointer(), save = nothing)
        -> CheckpointRecord

The averaged record `weights = run => Average(sel)` would load, returned on the host, and written
to `save` when one is given. The written file is an ordinary record: `weights = path` loads it,
and its `averaged_from` field (shown by [`checkpoint_info`](@ref) and recorded in an export's
provenance) names the files, epochs and weights it averages. It is not added to the manifest, so
retention never ranks or deletes it. `save` refuses an existing file.

```julia
average_checkpoints("runs/MyExp", :latest => Average(Last(10)); save = "runs/MyExp/avg-last10.jld2")
```
"""
function average_checkpoints(run_dir, src::Pair; checkpointer = Checkpointer(), save = nothing)
    run, avg = src
    avg isa Average || error(
        "ReactantNitro: `average_checkpoints` takes `run => Average(selector)`; got \
         `$(source_repr(src))`."
    )
    check_source(src, :weights)
    what = "`average_checkpoints($(repr(String(run_dir))), $(source_repr(src)))`"
    rec = average_records(
        checkpointer, resolve_average_source(checkpointer, String(run_dir), run, avg; what)
    )
    if save !== nothing
        path = String(save)
        endswith(path, ".jld2") || error("ReactantNitro: `save = $(repr(path))` must end in `.jld2`.")
        ispath(path) && error(
            "ReactantNitro: `save = $(repr(path))` already exists, and `average_checkpoints` does \
             not overwrite. Remove it or pick another name."
        )
        mkpath(dirname(abspath(path)))
        write_record(path, rec)
    end
    return rec
end

"""
    ReactantNitro.resolve_checkpoint(ckpt, run_dir, run, which; what) -> (; path, epoch, run)

Resolve a `run => checkpoint` source of `weights` or `resume` to a record, from the manifest of
the checkpointer's directory (its `dir`, else `run_dir`):

  * `run` is `:latest` (the directory's most recent run: the run of the entry written last, legacy
    entries counting as oldest), `:all` (every run, only with `which = :best`), or a run id
    (`"legacy"` for the unnamed run of a manifest written before runs were recorded).
  * `which` is `:latest` (the run's newest retained record) or `:best` (the best of the run's
    entries by the checkpointer's CURRENT `metric` and `mode`; across every run for `:all`).

Raises instead of returning `nothing`, since a caller who asked for trained weights and got
fresh ones would not notice. Every miss names the directory, the runs it holds and
`runs(run_dir)`. A `:best` over entries scored under another metric warns that they were not
ranked ([`warn_mixed_metrics`](@ref)), and a run with no entry under the current metric raises,
naming the ones it has. A checkpointer of the user's own knows nothing of runs and answers
`:latest => :best | :latest` through its own [`selected_checkpoint`](@ref) and
[`find_latest`](@ref) methods. `what` is how the messages name the request.
"""
function resolve_checkpoint(ckpt, run_dir, run, which::Symbol; what)
    ckpt === nothing && error(
        "ReactantNitro: $(what) is looked up through the checkpointer's manifest, and \
         `checkpointer = nothing` has none. Leave `checkpointer` at its default to read the run \
         directory the framework writes, or pass the checkpoint's path."
    )
    if !(ckpt isa Checkpointer)
        run === :latest || error(
            "ReactantNitro: $(what) names a run, and this `$(typeof(ckpt))` checkpointer records \
             none. Only `:latest => :best | :latest` or a path can be resolved through it."
        )
        found = which === :best ? selected_checkpoint(ckpt, run_dir) : find_latest(ckpt, run_dir)
        found === nothing && error(
            "ReactantNitro: $(what) found no checkpoint in `$(run_dir)` through this \
             `$(typeof(ckpt))` checkpointer."
        )
        path = which === :best ? found.path : found
        return (; path, epoch = nothing, run = nothing)
    end
    pool, label, dir, _ = run_pool(ckpt, run_dir, run; what)
    entries = read_manifest(dir)
    pick = if which === :latest
        pool[argmax([e.epoch for e in pool])]
    else
        ckpt = bind_selection(ckpt, pool)
        warn_mixed_metrics(pool, ckpt, what)
        scored = [e for e in pool if e.score !== nothing && same_selection(e, ckpt)]
        if isempty(scored)
            others = other_selections(pool, ckpt)
            has = isempty(others) ? "no scored checkpoint at all (a run without a `val` split \
                scores none; use `=> :latest`)" :
                "checkpoints scored only under " *
                join((_metric_label(m...) * " x" * string(c) for (m, c) in others), ", ")
            error(
                "ReactantNitro: $(what) selects by the checkpointer's \
                 $(_metric_label(ckpt.metric, ckpt.mode)), and $(label) has $(has). Set the \
                 checkpointer's `metric` and `mode` to match (`checkpointer = \
                 Checkpointer(; metric = ..., mode = ...)`), or pass the checkpoint's path. \
                 $(holds_text(dir, entries))."
            )
        end
        scored[(ckpt.mode === :max ? argmax : argmin)([e.score for e in scored])]
    end
    path = joinpath(dir, pick.file)
    isfile(path) || error(
        "ReactantNitro: $(what) resolved to `$(path)` (epoch $(pick.epoch) of $(run_label(pick.run))), \
         which the manifest lists and the directory no longer holds. $(holds_text(dir, entries))."
    )
    return (; path, epoch = pick.epoch, run = pick.run)
end

"""
    ReactantNitro.resumed_run(ckpt, path) -> (; run, parent, from)

Which run a resume from `path` continues. Resuming from a run's newest record in this
checkpointer's directory continues that run: its id is adopted, and the new checkpoints join its
retention. Anything else, an earlier record or a file the manifest does not list, starts a new run
with `parent = path`, since continuing the id from an earlier epoch would rewrite that run's later
epochs and rotate the originals away. `from` is `nothing` for a continuation and otherwise says
what was branched from, `(; run, epoch, newest)` for a listed record or `(;)` for an unlisted file.
"""
function resumed_run(ckpt::Checkpointer, path)
    dir = checkpoint_dir(ckpt)
    entries = isdir(dir) ? read_manifest(dir) : ManifestEntry[]
    here = !isempty(entries) && isdir(dirname(abspath(path))) &&
        samefile(dirname(abspath(path)), dir)
    hit = here ? findfirst(e -> e.file == basename(path), entries) : nothing
    hit === nothing && return (; run = new_run_id(), parent = String(path), from = (;))
    e = entries[hit]
    mine = [x for x in entries if isequal(x.run, e.run)]
    newest = mine[argmax([x.epoch for x in mine])]
    newest.file == e.file && return (; run = e.run, parent = e.parent, from = nothing)
    return (;
        run = new_run_id(), parent = String(path),
        from = (; run = e.run, epoch = e.epoch, newest = newest.epoch),
    )
end
