# Checkpoint.jl
#
# Checkpointing: the record, the top-K checkpointer, the manifest, and the resume compatibility
# checks. A "checkpoint" is full training state and "weights" are parameters alone, so optimizer
# state is saved on every checkpoint, including the top-K ones.

"""
    CheckpointRecord

The single authority on what a checkpoint holds: the `snapshot` handed to
[`save_checkpoint!`](@ref) plus the two framework-stamped fields.

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
| `stop_reason` | optional | `:completed`, `:early_stop`, or `:error` |

Every value is a host value: write host, read host, normalize to device residency on the way in.
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

"""
    TopKCheckpointer(; k = 3, metric = :val_loss, mode = :min, dir = nothing, name = nothing,
                       keep_latest = true, scope = :run)

The default checkpointer, writing into [`run_dir`](@ref). Constructible for an arbitrary
experiment because the default `metrics` guarantees `:val_loss`; a user `metrics` that does not
emit the configured metric is a setup error.

The retention rule is the top K by `metric` in union with the newest, whatever it scored: K+1
files when the newest is not among the best, K when it is. `latest` is a rule, not a symlink,
since a symlink into the top-K set dangles the moment rotation deletes its target.

**`keep_latest = false`** drops the newest from the rule: the run keeps its top K and nothing
else, and an epoch whose score cannot enter the current top K is not written at all, which
saves writing a full record (hundreds of MB on a remote filesystem) only to delete it. A score must
beat the K-th best strictly; a tie with it does not enter. The final `stop_reason` rewrite follows
the same rule, so the state training ended in is saved only when the last epoch was a top-K epoch,
and `weights = :latest => :latest` and `resume = :auto` resolve to the run's newest RETAINED
record, its most recent improvement, rather than to where training stopped. A crash loses every
epoch since that improvement, and a resume from it retrains them. Setup warns once about all of
this for a handle that trains, and refuses the flag for a training run with no `val` split, whose
epochs have no score and could never be kept. The default, `true`, is the rule above. A small
manifest (file, epoch, score, `stop_reason`, and the run, metric and mode that wrote it) is written
alongside so resume finds the newest without reading every file. `checkpointer = nothing` disables
checkpointing. `name` is the filename hook, `name(; epoch, step, metric, score)`; `nothing` adopts
the experiment's [`checkpoint_filename`](@ref) at setup, as `dir` adopts `run_dir`.

**Retention is per run by default (`scope = :run`).** One directory routinely holds several runs
(the default `run_dir` is the experiment's type name), so setup gives every run an id, `run`, and
the rule above applies to that run's entries alone: another run's checkpoints are never ranked,
rotated, deleted or overwritten. A fresh run gets a new id; `resume` continues the id of the run it
resumes when it resumes that run's newest record, and branches into a new one, recording the source
in `parent`, otherwise. Setup assigns both fields every time, so a checkpointer object reused across
handles belongs to the latest one. A checkpointer driven by hand, which never went through setup,
has `run = nothing`, the same unnamed run every manifest entry written before runs were recorded
belongs to.

**`scope = :dir`** is the space-saving strategy: the whole directory keeps only its top K by
`metric` across every run, plus this run's newest (by write time, and only under `keep_latest`),
and a write deletes every other entry scored the same way, another run's included. Ranking is by
score alone. Under `keep_latest = false` at most K such files remain in the directory. Setup warns
a training handle when the directory already holds checkpoints this rule may delete.

**Only the current `metric` and `mode` are ever ranked.** An entry scored under another pair
(written by an earlier run, or by this run before a resume changed the selection) is not
comparable, so under either scope and either `keep_latest` it is never ranked and never deleted.
Setup warns when the entries in the handle's retention scope are mixed this way, and so does a
`:best` lookup. An entry with no recorded metric predates the field and counts as matching.
Under either scope a name that would land on a file this run does not own gets `-run-<id>`
inserted before `.jld2`, so no write ever replaces another run's file by name.
"""
mutable struct TopKCheckpointer
    k::Int
    metric::Symbol
    mode::Symbol
    dir::Union{String, Nothing}
    name::Any
    manifest::Any
    # This run's id, and the checkpoint a branched run was resumed from. Assigned at setup.
    run::Union{String, Nothing}
    parent::Union{String, Nothing}
    # Whether the run's newest record is retained whatever it scored. `false` keeps the top K only.
    keep_latest::Bool
    # Which entries retention ranks and prunes: `:run` (this run's) or `:dir` (the directory's).
    scope::Symbol
end

function TopKCheckpointer(;
        k::Int = 3, metric::Symbol = :val_loss, mode::Symbol = :min,
        dir::Union{String, Nothing} = nothing, name = nothing, keep_latest::Bool = true,
        scope::Symbol = :run
    )
    check_scope(scope)
    return TopKCheckpointer(k, metric, mode, dir, name, nothing, nothing, nothing, keep_latest, scope)
end

check_scope(scope) = scope in (:run, :dir) || error(
    "ReactantNitro: the checkpointer's `scope` is $(repr(scope)); it must be `:run` (retention \
     within this run's own checkpoints, the default) or `:dir` (the top K across every run in the \
     directory)."
)

# A run id: eight characters from the OS entropy source. Not the global rng, which setup has
# seeded, so two runs with one seed would draw the same id and be one run to the manifest.
new_run_id() = Random.randstring(Random.RandomDevice(), 8)

"""
    save_checkpoint!(ckpt, epoch, metrics, snapshot) -> Bool or nothing

The checkpointer decides internally whether an epoch warrants a write, and returns `false` when it
declined; any other return, `nothing` included, means the epoch was written. `train!` reads it to
decide whether the final `stop_reason` rewrite has a record to rewrite. [`TopKCheckpointer`](@ref)
returns `true` or `false`, and declines only under `keep_latest = false`, for an epoch that cannot
enter the run's top K. Writes go through [`with_io_retry`](@ref) to a temporary path renamed into
place, and rotation deletes the displaced file only after the new one is durable. `e` is
deliberately not an argument: the filename comes from the checkpointer's bound `name`, so the one
object holding device leaves by design never reaches the function that serializes.
"""
function save_checkpoint! end
save_checkpoint!(::Nothing, epoch, metrics, snapshot) = nothing

function save_checkpoint!(ckpt::TopKCheckpointer, epoch, metrics, snapshot)
    dir = checkpoint_dir(ckpt)
    mkpath(dir)
    # The selection metric drives retention, which is control flow, so the readback is validated.
    # A metric that is only logged is not, and that asymmetry is deliberate.
    score = haskey(metrics, ckpt.metric) ?
        check_control_readback(
            getproperty(metrics, ckpt.metric), "the checkpointer",
            ckpt.metric
        ) : nothing
    (score === nothing && !isempty(metrics)) && error(
        """
        ReactantNitro: the checkpointer selects on `$(ckpt.metric)`, which `metrics` does not emit.
        The available metrics are $(keys(metrics)).
        `:val_loss` is not a magic name: it is what the framework substitutes when an experiment
        defines no `metrics`. An experiment defining its own names one of its own keys."""
    )
    (score === nothing && !ckpt.keep_latest) && error(
        """
        ReactantNitro: this checkpointer has `keep_latest = false` and epoch $(epoch) has no
        `$(ckpt.metric)` score (the run has no `val` split), so it could never be kept.
        `keep_latest = false` retains only the top K by score. Keep the default
        `keep_latest = true`, which retains the newest record whatever it scored, or add a `val`
        split."""
    )

    # This run's entries and everyone else's. The one-entry-per-epoch rule and the displaced file
    # are applied to this run's alone. Under `scope = :run` so is retention, and another run's
    # entries go back into the manifest exactly as read, their files never touched; under
    # `scope = :dir` retention ranks the whole directory and may prune another run's entries.
    prior = read_manifest(dir)
    own = ManifestEntry[e for e in prior if isequal(e.run, ckpt.run)]
    others = ManifestEntry[e for e in prior if !isequal(e.run, ckpt.run)]
    pool_others = ckpt.scope === :dir ? others : ManifestEntry[]
    # Without the newest protected, an epoch outside the top K would be written only for the
    # rotation below to delete it. Decided before the record is built or the name hook is called.
    ckpt.keep_latest || enters_top_k(vcat(pool_others, own), ckpt, Int(epoch), score) ||
        return false

    record = CheckpointRecord(
        FORMAT_VERSION, framework_version(),
        snapshot.ps, snapshot.st, snapshot.opt_state,
        snapshot.flat_permutation, snapshot.step, snapshot.epoch,
        snapshot.seed, snapshot.config, snapshot.devices, metrics,
        snapshot.run_id, snapshot.run_url,
        snapshot.logger_state, snapshot.logger_type,
        snapshot.anchor_checksum, snapshot.stop_reason,
        # `nothing` for a run that named no preset, which is most of them.
        hasproperty(snapshot, :preset) ? snapshot.preset : nothing
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
    ReactantNitro.enters_top_k(pool, ckpt, epoch, score) -> Bool

Whether a write of `epoch` scoring `score` would be among the top K, which is the write condition
under `keep_latest = false`. `pool` is the entries retention ranks: the run's own under
`scope = :run`, the whole directory's under `scope = :dir`. The ones ranked are those
[`retained`](@ref) ranks (scored, under this checkpointer's `metric` and `mode`), less this run's
entry for `epoch` itself, since the write replaces it. That exclusion is what lets the final
`stop_reason` rewrite of a top-K epoch through, and turns away the rewrite of an epoch that was
never written. With fewer than K rivals the answer is yes; otherwise the score must beat the K-th
best strictly, because a tie sorts after the entries already held and would be rotated out.
"""
function enters_top_k(pool, ckpt::TopKCheckpointer, epoch::Int, score)
    rivals = [
        e.score for e in pool
            if !(isequal(e.run, ckpt.run) && e.epoch == epoch) && e.score !== nothing &&
            same_selection(e, ckpt)
    ]
    length(rivals) < ckpt.k && return true
    kth = sort(rivals; rev = ckpt.mode === :max)[ckpt.k]
    return ckpt.mode === :max ? score > kth : score < kth
end

"""
    ReactantNitro.owned_name(name, dir, run, own, theirs) -> String

The name this run writes under: `name` itself when that file is absent or already this run's, and
otherwise `name` with `-run-<id>` inserted before `.jld2` (and a counter after it, should that be
taken too). Two runs of one seed and config produce the same default name, epoch for epoch, and a
write under it would replace the other run's record in place. An unlisted file is not this run's
either, so it is not overwritten.
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

The retention rule over the entries retention decides for (`pool`: one run's under
`scope = :run`, the directory's under `scope = :dir`): the top K by the selection metric, in union
with this run's newest whatever it scored, or the top K alone under `keep_latest = false`. The
newest is the highest epoch under `:run` and the most recent write under `:dir`, where the pool
holds other runs' epochs too. An entry with no score (a run with no `val` split) is never among the
top K and is retained only by being the newest, which keeps a train-only run resumable; under
`keep_latest = false` it is not retained. Only entries scored under the checkpointer's own
`metric` and `mode` are ranked, since a score is comparable only with scores of the same metric and
direction, and every entry written under another pair is retained, under either scope: selection
never prunes what it cannot rank. An entry with no recorded metric predates the field and is
ranked as before.
"""
function retained(pool, ckpt::TopKCheckpointer)
    mine = [e for e in pool if isequal(e.run, ckpt.run)]
    newest = if !ckpt.keep_latest || isempty(mine)
        String[]
    elseif ckpt.scope === :dir
        [mine[argmax([e.written for e in mine])].file]
    else
        [mine[argmax([e.epoch for e in mine])].file]
    end
    scored = [e for e in pool if e.score !== nothing && same_selection(e, ckpt)]
    by_score = sort(scored; by = e -> e.score, rev = ckpt.mode === :max)
    other = [e.file for e in pool if !same_selection(e, ckpt)]
    return Set(vcat([e.file for e in first(by_score, max(ckpt.k, 0))], newest, other))
end

# Whether an entry's score was produced by this checkpointer's selection. An entry with no recorded
# metric was written before the manifest carried one and is taken to match.
same_selection(e, ckpt::TopKCheckpointer) =
    e.metric === nothing || (e.metric === ckpt.metric && e.mode === ckpt.mode)

"""
    ReactantNitro.write_record(path, record) -> nothing

The write: through [`with_io_retry`](@ref), to a temporary path, `rename`d into place
**atomically**. A reader, including `resume = :auto` in another process, therefore sees either the
old file or the whole new one, never a half-written record.
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
This is the cheap question; go to a record with [`checkpoint_info`](@ref) only for what the
manifest does not carry. Returns an empty vector for a directory no run has written to.

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
                field(e, :run, nothing), field(e, :metric, nothing), field(e, :mode, nothing),
                Float64(field(e, :written, 0.0)), field(e, :parent, nothing),
            )
        )
            for e in raw
    ]
end

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

What a checkpoint says about itself, with none of its weights: epoch, step, preset, run, how it
stopped, and the validation metrics current when it was written.

```julia
i = checkpoint_info("runs/MyExp/epoch-0012-step-116520-mae=0.0253034.jld2")
i.epoch, i.step, i.preset, i.metrics.mae, i.run_id
```

This is the one thing to open a checkpoint with when asking a question about it: a record is one
JLD2 object four of whose fields are the parameter tree, so there is no cheaper way to read
`epoch`, and hand-rolled readers have printed the whole tree by accident. Read it in a process with
ReactantNitro loaded, or JLD2 reconstructs a look-alike type. For a whole directory use
[`read_manifest`](@ref) rather than calling this in a loop.
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
        metrics = r.metrics,
    )
end

# ── The filename ────────────────────────────────────────────────────────────────────

"""
    checkpoint_filename(e; epoch, step, metric, score) -> String

The checkpoint filename, and a user hook.

    epoch-0006-step-4500-val_loss=0.0416831.jld2
    epoch-0006-step-4500.jld2                          # a run with no `val` split

The epoch stays first and zero-padded so lexicographic order is training order; the step is
unpadded. `score === nothing` omits the segment. The score needs no sanitizing because it went
through [`check_control_readback`](@ref) and is a finite `Float64`; the metric name is a user
`Symbol` and is sanitized. A method must accept `kwargs...` so a later keyword addition is not a
`MethodError`:

```julia
ReactantNitro.checkpoint_filename(e::MyExp; epoch, score, kwargs...) =
    "e\$(lpad(epoch, 4, '0'))_\$(round(something(score, 0.0); digits = 4)).jld2"
```

The framework calls this through [`TopKCheckpointer`](@ref)'s `name`, bound at setup, so a revised
method takes effect on the next checkpoint of a live run.
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

Every character outside `[A-Za-z0-9_.]` becomes `_`, truncated to 40, and an empty result becomes
`metric`: a metric key is a user-supplied `Symbol` and a filename is not.
"""
function sanitize_metric_name(metric)
    s = replace(String(metric), r"[^A-Za-z0-9_.]" => "_")
    length(s) > 40 && (s = s[1:nextind(s, 0, 40)])
    return isempty(s) ? "metric" : s
end

"""
    ReactantNitro.checkpoint_name(ckpt, epoch, step, score) -> String

Resolve [`TopKCheckpointer`](@ref)'s `name` for one write and check the result: a non-empty `.jld2`
basename with no path separator and no `..`. A name that omits the epoch is allowed ("best only")
and collides across epochs, which is why the write deletes the file it displaced.
"""
function checkpoint_name(ckpt::TopKCheckpointer, epoch, step, score)
    ckpt.name === nothing && error("ReactantNitro: this `TopKCheckpointer` has no `name`. Setup \
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

Where a [`TopKCheckpointer`](@ref) writes. [`run_dir`](@ref) is the one path concept in the
framework, so the checkpointer adopts it at setup unless it was constructed with an explicit `dir`,
which pins it.
"""
function checkpoint_dir(ckpt::TopKCheckpointer)
    ckpt.dir === nothing && error("ReactantNitro: this `TopKCheckpointer` has no directory. Setup \
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

**Dispatches on the checkpointer, not on the path.** A checkpointer with a non-file backend, which
is a ten-line implementation, cannot implement a path-dispatched loader at all.
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

# Field-shape migrations: the `tunables` to `devices` rename, and the addition of `preset`.
# JLD2 reconstructs an old record as a ReconstructedMutable with the on-disk field names, and the
# rename is done here. A function because `checkpoint_info` reads the same records. FORMAT_VERSION
# stays 1, since the semantics did not change.
function _migrate_record(raw, path = "<record>")
    raw isa CheckpointRecord && return raw
    if hasproperty(raw, :tunables) || !hasproperty(raw, :preset)
        return CheckpointRecord(
            raw.format_version, raw.framework_version,
            raw.ps, raw.st, raw.opt_state, raw.flat_permutation,
            raw.step, raw.epoch, raw.seed, raw.config,
            hasproperty(raw, :devices) ? raw.devices : raw.tunables,
            raw.metrics, raw.run_id, raw.run_url, raw.logger_state,
            raw.logger_type, raw.anchor_checksum, raw.stop_reason,
            hasproperty(raw, :preset) ? raw.preset : nothing
        )
    end
    return error("ReactantNitro: `$path` does not hold a `CheckpointRecord`; it holds a \
        `$(typeof(raw))`.")
end

function load_checkpoint(ckpt::TopKCheckpointer, path)
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
    ReactantNitro.check_resume_compatible(record, e, layout; kwargs...) -> nothing

The compatibility check behind `resume = :auto` and `weights = path`, one route from a record
to usable state.

  * Config: the `GraphConst` fields are compared and a mismatch refuses with a diff. `Host` fields
    are excluded, since raising `max_epochs` on resume is normal; `Device` fields are excluded
    because they cannot affect the compiled program, and a change is visible in `devices`.
  * `step` is restored, not derived, so every stateless schedule resumes exactly.
  * The flat permutation is compared leaf by leaf, shapes included.
  * The decay anchor is verified by checksum: the model is rebuilt under the restored seed, `w0`
    is captured, and each anchored group's checksum must match the record, or the run would
    silently regularize toward a different point.

Accepted limitations: an adaptive schedule restarts its adaptation, the loader's sampling order is
not restored, and an anchored run with a nondeterministic init cannot be resumed at all, which is
the trade for not storing the anchor arrays.
"""
function check_resume_compatible(
        record, e, layout; anchors = nothing, logger = nothing,
        path = "the checkpoint"
    )
    check_config_compatible(record, e, path)
    layout === nothing || check_permutation_compatible(record, layout, path)
    anchors === nothing || check_anchors_compatible(record, layout, anchors, path)
    check_logger_compatible(record, logger, path)
    return nothing
end

"""
    ReactantNitro.check_config_compatible(record, e, path) -> nothing

The config half of the resume check, over `GraphConst` fields only, the same set the compile cache
is keyed on: a changed one means the resumed run would train a different compiled program. `Device`
fields are recorded separately in `devices`; `Host` fields are the normal thing to change on
resume. `derive` has merged before this runs, so a derived field is compared as whichever kind it
was declared.
"""
function check_config_compatible(record, e, path)
    now = graphconst_fields(e)
    was = record.config
    was isa NamedTuple || error("ReactantNitro: the checkpoint at `$path` has no usable `config` \
        field; it holds a `$(typeof(was))`.")
    added = setdiff(keys(now), keys(was))
    removed = setdiff(keys(was), keys(now))
    changed = [
        k for k in intersect(keys(now), keys(was))
            if !isequal(getproperty(now, k), getproperty(was, k))
    ]
    (isempty(added) && isempty(removed) && isempty(changed)) && return nothing
    diff = String[]
    for k in changed
        push!(diff, "  $k: $(repr(getproperty(was, k))) -> $(repr(getproperty(now, k)))")
    end
    for k in added
        push!(diff, "  $k: (absent in the checkpoint) -> $(repr(getproperty(now, k)))")
    end
    for k in removed
        push!(diff, "  $k: $(repr(getproperty(was, k))) -> (absent now)")
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
        Start a fresh run, or pass `resume = false` to train from scratch in this `run_dir`."""
    )
end

"""
    ReactantNitro.graphconst_fields(e) -> NamedTuple

The fields the compile cache is keyed on and the resume check compares: everything that is neither
`Device` nor `Host`. One definition, used by both, so the two cannot drift into disagreeing about
what a configuration is.
"""
function graphconst_fields(e)
    T = typeof(e)
    df, hf = device_fields(T), host_fields(T)
    ks = Tuple(f for f in fieldnames(T) if !(f in df) && !(f in hf))
    return NamedTuple{ks}(map(f -> getfield(e, f), ks))
end

"""
    ReactantNitro.device_values(e) -> NamedTuple

The record's `devices` field, read back to **host** values. Derived and adaptive `Device` values
live here so that a change is visible after the fact, which is what lets the resume check exclude
them from the config comparison without losing the information.
"""
function device_values(e)
    T = typeof(e)
    df = device_fields(T)
    return NamedTuple{df}(map(f -> to_host(getfield(e, f)), df))
end

"""
    ReactantNitro.check_permutation_compatible(record, layout, path) -> nothing

The permutation half of the resume check. A different permutation scrambles `opt_state` against
`ps`, so it is a refusal. A [`LeafRow`](@ref) carries keypath, group, offset, length and size, so
shapes are validated rather than only counts, and the error names the leaf that moved.
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

The optimizer half of `restore_optimizer = true`: the record's `opt_state` replaces the one setup
just built, so it must be the same optimizer over the same groups. Compared per group: the rule
(its type name, and each member's for an `OptimiserChain`) and the shape of every state leaf. A
resume needs no such check because it restores a state the run itself wrote under the same config;
a warm start with the optimizer state is a new run, with a new `optimizer` hook free to have
changed. Refuses, naming the group and both sides, rather than stepping moments meant for another
rule. The permutation check has already run, so the groups hold the same leaves.
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

The decay-anchor checksum per group, `nothing` for a `:zero` group. SHA256 over the host bytes,
which both sides agree on, and which outlives the process and the Julia version as `Base.hash`
does not.
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

The anchor half of the resume check, and what makes storing a checksum instead of the anchor
arrays sound: the fresh `w0` capture is checksummed against the record, a match proves it is the
same tensor, and a mismatch refuses, naming the group and both checksums. Continuing would
silently regularize toward a different point with a plausible loss curve.
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

Resuming into a different logger backend refuses with both type names rather than handing one
backend's state to another. A record with no `logger_state` checks nothing. `weights_only = true`
permits no logger at all: a weights-only restore continues no run, and without this every export
of a checkpoint written with a logger was impossible.
"""
function check_logger_compatible(record, logger, path; weights_only::Bool = false)
    record.logger_state === nothing && return nothing
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

The weights-only restore's derived `Device` values, from the record rather than recomputed, since
an evaluation process may not have the training data `derive` reads. Only fields the experiment
still declares as `Device` are taken.
"""
function restored_devices(record, e)
    record.devices isa NamedTuple || return (;)
    df = device_fields(typeof(e))
    ks = Tuple(k for k in keys(record.devices) if k in df)
    return NamedTuple{ks}(map(k -> getproperty(record.devices, k), ks))
end

"""
    ReactantNitro.check_checkpointer(ckpt, collection, routing) -> nothing

The checkpointer's setup-time checks: `k`, `mode`, and, when the experiment defines no `metrics`
(so the only key is `:val_loss`), the selection metric. Otherwise the keys exist only once the hook
has run, and [`save_checkpoint!`](@ref) raises at the first checkpoint. Selecting on a metric in a
run with no `val` split is not an error: the newest checkpoint is retained regardless. The one
exception is `keep_latest = false`, which retains nothing unscored, so a handle with a `train`
split and no `val` split is refused: every epoch it wrote would go unkept.
"""
check_checkpointer(::Nothing, collection, routing) = nothing

function check_checkpointer(ckpt::TopKCheckpointer, collection, routing)
    ckpt.mode in (:min, :max) || error("ReactantNitro: the checkpointer's `mode` is \
        $(repr(ckpt.mode)); it must be `:min` or `:max`.")
    check_scope(ckpt.scope)
    ckpt.k >= 1 || error("ReactantNitro: the checkpointer's `k` is $(ckpt.k); it must be at least \
        1. Pass `checkpointer = nothing` to disable checkpointing, which is the documented \
        opt-out.")
    (
        haskey(collection, :val) && routing !== nothing && routing.metrics === nothing &&
            ckpt.metric !== :val_loss
    ) && error(
        """
        ReactantNitro: the checkpointer selects on `$(ckpt.metric)`, and this experiment defines no
        `metrics` method, so the only metric the framework will emit is `:val_loss`.
        Either select on `:val_loss`, or define `metrics` emitting `$(ckpt.metric)`."""
    )
    (!ckpt.keep_latest && haskey(collection, :train) && !haskey(collection, :val)) && error(
        """
        ReactantNitro: the checkpointer has `keep_latest = false`, and this run has a `train` split
        and no `val` split, so no epoch gets a `$(ckpt.metric)` score.
        `keep_latest = false` retains only the top K by score, so nothing this run trained would
        ever be saved. Keep the default `keep_latest = true`, which retains the newest record
        whatever it scored, or add a `val` split."""
    )
    return nothing
end

"""
    ReactantNitro.warn_keep_latest(ckpt, resume) -> nothing

Said once, at setup of a handle that trains, when its checkpointer has `keep_latest = false`: what
the run gives up by not keeping its newest record. Everything it says follows from the retention
rule, and each is the kind of thing found out only after a crash or a resume, so it is said up
front. A `resume = :auto` in the same call is named, since that resume has already happened by the
rule: it continued from the newest RETAINED record and will retrain the epochs after it.
"""
warn_keep_latest(ckpt, resume) = nothing
function warn_keep_latest(ckpt::TopKCheckpointer, resume)
    ckpt.keep_latest && return nothing
    auto = resume === :auto ? " This handle was built with `resume = :auto`, so where the \
        directory held a record it continues from the run's newest RETAINED record, one of its top \
        $(ckpt.k), and retrains every epoch after it; under `keep_latest = false` that is not \
        where the earlier process stopped." : ""
    @warn "ReactantNitro: the checkpointer has `keep_latest = false`, so this run keeps only its \
           top $(ckpt.k) by `$(ckpt.metric)` and does not write an epoch that cannot enter them. \
           The state training ends in is not saved unless the last epoch is a top-$(ckpt.k) epoch. \
           `resume = :auto` and `weights = :latest => :latest` resolve to the most recent \
           improvement, not to where training stopped, and a crash loses every epoch since that \
           improvement.$(auto)"
    return nothing
end

"""
    ReactantNitro.warn_retention_scope(ckpt) -> nothing

Said at setup of a handle that trains, over the entries its retention will decide for: this run's
own under `scope = :run`, the directory's under `scope = :dir`. Two warnings, each about something
otherwise found out only by a checkpoint that is not where it was expected: the entries are mixed
across metrics ([`warn_mixed_metrics`](@ref); those are kept), and, under `scope = :dir`, the
directory already holds other runs' checkpoints that this run deletes as it beats them.
"""
warn_retention_scope(ckpt) = nothing
function warn_retention_scope(ckpt::TopKCheckpointer)
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
           under $(_metric_label(ckpt.metric, ckpt.mode)). The directory keeps only its top \
           $(ckpt.k) across every run$(ckpt.keep_latest ? " plus this run's newest" : ""), so they \
           may be deleted as this run beats them. Checkpoints under other metrics are kept. \
           `runs(\"$(dir)\")` lists what is there."
    return nothing
end

# The directory a lookup reads: the checkpointer's pinned `dir`, else the `run_dir` it was given.
lookup_dir(ckpt::TopKCheckpointer, run_dir) = ckpt.dir === nothing ? String(run_dir) : ckpt.dir

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

The newest record of one run in `run_dir`, through the manifest: what `:latest => :latest` and
`resume = :auto` resolve to. Newest, not best, since resuming from the best epoch would discard
every epoch after it. Newest RETAINED: under `keep_latest = false` the epochs after the run's most
recent improvement were never written, so "latest" is that improvement, one of the top K, and not
where training stopped. `run` is a run id, `nothing` for the unnamed run of a manifest written
before runs were recorded, or `:recent` (the default), the run that wrote the directory's most
recent entry. `nothing` when there is nothing to resume from; the raising form, which says what
the directory holds instead, is [`resolve_checkpoint`](@ref).
"""
find_latest(::Nothing, run_dir; run = :recent) = nothing

function find_latest(ckpt::TopKCheckpointer, run_dir; run = :recent)
    dir = lookup_dir(ckpt, run_dir)
    entries = run_scope(dir, run).entries
    isempty(entries) && return nothing
    newest = entries[argmax([e.epoch for e in entries])]
    path = joinpath(dir, newest.file)
    return isfile(path) ? path : nothing
end

"""
    ReactantNitro.selected_checkpoint(ckpt, run_dir; run = :recent) -> NamedTuple or `nothing`

Which checkpoint a run would hand you, `(; path, epoch, metric, score, run)`: the best of that run's
entries by the checkpointer's CURRENT `metric` and `mode`, answered from the manifest alone. `run`
is as for [`find_latest`](@ref), the most recent run by default; `train!` passes its own. Only
entries scored under the same metric and mode are ranked, since another pair's scores are not
comparable; an entry with no recorded metric predates the field and is ranked. `nothing` with no
checkpointer, no directory, no scored entry, or a winning file that is gone. This is the quiet
form `show` and `train!` use; `:latest => :best` goes through [`resolve_checkpoint`](@ref), which
raises and warns about mixed metrics.
"""
selected_checkpoint(::Nothing, run_dir; run = :recent) = nothing

function selected_checkpoint(ckpt::TopKCheckpointer, run_dir; run = :recent)
    dir = lookup_dir(ckpt, run_dir)
    scope = run_scope(dir, run)
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
own_selected_checkpoint(ckpt::TopKCheckpointer, run_dir) =
    selected_checkpoint(ckpt, run_dir; run = ckpt.run)
own_selected_checkpoint(ckpt, run_dir) = selected_checkpoint(ckpt, run_dir)

# The announcement for a lookup that named no run, in a directory holding more than one: which
# run it read, since the answer is otherwise indistinguishable from the other runs' files.
announce_recent_run(ckpt, run_dir, what) = nothing
function announce_recent_run(ckpt::TopKCheckpointer, run_dir, what)
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
`TopKCheckpointer`, and for a run continuing the unnamed run of a manifest written before runs were
recorded, which [`runs`](@ref) lists as `"legacy"`. Not [`run_id`](@ref), which is the logger's.
"""
checkpoint_run(nitro::Nitro) =
    nitro.checkpointer isa TopKCheckpointer ? nitro.checkpointer.run : nothing

# ── Mixed metrics ────────────────────────────────────────────────────────────────────

# The entries scored under a (metric, mode) other than the checkpointer's, grouped with counts.
# A legacy entry, with no recorded metric, is taken to match and is never "mixed".
function other_selections(entries, ckpt::TopKCheckpointer)
    counts = Dict{Tuple{Symbol, Union{Symbol, Nothing}}, Int}()
    for e in entries
        (e.score === nothing || same_selection(e, ckpt)) && continue
        counts[(e.metric, e.mode)] = get(counts, (e.metric, e.mode), 0) + 1
    end
    return sort!(collect(counts); by = p -> -last(p))
end

"""
    ReactantNitro.warn_mixed_metrics(entries, ckpt, what) -> nothing

The `@warn` for a set of entries some of which were scored under a metric or mode other than the
checkpointer's: which pairs, how many of each, and that they are kept. Selection only ever ranks
the current pair, so those checkpoints are invisible to `:best` and to retention, and a reader who
expected them to compete should hear so. Said at setup of a training handle, over the entries its
retention decides for, and by every `:best` lookup, over the entries it considered. The manifest
records a metric's NAME: a metric whose definition changed under the same name is not detectable
here.
"""
function warn_mixed_metrics(entries, ckpt::TopKCheckpointer, what)
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

"""
    ReactantNitro.resolve_checkpoint(ckpt, run_dir, run, which; what) -> (; path, epoch, run)

The one resolver behind every `run => checkpoint` source of `weights` and `resume`, answered from
the manifest of `lookup_dir(ckpt, run_dir)`:

  * `run` is `:latest` (the directory's most recent run: the run of the entry written last, legacy
    entries counting as oldest), `:all` (every run, only with `which = :best`), or a run id
    (`"legacy"` for the unnamed run of a manifest written before runs were recorded).
  * `which` is `:latest` (the run's newest retained record) or `:best` (the best of the run's
    entries by the checkpointer's CURRENT `metric` and `mode`; across every run for `:all`).

Raises rather than returning `nothing`: a caller who asked for the best weights and got fresh ones
would not find out until the numbers were wrong. Every miss names the directory, the runs it holds
and `runs(run_dir)`. A `:best` over entries scored under another metric warns that they were not
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
    if !(ckpt isa TopKCheckpointer)
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
    dir = lookup_dir(ckpt, run_dir)
    entries = isdir(dir) ? read_manifest(dir) : ManifestEntry[]
    pool, label = if run === :all
        entries, "every run in `$(dir)`"
    elseif run === :latest
        isempty(entries) && error("ReactantNitro: $(what) found no checkpoint: $(holds_text(dir, entries)).")
        r = announce_recent_run(ckpt, dir, what)
        ManifestEntry[e for e in entries if isequal(e.run, r)], run_label(r)
    else
        r = run_from_key(run)
        mine = ManifestEntry[e for e in entries if isequal(e.run, r)]
        isempty(mine) && error(
            "ReactantNitro: $(what) names run `$(run)`, which has no checkpoints here: \
             $(holds_text(dir, entries))."
        )
        mine, run_label(r)
    end
    isempty(pool) && error("ReactantNitro: $(what) found no checkpoint: $(holds_text(dir, entries)).")
    pick = if which === :latest
        pool[argmax([e.epoch for e in pool])]
    else
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
                 TopKCheckpointer(; metric = ..., mode = ...)`), or pass the checkpoint's path. \
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

Which run a resume from `path` continues. The newest record of a run in this checkpointer's
directory continues that run: its id is adopted, so the resumed run's checkpoints join it and its
retention rotates them. Anything else, an earlier record (a best epoch, say) or a file this
directory's manifest does not list, starts a NEW run, a branch, with `parent = path`: continuing
the original id from an earlier epoch would write that run's later epochs again and rotate the
originals away. `from` is `nothing` for a continuation and otherwise says what was branched from,
`(; run, epoch, newest)` for a listed record or `(;)` for an unlisted file.
"""
function resumed_run(ckpt::TopKCheckpointer, path)
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
