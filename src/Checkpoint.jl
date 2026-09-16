# Checkpoint.jl
#
# Checkpointing: the record, the top-K checkpointer, the manifest, and the resume compatibility
# checks.
#
# A "CHECKPOINT" IS FULL TRAINING STATE; "WEIGHTS" ARE PARAMETERS ALONE. An API that says checkpoint
# and returns something unresumable has picked the wrong word, so `save_optimizer_state = true` by
# default on EVERY checkpoint including top-K, not only a dedicated resume checkpoint. Whatever
# produces parameters alone for export or serving is named for weights.

"""
    CheckpointRecord

The single authority on what a checkpoint holds. The record on disk is the `snapshot` handed to
[`save_checkpoint!`](@ref) plus the two framework-stamped fields at the top.

| Field | Required | Why it is in the record |
| --- | --- | --- |
| `format_version` | yes | JLD2 files outlive framework versions |
| `framework_version` | yes | Diagnosing a restore that misbehaves |
| `ps` | yes | The parameters, as a tree |
| `st` | yes | Layer state, including running statistics |
| `opt_state` | yes | Opaque; carries moments **and** bias correction. **Stored as host values** |
| `flat_permutation` | yes | A different permutation scrambles `opt_state` against `ps` |
| `step` | yes | Optimizer steps; restores schedules exactly |
| `epoch` | yes | Loop position and reporting |
| `seed` | yes | A rebuild must reproduce the same initialization |
| `config` | yes | Flattened, for the resume compatibility check |
| `devices` | yes | Derived and adaptive `Device` values, so a change is visible after the fact |
| `metrics` | yes | The epoch's validation metrics, for top-K bookkeeping |
| `run_id`, `run_url` | optional | Informational; traces a checkpoint back to its experiment |
| `logger_state` | optional | Machine-readable resumption state, opaque and backend-specific |
| `logger_type` | optional | So resuming into a different backend refuses |
| `anchor_checksum` | optional | Per anchored group; **the arrays themselves are not stored** |
| `stop_reason` | optional | `:completed`, `:early_stop`, or `:error` |

**`opt_state` is stored as host values and re-normalized to device residency on restore.**
Serializing `ConcretePJRTNumber` leaves would tie a checkpoint to a device configuration and JLD2
has no reason to round-trip them faithfully, so the flow is uniform with exactly one normalization
point per path: **write host, read host, normalize on the way in.** Skipping the restore-path
normalization reacquires the frozen-step-counter bug in full, on the default path.

A metric-comparison gate can read its curves out of this record's `metrics` field, which is why such
a gate needs no logger and the framework ships none.
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

# ── Showing a record NEVER shows the weights ─────────────────────────────────────────
#
# The default struct `show` prints every field recursively, and four of them (`ps`, `st`,
# `opt_state`, `flat_permutation`) are the whole parameter tree. A 115 MB checkpoint therefore
# `display`s as tens of millions of characters: a 64x64 toy weight alone prints 90,080. In a REPL
# that is a wasted screen; in an agent session, where the REPL's output is the transcript, it is the
# context window. Printing a record to find one small field is the way that cost gets paid by
# accident: the parameter dump lands in the transcript and the field being looked for is buried in
# it.
#
# So the summary is the show, and the arrays are named rather than printed. This is not politeness:
# there is no way to ask for `epoch` without materializing the record, so the safe rendering has to
# be the DEFAULT rendering, or the trap stays one keystroke away. `checkpoint_info` is the supported
# way to ask, and the last line says so, because someone reading this output is mid-question.
#
# Nothing here can throw on a partly-migrated record: every field is read through `_shown`, which
# falls back to the type name for anything it does not recognize.
_shown(x::Union{Nothing, Symbol, AbstractString, Real}) = repr(x)
_shown(x::NamedTuple) = isempty(x) ? "(;)" :
    "(" * join(("$k = " * _shown(v) for (k, v) in pairs(x)), ", ") * ")"
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
    print(io, "  the weights reach a run through `Nitro(e; checkpoint = path)`; for the metadata \
               alone use `checkpoint_info(path)`")
    return nothing
end

"""
    TopKCheckpointer(; k = 3, metric = :val_loss, mode = :min, dir = nothing, name = nothing)

The default checkpointer, writing into [`run_dir`](@ref).

It is constructible for an arbitrary experiment **only because the default `metrics` guarantees
`:val_loss` exists**; an earlier draft left `metric` with no default and therefore had a
batteries-included default that could not be built. **If a user's own `metrics` does not emit the
configured metric, that is a setup error naming the metric and the ones actually emitted**, checked
before training rather than at the first checkpoint.

**Retain a `latest` alongside top-K**, as a retention *rule* rather than a second file: never rotate
out the newest, whatever it scored. On disk this is the union, K+1 files when the newest is not among
the best and exactly K when it is. Resuming from the *best* checkpoint is not resuming from where you
were.

**Do not symlink `latest` into the top-K set.** Rotation deletes the target when a better checkpoint
arrives, so the link dangles through entirely normal operation with no user action, and checkpoint
directories get copied between paths where `tar`/`rsync`/`scp` handle symlinks inconsistently.

Write a small manifest (file, epoch, metric, `stop_reason`) alongside; top-K bookkeeping needs it
anyway and it lets resume find the newest without reading every file.

**Passing `checkpointer = nothing` disables checkpointing** and is the documented opt-out, with its
`::Nothing` no-op methods. TopK is the default rather than `nothing` because `resume = :auto` only
works if something wrote a `latest`.

"Every N epochs", "best only", and "write to object storage" are ten-line implementations. The
serialization format stays concrete (JLD2); swapping it means replacing save and load together,
which is the only safe granularity.

**`name` is the checkpoint filename, and `nothing` means adopt the experiment's**
[`checkpoint_filename`](@ref) at setup, exactly as `dir` adopts [`run_dir`](@ref). Passing one pins
it and setup never overwrites it. It is called with keywords, `name(; epoch, step, metric, score)`.
"""
mutable struct TopKCheckpointer
    k::Int
    metric::Symbol
    mode::Symbol
    dir::Union{String, Nothing}
    name::Any
    manifest::Any
end

TopKCheckpointer(;
    k::Int = 3, metric::Symbol = :val_loss, mode::Symbol = :min,
    dir::Union{String, Nothing} = nothing, name = nothing
) =
    TopKCheckpointer(k, metric, mode, dir, name, nothing)

"""
    save_checkpoint!(ckpt, epoch, metrics, snapshot) -> nothing

**The checkpointer decides internally whether an epoch warrants a write**, so policy and action
collapse into one hook.

Writes go through [`ReactantNitro.with_io_retry`](@ref), to a temporary path renamed into place
atomically, and rotation deletes the displaced file only after the new one is durably in place.

**Four arguments, and `e` is deliberately not among them.** The filename comes from the
checkpointer's `name`, bound to the experiment's [`checkpoint_filename`](@ref) at setup, rather than
from an experiment passed in here: the record's host-value discipline exists to keep device-resident
leaves away from the function that serializes, and the experiment is the one object in a run that
holds them by design.
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

    path = joinpath(dir, checkpoint_name(ckpt, epoch, snapshot.step, score))
    write_record(path, record)

    entry = ManifestEntry((basename(path), Int(epoch), score, snapshot.stop_reason))
    prior = read_manifest(dir)
    # ONE ENTRY PER EPOCH, AND ONE PER FILE. The first is what top-K bookkeeping ranks over. The
    # second is because `file` is the only identity a checkpoint has, so an entry naming a file
    # this write just overwrote describes bytes that are gone. Under the default name the two
    # conditions coincide, since the name carries the epoch; they come apart for a `name` that does
    # not, which is exactly the "best only" ten-line implementation, and without the second
    # condition that one grows a manifest entry per epoch all naming the same file.
    entries = ManifestEntry[
        e for e in prior
            if e.epoch != Int(epoch) && e.file != entry.file
    ]
    push!(entries, entry)
    # REPLACING AN EPOCH DELETES THE FILE IT DISPLACED. The filter above drops the old same-epoch
    # entry, so a write whose name differs from the previous write of that epoch leaves a file that
    # is in no entry, which makes it invisible to the rotation below and to every later one:
    # permanent, not merely untidy. The default name cannot reach this, since all four of its
    # inputs are identical between the per-epoch write and the final rewrite at the end of a run,
    # but `name` is a hook and Revise can change one mid-run.
    displaced = String[
        e.file for e in prior
            if e.epoch == Int(epoch) && e.file != entry.file
    ]
    keep = retained(entries, ckpt)
    # THE MANIFEST IS WRITTEN BEFORE THE ROTATION DELETES ANYTHING. A rotation that deletes
    # first and then fails leaves a run with fewer checkpoints than its retention policy promises;
    # writing the manifest first leaves at worst a file on disk that the manifest does not list,
    # which the next rotation cleans up and which nothing reads.
    write_manifest(dir, [e for e in entries if e.file in keep])
    for f in Iterators.flatten((displaced, (e.file for e in entries if !(e.file in keep))))
        p = joinpath(dir, f)
        isfile(p) && with_io_retry(() -> rm(p))
    end
    return nothing
end

"""
    ReactantNitro.retained(entries, ckpt) -> Set{String}

The retention rule: **the top K by the selection metric, in union with the newest, whatever it
scored.** On disk that is K+1 files when the newest is not among the best and exactly K when it is.

**`latest` is a rule, not a second file and not a symlink.** A symlink into the top-K set dangles
the moment a better checkpoint rotates its target away, which is entirely normal operation, and
checkpoint directories get copied between paths where `tar`, `rsync`, and `scp` handle symlinks
inconsistently.

An entry with no score, which is what a run with no `val` split produces, is never among the top K
and is retained only by being the newest. That keeps a train-only run resumable without inventing an
ordering over metrics it never computed.
"""
function retained(entries, ckpt::TopKCheckpointer)
    newest = entries[argmax([e.epoch for e in entries])]
    scored = [e for e in entries if e.score !== nothing]
    by_score = sort(scored; by = e -> e.score, rev = ckpt.mode === :max)
    return Set(vcat([e.file for e in first(by_score, max(ckpt.k, 0))], newest.file))
end

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

# The manifest: file, epoch, metric, and `stop_reason`, so top-K bookkeeping and `resume = :auto`
# both work without reading every record. JLD2 for the same reason the records use it: the
# framework fixes one serialization format, and swapping it means replacing save and load together.
#
# THE ENTRY TYPE IS EXPLICIT, and the two optional fields are `Union`s rather than whatever the first
# entry happened to hold. A manifest built from bare NamedTuples types itself off its first row, so a
# run whose first epoch had no `stop_reason` produced a `Vector` of entries with `stop_reason` typed
# `Nothing`, and the final rewrite that carries the outcome could not be pushed into it. That failed
# loudly here; on a file format it would be the kind of thing that fails on the tenth epoch of a
# long run.
const ManifestEntry = @NamedTuple{
    file::String, epoch::Int, score::Union{Float64, Nothing},
    stop_reason::Union{Symbol, Nothing},
}

manifest_path(dir) = joinpath(dir, "manifest.jld2")

"""
    read_manifest(dir) -> Vector

**Which checkpoints a run directory holds, without opening one of them.** Each entry
carries `file` (a basename, not a path), `epoch`, `score` (the validation metric the retention rule
ranked that checkpoint by, or `nothing` for a run with no `val` split) and `stop_reason` (`nothing`
until a run records how it ended).

```julia
for e in sort(read_manifest("runs/MyExp"); by = e -> e.epoch)
    println(e.epoch, "  ", e.score, "  ", e.file)
end
```

**This is the cheap question, and it is the one worth asking first.** The manifest is a single small
file the checkpointer rewrites next to every record, so ranking a run's checkpoints, or finding the
newest one to resume from, costs one read instead of deserializing a parameter tree per file in the
directory. [`checkpoint_info`](@ref) is the other half of the pair: go to a record only for the
fields the manifest does not carry, and not in a loop over a directory.

Returns an empty vector when `dir` has no manifest. That is a directory no run has written to yet,
which is an answer rather than an error, and it is what makes `resume = :auto` safe in a fresh one.
"""
function read_manifest(dir)
    p = manifest_path(dir)
    isfile(p) || return ManifestEntry[]
    raw = with_io_retry() do
        JLD2.load(p, "entries")
    end
    return ManifestEntry[
        ManifestEntry((String(e.file), Int(e.epoch), e.score, e.stop_reason))
            for e in raw
    ]
end

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

**What a checkpoint says about itself, with none of its weights.** Which epoch, which step, which
preset, which run, how it stopped, and the validation metrics that were current when it was written.

```julia
i = checkpoint_info("runs/MyExp/epoch-0012-step-116520-mae=0.0253034.jld2")
i.epoch, i.step, i.preset, i.metrics.mae, i.run_id
```

**This is the ONLY thing you should open a checkpoint with when you are asking a question about it**,
and it exists because the alternative kept being reinvented, badly. A record is one JLD2 entry
holding one object, four of whose fields are the whole parameter tree, so there is no way to read
`epoch` without materializing all of it, and every session that tried built its own reader:
`JLD2.Group` (does not exist), `JLD2.names` (not public, wrong method), `keys` on the record (no
method), then `getfield` on a record read in a process without ReactantNitro loaded, where JLD2 hands
back a `ReconstructedMutable` whose fields answer to `getproperty` and NOT to `getfield`. Then the
one that actually costs something: `println` the record, and the parameter tree goes to stdout.
[`CheckpointRecord`](@ref)'s `show` now refuses that, and this function means nobody has to get
close to it.

**Read it in a process that has ReactantNitro loaded.** That is not a nicety either: the workspace
needs the type, or JLD2 reconstructs a look-alike and the field access rules change under you. This
function is also where the two field-shape migrations live, so an older record answers the same
questions as a new one.

**For a whole run directory, do not call this in a loop.** [`read_manifest`](@ref) answers "which
checkpoints are here, at which epochs, with which scores" from the run's own small manifest and opens
no record at all; go to a record only for the fields the manifest does not carry.
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

The checkpoint filename, and a user hook: define a method for your experiment type to name
checkpoints differently.

    epoch-0006-step-4500-val_loss=0.0416831.jld2
    epoch-0006-step-4500.jld2                          # a run with no `val` split
    epoch-0043-step-32250-mae=-1.23457e-07.jld2

Reading a run directory should answer "which checkpoint is good" without opening JLD2. The score is
in the manifest and in every record too, but both need deserializing to rank three files.

**The epoch stays first and stays zero-padded**, so lexicographic order is training order. The step
is unpadded, because the epoch already supplies the ordering and a width would be a promise that
breaks above its own digit count.

**`score === nothing` omits the segment rather than filling it**, which is what a run with no `val`
split produces; the retention rule keeps such a checkpoint by being newest. No metric segment
means no metric was computed, and a token such as `no_score` would reserve a name a real metric
could take.

**The value needs no sanitizing, and that is a consequence rather than a convention.** The
selection metric goes through [`check_control_readback`](@ref) before anything uses it, so `score`
is either `nothing` or a FINITE `Float64`, and `%g` on a finite double is always within
`[0-9.eE+-]`. A
`NaN` score does not produce an odd filename; it stops the run. The metric NAME is sanitized,
because it is a user-supplied `Symbol` and a filename is not.

**A method must accept `kwargs...`** unless it wants a later keyword addition to be a `MethodError`:

```julia
ReactantNitro.checkpoint_filename(e::MyExp; epoch, score, kwargs...) =
    "e\$(lpad(epoch, 4, '0'))_\$(round(something(score, 0.0); digits = 4)).jld2"
```

The framework calls this through [`TopKCheckpointer`](@ref)'s `name`, which is bound to the
experiment at setup, so `save_checkpoint!` keeps its four arguments and never sees `e`: the
host-value discipline exists to keep device-resident leaves away from the function that serializes,
and the experiment is the one object in a run that holds them by design. A consequence worth having:
the binding dispatches at WRITE time, so a revised method takes effect on the next checkpoint of a
live run, unlike a revised `checkpointer` accessor, which is called once at setup.
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

A metric key is a user-supplied `Symbol` and a filename is not, so every character outside
`[A-Za-z0-9_.]` becomes `_` and the result is truncated to 40. That covers glob metacharacters,
Windows-hostile characters, and a path separator smuggled in through `Symbol("a/b")`, without
needing to enumerate them. An empty result becomes `metric`, so a pathological `Symbol("")` still
names a file.
"""
function sanitize_metric_name(metric)
    s = replace(String(metric), r"[^A-Za-z0-9_.]" => "_")
    length(s) > 40 && (s = s[1:nextind(s, 0, 40)])
    return isempty(s) ? "metric" : s
end

"""
    ReactantNitro.checkpoint_name(ckpt, epoch, step, score) -> String

Resolve [`TopKCheckpointer`](@ref)'s `name` for one write, and CHECK WHAT IT RETURNED. A hook is a
user method, so the return is validated at the one moment the failure is cheap rather than after a
`joinpath` has quietly aimed the write somewhere else.

The rules are the minimum that keeps the manifest the single identity a checkpoint has: a non-empty
`.jld2` basename, no path separator, and no `..`. A name that omits the epoch is NOT refused, since
that is a legitimate choice ("best only" writes one file), but it does collide across epochs, which
is why the write deletes the file it displaced.
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
# it cannot drift from the release it shipped in.
const FORMAT_VERSION = 1

function framework_version()
    p = joinpath(pkgdir(ReactantNitro), "Project.toml")
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

# JLD2 warns "saved type CheckpointRecord is missing field <f> in workspace type; reconstructing"
# whenever a record predates a field ADDITION, which means every checkpoint written before `preset`
# existed. The warning is JLD2's, fires inside `load` before any of our code runs, and reads like
# breakage: it says "reconstructing" at a user who is resuming a run that is about to work.
#
# The migration below is EXPLICIT and tested, so the warning reports a problem that is not happening,
# and an old checkpoint is not a rare artifact: a long run's mid-training checkpoints are.
# Suppressed NARROWLY, matching both the phrase and the type name, so any other JLD2 warning still
# reaches the user. Same call as suppressing "Replacing docs" on a deliberate docstring merge: a
# warning about an operation the framework performs on purpose is noise, and noise trains people to
# ignore warnings.
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

# Migration (the Tunable -> Device rename). Records written before the rename carry a
# `tunables` field. JLD2 reconstructs a struct from the ON-DISK field names when they do not match
# the workspace type, so an old record arrives as a ReconstructedMutable holding `tunables` rather
# than as a `CheckpointRecord`, and the rename is done here, field by field. FORMAT_VERSION stays 1:
# nothing about the semantics of the stored data changed, only the field's name.
#
# A FUNCTION rather than a block inside `load_checkpoint`, because `checkpoint_info` reads the same
# records and two copies of a migration is how one of them gets a third migration and the other does
# not. `path` is for the error only.
function _migrate_record(raw, path = "<record>")
    raw isa CheckpointRecord && return raw
    if hasproperty(raw, :tunables) || !hasproperty(raw, :preset)
        # TWO migrations now, both field-shape only, so FORMAT_VERSION stays 1: nothing about the
        # semantics of the stored data changed. The `tunables` to `devices` rename, and the
        # ADDITION of `preset`, which an older record does not have and which reads back as
        # `nothing` rather than refusing.
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

**`resume = :auto` is the default**, and that is safe only because the compatibility check lives in
the framework rather than in a harness.

  * **Config.** The flattened config is in the record, so on mismatch **refuse with a diff of the
    offending fields**. Identical config continues, changed config errors, neither silently does the
    wrong thing. `Host` fields are excluded, since changing `max_epochs` on resume is the normal
    case.
  * **Derived values are excluded from the config comparison, and the three cases differ.** A derived
    **`Device`** may change freely: it is recomputed on resume by design, cannot affect the compiled
    program, and the change is recorded in `devices` so it is visible after the fact. A derived
    **`GraphConst`** field changing is an **error**, because it bakes as a trace-time constant and
    changes the cache key, so continuing would silently train a different program than the one being
    resumed.
  * **`step` is stored, not derived.** Deriving it as `epoch * opt_steps_per_epoch` is silently wrong
    whenever steps-per-epoch changed. Restoring `step` restores every stateless schedule exactly,
    since those are pure functions of `(step, total)`. The optimizer's bias correction is **not**
    restored from `step`; it lives in `opt_state`.
  * **Validate shapes, not just parameter counts**, and validate the flat permutation.
  * **Verify the decay anchor by checksum and refuse on mismatch.** Step 6 rebuilds the model under
    the restored `seed` and step 7 captures `w0`; for each group whose `decay_anchor` is not `:zero`,
    checksum the freshly captured anchor against the record. All match, proceed: the fresh capture is
    provably the same tensor the original run anchored to. Any mismatch, **refuse**, naming the
    group, both checksums, and the likely cause. Do not warn and continue: continuing means silently
    regularizing toward a *different* point, which produces a plausible loss curve with no error. The
    checksum is over the **host** value of each anchored group's flat slice, consistent with the
    record storing host values throughout.

The accepted limitations, all documented rather than hidden: **adaptive schedules do not resume
exactly** (a schedule closing over training state restarts its adaptation; carry that state in a
`Device` if it matters); **data order is the loader's** and the framework does not restore a
sampler's RNG; and **an anchored run with a nondeterministic init cannot be resumed at all**, which
is the deliberate trade for not storing the anchor arrays, since that would roughly double the record
for a `:w0`-anchored backbone.

The `checkpoint = path` construction reuses this same check, so there is one route from a checkpoint
to usable state rather than two that can drift.
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

The config half of the resume check: **identical config continues, changed config errors, and
neither silently does the wrong thing.**

The comparison is over **`GraphConst` fields only**, which is the same set the compile cache is
keyed on, and that is not a coincidence: a `GraphConst` field bakes as a trace-time constant, so a
changed one means the resumed run would train a **different compiled program** than the one being
resumed.

`Device` fields are excluded because they are traced inputs that cannot affect the graph, and they
are recorded separately in `devices` so a change stays visible after the fact. `Host` fields are
excluded because raising `max_epochs` on resume is the normal case. A **derived** `GraphConst` field
is in the comparison and a derived `Device` is not, which is exactly the three-case rule above and
needs no special handling here: `derive` merges into the struct before this runs, so each derived
value is already whichever kind it was declared as.
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

The permutation half of the resume check, and its shape half in the same check. A different
permutation scrambles `opt_state` against `ps`, so it is a refusal rather than a warning.

**This validates shapes rather than only parameter counts**, which is a separate requirement: a
[`LeafRow`](@ref) carries the leaf's `keypath`, `group`, `offset`, `len`, and `size`, so a model
whose leaf count matches but whose shapes moved differs here. Storing keypaths is also what makes
the error name the leaf that moved instead of reporting that two integer vectors differ.
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
    ReactantNitro.anchor_checksums(anchors) -> NTuple or nothing

The decay-anchor checksum, one per group, `nothing` for a group anchored at `:zero`.

**SHA256 over the host bytes**, for two reasons. The checksum must be over the same bytes on both
sides, and the record stores host values throughout, so the host value is the one both sides can
agree on. And it outlives the process and the Julia version, which `Base.hash` does not promise: a
checksum that drifted on a Julia upgrade would make resume refuse while blaming `build_model` for
being nondeterministic, which is the least debuggable failure this check could produce.
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

The anchor half of the resume check, and the reason a checkpoint can store a **checksum** instead
of the anchor arrays without that being a silent downgrade.

Step 6 rebuilds the model under the **restored** seed and step 7 captures `w0`; for each group whose
`decay_anchor` is not `:zero`, the freshly captured anchor is checksummed against the record. All
match: proceed, because the fresh capture is provably the same tensor the original run anchored to,
so nothing was lost by not storing it. Any mismatch: **refuse**, naming the group, both checksums,
and the likely cause.

**Do not warn and continue.** Continuing means silently regularizing toward a *different* point
than the run being resumed, which is the whole hazard of anchored decay: a plausible loss curve and
no error. The limitation this accepts is recorded above, that a run whose anchored parameters come
from a nondeterministic source cannot be resumed at all, and the two escapes from it.
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

The logger type check. The record stores the logger's type name alongside its state, so resuming
into a **different** backend refuses with both names rather than handing one backend's state to
another's logger, which would either raise somewhere deep in that backend or, worse, quietly start a
fresh experiment.

A record with no `logger_state` reattaches nothing and therefore checks nothing: a ten-line file
logger that defines neither hook keeps working across a resume, which is the whole point of making
the pair self-describing.

**`weights_only = true` permits no logger at all**, and the type check still applies to one that is
passed. A weights-only restore continues no run: there is no history to extend, so declining the
logger drops nothing, and `reattach!(::Nothing, state)` is already a no-op.

That distinction is not a nicety. This check ran on both restore paths, so **every export of a
checkpoint written with a logger was impossible**: `logger = nothing` was refused, and the alternative
was to pass the real backend and have setup reattach and re-log the config into the finished TRAINING
experiment, from an export, needing credentials a detached export may not have. The error's suggested
escape, `resume = false`, is not consulted on the `checkpoint = path` branch, so there was none. Found
by two independent model ports.
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

The weights-only restore: the derived `Device` values **from the record**, rather than recomputed
by `derive`. That difference is deliberate and is the one place the two restore paths diverge: an
evaluation process may not have the training data `derive` reads at all, so recomputing is not an
option there, while a resume recomputes.

Only fields the experiment still declares as `Device` are taken, so a record written before a field
was removed does not resurrect it as a merge error.
"""
function restored_devices(record, e)
    record.devices isa NamedTuple || return (;)
    df = device_fields(typeof(e))
    ks = Tuple(k for k in keys(record.devices) if k in df)
    return NamedTuple{ks}(map(k -> getproperty(record.devices, k), ks))
end

"""
    ReactantNitro.check_checkpointer(ckpt, collection, routing) -> nothing

The checkpointer's setup-time checks, the same shape as the early-stopping ones and partial for the
same reason: the metric key set is knowable before the run **only** when the experiment defines no
`metrics` method, where the framework's substitution fixes it at `(:val_loss,)`. Otherwise the keys
exist only once the hook has run, and [`save_checkpoint!`](@ref) raises at the first checkpoint
naming what was actually emitted.

`k`, `mode`, and the absence of a `val` split are checkable unconditionally. A checkpointer selecting
on a metric in a run with no validation is not an error, though: the retention rule keeps the
newest checkpoint whatever it scored, so a train-only run stays resumable and simply never ranks.
"""
check_checkpointer(::Nothing, collection, routing) = nothing

function check_checkpointer(ckpt::TopKCheckpointer, collection, routing)
    ckpt.mode in (:min, :max) || error("ReactantNitro: the checkpointer's `mode` is \
        $(repr(ckpt.mode)); it must be `:min` or `:max`.")
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
    return nothing
end

"""
    ReactantNitro.find_latest(ckpt, run_dir) -> path or nothing

`resume = :auto`'s lookup: the **newest** record in `run_dir`, found through the manifest rather than
by reading every file, which is one of the two jobs the manifest exists for.

**Newest, not best.** The retention rule keeps the newest checkpoint whatever it scored, for
exactly this: resuming from the *best* checkpoint is not resuming from where you were, and a run
that resumed from its best epoch would silently discard every epoch after it.

Returns `nothing` when there is nothing to resume from, without raising, or a first run in a fresh
directory could not start under the default `resume = :auto`.
"""
find_latest(::Nothing, run_dir) = nothing

function find_latest(ckpt::TopKCheckpointer, run_dir)
    dir = ckpt.dir === nothing ? run_dir : ckpt.dir
    isdir(dir) || return nothing
    entries = read_manifest(dir)
    isempty(entries) && return nothing
    newest = entries[argmax([e.epoch for e in entries])]
    path = joinpath(dir, newest.file)
    return isfile(path) ? path : nothing
end
