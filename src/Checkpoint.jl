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
    print(io, "  the weights reach a run through `Nitro(e; checkpoint = path)`; for the metadata \
               alone use `checkpoint_info(path)`")
    return nothing
end

"""
    TopKCheckpointer(; k = 3, metric = :val_loss, mode = :min, dir = nothing, name = nothing)

The default checkpointer, writing into [`run_dir`](@ref). Constructible for an arbitrary
experiment because the default `metrics` guarantees `:val_loss`; a user `metrics` that does not
emit the configured metric is a setup error.

The retention rule is the top K by `metric` in union with the newest, whatever it scored: K+1
files when the newest is not among the best, K when it is. `latest` is a rule, not a symlink,
since a symlink into the top-K set dangles the moment rotation deletes its target. A small
manifest (file, epoch, metric, `stop_reason`) is written alongside so resume finds the newest
without reading every file. `checkpointer = nothing` disables checkpointing. `name` is the
filename hook, `name(; epoch, step, metric, score)`; `nothing` adopts the experiment's
[`checkpoint_filename`](@ref) at setup, as `dir` adopts `run_dir`.
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

The checkpointer decides internally whether an epoch warrants a write. Writes go through
[`with_io_retry`](@ref) to a temporary path renamed into place, and rotation deletes the displaced
file only after the new one is durable. `e` is deliberately not an argument: the filename comes
from the checkpointer's bound `name`, so the one object holding device leaves by design never
reaches the function that serializes.
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
    # One entry per epoch and one per file: `file` is a checkpoint's only identity, and a `name`
    # that omits the epoch ("best only") would otherwise grow an entry per epoch naming one file.
    entries = ManifestEntry[
        e for e in prior
            if e.epoch != Int(epoch) && e.file != entry.file
    ]
    push!(entries, entry)
    # A write whose name differs from the previous write of that epoch would leave a file no entry
    # names, invisible to every rotation; `name` is a hook and Revise can change it mid-run.
    displaced = String[
        e.file for e in prior
            if e.epoch == Int(epoch) && e.file != entry.file
    ]
    keep = retained(entries, ckpt)
    # The manifest is written before the rotation deletes anything, so a failure leaves at worst
    # an unlisted file rather than fewer checkpoints than the policy promises.
    write_manifest(dir, [e for e in entries if e.file in keep])
    for f in Iterators.flatten((displaced, (e.file for e in entries if !(e.file in keep))))
        p = joinpath(dir, f)
        isfile(p) && with_io_retry(() -> rm(p))
    end
    return nothing
end

"""
    ReactantNitro.retained(entries, ckpt) -> Set{String}

The retention rule: the top K by the selection metric, in union with the newest whatever it
scored. An entry with no score (a run with no `val` split) is never among the top K and is retained
only by being the newest, which keeps a train-only run resumable.
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

# The manifest: file, epoch, metric and `stop_reason`, so top-K bookkeeping and `resume = :auto`
# work without reading every record. The entry type is explicit, with `Union` fields: a manifest
# built from bare NamedTuples typed itself off its first row, and the final rewrite carrying a
# `stop_reason` could not be pushed into it.
const ManifestEntry = @NamedTuple{
    file::String, epoch::Int, score::Union{Float64, Nothing},
    stop_reason::Union{Symbol, Nothing},
}

manifest_path(dir) = joinpath(dir, "manifest.jld2")

"""
    read_manifest(dir) -> Vector

Which checkpoints a run directory holds, without opening one. Each entry carries `file` (a
basename), `epoch`, `score` (the retention metric, or `nothing` for a run with no `val` split) and
`stop_reason` (`nothing` until the run records how it ended). This is the cheap question; go to a
record with [`checkpoint_info`](@ref) only for what the manifest does not carry. Returns an empty
vector for a directory no run has written to.

```julia
for e in sort(read_manifest("runs/MyExp"); by = e -> e.epoch)
    println(e.epoch, "  ", e.score, "  ", e.file)
end
```
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

The compatibility check behind `resume = :auto` and `checkpoint = path`, one route from a record
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
run with no `val` split is not an error: the newest checkpoint is retained regardless.
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

`resume = :auto`'s lookup: the newest record in `run_dir`, through the manifest. Newest, not
best, since resuming from the best epoch would discard every epoch after it. `nothing` when there
is nothing to resume from.
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

"""
    ReactantNitro.selected_checkpoint(ckpt, run_dir) -> NamedTuple or `nothing`

Which checkpoint the run would hand you, `(; path, epoch, metric, score)`: the best by the
checkpointer's `metric` and `mode`, answered from the manifest alone. `nothing` with no
checkpointer, no directory, no scored entry, or a winning file that is gone.
"""
selected_checkpoint(::Nothing, run_dir) = nothing

function selected_checkpoint(ckpt::TopKCheckpointer, run_dir)
    dir = ckpt.dir === nothing ? run_dir : ckpt.dir
    isdir(dir) || return nothing
    scored = [e for e in read_manifest(dir) if e.score !== nothing]
    isempty(scored) && return nothing
    pick = ckpt.mode === :max ? argmax : argmin
    best = scored[pick([e.score for e in scored])]
    path = joinpath(dir, best.file)
    isfile(path) || return nothing
    return (; path, epoch = best.epoch, metric = ckpt.metric, score = best.score)
end
