# Data.jl
#
# The data-source contract's checks, the prefetch pipeline, and the pad-then-slice path for a
# short final eval batch. Nothing here is exported. The prefetch puts the H2D transfer inside
# itself, so it is realized against a resolved run rather than against what `build_data` returned.

"""
    ReactantNitro.check_data_source(source, name::Symbol) -> Int

The data-source requirements, checked at setup with an error naming the source. Returns the batch
count. `length` is required and counts batches; it fixes the schedule horizon. The source must
also be restartable, which cannot be checked here: setup draws one batch for the schema, so a
one-shot source silently loses it, and [`check_epoch_length`](@ref) catches that after the first
epoch.
"""
function check_data_source(source, name::Symbol)
    applicable(length, source) || error(
        """
        ReactantNitro: the `$name` split is a `$(typeof(source))`, which does not support `length`.
        A data source must be iterable, yield concrete NamedTuples of host arrays, and support
        `length`, which counts BATCHES. It is read once at setup to fix the schedule
        horizon `total = max_epochs * div(steps_per_epoch, accum)`, so a source without it cannot be
        trained against on any schedule.
        `MLUtils.DataLoader` and a plain `Vector` of batches both qualify."""
    )
    n = length(source)
    n > 0 || error("ReactantNitro: the `$name` split reports `length == $n`. A split must yield at \
                    least one batch.")
    return n
end

"""
    ReactantNitro.check_source_options(source, name::Symbol, cfg) -> nothing

A hook for options a source type can carry that the resolved pipeline cannot honour, checked at
setup for every split. The default does nothing; `ReactantNitroMLUtilsExt` implements it for
`MLUtils.DataLoader`. `cfg` is the resolved [`prefetch_config`](@ref), because an option living
inside `Base.iterate` is live on the single-producer path and dead on the fan-out, which never
iterates. Called for `:inline` splits too, since a training loader keeping its partial final batch
is wrong however it is read.
"""
check_source_options(source, name::Symbol, cfg) = nothing

"""
    ReactantNitro.check_epoch_length(seen::Integer, expected::Integer, name::Symbol;
                                     horizon_dependent::Bool) -> nothing

What only an actual epoch can establish. A one-shot source yields `expected - 1` batches on the
first epoch, since setup consumed one, and the error says so. `length` must be stable across
epochs when a horizon-dependent schedule is in use, because `total` fixes the LR curve; a
variable-length loader with a constant learning rate is fine.
"""
function check_epoch_length(
        seen::Integer, expected::Integer, name::Symbol;
        horizon_dependent::Bool = true
    )
    seen == expected && return nothing
    if seen == expected - 1
        error(
            """
            ReactantNitro: the `$name` split yielded $seen batches but `length` promised $expected,
            short by exactly one. That is the signature of a source that is NOT RESTARTABLE:
            setup draws one batch to learn the batch schema and resolve routing, then discards
            it and the loop iterates from scratch, so a one-shot source such as a
            bare `Channel` loses that first batch permanently.
            Wrap the source so each `iterate` starts a fresh pass."""
        )
    end
    horizon_dependent || return nothing
    error(
        """
        ReactantNitro: the `$name` split yielded $seen batches but `length` promised $expected, and
        a horizon-dependent schedule is configured. `total` was fixed at setup from the promised
        count, so a changing length shifts the whole learning-rate curve.
        A variable-length loader is fine with constant hyperparameters; it is only the combination
        that is rejected."""
    )
end

"""
    ReactantNitro.check_train_divisibility(n_batches::Integer, accum::Integer, split::Symbol)

The setup-time half of the accumulation contract, `length(train) % accum == 0`, in batch units
because batches are the only unit the data contract exposes. It buys an invariant: an accumulation
group never spans an epoch boundary, so the accumulator is empty at every epoch edge and
checkpointing, resume and validation never reason about partial accumulation state.
"""
function check_train_divisibility(n_batches::Integer, accum::Integer, split::Symbol = :train)
    accum >= 1 || error("ReactantNitro: `accum` must be at least 1, and is $accum.")
    n_batches % accum == 0 && return nothing
    lower = n_batches - (n_batches % accum)
    upper = lower + accum
    error(
        """
        ReactantNitro: the `$split` split has $n_batches batches, which is not divisible by
        `accum = $accum`. The nearest batch counts that work are $lower and $upper.
        This is an error rather than a warning because it buys an invariant the rest of the
        framework leans on: an accumulation group never spans an epoch boundary, so the
        accumulator is always empty at an epoch edge and checkpointing, resume, and validation
        never have to reason about a partially accumulated gradient. It also makes the schedule
        horizon `total = max_epochs * div(steps_per_epoch, accum)` an exact division.
        Change the batch size, drop or add samples, or change `accum`."""
    )
end

"""
    ReactantNitro.batch_size_of(batch::NamedTuple, routing) -> Int

The batch size, inferred from the first batch rather than read from a config field, since an
inferred value cannot disagree with the loader; a `batch_size` field on the experiment is ordinary
user config the framework never reads. The batch dimension is last, and every routed field must
agree on it. This says nothing about metric denominators, which each metric reports itself.
"""
function batch_size_of(batch::NamedTuple, routing = nothing)
    ks = routing === nothing ? keys(batch) : routed_fields(routing)
    isempty(ks) && error("ReactantNitro: cannot infer the batch size, because no hook declares any \
                          field of a batch with fields $(keys(batch)).")
    sizes = [(k, size(getproperty(batch, k))) for k in ks]
    trailing = unique(last(sz) for (_, sz) in sizes if !isempty(sz))
    length(trailing) == 1 || error(
        """
        ReactantNitro: the routed batch fields disagree on their last dimension, so the batch size is
        ambiguous: $(join(("`$k` $(sz)" for (k, sz) in sizes), ", ")).
        The framework infers the batch size from the first batch and takes the LAST dimension as the
        batch dimension. Every routed field must agree on it, because the per-batch training
        assertion and the eval pad-and-slice path both depend on that number."""
    )
    return only(trailing)
end

"""
    ReactantNitro.check_train_batch_shape(batch, batch_size, split, idx, routing) -> nothing

The per-batch half of the accumulation contract: `size(leaf)[end] == batch_size` on every
training batch. The training loader is required to drop its partial final batch; the framework can
see a batch it was handed and cannot see samples a loader dropped, so a shortfall from drop-last
(3001 samples at 64 drop 57 every epoch) is reported in the binding report when the source
supports `MLUtils.numobs`, not caught here.
"""
function check_train_batch_shape(
        batch::NamedTuple, batch_size::Integer, split::Symbol,
        idx::Integer, routing = nothing
    )
    ks = routing === nothing ? keys(batch) : routed_fields(routing)
    for k in ks
        sz = size(getproperty(batch, k))
        isempty(sz) && continue
        last(sz) == batch_size && continue
        error(
            """
            ReactantNitro: the `$split` split yielded a batch whose last dimension is $(last(sz)),
            not $batch_size, at batch $idx (field `$k`, size $sz).
            The training loader must DROP its partial final batch (`partial = false` for
            `MLUtils.DataLoader`, `drop_last = true` elsewhere), because the compiled program has a
            fixed shape and training cannot pad: in train mode BatchNorm normalizes over the batch,
            so padded rows change the real rows' outputs and no downstream slice undoes it.
            Eval splits DO pad and slice, and should not drop."""
        )
    end
    return nothing
end

"""
    ReactantNitro.pad_batch(batch, batch_size, routing = nothing) -> (padded, n_real)
    ReactantNitro.slice_outputs(outputs, n_real, batch_size) -> outputs

The eval path: validation, test and inference loaders may yield a short final batch. The framework
pads it to `batch_size`, runs `forward` at one shape, slices the outputs and the routed fields back
to `n_real`, and only then calls `metrics`, which never sees padding.

Exact in eval mode, where each sample's output is independent of the rest of the batch; not
guaranteed bitwise, since two shapes are two compiled programs and XLA's tiling may depend on the
extent. `metrics` compiles twice per split, once at `batch_size` and once at the remainder.

`slice_outputs` asserts `size(leaf)[end] == batch_size` before slicing, against the width passed
in rather than one learned from the leaves, so a model whose batch dimension is not last fails
loudly. The padding rows replicate the last real sample rather than zeros, since an all-zero row
can produce a `NaN` in a normalization that a real row never would.
"""
function pad_batch(batch::NamedTuple, batch_size::Integer, routing = nothing)
    n_real = batch_size_of(batch, routing)
    n_real <= batch_size || error(
        """
        ReactantNitro: an eval batch is $n_real wide, which is MORE than the batch size $batch_size
        the compiled program was built at. The framework pads a SHORT final batch
        up to that width and slices back; it cannot make a wide one fit.
        Every split must use the same batch size as the one the programs were compiled at, which is
        the width of the first batch of `train`, or of whichever split exists."""
    )
    n_real == batch_size && return (batch, n_real)
    ks = keys(batch)
    padded = NamedTuple{ks}(map(k -> _pad_leaf(getproperty(batch, k), n_real, batch_size), ks))
    return (padded, n_real)
end

# Pad one field along its last dimension by replicating the last real sample. A field whose last
# dimension is not `n_real` is not batch-shaped and is left alone.
function _pad_leaf(x::AbstractArray, n_real::Integer, batch_size::Integer)
    N = ndims(x)
    (N == 0 || size(x, N) != n_real) && return x
    out = similar(x, ntuple(d -> d == N ? Int(batch_size) : size(x, d), N))
    copyto!(selectdim(out, N, 1:n_real), x)
    last_real = selectdim(x, N, n_real)
    for j in (n_real + 1):batch_size
        copyto!(selectdim(out, N, j), last_real)
    end
    return out
end
_pad_leaf(x, n_real, batch_size) = x

"""
    ReactantNitro.check_output_batch_dim(outputs, batch_size) -> nothing

The assertion `slice_outputs` makes before it slices: every array leaf's last dimension is the
batch, so a model whose batch dimension is elsewhere fails here rather than returning the wrong
samples with no error. Checked only on the path that slices, so a batch-free output leaf is not
rejected until a split has a short final batch.
"""
function check_output_batch_dim(outputs, batch_size::Integer)
    i = 0
    _each_array_leaf(outputs) do leaf
        i += 1
        sz = size(leaf)
        (!isempty(sz) && last(sz) == batch_size) && return nothing
        error(
            """
            ReactantNitro: output leaf $i has size $sz, whose last dimension is not the batch size
            $batch_size, and a short final batch has to be sliced back to the real sample count.
            The framework slices the LAST dimension, so it asserts that dimension is the batch
            rather than slicing something else and returning the wrong samples with no error.
            A model whose batch dimension is not last cannot use the eval pad-and-slice path;
            transpose the output in `forward`, or give the split a length that is a multiple of the
            batch size so no batch is ever short."""
        )
    end
    return nothing
end

function slice_outputs(outputs, n_real::Integer, batch_size::Integer)
    n_real == batch_size && return outputs
    check_output_batch_dim(outputs, batch_size)
    return slice_last(outputs, n_real)
end

"""
    ReactantNitro.slice_last(tree, n) -> tree

Take the first `n` entries of every array leaf's last dimension, unchecked. Separate from
[`slice_outputs`](@ref) because on the traced metric path the check is host-side and the slice is
inside the traced program.
"""
slice_last(tree, n::Integer) = _map_array_leaves(x -> _slice_leaf(x, n), tree)

function _slice_leaf(x::AbstractArray, n::Integer)
    N = ndims(x)
    (N == 0 || size(x, N) == n) && return x
    # A `copy` rather than a view: a view crossing a program boundary has a different type and
    # costs a recompile, and it would hand the user a view onto a buffer the eval loop frees.
    return copy(selectdim(x, N, 1:n))
end
_slice_leaf(x, n) = x

# Walk the array leaves of an output tree through Functors, with arrays as the leaf type. Under
# trace this runs on `TracedRArray` leaves, where the walk is host-side and free.
_is_array_leaf(x) = x isa AbstractArray && !(eltype(x) <: AbstractArray)

# ── The residency question, which is not the structural one above ───────────────────
#
# `_is_array_leaf` answers "is this one array to slice" (a view is). `_hidden_array` answers "could
# device memory be hiding under this" (a view of a device array: yes). A `SubArray` or a
# `ReshapedArray` over a `ConcretePJRTArray` is not an `AbstractConcreteArray`, and conflating the
# two questions let such wrappers pass both the host converter and the assertion built to catch
# it. Both residency walkers call this one function. `Base.parent` is the whole test.
_hidden_array(x::AbstractArray) = (p = parent(x); p === x ? nothing : p)
_map_array_leaves(f, tree) = Functors.fmap(f, tree; exclude = _is_array_leaf)

function _each_array_leaf(f, tree)
    _map_array_leaves(leaf -> (f(leaf); leaf), tree)
    return nothing
end

# ── The index-addressable trait, which is what makes N producers possible ────────────
#
# A plain sequential iterator cannot be consumed by N tasks, because the expensive work happens
# inside `iterate` and there is no way to ask for batch k. This trait was dropped once in a port,
# leaving one producer, and the model trained four times slower than its reference.

"""
    ReactantNitro.batch_at(source, i::Integer) -> batch
    ReactantNitro.batch_at(source, i::Integer, plan) -> batch

Optional, and half of the index-addressable opt-in: produce batch `i` of the current epoch,
independently of every other `i`, from one of many concurrent tasks. A source that stores its own
plan writes the two-argument form; an immutable declaration of a dataset has
[`begin_epoch!`](@ref) return the plan and writes the three-argument form, with the third argument
left untyped so the capability check can see it.

`i` is a BATCH index in `1:length(source)`, not a sample offset:

```julia
ReactantNitro.batch_at(dl::MyLoader, k::Integer) =
    _build_batch(dl, 1 + (k - 1) * dl.batch_size)     # RIGHT
ReactantNitro.batch_at(dl::MyLoader, k::Integer) = _build_batch(dl, k)   # WRONG, and it TRAINS
```

The wrong version trains an epoch on heavily overlapping windows of the first few hundred samples
with a loss curve that still falls; only [`check_batch_at`](@ref) catches it. The implementation
must be thread-safe with respect to shared state (per-task scratch in a `TaskLocalValue`), a pure
function of `i` and the epoch's plan (re-planning belongs in `begin_epoch!`), and never `nothing`
for a valid `i`. Define this AND [`begin_epoch!`](@ref) to opt in.
"""
function batch_at end

"""
    ReactantNitro.begin_epoch!(source) -> plan

Optional, and the other half of the index-addressable opt-in: re-plan the epoch. Called exactly
once per epoch, before any job is dispatched. Return `nothing` if the source stores its own plan
(the framework then calls the two-argument [`batch_at`](@ref)), or the plan itself if it cannot,
in which case every three-argument `batch_at` call for that epoch receives it.

The fan-out never calls `iterate`, so a source that re-plans in its iteration init and exposes
`batch_at` alone would train every epoch after the first on epoch 1's plan, silently. That is why
both methods are required: a source with only `batch_at` gets one producer and a warning.

```julia
ReactantNitro.begin_epoch!(::MySource) = nothing            # no re-plan needed
ReactantNitro.begin_epoch!(d::MyLoader) = shuffled_index(d.rng, d.n)   # a declaration's plan
Base.iterate(s::MySource) = (begin_epoch!(s); _first_batch(s))         # one implementation
```

`length(source)` is read after this returns. Setup's schema probe goes through `Base.iterate`, so
a source counting its own epochs sees `N + 1` calls across `N` epochs, and a loader that plans
eagerly in `build_data` needs a flag saying the first plan is already drawn.
"""
function begin_epoch! end

"""
    ReactantNitro.epoch_token(source)

An opaque value that must change on every [`begin_epoch!`](@ref), or `nothing` (the default) to
decline the check. The fan-out asserts its one call advanced the source by exactly one epoch,
catching both a missing re-plan and a doubled one, which skips a generation of epoch-seeded
augmentation. A monotone `Int` counter is the intended implementation.
"""
epoch_token(source) = nothing

"""
    ReactantNitro.fanout_capable(source) -> Bool

Whether `source` implements both halves of the index-addressable trait. Either [`batch_at`](@ref)
shape counts. There is deliberately no generic three-argument forwarding method, since one would
make `hasmethod` answer `true` for every source.
"""
fanout_capable(source) =
    applicable(begin_epoch!, source) && (
    applicable(batch_at, source, 1) ||
        hasmethod(batch_at, Tuple{typeof(source), Integer, Any})
)

"""
    ReactantNitro.default_prefetch_workers() -> Int

`Threads.nthreads(:default)`, the non-interactive pool size (`N` under `julia -t N,1`). Named
because `-t` is thereby a data-throughput setting, and the resolved value is logged as a
hyperparameter rather than left to be inferred.
"""
default_prefetch_workers() = max(1, Threads.nthreads(:default))

const DEFAULT_DEVICE_BATCHES = 1

"""
    ReactantNitro.PrefetchIterator(source; workers = default_prefetch_workers(),
                                   device_batches = 1, host_batches = 2 * workers, ordered = true)

Prefetch: `workers` producer tasks build host batches, one transfer task copies them to the device,
and a `Channel` of `device_batches` `(host, device)` pairs feeds the loop.

The framework wraps every split in one of these at these defaults (see [`auto_prefetch`](@ref)), so
a user normally never writes it. Construct it only to change the defaults; [`NoPrefetch`](@ref)
declines entirely.

  * **`workers` is the knob that matters.** Buffering is lookahead, not concurrency: a producer
    slower than the device starves it at any `device_batches`. Fanning out needs [`batch_at`](@ref)
    and [`begin_epoch!`](@ref) on the source; without them this falls back to one producer and
    setup warns.
  * **`device_batches`** bounds batches staged on the device. Resident device memory is
    `(device_batches + 2) x batch` (the channel, the transfer task's hand, the consumer's hand),
    hence the default of 1.
  * **`host_batches`** bounds batches existing on the host at once, across the workers' hands, the
    hand-off channel and the reorder buffer. It is a credit window at the coordinator and may not be
    below `workers`, or ordered delivery can deadlock waiting on a batch never handed out.
  * **`ordered = true`** emits in source order through a reorder buffer, so a fixed seed reproduces
    bitwise at any `workers`. `ordered = false` never waits on a straggler, at the cost that the
    epoch's accumulation groups repartition and bitwise reproducibility goes; every sample is still
    seen exactly once.
  * **The transfer stays on one task.** Device memory cannot be bounded by a semaphore: XLA runs
    asynchronously and the executable holds its inputs, so there is no "consumer is done" moment to
    release on. Buffers are freed explicitly, never left to the GC.

Batching, shuffling and splitting are not shipped; `MLUtils.jl` covers those.

**This is a declaration.** `build_data` runs before the batch routing and device placement exist, so
the wrapper carries only the source and its settings, and the training loop builds the pipeline.
`Base.iterate` is therefore a passthrough: iterating one by hand yields the source's batches with no
task and no channel, which makes a producer leak impossible and keeps `render` and
`predict(nitro, loader)` unaffected by the auto-wrap. Eval splits stream through
[`eval_stream`](@ref), which pads a short final batch inside the producer.
"""
struct PrefetchIterator{S}
    source::S
    device_batches::Int
    host_batches::Int
    workers::Int
    ordered::Bool

    function PrefetchIterator(
            source::S;
            workers::Integer = default_prefetch_workers(),
            device_batches::Integer = DEFAULT_DEVICE_BATCHES,
            host_batches::Integer = 2 * workers,
            ordered::Bool = true
        ) where {S}
        device_batches >= 1 || error(
            """
            ReactantNitro: `PrefetchIterator` device_batches is $device_batches; it must be \
            at least 1. It is how many batches sit on the DEVICE staged ahead, and resident device \
            memory is
            `(device_batches + 2) x batch`, which is why the default is 1.
            ZERO IS NOT HOW YOU TURN PREFETCH OFF. Running the host data path inline on the
            training task is an intentionally suboptimal choice, so it is not reachable by setting a
            number: it needs a marker a reviewer will question. Write `NoPrefetch(source)` if that is
            genuinely what you want."""
        )
        workers >= 1 || error("ReactantNitro: `PrefetchIterator` workers is $workers; it must be at \
            least 1. `workers = 1` is the single-producer path. To run the data path inline on the \
            training task instead, wrap the source in `NoPrefetch`.")
        # Not merely a sanity bound: ordered delivery holds finished batches until the one it is
        # waiting for arrives, and with fewer credits than workers that batch may never be handed
        # out, which is a deadlock rather than a stall.
        host_batches >= workers || error(
            """
            ReactantNitro: `PrefetchIterator` host_batches is $host_batches with $workers workers; it
            must be at least `workers`. It is the ceiling on batches existing on the HOST at once,
            across the producers' hands, the hand-off channel, and the reorder buffer together, and a
            window narrower than the producer count can leave the batch ordered delivery is waiting
            for outside it, which deadlocks the epoch rather than slowing it.
            To hold fewer batches, lower `workers` too."""
        )
        return new{S}(source, Int(device_batches), Int(host_batches), Int(workers), ordered)
    end
end

Base.length(p::PrefetchIterator) = length(p.source)
Base.iterate(p::PrefetchIterator, state...) = iterate(p.source, state...)
Base.eltype(::Type{PrefetchIterator{S}}) where {S} = eltype(S)

"""
    ReactantNitro.NoPrefetch(source)

The only way to decline prefetch. A marker rather than a number, so a split running its host data
path inline on the training task is visible to a reviewer; a `device_batches = 0` knob would read as
ordinary tuning, which is how one model ran inline for weeks unnoticed. `iterate` is a passthrough,
as `PrefetchIterator`'s is.
"""
struct NoPrefetch{S}
    source::S
end

Base.length(n::NoPrefetch) = length(n.source)
Base.iterate(n::NoPrefetch, state...) = iterate(n.source, state...)
Base.eltype(::Type{NoPrefetch{S}}) where {S} = eltype(S)

"""
    ReactantNitro.prefetch_source(x)
    ReactantNitro.prefetch_device_batches(x) -> Int
    ReactantNitro.prefetch_host_batches(x) -> Int
    ReactantNitro.prefetch_workers(x) -> Int

Unwrap a split, and read its settings. `prefetch_device_batches` returns `0` for anything that is
not a [`PrefetchIterator`](@ref), meaning "this split does not stream"; that `0` is an internal
dispatch result, never a user-settable value. Setup validates a wrapped source through
`prefetch_source`, as itself.
"""
prefetch_source(p::PrefetchIterator) = p.source
prefetch_source(n::NoPrefetch) = n.source
prefetch_source(x) = x
prefetch_device_batches(p::PrefetchIterator) = p.device_batches
prefetch_device_batches(x) = 0
prefetch_host_batches(p::PrefetchIterator) = p.host_batches
prefetch_host_batches(x) = 0
prefetch_workers(p::PrefetchIterator) = p.workers
prefetch_workers(x) = 0
prefetch_ordered(p::PrefetchIterator) = p.ordered
# `true` for everything else: the inline and single-producer paths walk the source's own `iterate`.
prefetch_ordered(x) = true

"""
    ReactantNitro.auto_prefetch(collection) -> collection

Wrap every split in a [`PrefetchIterator`](@ref) at the framework's defaults, unless it is already
a `PrefetchIterator` or a [`NoPrefetch`](@ref). The default has to be declined rather than
requested, because a model once ran its whole data path inline for weeks with nothing saying so.
Eval splits are included: a validation step is forward-only, so its host-to-device ratio is worse
than training's, not better. Called at setup after `derive`, the schema probe and the contract
checks, so all three see what `build_data` returned.
"""
auto_prefetch(collection::NamedTuple) =
    NamedTuple{keys(collection)}(map(_auto_wrap, values(collection)))

_auto_wrap(split::PrefetchIterator) = split
_auto_wrap(split::NoPrefetch) = split
_auto_wrap(split) = PrefetchIterator(split)

"""
    ReactantNitro.prefetch_config(split) -> (; device_batches, host_batches, workers, ordered, path)

The resolved prefetch settings for a split, which may differ from the requested ones. `path` is:

  * `:fanout`: `workers` producers over [`batch_at`](@ref), delivered in the source's order.
  * `:fanout_unordered`: the same with `ordered = false`, delivered as the workers finish.
  * `:single`: one producer over the source's own `iterate`, because `workers == 1` was asked for.
  * `:materialized`: one producer over a `Vector` of already-built batches; nothing to spread.
  * `:single_no_trait`: one producer because the source lacks the trait. The setup warning's case.
  * `:inline`: no stream at all, a [`NoPrefetch`](@ref) split.
"""
function prefetch_config(split)
    d = prefetch_device_batches(split)
    h = prefetch_host_batches(split)
    o = prefetch_ordered(split)
    d == 0 && return (;
        device_batches = 0, host_batches = 0, workers = 0, ordered = o, path = :inline,
    )
    w = prefetch_workers(split)
    src = prefetch_source(split)
    cfg(workers, path) = (;
        device_batches = d, host_batches = h, workers = workers, ordered = o, path = path,
    )
    if fanout_capable(src)
        w == 1 && return cfg(1, :single)
        return cfg(w, o ? :fanout : :fanout_unordered)
    end
    w == 1 && return cfg(1, :single)
    materialized_source(src) && return cfg(1, :materialized)
    return cfg(1, :single_no_trait)
end

"""
    ReactantNitro.materialized_source(x) -> Bool

Whether a source's batches already exist, so `workers > 1` would buy nothing. `Vector` exactly:
`getindex` on one is a pointer load, while a custom `AbstractVector` may compute in its `getindex`
and genuinely wants the trait. This decides only whether setup warns, never what runs.
"""
materialized_source(x) = x isa Vector

"""
    ReactantNitro.check_batch_at(source; values = false) -> nothing

Assert that [`batch_at`](@ref) agrees with the source's own `iterate`, batch for batch. This is the
only thing that catches a `batch_at` whose index units are wrong, and it needs no accelerator, so
it belongs in a model package's test suite. `values = false` compares field names and sizes, which
a stochastic loader permits; `values = true` compares contents and wants a deterministic split.
"""
function check_batch_at(source; values::Bool = false)
    fanout_capable(source) || error(
        """
        ReactantNitro: `check_batch_at` was given a `$(typeof(source))`, which does not implement both
        halves of the index-addressable trait, so there is nothing to check. It needs `batch_at` AND
        `begin_epoch!`."""
    )
    # Before the sequential pass: a source storing its own plan returns `nothing` and is re-planned
    # by `collect`, so the indexed pass reads the same plan; a source returning its plan draws two,
    # which is why `values = true` needs a deterministic source.
    plan = begin_epoch!(source)
    seq = collect(source)
    n = length(source)
    length(seq) == n || error("ReactantNitro: `check_batch_at`: the sequential pass yielded \
        $(length(seq)) batches and `length` promised $n. Fix that before checking `batch_at`.")
    for i in 1:n
        # The same call the fan-out makes, through the same dispatch, so what is checked is what runs.
        got = planned_batch_at(source, i, plan)
        want = seq[i]
        got === nothing && error("ReactantNitro: `check_batch_at`: `batch_at(source, $i)` returned \
            `nothing`, and every index in `1:length(source)` must produce a batch.")
        keys(got) === keys(want) || error(
            """
            ReactantNitro: `check_batch_at`: `batch_at(source, $i)` has fields $(keys(got)) and the
            sequential pass's batch $i has $(keys(want))."""
        )
        for k in keys(want)
            sg, sw = size(getproperty(got, k)), size(getproperty(want, k))
            sg == sw || error(
                """
                ReactantNitro: `check_batch_at`: at batch $i, field `$k` is $(sg) from `batch_at` and
                $(sw) from the sequential pass."""
            )
            values || continue
            isequal(getproperty(got, k), getproperty(want, k)) || error(
                """
                ReactantNitro: `check_batch_at`: at batch $i, field `$k` DIFFERS in value between
                `batch_at(source, $i)` and the sequential pass's batch $i.
                The overwhelmingly likely cause is that `batch_at`'s index is being used as a SAMPLE
                offset rather than a BATCH index: it must be
                `_build_batch(dl, 1 + (i - 1) * batch_size)`, not `_build_batch(dl, i)`.
                If this source SHUFFLES or its augmentation is stochastic, this check cannot be run
                with `values = true`: the sequential pass and the indexed pass each draw their own
                plan, so the two disagree for a reason that is not a bug. Run it on a deterministic
                split, or with `shuffle = false`."""
            )
        end
    end
    return nothing
end

"""
    ReactantNitro.batch_stream(split, routing)

The training loop's batch source. It yields `(host_batch, device_batch)` pairs: the loop needs the
host batch for the schema checks and a `:host` `train_metrics`, and the device batch for the
gradient program. A producer that throws surfaces at the consumer, since `Channel` closes with the
exception. The checks stay on the consumer, against the host half, so an error is raised on the
task the user's stack trace is about.

Three paths: the fan-out ([`PrefetchStream`](@ref)) when `workers > 1` and the source is
[`fanout_capable`](@ref); one producer feeding a `Channel` of `device_batches` otherwise; and an
inline generator for a [`NoPrefetch`](@ref) split. `prepare` is the step from host batch to device
payload, and passing it as a closure is what lets one pipeline serve training (transfer as is) and
evaluation (pad first, via [`eval_stream`](@ref)).
"""
batch_stream(split, routing, mesh = nothing) =
    prepared_stream(split, b -> to_device_batch(b, routing, mesh))

"""
    ReactantNitro.prepared_stream(split, prepare) -> stream

The shared pipeline behind [`batch_stream`](@ref) and [`eval_stream`](@ref). A distinct name
because `batch_stream(split, routing)` has the same arity and would be silently replaced.
"""
function prepared_stream(split, prepare)
    device_batches = prefetch_device_batches(split)
    device_batches == 0 && return ((b, prepare(b)) for b in split)
    src = prefetch_source(split)
    workers = prefetch_workers(split)
    (workers > 1 && fanout_capable(src)) && return fanout_stream(
        src, workers, device_batches, prefetch_host_batches(split),
        prepare, prefetch_ordered(split)
    )
    return Channel{Tuple{Any, Any}}(device_batches; spawn = true) do ch
        for b in src
            put!(ch, (b, prepare(b)))
        end
    end
end

"""
    ReactantNitro.eval_stream(split, xfer, batch_size, mesh) -> stream

[`batch_stream`](@ref) for an evaluation pass: the producer pads a short final batch to
`batch_size`, so the pad and the transfer overlap the previous batch's forward. `n_real` is not
carried through the stream; the consumer recomputes it from the same host batch, so the two cannot
disagree.
"""
eval_stream(split, xfer, batch_size::Integer, mesh = nothing) = prepared_stream(
    split, b -> to_device_batch(first(pad_batch(b, batch_size, xfer)), xfer, mesh)
)

"""
    ReactantNitro.PrefetchStream

The fan-out, as one object so [`close_stream!`](@ref) has one place to stop every task. Iterating
it yields the same `(host, device)` pairs the other paths yield.

```
coordinator  ->  Channel{Int}(workers)             the batch indices 1:n
   N workers ->  Channel{Tuple{Int,Any}}(workers)  HOST batches, with the index that produced them
      1 transfer task -> Channel{Tuple{Any,Any}}(device_batches)  (host, device) pairs
         the training loop

   credits  <-  Channel{Nothing}(host_batches)     one token per host batch, returned on emit
```

`credits` is `host_batches` made literal: a `Channel` rather than a `Semaphore` so that teardown
can close it and unwind a parked coordinator. `marks` is the exactly-once ledger, one byte per
batch index, `Vector{UInt8}` rather than `BitVector` because N workers write concurrently.
"""
struct PrefetchStream
    devch::Channel{Tuple{Any, Any}}
    hostch::Channel{Tuple{Int, Any}}
    jobs::Channel{Int}
    credits::Channel{Nothing}
    tasks::Vector{Task}
    marks::Vector{UInt8}
end

Base.IteratorSize(::Type{PrefetchStream}) = Base.SizeUnknown()
Base.eltype(::Type{PrefetchStream}) = Tuple{Any, Any}
Base.iterate(s::PrefetchStream) = iterate(s.devch)
Base.iterate(s::PrefetchStream, state) = iterate(s.devch, state)

# The error-propagation shape is reused from a prior data path that recorded two real deadlocks.
function fanout_stream(
        src, workers::Int, device_batches::Int, host_batches::Int, prepare, ordered::Bool = true
    )
    token_before = epoch_token(src)
    # Before `length`, since a plan may change the batch count. `plan` is `nothing` for a source
    # that stores its own, else the epoch's plan object, held here and handed to every worker so N
    # workers read one immutable object.
    plan = begin_epoch!(src)
    check_epoch_advanced(src, token_before)
    n = length(src)
    n > 0 || error("ReactantNitro: the training split reports `length == $n` after `begin_epoch!`; \
                    an epoch must have at least one batch.")

    jobs = Channel{Int}(workers)
    hostch = Channel{Tuple{Int, Any}}(workers)
    devch = Channel{Tuple{Any, Any}}(device_batches)
    marks = zeros(UInt8, n)

    # One token per batch allowed on the host, taken before a job is handed out and returned after
    # the batch is emitted. Both delivery modes take the window, so the knob means one thing.
    credits = Channel{Nothing}(host_batches)
    for _ in 1:min(host_batches, n)
        put!(credits, nothing)
    end

    # One shared job channel, so exactly-once delivery is a property of `Channel` rather than of
    # index arithmetic.
    coord = Threads.@spawn try
        for i in 1:n
            take!(credits)
            put!(jobs, i)
        end
    finally
        close_quiet!(jobs)
    end

    wtasks = [Threads.@spawn(prefetch_worker(src, plan, jobs, hostch, marks)) for _ in 1:workers]
    joiner = Threads.@spawn prefetch_joiner(coord, wtasks, hostch)
    xfer = Threads.@spawn prefetch_transfer(hostch, devch, prepare, ordered, credits)

    return PrefetchStream(devch, hostch, jobs, credits, [coord; wtasks; joiner; xfer], marks)
end

function prefetch_worker(src, plan, jobs::Channel{Int}, hostch::Channel, marks::Vector{UInt8})
    try
        for i in jobs
            b = planned_batch_at(src, i, plan)
            b === nothing && error(
                """
                ReactantNitro: `batch_at(source, $i)` returned `nothing`, and it must produce a batch
                for every index in `1:length(source)`. A loader whose sequential
                producer returns `nothing` at exhaustion must not forward that: `length` is what
                bounds the epoch, and returning `nothing` inside it DROPS a batch."""
            )
            # Marked before the hand-off, so the ledger records delivery even if teardown races it.
            marks[i] = 0x01
            put!(hostch, (i, b))
        end
    catch e
        # Closing `jobs` WITH the exception unblocks a coordinator parked in `put!` and stops the
        # siblings.
        close_quiet!(jobs, e)
        rethrow()
    end
    return nothing
end

"""
    ReactantNitro.planned_batch_at(source, i, plan) -> batch

The single call site for [`batch_at`](@ref), where the two shapes are told apart: `nothing` means
the source stores its own plan. Kept here rather than in a forwarding method so
[`fanout_capable`](@ref) can detect a three-argument implementation.
"""
planned_batch_at(source, i::Integer, ::Nothing) = batch_at(source, i)
planned_batch_at(source, i::Integer, plan) = batch_at(source, i, plan)

# `bind` is not used anywhere here: its close-on-done returns early while the channel `isready`,
# so error propagation would stall behind a buffered item. Every channel is closed explicitly.
function prefetch_joiner(coord::Task, wtasks::Vector{Task}, hostch::Channel)
    err = nothing
    try
        wait(coord)
    catch e
        err = unwrap_task_exc(e)
    end
    for w in wtasks
        try
            wait(w)
        catch e
            err === nothing && (err = unwrap_task_exc(e))
        end
    end
    close_quiet!(hostch, err)
    return nothing
end

function prefetch_transfer(
        hostch::Channel, devch::Channel, prepare, ordered::Bool, credits::Channel{Nothing}
    )
    err = nothing
    # The reorder buffer, keyed on batch index; host batches, so no explicit free at teardown.
    pending = Dict{Int, Any}()
    next = 1
    try
        # Iterating a channel the joiner closed WITH an exception rethrows it here, which is how a
        # worker's error reaches the consumer: this task then closes `devch` with it.
        for (i, b) in hostch
            if !ordered
                put!(devch, (b, prepare(b)))
                return_credit!(credits)
                continue
            end
            pending[i] = b
            # `haskey` rather than a `nothing` sentinel a batch could in principle take.
            while haskey(pending, next)
                hb = pop!(pending, next)
                put!(devch, (hb, prepare(hb)))
                next += 1
                # The credit returns once the batch has LEFT, so the window bounds what exists.
                return_credit!(credits)
            end
        end
    catch e
        err = e
    end
    close_quiet!(devch, err)
    err === nothing || rethrow(err)
    return nothing
end

"""
    ReactantNitro.return_credit!(credits) -> nothing

Hand one job credit back, and never raise doing it: [`close_stream!`](@ref) closes `credits` to
unblock the coordinator, and a `put!` racing that close would replace the real failure being
unwound with an `InvalidStateException`.
"""
function return_credit!(credits::Channel{Nothing})
    try
        put!(credits, nothing)
    catch
    end
    return nothing
end

# `wait(task)` wraps a failed task's result in `TaskFailedException`. Strip it so the consumer sees
# the loader's own exception type, which is what the inline path raises and what the tests assert.
function unwrap_task_exc(e)
    while e isa TaskFailedException
        r = e.task.result
        r isa Exception || break
        e = r
    end
    return e
end

# Closing is called from teardown paths that are usually already unwinding an exception, and from
# several tasks that may race each other to it, so it must never raise on its own.
function close_quiet!(ch::Channel, err = nothing)
    try
        err isa Exception ? close(ch, err) : close(ch)
    catch
    end
    return nothing
end

"""
    ReactantNitro.check_prefetch_delivery(stream) -> nothing

The exactly-once assertion, one byte per batch, checked once per epoch. Call it only on an epoch
that ran to completion; an early exit leaves the ledger legitimately partial. It catches a
duplicate paired with a drop, which [`check_epoch_length`](@ref) cannot; a wrong index mapping is
[`check_batch_at`](@ref)'s job. A no-op on the other stream types.
"""
check_prefetch_delivery(stream) = nothing

function check_prefetch_delivery(s::PrefetchStream)
    missed = findall(==(0x00), s.marks)
    isempty(missed) && return nothing
    error(
        """
        ReactantNitro: the prefetch fan-out finished an epoch without producing $(length(missed)) of
        its $(length(s.marks)) batches (first missing index $(first(missed))).
        Every index in `1:length(source)` is dispatched exactly once through one shared job channel,
        so this means a worker took a job and neither produced it nor raised. That is a DROPPED batch:
        the epoch trained on less data than it reported."""
    )
end

"""
    ReactantNitro.check_epoch_advanced(source, token_before) -> nothing

Assert that the fan-out's one [`begin_epoch!`](@ref) call advanced [`epoch_token`](@ref) by exactly
one. Declined by a source leaving the token at `nothing`. Unchanged means the re-plan is still in
`Base.iterate`, which the fan-out never calls; advanced by more than one means it fired twice.
"""
function check_epoch_advanced(source, token_before)
    token_before === nothing && return nothing
    after = epoch_token(source)
    after == token_before + 1 && return nothing
    error(
        """
        ReactantNitro: `begin_epoch!` on a `$(typeof(source))` moved `epoch_token` from
        $(repr(token_before)) to $(repr(after)); it must advance by exactly one.
        Unchanged means the source's epoch re-plan is still in `Base.iterate`'s initialization, which
        the index-addressable path NEVER CALLS, so every epoch after the first would train on the
        first epoch's plan. Advanced by more than one means it fired twice, which re-draws the plan
        and skips a generation of any epoch-seeded augmentation.
        Move the re-plan into `begin_epoch!` and have `Base.iterate` call that, so there is one
        implementation and the two entry points cannot drift."""
    )
end

"""
    ReactantNitro.close_stream!(stream) -> nothing

Stop a stream's producers and free the device buffers it still holds. Called from the consumer's
`finally`: an early exit (`request_stop!`, a
non-finite loss, an error) must stop the producers and free the device buffers still in the
channel, or it leaks a task and `device_batches x batch` of device memory. Closing each channel
unwinds each stage. On the fan-out it does not `wait` on the tasks, since a worker mid round trip
would hold teardown for seconds and holds no device memory; only the transfer task ever does, and
it is the easiest to stop. A no-op on the inline path.
"""
close_stream!(stream) = nothing

function close_stream!(ch::Channel)
    close_quiet!(ch)
    drain_and_free!(ch)
    return nothing
end

"""
    ReactantNitro.drain_and_free!(ch::Channel) -> nothing

Free the device halves still sitting in a closed stream's channel, and never raise doing it.
`isready` rather than iterating, since iterating a channel closed with an exception rethrows it.
The `try` is a bug fix: `isready` counts a blocked putter, so `take!` on a channel that is closed
with an empty buffer throws `InvalidStateException` from inside a `finally` and replaces the
caller's real exception. Freeing is best-effort over the GC anyway.
"""
function drain_and_free!(ch::Channel)
    try
        while isready(ch)
            _, b = take!(ch)
            free_batch!(b)
        end
    catch
    end
    return nothing
end

function close_stream!(s::PrefetchStream)
    # Upstream first, so no stage is refilled behind the drain below. `credits` leads, because the
    # coordinator can be parked on `take!(credits)` and closing `jobs` would not wake it there.
    close_quiet!(s.credits)
    close_quiet!(s.jobs)
    close_quiet!(s.hostch)
    close_quiet!(s.devch)
    # Only the device halves in the device channel are buffers nothing else has held.
    drain_and_free!(s.devch)
    return nothing
end

"""
    ReactantNitro.free_batch!(batch) -> nothing

Release a device batch's buffers now rather than at the GC's convenience. The pointer is nulled
after the free: `XLA.free_buffer` is a no-op on null and a double free on a live pointer that the
finalizer will visit again, and this stack asserts `buffer.buffer !== C_NULL` on readback, so a
use-after-free is loud. Only buffers the prefetch owns are freed here; a batch handed to a compiled
program is freed by the loop once the loss readback has awaited the step.
"""
function free_batch!(batch)
    _each_array_leaf(batch) do leaf
        leaf isa Reactant.ConcretePJRTArray || return nothing
        for async in leaf.data
            buf = async.buffer
            buf.buffer == C_NULL && continue
            Reactant.XLA.free_buffer(buf)
            buf.buffer = C_NULL
        end
        return nothing
    end
    return nothing
end
