# Data.jl
#
# The data-source contract's checks, the prefetch pipeline, and the pad-then-slice path for a
# short final eval batch. Nothing here is exported. The prefetch puts the H2D transfer inside
# itself, so it is realized against a resolved run rather than against what `build_data` returned.

"""
    ReactantNitro.check_data_source(source, name::Symbol) -> Int

Check at setup that a split supports `length` and is non-empty, and return its batch count.
`length` counts batches and fixes the schedule horizon. The source must also be restartable, which
cannot be checked here: setup draws one batch for the schema, so a one-shot source loses it, and
[`check_epoch_length`](@ref) catches that after the first epoch.
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

Reject or warn about source options the resolved pipeline cannot honour. Called at setup for
every split, `:inline` ones included, with the unwrapped source. The default does nothing;
`ReactantNitroMLUtilsExt` implements it for `MLUtils.DataLoader`. `cfg` is the split's resolved
[`prefetch_config`](@ref), since an option implemented inside `Base.iterate` takes effect on the
single-producer path and has no effect on the fan-out, which never calls `iterate`.
"""
check_source_options(source, name::Symbol, cfg) = nothing

"""
    ReactantNitro.check_epoch_length(seen::Integer, expected::Integer, name::Symbol;
                                     horizon_dependent::Bool = true) -> nothing

Check after an epoch that the split yielded the `expected` batches its `length` promised. A
one-shot source yields `expected - 1` on the first epoch, since setup consumed one, and the error
says so. Any other mismatch is an error only when `horizon_dependent`, because `total` was fixed
from `length` at setup and sets the learning-rate curve; a variable-length loader with constant
hyperparameters passes.
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
    ReactantNitro.check_train_divisibility(n_batches::Integer, accum::Integer,
                                           split::Symbol = :train) -> nothing

Require `n_batches % accum == 0` at setup, the setup-time half of the accumulation contract. It is
in batches because batches are the only unit the data contract exposes. It guarantees that an
accumulation group never spans an epoch boundary, so the accumulator is empty at every epoch edge
and checkpointing, resume and validation never handle partial accumulation state.
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
    ReactantNitro.batch_size_of(batch::NamedTuple, routing = nothing) -> Int

The batch size: the last dimension of the routed fields (all fields when `routing` is `nothing`),
which must all agree on it. Setup infers it from the first batch so it cannot disagree with the
loader; a `batch_size` field on the experiment is ordinary user config the framework never reads.
Metric denominators are separate; each metric reports its own.
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

Require `size(leaf)[end] == batch_size` for every routed field of a training batch, the
per-batch half of the accumulation contract. The training loader must drop its partial final batch.
Samples a loader drops never reach this check (3001 samples at batch size 64 drop 57 per epoch),
and nothing reports them.
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

Pad a short eval batch to `batch_size`, and slice the outputs back to `n_real`. Validation, test
and inference loaders may yield a short final batch; the framework pads it, runs `forward` at the
one compiled shape, and slices back before `metrics`, which never sees padding.

The result is exact in eval mode, where each sample's output does not depend on the rest of the
batch, but not guaranteed bitwise equal to an unpadded forward, since XLA's tiling may depend on
the batch extent. A `:device` `metrics` compiles twice per split, at `batch_size` and at the
remainder.

`slice_outputs` calls [`check_output_batch_dim`](@ref) before slicing, so a model whose batch
dimension is not last raises an error. Padding rows copy the last real sample rather than zeros,
since an all-zero row can produce a `NaN` in a normalization.
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

Assert that every array leaf of `outputs` has `batch_size` as its last dimension. Called before
slicing a short batch, so a model whose batch dimension is elsewhere raises an error instead of
returning the wrong samples. It runs only when there is something to slice, so an output leaf
without a batch dimension is accepted until a split has a short final batch.
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

Take the first `n` entries of every array leaf's last dimension, without checking. Separate from
[`slice_outputs`](@ref) because on the traced metric path the check runs on the host and the slice
runs inside the traced program.
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

Produce batch `i` of the current epoch. Optional; with [`begin_epoch!`](@ref) it opts a source into
the multi-producer fan-out, which calls it from many concurrent tasks, each `i` independently. A
source that stores its own plan defines the two-argument form. An immutable source has
`begin_epoch!` return the plan and defines the three-argument form; leave `plan` untyped, since
[`fanout_capable`](@ref) detects the method by asking whether it accepts `Any`.

`i` is a BATCH index in `1:length(source)`, not a sample offset:

```julia
ReactantNitro.batch_at(dl::MyLoader, k::Integer) =
    _build_batch(dl, 1 + (k - 1) * dl.batch_size)     # RIGHT
ReactantNitro.batch_at(dl::MyLoader, k::Integer) = _build_batch(dl, k)   # WRONG, and it TRAINS
```

The wrong version trains each epoch on overlapping windows of the first few hundred samples, and
the loss still falls; [`check_batch_at`](@ref) catches it. An implementation must be thread-safe
with respect to shared state (keep per-task scratch in a `TaskLocalValue`), depend only on `i` and
the epoch's plan (re-planning belongs in `begin_epoch!`), and never return `nothing` for an `i` in
`1:length(source)`.
"""
function batch_at end

"""
    ReactantNitro.begin_epoch!(source) -> plan

Plan the next epoch (shuffle, draw augmentation seeds). Optional; with [`batch_at`](@ref) it opts a
source into the multi-producer fan-out. The fan-out calls it once per epoch, before requesting any
batch. Return `nothing` if the source stores its own plan, and the framework calls the
two-argument `batch_at`; otherwise return the plan, and every three-argument `batch_at` call that
epoch receives it.

The fan-out never calls `Base.iterate`, and the other paths never call `begin_epoch!`, so a source
should re-plan here and have its `iterate` call this method, as below. A source that re-planned in
`iterate` and defined only `batch_at` would reuse epoch 1's plan for every epoch, so both methods
are required; a training split with only `batch_at` gets one producer and a setup warning.

```julia
ReactantNitro.begin_epoch!(::MySource) = nothing            # no re-plan needed
ReactantNitro.begin_epoch!(d::MyLoader) = shuffled_index(d.rng, d.n)   # a declaration's plan
Base.iterate(s::MySource) = (begin_epoch!(s); _first_batch(s))         # one implementation
```

`length(source)` is read after this returns, so a plan may change the batch count. Setup's schema
probe goes through `Base.iterate`, so a source that counts its own epochs sees `N + 1` calls across
`N` epochs, and a loader that plans eagerly in `build_data` needs a flag saying the first plan is
already drawn.
"""
function begin_epoch! end

"""
    ReactantNitro.epoch_token(source)

An epoch counter that each [`begin_epoch!`](@ref) call increments by one, or `nothing` (the default)
to skip the check. The fan-out asserts `epoch_token(source) == before + 1` around its one
`begin_epoch!` call, which catches a missing re-plan and a doubled one; a doubled one skips a
generation of epoch-seeded augmentation. Define it as an `Int` field the source increments.
"""
epoch_token(source) = nothing

"""
    ReactantNitro.release!(source) -> nothing

Release what a source holds open, such as a data server, a client or a file. Optional; the
default does nothing. The framework calls it once per handle on every split's source, unwrapped
from its [`PrefetchIterator`](@ref) or [`NoPrefetch`](@ref), at the first [`Terminal`](@ref) phase
(`Done` or `Failed`) the handle publishes, after the phase monitors have run. For a handle that
only evaluates or exports, and so never reaches a `Terminal` phase, call `release!(nitro)`. An
error a method raises is logged as a warning and does not stop the other splits.

A method must tolerate a second call, since a source may be released explicitly and then again at
`Terminal`. A collection shared between two handles (`Nitro(e2; data = nitro.data)`) is released
by whichever finishes first; give each handle its own collection.

The source's package defines this method with the rest of the trait. For a public loader the
framework may ship the extension, as with `ReactantNitroMLUtilsExt`; otherwise the loader's package
declares ReactantNitro as a weak dependency and ships the extension itself.
"""
release!(source) = nothing

"""
    ReactantNitro.fanout_capable(source) -> Bool

Whether `source` defines both [`begin_epoch!`](@ref) and either form of [`batch_at`](@ref). There
is no generic three-argument forwarding method, since one would make `hasmethod` return `true` for
every source.
"""
fanout_capable(source) =
    applicable(begin_epoch!, source) && (
    applicable(batch_at, source, 1) ||
        hasmethod(batch_at, Tuple{typeof(source), Integer, Any})
)

"""
    ReactantNitro.default_prefetch_workers() -> Int

The default producer count: `Threads.nthreads(:default)`, the default thread pool size (`N` under
`julia -t N,1`), at least 1. This makes `-t` a data-throughput setting, so the resolved value is
logged as a hyperparameter.
"""
default_prefetch_workers() = max(1, Threads.nthreads(:default))

const DEFAULT_DEVICE_BATCHES = 1

"""
    ReactantNitro.PrefetchIterator(source; workers = default_prefetch_workers(),
                                   device_batches = 1, host_batches = 2 * workers, ordered = true)

Prefetch settings for a split: `workers` producer tasks build host batches, one transfer task
copies them to the device, and a `Channel` of `device_batches` `(host, device)` pairs feeds the
loop.

The framework wraps every split in one of these at the defaults (see [`auto_prefetch`](@ref)).
Construct one only to change the defaults; [`NoPrefetch`](@ref) opts a split out.

  * **`workers`** sets throughput. Buffering only adds lookahead: a producer slower than the device
    starves it at any `device_batches`. More than one worker needs [`batch_at`](@ref) and
    [`begin_epoch!`](@ref) on the source; without them the split gets one producer, and setup warns
    for the training split.
  * **`device_batches`** bounds batches staged on the device. Resident device memory is
    `(device_batches + 2) x batch` (the channel, the transfer task's hand, the consumer's hand),
    hence the default of 1.
  * **`host_batches`** bounds batches on the host at once, across the workers, the hand-off
    channel and the reorder buffer. It must be at least `workers`, or ordered delivery can deadlock
    waiting on a batch that was never handed out.
  * **`ordered = true`** emits in source order through a reorder buffer, so a fixed seed reproduces
    bitwise at any `workers`. `ordered = false` emits batches as they finish, so the epoch's
    accumulation groups change and runs no longer reproduce bitwise; every batch is still seen
    exactly once. Applies only to the fan-out; one producer always emits in order.

The transfer runs on one task because device memory cannot be bounded by a semaphore: XLA runs
asynchronously and the executable holds its inputs, so there is no point at which the consumer is
known to be done. Device buffers are freed explicitly, not left to the GC.

Batching, shuffling and splitting are not provided; use `MLUtils.jl`.

The wrapper holds only the source and its settings, since `build_data` runs before batch routing
and device placement exist; the training loop builds the pipeline. `Base.iterate` therefore passes
through to the source with no task and no channel, so iterating one by hand cannot leak a producer,
and `render` and `predict(nitro, loader)` are unaffected by the wrapping. Eval splits stream
through [`eval_stream`](@ref), which pads a short final batch before the transfer.
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

Opt a split out of prefetch, so its host data path runs inline on the training task. This is the
only way to do so; `device_batches = 0` is an error, so the choice is visible in the code rather
than looking like tuning. `iterate` passes through to the source, as `PrefetchIterator`'s does.
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

Unwrap a split, and read its prefetch settings. For anything that is not a
[`PrefetchIterator`](@ref) the three counts return `0`, meaning the split does not stream; that `0`
is internal, never a user-settable value. Setup checks a wrapped split's source through
`prefetch_source`.
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

Wrap every split in a [`PrefetchIterator`](@ref) at the defaults, unless it is already a
`PrefetchIterator` or a [`NoPrefetch`](@ref). Eval splits are included, since a forward-only step
leaves less device time to hide the data path behind. Called at setup after `derive`, the schema
probe and the contract checks, so those see the collection as `build_data` returned it.
"""
auto_prefetch(collection::NamedTuple) =
    NamedTuple{keys(collection)}(map(_auto_wrap, values(collection)))

_auto_wrap(split::PrefetchIterator) = split
_auto_wrap(split::NoPrefetch) = split
_auto_wrap(split) = PrefetchIterator(split)

"""
    ReactantNitro.prefetch_config(split) -> (; device_batches, host_batches, workers, ordered, path)

The prefetch settings a split resolves to, which may differ from the requested ones. `path` is:

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

Whether a source's batches already exist, so `workers > 1` would gain nothing. True only for a
`Vector`: its `getindex` is a load, while a custom `AbstractVector` may compute in `getindex` and
benefit from the trait. This decides only whether setup warns, not what runs.
"""
materialized_source(x) = x isa Vector

"""
    ReactantNitro.check_batch_at(source; values = false) -> nothing

Assert that [`batch_at`](@ref) agrees with the source's own `iterate`, batch for batch. It is the
only check that catches a `batch_at` using the wrong index units, and it needs no accelerator, so
it belongs in a model package's test suite. `values = false` compares field names and sizes, which
works for a stochastic loader; `values = true` also compares contents and needs a deterministic
source, since the two passes may draw different plans.
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
    ReactantNitro.batch_stream(split, routing, mesh = nothing) -> stream

The training loop's batch source, yielding `(host_batch, device_batch)` pairs. The loop needs the
host batch for the schema checks and a `:host` `train_metrics`, and the device batch for the
gradient program. An exception in a producer is rethrown at the consumer, since the `Channel` is
closed with it. The checks run on the consumer, against the host batch, so errors are raised on the
training task.

Three paths: the fan-out ([`PrefetchStream`](@ref)) when `workers > 1` and the source is
[`fanout_capable`](@ref); one producer feeding a `Channel` of `device_batches` otherwise; and an
inline generator for a [`NoPrefetch`](@ref) split. All go through `prepared_stream` with a
`prepare` function from host batch to device payload: a plain transfer here, pad then transfer in
[`eval_stream`](@ref).
"""
batch_stream(split, routing, mesh = nothing) =
    prepared_stream(split, b -> to_device_batch(b, routing, mesh))

"""
    ReactantNitro.prepared_stream(split, prepare) -> stream

The shared pipeline behind [`batch_stream`](@ref) and [`eval_stream`](@ref), applying `prepare` to
each host batch. A separate name because a `batch_stream(split, prepare)` method would have the
same arity as `batch_stream(split, routing)` and overwrite it.
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

[`batch_stream`](@ref) for an evaluation pass: a short final batch is padded to `batch_size`
before the transfer, so on the prefetch paths the pad and the transfer overlap the previous
batch's forward. `xfer` is the routing whose fields are transferred. The yielded host batch is
unpadded, and the consumer recomputes `n_real` from it rather than carrying it through the
stream.
"""
eval_stream(split, xfer, batch_size::Integer, mesh = nothing) = prepared_stream(
    split, b -> to_device_batch(first(pad_batch(b, batch_size, xfer)), xfer, mesh)
)

"""
    ReactantNitro.PrefetchStream

The multi-producer fan-out, held as one object so [`close_stream!`](@ref) can stop every task.
Iterating it yields the same `(host, device)` pairs as the other paths.

```
coordinator  ->  Channel{Int}(workers)             the batch indices 1:n
   N workers ->  Channel{Tuple{Int,Any}}(workers)  HOST batches, with the index that produced them
      1 transfer task -> Channel{Tuple{Any,Any}}(device_batches)  (host, device) pairs
         the training loop

   credits  <-  Channel{Nothing}(host_batches)     one token per host batch, returned on emit
```

`credits` holds `host_batches` tokens; it is a `Channel` rather than a `Semaphore` so teardown can
close it and wake a blocked coordinator. `marks` records delivery, one byte per batch index; a
`Vector{UInt8}` rather than a `BitVector` because workers write it concurrently.
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

Call the right form of [`batch_at`](@ref): the two-argument form when `plan` is `nothing` (the
source stores its own plan), the three-argument form otherwise. A function rather than a forwarding
`batch_at` method, so [`fanout_capable`](@ref) can still detect a three-argument implementation.
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

Return one credit to `credits`, ignoring any error. [`close_stream!`](@ref) closes `credits` to
wake the coordinator, and a `put!` racing that close would otherwise replace the exception being
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

Assert that the fan-out produced every batch index of the epoch. Call it only after an epoch that
ran to completion, since an early exit leaves batches legitimately unproduced. It catches a
duplicate paired with a drop, which [`check_epoch_length`](@ref) cannot; a wrong index mapping is
caught by [`check_batch_at`](@ref). A no-op on the other stream types.
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
one. Skipped when the token is `nothing`. Unchanged means the re-plan is still in `Base.iterate`,
which the fan-out never calls; advanced by more than one means it ran twice.
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

Stop a stream's producers and free the device batches still in its channel. Called from the
consumer's `finally`, so an early exit (`request_stop!`, a non-finite loss, an error) does not leak
a task and `device_batches x batch` of device memory. It closes each channel, which stops each
stage, and does not `wait` on the tasks: a worker in the middle of a slow fetch could delay
teardown for seconds, and only the transfer task holds device memory. A no-op on the inline
path.
"""
close_stream!(stream) = nothing

function close_stream!(ch::Channel)
    close_quiet!(ch)
    drain_and_free!(ch)
    return nothing
end

"""
    ReactantNitro.drain_and_free!(ch::Channel) -> nothing

Free the device batches left in a closed stream's channel, ignoring any error. It polls `isready`
rather than iterating, since iterating a channel closed with an exception rethrows it. The `try`
is needed because `isready` counts a blocked putter, so `take!` on a closed channel with an empty
buffer throws `InvalidStateException` inside a `finally` and would replace the caller's exception.
Freeing is best-effort; the GC frees anything missed.
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

Free a device batch's buffers now rather than waiting for the GC. The pointer is set to `C_NULL`
after the free, because the finalizer calls `XLA.free_buffer` again, which is a no-op on null and a
double free otherwise; a later readback of a freed buffer then fails on the null check rather than
reading freed memory. Only the prefetch's own buffers are freed here; a batch passed to a compiled
program is freed by the loop after the loss readback has waited for the step.
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
