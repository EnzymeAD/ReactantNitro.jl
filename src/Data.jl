# Data.jl
#
# The data-source contract's checks, the staged prefetch pipeline, and the pad-then-slice path for a short final
# eval batch. Nothing here is exported.
#
# The prefetch puts the H2D transfer INSIDE itself, so it depends on the device placement and the
# sharding that setup resolves, which is why it is realized against a resolved run rather than
# against whatever `build_data` returned.

"""
    ReactantNitro.check_data_source(source, name::Symbol) -> Int

The data-source requirements, checked at setup with an error naming the source rather than failing
later. Returns the batch count.

**`length` is required**, and counts **batches**. It is read once, at setup step 11, to fix the
schedule horizon, and a source without it raises here naming the source and the requirement rather
than failing inside the horizon computation.

**The source must also be restartable**, which is a contract this function cannot check: setup draws
one batch to learn the schema and then discards it, and the loop iterates from scratch, so a one-shot
source such as a bare `Channel` silently loses its first batch. What the framework *can* do is catch
it after the fact, which [`check_epoch_length`](@ref) does at the end of the first epoch.
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
    ReactantNitro.check_source_options(source, name::Symbol) -> nothing

A hook for options a source type can carry that a PREFETCHED pipeline cannot honour, checked at
setup on every split the framework wrapped.

The default does nothing, because the framework knows nothing about any particular loader's
options. An extension for a loader it does know implements this; `ReactantNitroMLUtilsExt` refuses
`MLUtils.DataLoader`'s `buffer = true`, whose one reused batch is silently overwritten while it is
still queued for transfer, and warns on `parallel = true`, whose own thread pool breaks the ordered
delivery this framework promises.

**Separate from [`check_data_source`](@ref) and called later**, because it is a question about the
resolved pipeline rather than about the source: an inline split consumes each batch before the next
is built, so the aliasing that makes `buffer = true` dangerous does not arise there.
"""
check_source_options(source, name::Symbol) = nothing

"""
    ReactantNitro.check_epoch_length(seen::Integer, expected::Integer, name::Symbol;
                                     horizon_dependent::Bool) -> nothing

The two things about a data source that only an actual epoch can establish.

**A one-shot source is caught here.** Setup drew one batch for the schema and discarded it, so a
source that does not restart yields `expected - 1` batches on the first epoch. That is a specific,
recognizable shortfall and the error says so, rather than leaving a silently short epoch.

**`length` must be stable across epochs** when a horizon-dependent schedule is in use, because
`total` must be fixed or the LR curve shifts. A variable-length loader with a constant learning rate
is fine, so the check is conditioned rather than unconditional.
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

The setup-time half of the accumulation contract: `length(train) % accum == 0`, an error naming
the split, the batch count, `accum`, and the nearest batch counts that would work.

**Stated in batch units** because batches are the only unit the data-source contract exposes. The
sample form `n(train) % (batch_size * accum) == 0` is **not** equivalent: it is the conjunction of
"no short final training batch" and "batch count divisible by `accum`", and this is only the second.
Nothing supplies `n(train)`, and adding a `numobs` requirement to the data contract would burden
every loader, return the wrong number for a `Vector` of batches, and hand the framework a
`drop_last` assumption it cannot verify.

**This buys an invariant, not tidiness: an accumulation group never spans an epoch boundary.** The
accumulator is always empty at an epoch edge, which is why checkpointing, resume, and validation
never reason about partial accumulation state, and why the schedule horizon is an exact division.
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

The batch **shape**, inferred from the first batch rather than read from a config field, because
an inferred value cannot disagree with the loader that produced it whereas a `batch_size` field and a
`DataLoader(batchsize = ...)` argument can drift apart silently. A `batch_size` field on the
experiment is therefore ordinary user config, used to construct the loader, and the framework never
reads it.

**The batch dimension is last**, and every routed field must agree on it. A disagreement is an
error here rather than a wrong slice later, since the eval pad-and-slice path and the per-batch
training assertion both depend on this number.

Note this says nothing about metric **denominators**: each metric reports its own. The two facts
coexist because the framework knows the batch shape and still never assumes it is any metric's
denominator.
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

The per-batch half of the accumulation contract: assert `size(leaf)[end] == batch_size` on every
training batch, with the first violation naming the split, the batch index, and the two sizes.

This is the condition the batch-count check above cannot express, and this is the honest place to
catch it: **the framework can see a batch it was handed and cannot see samples a loader dropped
before handing anything over.** The training loader is required to drop its partial final batch, so a
short training batch reaches the loop only from a loader configured against that requirement.

**Samples silently dropped by drop-last are reported, not caught.** A train split of 3001 samples at
`batch_size = 64` yields 46 batches and drops 57 samples every epoch, which no check here detects
because the loader has already floored the count. The binding report states the shortfall when the
source supports `MLUtils.numobs`; that is a data-hygiene fact rather than a correctness invariant.
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

The eval path, shipped and **not optional**: validation, test, and inference loaders may yield a
short final batch and should not drop it. The framework pads up to `batch_size`, runs `forward` at
**one shape**, slices the outputs and the routed batch fields back to `n_real`, and only then calls
`metrics`. **`metrics` never sees padding**, so there is no mask for a user to forget to apply.

**This is exact, not an approximation.** In eval mode each sample's output is independent of the rest
of the batch: BatchNorm uses running statistics, Dropout is off, and no standard layer mixes across
the batch dimension. **It is not guaranteed bitwise**: two shapes are two compiled programs, and
XLA's tiling and accumulation order inside a matmul or convolution may depend on the batch
dimension's extent. The claim is numerically exact to the precision of the two programs, and bitwise
on CPU.

`slice_outputs` asserts `size(leaf)[end] == batch_size` on each output leaf **before** slicing, so a
model whose batch dimension is not last fails loudly rather than silently slicing the wrong axis.
**Both take the full `batch_size` as an argument**, and that third argument is load-bearing: the
assertion is a comparison against the width the compiled program was built at, and a `slice_outputs`
that learned that width from the leaves it is checking would be comparing each leaf against itself.

Cost: `metrics` compiles twice per split, once at `batch_size` and once at the remainder, since there
is at most one short batch. `forward` is unaffected.

**The padding rows are copies of the last real sample, not zeros.** Either is correct, because the
framework guarantees they are sliced away before any hook sees them and eval-mode per-sample
independence keeps them out of the real rows' outputs. Replication is the safer of the two: an
all-zero row can produce a `NaN` in a normalization or a division that a real row never would, and
a `NaN` in a padded lane is exactly the kind of thing that turns a documented-as-harmless fill into
a debugging session on some future model.
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

# Pad one field along its last dimension by REPLICATING the last real sample. Fields whose last
# dimension is not `n_real` are left alone: they are not batch-shaped, so there is nothing to pad,
# and `batch_size_of` has already established that every ROUTED field agrees on `n_real`.
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

The assertion `slice_outputs` makes before it slices: every array leaf of the output tree must have
`size(leaf)[end] == batch_size`. A model whose batch dimension is not last
**fails loudly here rather than silently slicing the wrong axis**, which would return the wrong
samples with no error and metrics that merely look a little off.

Checked only on the path that actually slices, which is the short final batch. A model with
a genuinely batch-free output leaf is therefore not rejected until it meets a split whose length is
not a multiple of the batch size, and the error names the leaf when it does.
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
[`slice_outputs`](@ref) because the check and the slice happen in different places on the traced
metric path: the check is host-side against the width the program was compiled at, and the slice is
**inside** the traced metric program, where the device arrays are. Everything that is not an array
leaf passes through.
"""
slice_last(tree, n::Integer) = _map_array_leaves(x -> _slice_leaf(x, n), tree)

function _slice_leaf(x::AbstractArray, n::Integer)
    N = ndims(x)
    (N == 0 || size(x, N) == n) && return x
    # A `copy` rather than a view: the ReshapedArray lesson is that a view-shaped value crossing a
    # program boundary has a different type from the array it came from, which costs a recompile
    # per call. Here it would also hand the user a view onto a buffer the eval loop frees.
    return copy(selectdim(x, N, 1:n))
end
_slice_leaf(x, n) = x

# Walk the array leaves of an output tree. `outputs` is whatever `forward` returned, which is
# constrained only to being a tree of arrays, so both helpers go through Functors with arrays as
# the leaf type. A traced program calls these at TRACE time on `TracedRArray` leaves, where the
# structure walk is host-side and free.
_is_array_leaf(x) = x isa AbstractArray && !(eltype(x) <: AbstractArray)

# ── The residency question, which is NOT the structural one above ───────────────────
#
# `_is_array_leaf` answers "is this ONE ARRAY TO SLICE", and `slice_last` needs it to keep
# answering exactly that: a view IS one array to slice, and descending into its parent would slice
# the wrong buffer.
#
# `_hidden_array` answers "COULD DEVICE MEMORY BE HIDING UNDER THIS", and a view of a device array
# answers YES where the structural question answers "leaf". Conflating the two let a device array
# behind `view`, `reshape`, or `vec` pass BOTH the host converter and the assertion built to catch
# the converter, because a `SubArray` and a `ReshapedArray` are not `AbstractConcreteArray`s.
#
# THE RECURRING BLIND SPOT IS WRAPPERS AROUND DEVICE ARRAYS, NOT DEVICE ARRAYS. This is the third
# time the shape has appeared: `vec(::ConcretePJRTArray)` returning a `ReshapedArray` is what hid
# the `_concat_group` scalar-indexing defect that made the framework unable to run on a GPU at all.
#
# Both residency walkers call this one function so they cannot drift apart again. `Base.parent` is
# the whole test: it returns `x` itself for a dense `Array` and for a `ConcretePJRTArray`, and the
# wrapped array for `SubArray`, `ReshapedArray`, `Adjoint`, `Transpose`, and `PermutedDimsArray`.
_hidden_array(x::AbstractArray) = (p = parent(x); p === x ? nothing : p)
_map_array_leaves(f, tree) = Functors.fmap(f, tree; exclude = _is_array_leaf)

function _each_array_leaf(f, tree)
    _map_array_leaves(leaf -> (f(leaf); leaf), tree)
    return nothing
end

# ── The index-addressable trait, which is what makes N producers possible at all ─────
#
# THE HISTORY MATTERS HERE, because this trait was deleted once already and cost a 4.2x regression.
# A prior framework fanned out to `num_workers = threads ÷ 2` producers over two methods a model
# supplied on its own loader, one returning the epoch's job list and one producing a batch from a
# start offset. The port to this framework dropped both as "framework-owned now", the replacement
# was ONE producer task with the device staging as a mere `Channel` capacity, and no layer picked the
# capability up. The first ported model then trained about four times slower than its reference,
# entirely in the host data path.
#
# A plain sequential Julia iterator CANNOT be consumed by N tasks safely, because the expensive work
# happens inside `iterate` and there is no way to ask for batch k without walking to it. So the
# capability needs a contract, and this is it.

"""
    ReactantNitro.batch_at(source, i::Integer) -> batch
    ReactantNitro.batch_at(source, i::Integer, plan) -> batch

**Optional, and half of the index-addressable opt-in.** Produce batch `i` of the current epoch,
independently of every other `i`, from a task that may be one of many running concurrently.

**Two shapes, and which one you write depends on where the epoch's plan lives.** A source that
stores its own plan writes the two-argument form; a source that is an immutable *declaration* of a
dataset, with nowhere to store one, has [`begin_epoch!`](@ref) return the plan and writes the
three-argument form. `MLUtils.DataLoader` is the second kind, and the extension that supports it is
two methods and no state.

**Leave the third argument untyped.** The capability check asks whether a three-argument method
accepts any plan at all, so annotating it with the plan's concrete type hides it and the source
falls back to one producer.

**`i` is a BATCH index in `1:length(source)`, not a sample offset.** That distinction is the single
easiest way to corrupt a run with this trait: a loader whose own producer takes a sample offset needs
the multiplication, and

```julia
ReactantNitro.batch_at(dl::MyLoader, k::Integer) =
    _build_batch(dl, 1 + (k - 1) * dl.batch_size)     # RIGHT
ReactantNitro.batch_at(dl::MyLoader, k::Integer) = _build_batch(dl, k)   # WRONG, and it TRAINS
```

The wrong version compiles, runs, and trains an epoch on `length(dl)` heavily overlapping windows of
the first few hundred samples, with a loss curve that still falls. No runtime assertion can see it,
because a permuted or overlapping index set produces perfectly well-shaped batches;
[`check_batch_at`](@ref) is what catches it, and it needs no accelerator.

Requirements on the implementation:

  * **Thread-safe with respect to the source's shared state.** Per-task scratch (a shared-memory
    segment, an RNG, a mutable buffer) belongs in a `TaskLocalValue` or is allocated per call.
  * **A pure function of `i` and the epoch's plan.** Anything that re-plans the epoch belongs in
    [`begin_epoch!`](@ref), which the framework calls exactly once before any `batch_at`. The
    three-argument form makes that literal: the plan arrives as an argument, so N workers read one
    immutable object rather than racing on the source's fields.
  * **Never `nothing`** for `i` in `1:length(source)`. A loader whose sequential producer returns
    `nothing` at exhaustion must not forward that; the framework raises on it.

Define this **and** [`begin_epoch!`](@ref) to opt in. Defining only one is a deliberate dead end (see
`begin_epoch!` for why).
"""
function batch_at end

"""
    ReactantNitro.begin_epoch!(source) -> plan

**Optional, and the other half of the index-addressable opt-in.** Re-plan the epoch. Called exactly
once per epoch, on the training task, **before any job is dispatched** to any worker.

**Return `nothing` if the source stores its own plan**, which is what every stateful loader does and
what this returned before there was anything to return; the framework then calls the two-argument
[`batch_at`](@ref). **Return the plan itself** if the source cannot store one, and the framework
hands it back to every three-argument `batch_at` call for that epoch. The returned value is opaque:
the framework holds it, passes it along, and drops it when the epoch ends.

This is what `Base.iterate`'s initialization used to do, and the reason it cannot stay there: the
fan-out never calls `iterate` on the source at all, so a source that re-plans in its iteration init
and exposes [`batch_at`](@ref) would **train every epoch after the first on epoch 1's plan**, silently
and with a plausible loss curve.

**That is why both methods are required to opt in**, rather than `batch_at` alone with a no-op
default here. A no-op default would make the stale-plan bug the *default* outcome for exactly the
loaders that most need the fan-out. With both required, a source that supplies only `batch_at` gets
the single-producer path and a warning naming this function, which is a loud opt-out instead of a
silent corruption.

A source that genuinely needs no re-plan opts in with a one-liner:

```julia
ReactantNitro.begin_epoch!(::MySource) = nothing
```

A source that is a declaration rather than a cursor returns its plan instead, and needs no mutable
field at all:

```julia
ReactantNitro.begin_epoch!(d::MyLoader) = shuffled_index(d.rng, d.n)
ReactantNitro.batch_at(d::MyLoader, i::Integer, plan) = _build_batch(d, plan, i)
```

**Implement it once and call it from `Base.iterate` too**, so the two entry points cannot drift:

```julia
Base.iterate(s::MySource) = (begin_epoch!(s); _first_batch(s))
```

`length(source)` is read **after** this returns, so a source whose batch count changes with the plan
is handled correctly.

**Setup calls it once, before any epoch.** Setup draws one batch through `first(source)` to learn
the batch schema, and that goes through `Base.iterate`, so a source counting its own epochs sees
`N + 1` calls across `N` training epochs. That is pre-existing behavior rather than something this
trait introduced, since the probe always called `iterate`, and it is why a loader that plans its
epoch eagerly in `build_data` needs a flag saying the first plan is already drawn: the probe is
what consumes it.
"""
function begin_epoch! end

"""
    ReactantNitro.epoch_token(source)

An opaque value that must change on every [`begin_epoch!`](@ref), or `nothing` (the default) to
decline the check.

Purpose: the fan-out asserts that its one `begin_epoch!` call advanced the source by **exactly one**
epoch. That catches the two ways the re-plan can go wrong once it lives in two places, a missing call
and a double call, and a double call is not hypothetical: it re-draws the plan and, for a loader whose
augmentation is seeded from an epoch counter, silently skips an epoch of the augmentation stream.

A monotone `Int` counter incremented inside `begin_epoch!` is the intended implementation.
"""
epoch_token(source) = nothing

"""
    ReactantNitro.fanout_capable(source) -> Bool

Whether `source` implements **both** halves of the index-addressable trait. Both, deliberately: see
[`begin_epoch!`](@ref).

Either [`batch_at`](@ref) shape satisfies the second half. **There is deliberately no generic
three-argument forwarding method**, because one would make `hasmethod` answer `true` for every
source alive and this check would stop meaning anything; the framework picks the shape at its one
call site instead, on whether the plan came back `nothing`.
"""
fanout_capable(source) =
    applicable(begin_epoch!, source) && (
    applicable(batch_at, source, 1) ||
        hasmethod(batch_at, Tuple{typeof(source), Integer, Any})
)

"""
    ReactantNitro.default_prefetch_workers() -> Int

`Threads.nthreads(:default)`, the **non-interactive** pool size, which is what `julia -t N,1` sets to
`N`.

Named rather than written inline at the call site, because the choice has a consequence a reader
should be able to find: `-t` is no longer only a CPU-politeness knob, it is a data-throughput setting,
and two runs of byte-identical code at different `-t` get different loader throughput AND a different
partition of each epoch into accumulation groups (the order the fan-out delivers in is not the
source's). The resolved value is therefore reported by the binding report and logged as a
hyperparameter, not left to be inferred.
"""
default_prefetch_workers() = max(1, Threads.nthreads(:default))

const DEFAULT_DEVICE_BATCHES = 1

"""
    ReactantNitro.PrefetchIterator(source; workers = default_prefetch_workers(),
                                   device_batches = 1, host_batches = 2 * workers, ordered = true)

Prefetch: `workers` producer tasks building **host** batches, one transfer task performing the H2D
copy, and a `Channel` of `device_batches` `(host, device)` pairs feeding the training loop. This is
what keeps the device fed across variable host latency.

**The two buffer knobs are one per side of the transfer, and each names what it costs.**
`device_batches` is how many batches sit ON THE DEVICE staged ahead, and `host_batches` is how many
may exist ON THE HOST at once: built but not yet transferred, counting the workers' hands, the
hand-off channel, and the reorder buffer together. Neither is per-worker.

**The framework wraps the `train` split with this automatically** (see [`auto_prefetch`](@ref)), at
these defaults, so a user normally never writes it. It stays public for the case where the defaults
are wrong, and [`NoPrefetch`](@ref) is how a split declines entirely.

  * **`workers` is the knob that matters.** Staging is lookahead, not concurrency: one producer at
    1.5 s/batch cannot feed a 0.4 s consumer at any `device_batches`, because the buffer simply stays
    empty. A worker count is what turns `host + device` per batch into `max(host / workers, device)`,
    and it requires [`batch_at`](@ref) and [`begin_epoch!`](@ref) on the source. Without them this
    falls back to one producer and setup says so.
  * **Resident device memory is `(device_batches + 2) x batch`**: `device_batches` in the channel,
    one in the transfer task's hand, one in the consumer's. That `+ 2` is why the default is 1.
  * **Resident host memory is about `host_batches x batch`**, and the default of `2 x workers` is
    what keeps every worker able to build one batch while another waits to be transferred.
  * **The two bound different failures.** `device_batches` back-pressures a SLOW CONSUMER: when the
    device channel fills, the transfer task blocks, stops draining the hand-off, and the workers
    block behind it. `host_batches` back-pressures a SLOW BATCH, which the device side cannot see at
    all: with ordered delivery a straggler leaves the device starving while finished batches pile up
    behind it, so the ceiling has to sit upstream at the coordinator.
  * **The transfer is on ONE task, not on the workers.** Two reasons, and the first is fatal to the
    alternative: bounding device memory would need a semaphore released when the consumer is *done*
    with a batch, and there is no such moment (XLA execution is asynchronous and the executable holds
    its inputs, which is why `free_batch!` refuses to free a consumed batch). Second, it keeps every
    PJRT buffer creation on one task, exactly as the single-producer path did.
  * **Buffer lifetime is a known sharp edge**: freeing must be explicit rather than left to the GC.
    The eval loop applies the same principle to validation outputs.
  * **Failure and cleanup.** A worker that throws must surface at the consumer rather than hanging it,
    and early exit must stop every task and free what the stream holds. This path only runs when
    something has already gone wrong, so it is tested by deliberately throwing mid-epoch.
  * **`ordered = true` is the default, and it is what makes fanning out free.** The workers finish
    out of order, so the transfer stage holds finished batches in a reorder buffer and emits them in
    the source's own order. A fixed seed therefore reproduces a run bitwise whatever `workers` is,
    and adopting [`batch_at`](@ref) on a source changes its throughput and nothing else.
  * **`ordered = false` trades that for throughput**, which is the prior framework's behaviour and
    this one's before the reorder buffer existed. Emission then never waits on a straggler, at the
    cost that reordering repartitions the epoch into different accumulation groups, so a fixed seed
    no longer reproduces bitwise. It stays statistically equivalent to a different shuffle: every
    sample is seen exactly once, and `check_train_divisibility`'s invariant (a group never spans an
    epoch boundary) is preserved either way.
  * **`host_batches` is enforced upstream, at the coordinator.** A reorder buffer that simply
    drained the workers would grow to a whole epoch behind one slow batch, so the coordinator takes
    a credit before handing out a job and the transfer stage returns one after emitting a batch.
    `host_batches` may not be smaller than `workers`, and the constructor says so: the batch ordered
    delivery is waiting for must be inside the window, or a straggler deadlocks instead of stalling.

**Batching, shuffling, and splitting are not shipped**: `MLUtils.jl` covers those.

## It is a DECLARATION, and the framework realizes it

The user constructs this in `build_data`, which setup runs early. The H2D transfer needs the
resolved batch routing and the device placement, **neither of which exists yet**, so this wrapper
cannot carry a producer: it carries the source and the settings, and the training loop builds the
pipeline once the transfer it is supposed to overlap is defined.

The consequence for the user is nil, and the consequence for a reader of this file is that
`Base.iterate` here is a **passthrough**. Iterating a `PrefetchIterator` by hand yields the
underlying source's batches, in order, with no task and no channel, so it is a valid data source
anywhere one is expected and cannot leak a producer when something other than the training loop
iterates it. That is deliberate: an `iterate` that spawned a task would have nowhere to put the
teardown, which is the exact leak this file warns about. It is also what keeps `render` and
`predict(nitro, loader)` unaffected by the auto-wrap.

**Prefetch applies to every split**, and `auto_prefetch` wraps them all. The eval splits go through
[`eval_stream`](@ref), which pads a short final batch inside the producer so that the pad and the
transfer both overlap the previous batch's forward.
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
        # NOT merely a lower bound for sanity's sake. Ordered delivery holds finished batches until
        # the one it is waiting for arrives, and that batch is only ever in flight if the window is
        # at least as wide as the number of workers. Below that, every worker can be holding a batch
        # the consumer cannot yet accept while the one it needs has not been handed out: a deadlock
        # rather than a stall.
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

**The only way to decline prefetch**, and it is deliberately a marker rather than a number.

The framework wraps the `train` split in a [`PrefetchIterator`](@ref) at its own defaults, so a split
that must run its host data path inline on the training task says so with this. It is a visible,
greppable declaration that a reviewer will question; a numeric `prefetch_device_batches = 0` field on an
experiment reads as ordinary tuning, which is precisely how one model ran its entire data path inline
for weeks with nothing contradicting it.

`iterate` is a passthrough, exactly as `PrefetchIterator`'s is, so this is a valid data source
anywhere one is expected.
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

Unwrap a split, and read its settings.

`prefetch_device_batches` returns `0` for anything that is not a [`PrefetchIterator`](@ref), which is
the "this split does not stream" case rather than a count. **That `0` is an internal dispatch result
and never a user-settable value**: the constructor rejects anything below 1, and [`NoPrefetch`](@ref)
is the supported way to say it. Do not resurrect `device_batches = 0` as an input on the strength of
this.

Setup goes through `prefetch_source` for the schema probe and the contract checks, so a wrapped
source is validated as itself rather than through the wrapper.
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
# `true` for everything else, because every other path delivers in the source's order anyway: the
# inline path and the single-producer path both walk the source's own `iterate`.
prefetch_ordered(x) = true

"""
    ReactantNitro.auto_prefetch(collection) -> collection

Wrap **every** split in a [`PrefetchIterator`](@ref) at the framework's defaults, unless it is
already a `PrefetchIterator` or a [`NoPrefetch`](@ref).

**This is the change that addresses the footgun rather than only the symptom.** Before it, prefetch
was opt-in through a wrapper the user had to remember in `build_data`, one model's comment asserted
the framework was handling it, and for weeks nothing contradicted that. The default now has to be
declined rather than requested, and the resolved settings appear in the binding report.

**Every split, including the eval ones**, which was not always true here: while `run_eval` iterated
its split directly, a worker count on an eval split would have been a number in the report that
nothing used. [`eval_stream`](@ref) is what changed that, and the reason it exists is that evaluation
is the phase MORE likely to be starved, not less: a validation step is forward-only, so device time
per batch falls sharply while host time per batch does not move.

Called at setup **after** `derive`, the schema probe, and the contract checks, so all three see
exactly what `build_data` returned.
"""
auto_prefetch(collection::NamedTuple) =
    NamedTuple{keys(collection)}(map(_auto_wrap, values(collection)))

_auto_wrap(split::PrefetchIterator) = split
_auto_wrap(split::NoPrefetch) = split
_auto_wrap(split) = PrefetchIterator(split)

"""
    ReactantNitro.prefetch_config(split) -> (; device_batches, host_batches, workers, ordered, path)

The **resolved** prefetch settings for a split, which is not the same as the requested ones: a
`workers > 1` request over a source that does not implement the index-addressable trait resolves to
one producer.

`path` is one of:

  * `:fanout`: `workers` producers over [`batch_at`](@ref), delivered in the source's order.
  * `:fanout_unordered`: the same, with `ordered = false`, delivered as the workers finish.
  * `:single`: one producer over the source's own `iterate`, because `workers == 1` was asked for.
  * `:materialized`: one producer, because the source is a `Vector` whose batches `build_data`
    already built. Fan-out cannot help it, so this is **not** a case the setup warning is about.
  * `:single_no_trait`: one producer, because the source implements neither or only one of
    [`batch_at`](@ref) and [`begin_epoch!`](@ref). **This is the case the setup warning is about.**
  * `:inline`: no stream at all, meaning a [`NoPrefetch`](@ref) split or one setup did not wrap.
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

Whether a source's batches already exist, so that producing one costs nothing and `workers > 1`
would buy nothing either.

**`Vector` exactly, not `AbstractVector`**, and the narrowness is the point. The premise of the
index-addressable trait is that the expensive work happens inside `iterate`; for a `Vector` that
`build_data` filled, `getindex` is a pointer load and there is no host work for N producers to
spread. A custom `AbstractVector` can compute in its `getindex` and genuinely wants the trait, so
it is not covered here and still hears the warning.

This decides only whether setup WARNS. It never changes what runs: the trait is what enables
fan-out, and a `Vector` that implements it fans out like anything else.
"""
materialized_source(x) = x isa Vector

"""
    ReactantNitro.check_batch_at(source; values = false) -> nothing

Assert that [`batch_at`](@ref) agrees with the source's own `iterate`, batch for batch. **This is the
only thing that catches a `batch_at` whose index units are wrong**, and it needs no accelerator, so it
belongs in a model package's test suite or in a CPU gate.

One epoch's plan is used for both halves: the sequential pass is taken first (which calls
[`begin_epoch!`](@ref) through `iterate`) and the indexed pass follows with no re-plan between them.

`values = false` compares field names and per-field sizes, which is what a loader with stochastic
augmentation permits. `values = true` compares contents and is the strong form; run it on a
**deterministic** split, which for these models means validation.
"""
function check_batch_at(source; values::Bool = false)
    fanout_capable(source) || error(
        """
        ReactantNitro: `check_batch_at` was given a `$(typeof(source))`, which does not implement both
        halves of the index-addressable trait, so there is nothing to check. It needs `batch_at` AND
        `begin_epoch!`."""
    )
    # BEFORE the sequential pass, which matters for the two kinds of source in opposite ways. A
    # source that stores its own plan returns `nothing` here and is then re-planned by `collect`'s
    # own `iterate`, so the indexed pass below reads exactly the plan the sequential pass used,
    # which is what this check compared before plans existed. A source that RETURNS its plan gets
    # one drawn here while `collect` draws another, so `values = true` needs a deterministic
    # source; the error below says so.
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

The training loop's batch source, yielding `(host_batch, device_batch)` pairs. One loop body serves
both paths, which is why the pair is the element type: the loop needs the **host** batch for the
schema and shape checks and for a `:host`-residency `train_metrics`, and the **device** batch for
the gradient program.

Without a [`PrefetchIterator`](@ref) this is a lazy generator and the transfer happens inline,
exactly as it did before prefetch existed. With one it is a `Channel` of `device_batches` fed by a
spawned producer, so the transfer overlaps the previous step's compute.

**A producer that throws surfaces at the consumer.** `Channel`'s task form closes the channel with
the exception, and iteration on the consumer side rethrows it, so a failing loader stops the run with
its own error rather than hanging the loop waiting on a channel nobody will feed. That is Julia's own
behavior and is asserted rather than reimplemented.

**The checks stay on the consumer**, against the host half of the pair, rather than moving into the
producer. Ordering is then unchanged from the inline path, and an error is raised on the task the
user's stack trace is about. The cost is that a batch with a bad schema is transferred before the
check fires, which is one wasted transfer on a path that is about to raise.

## The three paths, and which one a split takes

  1. **Fan-out** ([`PrefetchStream`](@ref)), when `workers > 1` and the source is
     [`fanout_capable`](@ref). N workers build host batches, one task transfers, the consumer sees the
     same pair type as always.
  2. **One producer**, a `Channel` of `device_batches` fed by a single task iterating the source. What this
     function used to do unconditionally, and the fallback when the source does not implement the
     index-addressable trait.
  3. **Inline**, a lazy generator, when `prefetch_device_batches == 0`: a [`NoPrefetch`](@ref) split.

## `prepare` is what makes one pipeline serve both phases

The producer's job is "host batch in, device payload out", and the only difference between training
and evaluation is what that payload is: training transfers the batch as it stands, evaluation pads a
short final batch to the compiled width first. Passing the step as a closure keeps ONE fan-out, one
credit window, one joiner, and one teardown, rather than a second copy of the machinery that the
tests would have to cover twice and that would drift the first time either is fixed.

The three-argument form is the training one and is what every existing caller writes;
[`eval_stream`](@ref) supplies the evaluation closure.
"""
batch_stream(split, routing, mesh = nothing) =
    prepared_stream(split, b -> to_device_batch(b, routing, mesh))

"""
    ReactantNitro.prepared_stream(split, prepare) -> stream

The shared pipeline behind [`batch_stream`](@ref) and [`eval_stream`](@ref), parameterized by the
step that turns one host batch into the device payload its phase wants.

**A distinct name rather than a two-argument `batch_stream` method.** `batch_stream(split, routing)`
already means "training, on the default mesh", and a `(split, prepare)` method has the identical
signature, so defining both silently replaces one with the other and every existing two-argument call
starts trying to call a routing `NamedTuple`.
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

[`batch_stream`](@ref) for an evaluation pass: the producer pads a short final batch up to
`batch_size` and transfers the padded one, so the pad and the H2D copy both overlap the previous
batch's forward instead of running between them.

**`n_real` is deliberately NOT carried through the stream.** It is `batch_size_of(batch, xfer)`, a
pure function of the unpadded host batch that the consumer already receives, so recomputing it there
costs one `size` read and cannot disagree with what the producer padded to. Carrying it would widen
the pair every stage of the pipeline passes around, for one consumer.

**Why evaluation wants this at all, given it is the cheaper phase:** it is the cheaper phase on the
DEVICE. A validation step is forward-only, so device time per batch falls sharply while host time per
batch does not move at all, which makes the host:device ratio worse than training's, not better.
"""
eval_stream(split, xfer, batch_size::Integer, mesh = nothing) = prepared_stream(
    split, b -> to_device_batch(first(pad_batch(b, batch_size, xfer)), xfer, mesh)
)

"""
    ReactantNitro.PrefetchStream

The fan-out, as one object so [`close_stream!`](@ref) has something to dispatch on and one place to
stop every task. Iterating it yields the same `(host, device)` pairs the other two paths yield, which
is what keeps the training loop's body identical across all three.

```
coordinator  ->  Channel{Int}(workers)             the batch indices 1:n
   N workers ->  Channel{Tuple{Int,Any}}(workers)  HOST batches, with the index that produced them
      1 transfer task -> Channel{Tuple{Any,Any}}(device_batches)  (host, device) pairs
         the training loop

   credits  <-  Channel{Nothing}(host_batches)     one token per host batch, returned on emit
```

`credits` is `host_batches` made literal. The coordinator takes a token before handing out a job and
the transfer stage returns one after emitting a batch, so at most that many host batches exist at
once across the workers, `hostch`, and the reorder buffer together. It is a
`Channel` rather than a `Semaphore` for one reason: teardown closes it, and a coordinator blocked on
`take!` of a closed channel unwinds into its own `finally`, whereas one blocked on a semaphore would
be a leaked task with no way to reach it.

`marks` is the exactly-once ledger: one byte per batch index, written by the worker that took that
job. `check_prefetch_delivery` asserts every byte is set at the end of an epoch that ran to
completion. It is `Vector{UInt8}` rather than a `BitVector` because distinct bytes are independent
memory and distinct bits of one word are not, and N workers write concurrently.
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

# The error-propagation shape here is a prior framework's, reused rather than reinvented: its data
# path carried two comments that each recorded a real deadlock in production, and a hang is the
# failure mode this path is most able to produce.
function fanout_stream(
        src, workers::Int, device_batches::Int, host_batches::Int, prepare, ordered::Bool = true
    )
    token_before = epoch_token(src)
    # BEFORE `length`, and before any job exists. A source whose plan changes its batch count is
    # correct only if the count is read after the re-plan.
    #
    # `plan` is `nothing` for a source that stores its own, and the epoch's plan object for one that
    # cannot. It is held here, for exactly this epoch, and handed to every worker: the framework
    # owning it is what lets an immutable declaration like `MLUtils.DataLoader` fan out at all, and
    # what keeps N workers reading one immutable object instead of racing on a source's fields.
    plan = begin_epoch!(src)
    check_epoch_advanced(src, token_before)
    n = length(src)
    n > 0 || error("ReactantNitro: the training split reports `length == $n` after `begin_epoch!`; \
                    an epoch must have at least one batch.")

    jobs = Channel{Int}(workers)
    hostch = Channel{Tuple{Int, Any}}(workers)
    devch = Channel{Tuple{Any, Any}}(device_batches)
    marks = zeros(UInt8, n)

    # `host_batches` made literal. One token per batch allowed to exist on the host: the coordinator
    # takes one before handing out a job and the transfer stage returns one after emitting, so the
    # ceiling covers the producers' hands, `hostch`, and the reorder buffer together.
    #
    # BOTH DELIVERY MODES take the window, so the knob means the same thing either way. Unordered
    # delivery would be bounded anyway, by `hostch` plus one batch per worker, but leaving it to fall
    # out of two unrelated capacities is how a documented number stops matching the code.
    credits = Channel{Nothing}(host_batches)
    for _ in 1:min(host_batches, n)
        put!(credits, nothing)
    end

    # One shared job channel rather than a per-worker stride, and that is the point: exactly-once
    # delivery is then a property of `Channel` (an item is taken by exactly one taker) instead of a
    # property of index arithmetic somebody has to get right.
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
            # Marked before the hand-off, so a batch is on the ledger even if teardown races the
            # `put!`. The ledger is about delivery, and this is the point of no return for it.
            marks[i] = 0x01
            put!(hostch, (i, b))
        end
    catch e
        # A worker that dies before consuming enough jobs to keep the coordinator moving leaves it
        # blocked forever in `put!` once `jobs` is full. Closing WITH the exception is what unblocks
        # it, and what stops the siblings' `for i in jobs`.
        close_quiet!(jobs, e)
        rethrow()
    end
    return nothing
end

"""
    ReactantNitro.planned_batch_at(source, i, plan) -> batch

The framework's single call site for [`batch_at`](@ref), and the only place the two shapes of it are
told apart.

`nothing` means the source stores its own plan, so the two-argument method is the one it wrote.
Anything else is the plan [`begin_epoch!`](@ref) returned, which goes to the three-argument method.
Keeping the choice here rather than in a forwarding method is what leaves
[`fanout_capable`](@ref) able to detect a three-argument implementation at all.
"""
planned_batch_at(source, i::Integer, ::Nothing) = batch_at(source, i)
planned_batch_at(source, i::Integer, plan) = batch_at(source, i, plan)

# `bind` is deliberately not used anywhere in this pipeline: its `close_chnl_on_taskdone` returns
# early while the channel `isready`, so error propagation would stall behind any buffered item. Every
# channel here is closed explicitly.
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
    # The reorder buffer. Keyed on the batch index the worker carried through `hostch`, which the
    # unordered path simply discards. It holds HOST batches, which are ordinary Julia arrays, so
    # unlike the device channel it needs no explicit free on teardown.
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
            # `haskey` rather than a `nothing` sentinel: `prefetch_worker` already refuses a
            # `nothing` batch, but a buffer that encodes "absent" as a value a batch could take is
            # the kind of thing that stops being true later.
            while haskey(pending, next)
                hb = pop!(pending, next)
                put!(devch, (hb, prepare(hb)))
                next += 1
                # The credit goes back only once the batch has LEFT, which is what makes the window
                # bound what exists rather than what has been started.
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

Hand one job credit back to the coordinator, and **never raise doing it**.

The one way this fails is teardown: [`close_stream!`](@ref) closes `credits` to unblock a
coordinator parked on `take!`, and a `put!` racing that close throws `InvalidStateException`. Inside
the transfer loop that exception would be caught as the epoch's error and close the device channel
with it, replacing whatever real failure the teardown was already unwinding. Same reasoning as
[`drain_and_free!`](@ref)'s `try`, and the same narrow scope: a credit nobody will ever take again.
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

The exactly-once assertion, and it is cheap enough to leave on: one byte per batch, checked once
per epoch.

**Call it only on an epoch that ran to completion.** An early exit (`request_stop!`, a non-finite loss,
an error) leaves the ledger legitimately partial, and asserting there would turn a clean stop into a
spurious failure.

What it catches that [`check_epoch_length`](@ref) cannot: a **duplicate paired with a drop**, where the
right number of batches arrived but one index twice and another never. What neither catches is a
`batch_at` whose index mapping is wrong, because that produces well-shaped batches at every index;
[`check_batch_at`](@ref) is for that.

A no-op on every other stream type, since the inline and single-producer paths deliver in the source's
own order by construction.
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

Assert that the fan-out's one [`begin_epoch!`](@ref) call advanced the source by exactly one epoch, as
reported by [`epoch_token`](@ref). Declined, silently, by a source that leaves `epoch_token` at its
`nothing` default.

This is the guard on the failure mode that worries this file most: the epoch re-plan used to live in
`Base.iterate`'s initialization, the fan-out does not call `iterate` at all, and an epoch trained on
the previous epoch's plan looks exactly like a healthy one. A missing call and a double call are both
caught here; a double call is not hypothetical, since a loader whose augmentation is seeded from an
epoch counter silently skips a generation of it.
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

The cleanup half, and the reason [`batch_stream`](@ref)'s consumer runs inside a `try`/`finally`:
**early exit must stop the producer and free the device buffers it is holding, or it leaks a task and
`device_batches x batch` of device memory.** An early exit is not exotic here: `request_stop!` from a monitor,
a non-finite loss, and any error mid-epoch all leave the loop with a full channel behind it.

`close` is what stops the producer: a blocked `put!` on a closed channel raises, which ends the task.
The buffers already in the channel are then **freed explicitly**, since they are the ones nothing
else has ever held.

A generator has neither, so this is a no-op on the inline path.

**On the fan-out it must stop several tasks and it must not block.** Closing each channel is what
unwinds each stage: the workers leave `for i in jobs`, a worker blocked in `put!` on the host channel
raises, and the transfer task blocked in `put!` on the device channel raises. It deliberately does
**not** `wait` on the tasks, because a worker in the middle of a network round trip would hold teardown
for seconds; it will raise on its next `put!` and exit, and nothing it holds is device memory. That is
the third reason the transfer lives on its own task: **only one task in the pipeline ever holds a
device buffer, and it is the easiest one to stop.**
"""
close_stream!(stream) = nothing

function close_stream!(ch::Channel)
    close_quiet!(ch)
    drain_and_free!(ch)
    return nothing
end

"""
    ReactantNitro.drain_and_free!(ch::Channel) -> nothing

Free the device halves still sitting in a closed stream's channel, and **never raise while doing it**.

`isready` rather than iterating, because iteration on a channel closed with an exception would rethrow
the producer's error, and this runs from a `finally` that is usually already unwinding one.

**The `try` is not defensive noise, it is a bug fix.** `isready` is `n_avail(c) > 0` and Julia's
`n_avail` counts `length(c.data) + length(c.cond_put.waitq)`, so a **blocked putter** makes `isready`
answer true with the buffer empty. `take!` then finds no data, sees the channel closed, and throws
`InvalidStateException`, from inside a `finally`, which **replaces the caller's real exception with
"Channel is closed."** That is exactly the state teardown runs in: a full channel and a producer
blocked on `put!`. It was latent only because the channel path used to require an explicit
`PrefetchIterator`; making prefetch the default made every early exit take it, and the framework's own
own non-finite-loss test started reporting a channel complaint instead of the divergence message.

Freeing is best-effort by nature (it is an optimization over the GC, per `free_batch!`), and masking a
user's error to finish it would be strictly worse than skipping one batch's buffers.
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
    # Host batches are ordinary Julia arrays and need no explicit free; only the device halves sitting
    # in the device channel are buffers nothing else has ever held.
    drain_and_free!(s.devch)
    return nothing
end

"""
    ReactantNitro.free_batch!(batch) -> nothing

Release a device batch's buffers **now** rather than at the GC's convenience.

The pointer is set to `C_NULL` after the free, and that is not tidiness: `XLA.free_buffer` is a
no-op on a null pointer and a **double free** on a live one, and `ConcretePJRTArray`'s buffers carry
a finalizer that will free them again at GC time. Nulling is also what makes a use-after-free loud
instead of silent: this stack asserts `buffer.buffer !== C_NULL` on readback, so a freed batch
that something still reads raises there instead of returning whatever the allocator has since put
in that memory.

**Only buffers the prefetch owns are freed**, which is the set still sitting in the channel at
teardown. A batch that has been handed to a compiled program is NOT freed by the loop: XLA execution
is asynchronous, the executable holds its inputs for the duration, and freeing at the point the Julia
call returns would race it. Those batches are dropped and reclaimed by the finalizer exactly as they
were before prefetch existed. The rule "explicit rather than left to the GC" is about the resident
`device_batches x batch` this iterator introduces, and that is what this frees.
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
