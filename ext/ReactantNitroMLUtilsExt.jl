# ReactantNitroMLUtilsExt.jl
#
# The index-addressable trait for `MLUtils.DataLoader`, so the loader everyone already writes fans
# out over every thread without the user doing anything.
#
# ── What this is, in one sentence ────────────────────────────────────────────────────
#
# It reads the loader's DECLARATION rather than iterating it: `DataLoader` is a description of a
# dataset (the data, a batch size, whether to shuffle, an RNG), and this rebuilds the epoch's plan
# from those fields with MLUtils' own `ObsView`, `shuffleobs` and `BatchView`. Nothing here
# reimplements batching, shuffling or collation; it asks MLUtils for the same objects
# `DataLoader.iterate` would have built, and then indexes them.
#
# ── Why it needs no state, and why that is the whole point ───────────────────────────
#
# `struct DataLoader` is immutable, so `begin_epoch!` has nowhere to stash an epoch's shuffled view.
# That is exactly what the three-argument `batch_at` exists for: `begin_epoch!` RETURNS the plan,
# the framework holds it for that one epoch and hands it to every worker, and this file stays two
# pure methods with no side table, no wrapper type, and nothing to collide when two handles share
# one loader (`Nitro(e; data = n.data)` is a documented pattern, and a side table would break it).
#
# It is also the right shape for a fan-out on its own terms: N workers read one immutable plan
# rather than racing on a source's fields.
#
# ── Reconstructed from PUBLIC fields, deliberately ───────────────────────────────────
#
# `DataLoader` keeps its already-wrapped view in `_data`, and `_shuffledata` is MLUtils-internal.
# Using either would tie this file to MLUtils' privates. `data`, `batchsize`, `partial`, `collate`,
# `shuffle` and `rng` are all documented fields, and `ObsView`, `BatchView`, `shuffleobs`, `numobs`
# and `getobs` are all exported, so the plan below is rebuilt entirely from the supported surface.
# The one thing that must keep matching upstream is the ORDER of operations, shuffle the
# observations and then batch them, which is what `_shuffledata` does and what makes a batch a
# contiguous window of the permuted stream rather than a permutation of fixed batches.

module ReactantNitroMLUtilsExt

import ReactantNitro
import MLUtils

using MLUtils: BatchView, DataLoader, ObsView, getobs, numobs, shuffleobs

"""
    ReactantNitro.begin_epoch!(dl::MLUtils.DataLoader) -> plan

This epoch's plan: an `ObsView` permuted when `shuffle = true`, wrapped in a `BatchView` when the
loader batches.

**Returned rather than stored**, because `DataLoader` is immutable. The framework holds it for the
epoch and hands it to [`ReactantNitro.batch_at`](@ref).

`shuffle = true` draws a NEW permutation here, once per epoch, from the loader's own `rng`, which is
the same thing `DataLoader`'s own `iterate` does at the top of each pass. That is the half of the
trait a `batch_at` alone cannot supply: the fan-out never calls `iterate`, so without this every
epoch after the first would train on the first epoch's permutation.
"""
function ReactantNitro.begin_epoch!(dl::DataLoader)
    obs = ObsView(dl.data, collect(1:numobs(dl.data)))
    shuffled = dl.shuffle ? shuffleobs(dl.rng, obs) : obs
    dl.batchsize > 0 || return shuffled
    return BatchView(shuffled; dl.batchsize, dl.partial, dl.collate)
end

"""
    ReactantNitro.batch_at(dl::MLUtils.DataLoader, i::Integer, plan) -> batch

Batch `i` of this epoch's plan, through MLUtils' own `getobs`, which is the same call
`DataLoader`'s unbuffered serial path makes.

`plan` is untyped on purpose: [`ReactantNitro.fanout_capable`](@ref) detects a three-argument method
by asking whether one accepts any plan at all, so annotating it here would hide it and the loader
would silently fall back to one producer.
"""
ReactantNitro.batch_at(dl::DataLoader, i::Integer, plan) = getobs(plan, i)

"""
    ReactantNitro.check_source_options(dl::MLUtils.DataLoader, name::Symbol, cfg) -> nothing

`partial` on the training split, plus `buffer` and `parallel` judged against the path the split
actually resolved to.

**Both options live inside `DataLoader`'s `Base.iterate`**, and that is the whole of it. `parallel`
selects which `iterate` method runs; `buffer` matters because the buffered `iterate` fills one
shared batch through `getobs!`. The trait in this file drives the loader by index instead, through
`getobs`, so on the FAN-OUT path neither option runs and neither can hurt anything. On the
single-producer path the framework iterates the source, so both are live. `workers = 1` is not
exotic: plain `julia` with no `-t` has one default thread and lands there.

**`buffer = true` is refused on the single-producer path**, because the framework reads ahead: the
loader would fill its one batch again while the previous one is still queued for its device
transfer, leaving the queued batch holding the wrong samples. Nothing raises and the loss curve
still looks plausible. On the fan-out path it is merely ignored, and said so, since someone who set
it to bound allocation should not be left believing it did something.

**`parallel = true` warns on the single-producer path** and is silent on the fan-out path. There it
is ignored and the user gets what they asked for anyway, from this framework's producers, in the
source's order. On the single-producer path it does run, and MLUtils documents that it breaks
ordering guarantees, so a fixed seed stops reproducing a run bitwise.

**`partial = false` on an EVAL split warns**, which is the mirror image. The framework pads a short
eval batch and slices the outputs back, so keeping it costs nothing and is the supported shape;
dropping it silently shrinks the set the metric is computed over, and a test accuracy quoted over
9,984 of 10,000 samples reads exactly like one over all of them. It stays legal because there are
two good reasons for it, a deliberately truncated set and a model that mixes across the batch axis
in test mode, for which dropping is how you avoid padding rather than tolerate it.

**`partial = true` on the `train` split is refused, and refused HERE**, before anything compiles.
The framework already rejects a short training batch, but it can only do so when one arrives, which
is the last batch of the first epoch: a compile and a full epoch of training after the mistake was
made. A `DataLoader` carries the answer in its fields, so the arithmetic is exact rather than a
guess, and `partial = true` over an observation count that happens to divide evenly is left alone
because no short batch will ever be produced.
"""
function ReactantNitro.check_source_options(dl::DataLoader, name::Symbol, cfg)
    _check_partial(dl, name)
    cfg.path === :inline && return nothing
    fanned = cfg.path === :fanout || cfg.path === :fanout_unordered
    if fanned
        dl.buffer === false || @warn """
        ReactantNitro: the `$name` split is an `MLUtils.DataLoader` with `buffer = true`, which has \
        NO EFFECT here. Buffering lives in the loader's `iterate`, and this split is driven by \
        index instead, so every batch is freshly allocated. Drop `buffer`; to bound host memory, \
        set `host_batches` on the split's `PrefetchIterator`.""" maxlog = 1
        return nothing
    end
    dl.buffer === false || error(
        """
        ReactantNitro: the `$name` split is an `MLUtils.DataLoader` with `buffer = true`, and it
        resolved to ONE producer, which iterates the loader. A buffered loader reuses one batch
        through `getobs!`, and this framework reads ahead, so the producer overwrites a batch that
        has not been transferred to the device yet. Nothing raises and the loss curve still looks
        plausible.
        Drop `buffer` (the default is `false`), or wrap the split in `NoPrefetch` to run its data
        path inline on the training task, where one batch is consumed before the next is built."""
    )
    dl.parallel && @warn """
    ReactantNitro: the `$name` split is an `MLUtils.DataLoader` with `parallel = true`, and it \
    resolved to ONE producer, so MLUtils' own worker threads are what build its batches. MLUtils \
    documents that they break ordering guarantees, so a fixed seed no longer reproduces a run \
    bitwise, whatever `ordered` says.
    Give the split more workers to use this framework's fan-out instead, which preserves the \
    source's order, or keep it and treat the run as unordered.""" maxlog = 1
    return nothing
end

# Exact rather than heuristic: a short final batch exists only when the observation count does not
# divide by the batch size, and both are fields on the loader. `batchsize <= 0` is MLUtils' iterate
# individual observations mode, where every batch is one observation and none can be short.
function _check_partial(dl::DataLoader, name::Symbol)
    dl.batchsize > 0 || return nothing
    n = numobs(dl.data)
    short = n % dl.batchsize
    # Nothing to say either way when the count divides: no batch is ever short, so `partial` picks
    # between two identical behaviours and the flag is inert.
    short == 0 && return nothing
    if name === :train
        dl.partial || return nothing
        error(
            """
            ReactantNitro: the `train` split is an `MLUtils.DataLoader` with `partial = true` over
            $n observations at a batch size of $(dl.batchsize), so its final batch is $short wide.
            A training loader must DROP its partial final batch: the compiled program has a fixed
            shape and training cannot pad, because in train mode BatchNorm normalizes over the
            batch, so padded rows change the real rows' outputs and no downstream slice undoes it.
            Pass `partial = false`. With a shuffled loader this costs nothing over a run, since a
            different tail is dropped each epoch. Eval splits should KEEP theirs; the framework pads
            and slices those."""
        )
    end
    dl.partial && return nothing
    @warn """
    ReactantNitro: the `$name` split is an `MLUtils.DataLoader` with `partial = false` over $n \
    observations at a batch size of $(dl.batchsize), so $short samples are dropped and every metric \
    on this split is computed over $(n - short) of them.
    Eval splits do not need to drop: the framework pads a short final batch, runs the one compiled \
    program, and slices the outputs back before `metrics` sees them, so padding is invisible. Pass \
    `partial = true` unless you meant it. Two cases where dropping IS the right call: evaluating on \
    a deliberately truncated set, and a model that mixes across the batch axis in test mode, for \
    which dropping is how you avoid padding rather than tolerate it.""" maxlog = 1
    return nothing
end

end # module
