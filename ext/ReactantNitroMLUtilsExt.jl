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
    ReactantNitro.check_source_options(dl::MLUtils.DataLoader, name::Symbol) -> nothing

The two `DataLoader` options that do not survive a prefetched pipeline, checked at setup rather than
discovered in a run's numbers.

**`buffer = true` is refused.** It allocates one batch and reuses it through `getobs!`, so every
batch the loader yields is the same memory. The framework holds several batches in flight at once,
so the producer overwrites a batch that is still queued for its device transfer. That is silent
corruption rather than an error, and it is invisible in a loss curve.

**`parallel = true` is a warning, not an error.** MLUtils runs its own worker threads, which is a
second, uncoordinated fan-out, and its own documentation says it breaks ordering guarantees. That
contradicts the framework's ordered delivery without either side being able to detect it. It stays
legal because it is a real way to get host concurrency from a loader that has not opted into the
trait, and because nothing about it is unsafe, only unreproducible.
"""
function ReactantNitro.check_source_options(dl::DataLoader, name::Symbol)
    dl.buffer === false || error(
        """
        ReactantNitro: the `$name` split is an `MLUtils.DataLoader` with `buffer = true`, which
        cannot be prefetched. A buffered loader reuses ONE batch through `getobs!`, and this
        framework keeps several batches in flight, so the producer overwrites a batch that has not
        been transferred to the device yet. Nothing raises and the loss curve still looks plausible.
        Drop `buffer` (the default is `false`), or wrap the split in `NoPrefetch` to run its data
        path inline on the training task, where one batch is consumed before the next is built."""
    )
    dl.parallel && @warn """
    ReactantNitro: the `$name` split is an `MLUtils.DataLoader` with `parallel = true`, which runs \
    MLUtils' own worker threads underneath this framework's pipeline. MLUtils documents that it \
    breaks ordering guarantees, so batches arrive in an order neither side controls and a fixed \
    seed no longer reproduces a run bitwise, whatever `ordered` says.
    Drop `parallel` to let the framework's fan-out supply the concurrency, which preserves the \
    source's order, or keep it and treat the run as unordered.""" maxlog = 1
    return nothing
end

end # module
