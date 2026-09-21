# ReactantNitroMLUtilsExt.jl
#
# The index-addressable trait for `MLUtils.DataLoader`, so the loader everyone writes fans out over
# every thread. It reads the loader's declaration rather than iterating it, rebuilding the epoch's
# plan from the documented fields (`data`, `batchsize`, `partial`, `collate`, `shuffle`, `rng`) with
# MLUtils' own exported `ObsView`, `shuffleobs` and `BatchView`, then indexing them. `DataLoader`
# is immutable, so `begin_epoch!` returns the plan rather than storing it, which is what the
# three-argument `batch_at` exists for; the file is two pure methods with no side table to collide
# when two handles share one loader. The one thing that must keep matching upstream is the order of
# operations: shuffle the observations, then batch them.

module ReactantNitroMLUtilsExt

import ReactantNitro
import MLUtils

using MLUtils: BatchView, DataLoader, ObsView, getobs, numobs, shuffleobs

"""
    ReactantNitro.begin_epoch!(dl::MLUtils.DataLoader) -> plan

This epoch's plan: an `ObsView`, permuted when `shuffle = true` from the loader's own `rng` as
`DataLoader`'s `iterate` does at the top of each pass, wrapped in a `BatchView` when the loader
batches. Returned rather than stored, since `DataLoader` is immutable.
"""
function ReactantNitro.begin_epoch!(dl::DataLoader)
    obs = ObsView(dl.data, collect(1:numobs(dl.data)))
    shuffled = dl.shuffle ? shuffleobs(dl.rng, obs) : obs
    dl.batchsize > 0 || return shuffled
    return BatchView(shuffled; dl.batchsize, dl.partial, dl.collate)
end

"""
    ReactantNitro.batch_at(dl::MLUtils.DataLoader, i::Integer, plan) -> batch

Batch `i` of this epoch's plan through MLUtils' own `getobs`. `plan` is untyped on purpose, since
[`ReactantNitro.fanout_capable`](@ref) detects the three-argument method by asking whether it
accepts any plan.
"""
ReactantNitro.batch_at(dl::DataLoader, i::Integer, plan) = getobs(plan, i)

"""
    ReactantNitro.check_source_options(dl::MLUtils.DataLoader, name::Symbol, cfg) -> nothing

`partial` on both kinds of split, plus `buffer` and `parallel` judged against the path the split
resolved to. Both of those live inside `DataLoader`'s `iterate`, which the fan-out never calls, so
on the fan-out path they are inert (and `buffer = true` is said to be); on the single-producer
path they are live. `buffer = true` is refused there, since the framework reads ahead and the
loader would refill its one batch while the previous is still queued for transfer, silently;
`parallel = true` warns, since MLUtils documents that it breaks ordering.

`partial = true` on the `train` split is refused here, before anything compiles, rather than at the
last batch of the first epoch, and only when the observation count does not divide by the batch
size. `partial = false` on an eval split warns: the framework pads and slices, so dropping silently
shrinks the set a metric is computed over. It stays legal for a deliberately truncated set or a
model that mixes across the batch axis in test mode.
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

# Exact rather than heuristic: both numbers are fields on the loader. `batchsize <= 0` is MLUtils'
# one-observation-per-batch mode, where none can be short.
function _check_partial(dl::DataLoader, name::Symbol)
    dl.batchsize > 0 || return nothing
    n = numobs(dl.data)
    short = n % dl.batchsize
    # When the count divides, `partial` picks between two identical behaviours.
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
