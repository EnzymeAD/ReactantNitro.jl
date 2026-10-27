# Data loading

[`build_data`](@ref) returns a NamedTuple of splits. For most experiments each split is an
`MLUtils.DataLoader`, and that is all this page's first section needs. The later sections cover
loading in worker processes, sharing a large dataset with those workers, how the framework drives
a split, and the interface for writing a source of your own.

## The DataLoader

A dataset is any type with `numobs` and `getobs`. `getobs` receives a vector of indices and returns
the batch as a `NamedTuple`, with the batch dimension last:

```julia
using MLUtils, Random

struct Images
    pixels::Array{UInt8, 4}     # (W, H, C, N)
    labels::Vector{Int}
end

MLUtils.numobs(d::Images) = length(d.labels)
MLUtils.getobs(d::Images, idxs::AbstractVector{<:Integer}) =
    (; img = d.pixels[:, :, :, idxs], label = d.labels[idxs])

function ReactantNitro.build_data(e::MyExperiment, dist)
    train, val = load_images(e.data_dir, "train"), load_images(e.data_dir, "val")
    return (;
        train = DataLoader(train; batchsize = e.batch_size, shuffle = true, partial = false,
                           rng = Xoshiro(e.seed)),
        val = DataLoader(val; batchsize = e.batch_size, partial = true),
    )
end
```

The field names are what the hooks route by, so `img` and `label` here are whatever `forward`,
`loss` and `metrics` ask for.

The framework loads these in the training process, fanned out over threads: each producer task
builds one batch with `getobs`, and one transfer task copies finished batches to the device while
the current step runs. A few things follow from that:

- **`julia -t N` sets the data throughput.** The number of producers defaults to the size of the
  default thread pool. Each epoch logs `data_wait_frac`, the fraction of the epoch the training
  loop spent waiting for a batch, and warns above 0.25; when it is high, raise `-t` first.
- **Runs reproduce.** Batches are delivered in the loader's order whatever the thread count, so a
  fixed `rng` reproduces a run bitwise. Random augmentation inside `getobs` keeps that only if it
  is seeded from the indices, for example `Xoshiro(hash(idxs))`, since any producer may build any
  batch.
- **The training split drops its short final batch** (`partial = false`), because the compiled
  program has a fixed shape. An eval split keeps it (`partial = true`): the framework pads it and
  slices the outputs back, so every sample counts in the metrics. Setup checks both.
- **`parallel` and `buffer` are not needed.** The framework does the parallelism, and both options
  have no effect on this path; setup warns if they are set.

To change the defaults, return the split already wrapped:
`PrefetchIterator(loader; workers = 8)`. [How a split is driven](@ref) lists
the settings.

## When to load in worker processes

Loading in the training process has one cost: `getobs` allocates there. Decoding, augmentation
and collation create garbage every batch, and Julia's garbage collector stops every thread to
collect it, including the task that drives the device. While it collects, no step is launched and
no batch is transferred. For a dataset whose batches are cheap to build, the pauses are small next
to a training step and nothing changes. When loading allocates heavily, such as decoding large
images or building many crops per sample, the pauses add up and the device sits idle during them.

Pauses show up as slower or uneven steps rather than as `data_wait_frac`. To check, run an epoch
with `GC.enable_logging(true)`, which prints each collection and its pause, or compare the `gctime`
that `@timed` reports for an epoch with its total time. If collections take a noticeable fraction
of the epoch, move loading out of the process with `num_workers`. The other reason to do so is a
`getobs` that does not scale over threads at all, such as a `PythonCall`-backed dataset, whose
reads share one interpreter lock.

Worker processes are not free: each starts its own Julia and loads the dataset's packages, setup
takes longer, and each holds its own copy of whatever the dataset contains (see
[Sharing data with workers](@ref)). If the device is already busy, they buy
nothing.

## Loading with num_workers

```julia
train = PrefetchIterator(
    DataLoader(train_images; batchsize = e.batch_size, shuffle = true, partial = false,
               rng = Xoshiro(e.seed), num_workers = 4);
    ordered = false)
```

With `num_workers > 0`, MLUtils starts worker processes with `Distributed`, sends each one the
dataset, and runs `getobs` there; on Unix the finished batches come back through shared memory.
The allocations of loading stay in the workers. The framework runs one producer over the loader's
own `iterate`, which drives the workers, so `-t` no longer sets the data throughput;
`num_workers` does.

!!! warning "Revise reaches the workers of a running job"
    In the training process, world age keeps a run on the code it started with: an edit that
    Revise applies during a run takes effect from the next run. Worker processes do not have that
    protection. Distributed runs each batch request on a worker with `invokelatest`, and Revise
    applies every revision to the live workers, so saving an edit to `getobs`, an augmentation or
    a collate function changes the batches of the job that is already training, from the next
    batch on, while `forward` and `loss` stay on the old code. In a [Kaimon](kaimon.md) session,
    Revise runs before every evaluation, so checking on a run is enough to apply a pending edit.
    Do not edit the data pipeline during a run that uses `num_workers`. The fix is tracked in
    [MLUtils.jl#267](https://github.com/JuliaML/MLUtils.jl/issues/267).

Three things differ from the in-process loader:

- **Runs no longer reproduce bitwise.** MLUtils delivers batches as the workers finish them, so the
  order within an epoch varies. Setup refuses the split unless it says so, with `ordered = false`
  on its `PrefetchIterator`. Every batch is still seen once per epoch.
- **Define the dataset type in a package.** Each worker imports the packages the dataset's types
  come from. A package that does not load Reactant keeps the workers small; one that does makes
  every worker load Reactant too.
- **The workers outlive the loader.** MLUtils keeps them warm across loaders and epochs and ends
  them when Julia exits. A [Kaimon](kaimon.md) session restart replaces the process without
  exiting Julia, and the workers survive it with their memory; call
  `MLUtils.close_dataloader_pool()` before restarting a session that used `num_workers`.

## Sharing data with workers

MLUtils sends the dataset to each worker once, with Julia's `Serialization`, and every array it
reaches is written out in full. A memory-mapped array is no exception: the worker rebuilds it as
an ordinary array, so each worker holds a private copy of the whole dataset rather than sharing the
file's pages. A new loader over the same dataset sends it again to the warm workers, and the
earlier copy is not freed. On a 13.5 GB memory-mapped image file, four workers held 14 GB of
private memory each, and about 30 GB each after a second handle built its loaders.

The fix is to send a handle and reopen the data on arrival. A dataset that keeps the file's path
next to its mapping can serialize the path and map the file again in `deserialize`, so the main
process and every worker read the same page cache:

```julia
using Mmap, Serialization
using Serialization: AbstractSerializer

struct Tiles
    images::Array{UInt8, 4}     # memory-mapped from `path`
    path::String
    labels::Vector{Int}
end

map_tiles(path, dims) = open(io -> Mmap.mmap(io, Array{UInt8, 4}, dims), path)

function Serialization.serialize(s::AbstractSerializer, d::Tiles)
    Serialization.writetag(s.io, Serialization.OBJECT_TAG)
    serialize(s, typeof(d))
    serialize(s, (d.path, size(d.images), d.labels))
end

function Serialization.deserialize(s::AbstractSerializer, ::Type{Tiles})
    path, dims, labels = deserialize(s)
    return Tiles(map_tiles(path, dims), path, labels)
end
```

With that method on the same 13.5 GB file, each worker received 720 KB (the labels) instead of
13.5 GB, and held 0.7 GB of private memory, most of it the Julia runtime and loaded packages.
The pages a worker reads count as shared, file-backed memory that the kernel can reclaim. Define
the methods in the package that defines the dataset, so each worker has them as soon as it imports
that package; methods defined in `Main` do not reach the workers intact.

The same pattern covers other sources. Anything that can be reopened from a description, such as
an HDF5 or Zarr file, a directory of images or a database, is cheaper to send as that description
than as its contents: reopen it in `deserialize`, or leave the field empty and open it on the
first `getobs` in each process. A resource that cannot be serialized at all, such as an open file
handle or a socket, needs the same treatment to work. Small per-sample metadata, such as labels,
can travel as it is.

## How a split is driven

The framework wraps every split in a [`PrefetchIterator`](@ref): producer tasks build host
batches, one transfer task copies them to the device, and a channel of `device_batches` feeds the
training loop. Eval splits are wrapped too. How many producers run, and what they call, depends on
the source:

| Path | When | What runs |
|---|---|---|
| `:fanout` | the source implements [`batch_at`](@ref) and [`begin_epoch!`](@ref) | `workers` producers call `batch_at`, delivered in order |
| `:fanout_unordered` | the same, with `ordered = false` | delivered as producers finish |
| `:delegated` | [`ReactantNitro.delegates_parallelism`](@ref) is `true`, as for `num_workers` | one producer over the source's own `iterate` |
| `:single` | `workers = 1` | one producer over `iterate` |
| `:materialized` | the split is a `Vector` of batches | one producer; there is nothing to build |
| `:single_no_trait` | none of the above | one producer over `iterate`, and setup warns for `train` |
| `:inline` | the split is wrapped in [`NoPrefetch`](@ref) | no producer; the training task iterates the source |

A plain `DataLoader` takes `:fanout`: the framework's MLUtils extension reads the loader's
settings and rebuilds each epoch's plan with MLUtils' own `shuffleobs` and `BatchView`, then calls
`getobs` per batch; the fan-out never calls the loader's `iterate`. `ReactantNitro.prefetch_config(split)`
reports the path a split resolved to. The `PrefetchIterator` settings are:

- `workers`: producer tasks, by default the default thread pool's size.
- `device_batches`: batches staged on the device, by default 1. Resident device memory is
  `device_batches + 2` batches.
- `host_batches`: batches on the host at once, by default `2 * workers`.
- `ordered`: delivery in source order, by default `true`.

## A source of your own

Any iterator works as a split, but only a source with the trait gets more than one producer. The
trait is a handful of methods, extended qualified:

- [`begin_epoch!`](@ref)`(source)` plans the next epoch: the order, and any per-epoch random
  draws. The fan-out calls it once per epoch before requesting a batch. Return `nothing` if the
  source stores the plan, or return the plan to have it passed to every `batch_at` that epoch.
- [`batch_at`](@ref)`(source, i)` or `batch_at(source, i, plan)` builds batch `i` of the epoch.
  Many producers call it at once, so it must depend only on `i` and the plan, and keep per-task
  scratch in a `TaskLocalValue`. `i` is a batch index, not a sample offset.
- [`ReactantNitro.epoch_token`](@ref)`(source)` is an optional counter that `begin_epoch!`
  increments, so the fan-out can catch a missed or doubled re-plan.
- [`ReactantNitro.release!`](@ref)`(source)` closes whatever the source holds open. The framework
  calls it once per handle when the run reaches `Done` or `Failed`.
- [`ReactantNitro.check_source_options`](@ref)`(source, name, cfg)` rejects or warns about options
  the resolved path cannot honour, before anything compiles.
- [`ReactantNitro.delegates_parallelism`](@ref)`(source)` returns `true` for a source that
  parallelizes its own `iterate`, as `num_workers` does, so the framework runs one producer over it.

The fan-out never calls `iterate`, so a source that also iterates by hand should call
`begin_epoch!` from its `iterate` rather than re-plan there. [`ReactantNitro.check_batch_at`](@ref)
compares `batch_at` against `iterate` batch for batch, needs no accelerator, and belongs in the
package's tests; it is the one check that catches a `batch_at` indexing samples instead of batches.

A package that owns a loader type can ship these methods as an extension with ReactantNitro as a
weak dependency, the way the framework's MLUtils extension does for `DataLoader`.

### Producers as clients of an external pipeline

The trait does not require the batches to be built in the training process. A source can be a thin
client of a data pipeline that runs elsewhere: a separate server process that decodes, augments
and collates, possibly written in another language or shared between several runs. Each producer
task then spends its time waiting on a request, so the fan-out becomes request concurrency:

- **`begin_epoch!` plans the epoch on the client.** It draws the order and makes every per-sample
  random choice a pure function of `(seed, epoch, batch, sample)`, so a batch is the same whichever
  producer asks for it, and a fixed seed still reproduces the run with ordered delivery.
- **`batch_at` sends one batch's request and waits for the reply.** The request carries the sample
  indices and their draws; the reply carries the collated arrays, through shared memory when the
  server is on the same host, so the bulk bytes never cross a socket. `workers` producers keep
  `workers` requests in flight, so `-t` sizes the concurrency, and the server parallelizes across
  requests rather than within one.
- **`release!` shuts the pipeline down** at `Done` or `Failed`, so no runtime field holds a handle
  and no run leaves a server behind.
- **`check_source_options` refuses what the pipeline cannot do**, such as a truncation that would
  drop batches from an eval split's metrics.

```julia
# `PipelineClient`, `request` and `fetch_batch` stand for your pipeline's client API.
mutable struct RemoteLoader
    client::PipelineClient
    n::Int                    # samples in the split
    batch_size::Int
    seed::UInt64
    epoch::Int
    order::Vector{Int}
end

Base.length(dl::RemoteLoader) = dl.n ÷ dl.batch_size

function ReactantNitro.begin_epoch!(dl::RemoteLoader)
    dl.epoch += 1
    dl.order = randperm(Xoshiro(hash((dl.seed, dl.epoch))), dl.n)
    return nothing
end

ReactantNitro.epoch_token(dl::RemoteLoader) = dl.epoch

function ReactantNitro.batch_at(dl::RemoteLoader, i::Integer)
    idxs = dl.order[(i - 1) * dl.batch_size .+ (1:dl.batch_size)]
    draws = [hash((dl.seed, dl.epoch, i, k)) for k in eachindex(idxs)]
    return fetch_batch(dl.client, request(idxs, draws))   # blocks this producer only
end

Base.iterate(dl::RemoteLoader, i = (ReactantNitro.begin_epoch!(dl); 1)) =
    i > length(dl) ? nothing : (ReactantNitro.batch_at(dl, i), i + 1)

ReactantNitro.release!(dl::RemoteLoader) = close(dl.client)   # must tolerate a second call
```

The client must accept concurrent requests from many tasks, or keep one connection per task in a
`TaskLocalValue`. Compared with `num_workers`, this keeps decoding and its allocations out of the
training process while preserving ordered delivery, and the pipeline manages its own memory instead
of holding a copy per worker.
