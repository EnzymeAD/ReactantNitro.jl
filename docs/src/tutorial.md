# Tutorial: MNIST end to end

MNIST, from configuration through training to prediction on new data. The model is an encoder and
a linear head over ten classes, with the README quick start's network, split, learning rate and
epoch budget. It differs in batch size (100 rather than 32) and in the features this page adds.

!!! tip "The whole thing, runnable"
    `examples/mnist/mnist_tutorial.jl` is this page's experiment as one file with its own
    environment. It keeps the default logger in place of the TSV logger below.

    ```
    julia --project=examples/mnist -e 'using Pkg; Pkg.instantiate()'
    julia --project=examples/mnist examples/mnist/mnist_tutorial.jl
    ```

    `NITRO_EXAMPLE_EPOCHS=1` turns it into a smoke test, and `NITRO_EXAMPLE_BACKEND=cuda` puts it
    on a GPU.

This page trains the automatic way: the framework sequences every optimizer step. An algorithm
that sequences its own steps, such as a GAN with one optimizer per network, defines `train_step`
instead; see [Manual training](manual.md).

!!! note "Binding cheat sheet"
    The [Binding cheat sheet](binding.md) draws every connection this page uses: batch fields to
    hooks, `forward`'s outputs to `loss` and `metrics`, the three markers to what traced code
    sees, schedule keys to their slots, run keywords to accessors to fields, and what recompiles.
    Check it when a name does not reach where you expected.

## Configuration, data, and the model

```julia
using ReactantNitro, Lux, Optimisers, Random
# Not ReactantNitro dependencies. MLUtils supplies batching and shuffling, which the framework does
# not ship; a schedule is any callable, so the framework needs no schedule library; and the
# framework never looks inside your data.
using MLDatasets, MLUtils
using ParameterSchedulers: OneCycle

@experiment struct MnistMLP
    "Width of the hidden layer. Structural: changes the compiled graph."
    width::GraphConst{Int} = 128

    "Label smoothing. A knob to sweep or schedule without recompiling."
    smoothing::Device{Float32} = 0.05f0

    "Softmax temperature, for post-hoc calibration. Swept at inference time, so it must be a
    traced input rather than a baked constant."
    temperature::Device{Float32} = 1f0

    """
    The training images, as a (784, 60000) matrix held in memory. Dataset-sized and driver-only.
    Unmarked, so `Host`: the default.
    """
    images::Matrix{Float32} = reshape(MLDatasets.MNIST(:train).features, 784, :)

    "Per-class weights, filled in by `derive` from the training split's label counts."
    class_weights::Device{Vector{Float32}} = Float32[]

    "Epochs to train for. Driver-only: never read inside a traced function."
    max_epochs::Int = 5
end
```

[`Host`](@ref) is the default: an unmarked field is driver-only and invisible to the tracer, which
suits most fields. A [`GraphConst`](@ref) field is baked into the compiled graph as a constant and
is part of the compile cache key, so changing `width` recompiles, because it is a different
program. A [`Device`](@ref) field is converted to a device value at setup and reaches traced code
as an input, so changing or scheduling it does not recompile.

### Keeping a dataset out of the tracer

Reactant and Enzyme traverse the whole experiment while tracing, because it is an argument to the
compiled program. A dataset-sized array reachable from it is walked element by element, on one
thread, on every compile. Nothing raises and the graph is unchanged; only compile time grows, which
is easy to mistake for a slow loader.

The framework therefore hands the tracer a **stripped view** of the experiment, in which every
[`Host`](@ref) field is replaced by a sentinel carrying only the field's name:

```julia
compile_view(e).images     # ReactantNitro.StrippedHost{:images}()
compile_view(e).width      # 128, unchanged: a GraphConst field still bakes
compile_view(e).smoothing  # unchanged: a Device is a traced input
```

The real `e` is used everywhere outside the trace, so [`build_data`](@ref), [`derive`](@ref),
`finalize_metrics`, checkpointing, and every accessor read `e.images` normally. Reading the
sentinel from traced code raises with the field named. The other option, often the better one, is
to keep the data off the experiment entirely and return it from [`build_data`](@ref).
[Experiments](experiments.md) has the full marker contract.

A large *derived* value should be [`Device`](@ref) rather than [`GraphConst`](@ref), which is why
`class_weights` is `Device`. A `GraphConst` array is baked into the graph and walked by the tracer;
a `Device` array is converted once and crosses as a single buffer.

```julia
# `build_data` runs host-side against the real `e`, before device conversion, so `e.images` is the
# (784, 60000) matrix itself here.
function ReactantNitro.build_data(e::MnistMLP, dist)
    labels = MLDatasets.MNIST(:train).targets
    onehot(y) = Float32.(0:9 .== permutedims(y))        # (10, n)

    # The batch dimension is LAST in everything, the framework's one shape requirement:
    # img is (784, n) and label is (10, n).
    train_idx, val_idx = 1:55_000, 55_001:60_000
    part(idx) = (img = e.images[:, idx], label = onehot(labels[idx]))
    test = MLDatasets.MNIST(:test)
    testset = (img = reshape(test.features, 784, :), label = onehot(test.targets))

    # A data source is anything iterable that yields NamedTuples of host arrays and supports
    # `length`. `train` must drop its partial final batch, and its batch count must divide by
    # `accum`: 55000 / 100 = 550 batches, and 550 % 2 == 0. The eval splits keep theirs; the
    # framework pads and slices them.
    (; train = MLUtils.DataLoader(part(train_idx); batchsize = 100, shuffle = true, partial = false),
       val   = MLUtils.DataLoader(part(val_idx);   batchsize = 100, partial = true),
       test  = MLUtils.DataLoader(testset;         batchsize = 100, partial = true))
end

# Values that depend on the data, computed once and merged into the experiment. Runs after
# `build_data` and before device conversion, so it returns host values and the framework places
# them. Anything computable from config alone belongs in an ordinary default instead.
function ReactantNitro.derive(e::MnistMLP, data)
    counts = [count(==(c), MLDatasets.MNIST(:train).targets) for c in 0:9]
    (; class_weights = Float32.(sum(counts) ./ (10 .* counts)))
end

function ReactantNitro.build_model(e::MnistMLP, rng)
    # The head emits RAW LOGITS, no softmax. Softmax in the model followed by a log in the loss is
    # the numerically unstable spelling of the same thing, and a logit-valued program exports
    # as-is. `predict` below is where softmax appears, once.
    model = Chain(Dense(784 => e.width, relu),      # encoder
                  Dense(e.width => 10))             # head, logits
    ps, st = Lux.setup(rng, model)
    (model, ps, st)
end
```

`dist` is reserved for a distribution handle and is `nothing` in this version; leave it untyped.

### Prefetching is automatic

The framework wraps every split in a [`PrefetchIterator`](@ref): by default one producer task per
thread of Julia's default pool (`julia -t N`), and one batch staged on the device, so the next
batch's transfer overlaps the current step. Eval splits are wrapped too, and benefit more: a
validation step is forward-only, so device time per batch falls while host time does not. Each
stream is built when its phase starts and torn down when it ends, so training and validation
prefetch memory never coexist. To change the defaults, return the split already wrapped in a
`PrefetchIterator` with its own `workers`, `device_batches`, `host_batches` or `ordered`;
[`NoPrefetch`](@ref) opts a split out.

Several producers need an index-addressable source: one with [`batch_at`](@ref) and
[`begin_epoch!`](@ref) methods. An `MLUtils.DataLoader` has both through the MLUtils extension,
which reads the loader's settings and rebuilds each epoch's plan with `shuffleobs` and `BatchView`
instead of iterating it. Any other source runs in one producer and setup warns; implementing the
two methods is the fix, and `ReactantNitro.check_batch_at` verifies an implementation. A `Vector`
of batches does not warn, because producing one is a pointer load.

A source that holds something open, a data server or a file, defines `ReactantNitro.release!`,
which the framework calls once per handle at `Done` or `Failed`; `ReactantNitro.release!(nitro)`
calls it for a handle that only evaluated or exported. A loader's package can ship these methods
itself, as a package extension with ReactantNitro as a weak dependency, so the framework never has
to name it.

Delivery is ordered by default, so a fixed seed reproduces a run bitwise at any worker count;
`ordered = false` on the split's `PrefetchIterator` trades that for throughput. Two `DataLoader`
options are checked at setup. On the multi-producer path both are inert, since that path never
calls the loader's `iterate`; `buffer = true` warns that it has no effect. On a single producer,
`buffer = true` is refused, because the loader reuses one batch through `getobs!` while the
framework holds several in flight, and `parallel = true` warns, because MLUtils' own worker threads
break ordering and a fixed seed stops reproducing.

## Forward and loss

Each hook declares the batch fields it wants as keyword arguments. The framework resolves the
method once at setup and passes that subset. [`forward`](@ref) declares only `img`, so it works on
a prediction batch with no labels. A field no hook declares is never transferred to the device, so
a loader can carry an index or a filename alongside the tensors.

```julia
function ReactantNitro.forward(e::MnistMLP, model, ps, st; img)
    logits, st_new = Lux.apply(model, img, ps, st)
    # `e.temperature` is a traced INPUT, so sweeping it later costs no compile. Default 1, so this
    # is the identity during training.
    return logits ./ e.temperature, st_new
end

# `forward` returns (outputs, st_new) and the framework strips st_new, so `logits` here is the
# (10, B) matrix itself. Do NOT unpack it again: `first(logits)` is element one of the matrix,
# every operation after that stays broadcast-legal, and the run trains on one number out of the
# batch without raising.
function ReactantNitro.loss(e::MnistMLP, logits; label)
    # `e.class_weights` is the derived (10,) vector, on the device and broadcastable against the
    # (10, B) target. `e.smoothing` is a device scalar. Both are traced INPUTS.
    smoothed = (1f0 - e.smoothing) .* label .+ e.smoothing / 10f0
    # NNlib's log-softmax subtracts the row max. The hand-rolled `logits .- log.(sum(exp, logits))`
    # overflows Float32 for any logit above 88.7, and the resulting `Inf` loss stops the run at the
    # non-finite check. Dividing by a small `e.temperature` is what produces such logits.
    logp = logsoftmax(logits; dims = 1)
    return -sum(smoothed .* logp .* e.class_weights) / size(label, 2)
end
```

Because [`forward`](@ref) takes inputs and never targets, one [`forward`](@ref) serves training,
validation, and inference, and prediction needs only inputs.

## Metrics carry their own denominators

A metric is `value => mode`. The usual mode is a count: the framework sums value and count across
the split, divides once at the end, and hands the result to [`finalize_metrics`](@ref). It never
supplies a sample count of its own, because the right denominator depends on the metric. `:sum`,
`:max`, `:min` and `:concat` are the other modes; see [Metrics](metrics.md).

```julia
function ReactantNitro.metrics(::MnistMLP, logits; label)
    pred = getindex.(argmax(logits; dims = 1), 1)          # (1, B), the predicted class index
    truth = getindex.(argmax(label; dims = 1), 1)

    (; # per IMAGE: the denominator is the batch's real sample count
       acc = sum(pred .== truth) => size(label, 2),

       # `:sum` means sum and never divide, which a confusion matrix needs. The (10, 10) value
       # is summed across the split exactly as a scalar would be. `confusion_matrix` is a
       # ten-line host helper, defined in the example script.
       confusion = confusion_matrix(pred, truth, 10) => :sum)
end
```

Without a [`metrics`](@ref) method, the framework reports `val_loss` with a count of 1, so its
denominator is the number of batches. Dividing every metric by an inferred batch size would scale
`confusion` into nonsense and report a per-batch mean as a per-image one.

[`metrics`](@ref) never sees padding. The `val` split has 5,000 images at a batch size of 100; give
it 5,050 and the final batch holds 50 real images. The framework pads it, runs the one compiled
[`forward`](@ref), slices the outputs and the routed batch fields back to 50, and then calls
[`metrics`](@ref). `size(label, 2)` is always a real count.

Derived metrics are finalized host-side, because a macro-averaged recall over a split is not the
mean of per-batch macro recalls:

```julia
function ReactantNitro.finalize_metrics(::MnistMLP, acc, split)
    # Counted keys arrive already divided; `:sum` keys arrive as raw totals, so
    # `acc.confusion` is the (10, 10) matrix for the WHOLE split.
    recall = [acc.confusion[c, c] / max(sum(acc.confusion[:, c]), 1) for c in 1:10]
    # `split` is a Symbol, so branching on :val versus :test is free here. As an argument to
    # `metrics` it would cost a second compiled program.
    return (; acc.acc, macro_recall = sum(recall) / 10)
end
```

What `finalize_metrics` returns replaces `acc`: it is what the logger, the checkpointer, the
early-stopping rule and `history` see. Here that is `acc` and `macro_recall`; `confusion` is not
passed on.

Training metrics are a different contract: scalars per step, traced inside the gradient program
alongside the loss, and not reduced across batches. They are per-step diagnostics, not a measure of
the model:

```julia
ReactantNitro.train_metrics(::MnistMLP, logits; label) =
    (; batch_acc = sum(argmax(logits; dims = 1) .== argmax(label; dims = 1)) / size(label, 2),
       logit_mag = sum(abs, logits) / length(logits))
```

### Host or device, per hook

A metric runs either **traced** on the accelerator, so the model's outputs never cross the device
boundary, or **host-side** in ordinary Julia, so it can use any Julia code.
[`metrics_residency`](@ref)`(e, hook)` picks. The defaults are `:device` for
[`train_metrics`](@ref), which runs once per micro-batch and would otherwise transfer a full output
batch every step, and `:host` for [`metrics`](@ref), which runs once per eval batch per epoch. There
the transfer is small next to the epoch, and evaluation code is often not traceable: matching,
connected components, sorting with tie-breaking, any library that knows nothing about Reactant.

```julia
# The defaults, written out. You would not normally write either line.
ReactantNitro.metrics_residency(::MnistMLP, hook) =
    hook === :train_metrics ? :device : :host

# Trace the validation metric instead. Every value it computes must then be traceable, which the
# loop in `confusion_matrix` is not.
ReactantNitro.metrics_residency(::MnistMLP, ::Symbol) = :device
```

The hook's code sees one difference: `outputs` are device arrays under `:device` and host arrays
under `:host`. The other difference is whether editing the metric recompiles. A traced
[`train_metrics`](@ref) is part of the gradient program, the expensive one; a traced
[`metrics`](@ref) is part of the cheaper evaluation program; a host metric is part of no program,
so adding a diagnostic mid-session compiles nothing. [Metrics](metrics.md) has the full comparison.

## The logger is your backend object

The framework ships one logging backend, the JSON default, and defines no logger supertype. The
contract is ten functions on your own type; implement the ones the framework calls for your run:

```julia
struct TSVLog          # no supertype, no registration, no extension
    io::IO
end

function ReactantNitro.log_metrics!(lg::TSVLog, m; step, epoch, context, kwargs...)
    # Every call carries `step` and `epoch`, so train and validation lines overlay on one axis.
    # `context` is "train" (the loss, `train_metrics` and `schedule/` values, once per logged
    # optimizer step), "validate" (once per epoch), or "data" (the epoch's `data_wait_frac`).
    for (k, v) in pairs(m)
        println(lg.io, join((context, epoch, step, k, v), '\t'))
    end
    flush(lg.io)
end

function ReactantNitro.log_params!(lg::TSVLog, params)
    for (k, v) in pairs(params)
        println(lg.io, join(("param", k, repr(v)), '\t'))
    end
end

ReactantNitro.log_other!(lg::TSVLog, key, value) = println(lg.io, join(("other", key, value), '\t'))
ReactantNitro.finish!(lg::TSVLog, status) = (println(lg.io, "finish\t$status"); close(lg.io))

# Read into every checkpoint record, so a run that checkpoints needs them.
ReactantNitro.run_id(::TSVLog) = nothing
ReactantNitro.run_url(::TSVLog) = nothing

# The logger is a run accessor like any other. The framework calls it exactly ONCE, at setup, so an
# accessor may have a side effect: opening a file, or registering a run with a hosted tracker.
function ReactantNitro.logger(e::MnistMLP)
    dir = run_dir(e)
    mkpath(dir)
    TSVLog(open(joinpath(dir, "metrics.tsv"), "w"))
end
```

Leaving [`logger`](@ref) unset gives [`JSONLogger`](@ref), one JSON object per line in
`metrics.jsonl` under the run directory: params and the binding report at setup, a train line per
optimizer step, a validation line and a data line per epoch, and a `finish` line.
[`log_every_n_steps`](@ref) writes the train line on every `n`th step only. A schedule that varies
is logged on the train line without any code from you, as `schedule/<field>` for a `Device` field
and `schedule/<group>/<field>` for an optimizer field; setting [`autolog_schedules`](@ref) to
`false` turns that off. `logger = nothing` disables logging.

A verb you did not define is a `MethodError` when the framework first calls it, rather than a
silent no-op, so a wrapper that forgets a verb fails instead of dropping data. The logger you pass
*is* your backend object, so an experiment tracker's own run handle goes in directly and its client
functions stay callable on it; if you wrap one, [`backend`](@ref)`(lgr)` returns it.
[`logger_info`](@ref)`(lgr)` returns the backend's identifying parameters as a `NamedTuple`, the
path for the JSON default or the URL and experiment key for a hosted tracker;
[`logger_info`](@ref)`(nitro)` reads the same table off a running experiment.

## Optimizer and schedule belong to the experiment

Everything that defines the experiment goes on the experiment type, so re-running it needs nothing
remembered from the last call:

```julia
# Two parameter groups. The defaults give one group, RAdam, and a fixed learning rate. `ks` is the
# parameter's keypath, so this reads "layer_1 is the encoder". A per-group `learning_rate` method
# is a RATIO against the base, which a schedule scales as a whole, so a group set to a tenth stays
# a tenth for the entire curve. The encoder here differs only in its decay.
ReactantNitro.param_group(::MnistMLP, ks) = ks[1] === :layer_1 ? :encoder : :default
ReactantNitro.learning_rate(::MnistMLP) = 1f-3
ReactantNitro.lambda(::MnistMLP, ::Val{:encoder}) = 1f-4   # decoupled decay, toward zero

# Every schedule entry is a factory of the horizon: the framework calls it once with the total
# number of optimizer steps, then calls what it returned once per step. The key is `eta` because
# that is the rule's own field name. A bare key applies to every group.
ReactantNitro.schedules(::MnistMLP) = (; eta = total -> OneCycle(total, 1f-3))

# Global norm over the fully accumulated gradient. 0 means off, and is the default.
ReactantNitro.gradient_clip_norm(::MnistMLP) = 1f0

# The run knobs are accessors too, so the experiment carries its own defaults. Each is also a
# `Nitro` keyword, and the keyword wins for that one run.
ReactantNitro.accum(::MnistMLP)        = 2        # 550 batches per epoch, so 275 optimizer steps
ReactantNitro.run_dir(::MnistMLP)      = "runs/mnist"
ReactantNitro.checkpointer(::MnistMLP) =
    TopKCheckpointer(; k = 3, metric = :macro_recall, mode = :max)
ReactantNitro.early_stop(::MnistMLP)   =
    EarlyStopping(; metric = :macro_recall, mode = :max, patience = 5)

# `max_epochs` needs no accessor: it is already a `Host` field on the struct, and each run-knob
# accessor in this block reads a field of its own name when the experiment declares one. Keep the
# driver-only ones Host: a `GraphConst` `seed` field would recompile once per seed. The framework
# rejects that at setup and names the marker as the fix.
```

The one-cycle curve varies, so each train line carries it per group as `schedule/encoder/eta` and
`schedule/default/eta`, plus `schedule/encoder/lambda`, because the encoder's decoupled decay is
scaled by its learning rate.

[Optimization](optimization.md) covers the optimizer contract, parameter groups, and decay;
[Schedules](schedules.md) covers the schedule contract.

## The run: keywords are the per-run knobs

The experiment is the run configuration, and a run is one call:

```julia
e = MnistMLP(; width = 128, max_epochs = 20)

nitro = train!(e)
```

The accelerator is chosen per session, not per run. A process initializes its XLA client once, for
Reactant's default backend (GPU where visible, else CPU). To choose, call [`setup_devices!`](@ref)
once before the first `Nitro`; the Kaimon tool `nitro_setup` is the same function:

```julia
setup_devices!(backend = "cpu")               # run everything on CPU
setup_devices!(backend = "cuda", n_devs = 2)  # two of the visible CUDA devices
```

`n_devs` defaults to every visible device and shards the batch across them; restrict the visible
set with `CUDA_VISIBLE_DEVICES` before starting the process.

[`train!`](@ref)`(e; ...)` is sugar for [`train!`](@ref)`(Nitro(e; ...))`. Every keyword belongs to
the constructor and [`train!`](@ref)`(nitro)` takes none, so there is one keyword surface.

Thirteen of the keywords default to an accessor of the same name: [`seed`](@ref),
[`run_dir`](@ref), [`n_devs`](@ref), [`accum`](@ref), [`max_epochs`](@ref), [`schedules`](@ref),
[`gradient_clip_norm`](@ref), [`check_divergence`](@ref), [`autolog_schedules`](@ref),
[`log_every_n_steps`](@ref), [`logger`](@ref), [`checkpointer`](@ref), and [`early_stop`](@ref).
Passing the keyword replaces the accessor for that run, without editing the struct or redefining a
method:

```julia
# A follow-up run of the SAME experiment: longer, another seed, a flat learning rate instead of
# the one-cycle curve, written somewhere else. Parameter groups, decay, checkpointer, stopping rule
# and logger still come from `e`.
train!(e; max_epochs   = 60,
          seed         = 43,
          schedules    = (; eta = _ -> t -> 1f-4),
          run_dir      = "runs/mnist-flat")
```

Nothing in that call recompiles. The learning rate reaches the optimizer as a device scalar
rewritten each step, so a schedule and a constant use the same slot; [`seed`](@ref),
[`run_dir`](@ref) and [`max_epochs`](@ref) never reach a traced function. The second run reuses
every program the first compiled.

Two keywords bake into a program as trace-time constants. [`accum`](@ref) reaches the gradient
program as `1/N`, so changing it compiles a new gradient program, the expensive one.
[`gradient_clip_norm`](@ref) is applied at the top of the optimizer program, so changing it compiles
a new optimizer program and reuses the gradient program:

```julia
train!(e; gradient_clip_norm = 0f0)   # new optimizer program; the gradient program is reused
train!(e; accum = 4)                  # new gradient program, the costly one
```

The other keywords have no accessor, because each describes this call rather than the experiment.
`data` substitutes for [`build_data`](@ref). `weights` and `resume` name where this call's
parameters come from: a file, a `run => checkpoint` pair such as `:latest => :best`, or another
handle. `run_ref` is a `Ref` that setup fills with the new handle before training starts. `w0`,
`restore_optimizer` and `restore_best` refine `weights` and `resume`; the [API](api.md) has them
all.

## Binding

Four things connect by name at setup, with nothing wired by hand. The binding report, appended to
`show(nitro)` and handed to [`log_other!`](@ref), says where each one landed.

!!! tip "Drawn out"
    The [Binding cheat sheet](binding.md) has each of these as a diagram with the rule under it.

| What | Binds to | By |
| --- | --- | --- |
| a batch field the loader yields | the hooks that declare it as a keyword | the keyword's name |
| a schedule key | an optimizer rule field, or a `Device` field on the experiment | the key's name, resolved against both namespaces |
| a `Nitro` keyword | the accessor of the same name, which usually reads the field of the same name | the keyword's name |
| a data source | the prefetch pipeline, fanned out over every thread for a `DataLoader` | the source's type |

**Batch to hooks.** [`forward`](@ref) declared `img`, [`loss`](@ref) declared `label`, and the
framework transferred exactly those two fields and passed each hook its own. A loader field no hook
declares never leaves the host. A required keyword the batch lacks raises on the hook's first call,
naming the batch's fields and the hook's; a keyword with a default takes it, which is how an
optional batch field works.

**Schedule to slot.** `eta` above resolved to the optimizer rules' learning-rate field. A key
naming a `Device` field, `smoothing` say, would instead be written into the experiment each step
and read by traced code as `e.smoothing`. A key matching both namespaces is an ambiguity error that
shows the two qualified forms, `(; opt = (; lambda = ...))` and `(; device = (; lambda = ...))`. A
group path, `(; opt = (; encoder = (; eta = ...)))`, binds one group's chain; for that group it
overrides a bare `eta`, which still applies to every other group. [Schedules](schedules.md) has the
rules.

**Keyword to accessor to field.** `max_epochs` was never given a method: the accessor read the
struct field. `run_dir` was given a method. Twelve of the thirteen run-knob accessors fall back to
a field of their own name; `schedules` is the exception. The keyword wins for one run either way.

**Per-group values have no keyword.** [`learning_rate`](@ref)`(e, Val(g))` and
[`lambda`](@ref)`(e, Val(g))` are methods only, and a bare `learning_rate` keyword would compete
with `schedules` for the same quantity.

`show(nitro)` prints the handle's state, seed, accumulation, horizon, batch size and preset, and
then the report's bands. The data band has one row per split: batch count, sample count, samples
dropped by drop-last, whether the final batch is padded, and the prefetch path that bound, with its
worker count, batches staged on device and on host, and a warning if delivery is unordered. The
bindings band lists the clip threshold and every schedule key with its source, keyword or
accessor. The parameter groups band has one row per group: rate, ratio, anchor, decay, rule, and
parameter count. The prefetch settings are also logged as params, because the worker count follows
the process's thread count, and comparing two runs needs the number.

## Device values: sweep without compiling

A [`Device`](@ref) field is a traced input, so its value cannot affect the graph and is excluded
from the compile key. A new experiment differing only in a `Device` value is a new run on the same
compiled programs:

```julia
# More smoothing. A different experiment value, trained from scratch, and it compiles NOTHING.
train!(MnistMLP(; width = 128, max_epochs = 20, smoothing = 0.15f0))

# A sweep over it is one compile for the whole sweep.
for s in (0f0, 0.05f0, 0.1f0, 0.2f0)
    train!(MnistMLP(; width = 128, smoothing = s); run_dir = "runs/mnist-smooth-$s")
end

# `width` is GraphConst, so this one compiles: it is a different graph.
train!(MnistMLP(; width = 512))
```

On a live handle, [`set_device!`](@ref) writes a `Device` field in place and takes effect
immediately. This is the supported inference-sweep loop:

```julia
# The best checkpoint of the latest run in `run_dir`. No training split needed.
n = Nitro(e; data = (;), weights = :latest => :best)

for t in (0.5f0, 1f0, 1.5f0, 2f0)
    set_device!(n; temperature = t)
    logits = predict(n, batch)                   # ZERO compiles, every iteration
    @show t, expected_calibration_error(logits, labels)
end

device_value(n, :temperature)                    # reads it back as a HOST value
```

It refuses anything that would break the no-recompile promise, and the error says which case: a
`GraphConst` field, a `Host` field, a scheduled field, a value that does not convert to the field's
type, or an array of a different size. In each case the answer is a new `Nitro`.

So mark `Device` the values you want to sweep for free, and `GraphConst` only the values that
define a different program.

## A `Nitro` is a fixed point

Every keyword is resolved once, at construction, into the handle's own fields, and every read
afterwards goes to the field. Nothing you revise changes an existing handle, and a handle never
recompiles underneath you.

**Build a new `Nitro` for a new training job. Reuse one for more validation, evaluation,
prediction, and `Device` sweeps.**

A rebuild is cheap because the compile cache is module-level, shared by every handle. A fresh
handle on the same experiment hits the entries the previous one compiled:

```julia
n = Nitro(e); train!(n)                                # pays the compiles once, this session

# "Now train it longer." A NEW handle, and it compiles nothing.
n2 = Nitro(e; data = n.data, max_epochs = 60, resume = :auto)   # restores the weights
train!(n2)
ReactantNitro.cache_stats()     # (; hits, misses, entries): `misses` has not moved since `train!(n)`
```

`data = n.data` reuses the collection instead of re-running [`build_data`](@ref). `resume = :auto`
continues from the newest checkpoint of the most recent run in [`run_dir`](@ref), and starts fresh
with a warning when there is none; the default is `resume = false`, so a `Nitro` never picks up
weights nobody named. For evaluation, `weights = path` or `weights = :latest => :best` restores the
weights and the derived values from the record, skipping [`derive`](@ref). What a rebuild costs is
setup: device conversion, one batch to infer routing, and `derive` on the resume path. That is
seconds, where a recompile can take hundreds.

A **warm start** is the third way weights enter a handle, for a REPL where the trained handle is
at hand. `weights = n` takes `n`'s parameters and layer state as the new run's initial ones;
everything else is a fresh run, so the new experiment's `derive` runs, the optimizer state is
fresh, and the step count starts at zero:

```julia
n = train!(MnistMLP(; max_epochs = 20))
# A new experiment starting from the trained weights: fresh optimizer, epoch 0, its own run_dir.
fine = train!(MnistMLP(; smoothing = 0.1f0, max_epochs = 5); weights = n, run_dir = "runs/mnist-ft")

# L2-SP fine-tuning: `w0 = :weights` anchors `decay_anchor = :w0` groups to the transferred
# weights rather than to `build_model`'s initialization, the default.
Nitro(e2; weights = n, w0 = :weights)
```

The trees must match leaf for leaf; a `width = 512` experiment refuses `weights` from a
`width = 128` run with a diff naming the leaves. `weights` with `resume` is an
error, and `show` says where the weights came from.

The same rule covers Revise:

```julia
# ── NOTHING here lands on an existing handle ────────────────────────────────────────
ReactantNitro.loss(e::MnistMLP, logits; label) = my_new_loss(logits, label)
train!(n)                                     # still the OLD loss, and it tells you

ReactantNitro.accum(::MnistMLP) = 4
train!(n)                                     # still the OLD accum

# ── The fix, which compiles only what changed ───────────────────────────────────────
n = Nitro(e; data = n.data)
train!(n)

# ── What DOES land: a host-side method on a stored object ───────────────────────────
ReactantNitro.should_stop(es::MyStopper, epoch, metrics) = ...
train!(n)                                     # no recompile, new behaviour next epoch
```

A stale handle reports itself. When something has been redefined since construction, every entry
point prints what: a scalar run-knob accessor whose value has changed, and any hook redefined
since. With nothing redefined it prints nothing.

```text
ReactantNitro: MnistMLP has values fixed at construction that have since been redefined. This
handle keeps what it froze; rebuild the `Nitro` to pick up the new ones, and see `show(nitro)` for
what it is currently holding.
  ! `accum(e)` was redefined: it returned 2 at construction and returns 4 now. This handle still
    uses 2; rebuild to pick up 4.
  ! hooks were redefined after this `Nitro` was built (worlds_train, worlds_opt, worlds_eval). It
    will keep running the programs it was built with; rebuild the `Nitro` to pick up the new code.
    The compile cache is module-level, so a rebuild recompiles only what actually changed.
```

The handle keeps its values because [`accum`](@ref) and [`max_epochs`](@ref) determine the
schedule horizon, so applying one in isolation would leave every schedule resolved against a
horizon that no longer exists. The accessor check covers the scalar knobs; a redefined `logger`,
`checkpointer`, `early_stop` or `schedules` is not detected. Silence the report in a scripted
driver with `ReactantNitro.set_config_report!(false)`. A running [`train!`](@ref) is pinned to the
world age it started in, so mid-run edits do nothing. [Experiments](experiments.md) has the limits
of the report, and [Recompilation](recompilation.md) has what the cache key contains.

## Reading the run: `history`

After [`train!`](@ref) returns, [`history`](@ref)`(n)` is the run as data: one row per validated
epoch with the epoch's mean train loss and what [`finalize_metrics`](@ref) returned. It displays as
a table sized to the terminal, and indexes by epoch and by metric name:

```julia
h = history(n)
h                          # every epoch that fits, thinned around the first, best and last
h[10:20]                   # epochs 10 to 20
h[:acc, :macro_recall]     # two metrics, every epoch
h[10:20, :acc]             # both
h[step = 2_000:6_000]      # the epochs that closed within those steps
h[end]                     # the last epoch's row, a NamedTuple
h.macro_recall             # the column, as a Vector
```

```text
history of MnistMLP  (runs/mnist, 13 of 40 epochs)
┌──────────────────────────────────────────────┐
│ epoch   step     loss  macro_recall     acc  │
├──────────────────────────────────────────────┤
│     1    275  0.70000        0.9158  0.9124  │
│     2    550  0.37512        0.9194  0.9162  │
│     3    825  0.26043        0.9230  0.9200  │
│     4   1100  0.20102        0.9265  0.9237  │
│     ⋮                                        │
│    21   5775  0.04520        0.9749  0.9749  │
│    22   6050  0.04334        0.9765  0.9766  │
│    23   6325  0.04164        0.9776  0.9778 *│
│    24   6600  0.04008        0.9765  0.9766  │
│    25   6875  0.03863        0.9749  0.9749  │
│    26   7150  0.03729        0.9730  0.9729  │
│     ⋮                                        │
│    38  10450  0.02650        0.9400  0.9380  │
│    39  10725  0.02589        0.9368  0.9346  │
│    40  11000  0.02531        0.9334  0.9310  │
└──────────────────────────────────────────────┘
  * best macro_recall (max), the checkpointer's metric
  27 of 40 epochs thinned to fit; select fewer, `h[16:30]` say, for every row
```

The columns are fixed: epoch, step, the train loss, the checkpointer's metric, then the rest. A
metric that is not a scalar, such as a confusion matrix returned from `finalize_metrics`, is named
in a `not tabulated:` footer line and reachable as a column, `h.confusion`. The history belongs to
the handle: a resumed handle starts at the epoch it resumed from, and the footer says so.

The history is also a Tables.jl table, so `DataFrame(h)` and `CSV.write("run.csv", h)` take it
directly, and in a notebook it displays as the same table in HTML with every row. The handle and
the experiment display the same way: one description, drawn as a framed text table in a terminal
and as an HTML table wherever `text/html` is asked for. A slice keeps all of that:
`h[10:20, :acc]` is a `MetricHistory` too, and shows, plots and converts the same way.

With a Makie backend loaded, `plot(h)` or `plot(n)` draws the same history. The default is one
axis, the checkpointer's metric over epochs with the best epoch starred, since that is the curve
the run selected on. The same selections change the figure: `plot(h[10:20])` for a window,
`plot(h[:acc, :loss])` for those two curves as two axes, `plot(h; x = :step)` for the step
axis, and `plot(h; metrics = :all)` for every column. The extension is on Makie itself, so
CairoMakie, GLMakie and WGLMakie all activate it, and `plot(fig[2, 1], h)` places the whole grid
inside a figure of your own.

```julia
using CairoMakie
save("history.png", plot(h))
```

## Progress, in a terminal or a notebook

`train!` reports each stretch of work, an epoch or an evaluation pass, through one reporter
contract, and the default reporter picks the display per stretch. Either way the display is one
bar for the whole run, filled by epochs completed with the running stretch interpolated, so the
ETA is the run's. It is labelled with the current stretch, `epoch 3/40: train [compiling gradient]`,
and ends as `done: 40/40 epochs`. In an interactive terminal it draws a ProgressMeter bar. Where
there is no terminal but the current logger accepts
[ProgressLogging](https://github.com/JuliaLogging/ProgressLogging.jl) records, which is Pluto, VS
Code, or a REPL running [TerminalLoggers](https://github.com/JuliaLogging/TerminalLoggers.jl), it
emits those records and the environment draws them. In a CI log or a captured transcript it emits
nothing. `ReactantNitro.progress_reporter!` installs a reporter for every later call: one of the two
built-ins (`ReactantNitro.progress_bar_reporter`, `ReactantNitro.progress_log_reporter`), your own
function of the same five arguments, or `nothing` to silence it.

`Nitro(e)` reports its setup the same way, as one line naming the current step
(`setup [building data]`) and closing as `setup done`. Nothing compiles during setup; the first
compile is reported by the first verb that needs it.

A hook that takes a while can say what it is doing. [`with_progress_note`](@ref) shows a note
beside the phase for the length of a block, with the innermost block shown when they nest, and
[`progress_note!`](@ref) updates the current one. A new phase clears them.

```julia
function ReactantNitro.build_data(e::MyExp, dist)
    index = with_progress_note(() -> load_index(e), "loading index")
    with_progress_note("decoding") do
        for (i, f) in enumerate(files)
            progress_note!("decoding $i/$(length(files))")
            ...
        end
    end
    ...
end
```

## Predicting on new data

[`predict`](@ref) is [`forward`](@ref) alone, in eval mode, on any `Nitro`. It takes a batch
`NamedTuple` or anything that iterates them. It does not take a bare array, because
[`forward`](@ref) is routed by keyword and needs the field names. Only the fields
[`forward`](@ref) declares are required, so a prediction batch needs no labels.

```julia
# One batch. The batch dimension is last in everything you pass and everything you get back.
batch = (; img = randn(Float32, 784, 6))            # 6 new images
logits = predict(nitro, batch)

# logits :: Matrix{Float32}, size (10, 6), as a HOST array. 6 is not the training batch size of
# 100: the framework padded to 100, ran the one compiled `forward`, and sliced every output leaf
# back to 6. Host arrays, because a device array that outlives its run can be freed under you.
# A batch WIDER than 100 is an error.

# The head emits logits, so softmax goes HERE: outside the compiled program, on host arrays, once.
probs = exp.(logits) ./ sum(exp, logits; dims = 1)
pred = getindex.(argmax(logits; dims = 1), 1) .- 1  # 0-based class labels

predict(nitro, randn(Float32, 784, 6))
# ERROR: ReactantNitro: `predict` takes a batch `NamedTuple` or something iterating them, not a
# bare `Matrix{Float32}`. ... Call it as `predict(nitro, (; img = ...))`.

# A loader in, a LAZY iterator out: one element per batch, nothing materialized up front.
unseen = MLUtils.DataLoader((; img = randn(Float32, 784, 500)); batchsize = 100, partial = true)
for logits in predict(nitro, unseen)
    # 5 batches of (10, 100). With 520 images the final one is (10, 20), sliced from the padded run.
    write_predictions(logits)
end

all_logits = reduce(hcat, predict(nitro, unseen))   # (10, 500), if you want them all
```

[`validate`](@ref)`(nitro)`, [`evaluate`](@ref)`(nitro; split = :test)` and [`predict`](@ref) share
one compiled [`forward`](@ref) and one compiled [`metrics`](@ref), so moving between them never
recompiles. [`evaluate`](@ref) errors on a split [`build_data`](@ref) did not return, naming the
ones it did.

## No training anywhere in the process

Setup produces the state those functions read; training is a separate step.

```julia
# Fresh weights from `build_model`, for a shape check on new data. With no `train` split, setup
# allocates no optimizer state, which for Adam-family rules would be twice the parameter memory.
fresh = Nitro(e; data = (;))
predict(fresh, batch)

# Trained weights from the latest run's best checkpoint, no training in this process.
trained = Nitro(e; weights = :latest => :best)
evaluate(trained; split = :test)

# Serving: `data = (;)` skips `build_data` entirely. Routing and the batch width are resolved from
# the first batch `predict` is handed; later batches pad to that width and may not exceed it.
serving = Nitro(e; weights = :latest => :best, data = (;))
predict(serving, batch)
```

## Further reading

[Experiments](experiments.md) for the markers and the stripped view,
[Recompilation](recompilation.md) for the compile cache, [Optimization](optimization.md) for the
optimizer contract, [Schedules](schedules.md) for the schedule contract, and the [API](api.md) for
every docstring.
