# Tutorial: MNIST end to end

The example is MNIST, carried from configuration through training to prediction on new data. It is
MNIST on purpose: the task is one nobody has to be told, so every line below is about the framework
rather than about the problem. The model is the small one you would reach for first, an encoder, a
`tanh` hidden layer, and a linear head over the ten classes.

**The example below runs start to finish today, and so does an experiment configuring nothing at
all.**

!!! tip "The whole thing, runnable"
    `examples/mnist_tutorial.jl` in the repository is this model as one self-contained file, with
    its own environment. This page is the reasoning; that file is the code, filled in and runnable:

    ```
    julia --project=examples -e 'using Pkg; Pkg.instantiate()'
    julia --project=examples examples/mnist_tutorial.jl
    ```

    `NITRO_EXAMPLE_EPOCHS=1` turns it into a smoke test, and `NITRO_EXAMPLE_BACKEND=cuda` puts it
    on a GPU.

This page trains the automatic way, with the framework sequencing every optimizer step. When your
algorithm needs to sequence its own steps, a GAN with one optimizer per network, say, the
[Manual training](manual.md) page shows the other mode: you define `train_step` and own the step;
the framework still owns everything outside it.

## Configuration, data, and the model

```julia
using ReactantNitro, Lux, Optimisers, Random
# None of the next three is a ReactantNitro dependency. MLUtils supplies batching and shuffling,
# which the framework deliberately does not ship; a schedule is just a callable, so the framework
# never dispatches on one and takes no dependency on any schedule library; and the framework never
# looks inside your data, so it does not care where the images came from.
using MLDatasets, MLUtils
using ParameterSchedulers: OneCycle

@experiment struct MnistMLP
    "Width of the hidden layer. Structural: changes the compiled graph."
    width::GraphConst{Int} = 128

    "Label smoothing. A knob you may want to sweep or schedule without recompiling."
    smoothing::Device{Float32} = 0.05f0

    "Softmax temperature, for post-hoc calibration. Swept at INFERENCE time, so it has to be a
    traced input rather than a baked constant."
    temperature::Device{Float32} = 1f0

    """
    The training images, as a (784, 60000) matrix held in memory. Dataset-sized, and
    driver-only. Unmarked, so `Host`: the default.
    """
    images::Matrix{Float32} = reshape(MLDatasets.MNIST(:train).features, 784, :)

    "Per-class weights, filled in by `derive` from the training split's label counts."
    class_weights::Device{Vector{Float32}} = Float32[]

    "Epochs to train for. Driver-only: never read inside a traced function."
    max_epochs::Int = 20
end
```

Three markers, three jobs. **[`Host`](@ref) is the default**: an unmarked field is driver-only and
invisible to the tracer entirely, which is the right category for the majority of real fields. A
**[`GraphConst`](@ref)** field bakes into the compiled graph as a constant and is part of the
compile cache key, so changing `width` recompiles, which is correct because it changes the program.
A **[`Device`](@ref)** field is converted to a device value at setup and reaches traced code as an
*input*, so revising it or putting it on a schedule does not recompile.

That third one has two jobs, and the second is the one that will cost you a day if you miss it.

### Keeping a dataset out of the tracer

Holding your data on the experiment is the natural thing to do. You load it once, keep it, and use
it to produce examples. The trouble is that **Reactant and Enzyme traverse the whole experiment
while tracing**, because it is an argument to the compiled program. Anything dataset-sized that they
can reach from it, a matrix of images, a DataFrame, a sampler, a materialized index, a columnar
store, gets walked element by element, on one thread, every single time a program is compiled. MNIST
is small and would merely be slow; the same field holding a real corpus is the version that stops
the session.

What makes this worth a section rather than a footnote is how it presents. It does not raise. It
does not change the graph that comes out, so nothing about the trained model looks wrong. Compile
time simply grows with the size of your dataset, which reads like a data-loading problem and is not
one: the loader is fine, and the cost is being paid before the loader has done anything.

Marking a field [`Host`](@ref) is the whole fix when a field needs it explicitly (an unmarked field
already is Host). The framework hands the tracer a **stripped view** of the experiment, in which
every [`Host`](@ref) field has been replaced by a sentinel that carries only the field's name, so
there is nothing left to walk:

```julia
compile_view(e).images     # ReactantNitro.StrippedHost{:images}()
compile_view(e).width      # 128, unchanged: a GraphConst field still bakes
compile_view(e).smoothing  # unchanged: a Device is a traced input
```

The real `e` is what runs everywhere outside the trace, so [`build_data`](@ref), [`derive`](@ref),
metric finalization, checkpointing, and every accessor read `e.images` normally. Only traced code
sees the sentinel, and using it there raises an error naming the field rather than quietly computing
something. That is deliberate: `nothing` would have passed through an `::Any` signature and survived
in a returned tuple, and this catches the cases that matter, arithmetic, property access, and
conversion.

The alternative, and the one to prefer when it fits, is to keep the data out of the experiment
altogether: [`build_data`](@ref) hands the collection to the framework, which holds it separately
from `e`. A field on the experiment is the case that needs the marker.

**One consequence worth knowing.** A large *derived* value should be [`Device`](@ref), not
[`GraphConst`](@ref), which is why `class_weights` above is marked that way. [`derive`](@ref) merges
its result into the experiment, and a [`GraphConst`](@ref) array field would be both baked into the
graph as a constant and walked by the tracer. As a [`Device`](@ref) it is converted once and crosses
as a single buffer.

```julia
# `build_data` sees the experiment BEFORE device conversion, and it sees the real `e` rather than
# the stripped view, so `e.images` is the (784, 60000) matrix itself here.
function ReactantNitro.build_data(e::MnistMLP, dist)
    labels = MLDatasets.MNIST(:train).targets
    onehot(y) = Float32.(0:9 .== permutedims(y))        # (10, n)

    # The batch dimension is LAST in everything, which is the framework's one shape requirement:
    # img is (784, n) and label is (10, n).
    train_idx, val_idx = 1:55_000, 55_001:60_000
    part(idx) = (img = e.images[:, idx], label = onehot(labels[idx]))
    test = MLDatasets.MNIST(:test)
    testset = (img = reshape(test.features, 784, :), label = onehot(test.targets))

    # A data source is anything iterable that yields concrete NamedTuples of host arrays and
    # supports `length`, which counts batches. MLUtils supplies batching and shuffling; the
    # framework owns the device transfer. `train` must drop its partial final batch, and its batch
    # count must divide by `accum`: 55000 / 100 = 550 batches, and 550 % 2 == 0. The eval splits
    # keep theirs, and the framework pads then slices them.
    (; train = MLUtils.DataLoader(part(train_idx); batchsize = 100, shuffle = true, partial = false),
       val   = MLUtils.DataLoader(part(val_idx);   batchsize = 100, partial = true),
       test  = MLUtils.DataLoader(testset;         batchsize = 100, partial = true))
end

# Values that genuinely depend on the data, computed once and merged into the experiment. This runs
# after `build_data` and before device conversion, so it returns plain host values and the framework
# places them. Reserve it for real dataset dependence: anything computable from config alone belongs
# in an ordinary default.
function ReactantNitro.derive(e::MnistMLP, data)
    counts = [count(==(c), MLDatasets.MNIST(:train).targets) for c in 0:9]
    (; class_weights = Float32.(sum(counts) ./ (10 .* counts)))
end

function ReactantNitro.build_model(e::MnistMLP, rng)
    # Encoder, hidden, head. The head emits RAW LOGITS and no softmax: softmax inside the model
    # followed by a log in the loss is the numerically unstable spelling of the same thing, and
    # keeping the compiled program logit-valued is also what makes it exportable as-is, with the
    # normalization applied outside it. `predict` below is where softmax appears, once.
    model = Chain(Dense(784 => e.width, relu),      # encoder
                  Dense(e.width => e.width, tanh),  # hidden
                  Dense(e.width => 10))             # head, logits
    ps, st = Lux.setup(rng, model)
    (model, ps, st)
end
```

The framework wraps **every** split in a `PrefetchIterator` automatically (one producer per thread,
one batch staged on the device), so the H2D transfer of the next batch overlaps the current step.
That includes the eval splits, and validation is the phase more exposed to a slow host path rather
than less: a validation step is forward-only, so device time per batch falls sharply while host time
per batch does not move. Each stream is built when its phase starts and torn down when it ends, so
training and validation prefetch memory never coexist.

Several producers need the index-addressable trait, `ReactantNitro.batch_at` and
`ReactantNitro.begin_epoch!` on the source, because a sequential Julia iterator cannot be consumed by
several tasks: the expensive work happens inside `iterate` and there is no way to ask for batch *k*
without walking to it. **An `MLUtils.DataLoader` gets both for free**, from the extension that loads
with MLUtils: it reads the loader's declaration (the data, the batch size, the RNG, whether to
shuffle) and rebuilds each epoch's plan with MLUtils' own `shuffleobs` and `BatchView` rather than
iterating it, so the loader above uses every thread with nothing asked of you. A source the framework
does not know still runs in ONE producer task and says so at setup; implementing the two methods is
the fix, and `ReactantNitro.check_batch_at` is what verifies an implementation before a run does.

Two `DataLoader` options do not survive a prefetched pipeline and are checked at setup.
`buffer = true` is refused: it reuses one batch through `getobs!`, and the framework holds several
batches in flight, so the producer would overwrite one still queued for its device transfer.
`parallel = true` warns: MLUtils' own worker threads are a second, uncoordinated fan-out, and its
documentation notes that they break ordering guarantees, so a fixed seed stops reproducing a run
bitwise.

The binding report carries `prefetch_workers`, `prefetch_device_batches`, `prefetch_host_batches`
and `prefetch_ordered`. The two batch counts are the pipeline's two buffers, one per side of the
transfer: `device_batches` is how many sit on the device staged ahead, and `host_batches` is how many
may exist on the host at once, built but not yet transferred.

Two things follow that are worth knowing before you go implementing the trait to silence a warning.
**Adopting it is free**, because delivery is ordered by default: a reorder buffer emits batches in
the source's own order, so a fixed seed still reproduces a run bitwise whatever the worker count is,
and `ordered = false` is the explicit opt-out that trades that back for throughput. And **a `Vector`
of batches never warns**, because `build_data` already built them: producing one is a pointer load,
so there is no host work for several producers to spread and the trait would buy nothing. The
warning is about a loader that does real work per batch, like the one above.

`dist` is the distribution handle, `nothing` in this version, and nothing dispatches on it.

Note where [`build_data`](@ref) reads `e.images`: it runs **host-side**, against the real
experiment. The stripped view exists only at the trace boundary, so a [`Host`](@ref) field is
perfectly ordinary everywhere a [`Host`](@ref) field is supposed to be used.

The full marker contract, and exactly what the tracer does and does not see, is the
[Experiments](experiments.md) page's topic.

## Forward and loss

Each hook declares exactly the batch fields it wants as keyword arguments, and the framework
resolves the method once at setup and passes that subset. [`forward`](@ref) below declares only
`img`, so on a prediction batch carrying no labels it still works, and a field no hook declares
reaches nobody and is never transferred to the device, which is what lets a loader carry bookkeeping
such as an index or a filename alongside the tensors.

```julia
function ReactantNitro.forward(e::MnistMLP, model, ps, st; img)
    logits, st_new = Lux.apply(model, img, ps, st)
    # `e.temperature` is a device scalar and a traced INPUT, so sweeping it later costs no compile.
    # It defaults to 1, so this is the identity during training.
    return logits ./ e.temperature, st_new
end

# `forward` returns (outputs, st_new) and the framework strips st_new, so `logits` here is the
# (10, B) matrix itself. Do NOT unpack it a second time: `first(logits)` is element one of the
# matrix, every operation after that stays broadcast-legal, and the run would train on one number
# out of the batch without ever raising.
function ReactantNitro.loss(e::MnistMLP, logits; label)
    # `e.class_weights` is the derived (10,) vector, on the device and already broadcastable against
    # the (10, B) target. `e.smoothing` is a device scalar. Both are `Device`, so both are traced
    # INPUTS and neither needs unwrapping; had `class_weights` been a `GraphConst` field it would have been
    # baked into the graph as a constant and walked by the tracer on every compile.
    smoothed = (1f0 - e.smoothing) .* label .+ e.smoothing / 10f0
    # NNlib's log-softmax, re-exported by Lux, which subtracts the row max. Reach for it rather
    # than writing `logits .- log.(sum(exp, logits; dims = 1))`: the two are the same function on
    # paper, and the hand-rolled one overflows Float32 for any logit above 88.7, producing an `Inf`
    # loss that the framework's non-finite check turns into a dead run. `forward` divides by
    # `e.temperature`, so a calibration sweep toward zero is exactly the case that reaches it.
    logp = logsoftmax(logits; dims = 1)
    return -sum(smoothed .* logp .* e.class_weights) / size(label, 2)
end
```

[`forward`](@ref) takes inputs and never targets. That discipline is what makes prediction from
inputs alone possible, and it is why the same [`forward`](@ref) serves training, validation, and
inference.

## Metrics, and why they carry their own denominators

A metric returns a `(sum, count)` pair. The framework adds both up across the split, divides at the
end, and hands the result to [`finalize_metrics`](@ref). **It never supplies a sample count of its
own**, because there is no single right one, and MNIST is enough to show it:

```julia
function ReactantNitro.metrics(::MnistMLP, logits; label)
    pred = getindex.(argmax(logits; dims = 1), 1)          # (1, B), the predicted class index
    truth = getindex.(argmax(label; dims = 1), 1)

    (; # per IMAGE: the denominator is the batch's real sample count
       acc = (sum(pred .== truth), size(label, 2)),

       # `count === nothing` means accumulate by summation and do NOT divide, which is what a
       # confusion-matrix-shaped quantity needs. The value is a (10, 10) matrix, and it is summed
       # across the split exactly as a scalar would be.
       confusion = (confusion_matrix(pred, truth, 10), nothing))
end
```

Two shapes in one call: a real denominator, and none at all. The framework's own default metric is
the third, and it is the one you get if you delete the method above entirely: with no
[`metrics`](@ref) defined it substitutes `val_loss` with a count of **1**, so its denominator is the
number of **batches**. Images, batches, none. Had the framework divided everything by a batch size
it inferred, `confusion` would have been silently scaled into nonsense and a per-batch mean would
have been reported as a per-image one, and neither would have raised.

That is the whole reason the contract is a pair rather than a number. On a harder task the gap gets
wider rather than narrower: a per-token loss, a per-detected-object score, and a per-image accuracy
computed from the same batch have three denominators that differ by orders of magnitude, and only
the metric itself knows which one it meant.

[`metrics`](@ref) never sees padding, whichever residency it runs at. The `val` split holds 5,000
images at a batch size of 100, so it divides evenly here; give it 5,050 and the final batch holds 50
real images out of 100. The framework pads it, runs the one compiled [`forward`](@ref), slices the
outputs and the routed batch fields back to 50, and only then calls [`metrics`](@ref). So
`size(label, 2)` above is a real count, and there is no mask for a user to remember to apply.

Derived metrics are finalized host-side, because a macro-averaged recall over a split is not the
mean of the per-batch macro recalls:

```julia
function ReactantNitro.finalize_metrics(::MnistMLP, acc, split)
    # `acc`'s counted keys arrive already divided; the `nothing`-counted ones arrive as raw totals,
    # so `acc.confusion` is the (10, 10) matrix for the WHOLE split.
    recall = [acc.confusion[c, c] / max(sum(acc.confusion[:, c]), 1) for c in 1:10]
    # `split` is a Symbol, so branching between :val and :test is free here. As an argument to
    # `metrics` it would have cost a second compiled program even if the code ignored it.
    return (; acc.acc, macro_recall = sum(recall) / 10)
end
```

Training metrics are a different contract, deliberately. They are scalars per step rather than
`(sum, count)` per batch, they are traced inside the gradient program alongside the loss, and they
are not reduced over anything. They exist to answer "is this step doing something sane", not to
summarize a set:

```julia
ReactantNitro.train_metrics(::MnistMLP, logits; label) =
    (; batch_acc = sum(argmax(logits; dims = 1) .== argmax(label; dims = 1)) / size(label, 2),
       logit_mag = sum(abs, logits) / length(logits))
```

Smoothing, if you want it, is your logger's job.

### Host or device: you choose, per hook

A metric can run two ways, and neither is right for both hooks:

- **traced**, compiled into a program and executed on the accelerator, so the model's raw outputs
  never cross the device boundary;
- **host-side**, in ordinary Julia on transferred arrays, so it can do anything Julia can.

[`metrics_residency`](@ref)`(e, hook)` picks, per hook. The defaults are **`:host` for
[`metrics`](@ref)** and **`:device` for [`train_metrics`](@ref)**, and the reason is cadence.

[`train_metrics`](@ref) runs once per **micro-batch**. Running it on the host would transfer a full
output batch every micro-batch, which for anything image-shaped dwarfs the handful of scalars you
wanted, so it is traced by default: the step returns `(loss, st_new, stats)` and only those scalars
come back. Readback is lazy on top of that, so you can compute them every step and read them only on
the steps you log, with no need for one program for logged steps and another for unlogged ones.

[`metrics`](@ref) runs once per eval batch, once per **epoch**, against a training epoch that has
just executed thousands of steps. The same transfer is noise at that cadence, so the argument that
justifies tracing does not apply, and what does apply is the other direction: a traced metric has to
be expressible as a traced program, and evaluation code often is not. Matching or assignment steps,
connected components, sorting with tie-breaking, data-dependent control flow, or any call into a
library that knows nothing about Reactant are all ordinary in validation and all untraceable. So
[`metrics`](@ref) is host-side by default.

```julia
# The defaults, written out. You would not normally write either line.
ReactantNitro.metrics_residency(::MnistMLP, hook) =
    hook === :train_metrics ? :device : :host

# Trace the validation metric instead, because it is cheap to express and the eval set is large.
ReactantNitro.metrics_residency(::MnistMLP, ::Symbol) = :device
```

**Only `outputs` changes for you**: device arrays under `:device`, host arrays under `:host`.
Keyword routing, the `(sum, count)` contract, `count === nothing`, and the guarantee that a metric
never sees padding are identical either way.

**What changes underneath is whether editing the metric recompiles.** A traced metric is part of a
program's graph, so editing it is editing the program and the compile cache correctly misses rather
than silently reporting your previous metric. Editing a traced [`train_metrics`](@ref) invalidates
the **gradient** program, the expensive one. Editing a traced [`metrics`](@ref) invalidates the
**evaluation** program, which is much cheaper. A **host** metric is part of no program, so editing
it recompiles nothing at all, which is what makes adding a diagnostic mid-session free.

**A future direction.** Reactant is gaining an eager mode, which would let a metric written against
device arrays run without being compiled into a program. That would collapse the choice above into
something closer to a performance hint than a contract, and it is something we intend to look into
for both hooks. Nothing in the metric contract would have to change, since a metric is already a
plain function of the outputs.

## The logger is your backend object

The framework ships exactly one logging backend, the JSON default, and defines no logger
supertype. The contract is ten functions; implement the ones your backend needs, on your own type,
in your own code:

```julia
struct TSVLog          # no supertype, no registration, no extension
    io::IO
end

function ReactantNitro.log_metrics!(lg::TSVLog, m; step, epoch, context, kwargs...)
    # Train metrics arrive with context "train" and carry the step; validation metrics arrive with
    # context "validate" and carry the epoch plus the current step, so the two overlay on one axis.
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

# The logger is a run accessor like any other, so the experiment can supply its own and derive the
# path from the run directory it already owns. The framework calls this exactly ONCE, at setup, which
# is what makes it safe for an accessor to have a side effect: opening a file here, or registering a
# run with a hosted tracker.
function ReactantNitro.logger(e::MnistMLP)
    dir = run_dir(e)
    mkpath(dir)
    TSVLog(open(joinpath(dir, "metrics.tsv"), "w"))
end
```

A bare run still leaves a machine-readable record: leaving [`logger`](@ref) unset gives you
[`JSONLogger`](@ref), which writes one JSON object per line to `metrics.jsonl` in the run's
directory (params and the binding report at setup, one line per optimizer step, one per epoch of
validation, and a `finish` line with the run's outcome). `logger = nothing` is the documented
opt-out and stays the public "no logging" value, with a method for all ten verbs.

Three things follow from the contract being duck-typed rather than a hierarchy. A verb you did not
define is a loud `MethodError` when the framework first reaches it, because a contract of optional
no-ops makes wrappers silently lossy and silence should be opted into per verb. In the common case
there is nothing to unwrap: the logger you pass *is* your backend object, so an experiment
tracker's own run handle goes in directly and every other client function in that package remains
callable on the same handle. If you do wrap one, [`backend`](@ref)`(lgr)` is the one accessor that
gets it back.

And one verb answers "where is this run, and what is it called": [`logger_info`](@ref)`(lgr)`
returns the backend's key identifying parameters as a plain `NamedTuple`. The JSON default reports
its `path`; a hosted tracker reports its URL, experiment key, workspace, and whatever else it
defines. [`logger_info`](@ref)`(nitro)` reads the same table off a running experiment, and the
Kaimon gate's `nitro_logger` tool renders it.

## Optimizer and schedule: the recipe belongs to the experiment

Everything that is part of what the experiment *is* goes on the experiment type, so re-running it
does not require remembering what it needed:

```julia
# Two parameter groups. The defaults give one group, RAdam, and a fixed learning rate. The per-group
# accessors define RATIOS against the base, which a schedule then scales as a whole, so the encoder
# stays a tenth of the rest for the entire curve rather than drifting relative to it. `ks` is the
# parameter's keypath, so this reads "layer_1 is the encoder".
ReactantNitro.param_group(::MnistMLP, ks) = ks[1] === :layer_1 ? :encoder : :default
ReactantNitro.learning_rate(::MnistMLP)   = 3f-4
ReactantNitro.learning_rate(::MnistMLP, ::Val{:encoder}) = 3f-5
ReactantNitro.lambda(::MnistMLP, ::Val{:encoder})        = 1f-4   # decoupled decay, toward zero

# The schedule is part of the recipe too. Every entry is a factory of the horizon: the framework
# calls it once, with the total number of optimizer steps, and then calls what it returns once per
# step. The key is `eta` because that is the rule's own field name, not `lr`.
ReactantNitro.schedules(::MnistMLP) = (; eta = total -> OneCycle(total, 3f-4))

# Global norm over the fully accumulated gradient. 0 means off, and is the default.
ReactantNitro.gradient_clip_norm(::MnistMLP) = 1f0

# The run knobs are accessors too, so the experiment carries its own defaults and a REPL call does
# not retype them. Each of these is also a keyword, and the keyword wins for that one run.
ReactantNitro.accum(::MnistMLP)        = 2        # 550 batches per epoch, so 275 optimizer steps
ReactantNitro.run_dir(::MnistMLP)      = "runs/mnist"
ReactantNitro.checkpointer(::MnistMLP) =
    TopKCheckpointer(; k = 3, metric = :macro_recall, mode = :max)
ReactantNitro.early_stop(::MnistMLP)   =
    EarlyStopping(; metric = :macro_recall, mode = :max, patience = 5)

# `max_epochs` needs no accessor here: it is already a `Host` field on the struct (unmarked, so
# Host is the default), and every one of these accessors reads a field of its own name when the
# experiment has one. Keeping the driver-only ones Host is not optional: a `GraphConst` field bakes
# into the compiled graph and enters the compile cache key, so a `GraphConst` `seed` field would
# recompile once per seed and defeat the point of a seed sweep. The framework rejects that at setup
# and names the marker as the fix.
```

The optimizer contract, parameter groups, and decay are the [Optimization](optimization.md) page's
subject;

the schedule contract has its own page, [Schedules](schedules.md).

## The run: keywords are the per-run knobs

With all of that on the experiment, the experiment IS the run configuration, and a run is one call:

```julia
e = MnistMLP(; width = 128, max_epochs = 20)

nitro = train!(e)
```

**Which accelerator the run executes on is a session-level choice, not a run keyword.** A run
compiles for Reactant's default backend (GPU where one is visible, else CPU), and a process
initializes its XLA client once. To be explicit about it, CPU versus CUDA versus ROCm versus
TPU, call [`setup_devices!`](@ref) once before training (the Kaimon tool `nitro_setup` is the
same function):

```julia
using ReactantNitro
setup_devices!(backend = "cpu")            # run everything on CPU
setup_devices!(backend = "cuda", n_devs = 2)  # two of the visible CUDA devices
```

`n_devs` defaults to every visible device and shards the batch across them; on a GPU host,
restrict the visible set with `CUDA_VISIBLE_DEVICES` before starting the process.

[`train!`](@ref)`(e; ...)` is sugar for [`train!`](@ref)`(Nitro(e; ...))`. Every keyword belongs to
the constructor and [`train!`](@ref)`(nitro)` takes none, so there is one keyword surface rather
than two lists to keep in sync.

**Ten of the fourteen keywords default to an accessor of the same name**: [`seed`](@ref),
[`run_dir`](@ref), [`n_devs`](@ref), [`accum`](@ref), [`max_epochs`](@ref), [`schedules`](@ref),
[`gradient_clip_norm`](@ref), [`logger`](@ref), [`checkpointer`](@ref), and [`early_stop`](@ref).
Declaring one on the experiment and omitting the keyword is the normal path; passing the keyword
replaces it for that run, without editing the struct or redefining a method:

```julia
# A follow-up run of the SAME experiment, longer, on a different seed, with a flat learning rate
# instead of the one-cycle curve, writing somewhere else. The parameter groups, the decay, the
# checkpointer, the stopping rule and the logger still come from `e`.
train!(e; max_epochs   = 60,
          seed         = 43,
          schedules    = (; eta = _ -> t -> 1f-4),
          run_dir      = "runs/mnist-flat")
```

**Nothing in that call recompiles.** The learning rate reaches the optimizer as a device scalar that
the framework rewrites each step, so a schedule and a constant are the same slot and switching
between them costs nothing; [`seed`](@ref), [`run_dir`](@ref) and [`max_epochs`](@ref) never reach a
traced function at all. The second run reuses both compiled programs from the first.

Two keywords are the exception, and they are the two that bake into a program as trace-time
constants. [`accum`](@ref) reaches the gradient program as `1/N`, so changing it compiles a new
gradient program, the expensive one. [`gradient_clip_norm`](@ref) is applied at the top of the
optimizer program, so changing it compiles a new optimizer program and reuses the gradient program,
which is why a clip sweep is cheap and an accumulation sweep is not:

```julia
train!(e; gradient_clip_norm = 0f0)   # new optimizer program; the gradient program is reused
train!(e; accum = 4)                  # new gradient program, which is the costly one
```

## Revising a `Device`, without compiling anything

A [`Device`](@ref) field is a traced input rather than a baked constant, so its value cannot affect
the compiled graph and is excluded from the compile cache key by construction. Revising one is a new
experiment and the same two programs:

```julia
# More label smoothing than we started with. This is a different experiment value, trained from
# scratch, and it compiles NOTHING: `smoothing` is `Device`.
heavier = MnistMLP(; width = 128, max_epochs = 20, smoothing = 0.15f0)
train!(heavier)

# A sweep over it is therefore one compile for the whole sweep.
for s in (0f0, 0.05f0, 0.1f0, 0.2f0)
    train!(MnistMLP(; width = 128, smoothing = s); run_dir = "runs/mnist-smooth-$s")
end

# `width` is a GraphConst field, so it is a trace-time constant and part of the cache key. This one
# does compile, and correctly so: it is a different graph.
train!(MnistMLP(; width = 512))
```

That is the whole point of the markers. Deciding which fields are [`Device`](@ref) is deciding which
sweeps are free, and deciding which are [`GraphConst`](@ref) is deciding which changes are new
programs.

## What a `Nitro` freezes, and when to build a new one

**A [`Nitro`](@ref) is a snapshot of one run's configuration.** The fourteen [`Nitro`](@ref)
keywords are resolved once, at construction, into the handle's own fields, and every read afterwards
goes to the field. So the rule is short:

> **Build a new [`Nitro`](@ref) for a new training job. Re-use an existing one for more validation,
> evaluation, prediction, and [`Device`](@ref) sweeps.**

Building a new one is cheap, and this is the part that surprises people: **the compile cache is
module-level, not per-[`Nitro`](@ref).** A fresh handle on the same experiment hits the entries the
previous one compiled. [`max_epochs`](@ref) is a [`Host`](@ref) field and is excluded from the key,
so:

```julia
n = Nitro(e); train!(n)                                # pays the compiles once, this session

# "Now train it longer." A NEW handle, and it compiles nothing.
n2 = Nitro(e; data = n.data, max_epochs = 60, resume = :auto)   # restores the weights
train!(n2)
ReactantNitro.cache_stats()                                          # (; hits = ..., misses = 0)
```

Two details make that cheap. `data = n.data` re-uses the collection instead of re-running
[`build_data`](@ref), and `resume = :auto` searches `run_dir`, so the new handle finds the
latest checkpoint in [`run_dir`](@ref) and continues from the trained weights rather than fresh
ones. **Resuming is opt-in**: the default is `resume = false`, so constructing a [`Nitro`](@ref)
never picks up weights nobody named. For evaluation and inference, `checkpoint = path` restores weights *and* the derived values
from the record, so it skips [`derive`](@ref) as well.

What a rebuild still costs is setup, not compilation: device conversion, one batch pulled to infer
routing, and [`derive`](@ref) on the resume path. Seconds, against the hundreds a recompile costs.

How the freeze, the module-level compile cache, and the rebuild path interact is the
[Recompilation](recompilation.md) page's subject.

## Revising with Revise: build a new `Nitro`

The line is short: **a [`Nitro`](@ref) is a fixed point. Revise, then build a new one.**

Everything that decides which compiled program runs is resolved at construction, so **nothing you
revise changes an existing handle**, and a handle can never silently recompile underneath you. The
one exception is a [`Device`](@ref) value, which is exactly the thing the compile key excludes by
construction, and [`set_device!`](@ref) below is how you write one.

```julia
# ── NOTHING here lands on an existing handle ────────────────────────────────────────
# A traced hook. `n` keeps the program it was built with, and says so.
ReactantNitro.loss(e::MnistMLP, logits; label) = my_new_loss(logits, label)
train!(n)                                     # still the OLD loss, and it tells you

# A run accessor. Resolved at construction; `n.accum` is still whatever it was.
ReactantNitro.accum(::MnistMLP) = 4
train!(n)                                     # still trains at the OLD accum

# An accessor that CONSTRUCTS. The object was built at setup, so a revised constructor
# is inert, and the patience counter inside the old one keeps counting.
ReactantNitro.early_stop(::MnistMLP) = EarlyStopping(; patience = 10)
train!(n)                                     # still the old stopper

# ── The fix is always the same, and it compiles only what really changed ────────────
n = Nitro(e; data = n.data)                   # picks up every edit above
train!(n)

# ── What DOES land on an existing handle ────────────────────────────────────────────
# A host-side method on a stored object. Nothing is compiled and nothing is snapshotted
# except the object, so the new code simply runs next epoch, for free.
ReactantNitro.should_stop(es::MyStopper, epoch, metrics) = ...
train!(n)                                     # no recompile, new behaviour
```

**A stale handle is never silent**, which is the whole reason the fixed point is safe. Every entry
point prints what it fixed, flags a scalar accessor that has drifted away from it, and names any
hook you have redefined since:

```text
ReactantNitro: MnistMLP values fixed at construction (rebuild the `Nitro` to change them)
  seed 42   accum 2   max_epochs 20   total 5500   clip 1.0   run_dir runs/mnist
  ! `accum(e)` was redefined: it returned 2 at construction and returns 4 now. This handle still
    uses 2; rebuild to pick up 4.
  ! hooks were redefined after this `Nitro` was built (worlds_train, worlds_eval). It will keep
    running the programs it was built with; rebuild the `Nitro` to pick up the new code. The
    compile cache is module-level, so a rebuild recompiles only what actually changed.
```

The flag compares what the accessor returned *at construction* against what it returns now, so a
keyword you passed deliberately (`run_dir = mktempdir()`) is never mistaken for a stale accessor.
Only a genuine redefinition trips it.

The handle always wins, deliberately: [`accum`](@ref) and [`max_epochs`](@ref) determine the
schedule horizon `total`, so applying one in isolation would leave every schedule resolved against a
horizon that no longer exists. Rebuild instead, which costs no compilation. Silence it in a scripted
driver with `ReactantNitro.set_config_report!(false)`.

**Why freezing is cheaper than it sounds.** An earlier design let a redefined hook move the key on
an existing handle, so [`train!`](@ref)`(n)` recompiled and ran the new code. That is convenient
exactly once and confusing thereafter: the same call sometimes recompiled and sometimes did not, and
a redefined *run accessor* bought a full recompile that produced a byte-identical program. Freezing
makes the handle predictable and pushes the choice to you, and it costs nothing, because a rebuilt
handle shares the module-level cache and recompiles only the programs your edit actually changed.

Two limits worth knowing. Revising an accessor that constructs an object ([`early_stop`](@ref),
[`checkpointer`](@ref), [`logger`](@ref)) is **not** flagged, because probing it would fire the side
effect the framework promises to trigger exactly once. And a `const` or global read from inside a
hook body is invisible to the key entirely: the method's world only moves when *the method itself*
is edited, so edit a helper that [`forward`](@ref) calls and you get a stale program.
`ReactantNitro.cache_reset!()` after any edit the report cannot see, or restart.

Mid-run edits do nothing at all: a running [`train!`](@ref) is pinned to the world age it started
in, and Revise does not even apply the change until the REPL gets a prompt back.

## Changing a `Device` on a live handle

[`set_device!`](@ref) is the one thing you can change on a built [`Nitro`](@ref) and have take
effect immediately, because a [`Device`](@ref) field is a traced input and is excluded from the
cache key by construction. This is the supported inference-sweep loop:

```julia
n = Nitro(e; data = (;), checkpoint = "runs/mnist/best.jld2")   # no training split needed

for t in (0.5f0, 1f0, 1.5f0, 2f0)
    set_device!(n; temperature = t)
    logits = predict(n, batch)                   # ZERO compiles, every iteration
    @show t, expected_calibration_error(logits, labels)
end

device_value(n, :temperature)                    # reads it back as a HOST value
```

It refuses anything that would break the no-recompile promise, and the error says which case you
hit: a [`GraphConst`](@ref) field (bakes, so it is genuinely a different program), a [`Host`](@ref)
field (reaches no trace, so writing it would change nothing compiled), a scheduled field (the
schedule owns the value), or a value of a different element type or size. In every one of those
cases the answer is a new [`Nitro`](@ref).

**Four keywords have no accessor, because each names a fact about this invocation rather than a
property of the experiment.** `data` substitutes for [`build_data`](@ref), which is its accessor
under another name. `checkpoint` and `resume` point at a file this particular call should read.
`run_ref` is a channel back to the caller, filled before the loop starts so you can reach the handle
while it runs.

The per-group accessors, [`learning_rate`](@ref)`(e, Val(g))` and [`lambda`](@ref)`(e, Val(g))`,
stay outside the keyword mechanism entirely: there is no sensible keyword form for a value that
varies by group, and a bare [`learning_rate`](@ref) keyword would compete with [`schedules`](@ref)
for the same quantity. Where a value could have come from three places, the binding report says
which one won.

Setup prints that report: where each configured value actually bound, which sources of data it
found, and how many samples the training loader's drop-last discarded. It hands the same text to
[`log_other!`](@ref), so it lands in your run's record rather than only on your terminal.

## Predicting on new data

[`predict`](@ref) is [`forward`](@ref) alone, in eval mode, and it works on any [`Nitro`](@ref). It
takes a batch `NamedTuple` or anything that iterates them, which is the same data source contract
the training loader satisfies. It does not take a bare array, because [`forward`](@ref) is routed by
keyword and the framework needs the field names. Only the fields [`forward`](@ref) declares are
required, so a prediction batch needs no labels.

```julia
# One batch. The batch dimension is last in everything you pass and everything you get back.
batch = (; img = randn(Float32, 784, 6))            # 6 new images
logits = predict(nitro, batch)

# logits :: Matrix{Float32}, size (10, 6), the same shape `forward` returned, as a HOST array.
#
# 6 is not the training batch size of 100. The framework padded the batch to 100, ran the one
# compiled `forward`, and sliced every output leaf back to 6 before returning, so the padding is
# invisible here. Host arrays rather than device ones, because you are leaving the framework and a
# device array that outlives its run is a footgun.

# The head emits logits, so THIS is where softmax goes: outside the compiled program, on host
# arrays, once. Keeping it out of the model is what makes the program numerically stable to
# differentiate and exportable without a postprocessing step baked into it.
probs = exp.(logits) ./ sum(exp, logits; dims = 1)
pred = getindex.(argmax(logits; dims = 1), 1) .- 1  # 0-based class labels

predict(nitro, randn(Float32, 784, 6))
# ERROR: `forward` declares the batch field `img`, so `predict` needs a NamedTuple: (; img = ...)

# A loader in, a LAZY iterator out: one element per batch, nothing materialized up front.
unseen = MLUtils.DataLoader((; img = randn(Float32, 784, 500)); batchsize = 100, partial = true)
for logits in predict(nitro, unseen)
    # 5 batches of (10, 100). Give it 520 images instead and the final one is (10, 20), sliced
    # from the padded run.
    write_predictions(logits)
end

all_logits = reduce(hcat, predict(nitro, unseen))   # (10, 500), if you want them all
```

[`validate`](@ref)`(nitro)` and [`evaluate`](@ref)`(nitro; split = :test)` share the same compiled
[`forward`](@ref) and the same compiled [`metrics`](@ref) with each other and with
[`predict`](@ref), so moving between them never recompiles. [`evaluate`](@ref) errors on a split
name [`build_data`](@ref) did not return, naming the ones it did.

## No training anywhere in the process

Everything above about prediction holds for a [`Nitro`](@ref) that has never trained. That is not a
special case bolted on: the setup sequence is what produces the state those functions read, and
training is a separate thing that happens afterward.

```julia
# Fresh weights from `build_model`, for a forward-pass sanity check or a shape check on new data.
# No optimizer state is allocated, which for Adam-family rules would otherwise be twice the
# parameter memory on device for nothing.
fresh = Nitro(e)
predict(fresh, batch)

# Trained weights, restored from a checkpoint, with no training in this process.
trained = Nitro(e; checkpoint = "runs/mnist/latest")
evaluate(trained; split = :test)

# Serving: `data = (;)` skips `build_data` entirely, so a process that only predicts does not have
# to produce a training split to get started. Keyword routing and the batch width are then resolved
# from the first batch `predict` is handed, and nothing needs padding because you supplied it whole.
serving = Nitro(e; checkpoint = "runs/mnist/latest", data = (;))
predict(serving, batch)
```

## Further reading

The four guides take each of this tutorial's topics further: [Experiments](experiments.md) covers
the markers and the stripped view, [Recompilation](recompilation.md) the compile cache and when a
rebuild is needed, [Optimization](optimization.md) the optimizer contract, and [Schedules](schedules.md) the
schedule contract. Every exported name, with its docstring, is on the [API](api.md)
page.
