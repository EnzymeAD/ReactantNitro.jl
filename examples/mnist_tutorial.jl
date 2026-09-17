# The tutorial, as one runnable file.
#
# `docs/src/tutorial.md` is the prose: it builds this model a hook at a time and explains each
# decision. This is the same model with the explanations removed and the gaps filled in, so it
# runs end to end. The two are meant to be read together, and the tutorial is where the reasoning
# lives; the comments kept here are the ones that stop a reader mis-copying the code.
#
#     julia --project=examples -e 'using Pkg; Pkg.instantiate()'
#     julia --project=examples examples/mnist_tutorial.jl
#
# The full run is 5 epochs, on the same network, split, learning rate and epoch budget as the
# README's quick start. Measured on CPU: acc 0.9584, macro_recall 0.9579, about 3 minutes.
#
# It lands below the quick start's 0.9796, and the reason is the STEP COUNT rather than the recipe.
# At a batch size of 100 with `accum = 2` the effective batch is 200, so 5 epochs is 1,375 optimizer
# steps against the quick start's 8,590 at a batch of 32. The larger batch is what makes gradient
# accumulation worth demonstrating here; it also means an epoch buys fewer updates.
#
# `NITRO_EXAMPLE_EPOCHS=1` makes it a smoke test, and `NITRO_EXAMPLE_BACKEND=cuda` runs it on a GPU.
# The schedule is sized for the full run, so a one-epoch smoke test is not representative. Use it to
# check that the thing runs, not to judge the model.
#
# MLDatasets downloads MNIST on first use and prompts before it does; this file accepts on your
# behalf, which is the one thing here you might not want done silently.

ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

using ReactantNitro, Lux, Optimisers, Random
# None of the next three is a ReactantNitro dependency. MLUtils supplies batching and shuffling,
# which the framework deliberately does not ship; a schedule is just a callable, so the framework
# never dispatches on one and takes no dependency on any schedule library; and the framework never
# looks inside your data, so it does not care where the images came from.
using MLDatasets, MLUtils
using ParameterSchedulers: OneCycle

# ── The experiment ──────────────────────────────────────────────────────────────────

@experiment struct MnistMLP
    "Width of the hidden layer. Structural: changes the compiled graph."
    width::GraphConst{Int} = 128

    "Label smoothing. A knob you may want to sweep or schedule without recompiling."
    smoothing::Device{Float32} = 0.05f0

    "Softmax temperature, for post-hoc calibration. Swept at INFERENCE time, so it has to be a
    traced input rather than a baked constant."
    temperature::Device{Float32} = 1.0f0

    """
    The training images, as a (784, 60000) matrix held in memory. Dataset-sized, and
    driver-only. Unmarked, so `Host`: the default.
    """
    images::Matrix{Float32} = reshape(MLDatasets.MNIST(:train).features, 784, :)

    "Per-class weights, filled in by `derive` from the training split's label counts."
    class_weights::Device{Vector{Float32}} = Float32[]

    "Epochs to train for. Driver-only: never read inside a traced function."
    max_epochs::Int = 5
end

# ── Data ────────────────────────────────────────────────────────────────────────────

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
    # supports `length`, which counts batches. `train` must drop its partial final batch, and its
    # batch count must divide by `accum`: 55000 / 100 = 550 batches, and 550 % 2 == 0. The eval
    # splits keep theirs, and the framework pads then slices them.
    return (;
        train = MLUtils.DataLoader(part(train_idx); batchsize = 100, shuffle = true, partial = false),
        val = MLUtils.DataLoader(part(val_idx); batchsize = 100, partial = true),
        test = MLUtils.DataLoader(testset; batchsize = 100, partial = true),
    )
end

# Values that genuinely depend on the data, computed once and merged into the experiment. This runs
# after `build_data` and before device conversion, so it returns plain host values and the framework
# places them.
function ReactantNitro.derive(e::MnistMLP, data)
    counts = [count(==(c), MLDatasets.MNIST(:train).targets) for c in 0:9]
    return (; class_weights = Float32.(sum(counts) ./ (10 .* counts)))
end

# ── Model, forward, loss ────────────────────────────────────────────────────────────

function ReactantNitro.build_model(e::MnistMLP, rng)
    # The head emits RAW LOGITS and no softmax: softmax inside the model followed by a log in the
    # loss is the numerically unstable spelling of the same thing, and keeping the compiled program
    # logit-valued is what makes it exportable as-is.
    model = Chain(
        Dense(784 => e.width, relu),      # encoder
        Dense(e.width => 10),             # head, logits
    )
    ps, st = Lux.setup(rng, model)
    return (model, ps, st)
end

function ReactantNitro.forward(e::MnistMLP, model, ps, st; img)
    logits, st_new = Lux.apply(model, img, ps, st)
    # `e.temperature` is a device scalar and a traced INPUT, so sweeping it later costs no compile.
    # It defaults to 1, so this is the identity during training.
    return logits ./ e.temperature, st_new
end

# `forward` returns (outputs, st_new) and the framework strips st_new, so `logits` here is the
# (10, B) matrix itself. Do NOT unpack it a second time.
function ReactantNitro.loss(e::MnistMLP, logits; label)
    # `e.class_weights` is the derived (10,) vector, on the device and already broadcastable against
    # the (10, B) target. `e.smoothing` is a device scalar. Both are `Device`, so both are traced
    # INPUTS and neither needs unwrapping.
    smoothed = (1.0f0 - e.smoothing) .* label .+ e.smoothing / 10.0f0
    # NNlib's log-softmax, re-exported by Lux, which subtracts the row max. Reach for it rather
    # than writing `logits .- log.(sum(exp, logits; dims = 1))`: the two are the same function on
    # paper, and the hand-rolled one overflows Float32 for any logit above 88.7. `forward` divides
    # by `e.temperature`, so a calibration sweep toward zero is exactly the case that reaches it.
    return -sum(smoothed .* logsoftmax(logits; dims = 1) .* e.class_weights) / size(label, 2)
end

# ── Metrics ─────────────────────────────────────────────────────────────────────────

# Not supplied by the framework, and small enough not to be worth a dependency.
function confusion_matrix(pred, truth, k)
    cm = zeros(Int, k, k)
    for (p, t) in zip(pred, truth)
        cm[p, t] += 1
    end
    return cm
end

function ReactantNitro.metrics(::MnistMLP, logits; label)
    pred = getindex.(argmax(logits; dims = 1), 1)          # (1, B), the predicted class index
    truth = getindex.(argmax(label; dims = 1), 1)
    return (;
        # Per IMAGE: the denominator is the batch's real sample count, and the framework sums both
        # halves across the split and divides ONCE at the end.
        acc = (sum(pred .== truth), size(label, 2)),
        # `count === nothing` means accumulate by summation and do NOT divide, which is what a
        # confusion-matrix-shaped quantity needs.
        confusion = (confusion_matrix(pred, truth, 10), nothing),
    )
end

function ReactantNitro.finalize_metrics(::MnistMLP, acc, split)
    # `acc`'s counted keys arrive already divided; the `nothing`-counted ones arrive as raw totals,
    # so `acc.confusion` is the (10, 10) matrix for the WHOLE split.
    recall = [acc.confusion[c, c] / max(sum(acc.confusion[:, c]), 1) for c in 1:10]
    # `split` is a Symbol, so branching between :val and :test is free here.
    return (; acc.acc, macro_recall = sum(recall) / 10)
end

ReactantNitro.train_metrics(::MnistMLP, logits; label) =
    (;
    batch_acc = sum(argmax(logits; dims = 1) .== argmax(label; dims = 1)) / size(label, 2),
    logit_mag = sum(abs, logits) / length(logits),
)

# ── Optimizer, schedule, run knobs ──────────────────────────────────────────────────

# Two parameter groups. The per-group accessors define RATIOS against the base, which a schedule
# then scales as a whole, so the encoder stays a tenth of the rest for the entire curve. `ks` is the
# parameter's keypath, so this reads "layer_1 is the encoder".
# The encoder runs at the BASE rate and differs only in its decay, which keeps this run comparable
# to the README's quick start. A per-group `learning_rate` method would define a ratio against the
# base instead, which a schedule then scales as a whole.
ReactantNitro.param_group(::MnistMLP, ks) = ks[1] === :layer_1 ? :encoder : :default
ReactantNitro.learning_rate(::MnistMLP) = 1.0f-3
ReactantNitro.lambda(::MnistMLP, ::Val{:encoder}) = 1.0f-4   # decoupled decay, toward zero

# Every schedule entry is a factory of the horizon: the framework calls it once, with the total
# number of optimizer steps, and then calls what it returns once per step. The key is `eta` because
# that is the rule's own field name, not `lr`.
ReactantNitro.schedules(::MnistMLP) = (; eta = total -> OneCycle(total, 1.0f-3))

# Global norm over the fully accumulated gradient. 0 means off, and is the default.
ReactantNitro.gradient_clip_norm(::MnistMLP) = 1.0f0

# The run knobs are accessors too, so the experiment carries its own defaults. Each is also a
# keyword, and the keyword wins for that one run.
ReactantNitro.accum(::MnistMLP) = 2                  # 550 batches per epoch, so 275 optimizer steps
ReactantNitro.run_dir(::MnistMLP) = "runs/mnist"
ReactantNitro.checkpointer(::MnistMLP) =
    TopKCheckpointer(; k = 3, metric = :macro_recall, mode = :max)
ReactantNitro.early_stop(::MnistMLP) =
    EarlyStopping(; metric = :macro_recall, mode = :max, patience = 5)

# ── The run ─────────────────────────────────────────────────────────────────────────

function main()
    setup_devices!(backend = get(ENV, "NITRO_EXAMPLE_BACKEND", "cpu"))
    # The keyword WINS over the experiment's `max_epochs` field, which is the point of the run
    # knobs being keywords too. Kept in agreement with the field so the two do not drift.
    epochs = parse(Int, get(ENV, "NITRO_EXAMPLE_EPOCHS", "5"))

    e = MnistMLP()
    nitro = Nitro(e; max_epochs = epochs)
    train!(nitro)

    # What the run produced, without a logger backend: the handle carries the last epoch's
    # validation metrics, the elapsed time, and the checkpoint the run selected.
    show(stdout, MIME"text/plain"(), nitro)
    println()

    # The test split, which training never looked at. `evaluate` runs the same compiled eval
    # program `validate` used, so this costs no compile. Note the keyword: `split` is not
    # positional.
    @info "test split" evaluate(nitro; split = :test)...

    # Prediction from inputs alone, on a batch that is not the training batch size: the framework
    # pads to 100, runs the one compiled `forward`, and slices every output back to 6.
    logits = predict(nitro, (; img = nitro.e.images[:, 1:6]))
    @info "predicted classes" classes = getindex.(argmax(logits; dims = 1), 1)

    # `temperature` is a `Device` field, so sweeping it is a device write and not a compile.
    for t in (0.5f0, 1.0f0, 2.0f0)
        set_device!(nitro; temperature = t)
        p = predict(nitro, (; img = nitro.e.images[:, 1:1]))
        @info "temperature sweep" t max_logit = maximum(p)
    end
    ReactantNitro.cache_stats()   # misses stays flat across everything above

    return nitro
end

main()
