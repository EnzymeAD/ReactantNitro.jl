---
name: reactantnitro-experiment
description: >
  Author or read a ReactantNitro experiment: the four required hooks, the three field
  markers (Device, Host, GraphConst) and which one a field wants, why every knob is a
  FLAT field and never a configuration struct held in one, the batch contract
  and keyword routing, what a hook may return and why a batch-reduced scalar may not be an
  output leaf, derived values, and what `Nitro(e)` does. Start here for any ReactantNitro
  task; this skill also indexes the rest. Invoke when writing a new experiment,
  porting a training loop onto ReactantNitro, adding a knob to an existing one, or reading
  one you did not write.
---

# Authoring a ReactantNitro experiment

An experiment is a struct plus hooks. The framework owns the compiled programs, the device
transfers, the optimizer, the schedules, and the lifecycle. `Nitro(e)` runs setup and nothing else,
so evaluation and serving never require a `train!` to have happened; `train!(nitro)` runs the loop.

## The four required hooks

Everything else has a default. These do not.

```julia
build_model(e, rng) -> (model, ps, st)      # standard Lux setup
build_data(e, dist) -> NamedTuple of splits # (; train, val, test); `dist` is nothing today
forward(e, model, ps, st; <batch fields>) -> (outputs, st_new)
loss(e, outputs; <batch fields>) -> scalar
```

An experiment defining only these four trains, validates, and checkpoints with nothing else passed.
When you are unsure whether something needs a hook, assume it does not and check the default first.

`forward` is the exportable program and is what `predict` calls, which is why the three-function
split exists: `forward` sees only inputs, `loss` and `metrics` see the outputs plus the label fields.
Do not fold the loss into `forward`.

## The three markers, and the default

```julia
@experiment struct MyExp
    "Structural: changes the emitted graph, so it bakes and is in the compile key."
    width::GraphConst{Int} = 128

    "A traced INPUT: sweep it or schedule it without recompiling."
    smoothing::Device{Float32} = 0.05f0

    "Unmarked, therefore Host: driver-only and invisible to the tracer."
    max_epochs::Int = 20
end
```

**Unmarked means `Host`, and that is the default on purpose.** The first real model ported to this
framework was 83% `Host` by field count, so the category you get by saying nothing is the one you
almost always want. Writing `Host{T}` explicitly is legal and worth it only where a field's
host-ness would otherwise surprise a reader.

| Marker | Reaches traced code as | In the compile key? | Reach for it when |
| --- | --- | --- | --- |
| `GraphConst{T}` | a baked literal, ordinary Julia semantics | **yes**, so changing it recompiles | the value changes the graph: a shape, a class count, a layer width, a variant selector |
| `Device{T}` | a device-resident traced input | no, by construction | a numeric knob you might sweep or schedule, **or a buffer set the graph reads and nothing trains** (see below) |
| unmarked / `Host{T}` | not at all | no | everything else, including anything dataset-sized |

**Getting it wrong is loud, in both directions.** Reading an unmarked field from traced code raises
with the field named and all three fixes spelled out. Over-marking `GraphConst` costs a recompile per
distinct value, which is visible in the cache counters. Neither mistake produces a plausible-looking
run. See `reactantnitro-recompiles` for what the key actually contains.

**Keep dataset-sized state unmarked, or off the experiment entirely.** Reactant and Enzyme traverse
the whole experiment while tracing, so anything dataset-sized reachable from it is walked element by
element, on one thread, every time a program compiles. It does not raise and does not change the
emitted graph: compile time simply grows with your data, which reads like a data-loading problem and
is not one. `Host` is the fix, and it is now the default, so this bites only a field someone marked
`GraphConst` by hand.

The generated accessors are `device_fields(T)`, `host_fields(T)`, and `config_metadata(T)`. There is
deliberately no `graphconst_fields`: two traits determine three categories and the third is the
`setdiff`. The macro is optional; define those three yourself and a hand-written struct works.

## A `Device` field may hold BUFFERS, not just a knob

The marker table above is written around numeric knobs, and that undersells it. A `Device` field may
hold a **buffer set**: arrays the graph reads and nothing differentiates, which is what PyTorch calls
buffers. The worked case is a FROZEN PRETRAINED NETWORK that a model calls inside its own `forward`,
through `Reactant.Ops.hlo_call` on an exported StableHLO module:

```julia
@experiment struct MyProbe
    feat_dim::GraphConst{Int} = 4096
    backbone_code::GraphConst{String} = ""    # the module text IS the graph: baked, and in the key
    backbone_w::Device{Any} = ()              # its weight buffers: traced inputs, never differentiated
end

ReactantNitro.derive(e::MyProbe, data) = (; backbone_code = ..., backbone_w = ...)   # host arrays

function ReactantNitro.forward(e::MyProbe, model, ps, st; img)
    feats = Reactant.Ops.hlo_call(e.backbone_code, img, e.backbone_w...)[2]
    return Lux.apply(model, feats, ps, st)[1], st
end
```

**Why the field and not `ps`.** Nothing trains them, so they do not belong in the parameter tree:
put them there and the optimizer carries state for every one of them, `make_zero` allocates a
gradient tree the size of the backbone, and the checkpoint records it all. Freezing with
`Optimisers.freeze!` does not fix that: it stops the UPDATE, not the differentiation.

**Why not `GraphConst`.** That bakes the values into the graph and the cache key. For a knob that is
the point; for tens of millions of weights it is a compile-time disaster.

**Why it is cheap to differentiate around**, which is the part that makes the pattern work: the
experiment reaches the gradient program as `Enzyme.Const(ev)`, so everything on it is constant to
the gradient. Enzyme has nothing to propagate through the backbone, keeps no tape for its forward,
and the only `Duplicated` argument in the program is the head's `ps`. A large frozen backbone in
front of a small head costs only the head.

**And it transfers once.** `to_device_config` converts `Device` leaves at setup; the per-step rebuild
re-converts the SCHEDULED entries only, and nothing converts inside the traced step. A buffer set is
not schedulable, so it crosses the bus once at setup and is passed by reference every step
thereafter. That is worth knowing before anyone "optimizes" it: at hundreds of megabytes, a
regression that started converting per step would present as an inexplicably slow epoch rather than
as an error.

**`derive` is usually the right place to fill it** rather than a field default, because a default is
eager: every `from_preset` would then read the weights off disk to answer a question about a preset.
`derive` runs once per `Nitro`, before device conversion, so it returns HOST arrays and the framework
places them.

**Then keep the backbone out of your dataloader.** The alternative shape, running the frozen
network in `build_data` and yielding features, puts a device program where the framework cannot
account for it, and the framework's prefetch fan-out will then run as many copies of it as there
are producer tasks. Many producers each holding a copy of the backbone's workspace exhaust device
memory within minutes, and the run ends in `RESOURCE_EXHAUSTED`.

## One flat struct: no configuration struct inside a field

**Every knob is a flat field on the experiment. Do not group knobs into a configuration struct and
hold it in a field.**

```julia
# YES
@experiment struct MyExp
    width::GraphConst{Int} = 128
    depth::GraphConst{Int} = 8
    smoothing::Device{Float32} = 0.05f0
end

# NO
@experiment struct MyExp
    model::GraphConst{ModelCfg} = ModelCfg()
end
```

If a model builder wants a config struct as its argument, assemble one inside `build_model` from the
flat fields, or have the builder take the experiment. Either keeps one declaration of each value.

**The reason that bites first: you cannot override one field of a nested struct.** `from_preset`
validates every key against `fieldnames(E)`, and the generated keyword constructor takes one keyword
per field. Neither reaches inside a struct. So a recipe wanting one different number has to name the
whole struct, and **that fills every field it did not name from the CONFIG's `@kwdef` defaults rather
than from the EXPERIMENT's default for that field**:

```julia
# experiment declares:  model::GraphConst{ModelCfg} = ModelCfg(width = 256, depth = 8)
# a preset writes:      (; model = ModelCfg(width = 512))
# and the run trains at depth = 4, the CONFIG's default. Nothing raises.
```

A call-site override has the same shape, since `MyExp(; model = ModelCfg(width = 512))` replaces the
field rather than merging into it. The workaround people reach for is a `_with(cfg; kw...)` merge
helper, which is a second mechanism to keep in sync and a second place to forget. Flat fields need
none: `from_preset(MyExp, :recipe; width = 512)` works and says exactly what it changes.

**A nested config is safe only by coincidence.** If the experiment's default for the field happens to
be a bare `ModelCfg()`, the fallback lands on the same values and nothing looks wrong. It stops being
true the day that default gains an argument, and then every partial preset silently reverts every
field it did not name.

Three more reasons, in the order they matter:

- **Residency is a per-FIELD property**, so a nested struct takes ONE marker for its whole group.
  Nothing inside a `GraphConst` struct can be `Device`, and `Device` is what makes a value sweepable
  and schedulable with no recompile. Burying a loss weight in a config struct decides, without
  meaning to, that changing it recompiles the gradient program.
- **A struct in a `GraphConst` field is a hashing hazard, and the WRAPPER is the whole defect.**
  Julia's fallback `hash` for a struct with no method of its own is `hash(objectid(x), h)`, and
  `objectid` reaches a mutable field such as a `Vector` by identity rather than descending into it.
  Two identical constructions then key differently (a recompile per handle, and a resume refused with
  a diff whose sides print identically), and one mutated in place keys the SAME, which serves a
  program compiled for the old value while the resume check passes. The same vector held directly as
  a `GraphConst{Vector{Int}}` field is content-hashed in both directions by Base, with nothing
  written by hand. `assert_graphconst_hashable` refuses the struct case at setup and recommends
  flattening first.
- **`config_metadata` and the logged hyperparameter table are per FIELD.** A non-scalar field logs as
  the bare string of its type name, so a nested config contributes one useless entry and none of its
  values reach the table or any run comparison. Flat fields each log by name.

**What not to flatten.** A `Host` field holding dataset-sized state (samplers, in-memory frames,
handles) stays one field: it is not a group of knobs, and the point is that `compile_view` replaces
the whole field with a sentinel so the tracer never walks it. A config type that is some other
function's signature stays a type, but assemble it at the call site rather than storing it. And note
the cost is asymmetric: `@experiment` generates one type parameter per `Device` and per `Host` field
and **none** per `GraphConst`, so flattening an architecture config is free in that respect while
flattening a large `Host` config is not.

**Keep the named recipes in their own file.** Once the fields are flat, `presets` is a table of
partial `NamedTuple`s, and it reads better beside the experiment than inside it, organized by
whatever your recipes vary over.

**Write the table as METHODS, and anchor it to no `const`.** Revise updates a method body and
cannot rebind a `const`, so a shared base held in one pins the whole table to whatever it contained
when the session loaded: an edited preset keeps serving the old numbers, silently, until the
process restarts. That is worse here than in ordinary code, because a restart also gives up
whatever the process was holding: on a shared accelerator host, the GPU goes back when the process
exits, so a one-character preset edit turns into acquiring a card again and paying the package load
a second time.

```julia
# The shared base as a function, so editing it takes effect on the next call.
my_preset_base() = (; width = 256, max_epochs = 40, batch_size = 64)

ReactantNitro.presets(::Type{MyExp}) = (
    smoke   = (; my_preset_base()..., max_epochs = 1, batch_size = 8),
    current = (; my_preset_base()..., dataset = :full),
)
```

```julia
const MY_PRESET_BASE = (; width = 256, ...)      # DON'T: Revise cannot rebind it
```

The cost is one allocation per `from_preset`, on a call made once per run. Restarting to pick up a
preset edit also releases and reacquires any leased accelerator, which is expensive on a shared
host.

(A `@config` macro that generated the three hashing methods existed briefly and was **withdrawn**.
Do not look for it: shipping a convenience for the discouraged shape worked against the layout the
rest of the framework assumes.)

## The batch contract

A data source is anything iterable yielding concrete `NamedTuple`s of host arrays and supporting
`length`. The framework never looks inside your data and ships no batching or shuffling.

**That is the whole contract, and it is deliberately small enough to adapt to rather than adopt.**
Three requirements: iterate, yield `NamedTuple`s of host arrays with concrete element types, and
answer `length`.

**If you have no loader, use the ecosystem's.** `MLUtils.jl` supplies batching and shuffling, and
`MLUtils.DataLoader` satisfies the contract directly once its tuples are named. Reach for it rather
than writing a sampler, unless you have a reason not to.

**If you already have a loader, adapt it; do not replace it.** Three requirements is usually less
work than porting a data pipeline, and a loader that already knows your storage, your augmentation,
and your sampling is not something to rewrite for the sake of a batching helper. A wrapper that maps
the loader's item to a `NamedTuple` and forwards `length` is the whole adapter.

**Prefetch is a framework default, and it changes nothing about the contract.** Setup wraps every
split, eval included, in a `PrefetchIterator` at `workers = Threads.nthreads(:default)`,
`device_batches = 1` (staged on the device), `host_batches = 2 * workers` (existing on the host) and
ordered delivery, unless the source is already a `PrefetchIterator` or a `NoPrefetch`. The wrapper
forwards `length` and its `iterate` is a passthrough, so the three requirements above are the whole
contract. `NoPrefetch` declines; `ordered = false` trades a bitwise-reproducible epoch order for
immunity to a straggling batch. Real concurrency needs the two-method trait, `batch_at(source, i)`
and `begin_epoch!(source)`. The handle's display shows each split's resolved `workers`,
`device_batches`, `host_batches` and path.

**The source trait, in full, and who implements it.** Five functions, all owned by ReactantNitro
and extended on the source type: `begin_epoch!(source)` (the per-epoch plan; return it, or
`nothing` when the source stores it), `batch_at(source, i)` or `batch_at(source, i, plan)`
(the two storage choices; `i` is a BATCH index), `epoch_token(source)` (a counter the fan-out
asserts advances once per epoch), `check_source_options(source, name, cfg)` (setup-time refusal
of options the resolved pipeline cannot honour), and `release!(source)` (close what the source
holds open, called once per handle at `Done` or `Failed`, or by `release!(nitro)` for a handle
that only evaluated or exported). All but the first two are unexported and written qualified.
One implementer per source type: the framework carries the extension for a public loader it
chooses to support (`ReactantNitroMLUtilsExt`); any other loader's package declares
ReactantNitro as a weak dependency and ships the extension itself. Defining the trait on a type
you do not own is the last resort.

**Runs leave the interactive thread, and ^C is a graceful stop.** Every entry point
(`train!`, `validate`, `evaluate`, `predict`, `export_model`) runs its body on a default-pool
worker thread when one exists, so a long XLA compile or execute never starves your logger tasks
or the REPL; the caller's task parks, and the REPL keeps working while the loop runs. ^C then
requests the graceful stop: `train!` finishes the epoch, validates, checkpoints, and returns
with `stop_reason = :requested` (the `request_stop!` wind-down), and the eval entry points stop
at the next batch boundary. Single-threaded Julia runs inline, unchanged.

**A field no hook declares reaches nobody**, so a bookkeeping field costs nothing and is allowed to
ride along, which means an adapter usually does not have to strip anything either.

**Hooks are routed by their keyword names.** `forward(e, model, ps, st; x)` declares it wants `x`;
`loss(e, ŷ; y)` declares `y`. Only the fields some hook declares are transferred to device, so a
bookkeeping field nothing declares costs nothing and is allowed. A required keyword the batch does
not provide is an error naming the hook, the keyword, and the batch's fields.

Every array leaf must have a concrete element type. A batch field like `Vector{NamedTuple}` fails
inside the tracer with an error that names neither the field nor the batch, so the framework rejects
it first.

## What a hook may RETURN, which is the mirror rule

`forward` may return any tree: an array, a tuple, a `NamedTuple` of several outputs. **The batch
dimension must be last on every array leaf of it**, exactly as it is on every batch field, because
the framework slices a short final evaluation batch back to its real width and slices the last
dimension to do it.

**So a reduction over the batch cannot be an output leaf.** A total, a mean, an accumulated penalty,
anything of shape `()`: each has no batch dimension, and each is rejected. Broadcast it to a `(1, N)`
row and reduce it again wherever you consume it.

Two reasons that rule is worth knowing before you meet it:

- **The check runs only on a short final batch**, so a split whose length divides the batch size
  never triggers it. A loader that drops its partial batch never triggers it at all, and then the
  defect is unreachable on that dataset in either direction.
- **A device-resident scalar cannot be a program output on more than one device**, whatever the
  slicing does, because a replicated array can be reconstructed and a replicated scalar cannot. A
  `(1, N)` row satisfies both rules at once.

## Derived values

`derive(e, data) -> NamedTuple` is merged into the experiment, for anything that genuinely depends on
the dataset (class weights from label counts, a normalization constant). It runs **before** device
conversion, so it always returns host values, and it is recomputed rather than restored on resume.

Prefer constants for anything structural. Deriving a shape makes the compiled program a function of
your data, so a different split silently changes the graph.

## Where to go next

- `reactantnitro-metrics`: the `(sum, count)` contract, `finalize_metrics`, residency
- `reactantnitro-optimizer`: parameter groups, decay, clipping, schedules
- `reactantnitro-manual`: manual training mode, where the experiment owns the step
- `reactantnitro-recompiles`: the compile key, `cache_stats()`, why an edit did not take effect
- `reactantnitro-checkpoint-resume`: top-K retention, what resume refuses
- `reactantnitro-device-boundary`: why a green CPU test suite says nothing about GPU behaviour
- `reactantnitro-visualization`: the two hooks and the `render` driver, and what to draw at all
- `reactantnitro-export`: shipping `forward` as an artifact, and why what ships is not what trained
- `reactantnitro-kaimon`: driving runs from a Kaimon-hosted session, where an experiment is named as a type string
