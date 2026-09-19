# Binding cheat sheet

Everything in ReactantNitro connects by **name**. A batch field reaches the hook that declares a
keyword of that name. A schedule key names the slot it drives. A run keyword names the accessor it
overrides, which names the field it reads. This page is every one of those connections on one
screen, drawn once, with the rule under each drawing.

## The three markers: where a field goes

```julia
@experiment struct MnistMLP
    width::GraphConst{Int} = 128          # baked into the compiled graph as the literal 128;
                                          # in the compile key, so a new value is a new program
    smoothing::Device{Float32} = 0.05f0   # a device scalar, a traced INPUT to every program;
                                          # not in the key: sweep it, schedule it, set_device! it
    images::Matrix{Float32} = load_mnist() # Host (unmarked): stays on the driver;
                                          # the trace sees StrippedHost{:images}(), nothing walked
end
```

What each hook sees when it reads `e.<field>`:

| hook | `e.width` (GraphConst) | `e.smoothing` (Device) | `e.images` (Host) |
| --- | --- | --- | --- |
| `build_data`, `derive` | `128` | `0.05f0`, still a host value | the matrix |
| `build_model` | `128` | a device scalar | the matrix |
| `forward`, `loss`, traced `metrics` | `128`, a constant | a traced scalar | **error** naming the field |
| `finalize_metrics`, accessors, checkpointing | `128` | a device scalar | the matrix |

Unmarked is `Host`. `Device` is for a number you might change without recompiling. `GraphConst` is
for a value that changes the graph.

## The batch: fields to hooks, by keyword name

```julia
batch = (; img, label, idx)          # what the loader yields; the batch dimension is LAST in each
forward(e, model, ps, st; img)       # declares `img`, receives `img`
loss(e, outputs; label)              # declares `label`, receives `label`
metrics(e, outputs; label)           # the same
                                     # `idx`: no hook declares it, so it never leaves the host
```

- A hook's keyword arguments are the batch fields it wants, by name. The framework resolves the
  method once at setup and passes exactly that subset.
- Only declared fields are transferred to the device. A field nobody declares rides along for free.
- A required keyword the batch lacks raises on the hook's first call, naming both sides. A keyword
  with a default is optional. A method ending in `kwargs...` receives the whole batch.
- `forward` declares inputs and never targets, which is why `predict` works on a batch with no
  labels.

## The outputs: `forward` to everything downstream

```julia
outputs, st_new = forward(e, model, ps, st; img)  # st_new is threaded back into the layer state
                                                  # and discarded in eval mode; `outputs` moves on
                                                  # EXACTLY as returned, as one positional argument:
loss(e, outputs; label)                 # a scalar, differentiated in the gradient program
train_metrics(e, outputs; label)        # scalars per step, traced beside the loss
metrics(e, outputs; label)              # (sum, count) per eval batch; padding already sliced off
predict(nitro, batch)                   # the same outputs, returned as host arrays
```

- Whatever `forward` puts in the first slot is what `loss` receives in its second. Nothing is
  splatted, unpacked, or renamed in between.
- **Do not unpack `outputs` a second time.** If `forward` returns a bare array, `first(outputs)`
  is element one of it, every later operation stays broadcast-legal, and the run trains on one
  number without raising.
- Several outputs: return a `NamedTuple` and destructure by name downstream. Every array leaf keeps
  the batch dimension last, and nothing already reduced over the batch belongs in `outputs`.

## Metrics: the pair, and where each hook runs

```julia
metrics(e, outputs; label) = (;
    acc = (sum(correct), size(label, 2)),   # per image: summed, then divided by the summed count
    confusion = (cm, nothing),              # summed across the split, never divided
)
finalize_metrics(e, acc, split)             # acc.acc is a mean; acc.confusion is the split's matrix
```

| hook | default residency | cadence | edit it and... |
| --- | --- | --- | --- |
| `train_metrics` | `:device`, traced with the loss | every micro-batch | the gradient program recompiles |
| `metrics` | `:host`, ordinary Julia | every eval batch | nothing recompiles |

`metrics_residency(e, hook)` flips either. No `metrics` method at all substitutes `val_loss` with a
count of 1.

## Schedules: a key to its slot

```julia
schedules(e) = (;
    eta = total -> OneCycle(total, 1f-3),   # a rule field: every group's `eta`, ratio-scaled
    smoothing = _ -> t -> 0.1f0 * t / 5000, # a Device field: e.smoothing, written each step
    opt = (; encoder = (; eta = ...)),      # path-bound: the :encoder group's chain only
    device = (; lambda = _ -> t -> 0.5f0),  # qualified: e.lambda; `lambda` is a rule field too
)
```

- A bare key resolves against two namespaces: the optimizer rules' field names (`opt`) and the
  experiment's `Device` fields (`device`). One match binds; two is an ambiguity error that shows
  the qualified forms; none is an error naming the fix.
- Every entry is a factory of the horizon: called once with the total optimizer steps, then once
  per step with the upcoming step, counting from 1. A bare `Number` is a constant.
- A `GraphConst` cannot be scheduled: its value is the program. A `Host` field is in neither
  namespace.
- `opt.eta` scales every group by that group's ratio, `learning_rate(e, Val(g)) / learning_rate(e)`,
  so one curve moves the groups together.

## Parameters: leaves to groups to one buffer each

```julia
param_group(e, keypath)         # each leaf of `ps` to a group: :encoder, :default, ...
                                # one flat buffer and one rule chain per group
learning_rate(e, Val(g))        # a RATIO against learning_rate(e); a schedule scales the base
lambda(e, Val(g))               # decoupled decay toward decay_anchor(e, Val(g)): :zero, :w0, array
no_decay(e, keypath, x)         # per leaf: excluded from decay; the default excludes 1-D leaves
```

The optimizer program applies each chain once per group, not once per array, which is what keeps
its compile small. `gradient_clip_norm(e)` is a global norm over the whole accumulated gradient,
applied before any group, and is not per group or schedulable.

## Run keywords: keyword to accessor to field

```julia
Nitro(e; max_epochs = 60)   # the keyword wins for THIS run;
max_epochs(e)               # else the accessor: define a method for the experiment's own default;
e.max_epochs                # else a field of the same name, when the struct declares one;
1                           # else the framework's value
```

Ten keywords follow this chain: `seed`, `run_dir`, `n_devs`, `accum`, `max_epochs`, `schedules`,
`gradient_clip_norm`, `logger`, `checkpointer`, `early_stop`. `schedules` is the one whose accessor
reads no field. Four keywords name a fact about this construction and have no accessor: `data`,
`checkpoint`, `resume`, `run_ref`. `weights` and `w0` are two more of that kind.

## Weights: the four ways in

| construction | weights | optimizer | epoch | `derive` |
| --- | --- | --- | --- | --- |
| `Nitro(e)` | `build_model`'s init | fresh | 0 | runs |
| `Nitro(e; checkpoint = path)` | the record's | fresh | 0 | skipped, values restored |
| `Nitro(e; resume = :auto)` | the latest record's | restored | restored | runs |
| `Nitro(e; weights = other)` | `other`'s `ps` and `st` | fresh | 0 | runs |

`w0`, the anchor for `decay_anchor = :w0`, is `build_model`'s init unless `w0 = :weights` or a tree
says otherwise.

## What recompiles

| you change | what compiles again |
| --- | --- |
| a `Device` field, a `Host` field, `seed`, `run_dir`, `max_epochs`, a schedule | nothing |
| a `GraphConst` field | every program that reads it |
| `accum` | the gradient program |
| `gradient_clip_norm` | the optimizer program |
| `forward` or `loss`, or a helper they call | the gradient and evaluation programs |
| a `:device` metric | that metric's program |
| a `:host` metric, `finalize_metrics`, an accessor | nothing |

A `Nitro` is a fixed point: none of these land on an existing handle. Build a new one and the
module-level cache recompiles only what moved. `ReactantNitro.cache_stats()` is the check.

## Where to read what bound

`show(nitro)` prints the handle and, under it, the bindings: each split with its batch count and
prefetch path, the clip threshold and every schedule key with its source, and one row per parameter
group with rate, ratio, anchor, decay and rule. The same text goes to the logger at setup as
`binding_report`, so the run's record carries it. After training, [`history`](@ref)`(nitro)` is the
run as a table.
