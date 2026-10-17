# Experiments: declaring a run

## An experiment is a struct plus hooks

An experiment is an ordinary Julia struct plus hook methods. The struct declares the knobs and
their markers; the hooks define what the run does with them. The framework owns everything else:
the compiled programs, the device transfers, the optimizer (see [Optimization](optimization.md)),
the schedules (see [Schedules](schedules.md)), and the run's lifecycle. [`Nitro`](@ref)`(e)` runs
setup and nothing else, so evaluation, prediction and export work on a handle that has never
trained; `train!(nitro)` runs the loop.

Four hooks are required:

```julia
build_model(e, rng) -> (model, ps, st)      # standard Lux setup
build_data(e, dist) -> NamedTuple of splits # (; train, val, test); `dist` is nothing today
forward(e, model, ps, st; <batch fields>) -> (outputs, st_new)
loss(e, outputs; <batch fields>) -> scalar
```

- [`build_data`](@ref) returns the data as a NamedTuple of splits, `(; train, val, test)`. `dist`
  is the distribution handle, always `nothing` in this version. It runs against the real
  experiment, before device conversion.
- [`build_model`](@ref) builds the model and its initial parameters and state with the standard
  Lux setup, returning `(model, ps, st)`. It sees the experiment, so it can read structural fields
  such as widths and class counts.
- [`forward`](@ref) is the exportable program, the one `predict` calls. It takes inputs only,
  never targets, so the same method serves training, validation, and inference. It returns
  `(outputs, st_new)`.
- [`loss`](@ref) turns the outputs into a scalar. It sees the outputs plus the label fields, which
  `forward` never sees. Do not fold the loss into `forward`, or `predict` and export would need
  labels.

Every other hook has a default. An experiment that defines only these four trains, validates, and
checkpoints.

## What each hook receives, and what it passes on

Two things move between the hooks: the **batch**, which the framework routes, and the **outputs**,
which [`forward`](@ref) produces and the framework hands on unchanged.

### The batch arrives as keywords, and only the fields you declare

A batch is a `NamedTuple` of host arrays with the batch dimension last. Each hook declares the
fields it wants as keyword arguments; the framework resolves the routing once at setup and passes
each hook only its fields:

```julia
forward(e, model, ps, st; img)        # gets `img` only
loss(e, outputs; label)               # gets `label` only
```

A field no hook declares is never transferred to the device. A required keyword missing from the
batch raises; a keyword with a default is an optional field and takes its default when absent, so a
typo in a defaulted keyword also takes the default. A method ending in `kwargs...` receives the
whole batch, and nothing is checked for it.

### `forward` returns `(outputs, st_new)`, and the framework keeps only the first

```julia
outputs, st_new = forward(e, model, ps, st; img)
```

The framework carries `st_new` into the next step in training and discards it in eval mode.
Every downstream hook receives `outputs` as **one positional argument**:

```julia
loss(e, outputs; label)
metrics(e, outputs; label)
train_metrics(e, outputs; label)
```

Nothing is splatted, unpacked, or renamed in between: `loss` receives, as its second argument,
exactly what `forward` returned first.

Do not unpack `outputs` a second time. If `forward` returns a bare array, `first(outputs)` inside
`loss` is element one of that array. Every operation after it still broadcasts, so the run trains on
one number out of the batch and never raises.

### Multiple outputs are a `NamedTuple`

When a model produces more than one thing, return them named and destructure by name downstream:

```julia
function ReactantNitro.forward(e::Seq2Seq, model, ps, st; tokens)
    logits, st_dec = Lux.apply(model.decoder, tokens, ps.decoder, st.decoder)
    energy, st_aux = Lux.apply(model.aux, tokens, ps.aux, st.aux)
    return (; logits, energy), merge(st, (; decoder = st_dec, aux = st_aux))
end

ReactantNitro.loss(e::Seq2Seq, out; target) =
    cross_entropy(out.logits, target) + e.aux_weight * mean(abs2, out.energy)

ReactantNitro.metrics(e::Seq2Seq, out; target) =
    (; acc = n_correct(out.logits, target) => size(target)[end])
```

A `Tuple` or a nested structure also works: the framework walks the output tree with `Functors`.
Prefer a `NamedTuple`, because [`export_outputs`](@ref) names the leaves to export by key, and
`out.logits` is readable at every call site.

**Every leaf of the output tree must be an array with the batch dimension last.** A short final
eval batch is sliced along the last dimension; the framework checks that dimension is the batch
size and raises otherwise. A scalar already reduced over the batch has no batch dimension to slice,
so reduce it in [`loss`](@ref) or [`metrics`](@ref) instead of returning it from `forward`.

### Where the outputs end up

| Stage | What runs | What `outputs` is by the time you see it |
| --- | --- | --- |
| training | `forward` then `loss`, in one traced program | exactly what `forward` returned |
| training metrics | `forward` then `train_metrics` | the same, once per micro-batch |
| validation | `forward` then `metrics` | sliced back to the real sample count, so padding is invisible |
| prediction | `forward` alone | sliced, and converted to host arrays |
| export | `export_preprocess` then `forward` | the leaves [`export_outputs`](@ref) names, which may be a subset |

Validation and prediction differ from training because of padding: a short final eval batch is
padded up to the compiled batch size, the program runs at that size, and the outputs are sliced back
before any of your hooks sees them. [`metrics`](@ref) never sees a padded sample.

## The three markers, and the default

Each experiment field is one of three kinds, chosen with a marker; an unmarked field is `Host`:

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

| Marker | Reaches traced code as | In the compile key? | Reach for it when |
| --- | --- | --- | --- |
| `GraphConst{T}` | a baked literal, ordinary Julia semantics | yes, changing it recompiles | the value changes the graph: a shape, a class count, a layer width, a variant selector |
| `Device{T}` | a device-resident traced input | no, by construction | a numeric knob you might sweep or schedule |
| unmarked, i.e. `Host{T}` | not at all | no | everything else, including anything dataset-sized |

Unmarked means `Host` because most fields are. Writing `Host{T}` explicitly is legal, and useful
where a reader would not expect the field to be host-only.

A `GraphConst` field bakes as a trace-time constant and is in the compile cache key, so changing it
recompiles: a different value is a different program. A `Device` field is moved to the device at
setup and reaches traced code as an input, so changing or scheduling its value does not recompile.
[Recompilation](recompilation.md) lists what the key contains; [Schedules](schedules.md) covers
scheduling.

Both mistakes are visible. Using an unmarked field in traced code raises, naming the field and the
fixes. Marking a field `GraphConst` that should be `Device` costs a recompile per distinct value,
visible in `ReactantNitro.cache_stats()`.

Keep dataset-sized state unmarked, or off the experiment entirely; the `compile_view` section below
says why.

## Why a macro

The markers are recorded by a macro because they cannot be recovered later. After setup a
`Device` field holds a device value (a `ConcretePJRTNumber` for a scalar, a `ConcretePJRTArray` for
an array), not the marker type, so the field types at runtime do not say which fields were marked.
The macro records the markers as traits at declaration.

Each `Device` and `Host` field gets its own free type parameter on the generated struct. That is
what lets one field hold a host value before setup and a device value after, and lets a `Host`
field hold its real value in `e` and a `ReactantNitro.StrippedHost` in `compile_view(e)`. `GraphConst` fields
keep their declared concrete type, and the generated keyword constructor converts to it, so
`width::GraphConst{Int}` always holds an `Int`.

The recorded traits are [`device_fields`](@ref), [`host_fields`](@ref), and
[`config_metadata`](@ref). There is no `graphconst_fields`: the `GraphConst` fields are the rest,
`setdiff(fieldnames(E), device_fields(E), host_fields(E))`. The macro also generates the keyword
constructor and a specialized [`compile_view`](@ref) method, and it appends a marker table beneath
the struct's docstring.

The macro is optional. A hand-written struct plus those three trait definitions works, which is why
the three are exported; `compile_view` and `config_metadata` have fallbacks that read the traits for
a type that never used the macro. The [API](api.md) page links each of them.

## compile_view and the stripped view

While tracing, Reactant and Enzyme traverse the whole experiment, because it is an argument to
the compiled program. A dataset-sized field reachable from it is walked element by element, on one
thread, on every compile. The cost grows with the data, does not change the emitted graph, and
raises nothing, so it looks like slow data loading.

The guard is [`compile_view`](@ref), which every trace site receives instead of `e`. It replaces
every `Host` field with a `ReactantNitro.StrippedHost` sentinel that carries only the field's
name, as a type parameter, so there is nothing left to walk:

```julia
compile_view(e).images     # ReactantNitro.StrippedHost{:images}()
compile_view(e).width      # 128, unchanged: a GraphConst field still bakes
compile_view(e).smoothing  # unchanged: a Device is a traced input
```

The real `e` is used everywhere outside the trace: `build_data`, `derive`, metric finalization,
checkpointing, and every driver decision read `e.images` normally. Only traced code sees the
sentinel, and using it there (arithmetic, indexing, property access, conversion, iteration) raises
with the field named. A sentinel that is only stored or passed through is not detected. It is a
sentinel rather than `nothing` because `nothing` passes through an `::Any` signature without an error.

Keep dataset-sized state unmarked (`Host`), or off the experiment entirely. Off the experiment is
usually better: `build_data` hands the collection to the framework, which holds it apart from `e`.
Since unmarked already means `Host`, the remaining risk is a dataset-sized field marked
`GraphConst`.

## One flat struct: no configuration struct inside a field

Every knob is a flat field on the experiment. Do not group knobs into a configuration struct and
hold it in one field:

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

If a model builder wants a config struct as its argument, assemble one inside `build_model` from
the flat fields, or have the builder take the experiment. Either keeps one declaration of each
value.

**Overrides replace the whole struct.** [`from_preset`](@ref) validates every key against
`fieldnames(E)`, and the generated keyword constructor takes one keyword per field; neither reaches
inside a struct. A recipe that changes one number has to supply the whole struct, and every field it
does not name takes the config type's own default, not the experiment's default for that field:

```julia
# ModelCfg's own default:  depth = 4
# experiment declares:     model::GraphConst{ModelCfg} = ModelCfg(width = 256, depth = 8)
# a preset writes:         (; model = ModelCfg(width = 512))
# and the run trains at depth = 4, ModelCfg's default. Nothing raises.
```

A call-site override behaves the same: `MyExp(; model = ModelCfg(width = 512))` replaces the field
rather than merging into it. A merge helper such as `_with(cfg; kw...)` is a second mechanism to
keep in sync. With flat fields, `from_preset(MyExp, :recipe; width = 512)` changes exactly one
value.

The problem is hidden while the experiment's default is a bare `ModelCfg()`, because the config
type's defaults are then the same values. Once the default gains an argument, every partial preset
reverts every field it did not name.

There are three more reasons. Residency is per field, so a nested struct takes one marker for its
whole group: nothing inside a `GraphConst` struct can be `Device`, so a loss weight in a config struct
recompiles the gradient program whenever it changes.

A struct in a `GraphConst` field may hash by identity. Julia's fallback `hash` for a struct with no
method of its own hashes `objectid(x)`, and `objectid` reaches a mutable field such as a `Vector`
by identity rather than by content. Two identical constructions then key differently, and one
mutated in place keys the same, which would serve a program compiled for the old value. The same
vector held directly as a `GraphConst{Vector{Int}}` field is hashed by content.
`ReactantNitro.assert_graphconst_hashable` refuses the struct case at setup and recommends
flattening first.

The logged hyperparameters are per field too. A non-scalar `GraphConst` or `Device` field logs as
its type name (an array as its type name and size), and a non-scalar `Host` field is not logged, so
none of a nested config's values reach the logger or any run comparison.

**What not to flatten.** A `Host` field holding dataset-sized state (samplers, in-memory frames,
handles) stays one field: it is not a group of knobs, and `compile_view` replaces the whole field
with a sentinel so the tracer never walks it. A config type that another function takes as its
argument stays a type, but assemble it at the call site rather than storing it. `@experiment`
generates one type parameter per `Device` and per `Host` field and none per `GraphConst`, so
flattening an architecture config adds no type parameters, while flattening a large `Host` config
adds one per field.

## derive and named configurations

[`derive`](@ref)`(e, data)` returns a NamedTuple that the framework merges into the experiment, for
values that depend on the dataset: class weights from label counts, a normalization constant. Every
key must name an existing field. It runs after `build_data` and before device conversion, so it
returns host values and the framework places them. A resume recomputes it; `weights = path`
restores the derived values from the checkpoint instead, so an evaluation process needs no training
data:

```julia
function ReactantNitro.derive(e::MyExp, data)
    counts = [count(==(c), labels) for c in 0:9]         # from the training split
    (; class_weights = Float32.(sum(counts) ./ (10 .* counts)))
end
```

Prefer constants for anything structural. A derived shape makes the compiled program depend on
the data, so a different split changes the graph. And a large derived value should be
`Device`, not `GraphConst`: a `GraphConst` array would be both baked into the graph as a constant
and walked by the tracer on every compile, while a `Device` value is converted once and crosses as
a single buffer.

[`presets`](@ref) is a table of named configurations on the experiment type: partial NamedTuples of
field values, empty by default. [`from_preset`](@ref)`(E, name; overrides...)` builds the
experiment:

```julia
ReactantNitro.presets(::Type{MyExp}) = (
    reference_v1 = (; width = 128, smoothing = 0f0),
    current      = (; width = 256, smoothing = 0.05f0),
)

e = from_preset(MyExp, :current)                   # the experiment, from the recipe
e = from_preset(MyExp, :current; max_epochs = 40)  # overrides win over the preset
```

Every preset key is validated against `fieldnames(E)`, since an unknown key would otherwise be
ignored. A preset is only field values, so the markers apply as usual: switching presets recompiles
only when a `GraphConst` field differs, so two recipes differing only in `Host` or `Device` fields
share every compiled program. A preset may set a `Device` field too; `derive` wins for anything
it computes. `from_preset` returns a bare experiment and records no name; `Nitro(E, name; ...)`
also logs the preset name with the run's hyperparameters. Keep the recipes in their own file,
beside the experiment.

## The Revise workflow

The [Tutorial](tutorial.md) walks through this workflow; this section is the reference.

Everything that decides which compiled program a `Nitro` runs is resolved at construction: the run
keywords, the accessor values, and the hook methods are frozen into the handle. A revision does not
change an existing handle, and a handle never recompiles because of one.

Revise, then build a new `Nitro`:

```julia
# You edit `loss` in your editor, then:
train!(n)                                        # still the OLD loss, and it tells you

n = Nitro(e; data = n.data)                      # picks up the edit, reuses the collection
train!(n)                                        # compiles only what actually changed
```

A rebuild is cheap because the compile cache is module-level rather than per `Nitro`: a new
handle on the same experiment reuses the entries the previous one compiled and recompiles only what
the edit changed. [Recompilation](recompilation.md) covers the key. The rest of a rebuild is setup:
device conversion, one batch to infer routing, and `derive`. That takes seconds; a recompile can
take minutes.

A new method on a host-side object the handle stores does apply to an existing handle. The handle
stores the object, not its methods, so the new code runs on the next `train!` call:

```julia
# A `should_stop` method on your own stopper. No recompile, new behaviour.
ReactantNitro.should_stop(es::MyStopper, epoch, metrics) = ...
train!(n)
```

The one exception to freezing is a `Device` value, changed with [`set_device!`](@ref) on the live
handle. `Device` fields are not in the compile key, and `set_device!` checks that the key is
unchanged before writing, so it reuses the compiled programs:

```julia
for t in (0.5f0, 1f0, 1.5f0, 2f0)
    set_device!(n; temperature = t)
    logits = predict(n, batch)                   # zero compiles, every iteration
end
device_value(n, :temperature)                    # reads it back as a HOST value
```

`set_device!` refuses, with an error naming the fix: a `GraphConst` field (a different program),
a `Host` field (no compiled program reads it), and a scheduled field (the schedule would overwrite
the write). For those, build a new `Nitro`. It converts the value to the field's host type, so
`0.5` for a `Device{Float32}` is accepted, and refuses a value that does not convert. It also
refuses an array of a different size: size is not in the key, so the old program would run and
fail inside XLA.

A stale handle says so. When something changed since construction, `train!`, `validate` and
`evaluate` print a report naming any redefined hook, any compiled program whose dependency methods
were redefined, and any scalar accessor whose value changed (`train!` checks every scalar accessor;
`validate` and `evaluate` check `seed` and `run_dir`):

```julia
ReactantNitro: MyExp has values fixed at construction that have since been redefined. This handle
keeps what it froze; rebuild the `Nitro` to pick up the new ones, and see `show(nitro)` for what it
is currently holding.
  ! `accum(e)` was redefined: it returned 2 at construction and returns 4 now. This handle still
uses 2; rebuild to pick up 4.
  ! hooks were redefined after this `Nitro` was built (loss). It will keep running the programs it
    was built with; rebuild the `Nitro` to pick up the new code. ...
```

Nothing is printed when nothing changed; `show(nitro)` lists the frozen values. The accessor check
compares the accessor's value now against its value at construction, so a keyword you passed is
not reported as stale. The handle keeps its frozen values because `accum` and `max_epochs` together
determine the schedule horizon `total`, and applying one revised value would leave every schedule
resolved against the old horizon. Silence the report in a scripted driver with
`ReactantNitro.set_config_report!(false)`.

Two limits. Only scalar accessors are compared: an accessor that constructs an object
(`early_stop`, `checkpointer`, `logger`) is never called again, since calling it could repeat a side
effect the framework runs exactly once. And a `const` or global value read from a hook is not in
the key: changing it reuses the old program until the REPL restarts. A redefined helper that
`forward` calls is caught by the dependency check. Edits made while `train!` is running do not
affect that call: it keeps the world age it started in.

Build a new `Nitro` for a new training job; reuse an existing one for more validation,
evaluation, prediction, and `Device` sweeps.
