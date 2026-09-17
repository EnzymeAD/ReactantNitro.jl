# Experiments: declaring a run

## An experiment is a struct plus hooks

An experiment is an ordinary Julia struct plus a handful of hook methods. The struct declares the
knobs and their markers; the hooks define what the run does with them. The framework owns
everything else: the compiled programs, the device transfers, the optimizer (see
[Optimization](optimization.md)), the schedules (see [Schedules](schedules.md)), and the run's
lifecycle. [`Nitro`](@ref)`(e)` runs setup and nothing else, so evaluation and serving never
require a `train!` to have happened; `train!(nitro)` runs the loop.

Four hooks are required, and these are all of them:

```julia
build_model(e, rng) -> (model, ps, st)      # standard Lux setup
build_data(e, dist) -> NamedTuple of splits # (; train, val, test); `dist` is nothing today
forward(e, model, ps, st; <batch fields>) -> (outputs, st_new)
loss(e, outputs; <batch fields>) -> scalar
```

- [`build_data`](@ref) produces the data as a NamedTuple of splits, `(; train, val, test)`; `dist`
  is the distribution handle, `nothing` in this version. It runs against the real experiment,
  before device conversion.
- [`build_model`](@ref) builds the model and its initial parameters and state with the standard
  Lux setup, returning `(model, ps, st)`. It sees the experiment, so it can read structural fields
  such as widths and class counts.
- [`forward`](@ref) is the exportable program, the one `predict` calls. It takes inputs and never
  targets, which is why the same method serves training, validation, and inference. It returns
  `(outputs, st_new)`.
- [`loss`](@ref) turns the outputs into a scalar. It sees the outputs plus the label fields, which
  is exactly why the split exists: `forward` sees only inputs. Do not fold the loss into `forward`.

Everything else has a default, so an experiment defining only these four trains, validates, and
checkpoints with nothing else passed. That claim is stronger than each default being individually
stated, and it is tested. When you are unsure whether something needs a hook, assume it does not and
check the default first.

## What each hook receives, and what it passes on

Two things move between the hooks: the **batch**, which the framework routes, and the **outputs**,
which [`forward`](@ref) produces and the framework hands on unchanged.

### The batch arrives as keywords, and only the fields you declare

A batch is a `NamedTuple` of host arrays with the batch dimension last. Each hook declares the
fields it wants as keyword arguments, and the framework resolves that once at setup and passes
exactly that subset:

```julia
forward(e, model, ps, st; img)        # gets `img` only
loss(e, outputs; label)               # gets `label` only
```

A field no hook declares is never transferred to the device. A keyword with a default is an optional
field. A method ending in `kwargs...` receives the whole batch, and nothing is checked for it.

### `forward` returns `(outputs, st_new)`, and the framework keeps only the first

```julia
outputs, st_new = forward(e, model, ps, st; img)
```

The framework threads `st_new` back for you and discards it in eval mode. Everything downstream
sees `outputs`, and sees it as **one positional argument**:

```julia
loss(e, outputs; label)
metrics(e, outputs; label)
train_metrics(e, outputs; label)
```

Nothing is splatted, unpacked, or renamed in between. Whatever `forward` put in the first slot is
exactly what `loss` receives in its second.

**The trap that follows from that** is worth stating once: if `forward` returns a bare array, then
`first(outputs)` inside `loss` is element one of that array, not "the outputs". Every operation
after it stays broadcast-legal, so the run trains on one number out of the batch and never raises.
Do not unpack `outputs` a second time.

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
    (; acc = (n_correct(out.logits, target), size(target)[end]))
```

A `Tuple` works too, and so does a nested structure: the framework walks the output tree with
`Functors`, so anything it walks is a legal shape. A `NamedTuple` is the one to prefer, because
[`export_outputs`](@ref) names the leaves that ship by key, and because `out.logits` says what it is
at every call site.

**Two rules apply to every array leaf of the output tree**, and both exist because of the short
final batch. The batch dimension must be **last**, and the framework asserts it before slicing
rather than slicing the wrong axis and returning the wrong samples. And a leaf must be an array:
a scalar you already reduced over the batch has no batch dimension to slice, so reduce it in
[`loss`](@ref) or [`metrics`](@ref) instead of returning it from `forward`.

### Where the outputs end up

| Stage | What runs | What `outputs` is by the time you see it |
| --- | --- | --- |
| training | `forward` then `loss`, in one traced program | exactly what `forward` returned |
| training metrics | `forward` then `train_metrics` | the same, once per micro-batch |
| validation | `forward` then `metrics` | sliced back to the real sample count, so padding is invisible |
| prediction | `forward` alone | sliced, and converted to host arrays |
| export | `export_preprocess` then `forward` | the leaves [`export_outputs`](@ref) names, which may be a subset |

The padding is the reason two of those rows differ. A short final eval batch is padded up to the
compiled width, the program runs at that width, and the outputs are sliced back before any hook of
yours sees them. [`metrics`](@ref) never sees a padded sample.

## The three markers, and the default

An experiment field is one of three categories, chosen with a marker or, for the default, with
nothing at all:

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

Unmarked means `Host`, and that is the default on purpose. The first real model ported to this
framework was 83% `Host` by field count, so the category you get by saying nothing is the one you
almost always want. Writing `Host{T}` explicitly is legal, and worth it only where a field's
host-ness would otherwise surprise a reader.

A `GraphConst` field bakes as a trace-time constant with ordinary Julia semantics, and because it
enters the compile cache key, changing it recompiles, which is correct: a different value is a
different program. A `Device` field is converted to device residency at setup and reaches traced
code as an input, so revising its value or scheduling it does not recompile.
[Recompilation](recompilation.md) says what the key actually contains; [Schedules](schedules.md)
owns the scheduling side.

Getting it wrong is loud in both directions. Reading an unmarked field from traced code raises with
the field named and all three fixes spelled out. Over-marking `GraphConst` costs a recompile per
distinct value, which is visible in the cache counters. Neither mistake produces a plausible-looking
run.

Keep dataset-sized state unmarked, or off the experiment entirely. The `compile_view` section below
says why.

## Why a macro

A macro rather than a helper function, because the declaration-time knowledge cannot be recovered
later. After setup a `Device` field holds a device value, a `ConcretePJRTNumber` for a scalar or a
`ConcretePJRTArray` for an array, not the marker type, so inspecting field types at runtime cannot
tell which fields were marked. The macro records that knowledge as traits while the declaration is
still in front of it.

Each `Device` and `Host` field gets its own free type parameter on the generated struct. That is
what lets one field hold a host value before setup and a device value after, and lets a `Host`
field hold its real value in `e` and a `ReactantNitro.StrippedHost` in `compile_view(e)`. `GraphConst` fields
keep their declared concrete type, and the generated keyword constructor converts to it, so
`width::GraphConst{Int}` means what it says whether or not the experiment happens to declare a
`Device` alongside it.

The recorded traits are [`device_fields`](@ref), [`host_fields`](@ref), and
[`config_metadata`](@ref). There is deliberately no `graphconst_fields`: two traits determine three
categories and the third is the complement, `setdiff(fieldnames, device_fields, host_fields)`. The
macro also generates the keyword constructor and a specialized [`compile_view`](@ref) method, and it
appends a marker table beneath the struct's docstring.

The macro is optional. A hand-written struct plus those three trait definitions works, which is why
the three are exported; `compile_view` and `config_metadata` have fallbacks that read the traits for
a type that never used the macro. The [API](api.md) page links each of them.

## compile_view and the stripped view

While tracing, Reactant and Enzyme traverse the whole experiment, because it is an argument to
the compiled program. A dataset-sized field reachable from it is therefore walked element by
element, on one thread, on every compile. The cost is O(n_train): it grows with the data, it does
not change the emitted graph, and it does not raise, so nothing about the trained model looks
wrong. Compile time simply reads like a data-loading problem, and is not one.

The guard is [`compile_view`](@ref), which every trace site receives instead of `e`. It replaces
every `Host` field with a `ReactantNitro.StrippedHost` sentinel that carries only the field's
name, as a type parameter, so there is nothing left to walk:

```julia
compile_view(e).images     # ReactantNitro.StrippedHost{:images}()
compile_view(e).width      # 128, unchanged: a GraphConst field still bakes
compile_view(e).smoothing  # unchanged: a Device is a traced input
```

The real `e` is what runs everywhere outside the trace: `build_data`, `derive`, metric
finalization, checkpointing, and every driver decision read `e.images` normally. Traced code is
the one place that sees the sentinel, and reading it there raises with the field named. The
sentinel deliberately does not quietly compute: `nothing` would pass through an `::Any` signature
and survive in a returned tuple, while the sentinel catches the paths that matter, arithmetic,
property access, and conversion.

Keep dataset-sized state unmarked (`Host`), or off the experiment entirely. The second is usually
the better fit: `build_data` hands the collection to the framework, which holds it separately from
`e`. A field on the experiment is the case that needs the marker, and since unmarked already means
`Host`, the case that still bites is a field someone marked `GraphConst` by hand.

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

**The override hazard is the reason that bites first.** [`from_preset`](@ref) validates every key
against `fieldnames(E)`, and the generated keyword constructor takes one keyword per field; neither
reaches inside a struct. A recipe that wants one different number therefore has to name the whole
struct, and naming the whole struct fills every field it did not name from the CONFIG's own
defaults rather than from the experiment's default for that field:

```julia
# experiment declares:  model::GraphConst{ModelCfg} = ModelCfg(width = 256, depth = 8)
# a preset writes:      (; model = ModelCfg(width = 512))
# and the run trains at depth = 4, the CONFIG's default. Nothing raises.
```

A call-site override has the same shape: `MyExp(; model = ModelCfg(width = 512))` replaces the
field rather than merging into it. The workaround people reach for is a `_with(cfg; kw...)` merge
helper, a second mechanism to keep in sync and a second place to forget. Flat fields need none, and
`from_preset(MyExp, :recipe; width = 512)` says exactly what it changes.

The nested config is safe only by coincidence. If the experiment's default happens to be a bare
`ModelCfg()`, the fallback lands on the same values and nothing looks wrong. That stops being true
the day the default gains an argument, and then every partial preset silently reverts every field
it did not name.

Three more reasons, in the order they matter. Residency is a per-field property, so a nested struct
takes one marker for its whole group: nothing inside a `GraphConst` struct can be `Device`, and
`Device` is what makes a value sweepable and schedulable with no recompile. Burying a loss weight
in a config struct decides, without meaning to, that changing it recompiles the gradient program.

A struct in a `GraphConst` field is a hashing hazard, and the wrapper is the whole defect. Julia's
fallback `hash` for a struct with no method of its own hashes `objectid(x)`, and `objectid` reaches
a mutable field such as a `Vector` by identity rather than descending into it. Two identical
constructions then key differently, and one mutated in place keys the same, which would serve a
program compiled for the old value. The same vector held directly as a `GraphConst{Vector{Int}}`
field is content-hashed by Base in both directions, with nothing written by hand.
`ReactantNitro.assert_graphconst_hashable` refuses the struct case at setup and recommends
flattening first.

And [`config_metadata`](@ref), which feeds the logged hyperparameter table, is per field. A
non-scalar field logs as the bare string of its type name, so a nested config contributes one
useless entry and none of its values reach the table or any run comparison. Flat fields each log by
name.

**What not to flatten.** A `Host` field holding dataset-sized state (samplers, in-memory frames,
handles) stays one field: it is not a group of knobs, and the point is that `compile_view` replaces
the whole field with a sentinel so the tracer never walks it. A config type that is some other
function's signature stays a type, but assemble it at the call site rather than storing it. And the
cost is asymmetric: `@experiment` generates one type parameter per `Device` and per `Host` field and
none per `GraphConst`, so flattening an architecture config is free in that respect while
flattening a large `Host` config is not.

## derive and named configurations

[`derive`](@ref)`(e, data)` returns a NamedTuple that the framework merges into the experiment, for
anything that genuinely depends on the dataset: class weights from label counts, a normalization
constant. It runs after `build_data` and before device conversion, so it always returns host values
and the framework places them, and resume recomputes it rather than restoring it:

```julia
function ReactantNitro.derive(e::MyExp, data)
    counts = [count(==(c), labels) for c in 0:9]         # from the training split
    (; class_weights = Float32.(sum(counts) ./ (10 .* counts)))
end
```

Prefer constants for anything structural. Deriving a shape makes the compiled program a function of
your data, so a different split silently changes the graph. And a large derived value should be
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

Every preset key is validated against `fieldnames(E)`, and that validation is the mechanism's one
correctness argument: an unknown key in a splatted NamedTuple would otherwise be a silent no-op. A
preset is field values, so the marker semantics are untouched: switching presets recompiles exactly
when the emitted graph really differs, two recipes differing only in `Host` fields share both
compiled programs, and one that changes a class count correctly does not. A preset may set a
`Device` field too; `derive` wins for anything it computes. `from_preset` returns a bare experiment
and records no name; `Nitro(E, name; ...)` records which recipe produced the run. Keep the recipes
in their own file, beside the experiment and organized by whatever they vary over.

## The Revise workflow, which is the point of the markers

The markers exist so that editing code has a predictable, cheap, and loud lifecycle.
[Tutorial](tutorial.md) walks through the story in narrative form; this is the reference version.

A `Nitro` is a fixed point. Everything that decides which compiled program runs is resolved at
construction: the run keywords and accessor values are frozen into the handle's own fields, and
every read afterwards goes to the field. So nothing you revise changes an existing handle, and a
handle can never silently recompile underneath you.

The fix is always the same: revise, then build a new `Nitro`.

```julia
# You edit `loss` in your editor, then:
train!(n)                                        # still the OLD loss, and it tells you

# The fix is always the same:
n = Nitro(e; data = n.data)                      # picks up the edit, reuses the collection
train!(n)                                        # compiles only what actually changed
```

A rebuild is cheap because the compile cache is module-level rather than per-`Nitro`: a fresh
handle on the same experiment hits the entries the previous one compiled, so it recompiles only
what the edit changed. [Recompilation](recompilation.md) covers the key mechanics. The remaining
cost of a rebuild is setup, not compilation: device conversion, a single batch to infer routing,
and `derive` on the resume path. Seconds, against the hundreds a recompile costs.

What does land on an existing handle is a host-side method on a stored object. Nothing is compiled
and nothing is snapshotted except the object, so the new code simply runs next epoch, for free:

```julia
# A `should_stop` method on your own stopper. No recompile, new behaviour.
ReactantNitro.should_stop(es::MyStopper, epoch, metrics) = ...
train!(n)
```

The one exception to freezing is a `Device` value, changed with [`set_device!`](@ref) on the live
handle. It is exactly the thing the compile key excludes by construction, so writing one provably
reuses the compiled programs:

```julia
for t in (0.5f0, 1f0, 1.5f0, 2f0)
    set_device!(n; temperature = t)
    logits = predict(n, batch)                   # zero compiles, every iteration
end
device_value(n, :temperature)                    # reads it back as a HOST value
```

`set_device!` refuses the cases that would break the no-recompile promise, and each error names
its fix: a `GraphConst` field bakes, so it is genuinely a different program; a `Host` field reaches
no trace, so writing it would change nothing compiled; a scheduled field belongs to the schedule;
a value of a different element type or size would move the key. In every case the fix is a new
`Nitro`.

The handle is never silently stale. Every entry
point prints what it fixed, names any hook you have redefined since construction, and flags a
scalar accessor that has drifted:

```julia
ReactantNitro: MyExp values fixed at construction (rebuild the `Nitro` to change them)
  seed 42   accum 2   max_epochs 20   total 5500   clip 1.0   run_dir runs/myexp
  ! `accum(e)` was redefined: it returned 2 at construction and returns 4 now.
  ! hooks were redefined after this `Nitro` was built. It will keep running the programs it was
    built with; rebuild the `Nitro` to pick up the new code.
```

The comparison is against what the accessor returned at construction, so a
keyword you passed deliberately is never mistaken for a stale accessor; only a genuine redefinition
trips it. When accessor and keyword disagree, the handle wins on purpose: `accum` and `max_epochs`
jointly determine the schedule horizon `total`, and applying a revised value in isolation would
leave every schedule resolved against a horizon that no longer exists. Rebuild instead, which costs
no compilation, or silence the report in a scripted driver with
`ReactantNitro.set_config_report!(false)`.

Two limits. An accessor that constructs an object (`early_stop`,
`checkpointer`, `logger`) is never flagged: probing it would fire the side effect the framework
promises to trigger exactly once. And a `const` or global read from inside a hook body is invisible
to the key, since the method's world only moves when the method itself is edited; edit a helper
that `forward` calls and you get a stale program. Mid-run edits do nothing at all: a running
`train!` is pinned to the world age it started in.

The rule in one line: build a new `Nitro` for a new training job; reuse an existing one for more
validation, evaluation, prediction, and `Device` sweeps.
