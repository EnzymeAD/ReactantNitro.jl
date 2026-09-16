---
name: reactantnitro-export
description: >
  Export a trained ReactantNitro model to a servable artifact: the six export hooks and
  which are optional, why the exported graph is not the graph that trained, the wire
  convention, what the framework derives and verifies versus what you declare, and what
  provenance it stamps versus what you must supply. Invoke when exporting a model,
  writing the export hooks for a new experiment, or reading a bundle that does not
  behave the way its manifest says it should.
---

# Exporting a ReactantNitro model

An export is one call. Everything else on this page is about the six declarations that call
reads, and about three facts that are not obvious and are expensive to discover by trial.

**When driving from a Kaimon session, the tool form is `nitro_export`** (`reactantnitro-kaimon`):
it takes the same hooks and adds the run-registry forms, `nitro_export(run_id = ...)` to reuse a
completed train run's `Nitro`, or `nitro_export(experiment = ..., checkpoint = ...)` to build a
weights-only handle, with `dir`/`name` required and `batch_sizes` defaulting to `[1]`. Everything
on this page about the hooks, the wire seam, and provenance still applies; the tool is the driver.

```julia
using ReactantServerExport                   # the extension that makes a backend exist

nitro = Nitro(e; checkpoint = "runs/x/best.jld2", data = (;))
export_model(nitro, ReactantServerBundle(); dir = "export_out", name = "my_model_v1")
```

## Three facts to have before writing anything

**The exported graph is not the graph that trained.** It is `export_preprocess ∘ forward`. A client
sends what a client can cheaply send, usually `UInt8` pixels; `forward` takes what the model trained
on, usually normalized `Float32`; and the conversion happens inside the compiled program. So the
artifact you ship contains an operation your training never ran. That is the intended design and not
a compromise: folding the conversion into `forward` would make every training step pay it, and doing
it outside the graph would make every client reimplement it.

**Export retraces.** It does not reuse the executable `predict` runs and it never touches the compile
cache. What the two share is the definition of `forward`, which is the whole reason the framework
splits `forward`, `loss` and `metrics` in the first place. Expect an export to cost a compile.

**Export publishes a phase.** The backend call is wrapped in the framework's `ExportCompiling`
phase, the one compile leaf that fires per call rather than per cache miss, so a phase
monitor sees the minutes-long trace rather than a silent stall. From a Kaimon session,
`nitro_status` shows `phase=ExportCompiling` while the export run is live.

**Export is a CPU trace.** It needs no accelerator, leases nothing, and asserts a single device: a
sharded program is not servable as one artifact, so build the export handle with `n_devs = 1`.

## The handle, and why export never reads a checkpoint

The framework distinguishes a **checkpoint**, which is full training state, from **weights**, which
are parameters alone. Export wants the second and `Nitro(e; checkpoint = path)` already produces it:
a weights-only restore that takes the parameters and the derived `Device` values from the record and
runs setup and nothing else.

**So no model's export code should contain the words "load" or "checkpoint".** If you are reaching
for a checkpoint parser you have gone around the surface rather than through it.

`data = (;)` is not a workaround either. `Nitro(e)` runs setup and nothing else, so an evaluation or
serving construction never needs the training data, and export is the purest case of that.

## The hooks

Two required, four optional.

```julia
export_inputs(e)  -> Vector{ExportSpec}       # required. what a client sends, IN ORDER
export_outputs(e) -> Vector{ExportSpec}       # required. which `forward` leaves ship
export_preprocess(e, wire...) -> NamedTuple   # optional. wire -> the batch `forward` declares. TRACED
export_postprocess(e) -> String               # optional. the model.jl source that ships
export_client_outputs(e) -> Vector{ExportSpec} # optional. what a client receives after model.jl
export_client_inputs(e)  -> Vector{ExportSpec}  # optional. the client-facing INPUT spec; carries `axis_letters`
```

A minimal image classifier is four lines:

```julia
ReactantNitro.export_inputs(e::MyExp) = [ExportSpec("img", UInt8, [e.sz, e.sz, 1, 1])]
ReactantNitro.export_outputs(::MyExp) = [ExportSpec("logits")]

# TWO methods, and read the next section before writing one method and moving on.
_wire_to_f32(x::AbstractArray) = Float32.(x) ./ 255.0f0
_wire_to_f32(x::Reactant.TracedRArray) =
    Reactant.Ops.convert(Reactant.TracedRArray{Float32, ndims(x)}, x) ./ 255.0f0
ReactantNitro.export_preprocess(::MyExp, img) = (; img = _wire_to_f32(img))
```

### `export_inputs` order is load-bearing

It fixes the positional order the program is traced with, the order of the tensor names in the
artifact, and therefore the order a client must supply. **Reordering this vector changes the wire
contract of the next bundle, silently.** Treat it the way you would treat field order in any other
serialized format, not as a presentation choice.

### `export_outputs` names a subset, and that is the point

`forward` returns what training needs, which is routinely more than a client wants: an ODE model
returning kinetic-energy rows alongside its predictions is the ordinary case. Name the leaves that
ship and the rest stay in training. A second `export_forward` program would be a second thing to keep
in agreement with the first, which is a defect this framework has already paid for once elsewhere.

Names must be keys of the `NamedTuple` `forward` returns. **A plain `Tuple` return is refused**,
deliberately: its leaves have no names, so bundle tensor names would be positional and a reordering
inside `forward` would rename a client's outputs with nothing to say so.

### `export_preprocess` is traced, AND ALSO CALLED EAGERLY

The traced half is the half people expect: no `Host` field reads, no data-dependent control flow,
nothing that is not a Reactant operation.

**The eager half is the one that costs an afternoon.** The hook is called on plain host arrays before
anything is traced, twice, once by the framework's verification probe and once by the backend
discovering output shapes. So it runs in two modes, and **the single most common thing it will ever
do works in only one of them**:

```julia
Float32.(img)     # MethodError: no method matching Float32(::Reactant.TracedRNumber{UInt8})
```

An integer-to-float conversion has no traced broadcast method. Every model doing the ordinary `UInt8`
to normalized `Float32` conversion hits this on its first export, so write two methods from the
start:

```julia
_wire_to_f32(x::AbstractArray) = Float32.(x) ./ 255.0f0
_wire_to_f32(x::Reactant.TracedRArray) =
    Reactant.Ops.convert(Reactant.TracedRArray{Float32, ndims(x)}, x) ./ 255.0f0
```

The eager method is not dead code kept for symmetry. It is what runs during shape discovery, so if it
disagrees with the traced one, the bundle's declared shapes describe a different computation than the
one it ships.

With no method at all the framework maps each wire input to a batch field of the same name, which is
right whenever `forward` already declares the wire tensors.

### Client-facing specs require `export_postprocess`

The postprocess is the Julia that ships beside the graph for work that does not belong in it: a
softmax, a decode, an assembly of raw logits into what a client actually wants. It runs on the
server, not in the compiled program.

**Declaring a client-facing spec, `export_client_inputs` or `export_client_outputs`, without a
postprocess is an error, and the framework raises it rather than letting you find out at serve
time.** Client-facing specs only mean something when something transforms the executable tensors,
and an artifact carrying them without one is rejected when it is loaded. Implement the postprocess,
or drop the client-facing hook and let the executable specs be the client's contract.

## What you declare, and what the framework refuses to let you declare

The split is not arbitrary. **The framework derives everything it can and asks only for what it
cannot.**

| | dtype and shape | batch axis | `axis_letters`, `-1` |
| --- | --- | --- | --- |
| `export_inputs` | **required**, example arrays are built from them | must be last | rejected |
| `export_outputs` | optional, derived from the trace, an assertion if given | must be last | rejected |
| `export_client_outputs` | **required** | free | allowed |
| `export_client_inputs` | **required** | free | allowed |

The executable side is traced, so its shapes are facts rather than declarations, and the batch-last
rule is one the framework already enforces on every array leaf of a batch and of `forward`'s return.
The client side is the output of a `model.jl` the framework never runs, so there it declares nothing
and asks for everything, including a `-1` for a variable axis and letters that give the manifest
meaningful axis names.

### Batch-last applies to the client side too, even though nothing enforces it there

**"Free" in that table means unenforceable, not encouraged.** The verification stops at the executable
boundary because a postprocess is Julia the framework never executes, so a client spec whose batch
axis is not last is accepted. It should still not exist.

Existing exports that do it stay, for compatibility. **There should be no new instances.** A
divergent batch axis is not a local quirk; it is one more per-model convention that every consumer
of the bundle has to remember, and that cost lands on someone other than the author who wrote it.

This is cheap to police, because a non-last batch axis is always an explicit keyword. The default is
last, on every `ExportSpec` constructor, so **any `batch_axis` appearing anywhere in an export hook is
either redundant or an exception**, and both are worth a question in review:

```bash
grep -rn 'batch_axis' <model packages>/src/
```

On the executable hooks a non-last value raises, so a hit there is only ever redundant and can be
deleted. On the client hooks a hit is the thing to look at. If a postprocess wants to emit a
batch-middle tensor, change the postprocess.

**Batch axes are derived, not declared, and then verified.** A tracer takes every tensor's batch axis
to be its last axis. The framework asserts the same rule everywhere else. Those two agree only
because export checks them against each other on a real forward pass before anything is traced, and
raises when they disagree. Do not hand-write batch axes anywhere; there is no keyword for it.

## What the framework absorbs so you do not have to

**The boundary copy.** A head whose output arrives through a reshape or a view serializes as its
*producer's* shape unless it is copied at the export boundary. The result is an artifact whose
declared output shape and actual output shape disagree, with no error anywhere. The framework copies
every selected output leaf, so this trap is gone rather than being something to remember.

**The calling convention.** `forward` is keyword-routed by batch field name and a tracer wants one
positional input object. The adapter between them is the framework's, resolved from your
`export_inputs` order.

## Provenance: what is stamped and what is yours

`export_provenance(nitro)` returns what the framework knows: the flat config, the **preset name**
recorded on the handle, the package version, the seed and the run directory. The preset travelling
with the artifact is what finally lets a served model answer "which recipe produced you", instead of
that fact living in a launch script nobody kept.

**The preset only reaches the bundle if the handle was built the recording way.** `Nitro(E, :name)`
records the name; `Nitro(from_preset(E, :name))` does not, because the second form is just an
experiment value and the framework has nothing to read the name off. So an export built the second way
produces a manifest with no `preset` key at all, silently, which defeats the one thing provenance is
most useful for. Build export handles as `Nitro(E, :name; checkpoint = path, data = (;))`.

**It does not guess at repository state.** A commit, a tree hash and a working-tree patch are site
policy, and a framework that shelled out to `git` would be asserting that the process's working
directory is the model's repository. Pass them yourself:

```julia
export_model(nitro, ReactantServerBundle(); dir, name,
             provenance = Dict("git_commit" => read(`git rev-parse HEAD`, String) |> strip))
```

Caller keys win on collision, and the backend stamps its own facts underneath.

**Two things provenance cannot currently carry, stated rather than papered over:** the checkpoint's
epoch and its metric. A weights-only restore takes the parameters from the record and then discards
it, and deliberately zeroes the epoch counter rather than continuing it. Retaining them would be a
change to the handle, not to the export path.

## Batch sizes, and one thing that is out of scope

`batch_sizes` defaults to `[1]` and **each entry is a separately compiled program**, traced and
stored independently. Widening it costs compile time and artifact size in proportion, so ask for the
sizes you will actually serve.

**Multiple compiled input shapes are not supported** by this surface. The bundle format has a
mechanism for it; `export_model` does not drive it. A model that genuinely needs several input shapes
should call the backend directly, and should say why in a comment, rather than being worked around
here.

## The bundle contract is not yours to change

The wire convention, the raw-logits executable, the postprocess boundary and the naming conventions
belong to the artifact format, and the backend package is the authority on all of them. The framework
translates into that format and never reimplements it. **If an export seems to need the artifact to
look different, that is a finding to raise, not a change to make in a model's hooks.**
