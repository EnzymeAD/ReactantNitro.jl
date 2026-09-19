# Export.jl
#
# Exporting a trained model: the model's hooks, the framework's resolution of them, and the one
# backend verb a package extension answers.
#
# Export is an extension point rather than a specification, on the same rule logging follows: THE
# FRAMEWORK OWNS WHAT IT KNOWS AND NEVER OWNS THE ARTIFACT FORMAT. It knows the program, the
# shapes, and the provenance. It does not know what a bundle is, and a surface here that starts
# inventing bundle structure has gone wrong.
#
# THE EXPORTED PROGRAM IS NOT THE PROGRAM THAT TRAINED, and that is the sharp edge of the whole
# surface. The wire carries whatever a client can cheaply send, usually `UInt8` pixels, and `forward`
# takes what the model trained on, usually normalized `Float32`. The conversion happens INSIDE the
# traced graph, through `export_preprocess`, so what ships is `export_preprocess ∘ forward`. Nothing
# about that is hidden here: it is a named hook precisely so it cannot be a line buried in a driver.
#
# Export also RETRACES. It does not reuse the executable `predict` runs, and it never touches the
# compile cache. What is shared between them is the DEFINITION of `forward`, which is the whole
# reason the `forward` / `loss` / `metrics` split exists.

# ── The spec carrier ────────────────────────────────────────────────────────────────

"""
    ExportSpec(name; from = nothing)
    ExportSpec(name, dtype, shape; batch_axis = length(shape), axis_letters = nothing, from = nothing)

One tensor's declaration, in the framework's own vocabulary rather than any backend's. A backend
translates it; nothing outside a backend extension should know what it translates to.

`shape` is the **Julia** shape and `batch_axis` is a **1-based Julia axis**, both matching every
other shape statement in this framework. A backend that wants row-major network axes converts, which
is the sort of thing having a framework type at all is for.

**Which fields are required depends on which hook returns it**, and the split is not arbitrary: the
framework derives everything it can and asks only for what it cannot.

| hook | `dtype`/`shape` | `batch_axis` | `axis_letters`, `-1` |
| --- | --- | --- | --- |
| [`export_inputs`](@ref) | **required**, the framework builds example arrays from them | must be last | rejected |
| [`export_outputs`](@ref) | optional, derived from the trace and verified if given | must be last | rejected |
| [`export_client_outputs`](@ref) | **required**, a postprocess is opaque Julia | free | allowed |

The two rejections are not tidiness. The executable side of a bundle is traced, so its shapes are
facts rather than declarations, and the batch-last rule is one the framework already enforces on
every array leaf of a batch and of `forward`'s return. The client side is the output of a
`model.jl` the framework never runs, so there it declares nothing and asks for everything.

`axis_letters` gives the manifest meaningful axis names, one `Char` per **non-batch** axis in `shape`
order, e.g. `['w', 'h']` for an image. A `-1` in `shape` marks a variable non-batch axis, which is
how a postprocess with a data-dependent output width (a detector's detection count) is declared.

`from` names the SOURCE of an output, separately from what the bundle calls it. It is meaningful only
on [`export_outputs`](@ref) and is refused elsewhere. See that hook for what it is for.
"""
struct ExportSpec
    name::String
    dtype::Union{DataType, Nothing}
    shape::Vector{Int}
    batch_axis::Union{Int, Nothing}
    axis_letters::Union{Nothing, Vector{Char}}
    from::Union{Nothing, Symbol}
end

ExportSpec(name; from = nothing) =
    ExportSpec(String(name), nothing, Int[], nothing, nothing, from === nothing ? nothing : Symbol(from))

function ExportSpec(
        name, dtype::DataType, shape;
        batch_axis::Union{Integer, Nothing} = length(shape),
        axis_letters = nothing, from = nothing
    )
    sh = Int[shape...]
    ba = batch_axis === nothing ? nothing : Int(batch_axis)
    if ba !== nothing && !(1 <= ba <= length(sh))
        error(
            """
            ReactantNitro: `ExportSpec("$name")` has `batch_axis = $ba`, which is not an axis of its
            $(length(sh))-dimensional shape $sh. `batch_axis` is a 1-based JULIA axis, like every
            other axis statement in this framework."""
        )
    end
    letters = axis_letters === nothing ? nothing : Char[axis_letters...]
    if letters !== nothing
        nax = ba === nothing ? length(sh) : length(sh) - 1
        length(letters) == nax || error(
            """
            ReactantNitro: `ExportSpec("$name")` has $nax non-batch axes but $(length(letters))
            entries in `axis_letters`. Give one letter per NON-batch axis, in `shape` order; the
            batch axis is named by the backend and is not yours to letter."""
        )
        allunique(letters) || error(
            "ReactantNitro: `ExportSpec(\"$name\")` has duplicate `axis_letters` $(letters)."
        )
    end
    return ExportSpec(String(name), dtype, sh, ba, letters, from === nothing ? nothing : Symbol(from))
end

spec_names(specs::AbstractVector{ExportSpec}) = Tuple(Symbol(s.name) for s in specs)

"""
    ReactantNitro.spec_sources(specs) -> Tuple{Vararg{Symbol}}

Where each output spec's data comes FROM, which is its `from` when it declares one and its own name
otherwise. Keeping the two separable is what lets a bundle's tensor name differ from the leaf that
produced it, and [`export_outputs`](@ref) is where that is explained.
"""
spec_sources(specs::AbstractVector{ExportSpec}) =
    Tuple(s.from === nothing ? Symbol(s.name) : s.from for s in specs)

# ── The model's hooks ───────────────────────────────────────────────────────────────

"""
    export_inputs(e) -> Vector{ExportSpec}

**Required.** The wire contract: what a client sends, in the order it sends it.

The order is load-bearing and is not a presentation choice. It fixes the positional order the
program is traced with, the order of the tensor names in the bundle, and therefore the order a
client must supply. Reordering this vector changes the wire contract of the next bundle, silently,
so treat it the way you would treat a struct's field order in a serialized format.

Each spec needs a `dtype` and a `shape`, because the framework builds the example arrays it traces
from them. The `batch_axis` must be the last axis: batch-last is the framework's rule everywhere,
and export does not get an exemption from it. The size at the batch axis is a placeholder,
overwritten by each entry of `export_model`'s `batch_sizes`.

```julia
export_inputs(e::CropClassifier) = [ExportSpec("img", UInt8, [e.sz, e.sz, 1, 1])]
```
"""
function export_inputs end

"""
    export_outputs(e) -> Vector{ExportSpec}

**Required.** Which leaves of `forward`'s output tree ship, and what they are called.

`forward` returns what training needs, which is routinely more than a client wants: an ODE model's
return carrying kinetic-energy rows is the case this hook exists for. Naming a subset is honest and
needs no second program, and a separate `export_forward` would be a second thing to keep in agreement
with the first.

Each name must be a key of the `NamedTuple` `forward` returns, or, when `forward` returns a bare
array, this must be the single spec that names it. `dtype` and `shape` are derived from the trace;
supplying them turns them into assertions, which is worth doing for an output whose shape you want a
bundle to fail on rather than drift on.

```julia
export_outputs(e::CropClassifier) = [ExportSpec("region_logits")]
```

## `from`, and why it exists

**A bundle's tensor name and the leaf that produced it are two different things, and forcing them to
be one made an export concern rewrite the trained program.** Two cases need separating, both found by
porting a real model rather than by design review:

  * **A rename.** The wire contract calls a tensor `state_out` and `forward` calls that leaf
    `g_pred`. Without `from` the only way to ship it is for `forward` to return a second, aliased
    leaf, which recompiles the gradient program to serve a serving detail.
  * **An echo.** A serve-time postprocess receives the program's OUTPUTS and never its inputs, so a
    postprocess that masks by `valid` or scales by `scale_px` needs those wire tensors back as
    outputs. Without `from` the only way was, again, to make `forward` return them.

```julia
export_outputs(e::Refiner) = [
    ExportSpec("state_out"; from = :g_pred),   # rename: a leaf `forward` returns
    ExportSpec("path_len"),                    # the ordinary case, name IS the source
    ExportSpec("valid"; from = :valid),        # echo: a wire input, returned to the client
]
```

**An echo must say so.** A spec whose name happens to match a wire input is refused unless it writes
`from` explicitly, because turning a client's own tensor into a program output on the strength of a
name collision would be a guess. The value echoed is the WIRE value, which is the tensor the client
sent, not the preprocessed one.
"""
function export_outputs end

"""
    export_preprocess(e, wire...) -> NamedTuple

**Optional, and TRACED.** The seam between the wire and the batch, and the reason the exported graph
is not the graph that trained.

It receives one positional argument per [`export_inputs`](@ref) spec, in that order, and returns the
`NamedTuple` batch `forward` declares its keywords against. Whatever it does happens inside the
compiled program, so a client sends bytes and the executable converts them.

With no method, the framework maps each wire input to a batch field of the same name, which is right
whenever `forward` already declares the wire tensors directly.

**It obeys every rule a traced hook obeys**: no `Host` field reads, since the compile view strips
them, no data-dependent control flow, nothing that is not a Reactant operation.

**AND IT IS ALSO CALLED EAGERLY, ON HOST ARRAYS, WHICH IS THE PART THAT SURPRISES PEOPLE.** Twice,
in fact: once by `export_model`'s verification probe and once by the backend's own shape discovery,
both before anything is traced. So it has to work in both modes, and the single most common thing it
will ever do does not:

```julia
Float32.(img)     # MethodError: no method matching Float32(::Reactant.TracedRNumber{UInt8})
```

An integer-to-float conversion has no traced broadcast method, so the wire conversion needs two
methods, one per mode. This is not a nicety; every model doing the ordinary `UInt8` to normalized
`Float32` conversion hits it on the first export.

```julia
_wire_to_f32(x::AbstractArray) = Float32.(x) ./ 255.0f0
_wire_to_f32(x::Reactant.TracedRArray) =
    Reactant.Ops.convert(Reactant.TracedRArray{Float32, ndims(x)}, x) ./ 255.0f0

export_preprocess(e::CropClassifier, img) = (; img = _wire_to_f32(img))
```
"""
function export_preprocess end

"""
    export_postprocess(e) -> String

**Optional.** The source of the `model.jl` that ships in the bundle, as text.

The framework never runs it and has no opinion about its contents beyond writing it where the
backend says it goes. It is the place for work that does not belong in a compiled graph: a softmax,
a decode, an assembly of raw logits into whatever a client actually wants.

Returning a source string commits you to also implementing [`export_client_outputs`](@ref) or
[`export_client_inputs`](@ref) when the postprocess changes what a client receives, and returning
`nothing` (the default) commits you to implementing neither.
"""
export_postprocess(e) = nothing

"""
    export_client_outputs(e) -> Vector{ExportSpec}

**Optional.** What a client receives after [`export_postprocess`](@ref) has run, when that differs
from what the executable emits.

Only meaningful alongside a postprocess, and the framework refuses the combination that does not
make sense: declaring client outputs with no `model.jl` writes a bundle that fails when the server
loads it, so it fails here instead.

These specs are the one place `ExportSpec`'s full expressiveness applies. The framework cannot derive
them, because a postprocess is opaque Julia it never executes, so `dtype` and `shape` are required,
`batch_axis` is unconstrained, and both `axis_letters` and `-1` variable axes are available.

**`batch_axis` being unconstrained here means unenforceable, not encouraged.** Batch-last is the
convention everywhere else and the verification simply cannot reach past the executable boundary.
One model predates the rule and keeps its batch-middle client tensor; there should be no new ones.
Since `batch_axis` defaults to last, any explicit value is either redundant or an exception, which
makes both greppable and neither silent.

```julia
export_client_outputs(e::CropClassifier) = [
    ExportSpec("region_prob", Float32, [e.num_classes, 1]; batch_axis = 2),
    ExportSpec("region_logits", Float32, [e.num_classes, 1]; batch_axis = 2),
]
```
"""
export_client_outputs(e) = nothing

"""
    export_client_inputs(e) -> Vector{ExportSpec}

**Optional.** The client-facing spec of the INPUTS, for the one thing that is not derivable about
them.

Their dtypes and shapes are not that thing. The wire preprocess lives inside the traced graph, so
the executable inputs already ARE the wire inputs, and a manifest that repeated them would be
repeating itself; with no method here the server falls back to the executable specs, which is the
same answer.

**What is not derivable is `axis_letters`.** The tracer derives the executable input specs itself and
has no way to carry letters into them, so a model that wants `"whcn"` in its manifest rather than
auto-allocated letters has to say so on the client side. That is the whole reason this hook exists,
and it is worth having because an axis named `w` documents a wire contract in a way `a` does not.

Like [`export_client_outputs`](@ref) it requires [`export_postprocess`](@ref), and for the same
reason: client-facing specs are rejected by the server when no `model.jl` is present.

```julia
export_client_inputs(e::Refiner) = [
    ExportSpec("img", UInt8, [e.w, e.h, 1, 1]; axis_letters = ['w', 'h', 'c']),
    ExportSpec("g0", Float32, [e.dims, e.k, 1]; axis_letters = ['g', 'k']),
]
```
"""
export_client_inputs(e) = nothing

"""
    export_provenance_extra(e; checkpoint = nothing) -> Dict{String,Any}

**The MODEL's half of a bundle's provenance**, merged on top of the framework's and the site's. The
default is empty, so this is opt-in and an experiment that has nothing to add implements nothing.

`export_provenance` stamps what the framework knows and a backend stamps its own facts underneath.
Neither can know what a *consumer* of this particular model needs in order to use it: which
labeling variant produced the head, the class names in the labeler's order, the crop geometry a client
has to reproduce, the contract strings that make a served tensor interpretable. None of that is
inferable from the tensor shapes, and a class count alone routinely fails to identify a model, since
two different labeling schemes can be the same width.

**Why this is a hook rather than something the caller merges in.** It was a hand-merged dictionary
first, and that shape has one failure mode which is worse than an error: omit the argument and the
export SUCCEEDS, the arity check passes, the manifest carries the framework's stamps, and the bundle
is untraceable while looking complete. Documentation had to warn about remembering the argument,
and more than one model independently wrote the same function under a name of its own, so nothing
generic could call any of them. A hook the framework merges removes the argument from both entry
points at once, so the tool path and the hand-written path get the same bundle.

`checkpoint` is the file the handle's weights came from, or `nothing` for freshly initialized weights,
and [`export_model`](@ref) passes `Nitro.checkpoint_source` rather than asking the caller to name it
again. **The `nothing` case is a discriminator and not merely an absence:** a model whose provenance
asserts anything about what it was trained against should report those fields as unverifiable rather
than as verified when there is no checkpoint behind them.

```julia
function ReactantNitro.export_provenance_extra(e::MyExp; checkpoint = nothing)
    prov = Dict{String, Any}(
        "model" => "MyModel",
        "class_names" => collect(String, class_names(e.variant)),
        "crop_sz" => [e.sz, e.sz],
    )
    checkpoint === nothing || (prov["trained_from"] = abspath(String(checkpoint)))
    return prov
end
```

Keys collide with the framework's and the site's at the model's own risk: [`export_model`](@ref)
states the precedence and this half wins over both. The explicit `provenance` argument still wins
over this, so a human can always override a hook.
"""
export_provenance_extra(e; checkpoint = nothing) = Dict{String, Any}()

# ── The backend contract ────────────────────────────────────────────────────────────

"""
    ReactantNitro.ExportBackend

The supertype of an export target. One verb, [`write_export`](@ref), dispatches on the backend
value, exactly as the logging contract dispatches on the logger object.

The framework ships no method for any backend. `ReactantServerBundle` is a package extension on a
weak dependency, which is the same care the logging contract takes with backends and the optimizer
takes with schedules: this package depends on no logger, no schedule library, no batching library,
no plotting package, and export is not the exception that breaks the pattern.
"""
abstract type ExportBackend end

"""
    ReactantNitro.ReactantServerBundle()

The StableHLO bundle target consumed by ReactantServer. The type is declared here so it can be named
and exported; its [`write_export`](@ref) method lives in the `ReactantServerExport` extension, so
`using ReactantServerExport` is what makes an export actually happen.
"""
struct ReactantServerBundle <: ExportBackend end

"""
    write_export(backend, model, ps, st, example_inputs; kwargs...) -> String

The one verb a backend implements. Everything above it is the framework's and is already resolved by
the time this is called; everything below it is the artifact format's and is none of the framework's
business.

`model` is a callable the framework built, with the `model(inputs, ps, st) -> (outputs, st)` shape a
Lux-style tracer expects. It already carries `compile_view(e)`, the resolved routing, the traced
preprocess, and the output selection, so a backend traces it exactly as it would trace any model and
needs to know nothing about experiments.

Keywords, all supplied by [`export_model`](@ref):

  * `dir`, `name`: where the artifact goes and what it is called.
  * `input_names`, `output_names`: `Vector{String}`, in the order the program takes and returns them.
  * `output_select`: maps the raw `forward` return to the ordered tuple of arrays that ship.
  * `client_inputs`, `client_outputs`: `nothing`, or the `Vector{ExportSpec}` the client side uses.
  * `postprocess`: `nothing`, or the `model.jl` source to write into the artifact.
  * `batch_sizes`: each one is a separately compiled program.
  * `provenance`: `Dict{String,Any}`, already merged (framework first, caller's on top).

A backend is expected to return the path it wrote.
"""
function write_export end

# The fallback exists so a missing extension says which package to load, rather than raising a
# MethodError whose argument list is thirty keywords long and names nothing useful.
function write_export(backend::ExportBackend, args...; kwargs...)
    hint = backend isa ReactantServerBundle ?
        "`ReactantServerBundle` is provided by a package extension, so add `using ReactantServerExport` and the method appears." :
        "A backend supplies its own `write_export` method; the ReactantServerExport extension is the shape of one."
    return error(
        """
        ReactantNitro: no `write_export` method for backend `$(typeof(backend))`.
        $hint"""
    )
end

"""
    site_provenance(backend, root) -> Dict{String,Any}

**The SITE's half of a bundle's provenance: repository state, collected at `root`.** The second verb a
backend answers, and the reason it is a verb at all is the one [`export_provenance`](@ref) gives:
a git commit, a tree hash and a working-tree patch are site policy rather than framework knowledge,
and a framework that shelled out to `git` would be asserting that the process's working directory
is the model's repository.

**The caller names the root, so nothing is guessed.** That is what makes this compatible with
`export_provenance`'s refusal to look: the framework still does not decide what repository a model
lives in, it forwards a root it was given to a backend that knows how to read one.

It dispatches on the backend rather than being hardcoded because the artifact format owns what it can
record. `ReactantServerBundle` answers with `ReactantServerExport.collect_provenance`, whose `git_diff`
the writer materializes as `working_tree.patch` in the bundle. On a dirty tree that patch is the only
thing tying the artifact to the code that produced it, and it is a multi-line unified diff, which is
exactly the shape that cannot ride a flat `name=value` list. That is why this is reached through a
ROOT rather than through the provenance dictionary.

There is no default method, deliberately. A backend that cannot collect site provenance must say so
rather than return an empty dictionary, because a silently empty result is the failure this whole
surface is designed against: a bundle that looks complete and is untraceable.
"""
function site_provenance end

# Same shape as `write_export`'s fallback and for the same reason: the error names what to do instead
# of reporting a thirty-keyword MethodError. Erroring rather than returning `Dict()` is the point.
function site_provenance(backend::ExportBackend, root)
    hint = backend isa ReactantServerBundle ?
        "`ReactantServerBundle` collects it through `ReactantServerExport.collect_provenance`, so add `using ReactantServerExport` and the method appears." :
        "A backend supplies its own `site_provenance` method; the ReactantServerExport extension is the shape of one."
    return error(
        """
        ReactantNitro: `export_model` was given `provenance_root = $(repr(root))` and backend
        `$(typeof(backend))` has no `site_provenance` method, so the repository state it asked for
        would be silently missing from the bundle.
        $hint
        Leave `provenance_root` unset and pass what you want stamped through `provenance` if this
        backend genuinely records no repository state."""
    )
end

# ── The traced adapter ──────────────────────────────────────────────────────────────

# The framework's `forward` is keyword-routed by batch field name; a tracer wants one
# positional input object. This is the whole of the mismatch, and it is resolved here rather than
# asked of either side.
#
# `K` is a type parameter for the same reason `Router`'s key set is one: the NamedTuple construction
# then resolves at trace time and the traced graph sees only the selected fields.
struct ExportForward{K, ECHO, E, M, R}
    ev::E
    inner::M
    router::R
    has_preprocess::Bool
end

function ExportForward(input_names::Tuple, echoes::Tuple, ev, inner, router, has_preprocess::Bool)
    return ExportForward{input_names, echoes, typeof(ev), typeof(inner), typeof(router)}(
        ev, inner, router, has_preprocess
    )
end

function (f::ExportForward{K, ECHO})(a, ps, st) where {K, ECHO}
    wire = a isa Tuple ? a : (a,)
    batch = f.has_preprocess ? export_preprocess(f.ev, wire...) : NamedTuple{K}(wire)
    batch isa NamedTuple || error(
        """
        ReactantNitro: `export_preprocess` must return a NamedTuple, the batch `forward` declares its
        keywords against, and returned a `$(typeof(batch))`."""
    )
    outputs, st_new = call_hook(forward, :forward, f.router, batch, f.ev, f.inner, ps, st)
    # THE ECHO: a wire tensor the postprocess needs back. It is merged here, at the framework's
    # own boundary, rather than being asked of `forward`, because a serve-time postprocess receives
    # the program's OUTPUTS only and never its inputs. Doing it here is what keeps an export concern
    # out of the trained program: the alternative is a model returning a leaf that training does not
    # want, which recompiles the gradient program to serve a serving detail.
    #
    # The WIRE value is echoed, not the preprocessed one: what a client gets back is the tensor it
    # sent. Merged after `forward`, so a name it already returned is deliberately NOT overwritten;
    # `check_export_sources` refuses that collision before it can happen.
    isempty(ECHO) && return outputs, st_new
    wired = NamedTuple{K}(wire)
    return merge(outputs, NamedTuple{ECHO}(map(k -> getfield(wired, k), ECHO))), st_new
end

# Select the named leaves of `forward`'s return, in declared order.
#
# `copy` is not decoration. A head whose output arrives through a reshape (a view chain) serializes
# as its raw PRODUCER shape unless it is copied at the export boundary, which produces a bundle whose
# declared output shape and actual output shape disagree with no error anywhere. That trap used to
# live as a comment in a hand-written driver, which is exactly the kind of thing a framework should
# absorb once instead of asking every model author to remember.
struct ExportSelect{K} end

ExportSelect(names::Tuple) = ExportSelect{names}()

function (::ExportSelect{K})(outputs) where {K}
    if outputs isa NamedTuple
        return map(K) do k
            haskey(outputs, k) || error(
                """
                ReactantNitro: `export_outputs` sources `$k`, which is neither a leaf of what
                `forward` returned nor a wire input. What is available is $(keys(outputs)).
                A spec's source is its `from` when it declares one and its own name
                otherwise."""
            )
            copy(getfield(outputs, k))
        end
    elseif outputs isa AbstractArray
        length(K) == 1 || error(
            """
            ReactantNitro: `forward` returned a bare array, so `export_outputs` may name exactly one
            output, and it names $(length(K)): $(K).
            Return a NamedTuple from `forward` if several leaves should ship, so they can be
            named."""
        )
        return (copy(outputs),)
    end
    return error(
        """
        ReactantNitro: `forward` returned a `$(typeof(outputs))`, which `export_outputs` cannot name
        into. Return a NamedTuple, whose keys are the names, or a single array.
        A plain Tuple is refused deliberately: its leaves have no names, so the bundle's tensor names
        would be positional and a reordering in `forward` would rename a client's outputs silently."""
    )
end

# ── The export view: what a FROZEN graph is allowed to see ──────────────────────────

"""
    export_view(e) -> e_export

The view of an experiment a **frozen** trace sees: [`compile_view`](@ref)'s stripping of every
`Host` field, and then every device-resident value that survives it read back to the host.

**This is deliberately not the view a training step sees.** A [`Device`](@ref) field is a traced
INPUT, which is exactly right while the framework owns both ends of the call: it supplies the
value at every invocation and changing the value costs no recompile. An exported bundle has no such
owner. It is `executable(inputs..., weights...)` and nothing more, so a value that is neither a
declared input nor a serialized weight has nobody to supply it.

**Reactant lifts every device-resident value reachable from a traced closure into an argument**,
whether the program reads it or not. A `Device` field therefore becomes an argument of the exported
module that the bundle declares no name for and no client can pass. Measured on a bundle of N
weights and one input whose experiment carried two `Device` arrays and whose layer state carried an
RNG seed:

    Execution supplied N+1 arguments but compiled program expected N+4

Both readable halves of that bundle were correct. The manifest declared its one input, the
safetensors file held its N weights, and the three surplus arguments existed only inside the
compiled graph, where nothing was looking.

**So the export view FREEZES rather than filters, and that is deliberate.** A `Device` field the
exported `forward` never reads (a loss weight, a class-weight vector, a soft-target matrix) becomes
a host value nothing reads, and no constant is emitted for it at all. A `Device` field `forward`
DOES read bakes into the graph as the value it held at export. Both are the right answer for an
artifact that is a fixed function, and neither one adds an argument.

**Filtering could not have worked**, which is worth stating because dropping the loss-only fields is
the obvious design. The framework cannot know which `Device` fields `forward` reads: a hook reaches
them through `ev` inside its own body and nothing declares it. Dropping the fields `forward` does not
read would leave every field it does read still lifted, so the defect would survive on precisely the
models whose field carries something that matters.

**The cost, stated rather than hidden.** A frozen `Device` array is a constant in each compiled
module, so a large one is paid for once per batch size in artifact size. A `Device` field is meant
for knobs and per-class statistics, so it is small in practice; an experiment carrying something
big enough to matter should override this method and say why.

See also [`compile_view`](@ref), which is what a TRAINING trace sees and which leaves `Device` fields
device-resident on purpose.
"""
export_view(e) = host_tree(compile_view(e))

"""
    ReactantNitro.host_rngs(x) -> x

Every `AbstractRNG` in a tree replaced by a host RNG, structure preserved. Applied to the parameters
and the layer state on the way to a backend, and nowhere else.

**This closes [`host_tree`](@ref)'s one deliberate blind spot, at the one boundary where it is
fatal.** `Reactant.ReactantRNG`'s type parameter forbids host contents (see [`HostRNG`](@ref)), so
`host_tree` passes one through unchanged, reasoning that an RNG's seed is not model data and that an
eval-mode `Dropout` never draws from it. Both of those are true and the conclusion still does not
follow: at export the state is TRACED, and tracing a device-resident seed lifts it into an MLIR
argument exactly as a `Device` field is lifted (see [`export_view`](@ref)). `st.<layer>.rng.seed`
is two `UInt64`s, and it is the difference between a servable bundle and one that fails every
inference.

**Replaced, not removed.** A testmode `Lux.Dropout` still CALLS `dropout(rng, x, p, Val(false), ...)`
and dispatches on the RNG's type, so substituting `nothing` raises a `MethodError` instead
(measured). The RNG is passed and never drawn from, so any concrete host RNG is numerically inert
here; all that matters is that Reactant is left nothing device-resident to hoist.

**What this assumes, plainly:** that the exported program does not draw from an RNG. Export is eval
mode by construction, and the substituted RNG is NOT the one training used, so a layer that samples
at inference would bake one fixed draw into the artifact. Such a layer is outside what this surface
contracts for, and nothing here detects it.
"""
host_rngs(::Random.AbstractRNG) = Random.Xoshiro(0)

function host_rngs(x::Union{Tuple, NamedTuple})
    y = map(host_rngs, x)
    return all(i -> y[i] === x[i], 1:length(x)) ? x : y
end

# Same identity-preserving struct walk as `to_host` and `host_tree`, for the same reason: an RNG in
# layer state lives inside a struct (`st.<layer>.rng`), which is where both of those walkers had to
# learn to look, and a subtree with no RNG in it comes back `===` what it was.
function host_rngs(x)
    (x isa AbstractString || x isa Symbol || x isa Number || x isa AbstractArray) && return x
    T = typeof(x)
    isstructtype(T) || return x
    fs = fieldnames(T)
    isempty(fs) && return x
    vals = ntuple(i -> host_rngs(getfield(x, fs[i])), length(fs))
    all(i -> vals[i] === getfield(x, fs[i]), 1:length(fs)) && return x
    return T.name.wrapper(vals...)
end

"""
    ReactantNitro.check_export_residency(program, ps, st) -> nothing

**The export view's invariant, asserted: nothing reachable from the traced closure is
device-resident.** Every such value becomes an argument of the compiled module, the bundle has no
name for it, and the artifact fails every inference with an argument count no reader of the bundle
can account for.

This runs BEFORE the trace, which is the whole of its value. The same fault costs a minutes-long
compile, a bundle write, a deploy and a serve-time round trip to reach as
`Execution supplied N arguments but compiled program expected N+k`, from a process that knows
neither which value it is nor which model produced it. Here it is a path and a type.

It walks `program` rather than converting it, and the asymmetry is deliberate: `ExportForward`
carries its input names and echoes as TYPE parameters, so it cannot be rebuilt field-by-field the
way a `NamedTuple` can. The pieces are converted ([`export_view`](@ref), [`host_tree`](@ref),
[`host_rngs`](@ref)) and the assembled whole is asserted, so a device value reachable through a route
the conversions do not cover, a constant baked into a model by `build_model`, say, is still named
here rather than shipped.
"""
function check_export_residency(program, ps, st)
    leaked = String[]
    device_paths(program, "program", leaked)
    device_paths(ps, "ps", leaked)
    device_paths(st, "st", leaked)
    isempty(leaked) && return nothing
    return error(
        """
        ReactantNitro: `export_model` is about to trace values that are still DEVICE-resident, at:
            $(join(leaked, "\n    "))
        Reactant lifts each one into an argument of the compiled module. A bundle declares its inputs
        and serializes its weights, and can name neither of those for a lifted value, so the artifact
        writes and registers successfully and then fails every inference with
        `Execution supplied N arguments but compiled program expected N+$(length(leaked))`.

        A path under `program.ev` is an experiment field: `export_view` freezes `Device`
        fields to host values, so reaching this means an `export_view` method for this experiment
        overrode that, or the field holds something the generic walk cannot rebuild.
        A path ending in `.rng.seed` is an RNG in layer state, which `host_rngs` replaces.
        Anything else is a device value baked into the model itself by `build_model`; it belongs in
        `ps` (where it is serialized) or in a `GraphConst` field (where it bakes into the graph)."""
    )
end

# ── Validation, which is where the framework earns its keep ─────────────────────────

# Batch-last is the framework's own rule, asserted on every array leaf of a batch and of `forward`'s
# return. A tracer derives each tensor's batch axis as its last axis, and those two facts
# agree only as long as somebody checks. This is that check.
function check_export_batch_last(specs::AbstractVector{ExportSpec}, hook::Symbol)
    for s in specs
        isempty(s.shape) && continue
        s.batch_axis === nothing && error(
            """
            ReactantNitro: `$hook` declares `$(s.name)` with no batch axis. Every executable tensor
            is batched in this framework, because the batch-last rule requires the batch dimension
            last on every array leaf of a batch and of `forward`'s return.
            An unbatched tensor belongs in `export_client_outputs`, which the framework derives
            nothing about and constrains nothing about."""
        )
        s.batch_axis == length(s.shape) || error(
            """
            ReactantNitro: `$hook` declares `$(s.name)` with `batch_axis = $(s.batch_axis)` and a
            $(length(s.shape))-dimensional shape, so its batch axis is not last.
            Batch-last is required on every array leaf, the framework asserts it everywhere else,
            and a tracer derives each tensor's batch axis as its last axis. A bundle written from a
            non-last declaration would disagree with the graph it ships."""
        )
    end
    return nothing
end

function check_export_derived_only(specs::AbstractVector{ExportSpec}, hook::Symbol)
    for s in specs
        s.axis_letters === nothing || error(
            """
            ReactantNitro: `$hook` gives `axis_letters` for `$(s.name)`, and the executable side of a
            bundle cannot carry them: its specs are derived from the trace by the backend, not
            declared here.
            `axis_letters` applies to `export_client_outputs`, which the framework does not
            derive."""
        )
        any(==(-1), s.shape) && error(
            """
            ReactantNitro: `$hook` declares `$(s.name)` with a `-1` axis. A variable axis is a
            property of a postprocess, whose output the framework never sees; an executable tensor is
            traced and its shape is a fact.
            Declare the variable axis in `export_client_outputs` instead."""
        )
    end
    return nothing
end

function check_export_inputs(specs)
    (specs isa AbstractVector{ExportSpec} && !isempty(specs)) || error(
        """
        ReactantNitro: `export_inputs` must return a non-empty `Vector{ExportSpec}`, and returned
        $(specs isa AbstractVector ? "an empty vector" : "a `$(typeof(specs))`")."""
    )
    for s in specs
        (s.dtype !== nothing && !isempty(s.shape)) || error(
            """
            ReactantNitro: `export_inputs` declares `$(s.name)` without a dtype and shape, and both
            are required: the framework builds the example arrays it traces from them, and there is
            nothing to derive them from on the input side.
            Write `ExportSpec("$(s.name)", UInt8, [w, h, c, 1])`."""
        )
    end
    for sp in specs
        sp.from === nothing || error(
            """
            ReactantNitro: `export_inputs` gives `from` for `$(sp.name)`. `from` says where an OUTPUT's
            data comes from, and an input is where data comes from."""
        )
    end
    check_export_derived_only(specs, :export_inputs)
    check_export_batch_last(specs, :export_inputs)
    names = spec_names(specs)
    allunique(names) || error("ReactantNitro: `export_inputs` has duplicate names $(names).")
    return specs
end

"""
    ReactantNitro.check_export_sources(specs, sources, in_names) -> Tuple{Vararg{Symbol}}

Split `export_outputs`'s sources into the ones `forward` produces and the ones that are ECHOES of wire
inputs, returning the second set. Refuses the one combination that would be silently wrong: a source
that is both a wire input and something `forward` returns, where merging could shadow either.
"""
function check_export_sources(specs::AbstractVector{ExportSpec}, sources::Tuple, in_names::Tuple)
    echoes = Symbol[]
    for (i, k) in enumerate(sources)
        k in in_names || continue
        specs[i].from === nothing && error(
            """
            ReactantNitro: `export_outputs` names `$k`, which is also a wire input. If echoing that
            input back to the client is what you mean, say so: `ExportSpec("$k"; from = :$k)`.
            An echo is worth being explicit about, because it makes a client's own tensor a program
            OUTPUT, and reading it as a coincidence of names would be a guess."""
        )
        push!(echoes, k)
    end
    return Tuple(echoes)
end

function check_export_outputs(specs)
    (specs isa AbstractVector{ExportSpec} && !isempty(specs)) || error(
        """
        ReactantNitro: `export_outputs` must return a non-empty `Vector{ExportSpec}`, and returned
        $(specs isa AbstractVector ? "an empty vector" : "a `$(typeof(specs))`").
        A bundle with no outputs is not a model."""
    )
    check_export_derived_only(specs, :export_outputs)
    check_export_batch_last(specs, :export_outputs)
    names = spec_names(specs)
    allunique(names) || error("ReactantNitro: `export_outputs` has duplicate names $(names).")
    return specs
end

# The pairing the SERVER enforces at load time, enforced here instead. `client_inputs`/`client_outputs`
# are valid in a manifest only when a `model.jl` is present, so declaring either without a postprocess
# writes a bundle that fails when something tries to serve it, which is both the latest and the most
# expensive moment to find out.
function check_export_postprocess(postprocess, client_outputs, client_inputs)
    if postprocess !== nothing && !(postprocess isa AbstractString)
        error(
            """
            ReactantNitro: `export_postprocess` must return the `model.jl` SOURCE as a string, or
            `nothing`, and returned a `$(typeof(postprocess))`."""
        )
    end
    check_client_specs(client_outputs, :export_client_outputs, postprocess)
    check_client_specs(client_inputs, :export_client_inputs, postprocess)
    return nothing
end

function check_client_specs(specs, hook::Symbol, postprocess)
    specs === nothing && return nothing
    if postprocess === nothing
        error(
            """
            ReactantNitro: `$hook` declares a client-facing spec, and `export_postprocess` returns
            nothing, so no `model.jl` ships.
            Client-facing specs are only meaningful when a postprocess sits between the executable
            tensors and the client, and a bundle carrying them without one is rejected when the
            server loads it. Implement `export_postprocess`, or drop `$hook` and let the executable
            specs be the client's contract."""
        )
    end
    (specs isa AbstractVector{ExportSpec}) || error(
        """
        ReactantNitro: `$hook` must return a `Vector{ExportSpec}` or `nothing`, and returned a
        `$(typeof(specs))`."""
    )
    for s in specs
        (s.dtype !== nothing && !isempty(s.shape)) || error(
            """
            ReactantNitro: `$hook` declares `$(s.name)` without a dtype and shape, and both are
            required. A postprocess is Julia the framework never runs, so there is nothing here to
            derive them from."""
        )
        s.from === nothing || error(
            """
            ReactantNitro: `$hook` gives `from` for `$(s.name)`. `from` resolves a source inside the
            traced program, and the client side is on the far side of a `model.jl` the framework never
            runs, so there is nothing there for it to resolve against."""
        )
    end
    return nothing
end

# Verify a declared output spec against what the trace actually produced. Declaring is optional; when
# it is done it is an assertion, and an assertion that does not fire is worth nothing.
function check_declared_output(s::ExportSpec, got::AbstractArray)
    if s.dtype !== nothing && eltype(got) !== s.dtype
        error(
            """
            ReactantNitro: `export_outputs` declares `$(s.name)` as `$(s.dtype)` and `forward`
            produced `$(eltype(got))`."""
        )
    end
    if !isempty(s.shape)
        # The batch axis is whatever `batch_sizes` asked for, so it is not part of the assertion.
        want = copy(s.shape)
        have = collect(Int, size(got))
        length(want) == length(have) || error(
            """
            ReactantNitro: `export_outputs` declares `$(s.name)` with a $(length(want))-dimensional
            shape $(want) and `forward` produced a $(length(have))-dimensional $(have)."""
        )
        b = s.batch_axis
        b === nothing || (want[b] = have[b])
        want == have || error(
            """
            ReactantNitro: `export_outputs` declares `$(s.name)` as $(s.shape) and `forward` produced
            $(have), which differ outside the batch axis.
            Declaring a shape here is an assertion; drop it to take the traced shape instead."""
        )
    end
    return nothing
end

# ── Provenance, which belongs to the framework and to no backend ────────────────────

"""
    export_provenance(nitro) -> Dict{String,Any}

What the framework knows about how these weights came to exist, and nothing else.

Provenance is the same question whatever the artifact is, which is why it is resolved here rather
than in a backend: the `preset` name travelling with the bundle is what finally lets a served model
answer "which recipe produced you", instead of that fact living in a launch script nobody kept.

It returns the flat config from `config_params`, the preset name recorded on the handle, this
package's version, the seed, and the run directory.

**It deliberately does not guess at repository state.** A git commit, a tree hash and a working-tree
patch are site policy, not framework knowledge, and a framework that shelled out to `git` would be
asserting that the process's working directory is the model's repository. `export_model` takes a
`provenance` dictionary that is merged on top, which is where that half belongs. A backend may stamp
its own facts underneath, and its own version string is its own business.

**It stamps `checkpoint` when the handle restored from one**, naming the file the restore actually
read. That is framework knowledge and not a guess: the path was handed to the constructor, or, under
`resume = :auto`, resolved by the framework's own search, and `Nitro.checkpoint_source` retains
whichever it was. A handle built from freshly initialized weights omits the key rather than carrying
an empty one, so "no checkpoint" and "some checkpoint" are distinguishable in the manifest.

**It stamps the TRAINING RUN's id and url** as `trained_run_id` and `trained_run_url`, taken from the
restored record and not from this handle's logger. That distinction is the point: a
`checkpoint = path` construction gets a fresh logger, so the handle's own `run_id` names the process
doing the exporting. A manifest carrying that would name an experiment holding an export trace and no
training metrics. With the record's id in the manifest, everything else about the run, the training
commit, the branch, the full logged hyperparameter set, is one link away rather than something a
reader has to infer from a run directory's name.

**One thing it still does not carry:** the checkpoint's epoch and its metric. A weights-only restore
deliberately zeroes the epoch counter rather than continuing it, so the handle's `epoch` is not the
checkpoint's; the record's `epoch`, `step` and `metrics` are available at construction and simply are
not retained. Stamping them is the same two lines as the run id above if it turns out to be wanted.
"""
function export_provenance(nitro::Nitro)
    cfg = config_params(nitro.e; seed = nitro.seed)
    prov = Dict{String, Any}(
        "framework" => "ReactantNitro.jl",
        "reactantnitro_version" => framework_version(),
        "seed" => nitro.seed,
        "run_dir" => nitro.run_dir,
        "config" => Dict{String, Any}(String(k) => _prov_value(v) for (k, v) in pairs(cfg)),
    )
    nitro.preset === nothing || (prov["preset"] = String(nitro.preset))
    nitro.checkpoint_source === nothing ||
        (prov["checkpoint"] = String(nitro.checkpoint_source))
    # A warm start names the handle it took its weights from, which has no path: the experiment,
    # the epoch and step it had reached, and its run directory are what a reader can chase.
    ws = nitro.weights_source
    ws === nothing || (
        prov["weights_from"] = Dict{String, Any}(
            "experiment" => String(ws.experiment), "epoch" => ws.epoch,
            "step" => ws.step, "run_dir" => ws.run_dir,
        )
    )
    # The training run, from the record rather than from this handle's logger, which on an export is
    # a fresh one. Omitted rather than empty when there is none, exactly as `checkpoint` is.
    nitro.trained_run_id === nothing ||
        (prov["trained_run_id"] = String(nitro.trained_run_id))
    nitro.trained_run_url === nothing ||
        (prov["trained_run_url"] = String(nitro.trained_run_url))
    return prov
end

# A manifest is a serialized document, so a value that only Julia can read is a value that reaches
# the file as something unpredictable. Numbers stay numbers and everything else becomes its printed
# form, which is what a hyperparameter table wanted anyway.
_prov_value(x::Union{Real, Bool}) = x
_prov_value(x::AbstractString) = String(x)
_prov_value(x) = string(x)

# Every provenance layer is keyed by STRING before it is merged, so a hook returning a `NamedTuple` or
# a `Dict{Symbol}` merges with the framework's keys instead of landing beside them. Values are passed
# through untouched here, unlike `config_params`': a site collector's `git_diff` is a multi-line patch
# and a model's `class_names` is a vector, and both have to reach the writer as themselves.
_prov_dict(d) = Dict{String, Any}(string(k) => v for (k, v) in pairs(d))

# ── The entry point ─────────────────────────────────────────────────────────────────

"""
    export_model(nitro, backend; dir, name, batch_sizes = [1],
                provenance_root = nothing, provenance = Dict()) -> String

Export a trained model to `backend`, and return the path written.

**It takes a `Nitro`, not a path**, which is the single largest simplification this surface makes.
The handle already carries the restored weights, the layer state, the experiment, the preset and the
seed, so export never parses a checkpoint file and no model's export code contains the words "load"
or "checkpoint" anywhere:

```julia
using ReactantServerExport                       # the extension that provides the backend

nitro = Nitro(e; checkpoint = "runs/x/best.jld2", data = (;))
export_model(nitro, ReactantServerBundle(); dir = "export_out", name = "my_model_v1")
```

`data = (;)` is not a workaround. `Nitro(e)` runs setup and nothing else, so an evaluation or
serving construction never needs the training data, and export is the purest case of that: it reads
weights and traces a graph.

**`batch_sizes` is a list because each entry is a separately compiled program**, traced and stored
independently. It defaults to `[1]`, and widening it costs compile time and artifact size in
proportion.

**Export is a CPU trace and asserts one device.** A sharded program is not servable as a bundle, and
the alternative to asserting it here is discovering it when something tries to load the result.

**Provenance is assembled from four sources, and this is the precedence**, lowest first, because
somebody will need to know which one wins:

| Layer | Source | Reaches the bundle when |
| --- | --- | --- |
| backend | the backend's own facts, stamped under everything | always |
| framework | [`export_provenance`](@ref)`(nitro)`: flat config, preset, version, seed, run dir, checkpoint | always |
| site | [`site_provenance`](@ref)`(backend, provenance_root)`: repository state | `provenance_root` is given |
| model | [`export_provenance_extra`](@ref)`(e; checkpoint)` | the experiment implements it |
| explicit | this call's `provenance` | always, and it wins over all of the above |

**`provenance_root` is how repository state reaches a bundle.** Without it the bundle carries no git
commit, no tree hash and no working-tree patch, and it says so by omitting those keys rather than
writing empty ones. That case is a legitimate choice and it is also the failure mode worth naming: an
export with no root SUCCEEDS, every check passes, and the manifest looks complete while being unable
to say which code produced the artifact. Pass the repository root and the backend collects the rest,
patch included. [`export_provenance`](@ref) explains why the framework will not go looking for it
by itself.

The model layer needs no argument at all, which is the point of it being a hook: `export_provenance_extra`
is called with the experiment and with the checkpoint the handle actually restored from, so neither
entry point can forget to pass it and neither can pass a path that disagrees with the loaded weights.

What happens, in order: check the hooks agree with each other and with the batch-last rule, build
example wire arrays at the first batch size, run the program eagerly once to derive the output
shapes and verify the batch-last rule the backend's own derivation depends on, resolve the
provenance, and hand all of it to [`write_export`](@ref). The backend call is published as the
[`ExportCompiling`](@ref) phase, so a phase monitor sees the minutes-long trace rather than a
silent stall; the previous phase is restored when the bundle is written, or when the call fails.

The trace is minutes-long, one CPU compile per batch size, so, like every entry point, this runs
on a worker thread when one is available: the interactive thread's logger tasks and REPL keep
running for the whole export. ^C aborts the wait; the export itself finishes in the background and
completes the bundle, because a partial bundle is worse than a late one.
"""
function export_model(
        nitro::Nitro, backend::ExportBackend;
        dir::AbstractString, name::AbstractString,
        batch_sizes::AbstractVector{<:Integer} = [1],
        provenance_root::Union{AbstractString, Nothing} = nothing,
        provenance = Dict{String, Any}()
    )
    nitro.mesh === nothing || error(
        """
        ReactantNitro: `export_model` requires a single-device handle and this one has a mesh
        ($(nitro.mesh)).
        Export is a CPU trace producing one servable program, and a sharded graph is not that. Build
        the export handle with `n_devs = 1`; the weights are the same weights."""
    )
    isempty(batch_sizes) && error(
        "ReactantNitro: `batch_sizes` must name at least one size, and each one is a separately \
         compiled program."
    )
    all(>(0), batch_sizes) || error("ReactantNitro: every entry of `batch_sizes` must be positive.")

    # `with_repl` wraps only the trace, not the argument checks above: an error raised before any
    # work started is not a phase transition (same rule as `render`). The default interrupt handler
    # applies: ^C aborts the caller's wait while the export completes in the background.
    return with_repl(nitro) do
        # The EXPORT view, not the compile view. A `Device` field is a traced input while
        # the framework owns both ends of the call, and an exported bundle owns neither, so the
        # export view freezes them to host values instead.
        e, ev = nitro.e, export_view(nitro.e)

        inputs = check_export_inputs(export_inputs(e))
        outputs = check_export_outputs(export_outputs(e))
        postprocess = export_postprocess(e)
        client_outputs = export_client_outputs(e)
        client_inputs = export_client_inputs(e)
        check_export_postprocess(postprocess, client_outputs, client_inputs)

        in_names = spec_names(inputs)
        out_names = spec_names(outputs)
        sources = spec_sources(outputs)
        echoes = check_export_sources(outputs, sources, in_names)

        # Host arrays throughout: export is a CPU trace, and a backend that receives device-resident
        # weights would be transferring them back before it could serialize them anyway.
        # `host_rngs` after `host_tree`, because it is the leaf `host_tree` deliberately does not
        # convert: a `ReactantRNG`'s type forbids host contents, so it is passed through, and
        # passing a device seed through to a TRACE is what lifts it into an argument.
        ps = host_rngs(host_tree(nitro.ps))
        st = host_rngs(host_tree(Lux.testmode(nitro.st)))   # the framework owns eval mode

        nb = Int(first(batch_sizes))
        example = ntuple(i -> _example_array(inputs[i], nb), length(inputs))

        has_pre = hasmethod(export_preprocess, Tuple{typeof(ev), map(typeof, example)...})
        router = _export_router(ev, in_names, example, has_pre, nitro.model, ps, st)
        program = ExportForward(in_names, echoes, ev, nitro.model, router, has_pre)
        select = ExportSelect(sources)

        # The residency assertion, BEFORE the probe and the compile because this is the cheapest
        # possible place to learn it. Everything the backend traces has now been converted; this
        # asserts the conversion was total, and names the path of anything that survived it.
        check_export_residency(program, ps, st)

        # One eager pass, host-side, before anything is traced.
        #
        # This is the check that makes the backend's derivation trustworthy rather than merely
        # conventional. A tracer takes each tensor's batch axis to be its last axis; the batch-last
        # rule says the same thing about every leaf `forward` returns; and nothing verifies the two
        # agree unless the framework does it. The cost is one CPU forward, which a tracer performs
        # anyway to learn its own output shapes.
        probe = select(first(program(length(example) == 1 ? example[1] : example, ps, st)))
        check_output_batch_dim(probe, nb)
        for (i, s) in enumerate(outputs)
            check_declared_output(s, probe[i])
        end

        # The backend call is the compile, so it is published as `ExportCompiling` rather than
        # left as a silent stall a monitor cannot distinguish from a wedged process. Restored in a
        # `finally`, because export publishes no `Failed` of its own and a failed backend write must
        # not leave the handle reading as still compiling.
        prev_phase = nitro.phase
        set_phase!(nitro, ExportCompiling())
        try
            # The precedence table above, bottom to top. Each layer is a separate `merge` rather
            # than one call so that the order is readable as the order, and so a layer that
            # contributes nothing (no root given, no hook defined) is visibly a no-op rather than
            # an empty argument.
            prov = export_provenance(nitro)
            provenance_root === nothing ||
                (prov = merge(prov, _prov_dict(site_provenance(backend, String(provenance_root)))))
            prov = merge(
                prov,
                _prov_dict(export_provenance_extra(e; checkpoint = nitro.checkpoint_source))
            )
            prov = merge(prov, _prov_dict(provenance))

            return write_export(
                backend, program, ps, st, example;
                dir = String(dir), name = String(name),
                input_names = String[s.name for s in inputs],
                output_names = String[s.name for s in outputs],
                output_select = select,
                client_inputs = client_inputs,
                client_outputs = client_outputs,
                postprocess = postprocess,
                batch_sizes = collect(Int, batch_sizes),
                provenance = prov,
            )
        finally
            set_phase!(nitro, prev_phase)
        end
    end
end

function _example_array(s::ExportSpec, nb::Integer)
    sz = copy(s.shape)
    s.batch_axis === nothing || (sz[s.batch_axis] = Int(nb))
    return zeros(s.dtype, sz...)
end

# `forward`'s router, resolved against the EXPORT batch rather than a training one.
#
# It has to be resolved here rather than taken from `nitro.routing`, and not only because an export
# handle built with `data = (;)` has none: the batch `export_preprocess` produces is the batch this
# program routes, and it is free to differ from anything a loader ever emitted.
function _export_router(ev, in_names::Tuple, example::Tuple, has_pre::Bool, model, ps, st)
    batch = has_pre ? export_preprocess(ev, example...) : NamedTuple{in_names}(example)
    batch isa NamedTuple || error(
        """
        ReactantNitro: `export_preprocess` must return a NamedTuple, the batch `forward` declares its
        keywords against, and returned a `$(typeof(batch))`."""
    )
    routing = resolve_routing(ev, batch; model = model, ps = ps, st = st)
    routing.forward === nothing && error(
        """
        ReactantNitro: no `forward` method for `$(typeof(ev))`, so there is no program to export.
        `forward` is the exportable program and is what `predict` calls, which is why the
        `forward` / `loss` / `metrics` split exists."""
    )
    return routing.forward
end
