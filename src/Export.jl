# Export.jl
#
# Exporting a trained model: the model's hooks, the framework's resolution of them, and the one
# backend verb a package extension answers. The framework owns what it knows (the program, the
# shapes, the provenance) and never the artifact format.
#
# The exported program is not the program that trained: the wire carries what a client can send,
# usually `UInt8`, and `forward` takes what the model trained on, so what ships is
# `export_preprocess ∘ forward`. Export retraces and never touches the compile cache; what it
# shares with `predict` is the definition of `forward`.

# ── The spec carrier ────────────────────────────────────────────────────────────────

"""
    ExportSpec(name; from = nothing)
    ExportSpec(name, dtype, shape; batch_axis = length(shape), axis_letters = nothing, from = nothing)

One tensor's declaration, in the framework's vocabulary; a backend translates it. `shape` is the
Julia shape and `batch_axis` a 1-based Julia axis. Which fields are required depends on the hook:

| hook | `dtype`/`shape` | `batch_axis` | `axis_letters`, `-1` |
| --- | --- | --- | --- |
| [`export_inputs`](@ref) | **required**, the framework builds example arrays from them | must be last | rejected |
| [`export_outputs`](@ref) | optional, derived from the trace and verified if given | must be last | rejected |
| [`export_client_outputs`](@ref) | **required**, a postprocess is opaque Julia | free | allowed |

The executable side is traced, so its shapes are facts and batch-last is enforced; the client side
is the output of a `model.jl` the framework never runs, so it declares everything. `axis_letters`
names the non-batch axes in `shape` order, e.g. `['w', 'h']`; a `-1` in `shape` marks a variable
non-batch axis. `from` names the source of an output separately from what the bundle calls it, and
is meaningful only on [`export_outputs`](@ref).
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

Where each output spec's data comes from: its `from` when it declares one, else its own name.
"""
spec_sources(specs::AbstractVector{ExportSpec}) =
    Tuple(s.from === nothing ? Symbol(s.name) : s.from for s in specs)

# ── The model's hooks ───────────────────────────────────────────────────────────────

"""
    export_inputs(e) -> Vector{ExportSpec}

Required. The wire contract: what a client sends, in the order it sends it. The order fixes the
positional order the program is traced with and the tensor names in the bundle, so reordering
changes the wire contract silently. Each spec needs a `dtype` and a `shape`, since the framework
builds the example arrays it traces from them, and the batch axis must be last; its size is a
placeholder overwritten by each of `export_model`'s `batch_sizes`.

```julia
export_inputs(e::CropClassifier) = [ExportSpec("img", UInt8, [e.sz, e.sz, 1, 1])]
```
"""
function export_inputs end

"""
    export_outputs(e) -> Vector{ExportSpec}

Required. Which leaves of `forward`'s output tree ship, and what they are called. `forward` returns
what training needs, routinely more than a client wants, and naming a subset needs no second
program. Each name must be a key of the `NamedTuple` `forward` returns, or the single spec naming a
bare array return. `dtype` and `shape` are derived from the trace; supplying them makes them
assertions.

`from` separates a bundle's tensor name from the leaf that produced it, so an export concern never
rewrites the trained program: a rename (`state_out` from `forward`'s `g_pred`) and an echo (a wire
input a serve-time postprocess needs back, since it receives outputs only) would otherwise both
require `forward` to return an extra leaf and recompile the gradient program.

```julia
export_outputs(e::Refiner) = [
    ExportSpec("state_out"; from = :g_pred),   # rename: a leaf `forward` returns
    ExportSpec("path_len"),                    # the ordinary case, name IS the source
    ExportSpec("valid"; from = :valid),        # echo: a wire input, returned to the client
]
```

An echo must say so: a spec whose name matches a wire input is refused without an explicit `from`.
The value echoed is the wire value, not the preprocessed one.
"""
function export_outputs end

"""
    export_preprocess(e, wire...) -> NamedTuple

Optional, and traced: the seam between the wire and the batch. It receives one positional argument
per [`export_inputs`](@ref) spec and returns the `NamedTuple` batch `forward` declares its keywords
against, inside the compiled program. With no method, each wire input maps to a batch field of the
same name. It obeys every rule a traced hook obeys.

It is also called eagerly, on host arrays, by `export_model`'s verification probe and the backend's
shape discovery, so it has to work in both modes, and the most common thing it does works in only
one: `Float32.(img)` has no traced broadcast method for an integer array. The wire conversion needs
two methods:

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

Optional. The source of the `model.jl` that ships in the bundle, for work that does not belong in
a compiled graph: a softmax, a decode. The framework never runs it. Returning a source string
commits you to [`export_client_outputs`](@ref) or [`export_client_inputs`](@ref) when the
postprocess changes what a client receives; `nothing` (the default) commits you to neither.
"""
export_postprocess(e) = nothing

"""
    export_client_outputs(e) -> Vector{ExportSpec}

Optional. What a client receives after [`export_postprocess`](@ref) has run, when that differs from
what the executable emits; refused without a postprocess, since the server would reject the bundle.
The framework cannot derive these, so `dtype` and `shape` are required, `batch_axis` is
unconstrained (unenforceable rather than encouraged; batch-last remains the convention), and
`axis_letters` and `-1` axes are available.

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

Optional. The client-facing spec of the inputs, for the one thing not derivable about them:
`axis_letters`. The executable inputs already are the wire inputs, so with no method the server
falls back to them; the tracer cannot carry letters, so a model that wants `"whcn"` in its manifest
says so here. Requires [`export_postprocess`](@ref), as [`export_client_outputs`](@ref) does.

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

The model's half of a bundle's provenance, merged on top of the framework's and the site's; the
default is empty. It carries what a consumer of this particular model needs and no tensor shape
reveals: the labeling variant, the class names in order, the crop geometry. A hook rather than a
caller-merged dictionary because omitting the argument produced a bundle that looked complete and
was untraceable.

`checkpoint` is the file the weights came from, or `nothing` for fresh weights, passed from
`Nitro.checkpoint_source`; a model asserting anything about its training data should report those
fields as unverifiable when there is no checkpoint behind them.

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

This half wins over the framework's and the site's keys; the explicit `provenance` argument wins
over it.
"""
export_provenance_extra(e; checkpoint = nothing) = Dict{String, Any}()

# ── The backend contract ────────────────────────────────────────────────────────────

"""
    ReactantNitro.ExportBackend

The supertype of an export target. One verb, [`write_export`](@ref), dispatches on the backend
value. The framework ships no method for any backend; `ReactantServerBundle`'s lives in a package
extension, so this package depends on no artifact format.
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

The one verb a backend implements, returning the path it wrote. `model` is a callable with the
`model(inputs, ps, st) -> (outputs, st)` shape a Lux-style tracer expects, already carrying
`compile_view(e)`, the routing, the traced preprocess and the output selection. Keywords, all
supplied by [`export_model`](@ref): `dir`, `name`; `input_names` and `output_names` in program
order; `output_select`, mapping the raw `forward` return to the ordered tuple that ships;
`client_inputs`, `client_outputs`, `postprocess` (`nothing` or the client-side pieces);
`batch_sizes`, each a separately compiled program; `provenance`, already merged.
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

The site's half of a bundle's provenance: repository state collected at `root`, which the caller
names so nothing is guessed. A verb on the backend because the artifact format owns what it can
record: `ReactantServerBundle` answers with `ReactantServerExport.collect_provenance`, whose
`git_diff` becomes `working_tree.patch` in the bundle. No default method, deliberately: a backend
that cannot collect site provenance must say so rather than return an empty dictionary.
"""
function site_provenance end

# Erroring rather than returning `Dict()` is the point.
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

# The framework's `forward` is keyword-routed by batch field name; a tracer wants one positional
# input object. `K` is a type parameter so the NamedTuple construction resolves at trace time.
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
    # The echo: a wire tensor the postprocess needs back, merged here rather than asked of `forward`
    # so an export concern stays out of the trained program. The WIRE value is echoed. Merged after
    # `forward`, so a name it returned is not overwritten; `check_export_sources` refuses that.
    isempty(ECHO) && return outputs, st_new
    wired = NamedTuple{K}(wire)
    return merge(outputs, NamedTuple{ECHO}(map(k -> getfield(wired, k), ECHO))), st_new
end

# Select the named leaves of `forward`'s return, in declared order. `copy` is load-bearing: an
# output arriving through a reshape serializes as its raw producer shape unless copied here, and
# the bundle's declared and actual output shapes then disagree with no error.
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

The view of an experiment a frozen trace sees: [`compile_view`](@ref)'s stripping of every `Host`
field, then every surviving device-resident value read back to the host.

A [`Device`](@ref) field is a traced input while the framework owns both ends of the call. An
exported bundle is `executable(inputs..., weights...)` and nothing more, and Reactant lifts every
device-resident value reachable from a traced closure into an argument the bundle has no name for,
so the artifact fails every inference with `Execution supplied N+1 arguments but compiled program
expected N+4`. So the export view freezes: a `Device` field `forward` never reads becomes a host
value nothing reads, and one it does read bakes into the graph as the value it held. Filtering
could not work, since nothing declares which fields `forward` reads. A large frozen array costs
artifact size once per batch size; an experiment carrying one should override this method.
"""
export_view(e) = host_tree(compile_view(e))

"""
    ReactantNitro.host_rngs(x) -> x

Every `AbstractRNG` in a tree replaced by a host RNG, on the parameters and layer state on the way
to a backend. This closes [`host_tree`](@ref)'s one deliberate blind spot: `Reactant.ReactantRNG`
passes through it unchanged, and at export the state is traced, so a device-resident seed is lifted
into an MLIR argument exactly as a `Device` field is. Replaced rather than removed, since a
testmode `Dropout` still dispatches on the RNG's type. This assumes the exported program never
draws from an RNG; a layer that samples at inference would bake one fixed draw.
"""
host_rngs(::Random.AbstractRNG) = Random.Xoshiro(0)

function host_rngs(x::Union{Tuple, NamedTuple})
    y = map(host_rngs, x)
    return all(i -> y[i] === x[i], 1:length(x)) ? x : y
end

# The same identity-preserving struct walk as `to_host` and `host_tree`: an RNG in layer state
# lives inside a struct.
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

The export view's invariant, asserted before the trace: nothing reachable from the traced closure
is device-resident. The same fault otherwise costs a compile, a write, a deploy and a serve-time
round trip to surface as an argument-count mismatch. It walks `program` rather than converting it,
since `ExportForward` carries its names as type parameters; the pieces are converted and the whole
is asserted, so a device value baked into the model by `build_model` is named here.
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

# Batch-last is asserted on every array leaf of a batch and of `forward`'s return, and a tracer
# derives each tensor's batch axis as its last axis; this is what keeps the two in agreement.
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

# The pairing the server enforces at load time, enforced here: client specs are valid only with a
# `model.jl`.
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

What the framework knows about how these weights came to exist: the flat config, the preset name,
this package's version, the seed and the run directory. It does not guess at repository state,
which is [`site_provenance`](@ref)'s job.

It stamps `checkpoint` when the handle restored from one, naming the file the restore actually
read (`Nitro.checkpoint_source`), and omits the key for fresh weights. It stamps the training run's
`trained_run_id` and `trained_run_url` from the restored record rather than this handle's logger,
which on a `checkpoint = path` construction is a fresh one naming the exporting process. The
checkpoint's epoch and metric are not carried; the record has them if that is ever wanted.
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

# A manifest is a serialized document: numbers stay numbers, everything else becomes its printed
# form.
_prov_value(x::Union{Real, Bool}) = x
_prov_value(x::AbstractString) = String(x)
_prov_value(x) = string(x)

# Every layer is keyed by string before merging, so a hook returning a `NamedTuple` merges with the
# framework's keys. Values pass through untouched: a `git_diff` is a multi-line patch.
_prov_dict(d) = Dict{String, Any}(string(k) => v for (k, v) in pairs(d))

# ── The entry point ─────────────────────────────────────────────────────────────────

"""
    export_model(nitro, backend; dir, name, batch_sizes = [1],
                provenance_root = nothing, provenance = Dict()) -> String

Export a trained model to `backend`, and return the path written. It takes a `Nitro`, not a path:
the handle carries the restored weights, the experiment, the preset and the seed, so no model's
export code loads a checkpoint.

```julia
using ReactantServerExport                       # the extension that provides the backend

nitro = Nitro(e; checkpoint = "runs/x/best.jld2", data = (;))
export_model(nitro, ReactantServerBundle(); dir = "export_out", name = "my_model_v1")
```

`data = (;)` is not a workaround: export reads weights and traces a graph. Each entry of
`batch_sizes` is a separately compiled program. Export is a CPU trace and requires a single-device
handle.

Provenance is assembled from four sources, lowest precedence first:

| Layer | Source | Reaches the bundle when |
| --- | --- | --- |
| backend | the backend's own facts | always |
| framework | [`export_provenance`](@ref)`(nitro)`: config, preset, version, seed, run dir, checkpoint | always |
| site | [`site_provenance`](@ref)`(backend, provenance_root)`: repository state | `provenance_root` is given |
| model | [`export_provenance_extra`](@ref)`(e; checkpoint)` | the experiment implements it |
| explicit | this call's `provenance` | always, and it wins |

Without `provenance_root` the bundle carries no commit, tree hash or working-tree patch, and says
so by omitting the keys: an export with no root succeeds and cannot say which code produced it.

In order: check the hooks against each other and the batch-last rule, build example wire arrays,
run the program eagerly once to derive and verify the output shapes, resolve the provenance, and
hand everything to [`write_export`](@ref), published as [`ExportCompiling`](@ref). Like every entry
point it runs on a worker thread; ^C aborts the wait and the export completes in the background,
since a partial bundle is worse than a late one.
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

    # `with_repl` wraps only the trace: an error before any work started is not a phase transition.
    return with_repl(nitro) do
        # The export view, not the compile view: `Device` fields frozen to host values.
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

        # Host arrays throughout. `host_rngs` after `host_tree`, which passes a `ReactantRNG`
        # through, and a device seed reaching a trace is lifted into an argument.
        ps = host_rngs(host_tree(nitro.ps))
        st = host_rngs(host_tree(Lux.testmode(nitro.st)))   # the framework owns eval mode

        nb = Int(first(batch_sizes))
        example = ntuple(i -> _example_array(inputs[i], nb), length(inputs))

        has_pre = hasmethod(export_preprocess, Tuple{typeof(ev), map(typeof, example)...})
        router = _export_router(ev, in_names, example, has_pre, nitro.model, ps, st)
        program = ExportForward(in_names, echoes, ev, nitro.model, router, has_pre)
        select = ExportSelect(sources)

        # Before the probe and the compile: the cheapest place to learn a value is still on device.
        check_export_residency(program, ps, st)

        # One eager pass, host-side: a tracer takes each tensor's batch axis to be its last, the
        # batch-last rule says the same of every leaf `forward` returns, and nothing verifies the
        # two agree unless the framework does.
        probe = select(first(program(length(example) == 1 ? example[1] : example, ps, st)))
        check_output_batch_dim(probe, nb)
        for (i, s) in enumerate(outputs)
            check_declared_output(s, probe[i])
        end

        # The backend call is the compile, published as `ExportCompiling` and restored in a
        # `finally` since export publishes no `Failed` of its own.
        prev_phase = nitro.phase
        set_phase!(nitro, ExportCompiling())
        try
            # The precedence table, bottom to top, one `merge` per layer so the order reads as the
            # order.
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

# `forward`'s router, resolved against the export batch rather than `nitro.routing`: an export
# handle built with `data = (;)` has none, and the batch `export_preprocess` produces may differ
# from anything a loader emitted.
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
