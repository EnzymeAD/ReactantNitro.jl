# Config.jl
#
# The `@experiment` macro, the `Device`, `Host`, and `GraphConst` markers, config metadata,
# `compile_view`, and the `StrippedHost{name}` sentinel. Docstrings are read by the macro as bare
# adjacent `String`s; comments are stripped by the parser and are never visible to a macro.

# ── The three field markers ─────────────────────────────────────────────────────────

"""
    Device{T}

Declaration-time marker for an [`@experiment`](@ref) field; the macro consumes it and records the
field in [`device_fields`](@ref). A `Device` field is converted to device residency at setup and
reaches the traced step as an input, so `e.aux_weight` in a hook is the device value. One marker,
three jobs: device placement, exclusion from the compile cache key (a traced input cannot affect
the graph), and schedulability (constant and scheduled are the same slot). Mark what you intend to
revise or schedule; a `Device` threshold of zero emits ops a host `0f0` folds away.

```julia
@experiment struct MyExp
    "Weight of the auxiliary heatmap loss relative to the primary term."
    aux_weight::Device{Float32} = 0.25f0
end
```

See also [`Host`](@ref), the default, and [`GraphConst`](@ref), which bakes.
"""
struct Device{T} end

"""
    Host{T}

Declaration-time marker for an [`@experiment`](@ref) field, and the default for an unmarked one:
`max_epochs::Int = 40` means `max_epochs::Host{Int} = 40`. A `Host` field is never converted,
never hashed into the compile cache key, and never visible to the tracer, which keeps driver knobs
out of the key and dataset-sized state away from Reactant and Enzyme, which walk the whole
`Const(e)` argument element-wise at every compile. [`compile_view`](@ref) replaces every `Host`
field with a [`StrippedHost`](@ref) sentinel at every trace site.

See also [`Device`](@ref) and [`GraphConst`](@ref).
"""
struct Host{T} end

"""
    GraphConst{T}

Declaration-time marker for an [`@experiment`](@ref) field. A `GraphConst` field bakes as a
trace-time constant and enters the compile cache key: the tracer sees a real `Int` with ordinary
control flow, and a changed value recompiles. The category for structure: layer counts, widths, a
seed deliberately meant to be part of the program. Under-marking is a loud `StrippedHost` error
when traced code reads the field, never a silent wrong program.

```julia
@experiment struct MyExp
    "Number of decoder blocks. Structural: changes the compiled graph."
    n_layers::GraphConst{Int} = 4
end
```
"""
struct GraphConst{T} end

"""
    StrippedHost{name}

The value [`compile_view`](@ref) substitutes for a [`Host`](@ref) field, carrying the field name as
a type parameter. Using one raises through [`_stripped_error`](@ref), which names the field and the
two fixes. The high-traffic operations carry real methods so the message is useful; detection is
unchanged, and a `Host` value merely stored, compared or passed through still escapes.
"""
struct StrippedHost{name} end

"""
    ReactantNitro._stripped_error(::StrippedHost{name}, op)

The message when traced code reads an unmarked field. The framework cannot know which fix was
meant, so it offers `GraphConst` (bake, recompile per value) and `Device` (traced input, no
recompile), plus the commonest answer: read it host-side.
"""
@noinline function _stripped_error(::StrippedHost{name}, op) where {name}
    return error(
        """
        ReactantNitro: `e.$name` was read inside a compiled program (via `$op`), and `$name` is not
        marked, so it is a `Host` field, which is the default.

        A `Host` field is invisible to the tracer: `compile_view(e)` replaces it with a sentinel
        before Reactant and Enzyme ever see the experiment, so there is no value here to compute
        with. This is not a bug in your hook; it is a missing marker on the field.

        Pick the one you meant, in the `@experiment` block:

            $name::GraphConst{T} = ...
                Rolled into the graph as a literal, and part of the compile cache key. Changing it
                RECOMPILES, which is correct: a different value is a different program. Use this for
                anything structural, a shape, a class count, a layer width, a variant selector.

            $name::Device{T} = ...
                Crosses as a traced INPUT, device-resident. Changing it does NOT recompile, and it
                can be put on a schedule. Use this for a numeric knob you might sweep, a loss
                weight, a temperature, a derived per-class statistic.

        If you did not mean to read it under trace at all, read it host-side instead: `build_data`,
        `derive`, `finalize_metrics`, and every accessor see the real `e` with the real value."""
    )
end

# The high-traffic ways a value gets used, each naming the operation. Not exhaustive: an operation
# missing here falls back to Base and still errors, less helpfully.
Base.getproperty(s::StrippedHost, f::Symbol) = _stripped_error(s, "getproperty(e.<field>, :$f)")
Base.convert(::Type{T}, s::StrippedHost) where {T <: Number} = _stripped_error(s, "convert($T, _)")
Base.getindex(s::StrippedHost, i...) = _stripped_error(s, "getindex")
Base.iterate(s::StrippedHost, ::Any...) = _stripped_error(s, "iterate")
Base.length(s::StrippedHost) = _stripped_error(s, "length")
Base.size(s::StrippedHost, ::Any...) = _stripped_error(s, "size")
Base.broadcastable(s::StrippedHost) = _stripped_error(s, "broadcast")
for op in (:+, :-, :*, :/, :^)
    @eval Base.$op(s::StrippedHost, ::Any) = _stripped_error(s, $(string(op)))
    @eval Base.$op(::Any, s::StrippedHost) = _stripped_error(s, $(string(op)))
end

"""
    ReactantNitro.NO_DEFAULT

The value [`config_metadata`](@ref) reports in a field's `default` slot when that field was
declared with no default, i.e. is a required keyword of the generated constructor. It is a distinct
singleton rather than `nothing`, because `nothing` is a legal default.
"""
struct NoDefault end
const NO_DEFAULT = NoDefault()

"""
    ReactantNitro._merge_field_table!(mod, name, table) -> nothing

Append [`@experiment`](@ref)'s generated marker table beneath any docstring the user wrote.
`Base.@__doc__` on the generated struct is what lets `\"\"\"prose\"\"\" @experiment struct ...` parse,
and a second `@doc` would overwrite it, so the two are merged: the user's prose is intent, the table
records which fields bake, cross, or are stripped. This reaches into `Base.Docs`, so it degrades
rather than breaks, and `test/config.jl` asserts the merged result.
"""
function _merge_field_table!(mod::Module, name::Symbol, table::AbstractString)
    prose = ""
    try
        meta = Base.Docs.meta(mod; autoinit = false)
        b = Base.Docs.Binding(mod, name)
        if meta !== nothing && haskey(meta, b)
            md = meta[b]
            if haskey(md.docs, Union{})
                prose = join(filter(x -> x isa AbstractString, collect(md.docs[Union{}].text)))
            end
        end
    catch
        prose = ""
    end
    text = isempty(strip(prose)) ? table : string(rstrip(prose), "\n\n", table)
    try
        # `:module` is required: `REPL.parsedoc` reads it, and a `DocStr` without it throws at
        # `?MyExp`.
        data = Dict{Symbol, Any}(:module => mod, :path => "", :linenumber => 0)
        b = Base.Docs.Binding(mod, name)
        # Installed directly rather than through `Base.Docs.doc!`, which warns "Replacing docs" on
        # every precompile for exactly the append this performs on purpose.
        Base.Docs.initmeta(mod)
        md = get!(Base.Docs.meta(mod), b, Base.Docs.MultiDoc())
        haskey(md.docs, Union{}) || push!(md.order, Union{})
        md.docs[Union{}] = Base.Docs.docstr(text, data)
    catch
        # Leave whatever the docsystem already has. A wrong docstring is worse than an unmerged one.
    end
    return nothing
end

# ── The generated interface, with total fallbacks ───────────────────────────────────
#
# The macro is optional, so all three are exported and every one is total for a hand-written struct.

"""
    device_fields(::Type{E}) -> NTuple{N,Symbol}

The names of `E`'s [`Device`](@ref) fields, in declaration order. Generated by
[`@experiment`](@ref); defaults to `()` for a type that declares none.

Defining this by hand, alongside [`host_fields`](@ref) and [`config_metadata`](@ref), is the
documented escape hatch for an experiment written without the macro.
"""
device_fields(::Type) = ()

"""
    host_fields(::Type{E}) -> NTuple{M,Symbol}

The names of `E`'s [`Host`](@ref) fields, in declaration order. Generated by
[`@experiment`](@ref); defaults to `()` for a type that declares none. This is what
[`compile_view`](@ref) reads.

A field is Host **by default**: an unmarked field is recorded here. The `GraphConst` fields are
the complement, `setdiff(fieldnames, device_fields, host_fields)`.
"""
host_fields(::Type) = ()

"""
    config_metadata(::Type{E}) -> NamedTuple

One entry per field of `E`, in declaration order:

```julia
(; aux_weight = (; type = Float32, default = 0.25f0, kind = :device,
                   doc = "Weight of the auxiliary ...", line = 3), ...)
```

`kind` is `:device`, `:host` or `:graphconst`; `type` is the declared type; `doc` and `line` come
from the declaration; `default` is `ReactantNitro.NO_DEFAULT` for a required field. The fallback
synthesizes a record for a hand-written experiment from `fieldnames` and `fieldtype`, with
`doc = nothing`, `line = 0` and no defaults.
"""
function config_metadata(::Type{E}) where {E}
    fns = fieldnames(E)
    isempty(fns) && return NamedTuple()
    df, hf = device_fields(E), host_fields(E)
    entries = map(fns) do f
        kind = f in df ? :device : f in hf ? :host : :graphconst
        (; type = fieldtype(E, f), default = NO_DEFAULT, kind, doc = nothing, line = 0)
    end
    return NamedTuple{fns}(entries)
end

"""
    compile_view(e) -> e_trace

The view every trace site sees: each [`Host`](@ref) field replaced with a [`StrippedHost`](@ref)
sentinel, everything else passed through, handed to the tracer as `Const`. The real `e` is used
everywhere outside the trace. [`@experiment`](@ref) generates a type-stable method per experiment;
the fallback here covers a hand-written one. It leaves `Device` fields in place, which is right for
every trace the framework invokes and wrong for the one it hands away; see [`export_view`](@ref).
"""
function compile_view(e)
    T = typeof(e)
    hf = host_fields(T)
    isempty(hf) && return e
    vals = map(fieldnames(T)) do f
        f in hf ? StrippedHost{f}() : getfield(e, f)
    end
    return Base.typename(T).wrapper(vals...)
end

"""
    ReactantNitro._field(e, name::Symbol, default)

`getproperty(e, name)` when `e` has that field, `default` otherwise, which is what lets a `Host`
field and a user method be interchangeable behind a defaulted accessor such as
`max_epochs(e) = _field(e, :max_epochs, 1)`. Accessors run host-side against the real `e`.
"""
@inline _field(e, name::Symbol, default) =
    hasfield(typeof(e), name) ? getproperty(e, name) : default

# ── @experiment ─────────────────────────────────────────────────────────────────────

"""
    @experiment struct MyExp ... end

Declare an experiment type. One declaration point generates the struct, a `@kwdef`-style
keyword constructor, the [`device_fields`](@ref) and [`host_fields`](@ref) traits, the
[`config_metadata`](@ref) table, a [`compile_view`](@ref) method, the `Base.show` methods, and a
`?MyExp` docstring.

```julia
@experiment struct MyExp
    "Weight of the auxiliary heatmap loss relative to the primary term."
    aux_weight::Device{Float32} = 0.25f0

    "Number of decoder blocks. Structural: changes the compiled graph."
    n_layers::GraphConst{Int} = 4

    "Epochs to train for. Driver-only: never read inside a traced function."
    max_epochs::Int = 40          # unmarked, so Host: the default
end
```

A [`Device`](@ref) field is a traced input, a [`GraphConst`](@ref) field bakes into the graph and
the cache key, and a [`Host`](@ref) field is invisible to the tracer and is the default. A macro is
necessary because after setup a `Device` field holds a `ConcretePJRTNumber` and nothing at runtime
says it was marked; every `Device` and `Host` field gets its own type parameter so what it holds
can change, while `GraphConst` fields keep their declared type and the constructor `convert`s to
it. Docstrings are bare strings above each field, since comments never reach a macro.

The generated `show` prints values and never a device buffer, summarizing an array as its eltype
and shape, because a `Device{NamedTuple}` of weights would otherwise print element by element. The
macro is optional: a hand-written struct defines the three exported traits itself and may opt into
the display with the `Base.show` one-liners the macro expands to (`_show_experiment`).
"""
macro experiment(expr)
    return _experiment(expr, __source__, __module__)
end

# ── Macro implementation. Split out so it is testable and so the macro body stays one line. ──

"""
    ReactantNitro._qual(name::Symbol, [caller::Module]) -> Expr

Name one of this module's functions for a method definition emitted by [`@experiment`](@ref):
through the symbol `ReactantNitro` when the caller binds it, and through the module object
otherwise. Hygiene would rename an unescaped short-form definition to a gensym, and `esc` cannot
extend a name reached through `using`. The symbolic form is preferred because Pluto's expression
explorer reads a definition head as a chain of symbols and drops a module object.
"""
_qual(name::Symbol) = Expr(:., @__MODULE__, QuoteNode(name))
function _qual(name::Symbol, caller::Module)
    bound = isdefined(caller, :ReactantNitro) && getfield(caller, :ReactantNitro) === @__MODULE__
    return bound ? Expr(:., esc(:ReactantNitro), QuoteNode(name)) : _qual(name)
end

struct _FieldDecl
    name::Symbol
    kind::Symbol          # :device, :host, :graphconst
    type::Any             # the DECLARED type expression: the marker's parameter, or the annotation
    default::Any          # an expression, or NO_DEFAULT
    doc::Union{String, Nothing}
    line::Int
end

# The bare name of a type expression, so `Device`, `ReactantNitro.Device`, and `Device{Float32}`
# all report `:Device`.
function _basename(ex)
    ex isa Symbol && return ex
    Meta.isexpr(ex, :curly) && return _basename(ex.args[1])
    Meta.isexpr(ex, :.) && ex.args[2] isa QuoteNode && return ex.args[2].value
    return nothing
end

# Recognize `Device{T}` / `Host{T}` / `GraphConst{T}`, qualified or not. A bare marker with no
# parameter is an error rather than a plain field of type `Device`.
function _marker(ex)
    name = _basename(ex)
    (name === :Device || name === :Host || name === :GraphConst) || return nothing
    (Meta.isexpr(ex, :curly) && length(ex.args) == 2) || error(
        "ReactantNitro.@experiment: `$name` takes exactly one type parameter, as in \
         `$name{Float32}`; got `$ex`."
    )
    return (name, ex.args[2])
end

function _parse_field(decl, default, doc, line)
    Meta.isexpr(decl, :(::)) || error(
        "ReactantNitro.@experiment: every field needs a type annotation, and got `$decl` on line \
         $line. Write `w::Device{Float32} = 0.25f0` for a traced input, \
         `n_layers::GraphConst{Int} = 4` for a baked constant, or `max_epochs::Int = 40` \
         for a driver-only field (Host, the default)."
    )
    length(decl.args) == 2 || error("ReactantNitro.@experiment: cannot read the field declaration \
                                     `$decl` on line $line")
    name, tyexpr = decl.args
    name isa Symbol || error("ReactantNitro.@experiment: field name `$name` on line $line is not a \
                              symbol")
    m = _marker(tyexpr)
    if m === nothing
        return _FieldDecl(name, :host, tyexpr, default, doc, line)
    end
    marker, inner = m
    _marker(inner) === nothing || error(
        "ReactantNitro.@experiment: field `$name` on line $line nests one marker inside another \
         (`$tyexpr`). A field is exactly one of Device, Host, or GraphConst."
    )
    kind = marker === :Device ? :device : marker === :Host ? :host : :graphconst
    return _FieldDecl(name, kind, inner, default, doc, line)
end

function _experiment(expr, source, mod = @__MODULE__)
    Meta.isexpr(expr, :struct) || error(
        "ReactantNitro.@experiment expects a struct definition, as in \
         `@experiment struct MyExp ... end`; got a `$(expr isa Expr ? expr.head : typeof(expr))`."
    )
    ismutable, sig, blk = expr.args

    # Name and optional supertype. Explicit type parameters are rejected: the macro generates one
    # per Device and Host field, and a user-supplied list would collide with them.
    if Meta.isexpr(sig, :<:)
        namepart, super = sig.args
    else
        namepart, super = sig, nothing
    end
    namepart isa Symbol || error(
        "ReactantNitro.@experiment: `$namepart` declares type parameters. The macro generates one \
         type parameter per Device and Host field, so an experiment struct is declared \
         without them."
    )
    name = namepart

    # Fields, docstrings, and line numbers. A per-field docstring reaches a macro as a bare String
    # adjacent to the field expression, with LineNumberNodes interleaved (verified on 1.12).
    fields = _FieldDecl[]
    line = source === nothing ? 0 : source.line
    doc = nothing
    for a in blk.args
        if a isa LineNumberNode
            line = a.line
        elseif a isa String
            doc === nothing || error("ReactantNitro.@experiment: two docstrings in a row near line \
                                      $line; each field takes at most one.")
            doc = a
        elseif Meta.isexpr(a, :(=))
            push!(fields, _parse_field(a.args[1], a.args[2], doc, line))
            doc = nothing
        elseif Meta.isexpr(a, :(::))
            push!(fields, _parse_field(a, NO_DEFAULT, doc, line))
            doc = nothing
        elseif Meta.isexpr(a, :const)
            error("ReactantNitro.@experiment: `const` fields are not supported; the framework \
                   rebuilds the experiment rather than mutating it.")
        else
            error("ReactantNitro.@experiment: cannot read `$a` near line $line as a field \
                   declaration or a docstring.")
        end
    end
    doc === nothing || error("ReactantNitro.@experiment: a trailing docstring near line $line \
                              belongs to no field.")
    allunique(f.name for f in fields) || error("ReactantNitro.@experiment: duplicate field name in \
                                                `$name`.")

    # ── The struct. Device and Host fields get a free type parameter each, because what they hold
    # changes; GraphConst fields keep their declared type, because they bake.
    tparam(f) = Symbol("__RN_T_", f.name)
    params = [tparam(f) for f in fields if f.kind !== :graphconst]
    fdecls = [
        f.kind === :graphconst ? Expr(:(::), f.name, f.type) : Expr(:(::), f.name, tparam(f))
            for f in fields
    ]
    structsig = isempty(params) ? name : Expr(:curly, name, params...)
    super === nothing || (structsig = Expr(:<:, structsig, super))
    # `Base.@__doc__` is what makes `"""docs""" @experiment struct ...` legal: any macro returning a
    # multi-expression block otherwise fails with "cannot document the following expression".
    structdef = Expr(
        :macrocall, GlobalRef(Base, Symbol("@__doc__")),
        source === nothing ? LineNumberNode(0) : source,
        esc(Expr(:struct, ismutable, structsig, Expr(:block, fdecls...)))
    )

    # The @kwdef-style constructor: a field with no default is a required keyword. A GraphConst
    # field is `convert`ed to its declared type, since Julia's own constructor for a parametric
    # struct does not convert and whether `n_layers` accepts a `UInt8` should not depend on an
    # unrelated field's marker; Device and Host fields are not, since what they hold changes.
    # A fieldless experiment gets no generated constructor: `MyExp(; ) = MyExp()` would replace
    # Julia's zero-argument constructor with one that calls itself.
    kwargs = [f.default === NO_DEFAULT ? f.name : Expr(:kw, f.name, f.default) for f in fields]
    ctorargs = [
        f.kind === :graphconst ? Expr(:call, :convert, f.type, f.name) : f.name
            for f in fields
    ]
    ctor = isempty(fields) ? nothing : esc(
            Expr(
                :(=),
                Expr(:call, name, Expr(:parameters, kwargs...)),
                Expr(:call, name, ctorargs...)
            )
        )

    # Dispatch on `Type{<:MyExp}` so the post-conversion instantiation matches too, which is what
    # lets `config_metadata` report declared types.
    dnames = Expr(:tuple, (QuoteNode(f.name) for f in fields if f.kind === :device)...)
    hnames = Expr(:tuple, (QuoteNode(f.name) for f in fields if f.kind === :host)...)
    df_def = :($(_qual(:device_fields, mod))(::Type{<:$(esc(name))}) = $dnames)
    hf_def = :($(_qual(:host_fields, mod))(::Type{<:$(esc(name))}) = $hnames)

    # ── config_metadata. Built in the function body rather than a module-level const, so a default
    # expression is resolved at call time exactly as the constructor resolves it.
    entry(f) = Expr(
        :tuple,
        Expr(:(=), :type, esc(f.type)),
        Expr(:(=), :default, f.default === NO_DEFAULT ? :NO_DEFAULT : esc(f.default)),
        Expr(:(=), :kind, QuoteNode(f.kind)),
        Expr(:(=), :doc, f.doc === nothing ? :nothing : f.doc),
        Expr(:(=), :line, f.line)
    )
    meta_body = isempty(fields) ? :(NamedTuple()) :
        Expr(
            :call, Expr(:curly, :NamedTuple, Expr(:tuple, (QuoteNode(f.name) for f in fields)...)),
            Expr(:tuple, (entry(f) for f in fields)...)
        )
    cm_def = :($(_qual(:config_metadata, mod))(::Type{<:$(esc(name))}) = $meta_body)

    # ── compile_view, specialized. Generated per type so the reconstruction is type-stable: the
    # tracer sees this on every trace, and the framework rebuilds `e` every optimizer step.
    hostset = Set(f.name for f in fields if f.kind === :host)
    cv_def = if isempty(hostset)
        :($(_qual(:compile_view, mod))(e::$(esc(name))) = e)
    else
        args = [
            f.name in hostset ? :($(_qual(:StrippedHost)){$(QuoteNode(f.name))}()) :
                :(getfield(e, $(QuoteNode(f.name)))) for f in fields
        ]
        :($(_qual(:compile_view, mod))(e::$(esc(name))) = $(esc(name))($(args...)))
    end

    docstr = _experiment_docstring(name, fields)
    # Merge rather than overwrite: `Base.@__doc__` attached the user's prose, and the table is the
    # half a user cannot write.
    doc_def = :($(_qual(:_merge_field_table!))($mod, $(QuoteNode(name)), $docstr))

    # The `show` methods, emitted per type since a generated experiment has no common supertype.
    # One method per MIME (a `Union` would be ambiguous with Base's `text/plain` fallback), spelled
    # `MIME{Symbol("text/plain")}` to avoid a string macro in macro output.
    show_def = :(Base.show(io::IO, e::$(esc(name))) = $(_qual(:_show_experiment))(io, e))
    showl_def = quote
        function Base.show(io::IO, mime::MIME{Symbol("text/plain")}, e::$(esc(name)))
            return $(_qual(:_show_experiment))(io, mime, e)
        end
        function Base.show(io::IO, mime::MIME{Symbol("text/html")}, e::$(esc(name)))
            return $(_qual(:_show_experiment))(io, mime, e)
        end
    end

    parts = Any[
        structdef, ctor, df_def, hf_def, cm_def, cv_def, show_def, showl_def, doc_def, esc(name),
    ]
    return Expr(:block, filter(!isnothing, parts)...)
end

# ── @nitrohook ──────────────────────────────────────────────────────────────────────
#
# Reactive-notebook support only; outside Pluto the macro defines one extra function nobody calls.
# Pluto builds its graph from global variable names, and a hook head is a qualified name
# (`ReactantNitro.loss`) that no cell references, so a hook cell is a reactive dead end and the
# `Nitro` cell never re-runs. The fix is a method of one token function per experiment on the
# defining side: Pluto's conflict test is signature-level while its edge resolution is name-level,
# so any number of cells may define methods of `MyExp_hooks` and every one becomes an upstream
# edge of a cell that names it. This only makes the rebuild fire.

"""
    ReactantNitro.@nitrohook <definition>

Define one or more hooks and, alongside them, one method of `<Experiment>_hooks` per definition,
so a reactive notebook can see that the hooks changed. Pluto's dependency graph is built from
variable names, and a hook is defined on a qualified name no cell references, so editing a hook
cell would otherwise re-run that cell alone and never rebuild the `Nitro`.

```julia
@nitrohook function ReactantNitro.loss(e::MnistMLP, logits; label)
    return -sum(label .* logsoftmax(logits; dims = 1)) / size(label, 2)
end

n = begin
    MnistMLP_hooks          # every hook cell for MnistMLP is now upstream of this one
    Nitro(MnistMLP())
end
```

The token is named for the type of the definition's first argument. For an extension point that
dispatches on something else (`batch_at` and `begin_epoch!` on a data source, `nonschedulable` on
a rule, `default_no_decay` on nothing), name the experiment explicitly:
`@nitrohook MnistMLP ReactantNitro.batch_at(src::MyLoader, i::Integer) = ...`. `@experiment`
needs no macro, since every cell already names the type.

Edit a hook where it is defined rather than appending a second definition; Julia has one method
table and Pluto refuses both cells. Invalidation is all-or-nothing by design, since dispatch
resolves over the whole table when the program is traced. Give `build_data` its own cell: hooks
bundled in one `begin` block are redefined together, and redefining `forward` even byte-identically
poisons the compiled programs, while `build_data` alone costs nothing. Rebuilding remains what puts
new code into effect; [`ReactantNitro.stale_hooks`](@ref) reports the drift.
"""
macro nitrohook(expr)
    return _nitrohook_expansion(nothing, expr)
end

macro nitrohook(experiment, expr)
    experiment isa Symbol || error(
        "ReactantNitro.@nitrohook: the optional first argument names the experiment whose token \
         to use, as in `@nitrohook MyExp ReactantNitro.batch_at(src::MyLoader, i) = ...`, and \
         got `$experiment`."
    )
    return _nitrohook_expansion(experiment, expr)
end

function _nitrohook_expansion(experiment, expr)
    tokens = _hook_tokens(expr, experiment)
    parts = Any[esc(expr)]
    for (tok, hook, key) in tokens
        # `Base.Val` rather than `Val` so a notebook that shadows `Val` still expands. Only the
        # token name is escaped: it is the one piece that must land in the caller's module.
        push!(
            parts,
            :($(esc(tok))(::Base.Val{$(QuoteNode(hook))}, ::Base.Val{$key}) = $(QuoteNode(hook)))
        )
    end
    # The token itself is the expansion's value, so a notebook shows "MyExp_hooks (generic
    # function with N methods)" and the count doubles as a tally of the hooks wired up so far.
    push!(parts, esc(first(last(tokens))))
    return Expr(:block, parts...)
end

# Every `(token, hook)` pair a `@nitrohook` expression calls for, in source order; a `begin` block
# is walked rather than rejected.
function _hook_tokens(expr, experiment = nothing)
    out = Tuple{Symbol, Symbol, UInt}[]
    _collect_hook_tokens!(out, expr, experiment)
    isempty(out) && error(
        "ReactantNitro.@nitrohook: expected a hook definition, as in \
         `@nitrohook function ReactantNitro.loss(e::MyExp, logits; label) ... end`, and got \
         `$expr`."
    )
    return out
end

function _collect_hook_tokens!(out, ex, experiment)
    ex isa LineNumberNode && return out
    if Meta.isexpr(ex, :block)
        for a in ex.args
            _collect_hook_tokens!(out, a, experiment)
        end
        return out
    end
    sig = _call_signature(ex)
    sig === nothing || push!(out, _hook_token(sig, experiment))
    return out
end

# The `:call` at the head of a definition, past any `where` clause and any return-type annotation,
# or `nothing` when the expression is not a named function definition at all.
function _call_signature(ex)
    (Meta.isexpr(ex, :function) || Meta.isexpr(ex, :(=))) || return nothing
    sig = ex.args[1]
    while true
        if Meta.isexpr(sig, :where)
            sig = sig.args[1]
        elseif Meta.isexpr(sig, :(::), 2) && Meta.isexpr(sig.args[1], :call)
            sig = sig.args[1]
        else
            break
        end
    end
    return Meta.isexpr(sig, :call) ? sig : nothing
end

# The token is named for the type the definition dispatches on, the experiment in the first
# positional argument; the two-argument form names it for hooks that dispatch on something else.
function _hook_token(sig, experiment)
    hook = _basename(sig.args[1])
    hook === nothing && error(
        "ReactantNitro.@nitrohook: could not read a hook name out of `$(sig.args[1])`. The \
         definition head must be a plain or qualified name, as in `ReactantNitro.loss`."
    )
    experiment === nothing || return (Symbol(experiment, "_hooks"), hook, _sig_key(sig))
    # The keywords parse into an `Expr(:parameters, ...)` that sorts BEFORE the positional
    # arguments, so the dispatched-on argument is the first non-`:parameters` entry, not `args[2]`.
    positional = filter(a -> !Meta.isexpr(a, :parameters), @view sig.args[2:end])
    isempty(positional) && error(
        "ReactantNitro.@nitrohook: `$hook` takes no positional argument, so there is no type to \
         name its token after. Name one explicitly: `@nitrohook MyExp <definition>`."
    )
    arg = first(positional)
    Meta.isexpr(arg, :(::)) || error(
        "ReactantNitro.@nitrohook: the first argument of `$hook` is `$arg`, which carries no type \
         annotation, so there is no type to name its token after. Write `e::MyExp`, or name the \
         experiment explicitly: `@nitrohook MyExp <definition>`."
    )
    E = _basename(arg.args[end])
    E === nothing && error(
        "ReactantNitro.@nitrohook: could not read a type out of `$arg`. Name the experiment \
         explicitly instead: `@nitrohook MyExp <definition>`."
    )
    return (Symbol(E, "_hooks"), hook, _sig_key(sig))
end

# The token's second `Val`, so two different methods of one hook (`batch_at(src, i)` and
# `batch_at(src, i, plan)`) do not collide on the token. Only the dispatched-on shape goes in, the
# same canonicalization Pluto applies to the definition itself, so a real conflict is still
# reported there, once.
function _sig_key(sig)
    kws, types = Symbol[], Any[]
    for a in @view sig.args[2:end]
        if Meta.isexpr(a, :parameters)
            append!(kws, map(_kwname, a.args))
        else
            push!(types, _argtype(a))
        end
    end
    return hash((Tuple(map(string, types)), Tuple(sort!(map(string, kws)))))
end

_argtype(a) =
    Meta.isexpr(a, :(::)) ? a.args[end] :
    Meta.isexpr(a, :kw) || Meta.isexpr(a, :...) ? _argtype(a.args[1]) : :Any

_kwname(k) = k isa Symbol ? k : Meta.isexpr(k, (:kw, :(::), :...)) ? _kwname(k.args[1]) : :_

# ── Showing an experiment never shows a device buffer ───────────────────────────────
#
# `Device{T}` takes any `T`, and the read-only buffer case puts weights there; under the default
# struct `show` a toy experiment with three small buffers printed 51,635 characters. `_shown`
# (Checkpoint.jl) summarizes an array as its eltype and shape, so the output grows with the field
# count and never with the model. Values, not just names: an experiment is configuration.
function _show_experiment(io::IO, e)
    T = typeof(e)
    fs = fieldnames(T)
    print(io, nameof(T), "(")
    print(io, join(("$f = " * _shown(getfield(e, f)) for f in fs), ", "))
    print(io, ")")
    return nothing
end

function _show_experiment(io::IO, mime::MIME, e)
    T = typeof(e)
    fs = fieldnames(T)
    df, hf = device_fields(T), host_fields(T)
    title = string(nameof(T)) * "  (@experiment; no device buffer is shown)"
    isempty(fs) && return _render_sections(io, mime, title, TableSection[]; note = "(no fields)")
    # The marker is the column that makes the table actionable: it is what a reader changes when a
    # field is in the wrong category, and it cannot be inferred from the value.
    rows = Vector{String}[
        [
            string(f),
            f in df ? "Device" : f in hf ? "Host" : "GraphConst",
            _shown(getfield(e, f)),
        ] for f in fs
    ]
    _render_table(io, mime, title, ["field", "marker", "value"], rows)
    return nothing
end

# Each note names its marker, the word a reader would type to move a field.
const _KIND_NOTE = (
    device = "`Device`, traced input",
    host = "`Host`, driver only, stripped from the trace; the default",
    graphconst = "`GraphConst`, baked constant, in the cache key",
)

function _experiment_docstring(name, fields)
    io = IOBuffer()
    println(io, "    ", name, "(; ", join((string(f.name) for f in fields), ", "), ")")
    println(io)
    println(
        io, "A ReactantNitro experiment, declared with [`@experiment`](@ref). ",
        "Its fields are:"
    )
    println(io)
    println(io, "| Field | Type | Kind | Default |")
    println(io, "| --- | --- | --- | --- |")
    for f in fields
        d = f.default === NO_DEFAULT ? "*required*" : string("`", f.default, "`")
        println(io, "| `", f.name, "` | `", f.type, "` | ", _KIND_NOTE[f.kind], " | ", d, " |")
    end
    docs = [f for f in fields if f.doc !== nothing]
    if !isempty(docs)
        println(io)
        for f in docs
            println(io, "- `", f.name, "`: ", f.doc)
        end
    end
    println(io)
    println(io, "See [`config_metadata`](@ref) for the same information as data.")
    return String(take!(io))
end

# ── Named configurations ────────────────────────────────────────────────────────────

"""
    presets(::Type{E}) -> NamedTuple

The table of named configurations for an experiment type; the contents belong to the model, the
mechanism here. Default is empty. Entries are partial, and everything else falls through to the
struct's defaults; inheritance is `merge` on NamedTuples.

```julia
ReactantNitro.presets(::Type{MyExp}) = (
    reference_v1 = (; n_classes = 8, batch_size = 16, rotate_deg = 7.5),
    current      = (; n_classes = 10, batch_size = 32, rotate_deg = 12.0),
)
```

The framework has an opinion so that the checkpoint record and the logger can say which recipe
produced a result, and so a typo in a key is an error rather than a silent no-op. It is not a
reproducibility mechanism; provenance is.
"""
function presets end
presets(::Type) = (;)

"""
    from_preset(E::Type, name::Symbol; overrides...) -> e

Build an experiment from [`presets`](@ref)`(E)[name]`, with `overrides` winning over the preset.
Every preset key is validated against `fieldnames(E)`. A preset is field values, so the marker
semantics are untouched: switching presets recompiles exactly when a `GraphConst` field differs.
This does not record the name; use [`Nitro`](@ref)`(E, name; ...)` for that.

```julia
e = from_preset(MyExp, :current; max_epochs = 40)
```
"""
function from_preset(E::Type, name::Symbol; overrides...)
    table = presets(E)
    haskey(table, name) || error(
        """
        ReactantNitro: `$(nameof(E))` has no preset `$(repr(name))`.
        $(
            isempty(table) ? "It declares no `presets` method at all." :
                "Its presets are: $(join(map(repr, collect(keys(table))), ", "))."
        )"""
    )
    p = getproperty(table, name)
    p isa NamedTuple || error(
        "ReactantNitro: preset $(repr(name)) on $(nameof(E)) is a `$(typeof(p))`; a preset is a \
         NamedTuple of field values."
    )
    valid = fieldnames(E)
    for k in keys(p)
        k in valid || error(
            """
            ReactantNitro: preset $(repr(name)) on $(nameof(E)) sets `$k`, which is not a field of
            $(nameof(E)). Its fields are: $(join(map(string, valid), ", ")).
            Validated rather than splatted, because an unknown key would otherwise be a silent
            no-op and the recipe would quietly not mean what it says."""
        )
    end
    return E(; merge(p, NamedTuple(overrides))...)
end
