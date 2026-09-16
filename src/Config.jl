# Config.jl
#
# The `@experiment` macro, the `Device`, `Host`, and `GraphConst` markers, config metadata,
# `compile_view`, and the `StrippedHost{name}` sentinel. Docstrings are read by the macro as bare
# adjacent `String`s; comments are stripped by the parser and are never visible to a macro.

# ── The three field markers ─────────────────────────────────────────────────────────

"""
    Device{T}

Declaration-time marker for an [`@experiment`](@ref) field, legal only in that position. It is
never instantiated and never appears in the generated struct: the macro consumes it and records
the field in [`device_fields`](@ref).

A `Device` field is converted to device residency at setup and reaches the traced step as an
**input**, so in-trace `e.aux_weight` is the device value and arithmetic works with no unwrapping.
The one marker does three jobs:

1. **Device placement.** Scalars become `ConcretePJRTNumber`, arrays `ConcretePJRTArray`.
2. **Cache-key exclusion.** A traced input cannot affect the graph, so it is excluded from the
   compile cache key by construction.
3. **Schedulability.** Constant and scheduled are the same device slot; only the write cadence
   differs, and switching between them does not recompile.

**Marking is opt-in.** Forgetting to mark costs a spurious recompile, which is visible and
harmless; marking a structural field `Device` would silently lose constant folding or fail at trace
time. The cost of over-marking is the lost constant folding: a `Device` threshold of zero emits the
ops a plain host `0f0` folds away. Mark what you intend to revise or schedule.

```julia
@experiment struct MyExp
    "Weight of the auxiliary heatmap loss relative to the primary term."
    aux_weight::Device{Float32} = 0.25f0
end
```

See also [`Host`](@ref), which is the default, and [`GraphConst`](@ref), which bakes, and
[`@experiment`](@ref).
"""
struct Device{T} end

"""
    Host{T}

Declaration-time marker for an [`@experiment`](@ref) field, legal only in that position. It is
never instantiated and never appears in the generated struct: the macro consumes it and records
the field in [`host_fields`](@ref).

A `Host` field is **never converted, never hashed into the compile cache key, and never visible to
the tracer**. It does two jobs:

1. **Keeps driver knobs out of the cache key**, so changing `max_epochs` from 40 to 41 does not
   force a full recompile.
2. **Keeps dataset-sized state away from the tracer.** Reactant and Enzyme traverse the whole
   `Const(e)` argument while tracing, so anything dataset-sized reachable from the experiment (a
   sampler, an in-memory table, a materialized index) is walked element-wise, on one thread, every
   time a program is compiled. The cost is O(n_train) and it does not change the emitted graph, so
   nothing about the trained model looks wrong; it just gets slower the more data you have.

**`Host` is the default: an unmarked field is Host.** In practice the vast majority of experiment
fields are driver knobs or dataset-sized state, so the marker is optional and `max_epochs::Int = 40`
means the same as `max_epochs::Host{Int} = 40`. Write the marker explicitly only where the field's
host-ness would otherwise be surprising.

The guard is [`compile_view`](@ref), which replaces every `Host` field with a
[`StrippedHost`](@ref) sentinel. The framework passes that view to every trace site and uses the
real `e` everywhere outside the trace.

Typical `Host` fields: `max_epochs`, output paths, log cadence, checkpoint retention,
early-stopping patience, and any materialized dataset an experiment chooses to carry.

See also [`Device`](@ref) and [`GraphConst`](@ref) and [`@experiment`](@ref).
"""
struct Host{T} end

"""
    GraphConst{T}

Declaration-time marker for an [`@experiment`](@ref) field, legal only in that position. It is
never instantiated and never appears in the generated struct: the macro consumes it and records
the field, which the framework reaches as `setdiff(fieldnames, device_fields, host_fields)`.

A `GraphConst` field **bakes as a trace-time constant and enters the compile cache key**. The
tracer sees the real value, so `e.n_layers` is a Julia `Int` with ordinary control-flow semantics;
changing it is a different program, which is exactly why it is hashed and why a changed value
recompiles rather than silently running the old graph. This is the field category for **structure**:
anything whose value changes the shape or meaning of the emitted graph, such as layer counts, widths,
or a seed that is deliberately meant to be part of the program.

**Marking is opt-in and the opposite of the default.** An unmarked field is [`Host`](@ref), so a
field that would previously have been left plain must now be written `GraphConst` to keep baking. The
cost of over-marking is a recompile per distinct value; the cost of under-marking is a loud
`StrippedHost` error when traced code reads the field, never a silent wrong program.

```julia
@experiment struct MyExp
    "Number of decoder blocks. Structural: changes the compiled graph."
    n_layers::GraphConst{Int} = 4
end
```

See also [`Device`](@ref) and [`Host`](@ref) and [`@experiment`](@ref).
"""
struct GraphConst{T} end

"""
    StrippedHost{name}

The value [`compile_view`](@ref) substitutes for a [`Host`](@ref) field, carrying the field name as
a **type parameter** so an error message can be built without the sentinel holding any data.

**Using one raises, and the error says what to do about it.** The sentinel used to define no methods
at all and rely on Base's fallbacks, which was loud but useless: you got
`MethodError: no method matching *(::StrippedHost{:sz}, ::Float32)`, which names the field only by
accident of the type parameter and offers no fix. Since `Host` became the **default**, reading an
unmarked field from traced code is the common mistake rather than an exotic one, so the high-traffic
operations now carry real methods and [`_stripped_error`](@ref) writes the message.

Stated honestly, and unchanged by that: the sentinel is still accepted by any `::Any` signature,
still compares with `===` and `==`, still hashes, still survives in a returned `NamedTuple`, and
still returns `false` from `isnothing` without raising. Methods improve the **message** on every
path that was already loud; they do not widen **detection**. A `Host` value that is merely stored,
compared, or passed through still escapes, exactly as before.
"""
struct StrippedHost{name} end

"""
    ReactantNitro._stripped_error(::StrippedHost{name}, op)

The message a user gets when traced code reads an unmarked field. **The framework cannot know which
fix was meant**, so it offers both with their consequences rather than guessing: a value that should
bake into the graph wants [`GraphConst`](@ref) and a recompile per distinct value, and a value that
should cross as a traced input wants [`Device`](@ref) and no recompile at all. Naming only one of
them would be advice half the time.

The third option is in there because it is the commonest real answer: most fields are read host-side
and should stay unmarked.
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

# The high-traffic ways a value gets USED. Each delegates to one message, and each names the
# operation so the report says how the field was reached rather than only that it was.
#
# This list is deliberately not exhaustive and cannot be: `StrippedHost` improves the message on
# paths that already raised, so an operation missing from it falls back to Base and still errors,
# just less helpfully. Add one when a real hook finds a gap.
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

Append [`@experiment`](@ref)'s generated marker table **beneath** any docstring the user wrote,
rather than replacing it.

`Base.@__doc__` on the generated struct is what lets `\"\"\"prose\"\"\" @experiment struct ...` parse at
all; without it the macro's multi-expression block is rejected with "cannot document the following
expression". But the macro then wants to document the same binding itself, and a second `@doc`
**overwrites**, so marking alone would have turned a loud precompilation error into silently
discarded documentation. Hence the merge: the user's prose is intent, the table records which fields
bake into the graph, which cross as traced inputs, and which are invisible to the tracer, and that
second half is the part a user cannot write for themselves.

**This reaches into `Base.Docs`**, which is not a public API, so it is written to degrade rather than
break: any failure to read or install leaves the generated table in place and the docstring merely
unmerged. `test/config.jl` asserts the merged result, so a docsystem change shows up as a failing
test rather than as silence.
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
        # `:module` is not optional: `REPL.parsedoc` reads it to render, so a `DocStr` without it
        # installs cleanly and then throws a `KeyError` at `?MyExp`, which is the worst place to
        # find out. `:path` and `:linenumber` are what the docsystem records for provenance.
        data = Dict{Symbol, Any}(:module => mod, :path => "", :linenumber => 0)
        b = Base.Docs.Binding(mod, name)
        # INSTALLED DIRECTLY RATHER THAN THROUGH `Base.Docs.doc!`, and the reason is the whole point
        # of this function. `doc!` warns "Replacing docs for `X :: Union{}`" whenever the signature
        # already has an entry, because for ordinary code a second docstring for one binding IS
        # usually an accident. Here it is deliberate: `Base.@__doc__` has just put the user's
        # prose there a moment ago and this deliberately appends the table beneath it. Going through
        # `doc!` printed that warning on every precompile of every package with a documented
        # experiment, telling the user their docstring had been thrown away at the exact moment it
        # had been preserved. A warning that states the opposite of what happened is worse than the
        # bug it was guarding against, so this does `doc!`'s bookkeeping without its guess.
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
# All three are exported, because the macro is OPTIONAL: a user must be able to hand-write the
# struct and define these themselves. The fallbacks make every one of them total, so framework
# code may call them on any experiment, hand-written or generated.

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

`kind` is one of `:device`, `:host`, `:graphconst`. `type` is the **declared** type, so a `Device`
field reports the type inside the marker rather than the device type it holds after setup. `doc` is
the field's docstring or `nothing`, and `line` is its declaration line, so a validation error can
point at the source. `default` is `ReactantNitro.NO_DEFAULT` for a field declared without
one.

This is the source of the logged hyperparameter table, and the input any tool that renders an
experiment's configuration reads.

The fallback synthesizes a record for a hand-written experiment from `fieldnames`, `fieldtype`,
[`device_fields`](@ref), and [`host_fields`](@ref), reporting `doc = nothing`, `line = 0`, and
`default = NO_DEFAULT` throughout. It reports `fieldtype`, which for a hand-written `Device` field
is the *device* type after setup rather than the declared one; a hand-written experiment that wants
the declared types in its metadata defines this method itself.
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

The stripped view of an experiment that every trace site sees: each [`Host`](@ref) field is
replaced with a [`StrippedHost`](@ref) sentinel and everything else is passed through unchanged.
Since an unmarked field is Host, this strips every field that is neither [`Device`](@ref) nor
[`GraphConst`](@ref).

The framework calls this at every trace site and passes the result as `Const`. The
real `e` is used everywhere outside the trace: `build_data`, `derive`, metrics finalization,
checkpointing, and every driver decision.

[`@experiment`](@ref) generates a method per experiment type, so the reconstruction is type-stable
and costs one struct copy. The generic fallback here covers a hand-written experiment that defines
[`host_fields`](@ref) itself, and is an ordinary accessor, so an experiment whose layout the default
does not suit overrides it directly.

Note it strips `Host` and **leaves `Device` in place**, which is why the compile cache key is
computed over this view's *GraphConst fields only* rather than over the view.

Leaving `Device` in place is right for every trace the framework itself invokes and wrong for the
one trace it hands away. See [`export_view`](@ref) for the frozen view export uses instead, and for
why a `Device` field that reaches a serialized artifact makes it unservable.
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

`getproperty(e, name)` when `e` has that field, `default` otherwise. This is what lets a Host field
and a user method be interchangeable behind a defaulted accessor:

```julia
max_epochs(e) = _field(e, :max_epochs, 1)
```

These accessors run **host-side against the real `e`**, never against [`compile_view`](@ref)'s
stripped view, which is exactly the split the `Host` marker exists to enforce.
"""
@inline _field(e, name::Symbol, default) =
    hasfield(typeof(e), name) ? getproperty(e, name) : default

# ── @experiment ─────────────────────────────────────────────────────────────────────

"""
    @experiment struct MyExp ... end

One declaration point, generating the struct, a `@kwdef`-style keyword constructor, the
[`device_fields`](@ref) and [`host_fields`](@ref) traits, the [`config_metadata`](@ref) table, a
[`compile_view`](@ref) method, two `Base.show` methods, and a `?MyExp` docstring.

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

The three markers are the three field categories the rest of the framework turns on. A
[`Device`](@ref) field is a traced input. A [`GraphConst`](@ref) field bakes as a trace-time
constant and enters the compile cache key. A [`Host`](@ref) field is invisible to the tracer
entirely, and **it is the default**: an unmarked field is Host, which is the right category for the
majority of real fields (driver knobs, dataset-sized state).

**Why a macro is necessary rather than merely nice.** Each `Device` field's storage varies
independently (scalar to `ConcretePJRTNumber`, array to `ConcretePJRTArray`), so N device fields
need N type parameters; and after setup a field holds a `ConcretePJRTNumber` rather than a
`Device`, so the framework cannot recover which fields were marked by inspecting types at runtime.
The declaration-time knowledge has to be recorded as a trait.

**Generated type parameters.** Every `Device` and `Host` field gets its own free type parameter,
so a `Device` can hold a host value before setup and a device value after, and a `Host` can hold
its real value in `e` and a [`StrippedHost`](@ref) in `compile_view(e)`. **GraphConst fields keep
their declared concrete type**, and the generated keyword constructor `convert`s to it, so
`n_layers::GraphConst{Int}` means what it says whether or not the experiment happens to declare a
`Device` alongside it. Parameterized fields are **not** converted, since the point of the parameter
is that what the field holds changes.

**Docstrings** are written as bare strings above each field. Comments are stripped by the parser
and are never visible to a macro, so `#` cannot carry a description. Per-field docs land in
`config_metadata` and in the generated type docstring.

**Display shows values and never a device buffer.** The generated `show` prints one line per
field with its marker and its value, summarizing an array as its eltype and shape and recursing
through tuples and NamedTuples to do it. That matters because `Device{T}` takes any `T`, and a
read-only buffer for an `hlo_call` lives in a `Device{NamedTuple}` or `Device{Tuple}` of weights:
under Julia's default struct `show`, printing such a config prints the arrays element by element.
The generated output grows with the field count and never with the model.

**The macro is optional.** A user may hand-write the struct and define `device_fields`,
`host_fields`, and `config_metadata` themselves; those three are exported for exactly that reason.
Such a struct keeps Julia's default `show`, and opts in with the one line the macro expands to:

```julia
Base.show(io::IO, e::MyExp) = ReactantNitro._show_experiment(io, e)
Base.show(io::IO, ::MIME"text/plain", e::MyExp) = ReactantNitro._show_experiment(io, e; long = true)
```
"""
macro experiment(expr)
    return _experiment(expr, __source__, __module__)
end

# ── Macro implementation. Split out so it is testable and so the macro body stays one line. ──

"""
    ReactantNitro._qual(name::Symbol) -> Expr

Name one of this module's functions through the module **object**, for a method definition emitted
by [`@experiment`](@ref).

This is not cosmetic. Hygiene treats a short-form definition `f(x) = y` as introducing a binding, so
an unescaped `device_fields(::Type{<:MyExp}) = ...` in macro output is renamed to a gensym and the
method lands on a function nobody can call: the generated traits silently do nothing and every
accessor falls through to the empty default. `esc` is not the fix either, since a name reaching the
user's module through `using` cannot be extended there without an explicit `import`.

Naming the module by its object rather than by the symbol `ReactantNitro` also drops the requirement
that the module be bound in the caller's scope, which `using ReactantNitro: @experiment` would not
provide.
"""
_qual(name::Symbol) = Expr(:., @__MODULE__, QuoteNode(name))

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

# Recognize `Device{T}` / `Host{T}` / `GraphConst{T}`, including a qualified
# `ReactantNitro.Device{T}`. A bare marker with no type parameter is an error rather than a plain
# field of type `Device`, which is what it would otherwise silently become.
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
    # `Base.@__doc__` is what makes `"""docs""" @experiment struct ...` legal at all. Measured: ANY
    # macro returning a multi-expression block fails with "cannot document the following
    # expression", regardless of what the block contains or ends with, so this is a Julia docsystem
    # rule rather than something this macro did wrong, and marking the one documentable expression
    # is the sanctioned answer.
    structdef = Expr(
        :macrocall, GlobalRef(Base, Symbol("@__doc__")),
        source === nothing ? LineNumberNode(0) : source,
        esc(Expr(:struct, ismutable, structsig, Expr(:block, fdecls...)))
    )

    # ── The @kwdef-style constructor. A field with no default is a required keyword, so omitting
    # it raises UndefKeywordError naming it.
    #
    # A GRAPHCONST field is `convert`ed to its declared type here, and that line is load-bearing
    # rather than tidy. Julia's own constructor for a parametric struct annotates a
    # concretely-typed field with that type and does not convert, so `MyExp(; n_layers = 0x03)`
    # would be a MethodError on an experiment that happens to declare a Device and would convert
    # on one that does not. Whether `n_layers::GraphConst{Int}` accepts a `UInt8` should not depend
    # on an unrelated field's marker. Device and Host fields are NOT converted: the whole point of
    # their type parameter is that what they hold changes, from a host value to a device one and,
    # for Host, to a StrippedHost.
    #
    # A FIELDLESS experiment gets NO generated constructor, and that guard is load-bearing rather
    # than an edge case nobody reaches: with no fields the expression above is literally
    # `MyExp(; ) = MyExp()`, which REPLACES Julia's own zero-argument constructor with one that
    # calls itself, so `MyExp()` is a `StackOverflowError` at 80,000 frames naming only `MyExp()`.
    # Julia already supplies exactly the constructor this would emit, so skipping it loses nothing.
    # A fieldless experiment is not exotic: it is an experiment defining only the four required
    # hooks and configuring nothing, which is the case every framework default exists to serve.
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

    # ── Traits. Dispatch is on `Type{<:MyExp}` so both the UnionAll and every instantiation match,
    # including the post-conversion type, which is what makes `config_metadata` report DECLARED
    # types rather than device ones.
    dnames = Expr(:tuple, (QuoteNode(f.name) for f in fields if f.kind === :device)...)
    hnames = Expr(:tuple, (QuoteNode(f.name) for f in fields if f.kind === :host)...)
    df_def = :($(_qual(:device_fields))(::Type{<:$(esc(name))}) = $dnames)
    hf_def = :($(_qual(:host_fields))(::Type{<:$(esc(name))}) = $hnames)

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
    cm_def = :($(_qual(:config_metadata))(::Type{<:$(esc(name))}) = $meta_body)

    # ── compile_view, specialized. Generated per type so the reconstruction is type-stable: the
    # tracer sees this on every trace, and the framework rebuilds `e` every optimizer step.
    hostset = Set(f.name for f in fields if f.kind === :host)
    cv_def = if isempty(hostset)
        :($(_qual(:compile_view))(e::$(esc(name))) = e)
    else
        args = [
            f.name in hostset ? :($(_qual(:StrippedHost)){$(QuoteNode(f.name))}()) :
                :(getfield(e, $(QuoteNode(f.name)))) for f in fields
        ]
        :($(_qual(:compile_view))(e::$(esc(name))) = $(esc(name))($(args...)))
    end

    docstr = _experiment_docstring(name, fields)
    # MERGE rather than overwrite, a deliberate choice. `Base.@__doc__` above attaches the USER's
    # docstring to the struct; this then appends the generated marker table beneath it, and
    # installs the table alone when there is no user docstring, which is the previous
    # behaviour. The table is the half a user cannot write, since it records which fields bake,
    # which are traced inputs, and which are invisible to the tracer; the prose is intent and
    # belongs above it. Neither is silently discarded.
    doc_def = :($(_qual(:_merge_field_table!))($mod, $(QuoteNode(name)), $docstr))

    # ── The two `show` methods ──────────────────────────────────────────────────────────
    #
    # Emitted per type rather than defined once, because a generated experiment has no common
    # supertype to dispatch on: the macro builds a bare struct, and `super` is whatever the user
    # wrote. A hand-written experiment (the macro is optional) keeps Julia's default `show` and can
    # opt in with the same one-liner these expand to.
    #
    # `MIME{Symbol("text/plain")}` rather than the `MIME"text/plain"` string macro: the latter is a
    # macrocall in macro output, and spelling the type directly is one less hygiene question.
    show_def = :(Base.show(io::IO, e::$(esc(name))) = $(_qual(:_show_experiment))(io, e))
    showl_def = :(
        function Base.show(io::IO, ::MIME{Symbol("text/plain")}, e::$(esc(name)))
            return $(_qual(:_show_experiment))(io, e; long = true)
        end
    )

    parts = Any[
        structdef, ctor, df_def, hf_def, cm_def, cv_def, show_def, showl_def, doc_def, esc(name),
    ]
    return Expr(:block, filter(!isnothing, parts)...)
end

# ── Showing an experiment NEVER shows a device buffer ───────────────────────────────
#
# The same hazard as `CheckpointRecord`'s and `Nitro`'s, arriving by a different door. A `Device`
# field is usually a scalar, and a table of scalars is exactly what an experiment should print. But
# `Device{T}` takes any `T`, and the read-only buffer case puts WEIGHTS there: `Device{NamedTuple}`
# or `Device{Tuple}` holding the arrays an `hlo_call` reads. Under Julia's default struct `show`
# those print element by element, so displaying a config dumps a model. Measured on a toy
# experiment carrying one 64x64 and two 16x16 buffers: 51,635 characters.
#
# The fix is the renderer `CheckpointRecord` already uses. `_shown` (Checkpoint.jl) summarizes an
# array as its eltype and shape and RECURSES THROUGH tuples and NamedTuples, so a buffer field
# renders as its shapes and a scalar field still renders as its value. The output's length grows
# with the FIELD COUNT and never with the model, which is the property worth having.
#
# Values, not just names: an experiment is configuration, and a config table that withheld its
# numbers would be useless in the case that is not a buffer, which is nearly all of them.
function _show_experiment(io::IO, e; long::Bool = false)
    T = typeof(e)
    fs = fieldnames(T)
    if !long
        print(io, nameof(T), "(")
        print(io, join(("$f = " * _shown(getfield(e, f)) for f in fs), ", "))
        print(io, ")")
        return nothing
    end
    df, hf = device_fields(T), host_fields(T)
    title = string(nameof(T)) * "  (@experiment; no device buffer is shown)"
    isempty(fs) && return print(io, title, "\n  (no fields)")
    # The marker is the column that makes the table actionable: it is what a reader changes when a
    # field is in the wrong category, and it cannot be inferred from the value.
    rows = Vector{String}[
        [
            string(f),
            f in df ? "Device" : f in hf ? "Host" : "GraphConst",
            _shown(getfield(e, f)),
        ] for f in fs
    ]
    _render_table(io, title, ["field", "marker", "value"], rows)
    return nothing
end

# Each note NAMES ITS MARKER, because the table's job is to be actionable: a reader deciding whether
# a field is in the wrong category needs the word they would type, not only what it does.
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

The table of named configurations for an experiment type. **The contents belong to the model; the
mechanism belongs here.** Default is empty, so a type that declares none loses nothing and "an
experiment defining only the four required hooks trains with nothing passed" is intact.

```julia
ReactantNitro.presets(::Type{MyExp}) = (
    reference_v1 = (; n_classes = 8, batch_size = 16, rotate_deg = 7.5),
    current      = (; n_classes = 10, batch_size = 32, rotate_deg = 12.0),
)
```

**Entries are PARTIAL**, and usually are: a recipe states what it changes and everything else falls
through to the struct's defaults, exactly as the ad-hoc tables this replaces already do.

**Why the framework has an opinion at all**, given three models each solved this privately in about
ten lines: none of those tables is visible to the checkpoint record or the logger, so a **result**
cannot say which recipe produced it. That, plus validating a key rather than letting a typo be a
silent no-op, is what earns the surface. It is **not** a reproducibility mechanism; provenance
already reproduces any run exactly from its git SHA.

Inheritance is `merge` on NamedTuples and needs nothing from the framework:
`wide_v2 = merge(wide_v1, (; max_epochs = 80))`.
"""
function presets end
presets(::Type) = (;)

"""
    from_preset(E::Type, name::Symbol; overrides...) -> e

Build an experiment from [`presets`](@ref)`(E)[name]`, with `overrides` winning over the preset.

```julia
e = from_preset(MyExp, :current)
e = from_preset(MyExp, :current; max_epochs = 40)
```

**Every preset key is validated against `fieldnames(E)`**, and it is the only correctness argument
for presets: today a typo in a splatted `NamedTuple` is a silent no-op, and a recipe that quietly
failed to set the field it names is worse than one that refuses to load.

**A preset is field values, so the marker semantics are untouched.** `GraphConst` fields enter the
compile cache key as usual, which means switching presets recompiles **exactly when the emitted
graph really differs**: two recipes differing only in `Host` fields share both compiled programs,
and one that changes a class count correctly does not. A preset may set a `Device` field too;
`derive` wins for anything it computes, which is already true of a hand-set value.

This does **not** record the name, because it returns a bare experiment and the framework has nowhere
to put it. Use [`Nitro`](@ref)`(E, name; ...)` for that.
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
