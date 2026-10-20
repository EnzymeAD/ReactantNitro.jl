# SourceSyntax.jl
#
# The string form of a checkpoint source, for drivers that pass strings (the Kaimon gate tools). It
# is Julia syntax, read with `Meta.parse` and walked by a whitelist, never evaluated. The value it
# returns is exactly what the `weights` and `resume` keywords take, and `check_source` validates it,
# so the string form cannot mean anything the keyword does not.

# Every constructor the string form may call. A new selector is one entry here; the round-trip test
# in `source_syntax_tests.jl` fails for a selector whose `show` does not parse back to it.
const SOURCE_SYNTAX = (; Best, Last, Epochs, Every, Average)

"""
    ReactantNitro.parse_source(s; kw = :weights) -> source

The `weights` or `resume` value a string names, validated by [`check_source`](@ref):

- no `=>`: a path (`"runs/x/epoch-0003.jld2"`); for `resume`, also `"auto"` and `"false"`
- `"run => checkpoint"`: the run is `latest`, `all` (with or without `:`) or a run id; the
  checkpoint is `best` or `latest` (with or without `:`), or a selector call in Julia syntax,
  `"latest => Average(Best(5; among = Last(20)))"`

Calls are limited to [`SOURCE_SYNTAX`](@ref ReactantNitro.SOURCE_SYNTAX), arguments to integer,
range, vector and symbol literals. Private while the grammar is iterated on.
"""
function parse_source(s::AbstractString; kw::Symbol = :weights)
    t = strip(s)
    if kw === :resume
        t == "auto" && return :auto
        t == "false" && return false
    end
    bare = lstrip(t, ':')
    bare in ("best", "latest") && error(
        "ReactantNitro: `$(kw) = \"$(t)\"` names a checkpoint and not the run it belongs to. Pass \
         `$(kw) = \"latest => $(bare)\"` for the most recent run in `run_dir`, or \
         `\"<run id> => $(bare)\"` for another (`runs(run_dir)` lists them)\
         $(bare == "best" ? ", or `\"all => best\"` for the best across every run" : "")."
    )
    occursin("=>", t) || return String(t)
    left, right = strip.(split(t, "=>"; limit = 2))
    (isempty(left) || isempty(right)) && error(
        "ReactantNitro: `$(kw) = \"$(t)\"` needs a run on the left of `=>` (`latest`, `all` or a run \
         id) and a checkpoint on the right (`best`, `latest`, or a selector such as \
         `Average(Last(10))`)."
    )
    run = _source_run(left)
    which = _source_checkpoint(right, t, kw)
    src = run => which
    check_source(src, kw)
    return src
end

function _source_run(left)
    l = lstrip(left, ':')
    l == "latest" && return :latest
    l == "all" && return :all
    # A run id; quotes are optional, since ids are written bare in `runs()` listings.
    return String(strip(left, '"'))
end

function _source_checkpoint(right, t, kw)
    r = lstrip(right, ':')
    r in ("best", "latest") && return Symbol(r)
    ex = _parse_julia(right, "`$(kw) = \"$(t)\"`")
    ex isa Symbol && error(
        "ReactantNitro: `$(kw) = \"$(t)\"` names no checkpoint of the run: `$(right)` is not \
         `best` or `latest`, and not a selector call such as `Average(Last(10))`."
    )
    return syntax_value(ex, "`$(kw) = \"$(t)\"`")
end

"""
    ReactantNitro.parse_restore_best(x) -> Bool or Average

The `restore_best` value a driver passes: a `Bool`, `"true"`, `"false"`, or an `Average` call in
the string form of [`parse_source`](@ref ReactantNitro.parse_source).
"""
parse_restore_best(x::Bool) = x
function parse_restore_best(s::AbstractString)
    t = strip(s)
    t in ("true", "false") && return t == "true"
    what = "`restore_best = \"$(t)\"`"
    v = syntax_value(_parse_julia(t, what), what)
    v isa Average || error(
        "ReactantNitro: $(what) is `true`, `false`, or an `Average` such as \
         `\"Average(Last(10))\"`; got a `$(typeof(v))`."
    )
    return v
end

"""
    ReactantNitro.syntax_value(ex, what) -> value

The value of one parsed expression under the whitelist: a call to a constructor in
`SOURCE_SYNTAX`, with positional and keyword arguments, or an integer, `a:b` / `a:s:b` range,
`[...]` vector or `:symbol` literal. Anything else is refused, naming what is allowed.
"""
function syntax_value(ex, what)
    ex isa Integer && return Int(ex)
    ex isa QuoteNode && ex.value isa Symbol && return ex.value
    if ex isa Expr
        ex.head === :vect && return Int[_int(a, what) for a in ex.args]
        if ex.head === :call && ex.args[1] === :(:)
            ints = map(a -> _int(a, what), ex.args[2:end])
            length(ints) in (2, 3) && return length(ints) == 2 ? (ints[1]:ints[2]) :
                (ints[1]:ints[2]:ints[3])
        end
        if ex.head === :call && ex.args[1] isa Symbol && haskey(SOURCE_SYNTAX, ex.args[1])
            pos, kws = Any[], Pair{Symbol, Any}[]
            for a in ex.args[2:end]
                if a isa Expr && a.head === :parameters
                    append!(kws, (_kwarg(p, what) for p in a.args))
                elseif a isa Expr && a.head === :kw
                    push!(kws, _kwarg(a, what))
                else
                    push!(pos, syntax_value(a, what))
                end
            end
            f = getfield(SOURCE_SYNTAX, ex.args[1])
            return try
                f(pos...; kws...)
            catch err
                err isa MethodError || rethrow()
                error(
                    "ReactantNitro: $(what): `$(ex)` does not match any form of \
                     `$(ex.args[1])`; see its docstring."
                )
            end
        end
    end
    return error(
        "ReactantNitro: $(what): `$(ex)` is not allowed in a source string. It may call \
         $(join(("`$(k)`" for k in keys(SOURCE_SYNTAX)), ", ")), with integer, range \
         (`3:5`), vector (`[3, 5]`) and symbol (`:latest`) arguments."
    )
end

# `Meta.parse` returns an `:incomplete` expression for a truncated string rather than throwing.
function _parse_julia(s, what)
    ex = try
        Meta.parse(s; raise = true)
    catch err
        error("ReactantNitro: $(what): `$(s)` does not parse: $(sprint(showerror, err)).")
    end
    ex isa Expr && ex.head in (:incomplete, :error) &&
        error("ReactantNitro: $(what): `$(s)` does not parse; it is incomplete or malformed.")
    return ex
end

function _int(a, what)
    v = syntax_value(a, what)
    v isa Int || error("ReactantNitro: $(what): `$(a)` must be an integer.")
    return v
end

function _kwarg(p, what)
    p isa Expr && p.head === :kw && p.args[1] isa Symbol ||
        error("ReactantNitro: $(what): `$(p)` is not a keyword argument (`name = value`).")
    return p.args[1] => syntax_value(p.args[2], what)
end
