# History.jl
#
# `history(nitro)`: the per-epoch series a handle's own `train!` calls produced, as data first (a
# `MetricHistory`, indexed by epoch and by metric name) and as a table second (its `show`, drawn in
# Render.jl). Deciding which rows and columns fit a terminal is arithmetic over the data and is
# done here; drawing the frame is PrettyTables' job.

"""
    ReactantNitro.MetricHistory

The per-epoch metric series a handle keeps. What [`history`](@ref)`(nitro)` returns: one row per
validated epoch, `(; epoch, step, loss,
metrics...)`, with `loss` the epoch's mean train loss and the rest that epoch's finalized
validation metrics as host values. Indexing selects by epoch for rows and by name for columns;
`epoch` and `step` always stay:

```julia
h = history(nitro)
h[7]                         # the row for epoch 7, a NamedTuple
h[end]                       # the last epoch's row
h[10:20]                     # epochs 10 to 20, still a MetricHistory
h[:acc, :macro_recall]       # two metrics, every epoch
h[10:20, :acc]               # both
h[step = 5_000:20_000]       # the epochs whose closing step is in that range
h.acc                        # the column as a Vector, for a plot or a threshold
h.epoch, h.step, h.loss      # the axes and the train loss
```

It is a Tables.jl table with column access, so `DataFrame(h)` and `CSV.write(path, h)` take it
directly; with a Makie backend loaded, `plot(h)` draws it (see [`history_series`](@ref)). Columns
are `epoch`, `step`, `loss`, the checkpointer's selection metric when the handle has one, then the
remaining scalar metrics in first-appearance order; an epoch lacking a metric has `missing`. A
non-scalar metric (a confusion matrix) is not a column of the table view, but `propertynames(h)`
lists it and `h.confusion` returns it per epoch.
"""
struct MetricHistory
    experiment::Symbol
    run_dir::String
    rows::Vector{NamedTuple}
    columns::Vector{Symbol}
    hidden::Vector{Pair{Symbol, String}}     # non-scalar metrics, with a shape note
    best::Union{Nothing, NamedTuple}         # (; metric, mode, epoch), from the checkpointer
    # The epoch the handle's history starts at, kept through every selection, so only a history
    # that itself began past epoch 1 reports earlier epochs in another process.
    first_epoch::Int
end

# The row appended once per validated epoch. `loss` is `NaN` for an epoch that ran no micro-batch.
history_row(nitro, loss_sum, loss_n, metrics_out) = merge(
    (; epoch = nitro.epoch, step = nitro.step, loss = loss_n == 0 ? NaN : loss_sum / loss_n),
    metrics_out,
)

"""
    history(nitro) -> MetricHistory

The per-epoch series this handle's `train!` calls produced, as a [`MetricHistory`](@ref). Fresh per
handle: a resumed handle starts at the epoch it resumed from, and the table says so. Nothing is read
from disk, so it works with any logger backend or none.
"""
function history(nitro::Nitro)
    rows = nitro.history
    ck = nitro.checkpointer
    sel = ck isa TopKCheckpointer ? (ck.metric, ck.mode) : nothing
    scalar, hidden = Symbol[], Pair{Symbol, String}[]
    for r in rows, (k, v) in pairs(r)
        k in (:epoch, :step, :loss) && continue
        if v isa Real
            k in scalar || push!(scalar, k)
        elseif !any(h -> first(h) == k, hidden)
            push!(hidden, k => _shape_note(v))
        end
    end
    columns = Symbol[:epoch, :step, :loss]
    if sel !== nothing && sel[1] in scalar
        push!(columns, sel[1])
        filter!(!=(sel[1]), scalar)
    end
    append!(columns, scalar)
    best = nothing
    if sel !== nothing && sel[1] in columns
        scored = [(r[sel[1]], r.epoch) for r in rows if haskey(r, sel[1]) && isfinite(r[sel[1]])]
        if !isempty(scored)
            pick = sel[2] === :max ? argmax : argmin
            best = (; metric = sel[1], mode = sel[2], epoch = scored[pick(first.(scored))][2])
        end
    end
    return MetricHistory(
        nameof(typeof(nitro.e)), nitro.run_dir, collect(NamedTuple, rows), columns, hidden, best,
        isempty(rows) ? 1 : first(rows).epoch
    )
end

_shape_note(v::AbstractArray) = join(size(v), "x")
_shape_note(v) = string(nameof(typeof(v)))

# ── The data interface ─────────────────────────────────────────────────────────────

# `getfield` throughout the implementation: `getproperty` below serves COLUMNS, so a metric named
# `rows` or `best` would shadow the struct's own field for a caller and must not for us.
_rows(h::MetricHistory) = getfield(h, :rows)
_columns(h::MetricHistory) = getfield(h, :columns)
_hidden(h::MetricHistory) = getfield(h, :hidden)
_best(h::MetricHistory) = getfield(h, :best)
_epochs(h::MetricHistory) = Int[r.epoch for r in _rows(h)]

Base.length(h::MetricHistory) = length(_rows(h))
Base.isempty(h::MetricHistory) = isempty(_rows(h))
Base.iterate(h::MetricHistory, i = 1) = iterate(_rows(h), i)
Base.eltype(::Type{MetricHistory}) = NamedTuple
Base.firstindex(h::MetricHistory) = isempty(h) ? 1 : first(_rows(h)).epoch
Base.lastindex(h::MetricHistory) = isempty(h) ? 0 : last(_rows(h)).epoch

Base.propertynames(h::MetricHistory) = Tuple(vcat(_columns(h), first.(_hidden(h))))

function Base.getproperty(h::MetricHistory, name::Symbol)
    if name in _columns(h) || any(p -> first(p) == name, _hidden(h))
        return _column(h, name)
    end
    return getfield(h, name)
end

# The narrowest element type the values allow, `Union{Missing, T}` when an epoch lacks the metric.
function _column(h::MetricHistory, name::Symbol)
    col = Any[get(r, name, missing) for r in _rows(h)]
    present = Any[v for v in col if !ismissing(v)]
    isempty(present) && return Vector{Missing}(missing, length(col))
    T = mapreduce(typeof, typejoin, present)
    return length(present) == length(col) ? Vector{T}(col) : Vector{Union{Missing, T}}(col)
end

# ── Tables.jl ──────────────────────────────────────────────────────────────────────
#
# Column access only: Pluto replaces the `text/html` show of a table reporting row access with its
# own grid.
Tables.istable(::Type{MetricHistory}) = true
Tables.columnaccess(::Type{MetricHistory}) = true
Tables.columns(h::MetricHistory) =
    NamedTuple{Tuple(_columns(h))}(Tuple(_column(h, c) for c in _columns(h)))
Tables.schema(h::MetricHistory) =
    Tables.Schema(_columns(h), [eltype(_column(h, c)) for c in _columns(h)])

function _row_at(h::MetricHistory, epoch::Integer)
    i = findfirst(r -> r.epoch == epoch, _rows(h))
    i === nothing && error(
        "ReactantNitro: this history has no epoch $epoch. " *
            (
            isempty(h) ? "It is empty: the handle has not validated an epoch." :
                "Its epochs run from $(firstindex(h)) to $(lastindex(h))."
        )
    )
    return _rows(h)[i]
end

# A row narrowed to the selected columns, so `h[:acc][7]` shows what `h[:acc]` shows.
function Base.getindex(h::MetricHistory, epoch::Integer)
    r = _row_at(h, epoch)
    keep = Tuple(k for k in keys(r) if k in _columns(h) || any(p -> first(p) == k, _hidden(h)))
    return NamedTuple{keep}(r)
end

function Base.getindex(h::MetricHistory, epochs::AbstractVector{<:Integer})
    rows = NamedTuple[r for r in _rows(h) if r.epoch in epochs]
    return MetricHistory(
        getfield(h, :experiment), getfield(h, :run_dir), rows, copy(_columns(h)),
        copy(_hidden(h)), _best(h), getfield(h, :first_epoch),
    )
end

function Base.getindex(h::MetricHistory, names::Symbol...; step = nothing)
    available = vcat(_columns(h), first.(_hidden(h)))
    for n in names
        n in available || error(
            "ReactantNitro: `$n` is not a metric in this history. It has: " *
                join(("`$c`" for c in available if !(c in (:epoch, :step))), ", ") * "."
        )
    end
    # No names selects every column, so `h[step = a:b]` is a row selection alone.
    if isempty(names)
        columns, hidden = copy(_columns(h)), copy(_hidden(h))
    else
        columns = Symbol[c for c in _columns(h) if c in (:epoch, :step) || c in names]
        hidden = Pair{Symbol, String}[p for p in _hidden(h) if first(p) in names]
    end
    rows = step === nothing ? copy(_rows(h)) : NamedTuple[r for r in _rows(h) if r.step in step]
    return MetricHistory(
        getfield(h, :experiment), getfield(h, :run_dir), rows, columns, hidden,
        _best(h), getfield(h, :first_epoch),
    )
end

Base.getindex(h::MetricHistory, epochs::AbstractVector{<:Integer}, names::Symbol...) =
    h[epochs][names...]
Base.getindex(h::MetricHistory, ::Colon, names::Symbol...) = h[names...]

# The compact form, and the whole display when no renderer is loaded: the three-argument
# `text/plain` method lives in Render.jl.
function Base.show(io::IO, h::MetricHistory)
    n = length(h)
    print(io, "history of ", getfield(h, :experiment), ": ", n, n == 1 ? " epoch" : " epochs")
    n == 0 || print(io, " (", firstindex(h), " to ", lastindex(h), ")")
    m = count(c -> !(c in (:epoch, :step, :loss)), _columns(h))
    print(io, ", ", m, m == 1 ? " metric" : " metrics", ", ", getfield(h, :run_dir))
    return nothing
end

# ── Fitting the table to a terminal ────────────────────────────────────────────────

"""
    ReactantNitro.thin_rows(n, best, budget) -> Vector{Int}

Which of `n` rows to show in `budget` lines: windows around the first, the `best` (or `nothing`)
and the last row, grown one row at a time round-robin until the budget is spent, with `0` marking
a gap. The best window grows in both directions, so it gets twice the context of the ends.
"""
function thin_rows(n::Int, best::Union{Nothing, Int}, budget::Int)
    n <= budget && return collect(1:n)
    keep = Set{Int}((1, n))
    best === nothing || push!(keep, best)
    cost() = length(keep) + _gaps(keep)
    fr, lr = 1, n                                   # the first and last windows' moving edges
    bl = br = something(best, 0)                    # the best window's two edges
    while cost() < budget && length(keep) < n
        for cursor in (:fr, :br, :bl, :lr)
            best === nothing && cursor in (:br, :bl) && continue
            x = cursor === :fr ? (fr += 1) : cursor === :br ? (br += 1) :
                cursor === :bl ? (bl -= 1) : (lr -= 1)
            1 <= x <= n || continue
            push!(keep, x)
            cost() < budget || break
        end
    end
    out = Int[]
    for i in sort!(collect(keep))
        isempty(out) || out[end] == 0 || out[end] == i - 1 || push!(out, 0)
        push!(out, i)
    end
    return out
end

function _gaps(keep::Set{Int})
    s = sort!(collect(keep))
    return count(i -> s[i + 1] > s[i] + 1, 1:(length(s) - 1))
end

# One cell, given the column's decimal count. A non-finite value prints as such, since a `NaN` in
# the table is a finding; a float prints with the column's decimals so the points line up.
_history_cell(::Missing, decimals) = ""
_history_cell(x::Bool, decimals) = string(x)
_history_cell(x::Integer, decimals) = string(x)
_history_cell(x::Real, decimals) = string(x)
function _history_cell(x::AbstractFloat, decimals)
    isfinite(x) || return string(x)
    decimals === nothing && return Printf.@sprintf("%.4g", x)
    return Printf.format(Printf.Format("%." * string(decimals) * "f"), x)
end

# The decimals a column prints with: enough that its smallest shown magnitude keeps four
# significant digits, capped at six; a column past that or past a million prints in `%g` form.
function _column_decimals(rows, picks, c)
    smallest, largest = Inf, 0.0
    for p in picks
        p == 0 && continue
        v = get(rows[p], c, missing)
        v isa AbstractFloat && isfinite(v) || continue
        v == 0 && continue
        smallest = min(smallest, abs(v))
        largest = max(largest, abs(v))
    end
    isfinite(smallest) || return 0
    (smallest < 1.0e-4 || largest >= 1.0e6) && return nothing
    # Four significant digits from the leading digit of the smallest magnitude.
    lead = floor(Int, log10(smallest))
    return clamp(3 - lead, 0, 6)
end

"""
    ReactantNitro.history_table(h, height, width) -> NamedTuple

Everything a renderer needs to draw `h` in a `height` x `width` terminal: `title`, `labels`,
`cells::Matrix{String}`, `alignment`, `best_row`, `gap_rows` and `note`. Rows are thinned with
[`thin_rows`](@ref); columns past the fixed four are dropped from the right until the widths fit,
and the note says what was left out and how to select it.
"""
function history_table(h::MetricHistory, height::Integer, width::Integer)
    rows, cols = _rows(h), _columns(h)
    best = _best(h)
    n = length(rows)
    ibest = best === nothing ? nothing : findfirst(r -> r.epoch == best.epoch, rows)
    # Nine lines of frame before a row of data; never fewer than five data lines.
    budget = max(5, Int(height) - 9)
    picks = thin_rows(n, ibest, budget)
    shown = count(!=(0), picks)

    # Width: the fixed columns always stay; the rest go from the right until the frame fits.
    fixed = min(length(cols), best === nothing ? 3 : 4)
    keep = collect(cols)
    dropped = Symbol[]
    while length(keep) > fixed && _table_width(rows, picks, keep) > width
        pushfirst!(dropped, pop!(keep))
    end

    labels = vcat(string.(keep), [" "])
    decimals = [_column_decimals(rows, picks, c) for c in keep]
    cells = Matrix{String}(undef, length(picks), length(labels))
    gap_rows = Int[]
    best_row = nothing
    for (i, p) in enumerate(picks)
        if p == 0
            cells[i, :] .= ""
            cells[i, 1] = "⋮"
            push!(gap_rows, i)
            continue
        end
        r = rows[p]
        for (j, c) in enumerate(keep)
            cells[i, j] = _history_cell(get(r, c, missing), decimals[j])
        end
        cells[i, end] = p == ibest ? "*" : ""
        p == ibest && (best_row = i)
    end
    alignment = vcat(fill(:r, length(keep)), [:l])

    title = "history of " * string(getfield(h, :experiment)) * "  (" * getfield(h, :run_dir) *
        (shown < n ? ", $shown of $n epochs" : "") * ")"
    notes = String[]
    best === nothing || push!(
        notes, "* best $(best.metric) ($(best.mode)), the checkpointer's metric"
    )
    if shown < n
        # A window of `budget` rows around the best epoch; the whole range would thin again.
        lo = clamp(something(ibest, n) - budget ÷ 2, 1, n - budget + 1)
        hi = lo + budget - 1
        push!(
            notes,
            "$(n - shown) of $n epochs thinned to fit; select fewer, " *
                "`h[$(rows[lo].epoch):$(rows[hi].epoch)]` say, for every row",
        )
    end
    isempty(dropped) || push!(
        notes,
        "not shown for width: " * join(string.(dropped), ", ") * "; select them with `h[" *
            join((":" * string(d) for d in dropped), ", ") * "]`",
    )
    isempty(_hidden(h)) || push!(
        notes, "not tabulated: " * join(("$k ($s)" for (k, s) in _hidden(h)), ", ")
    )
    fe = getfield(h, :first_epoch)
    fe > 1 && push!(notes, "epochs before $fe were trained in another process")
    # One note per line.
    return (; title, labels, cells, alignment, best_row, gap_rows, note = join(notes, "\n  "))
end

# An estimate of PrettyTables' layout, close enough to decide a drop.
function _table_width(rows, picks, keep)
    w = 0
    for c in keep
        cw = textwidth(string(c))
        d = _column_decimals(rows, picks, c)
        for p in picks
            p == 0 && continue
            cw = max(cw, textwidth(_history_cell(get(rows[p], c, missing), d)))
        end
        w += cw + 2
    end
    return w + 2 + 3          # the outer rules, and the marker column
end

# ── The plot ───────────────────────────────────────────────────────────────────────
#
# Same split as the table: what is drawn is decided here, the Makie extension draws it.

"""
    ReactantNitro.history_series(h; x = :epoch, metrics = nothing) -> NamedTuple

Everything a plot of `h` draws, computed without a plotting package: `title`, `xlabel`, and
`panels`, one per curve, each `(; name, x, y, best, subtitle)`. `x` is `:epoch` or `:step`.
`metrics` is `nothing` (the checkpointer's metric when the history has it, else every metric
column, else `loss`), `:all`, one name, or a collection. `best` marks the checkpointer's chosen
epoch on that metric's panel. A row lacking a metric is skipped; a non-finite value is kept, so the
line breaks where the run did. The Makie extension turns this into `plot(h)`.
"""
function history_series(h::MetricHistory; x::Symbol = :epoch, metrics = nothing)
    x in (:epoch, :step) || error("ReactantNitro: `x` must be `:epoch` or `:step`, got `$x`.")
    isempty(h) && error(
        "ReactantNitro: this history is empty, the handle has not validated an epoch; " *
            "there is nothing to plot."
    )
    plottable = Symbol[c for c in _columns(h) if !(c in (:epoch, :step))]
    best = _best(h)
    names = if metrics === nothing
        if best !== nothing && best.metric in plottable
            [best.metric]
        else
            rest = filter(!=(:loss), plottable)
            isempty(rest) ? [:loss] : rest
        end
    elseif metrics === :all
        plottable
    elseif metrics isa Symbol
        [metrics]
    else
        collect(Symbol, metrics)
    end
    for nm in names
        nm in plottable && continue
        i = findfirst(p -> first(p) == nm, _hidden(h))
        i === nothing && error(
            "ReactantNitro: `$nm` is not a metric in this history. It has: " *
                join(("`$c`" for c in plottable), ", ") * "."
        )
        error(
            "ReactantNitro: `$nm` is not a scalar metric ($(last(_hidden(h)[i]))); " *
                "it has no curve to draw."
        )
    end
    rows = _rows(h)
    panels = map(names) do nm
        xs, ys = Int[], Float64[]
        for r in rows
            v = get(r, nm, missing)
            v === missing && continue
            push!(xs, r[x])
            push!(ys, Float64(v))
        end
        mark, subtitle = nothing, string(nm)
        if best !== nothing && nm == best.metric
            i = findfirst(r -> r.epoch == best.epoch, rows)
            if i !== nothing && haskey(rows[i], nm)
                mark = (rows[i][x], Float64(rows[i][nm]))
                subtitle = "$nm ($(best.mode)), best at epoch $(best.epoch)"
            end
        end
        return (; name = nm, x = xs, y = ys, best = mark, subtitle)
    end
    return (; title = "history of " * string(getfield(h, :experiment)), xlabel = string(x), panels)
end
