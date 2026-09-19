# History.jl
#
# `history(nitro)`: the per-epoch series a handle's own `train!` calls produced, as data first and
# as a table second. The data is a `MetricHistory`, which indexes by epoch, selects metrics by
# name, and hands a column back as a vector; the table is its `show`, and lives in the
# PrettyTables extension because that is the one renderer this package has.
#
# ── Why a handle keeps a series at all ─────────────────────────────────────────────
#
# `last_metrics` answers "how did the last epoch do". The question a person asks after a run is
# "how did it GO", which is the shape of the curve, and before this the only place that shape
# existed was a logger backend: a file to open, or a hosted tracker to log into. A REPL that has
# the handle in hand should be able to ask the handle. What it keeps is small, one `NamedTuple` of
# host scalars per validated epoch, and it is fresh per handle: a resumed run's earlier epochs
# belong to the process that trained them and are not reconstructed here.
#
# ── Why the display logic is here and the rendering is not ─────────────────────────
#
# Deciding WHICH rows and columns fit a terminal is arithmetic over the data, and it is the part
# worth testing without a frame around it. Drawing the frame is PrettyTables' job. So
# `history_table` produces the cells, the labels, the alignment and the footer note for a given
# display size, and the extension's `show` does nothing but hand those to `pretty_table`. There
# is deliberately no plain renderer beside it; see `_render_sections`.

"""
    ReactantNitro.MetricHistory

What [`history`](@ref)`(nitro)` returns: one row per validated epoch, `(; epoch, step, loss,
metrics...)`, with `loss` the epoch's mean train loss and the rest that epoch's finalized
validation metrics as host values. Its display is the table; the object is the data.

Indexing selects, by **epoch** for rows and by **name** for columns; `epoch` and `step` always
stay:

```julia
h = history(nitro)
h[7]                         # the row for epoch 7, a NamedTuple
h[end]                       # the last epoch's row
h[10:20]                     # epochs 10 to 20, still a MetricHistory
h[:acc, :macro_recall]       # two metrics, every epoch
h[10:20, :acc]               # both
h[step = 5_000:20_000]       # the epochs whose closing step is in that range
h[:acc, step = 5_000:20_000] # both
h.acc                        # the column as a Vector, for a plot or a threshold
h.epoch, h.step, h.loss      # the axes and the train loss
```

A row's `step` is the global step at the END of its epoch, so `step = a:b` selects the epochs that
closed within it. With a Makie backend loaded, `plot(h)` draws it; see [`history_series`](@ref)
for what the plot shows by default.

It is a Tables.jl table with column access, so `DataFrame(h)`, `CSV.write(path, h)` and anything
else that consumes one take it directly. The table view has the scalar columns in the order above;
a column with a gap, an epoch that did not report the metric, has `missing` there and a
`Union{Missing, T}` element type. A non-scalar metric is not a column of the table view. In a
notebook the display is the same table as HTML, every row, since a notebook scrolls.

The column order is fixed: `epoch`, `step`, `loss`, the checkpointer's selection metric when the
handle has one, then the remaining scalar metrics in the order they first appeared. A metric that
is not a scalar, a confusion matrix say, is not tabulated; `propertynames(h)` still lists it and
`h.confusion` returns it per epoch.
"""
struct MetricHistory
    experiment::Symbol
    run_dir::String
    rows::Vector{NamedTuple}
    columns::Vector{Symbol}
    hidden::Vector{Pair{Symbol, String}}     # non-scalar metrics, with a shape note
    best::Union{Nothing, NamedTuple}         # (; metric, mode, epoch), from the checkpointer
    # The epoch the HANDLE's history starts at, kept through every selection so that `h[18:28]`
    # is not mistaken for a resumed run: only a history that itself began past epoch 1 has
    # earlier epochs in another process.
    first_epoch::Int
end

# The row appended once per validated epoch, from both training loops. `loss` is `NaN` for an epoch
# that ran no micro-batch, which cannot happen in a completed epoch and is the honest value if it
# ever does.
history_row(nitro, loss_sum, loss_n, metrics_out) = merge(
    (; epoch = nitro.epoch, step = nitro.step, loss = loss_n == 0 ? NaN : loss_sum / loss_n),
    metrics_out,
)

"""
    history(nitro) -> MetricHistory

The per-epoch series this handle's `train!` calls produced: `(; epoch, step, loss, metrics...)`
per validated epoch, as a [`MetricHistory`](@ref) that indexes by epoch and by metric name and
displays as a table sized to the terminal.

Fresh per handle. A handle built with `resume = :auto` starts its history at the epoch it resumed
from, and the table says so; the earlier epochs are in the logger's record. Nothing is read from
disk here, which is what lets it work with any logger backend, or none.
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
# Column access only: Pluto replaces the `text/html` show of any table that reports row access
# with its own grid of raw values. Tables.jl derives rows from columns for everything else.
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
# `text/plain` method lives in the PrettyTables extension and falls back to this without it.
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

Which of `n` rows to show when only `budget` lines are available: windows around the first, the
`best` (or `nothing`) and the last row, each grown one row at a time in turn until the budget is
spent, with `0` marking a gap between windows. A gap costs one line, so the result's length is at
most `budget`, and every row is shown when `n <= budget`.

The windows grow round-robin, the best one in both directions, so the best epoch gets roughly twice
the context of the ends. Windows that meet merge, which frees the gap's line for another row.
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

# One cell, given the column's decimal count. An integer is itself; a non-finite value prints as
# such, because a `NaN` in the table is a finding and a blank would hide it; a float prints with
# the column's decimals so the points line up down the column, or in `%g` form for a column that
# `_column_decimals` decided has no sensible fixed count.
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
# significant digits, so `0.7` and `0.02531` in one column print as `0.70000` and `0.02531` and
# the points align. Capped at six; a column with a value past that, or past a million, has no
# fixed count that reads and prints in `%g` form instead (`nothing`). A column with no finite float
# is left alone.
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

Everything a renderer needs to draw `h` in a `height` x `width` terminal, computed without one:
`title`, `labels`, `cells::Matrix{String}`, `alignment`, `best_row` (or `nothing`), `gap_rows`, and
`note`. Rows are thinned with [`thin_rows`](@ref) to the lines left after the frame; columns past
the fixed four are dropped from the right until the widths fit, and the note names what was left
out and how to select it. Each float column prints with one decimal count, chosen so its smallest
shown value keeps four significant digits, so the points align down the column.
"""
function history_table(h::MetricHistory, height::Integer, width::Integer)
    rows, cols = _rows(h), _columns(h)
    best = _best(h)
    n = length(rows)
    ibest = best === nothing ? nothing : findfirst(r -> r.epoch == best.epoch, rows)
    # Title, column labels, header rule, top and bottom rules, the note, and a line of air: nine
    # lines the table spends before a row of data. Never fewer than five data lines, which is the
    # three anchors and two gaps.
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
    # One note per line: a footer that wraps mid-sentence under a table is harder to read than
    # a frame that is one line taller.
    return (; title, labels, cells, alignment, best_row, gap_rows, note = join(notes, "\n  "))
end

# The frame's width for these columns: every cell padded to its column, two blanks of padding per
# column, and the outer rules. An estimate of PrettyTables' layout rather than a call into it,
# close enough to decide a drop and cheap enough to run per column.
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
`panels`, one per curve, each `(; name, x, y, best, subtitle)`.

`x` is `:epoch` or `:step`, the axis. `metrics` picks the curves:

- `nothing`, the default: the checkpointer's metric when the history has it as a column, which is
  the one curve the run was selecting on; otherwise every metric column; otherwise `loss`
- `:all`: every column, `loss` included
- one name, or a collection of names, in the history's column order

`best` is the `(x, y)` of the checkpointer's chosen epoch on that metric's panel when the selection
still holds that epoch, and `nothing` on every other panel. A row that lacks a metric is skipped; a
non-finite value is kept, so the drawn line breaks where the run did. The Makie extension turns
this into `plot(h)`, so `plot(h[10:20, :acc])` and `plot(h; x = :step, metrics = :all)` are both
how to ask for a different figure.
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
