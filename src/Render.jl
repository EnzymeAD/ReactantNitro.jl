# Render.jl
#
# The framed renderer every long `show` goes through, as text and as HTML, and the history table's
# two shows. PrettyTables
# is an ordinary dependency: Reactant loads it into every process anyway, and a renderer installed
# at definition needs no `__init__`, which matters for a tool that evaluates this package from
# source and skips initialisers. Sections become `row_group_labels` bands of ONE frame, drawn
# without vertical rules so a short band does not read as a row of empty cells.

# ── The palette: six roles, and the eight-colour ANSI set ────────────────────────────
#
# THE BASIC EIGHT, not 256-colour and not truecolor, and that is the whole reason this reads well
# on somebody else's terminal. The basic eight are the ones a theme remaps, so `:green` is
# whatever green that person chose and is legible against whatever background they chose with it.
# A hex colour picked here would look considered on the machine it was picked on and would be the
# one unreadable cell on a light theme.
#
# `:dark_gray` for `:muted` is the one entry to be careful with. It is Crayons' name for bright
# black, which is the theme's grey and is dim on both polarities; `faint` would be the obvious
# alternative and is a terminal attribute that several emulators drop entirely and one or two
# render as invisible.
#
# Nothing here sets a background. A background colour survives a copy-paste into a ticket as a
# block of highlight, and these tables get pasted.
#
# ── TWO GATES DECIDE WHETHER A CRAYON IS EMITTED, and they read different things ──────
#
# PrettyTables decides whether to STYLE from the `IOContext` it is handed. Crayons decides whether
# to emit the ESCAPE from the buffer it is printing into, which is one PrettyTables made and which
# does not carry that context, so it falls back to the process-global `Base.get_have_color()`.
#
# In a process with colour on, which is every REPL, both are satisfied and none of this is
# visible. In one with colour off, a caller that wraps the `io` in `IOContext(:color => true)`
# satisfies the first gate and not the second, and gets a table carrying `\e[0m` resets with no
# colour before them. `Crayons.force_color(true)` is what satisfies the second, and a test
# asserting on a crayon has to set both.
const _ROLE_CRAYONS = Dict{Symbol, PrettyTables.Crayon}(
    :good => PrettyTables.Crayon(foreground = :green),
    :busy => PrettyTables.Crayon(foreground = :cyan),
    :warn => PrettyTables.Crayon(foreground = :yellow),
    :bad => PrettyTables.Crayon(foreground = :red, bold = true),
    :muted => PrettyTables.Crayon(foreground = :dark_gray),
    :accent => PrettyTables.Crayon(foreground = :magenta),
)

# The sections flattened into one grid: the cells, the band labels by row, the header rows, and
# the roles keyed by the table's own row numbers rather than each section's.
function _table_layout(sections::Vector{TableSection})
    rows = TableRows()
    labels = Pair{Int, String}[]
    headers = Set{Int}()
    roles = Dict{Tuple{Int, Int}, Symbol}()
    for sec in sections
        # An empty section title draws no band label: the leading section is named by the
        # table's title. A label sits at the row it precedes, so it is placed before the rows.
        isempty(sec.title) || push!(labels, (length(rows) + 1) => sec.title)
        if any(!isempty, sec.header)
            push!(rows, sec.header)
            push!(headers, length(rows))    # an ordinary row to the table; bolded below
        end
        offset = length(rows)
        append!(rows, sec.rows)
        for ((r, c), role) in sec.styles
            roles[(offset + r, c)] = role
        end
    end
    isempty(rows) && return nothing
    cols = maximum(length, rows)
    data = [get(rows[r], c, "") for r in 1:length(rows), c in 1:cols]
    return (; data, labels, headers, roles)
end

# The renderer contract, as `table_renderer!` documents it, once per MIME type. Typed in full: it
# is reached through a `Ref{Any}`, so nothing else checks the signature.
function render_table(
        io::IO, ::MIME"text/plain", title::AbstractString, sections::Vector{TableSection},
        note::Union{AbstractString, Nothing}
    )
    t = _table_layout(sections)
    t === nothing && return print(io, title)
    # Rendered into a buffer for one reason: `pretty_table` ends its output with a newline and a
    # `show` method must not. `IOContext(buf, io)` carries `:color` across.
    buf = IOBuffer()
    PrettyTables.pretty_table(
        IOContext(buf, io), t.data;
        alignment = :l, title, title_alignment = :l,
        # Every section carries its own header as a data row; one label row cannot serve them.
        show_column_labels = false,
        row_group_labels = isempty(t.labels) ? nothing : t.labels,
        # ORDER IS THE PRECEDENCE: the first matching highlighter wins, so a header stays bold
        # where a role was set on the same cell. An unknown role leaves the cell alone.
        highlighters = [
            PrettyTables.TextHighlighter((_, i, _) -> i in t.headers; bold = true),
            PrettyTables.TextHighlighter(
                (_, i, j) -> haskey(_ROLE_CRAYONS, get(t.roles, (i, j), :none)),
                (_, _, i, j) -> _ROLE_CRAYONS[t.roles[(i, j)]]
            ),
        ],
        table_format = PrettyTables.TextTableFormat(;
            vertical_lines_at_data_columns = :none,
            horizontal_line_after_column_labels = false,
        ),
        # NO CROPPING: what would be dropped is the right-hand column, the sources and paths,
        # which are the reason someone printed the handle.
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
    )
    print(io, rstrip(String(take!(buf)), '\n'))
    note === nothing || print(io, "\n  ", note)
    return nothing
end

# The same roles as CSS. Named colours, so a notebook theme's text colour still shows through
# everywhere a role is not set, and `:bad` is bold as well as red, as in the terminal.
const _ROLE_CSS = Dict{Symbol, Vector{Pair{String, String}}}(
    :good => ["color" => "green"],
    :busy => ["color" => "teal"],
    :warn => ["color" => "darkorange"],
    :bad => ["color" => "red", "font-weight" => "bold"],
    :muted => ["color" => "gray"],
    :accent => ["color" => "purple"],
)

const _HTML_STYLE = PrettyTables.HtmlTableStyle(;
    title = ["font-weight" => "bold", "font-size" => "large", "text-align" => "left"],
)

function render_table(
        io::IO, ::MIME"text/html", title::AbstractString, sections::Vector{TableSection},
        note::Union{AbstractString, Nothing}
    )
    t = _table_layout(sections)
    if t === nothing
        print(io, "<p><b>", _html_escape(title), "</b></p>")
    else
        PrettyTables.pretty_table(
            io, t.data;
            backend = :html, alignment = :l, title,
            show_column_labels = false,
            row_group_labels = isempty(t.labels) ? nothing : t.labels,
            highlighters = [
                PrettyTables.HtmlHighlighter(
                    (_, i, _) -> i in t.headers, ["font-weight" => "bold"]
                ),
                PrettyTables.HtmlHighlighter(
                    (_, i, j) -> haskey(_ROLE_CSS, get(t.roles, (i, j), :none)),
                    (_, _, i, j) -> _ROLE_CSS[t.roles[(i, j)]]
                ),
            ],
            style = _HTML_STYLE,
        )
    end
    note === nothing || _html_notes(io, note)
    return nothing
end

# Footer notes, one paragraph per line, with backticked selections as `<code>`.
function _html_notes(io::IO, note::AbstractString)
    for line in split(note, "\n  "; keepempty = false)
        print(io, "<p style=\"font-size: smaller; margin: 2px 0;\">", _html_note(line), "</p>")
    end
    return nothing
end

# ── The history table ────────────────────────────────────────────────────────────────
#
# Everything decided here was decided in the core: `history_table` picked the rows that fit the
# terminal, dropped the columns that did not, formatted the cells and wrote the note. This method
# draws them, and it is the only `show` of a `MetricHistory` longer than one line, so a process
# without PrettyTables sees the compact form.
function Base.show(io::IO, ::MIME"text/plain", h::MetricHistory)
    isempty(h) && return show(io, h)
    height, width = displaysize(io)
    t = history_table(h, height, width)
    buf = IOBuffer()
    PrettyTables.pretty_table(
        IOContext(buf, io), t.cells;
        column_labels = [t.labels], alignment = t.alignment,
        title = t.title, title_alignment = :l,
        highlighters = [
            # The best epoch is the row a reader is looking for, so it is the one that earns
            # colour; a gap row is filler and is muted so the eye skips it.
            PrettyTables.TextHighlighter(
                (_, i, _) -> i == t.best_row, _ROLE_CRAYONS[:good]
            ),
            PrettyTables.TextHighlighter(
                (_, i, _) -> i in t.gap_rows, _ROLE_CRAYONS[:muted]
            ),
        ],
        table_format = PrettyTables.TextTableFormat(;
            vertical_lines_at_data_columns = :none,
        ),
        # The core already fitted the table to `displaysize(io)`; PrettyTables cropping on top of
        # that would elide what the thinning deliberately kept.
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
    )
    print(io, rstrip(String(take!(buf)), '\n'))
    isempty(t.note) || print(io, "\n  ", t.note)
    return nothing
end

# ── The history table, for a notebook ────────────────────────────────────────────────
#
# The same cells from `history_table`, unthinned and with every column, since a notebook scrolls.
function Base.show(io::IO, ::MIME"text/html", h::MetricHistory)
    if isempty(h)
        print(io, "<p><code>", _html_escape(sprint(show, h)), "</code></p>")
        return nothing
    end
    t = history_table(h, typemax(Int32), typemax(Int32))
    PrettyTables.pretty_table(
        io, t.cells;
        backend = :html,
        column_labels = [t.labels], alignment = t.alignment,
        title = t.title,
        highlighters = [
            PrettyTables.HtmlHighlighter(
                (_, i, _) -> i == t.best_row, ["font-weight" => "bold"]
            ),
        ],
        style = _HTML_STYLE,
    )
    isempty(t.note) || _html_notes(io, t.note)
    return nothing
end

_html_escape(s::AbstractString) =
    replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;")

# A footer note with its backticked selections, `h[:acc]` say, as `<code>`.
function _html_note(note::AbstractString)
    parts = split(_html_escape(note), '`')
    return join(
        (isodd(i) ? p : "<code>" * p * "</code>" for (i, p) in enumerate(parts))
    )
end

# The default, installed at definition. `table_renderer!` swaps it.
_TABLE_RENDERER[] = render_table
