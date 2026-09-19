# ReactantNitroPrettyTablesExt.jl
#
# The framed renderer for this package's long `show` methods, and the `history` table. Loading
# PrettyTables anywhere in a session is the trigger, and `__init__` points
# `ReactantNitro._TABLE_RENDERER` here.
#
# ── Why an extension and not a dependency ────────────────────────────────────────────
#
# A framework whose display pulls in a table library has made every deployment of it carry that
# library, including a serving process that renders nothing. In practice Reactant depends on
# PrettyTables, so every session that loads this package loads this extension too, and it is THE
# renderer: the core keeps no second one to maintain beside it. Without this extension a long
# `show` prints its title and a line saying so.
#
# ── How several sections become one frame ────────────────────────────────────────────
#
# `row_group_labels` is the feature this rests on: a full-width labelled band drawn between data
# rows. Each of our sections becomes one of those, and a section's own column header is emitted as
# an ordinary first row of its band, since a table has exactly one real column-label row and it
# would have to sit above every section at once.
#
# NO VERTICAL RULES, and that is the choice that makes the whole thing work rather than a style
# preference. A table has one column structure, so the widest cell anywhere in a column sets that
# column for every section, and a two-column band inside a three-column table therefore ends in a
# stretch of air. Unruled, that air is invisible and the band reads as a short line. Ruled, it
# reads as a cell somebody forgot to fill in.
#
# ── The one thing to know if the display changes under you ───────────────────────────
#
# This activates on LOAD, so a session that pulls PrettyTables in indirectly (Reactant does)
# gets the framed table without asking for it. `ReactantNitro.table_renderer!(nothing)` uninstalls
# it for the rest of the process, and this extension never reclaims it.
module ReactantNitroPrettyTablesExt

import ReactantNitro
import PrettyTables

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

# The renderer contract, as `ReactantNitro.table_renderer!` documents it: a title, the sections,
# and a trailing note.
#
# NAMED AND TYPED SPECIFICALLY on purpose. It is reached through a `Ref{Any}`, so the call is
# dynamic and nothing would check a looser signature; a bare `render` taking four untyped
# arguments is the shape that reads as applicable to any four-argument call and turns a stack
# trace from this display path into a puzzle.
function render_table(
        io::IO, title::AbstractString, sections::Vector{ReactantNitro.TableSection},
        note::Union{AbstractString, Nothing}
    )
    rows = ReactantNitro.TableRows()
    labels = Pair{Int, String}[]
    headers = Set{Int}()
    # A section's styles are keyed by its OWN row numbers; the table's are keyed by the table's.
    # This is where the two are reconciled, once, rather than at every lookup.
    roles = Dict{Tuple{Int, Int}, Symbol}()
    for sec in sections
        # An empty section title draws no band label. The leading section uses it: the table's
        # own title already names that band, and a label directly under the title is a heading
        # printed twice. A label is placed at the row it precedes, so it is computed BEFORE the
        # section's rows are appended.
        isempty(sec.title) || push!(labels, (length(rows) + 1) => sec.title)
        if any(!isempty, sec.header)
            push!(rows, sec.header)
            # Remembered so the highlighter below can BOLD it. A section's column header is an
            # ordinary data row as far as the table is concerned, so nothing else distinguishes
            # `group  settings  params` from the group rows underneath it, and a header that
            # reads as data is worse than no header at all.
            push!(headers, length(rows))
        end
        offset = length(rows)
        append!(rows, sec.rows)
        for ((r, c), role) in sec.styles
            roles[(offset + r, c)] = role
        end
    end
    isempty(rows) && return print(io, title)
    cols = maximum(length, rows)
    data = [get(rows[r], c, "") for r in 1:length(rows), c in 1:cols]

    # Rendered into a buffer for one reason: `pretty_table` ends its output with a newline and a
    # `show` method must not, or every display carries a blank line under it. `IOContext(buf, io)`
    # carries the caller's attributes across, `:color` above all, so the detour costs nothing else.
    buf = IOBuffer()
    PrettyTables.pretty_table(
        IOContext(buf, io), data;
        alignment = :l, title, title_alignment = :l,
        # The real column-label row is off: every section carries its own header as a data row,
        # because one label row cannot describe five sections with different columns.
        show_column_labels = false,
        row_group_labels = isempty(labels) ? nothing : labels,
        # ORDER IS THE PRECEDENCE: PrettyTables applies the FIRST highlighter that matches, so
        # the header rule comes first and a section's column header stays bold even where a role
        # was set on the same coordinates. Every crayon here is emitted solely when the
        # destination declares color, so a `sprint` in a test or a redirect to a file still gets
        # clean text.
        highlighters = [
            PrettyTables.TextHighlighter((_, i, _) -> i in headers; bold = true),
            # A ROLE THIS PALETTE DOES NOT KNOW LEAVES THE CELL ALONE, which is why the
            # predicate looks the role up here rather than only checking that one was set. The
            # roles are a contract between a `show` in the core and whatever renderer is
            # installed, so a core that grows a seventh must not take a `KeyError` out of the
            # display of every handle in a session that has not upgraded this extension with it.
            PrettyTables.TextHighlighter(
                (_, i, j) -> haskey(_ROLE_CRAYONS, get(roles, (i, j), :none)),
                (_, _, i, j) -> _ROLE_CRAYONS[roles[(i, j)]]
            ),
        ],
        table_format = PrettyTables.TextTableFormat(;
            vertical_lines_at_data_columns = :none,
            horizontal_line_after_column_labels = false,
        ),
        # NO CROPPING. The default fits the table to the display and drops what does not fit, and
        # what does not fit here is the right-hand column: the sources and the checkpoint path,
        # which are the reason someone printed the handle. A wrapped long line beats an elided one.
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
    )
    print(io, rstrip(String(take!(buf)), '\n'))
    note === nothing || print(io, "\n  ", note)
    return nothing
end

# ── The history table ────────────────────────────────────────────────────────────────
#
# Everything decided here was decided in the core: `history_table` picked the rows that fit the
# terminal, dropped the columns that did not, formatted the cells and wrote the note. This method
# draws them, and it is the only `show` of a `MetricHistory` longer than one line, so a process
# without PrettyTables sees the compact form.
function Base.show(io::IO, ::MIME"text/plain", h::ReactantNitro.MetricHistory)
    isempty(h) && return show(io, h)
    height, width = displaysize(io)
    t = ReactantNitro.history_table(h, height, width)
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
# Jupyter, VS Code and Pluto ask for `text/html` before `text/plain`, so without this method a
# notebook shows the terminal table in a `<pre>`, thinned against the 24 lines a terminal-less
# `IO` claims to have. This is the same table drawn by the HTML backend, with NO thinning and no
# column dropped: a notebook scrolls, and the cell is the record. Decimals, column order, the
# best-epoch mark and the footer notes are `history_table`'s, unchanged.
function Base.show(io::IO, ::MIME"text/html", h::ReactantNitro.MetricHistory)
    if isempty(h)
        print(io, "<p><code>", _html_escape(sprint(show, h)), "</code></p>")
        return nothing
    end
    t = ReactantNitro.history_table(h, typemax(Int32), typemax(Int32))
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
    )
    for note in split(t.note, "\n  "; keepempty = false)
        print(io, "<p style=\"font-size: smaller; margin: 2px 0;\">", _html_note(note), "</p>")
    end
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

function __init__()
    ReactantNitro.table_renderer!(render_table)
    return nothing
end

end
