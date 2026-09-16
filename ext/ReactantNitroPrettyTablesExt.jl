# ReactantNitroPrettyTablesExt.jl
#
# The PrettyTables renderer for this package's long `show` methods. Loading PrettyTables anywhere
# in a session is the trigger, and `__init__` points `ReactantNitro._TABLE_RENDERER` here.
#
# ── Why an extension and not a dependency ────────────────────────────────────────────
#
# A framework whose display pulls in a table library has made every deployment of it carry that
# library, including a serving process that renders nothing. The core keeps its own aligned-column
# renderer, which needs no dependency and is what CI and a log file see; this is the upgrade a
# human at a REPL gets for free once something in the session has already loaded PrettyTables.
#
# ── The one thing to know if the display changes under you ───────────────────────────
#
# This activates on LOAD, so a session that pulls PrettyTables in indirectly (DataFrames does)
# gets boxed tables without asking for them. That is the intended behaviour and it is also the
# reason `ReactantNitro.table_renderer!(nothing)` exists: it puts the plain renderer back for the
# rest of the process, and this extension never reclaims it.
module ReactantNitroPrettyTablesExt

import ReactantNitro
import PrettyTables

# The renderer contract, as `ReactantNitro.table_renderer!` documents it: a title, a column header,
# rows, and a trailing note. An all-empty header means there are no column names to show, which is
# the handle summary's shape (label and value, no heading over either).
#
# NAMED AND TYPED SPECIFICALLY on purpose. It is reached through a `Ref{Any}`, so the call is
# dynamic and nothing would check a looser signature; a bare `render` taking five untyped
# arguments is the shape that reads as applicable to any five-argument call and turns a stack
# trace from this display path into a puzzle.
function render_table(
        io::IO, title::AbstractString, header::Vector{String},
        rows::ReactantNitro.TableRows, note::Union{AbstractString, Nothing}
    )
    isempty(rows) && return print(io, title)
    cols = maximum(length, rows)
    data = [get(rows[r], c, "") for r in 1:length(rows), c in 1:cols]
    kwargs = any(!isempty, header) ?
        (; column_labels = [get(header, c, "") for c in 1:cols]) :
        (; show_column_labels = false)
    # Rendered into a buffer for one reason: `pretty_table` ends its output with a newline and a
    # `show` method must not, or every display carries a blank line under it. `IOContext(buf, io)`
    # carries the caller's attributes across, `:color` above all, so the detour costs nothing else.
    buf = IOBuffer()
    PrettyTables.pretty_table(
        IOContext(buf, io), data;
        alignment = :l, title,
        # NO CROPPING. The default fits the table to the display and drops what does not fit, and
        # what does not fit here is the right-hand column: the metrics and the checkpoint path,
        # which are the reason someone printed the handle. A wrapped long line beats an elided one.
        fit_table_in_display_horizontally = false,
        fit_table_in_display_vertically = false,
        kwargs...
    )
    print(io, rstrip(String(take!(buf)), '\n'))
    note === nothing || print(io, "\n  ", note)
    return nothing
end

function __init__()
    ReactantNitro.table_renderer!(render_table)
    return nothing
end

end
