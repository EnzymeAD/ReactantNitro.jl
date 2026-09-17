# The ReactantNitroPrettyTablesExt extension: the boxed renderer it installs for every long `show`
# in this package.
#
# The extension only activates when PrettyTables is loaded alongside ReactantNitro, so the
# `using PrettyTables` below is what triggers it. That also means this file CHANGES A PROCESS-WIDE
# SETTING: `__init__` installs the renderer, and every testitem sharing this worker would render
# through it afterwards. ReTestItems runs one testitem per worker process at a time, and the last
# testset here puts the previous renderer back, so neither this file's own later assertions nor a
# testitem that follows it sees a renderer it did not ask for.
@testitem "prettytables_ext" begin
    using Test
    using ReactantNitro
    using PrettyTables
    using Lux, Random

    @experiment struct PTExp
        width::GraphConst{Int} = 4
        scale::Device{Float32} = 0.5f0
        buffers::Device{NamedTuple{(:w,), Tuple{Matrix{Float32}}}} = (w = zeros(Float32, 32, 32),)
        max_epochs::Int = 1
    end
    # The minimum a `Nitro` construction needs. There is no training here: the handle exists to be
    # displayed, which is the only thing this file is about.
    ptexp_chain(w) = Lux.Chain(Lux.Dense(3 => w, tanh), Lux.Dense(w => 2))
    ReactantNitro.build_model(e::PTExp, rng) =
        (m = ptexp_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::PTExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::PTExp, ŷ; y) = sum(abs2, ŷ .- y) / size(y, 2)

    # Colour on for the duration of `f`, restored afterwards. A COLOURLESS PROCESS NEEDS BOTH
    # THIS AND `IOContext(io, :color => true)`, because two separate gates are involved and they
    # read different things. PrettyTables decides whether to style from the `IOContext` it is
    # handed; Crayons decides whether to emit the escape from the buffer it is printing INTO,
    # which is one PrettyTables made and which does not carry that context, so it falls back to
    # the process-global `Base.get_have_color()`. Set only the context and the output has resets
    # with no colour before them; set only this and there is no styling at all. A REPL has the
    # global on and needs neither, which is why this is a test problem and not a user one.
    function with_color(f)
        prev = PrettyTables.Crayons.FORCE_COLOR[]
        PrettyTables.Crayons.force_color(true)
        try
            return f()
        finally
            PrettyTables.Crayons.force_color(prev)
        end
    end

    @testset "loading PrettyTables installs the renderer" begin
        @test Base.get_extension(ReactantNitro, :ReactantNitroPrettyTablesExt) !== nothing
        @test ReactantNitro._TABLE_RENDERER[] !== nothing
    end

    @testset "the boxed form carries the same facts as the plain one" begin
        boxed = sprint(show, MIME"text/plain"(), PTExp())
        # Box drawing, which is the whole point, and the table's own header.
        @test occursin("│", boxed) && occursin("┌", boxed)
        @test occursin("field", boxed) && occursin("marker", boxed)
        # Every field, its marker, and its value survive the change of renderer.
        for s in ("width", "GraphConst", "4", "scale", "Device", "0.5f0", "max_epochs", "Host")
            @test occursin(s, boxed)
        end
        # And the reason all of this exists: a device buffer is still summarized, never printed.
        @test occursin("size 32x32", boxed)
        @test !occursin("0.0, 0.0", boxed)
        @test !occursin("Float32[", boxed)
    end

    # A `show` method must not end with a newline: the REPL supplies the line break, and
    # `pretty_table` writes one of its own, so the extension has to strip it. Getting this wrong
    # puts a blank line under every display in the session.
    @testset "no trailing newline, and the note survives" begin
        boxed = sprint(show, MIME"text/plain"(), PTExp())
        @test !endswith(boxed, "\n")

        n = Nitro(PTExp(); data = (;), checkpointer = nothing, run_dir = mktempdir())
        handle = sprint(show, MIME"text/plain"(), n)
        @test !endswith(handle, "\n")
        @test occursin("│", handle)
        # The trailing note is printed under the box rather than swallowed with the newline.
        @test occursin("binding_report", handle)
        @test endswith(handle, "`logger_info`")
    end

    # Cropping is the default and it drops the RIGHT-hand column, which is where the metrics and
    # the checkpoint path live: the reason someone printed the handle in the first place.
    @testset "long cells are not elided" begin
        long = "x"^200
        out = sprint() do io
            ReactantNitro._render_table(io, "t", ["", ""], [["k", long]])
        end
        @test occursin(long, out)
        @test !occursin("⋯", out)
        @test !occursin("columns omitted", out)
    end

    # ── several sections, one frame ─────────────────────────────────────────────────────
    #
    # `row_group_labels` is what makes this possible, and a section's own column header rides as
    # an ordinary data row because a table has exactly one real column-label row and it would
    # have to describe every section at once. What that costs, and what these assertions pin, is
    # that the header row is distinguishable from the data under it by weight alone.
    @testset "sections render as bands of ONE frame" begin
        n = Nitro(PTExp(); data = (;), checkpointer = nothing, run_dir = mktempdir())
        boxed = sprint(show, MIME"text/plain"(), n)

        # ONE frame: one top border and one bottom border, however many bands are inside it.
        @test count("┌", boxed) == 1
        @test count("└", boxed) == 1
        # The bands are separated by full-width rules rather than by a new box each.
        @test count("├", boxed) >= 1
        # NO INTERIOR VERTICAL RULES. A section's columns are shared with every other section, so
        # a two-column band inside a three-column table ends in air; ruled, that air reads as a
        # cell somebody forgot to fill in.
        for line in split(boxed, "\n")
            startswith(line, "│") || continue
            @test count("│", line) == 2
        end
        # The leading band carries no label: the title names it.
        @test occursin("Nitro for PTExp", boxed)
        @test occursin("gradient clip", boxed)

        # Clean text when the process has no color, which is a redirect to a file and is CI.
        @test !occursin("\e[", boxed)

        # And bold when it does; see `with_color` for why both switches are set.
        @test with_color() do
            occursin(
                "\e[1m",
                sprint(io -> show(IOContext(io, :color => true), MIME"text/plain"(), n))
            )
        end
    end

    # ── the palette ─────────────────────────────────────────────────────────────────────
    #
    # A role is a name in the core and a crayon here, and the split is what keeps an escape
    # sequence out of `binding_report`'s string. These assertions pin both ends of it: that a role
    # the core sets reaches a colour, and that a role this extension has never heard of leaves the
    # cell alone rather than taking the whole display down with a `KeyError`.
    @testset "roles map to the basic ANSI set, and an unknown role is inert" begin
        ext = Base.get_extension(ReactantNitro, :ReactantNitroPrettyTablesExt)
        crayon(c) = sprint(print, c; context = :color => true)

        # THE BASIC EIGHT, which is what makes this legible on a theme nobody here chose: these
        # are the codes a terminal remaps, unlike a 256-colour index or a hex triple.
        #
        # Matched as ONE PARAMETER OF THE SEQUENCE rather than the whole of it, because a crayon
        # carrying an attribute as well as a colour emits them together: `:bad` is red and bold,
        # so it writes `\e[31;1m` and an assertion on `\e[31m` would miss it.
        has_code(c, code) = occursin(Regex("\\e\\[(?:[0-9]+;)*$(code)(?:;[0-9]+)*m"), crayon(c))
        for (role, code) in
            (:good => 32, :busy => 36, :warn => 33, :bad => 31, :muted => 90, :accent => 35)
            @test has_code(ext._ROLE_CRAYONS[role], code)
        end
        # `:bad` is the one role that is not colour alone: a failed run is the cell that has to be
        # findable in a scrollback, and red on its own is the attribute a colourblind reader loses.
        @test has_code(ext._ROLE_CRAYONS[:bad], 1)
        # No backgrounds: these tables get pasted into tickets, where a background survives as a
        # block of highlight.
        for cr in values(ext._ROLE_CRAYONS)
            @test !occursin(r"\e\[4[0-7]m", crayon(cr))
        end

        rows = ReactantNitro.TableRows([["a", "b"]])
        rendered(styles) = with_color() do
            sprint(
                io -> ext.render_table(
                    IOContext(io, :color => true), "t",
                    [ReactantNitro.TableSection("", String[], rows, styles)], nothing
                )
            )
        end
        @test occursin("\e[35m", rendered(ReactantNitro.CellStyles((1, 2) => :accent)))
        # A role from a newer core than this extension: the cell renders, undecorated.
        out = rendered(ReactantNitro.CellStyles((1, 2) => :not_a_role_yet))
        @test occursin("a", out) && occursin("b", out)
        @test !occursin("\e[3", out)
    end

    @testset "the plain renderer is restorable" begin
        prev = ReactantNitro.table_renderer!(nothing)
        try
            plain = sprint(show, MIME"text/plain"(), PTExp())
            @test !occursin("│", plain)
            @test occursin("width", plain) && occursin("GraphConst", plain)
            @test !endswith(plain, "\n")
        finally
            ReactantNitro.table_renderer!(prev)
        end
        @test ReactantNitro._TABLE_RENDERER[] === prev
    end
end
