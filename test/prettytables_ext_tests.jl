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
