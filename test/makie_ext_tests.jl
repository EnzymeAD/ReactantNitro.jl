# The ReactantNitroMakieExt extension: `plot(h)` and `plot(nitro)` as a SpecApi layout, one axis
# per curve, for any Makie backend. Makie alone here, no backend: a `Figure` is built and
# inspected without rendering, which is everything the extension decides. What the curves ARE is
# `history_series`'s business and is tested in history_tests.jl without Makie.
@testitem "makie_ext" begin
    using Test
    using ReactantNitro
    using ReactantNitro: MetricHistory
    using Makie

    @testset "the extension loads with Makie" begin
        @test Base.get_extension(ReactantNitro, :ReactantNitroMakieExt) !== nothing
    end

    rows = NamedTuple[
        (; epoch = i, step = 4i, loss = 1.0 / i, mae = 0.5 + abs(i - 23) / 100, another = i / 7)
            for i in 1:40
    ]
    h = MetricHistory(
        :Synth, "runs/synth", rows, [:epoch, :step, :loss, :mae, :another],
        Pair{Symbol, String}[:cm => "2x2"], (; metric = :mae, mode = :min, epoch = 23), 1
    )
    axes_of(fig) = [c for c in fig.content if c isa Axis]

    @testset "plot(h) is one axis for the checkpointer's metric, with the best epoch starred" begin
        fap = plot(h)
        @test fap isa Makie.FigureAxisPlot
        fig = fap.figure
        axes = axes_of(fig)
        @test length(axes) == 1
        ax = axes[1]
        @test ax.title[] == "mae (min), best at epoch 23"
        @test ax.xlabel[] == "epoch"
        plots = ax.scene.plots
        @test count(p -> p isa Lines, plots) == 1
        stars = filter(p -> p isa Scatter, plots)
        @test length(stars) == 1
        @test stars[1].marker[] == Makie.to_spritemarker(:star5)   # a BezierPath once converted
        @test only(stars[1][1][]) == Point2f(23, 0.5)
        @test any(c -> c isa Label && c.text[] == "history of Synth", fig.content)
    end

    @testset "keywords reach the selection, and the grid grows with the panels" begin
        fig = plot(h; x = :step, metrics = :all).figure
        axes = axes_of(fig)
        @test length(axes) == 3
        @test all(ax -> ax.xlabel[] == "step", axes)
        @test [ax.title[] for ax in axes] == ["loss", "mae (min), best at epoch 23", "another"]
        # Only the checkpointer's panel carries a star.
        @test count(ax -> any(p -> p isa Scatter, ax.scene.plots), axes) == 1
        # Indexing narrows what the plot sees.
        fig2 = plot(h[10:20, :another]).figure
        ax = only(axes_of(fig2))
        @test ax.title[] == "another"
        @test length(ax.scene.plots[1][1][]) == 11
    end

    @testset "the layout embeds in a caller's figure" begin
        fig = Figure()
        plot(fig[1, 1], h)
        plot(fig[2, 1], h; metrics = :loss)
        @test length(axes_of(fig)) == 2
    end
end
