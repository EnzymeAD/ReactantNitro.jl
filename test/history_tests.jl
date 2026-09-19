# `history(nitro)`: the per-epoch series a handle keeps, its indexing, and the table that fits a
# terminal.
@testitem "history" begin
    using Test
    using ReactantNitro
    using ReactantNitro: MetricHistory, history_table, thin_rows
    using Lux, Random, Statistics, Tables
    # The shared toy model from the test kit, whose precompile workload already trained it once,
    # so this item starts with the framework's specializations for it warm. Its `metrics` emit one
    # scalar (`mae`) and one summed matrix (`cm`), which is the pair the table has to handle.
    using NitroTestKit

    mk(; kw...) = kit_nitro(; kw...)

    @testset "the series, as data" begin
        n = train!(
            mk(; max_epochs = 3, checkpointer = TopKCheckpointer(; metric = :mae, mode = :min))
        )
        h = history(n)
        @test h isa MetricHistory
        @test length(h) == 3
        @test h.epoch == [1, 2, 3]
        @test h.step == [4, 8, 12]                  # four batches per epoch at accum 1
        @test all(isfinite, h.loss)
        @test h.mae isa Vector{<:Real}
        @test length(h.mae) == 3
        @test h.cm[1] == 2 .* ones(Int, 2, 2)       # summed over the two val batches, never divided
        # The fixed column order: the axes, the train loss, the checkpointer's metric.
        @test h.columns == [:epoch, :step, :loss, :mae]
        @test first.(h.hidden) == [:cm]
        @test :cm in propertynames(h)
        @test h.best.metric === :mae
        @test h.best.mode === :min
        @test h.best.epoch == h.epoch[argmin(h.mae)]

        # Indexing is by EPOCH for rows and by name for columns.
        @test h[2].epoch == 2
        @test haskey(h[2], :mae)
        @test h[end].epoch == 3
        @test h[2:3].epoch == [2, 3]
        @test h[:mae].columns == [:epoch, :step, :mae]
        @test h[2:3, :mae].epoch == [2, 3]
        @test h[:, :mae].columns == [:epoch, :step, :mae]
        @test h[:cm].hidden == h.hidden               # a non-scalar is selectable too
        @test h[:mae].hidden == []                    # and dropped when not selected
        # By step: the epochs whose closing step falls in the range, columns untouched.
        @test h[step = 5:8].epoch == [2]
        @test h[step = 5:8].columns == h.columns
        @test h[step = 5:8].hidden == h.hidden
        @test h[:mae, step = 8:12].epoch == [2, 3]
        @test h[:mae, step = 8:12].columns == [:epoch, :step, :mae]
        @test isempty(h[step = 100:200])
        @test collect(h) == h.rows
        # A Tables.jl table with column access: the scalar columns, and not the matrix.
        @test Tables.istable(h)
        @test Tables.columnaccess(h)
        @test !Tables.rowaccess(h)
        @test keys(Tables.columntable(h)) == (:epoch, :step, :loss, :mae)
        @test Tables.columntable(h).mae == h.mae
        @test Tables.rowtable(h)[2].epoch == 2
        @test Tables.schema(h).names == (:epoch, :step, :loss, :mae)
        err = try
            h[9]
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("no epoch 9", err.msg)
        err = try
            h[:nope]
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("`mae`", err.msg)

        @test sprint(show, h) == "history of KitMLP: 3 epochs (1 to 3), 1 metric, $(n.run_dir)"
        # The table, through the PrettyTables extension Reactant's dependency loads.
        s = sprint(show, MIME"text/plain"(), h)
        @test occursin("history of KitMLP", s)
        @test occursin("mae", s)
        @test occursin("*", s)
        @test occursin("best mae (min)", s)
        @test occursin("not tabulated: cm (2x2)", s)
        @test !endswith(s, "\n")

        # Training on extends the same series.
        n2 = Nitro(
            KitMLP(); run_dir = n.run_dir, data = n.data, max_epochs = 5, resume = :auto,
            checkpointer = TopKCheckpointer(; metric = :mae, mode = :min)
        )
        train!(n2)
        h2 = history(n2)
        @test h2.epoch == [4, 5]                      # fresh per handle, from the resumed epoch
        @test occursin("before 4", history_table(h2, 40, 120).note)
    end

    @testset "an untrained handle has an empty history" begin
        h = history(mk(; checkpointer = nothing))
        @test isempty(h)
        @test length(h) == 0
        @test occursin("0 epochs", sprint(show, h))
        @test sprint(show, MIME"text/plain"(), h) == sprint(show, h)
    end

    @testset "thin_rows keeps the first, the best and the last, and marks gaps" begin
        @test thin_rows(5, 3, 10) == 1:5
        @test thin_rows(12, nothing, 12) == 1:12
        t = thin_rows(40, 23, 12)
        @test length(t) == 12
        nz = filter(!=(0), t)
        @test issorted(nz)
        @test allunique(nz)
        @test 1 in nz
        @test 23 in nz
        @test 40 in nz
        @test t[1] == 1
        @test t[end] == 40
        @test count(==(0), t) <= 2
        # A gap sits between two rows that are not consecutive, and never beside another gap.
        for i in 2:(length(t) - 1)
            t[i] == 0 || continue
            @test t[i - 1] != 0
            @test t[i + 1] > t[i - 1] + 1
        end
        # The best window grows in both directions, so the best epoch gets the most context.
        @test all(x -> x in nz, 21:25)
        # Windows that meet merge and free the gap.
        t2 = thin_rows(40, 39, 9)
        @test 39 in t2
        @test 40 in t2
        @test count(==(0), t2) == 1
        # No best: two windows, one gap.
        t3 = thin_rows(40, nothing, 8)
        @test t3[1] == 1
        @test t3[end] == 40
        @test count(==(0), t3) == 1
        # A budget too small for the anchors still shows all three.
        @test filter(!=(0), thin_rows(40, 20, 3)) == [1, 20, 40]
    end

    @testset "the table fits the terminal" begin
        # Synthetic, so forty epochs are fitted without training forty.
        rows = NamedTuple[
            (;
                epoch = i, step = 4i, loss = 1.0 / i, mae = 0.5 + abs(i - 23) / 100,
                extra_long_metric_name = Float32(i), another = i / 7,
            ) for i in 1:40
        ]
        h = MetricHistory(
            :Synth, "runs/synth", rows,
            [:epoch, :step, :loss, :mae, :extra_long_metric_name, :another],
            Pair{Symbol, String}[:cm => "2x2"], (; metric = :mae, mode = :min, epoch = 23), 1
        )
        t = history_table(h, 24, 80)
        @test t.labels[1:4] == ["epoch", "step", "loss", "mae"]
        @test t.labels[end] == " "
        @test size(t.cells, 1) <= 15
        @test t.cells[t.best_row, end] == "*"
        @test t.cells[t.best_row, 1] == "23"
        @test !isempty(t.gap_rows)
        @test all(i -> t.cells[i, 1] == "⋮", t.gap_rows)
        @test occursin("best mae (min)", t.note)
        @test occursin("thinned", t.note)
        # The suggested selection is one that fits: fifteen rows around the best epoch, not
        # the whole run, which would only come back thinned the same way.
        @test occursin("`h[16:30]`", t.note)
        @test size(history_table(h[16:30], 24, 80).cells, 1) == 15
        @test occursin("not tabulated: cm (2x2)", t.note)
        @test occursin("12 of 40 epochs", t.title) || occursin("of 40 epochs", t.title)

        # Narrow: metrics drop from the right, are named, and the checkpointer's never drops.
        tn = history_table(h, 24, 50)
        @test length(tn.labels) < length(t.labels)
        @test "mae" in tn.labels
        @test occursin("not shown for width", tn.note)
        @test occursin("extra_long_metric_name", tn.note)
        @test occursin("another", tn.note)
        @test occursin("`h[:extra_long_metric_name, :another]`", tn.note)

        # Tall and wide: every row, no thinning, and one decimal count per column so the points
        # align: the smallest loss shown is 0.025, so the column prints five decimals.
        tt = history_table(h, 60, 200)
        @test size(tt.cells, 1) == 40
        @test isempty(tt.gap_rows)
        @test !occursin("thinned", tt.note)
        @test tt.cells[1, 3] == "1.00000"
        @test tt.cells[3, 3] == "0.33333"
        @test tt.cells[40, 3] == "0.02500"
        @test tt.cells[40, 2] == "160"
        @test all(c -> length(split(c, '.')[end]) == 5, tt.cells[:, 3])
        # Notes are one per line.
        @test occursin("\n  ", t.note)

        # A selection that starts late is a selection, not a resumed run.
        @test !occursin("before 12", history_table(h[12:40], 60, 200).note)
        late = MetricHistory(
            :Synth, "runs/synth", rows[12:end], h.columns, h.hidden, h.best, 12
        )
        @test occursin("before 12", history_table(late, 60, 200).note)
        @test occursin("before 12", history_table(late[20:30], 60, 200).note)

        # Rendered, at two terminal sizes.
        s = sprint(show, MIME"text/plain"(), h; context = :displaysize => (24, 80))
        @test occursin("history of Synth", s)
        @test occursin("⋮", s)
        @test occursin("*", s)
        @test occursin("mae", s)
        @test !endswith(s, "\n")
        wide = sprint(show, MIME"text/plain"(), h; context = :displaysize => (60, 200))
        @test count("\n", wide) >= 40
        @test occursin("extra_long_metric_name", wide)

        # As HTML, for a notebook: every row and every column, the best row bold, the notes
        # under it with their selections as code, and nothing thinned.
        @test showable(MIME"text/html"(), h)
        html = sprint(show, MIME"text/html"(), h)
        @test occursin("<table", html)
        @test count("<tr", html) >= 41
        @test occursin("extra_long_metric_name", html)
        # Bold on the best row's cells and on no other data row.
        @test occursin("<td style = \"font-weight: bold; text-align: right;\">23</td>", html)
        @test occursin("<td style = \"text-align: right;\">22</td>", html)
        # Right-aligned bold cells: the column labels and the best row, less the marker column,
        # which is left-aligned in both.
        @test count("font-weight: bold; text-align: right;\">", html) == 2 * (length(t.labels) - 1)
        @test !occursin("⋮", html)
        @test !occursin("thinned", html)
        @test occursin("best mae (min)", html)
        @test occursin("not tabulated: cm (2x2)", html)
        @test occursin("<code>", html) || !occursin("`", html)
        @test !occursin("`", html)
        empty = MetricHistory(
            :Synth, "runs/synth", NamedTuple[], [:epoch, :step, :loss],
            Pair{Symbol, String}[], nothing, 1
        )
        @test occursin("0 epochs", sprint(show, MIME"text/html"(), empty))
    end

    @testset "columns are typed, and a gap is missing" begin
        rows = NamedTuple[(; epoch = i, step = i, loss = 1.0, acc = 0.5f0) for i in 1:4]
        rows[3] = (; epoch = 3, step = 3, loss = 1.0)
        h = MetricHistory(
            :Synth, "runs/synth", rows, [:epoch, :step, :loss, :acc], Pair{Symbol, String}[],
            nothing, 1
        )
        @test h.loss isa Vector{Float64}
        @test h.epoch isa Vector{Int}
        @test h.acc isa Vector{Union{Missing, Float32}}
        @test ismissing(h.acc[3])
        @test Tables.schema(h).types == (Int, Int, Float64, Union{Missing, Float32})
        @test ismissing(Tables.rowtable(h)[3].acc)
        # The HTML table shows the gap as an empty cell, like the terminal table.
        @test occursin("<td", sprint(show, MIME"text/html"(), h))
    end

    @testset "history_series picks the curves a plot draws" begin
        using ReactantNitro: history_series
        rows = NamedTuple[
            (; epoch = i, step = 4i, loss = 1.0 / i, mae = 0.5 + abs(i - 23) / 100, another = i / 7)
                for i in 1:40
        ]
        # Epoch 30 never reported `another`, and epoch 31's mae is NaN.
        rows[30] = (; epoch = 30, step = 120, loss = 1 / 30, mae = 0.57)
        rows[31] = (; epoch = 31, step = 124, loss = 1 / 31, mae = NaN, another = 31 / 7)
        best = (; metric = :mae, mode = :min, epoch = 23)
        h = MetricHistory(
            :Synth, "runs/synth", rows, [:epoch, :step, :loss, :mae, :another],
            Pair{Symbol, String}[:cm => "2x2"], best, 1
        )
        # The default is the checkpointer's metric alone, with its best epoch marked.
        s = history_series(h)
        @test s.title == "history of Synth"
        @test s.xlabel == "epoch"
        @test [p.name for p in s.panels] == [:mae]
        p = s.panels[1]
        @test p.x == 1:40
        @test length(p.y) == 40
        @test isnan(p.y[31])                           # kept, so the line breaks there
        @test p.best == (23, 0.5)
        @test p.subtitle == "mae (min), best at epoch 23"
        # By step, the mark moves with the axis.
        @test history_series(h; x = :step).panels[1].best == (92, 0.5)
        @test history_series(h; x = :step).xlabel == "step"
        # `:all` is every column but the axes; a name or names select in column order.
        @test [p.name for p in history_series(h; metrics = :all).panels] == [:loss, :mae, :another]
        @test [p.name for p in history_series(h; metrics = :another).panels] == [:another]
        @test [p.name for p in history_series(h; metrics = (:another, :loss)).panels] ==
            [:another, :loss]
        # A row without the metric is skipped, not zeroed; the other panels carry no mark.
        pa = history_series(h; metrics = :another).panels[1]
        @test length(pa.x) == 39
        @test !(30 in pa.x)
        @test pa.best === nothing
        @test pa.subtitle == "another"
        # A selection that dropped the best epoch draws no mark; one that kept it does.
        @test history_series(h[30:40]).panels[1].best === nothing
        @test history_series(h[20:25]).panels[1].best == (23, 0.5)
        # Without a checkpointer's metric in the columns, the default is every metric, and a
        # history of only loss draws the loss.
        @test [p.name for p in history_series(h[:another]).panels] == [:another]
        nb = MetricHistory(
            :Synth, "runs/synth", rows, [:epoch, :step, :loss, :mae, :another],
            Pair{Symbol, String}[], nothing, 1
        )
        @test [p.name for p in history_series(nb).panels] == [:mae, :another]
        @test [p.name for p in history_series(nb[:loss]).panels] == [:loss]
        # Refusals name the problem.
        for (kw, needle) in (
                ((; x = :time), "`:epoch` or `:step`"),
                ((; metrics = :cm), "not a scalar metric (2x2)"),
                ((; metrics = :nope), "not a metric in this history"),
            )
            err = try
                history_series(h; kw...)
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin(needle, err.msg)
        end
        empty = MetricHistory(
            :Synth, "runs/synth", NamedTuple[], [:epoch, :step, :loss],
            Pair{Symbol, String}[], nothing, 1
        )
        err = try
            history_series(empty)
        catch e
            e
        end
        @test occursin("nothing to plot", err.msg)
    end
end
