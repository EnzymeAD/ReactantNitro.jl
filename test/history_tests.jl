# `history(nitro)`: the per-epoch series a handle keeps, its indexing, and the table that fits a
# terminal.
@testitem "history" begin
    using Test
    using ReactantNitro
    using ReactantNitro: MetricHistory, history_table, thin_rows
    using Lux, Random, Statistics
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
        @test collect(h) == h.rows
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
        @test occursin("`h[1:40]`", t.note)
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
    end
end
