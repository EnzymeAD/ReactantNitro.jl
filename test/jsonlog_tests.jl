# The shipped JSON default logger: the ten verbs write JSON Lines, the pathless default
# adopts the run's resolved `run_dir`, append mode extends a resumed run's file, `logger_info`
# carries the path, and a pathless logger that never reaches a `Nitro` fails loudly.

@testitem "jsonlog" begin
    using Test
    using ReactantNitro
    using ReactantNitro: JSONLogger, METRICS_FILE, _adopt_logger!
    using JSON3, Lux, Random, Statistics

    # ── The experiment, one tiny train for the adoption and append testsets ─────────────

    @experiment struct JsonMLP
        width::GraphConst{Int} = 4
    end
    ReactantNitro.build_model(e::JsonMLP, rng) = begin
        m = Lux.Chain(Lux.Dense(3 => e.width, tanh), Lux.Dense(e.width => 1))
        (m, Lux.setup(rng, m)...)
    end
    ReactantNitro.forward(::JsonMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::JsonMLP, y; yhat) = mean(abs2, y .- yhat)

    json_batches(n; seed) = (
        rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, Float32, 3, 4), yhat = randn(rng, Float32, 1, 4)) for _ in 1:n]
    )
    ReactantNitro.build_data(::JsonMLP, dist) = (; train = json_batches(2; seed = 1), val = json_batches(1; seed = 2))

    lines(path) = [JSON3.read(ln) for ln in readlines(path)]

    # ── The verbs, driven directly: every line is one well-formed JSON object ───────────

    @testset "the ten verbs write one JSON object per line" begin
        dir = mktempdir()
        path = joinpath(dir, "metrics.jsonl")
        l = JSONLogger(path)

        @testset "params, metrics, tags, other, confusion, finish" begin
            log_params!(l, (; seed = 42, max_epochs = 2))
            log_metrics!(l, (; loss = 0.31, acc = 0.9); context = "train", epoch = 1, step = 4)
            log_metrics!(l, (; val_loss = 0.29); context = "validate", epoch = 1, step = 8)
            log_tags!(l, (; run = "probe"))
            log_other!(l, "binding_report", "report text")
            log_confusion!(l, [1 0; 0 1], ["a", "b"]; epoch = 1)
            finish!(l, :completed)
            ls = lines(path)

            @test length(ls) == 7
            @test ls[1]["type"] == "params" && ls[1]["seed"] == 42
            # The carrier fields win over a metric that shares a name.
            @test ls[2]["type"] == "metrics" && ls[2]["context"] == "train"
            @test ls[2]["epoch"] == 1 && ls[2]["step"] == 4
            @test ls[2]["loss"] == 0.31 && ls[2]["acc"] == 0.9
            @test ls[3]["type"] == "metrics" && ls[3]["context"] == "validate" && ls[3]["val_loss"] == 0.29
            @test ls[4]["type"] == "tags" && ls[4]["run"] == "probe"
            @test ls[5]["type"] == "other" && ls[5]["key"] == "binding_report"
            @test ls[5]["value"] == "report text"
            @test ls[6]["type"] == "confusion" && ls[6]["epoch"] == 1
            @test ls[6]["matrix"] == [[1, 0], [0, 1]] && ls[6]["labels"] == ["a", "b"]
            @test ls[7]["type"] == "finish" && ls[7]["status"] == "completed"
        end

        @testset "finish! writes the outcome and closes the handle" begin
            path2 = joinpath(dir, "finish.jsonl")
            l2 = JSONLogger(path2)
            finish!(l2, :early_stop)
            @test lines(path2)[1]["type"] == "finish"
            @test lines(path2)[1]["status"] == "early_stop"
            @test l2.io === nothing                       # closed
        end

        @testset "strings are escaped, non-finite floats become null" begin
            path3 = joinpath(dir, "esc.jsonl")
            l3 = JSONLogger(path3)
            log_params!(l3, (; name = "x\"y\\z\nw"))
            log_metrics!(l3, (; loss = NaN); context = "train", epoch = 1, step = 1)
            ls = lines(path3)
            @test ls[1]["name"] == "x\"y\\z\nw"
            @test ls[2]["loss"] === nothing               # NaN became JSON `null`
        end
    end

    # ── The default wiring: adoption, append, opt-out ───────────────────────────────────

    @testset "the pathless default adopts the run's resolved run_dir" begin
        dir = mktempdir()
        n = Nitro(JsonMLP(); run_dir = dir, checkpointer = nothing)
        @test n.logger isa JSONLogger
        @test n.logger.path == joinpath(dir, METRICS_FILE)
        @test n.logger.io isa IO                            # step 12 already wrote the params line
        # Step 12 logged params into the adopted file before the first compile.
        ls = lines(joinpath(dir, METRICS_FILE))
        @test ls[1]["type"] == "params" && ls[1]["width"] == 4
        @test any(x -> x["type"] == "other" && x["key"] == "binding_report", ls)
    end

    @testset "append mode: a resumed run extends, never truncates" begin
        dir = mktempdir()
        n = train!(Nitro(JsonMLP(); run_dir = dir, checkpointer = nothing, max_epochs = 1))
        path = joinpath(dir, METRICS_FILE)
        before = length(lines(path))
        @test before > 2
        @test lines(path)[end]["type"] == "finish"
        @test lines(path)[end]["status"] == "completed"

        # A second construction in the same directory (the resume story) appends its params line.
        n2 = Nitro(JsonMLP(); run_dir = dir, checkpointer = nothing, resume = :auto)
        after = length(lines(path))
        @test after > before
    end

    @testset "an explicit path is a user choice and is never overwritten" begin
        dir = mktempdir()
        custom = joinpath(dir, "custom.jsonl")
        n = Nitro(JsonMLP(); run_dir = dir, checkpointer = nothing, logger = JSONLogger(custom))
        @test n.logger.path == custom
        @test isfile(custom)
        @test !isfile(joinpath(dir, METRICS_FILE))
    end

    @testset "`logger = nothing` stays the documented opt-out" begin
        dir = mktempdir()
        n = Nitro(JsonMLP(); run_dir = dir, checkpointer = nothing, logger = nothing)
        @test n.logger === nothing
        train!(n)
        @test isempty(readdir(dir))
    end

    @testset "a pathless logger that never reaches a Nitro fails loudly" begin
        l = JSONLogger()
        @test_throws ErrorException log_params!(l, (; a = 1))
        err = try
            log_params!(l, (; a = 1))
            nothing
        catch ex
            ex
        end
        @test occursin("no path", err.msg)
        # `_adopt_logger!` is what a Nitro calls; applying it directly fixes the path.
        _adopt_logger!(l, "/tmp/some_run")
        @test l.path == joinpath("/tmp/some_run", METRICS_FILE)
    end

    # ── logger_info ─────────────────────────────────────────────────────────────────────

    @testset "`logger_info` reports the file's path" begin
        l = JSONLogger("runs/x/metrics.jsonl")
        @test logger_info(l) == (; path = "runs/x/metrics.jsonl")
        @test logger_info(nothing) == (;)
    end

end
