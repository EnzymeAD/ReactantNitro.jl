# The ReactantNitroTensorBoardLoggerExt extension: the ten verbs against a real `TBLogger`, read
# back out of the event files it wrote.
#
# ── Read the scalars BEFORE anything writes a text summary ──────────────────────────
#
# TensorBoardLogger 0.1.26's own deserializer is broken for tensor summaries, which is what
# `log_text` writes: `deserialize_tensor_summary` reaches for a `.tensor` field that its generated
# protobuf type does not have, and `map_summaries` raises a `FieldError` the moment it reaches one.
# The WRITE side is fine (the field is a `OneOf` on `value`, and TensorBoard itself reads the file),
# so this is a reader bug and not a reason to avoid text summaries.
#
# The consequence for this file is only an ordering one: the scalar assertions run against a log
# that contains scalars and nothing else, and everything written as text afterwards is asserted on
# the raw event-file bytes instead. If a later TensorBoardLogger fixes the reader, the second half
# can move to `map_summaries` too.
@testitem "tensorboardlogger_ext" begin
    using Test
    using ReactantNitro
    using TensorBoardLogger
    using TensorBoardLogger: TBLogger, logdir, map_summaries, tb_overwrite

    @testset "the extension loads with TensorBoardLogger" begin
        @test Base.get_extension(ReactantNitro, :ReactantNitroTensorBoardLoggerExt) !== nothing
    end

    # ── Scalars: the tag prefix, the axis, and the driver's step counter ────────────

    @testset "metrics become scalar summaries tagged by context at the driver's step" begin
        lg = TBLogger(joinpath(mktempdir(), "run"), tb_overwrite)

        log_metrics!(lg, (; loss = 0.31, acc = 0.9); step = 4, epoch = 1, context = "train")
        log_metrics!(lg, (; val_loss = 0.29); step = 8, epoch = 1, context = "validate")
        log_metrics!(lg, (; data_wait_frac = 0.01); step = 8, epoch = 1, context = "data")
        log_other!(lg, "elapsed", 12.5)

        # A TensorBoard scalar summary is a `Float32`, so these come back narrowed and the
        # literals below say so rather than hiding it behind an approximate comparison.
        seen = Dict{String, Tuple{Int, Float32}}()
        map_summaries((n, s, v) -> (seen[n] = (s, v)), logdir(lg))

        # Every context prefixes its own tags, so `train/loss` and `validate/val_loss` are
        # separate series rather than one series whose meaning depends on the caller.
        @test seen["train/loss"] == (4, 0.31f0)
        @test seen["train/acc"] == (4, 0.9f0)
        @test seen["validate/val_loss"] == (8, 0.29f0)
        @test seen["data/data_wait_frac"] == (8, 0.01f0)
        # The epoch rides along in each context, which is what lets a curve be read back against
        # epoch without the run directory.
        @test seen["train/epoch"] == (4, 1.0f0)
        @test seen["validate/epoch"] == (8, 1.0f0)
        @test seen["data/epoch"] == (8, 1.0f0)
        # A numeric `log_other!` is a scalar, at the last step a metric was logged at.
        @test seen["other/elapsed"] == (8, 12.5f0)

        # THE POINT OF THE WHOLE DESIGN: the driver owns the step. Nothing here goes through
        # `handle_message`, so the logger's own global counter never moves and cannot disagree
        # with the step in the checkpoint record.
        @test TensorBoardLogger.step(lg) == 0
    end

    @testset "a non-Real metric is a loud error naming the key" begin
        lg = TBLogger(joinpath(mktempdir(), "run"), tb_overwrite)
        err = try
            log_metrics!(lg, (; note = "text"); step = 1, epoch = 1, context = "train")
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("note::String", err.msg)
        @test occursin("\"train\"", err.msg)
    end

    # ── Text: params, tags, the report, the table, and the HParams record ───────────

    @testset "the text verbs land in the event file, markdown intact" begin
        dir = joinpath(mktempdir(), "run")
        lg = TBLogger(dir, tb_overwrite)

        log_params!(lg, (; seed = 42, max_epochs = 2, preset = :fast, flag = true))
        log_metrics!(lg, (; loss = 0.5); step = 3, epoch = 1, context = "train")
        log_tags!(lg, (; kind = "probe"))
        log_tags!(lg, ["alpha", "beta"])
        log_tags!(lg, "solo")
        log_other!(lg, "binding_report", "line one\nline two")
        log_confusion!(lg, [3 1; 0 4], ["a", "b"]; epoch = 1)
        finish!(lg, :early_stop)

        events = filter(f -> startswith(basename(f), "events."), readdir(dir; join = true))
        blob = String(read(only(events)))

        @testset "parameters, written immediately so a compile crash keeps them" begin
            @test occursin("params", blob)
            @test occursin("**seed**: 42", blob)
            @test occursin("**preset**: fast", blob)
        end

        @testset "tags, in all three shapes the verb accepts" begin
            @test occursin("**kind**: probe", blob)
            @test occursin("- alpha", blob) && occursin("- beta", blob)
            @test occursin("solo", blob)
        end

        # The regression that motivated the `_Markdown` wrapper: TensorBoardLogger renders a
        # `String` through `repr(MIME"text/plain"(), s)`, which quotes it and turns the newline
        # into a literal backslash-n. A binding report is multi-line and must survive.
        @testset "multi-line text is not escaped into one line" begin
            @test occursin("other/binding_report", blob)
            @test occursin("line one\nline two", blob)
            @test !occursin("line one\\nline two", blob)
        end

        @testset "the confusion matrix is a markdown table" begin
            @test occursin("confusion", blob)
            @test occursin("rows are true and columns are predicted", blob)
            @test occursin("| **a** | 3 | 1 |", blob)
            @test occursin("| **b** | 0 | 4 |", blob)
        end

        # The reason this file keeps per-logger state at all: `write_hparams!` wants the
        # hyperparameters and the metric tags in one call, and the contract logs parameters
        # before any metric name exists.
        @testset "finish! writes the deferred HParams record and the status" begin
            @test occursin("_hparams_/experiment", blob)
            @test occursin("_hparams_/session_start_info", blob)
            # The tag collected from the run, not from the parameters.
            @test occursin("train/loss", blob)
            @test occursin("run/status", blob)
            @test occursin("early_stop", blob)
        end
    end

    @testset "a confusion matrix that cannot be a table says so" begin
        lg = TBLogger(joinpath(mktempdir(), "run"), tb_overwrite)
        err = try
            log_confusion!(lg, [3 1 0; 0 4 1], ["a", "b"]; epoch = 1)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("2x3", err.msg) && occursin("2 labels", err.msg)
    end

    # ── The informational and resumption verbs ──────────────────────────────────────

    @testset "an event-file backend has a directory, not a hosted run" begin
        dir = joinpath(mktempdir(), "run")
        lg = TBLogger(dir, tb_overwrite)

        @test run_id(lg) === nothing
        @test run_url(lg) === nothing
        @test logger_info(lg).logdir == logdir(lg)
        @test logger_info(lg).step == 0
        @test logger_state(lg) == (; logdir = logdir(lg))
        # The contract's rule, checked: state must be plain serializable data.
        @test ReactantNitro.check_logger_state_serializable(logger_state(lg), lg) === nothing
    end

    # The failure this pair exists to catch: `TBLogger(dir)` defaults to `tb_increment`, so a
    # resume constructing its logger the obvious way lands in `dir_1` and splits one run's
    # history across two entries in TensorBoard's run list, with no error anywhere.
    @testset "reattach! refuses a resume that would write somewhere else" begin
        dir = joinpath(mktempdir(), "run")
        lg = TBLogger(dir, tb_overwrite)

        @test reattach!(lg, logger_state(lg)) === nothing

        err = try
            reattach!(lg, (; logdir = logdir(lg) * "_1"))
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin(logdir(lg), err.msg)
        @test occursin(logdir(lg) * "_1", err.msg)
        @test occursin("tb_append", err.msg)
    end

    @testset "the increment default is real, and reattach! catches it" begin
        base = joinpath(mktempdir(), "run")
        first_run = TBLogger(base, tb_overwrite)
        state = logger_state(first_run)

        # No policy passed, exactly as a resume would do it. The directory already exists.
        resumed = TBLogger(base)
        @test logdir(resumed) != state.logdir
        @test_throws ErrorException reattach!(resumed, state)

        # And the fix the error message names.
        appended = TBLogger(base, TensorBoardLogger.tb_append)
        @test logdir(appended) == state.logdir
        @test reattach!(appended, state) === nothing
    end
end
