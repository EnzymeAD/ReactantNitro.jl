# test/kaimon_ext.jl
#
# The ReactantNitroKaimonGateExt extension: the `nitro_*` GateTools it registers with a running Kaimon gate,
# and the run registry those tools are made of.
#
# The extension only activates when KaimonGate is loaded alongside ReactantNitro. `runtests.jl`
# loads ReactantNitro at the top, so the `using KaimonGate` below is what triggers it; the
# extension must be fetched with `Base.get_extension` because an extension module is not a
# loadable package.
#
# These tests drive the tool functions directly and exercise registration against a real gate
# started with `force = true` (headless processes skip `serve` otherwise). The background-task
# machinery, the registry, the experiment resolution, and the kwargs contract are all
# gate-independent and are exactly what the tools are made of; the live Kaimon session behavior
# is covered by the docs page and a manual smoke test outside the suite.
#
# The training runs below are deliberately tiny (a few batches, one or two epochs), like the
# rest of the CPU suite; the framework warns about the data path being idle during compile,
# which is the expected shape of a short run and is not an error.

@testitem "kaimon_ext" begin
    using Test
    using KaimonGate
    using ReactantNitro
    using Reactant
    using ReactantNitro: @experiment, ExportBackend, read_manifest
    using JSON3, Lux, Random, Statistics

    const EXT = Base.get_extension(ReactantNitro, :ReactantNitroKaimonGateExt)

    # ── The experiment ───────────────────────────────────────────────────────────────────
    #
    # One tiny experiment serves every tool: `build_data` ships all three splits, and the export
    # hooks make `export_model` reachable. The `width` field exercises the `overrides` path; the
    # unmarked `max_epochs` is a Host field like the tutorial's.

    @experiment struct GateMLP
        "Hidden width. Structural: changes the compiled graph."
        width::GraphConst{Int} = 4

        "Epochs to train for. Driver-only."
        max_epochs::Int = 2
    end

    ReactantNitro.build_model(e::GateMLP, rng) = begin
        m = Lux.Chain(Lux.Dense(3 => e.width, tanh), Lux.Dense(e.width => 1))
        (m, Lux.setup(rng, m)...)
    end
    ReactantNitro.forward(::GateMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::GateMLP, y; yhat) = mean(abs2, y .- yhat)

    gate_batches(n; seed) = (
        rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, Float32, 3, 4), yhat = randn(rng, Float32, 1, 4)) for _ in 1:n]
    )
    ReactantNitro.build_data(::GateMLP, dist) = (;
        train = gate_batches(4; seed = 1),
        val = gate_batches(2; seed = 2),
        test = gate_batches(2; seed = 3),
    )

    # The ReactantNitroKaimonGateExt resolves an experiment by NAME, the way a session host does when a model
    # package is loaded (`using MyModels`, then `MyModels.MyExperiment`). This @testitem runs in a
    # generated module, so the honest "model package" is that module itself: it is bound in Main
    # under its gensym name, and the resolver walks Main -> module -> struct exactly like a real
    # `Main.MyModels.MyExperiment` spec.
    const GATE_SPEC = string(nameof(@__MODULE__)) * ".GateMLP"
    ReactantNitro.export_inputs(::GateMLP) = [ReactantNitro.ExportSpec("x", Float32, [3, 1])]
    ReactantNitro.export_outputs(::GateMLP) = [ReactantNitro.ExportSpec("y", Float32, [1, 1])]

    # ── An experiment whose exportable handle is a different build ───────────────────────
    #
    # A real model's shape in miniature, and the reason `nitro_export` grew a `data` keyword. A
    # `Host` flag selects between a trainable program and a servable one, and each build refuses
    # the other's job in its own words:
    #
    #   * `build_data` refuses the flag, because an inference program has no arm for `loss` to
    #     score and a run that reached the first optimizer step with it set would fail there.
    #   * the first export hook refuses its absence, because the trainable program wants ground
    #     truth on the wire, which no client has, and exporting it would say so nowhere.
    #
    # Together those are a pincer on any caller that must build data to construct a handle, which
    # is what this tool was until the `data` keyword. `build_data` counts its calls so a test can
    # assert it was SKIPPED rather than infer it from an export that happened to work.
    @experiment struct ExportOnly
        "Build the inference program instead of the trainable one. Host: not part of the checksum."
        export_inference::Bool = false

        "Epochs to train for."
        max_epochs::Int = 1
    end

    const EXPORT_ONLY_SPEC = string(nameof(@__MODULE__)) * ".ExportOnly"
    const EXPORT_ONLY_BUILT = Ref(0)

    ReactantNitro.build_model(e::ExportOnly, rng) = begin
        m = Lux.Chain(Lux.Dense(3 => 2, tanh), Lux.Dense(2 => 1))
        (m, Lux.setup(rng, m)...)
    end
    ReactantNitro.forward(::ExportOnly, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::ExportOnly, y; yhat) = mean(abs2, y .- yhat)

    function ReactantNitro.build_data(e::ExportOnly, dist)
        EXPORT_ONLY_BUILT[] += 1
        e.export_inference && error(
            "ExportOnly: export_inference builds the image-only INFERENCE model, which has no \
             teacher-forced arms for `loss` to score, so it cannot train."
        )
        return (; train = gate_batches(2; seed = 11), val = gate_batches(1; seed = 12))
    end

    function ReactantNitro.export_inputs(e::ExportOnly)
        e.export_inference || error(
            "ExportOnly: this handle was built without `export_inference = true`, so its model is \
             the teacher-forced training program."
        )
        return [ReactantNitro.ExportSpec("x", Float32, [3, 1])]
    end
    ReactantNitro.export_outputs(::ExportOnly) = [ReactantNitro.ExportSpec("y", Float32, [1, 1])]

    # ── The recording export backend ─────────────────────────────────────────────────────
    #
    # The suite's export test uses a fake backend rather than a real StableHLO bundle (a bundle
    # needs the backend package, a real trace, and a server to load it; none belong in a CPU
    # suite). The tool's backend registry is how a fake gets in: the tool resolves the backend
    # by name, so the test registers one that records what `export_model` handed it.

    struct RecordingBundle <: ExportBackend end

    const RECORDED = Ref{Any}(nothing)
    function ReactantNitro.write_export(
            ::RecordingBundle, program, ps, st, example_inputs; kwargs...
        )
        RECORDED[] = kwargs
        return joinpath(String(kwargs[:dir]), String(kwargs[:name]))
    end

    # ── A backend whose methods arrive when its factory runs ─────────────────────────────
    #
    # The world-age regression, as a backend rather than as a package. `_reactant_server_backend`
    # loads ReactantServerExport with `Base.require` when an export names that backend, and the
    # extension's `site_provenance` and `write_export` methods only exist after that load. The load
    # raises the world counter, and a method added after a task STARTED is invisible to it, so an
    # export that resolved its backend from inside `_launch_run!`'s task dispatched to the erroring
    # generic `site_provenance(::ExportBackend, root)` and failed before tracing anything.
    #
    # The mechanism is the counter, not that package, so this reproduces it with a factory that
    # defines its own methods. `Core.eval` raises the counter exactly as loading a package does,
    # and the suite needs no backend package to prove it.
    struct LateBundle <: ExportBackend end

    const LATE_DEFINED = Ref(false)
    const LATE_RECORDED = Ref{Any}(nothing)

    function define_late_methods!()
        LATE_DEFINED[] && return LateBundle()
        Core.eval(
            @__MODULE__, quote
                function ReactantNitro.write_export(
                        ::LateBundle, program, ps, st, example_inputs; kwargs...
                    )
                    LATE_RECORDED[] = kwargs
                    return joinpath(String(kwargs[:dir]), String(kwargs[:name]))
                end
                ReactantNitro.site_provenance(::LateBundle, root) =
                    Dict{String, Any}("git_commit" => "late", "site_root" => String(root))
            end
        )
        LATE_DEFINED[] = true
        return LateBundle()
    end

    # ── Helpers ──────────────────────────────────────────────────────────────────────────

    """Poll the registry until `id` leaves `:running`, or fail after `timeout` seconds."""
    function wait_run(id; timeout = 240.0)
        deadline = time() + timeout
        while time() < deadline
            s = EXT._get_run(id)
            s.status !== :running && return s
            sleep(0.5)
        end
        return error("run $id did not finish within $(timeout)s; status=$(EXT._get_run(id).status)")
    end

    run_id(msg) = String(match(r"run ([0-9a-f]{8})", msg).captures[1])

    # ── The tests ────────────────────────────────────────────────────────────────────────

    @testset "the extension loads and builds the tool set" begin
        @test EXT !== nothing
        @test EXT isa Module
        names = [t.name for t in EXT._build_tools()]
        @test names == [
            "nitro_setup", "nitro_train", "nitro_validate", "nitro_evaluate", "nitro_predict",
            "nitro_export", "nitro_runs", "nitro_status", "nitro_logger", "nitro_stop",
        ]
        # The MCP schema comes from reflection: `experiment` must be a required string positional
        # and the run knobs typed optional kwargs, which is what an agent sees in tools/list.
        train = only(t for t in EXT._build_tools() if t.name == "nitro_train")
        meta = KaimonGate._reflect_tool(train)
        @test meta["name"] == "nitro_train"
        ex = only(a for a in meta["arguments"] if a["name"] == "experiment")
        @test ex["required"] == true
        @test ex["type_meta"]["kind"] == "string"
        me = only(a for a in meta["arguments"] if a["name"] == "max_epochs")
        @test me["required"] == false
        @test me["is_kwarg"] == true
        @test me["type_meta"]["kind"] == "integer"
        # nitro_setup takes only optional kwargs: it must reflect as a valid tool with none required.
        setup_tool = only(t for t in EXT._build_tools() if t.name == "nitro_setup")
        smeta = KaimonGate._reflect_tool(setup_tool)
        @test smeta["name"] == "nitro_setup"
        @test all(!a["required"] for a in smeta["arguments"])
        # `data` reaches the SCHEMA, which is the surface the whole keyword exists for: an agent
        # that cannot see it in tools/list writes a hand-rolled `export_model` call instead, and
        # keeps writing one for as long as the keyword stays invisible. The description carries the
        # docstring, so the default is discoverable without reading the extension.
        xmeta = KaimonGate._reflect_tool(
            only(t for t in EXT._build_tools() if t.name == "nitro_export")
        )
        dat = only(a for a in xmeta["arguments"] if a["name"] == "data")
        @test dat["required"] == false
        @test dat["is_kwarg"] == true
        @test dat["type_meta"]["kind"] == "string"
        @test occursin("does not build the training data", xmeta["description"])
    end

    @testset "nitro_setup / setup_devices!" begin
        # The suite runs on CPU, which reports exactly one visible device.
        @test length(Reactant.devices()) == 1
        # A report-only call changes nothing and reports the default (every visible device).
        r0 = EXT.nitro_setup()
        @test occursin("backend=cpu", r0) && occursin("n_devs=1 of 1", r0)
        @test occursin("default (every visible device)", r0)
        @test ReactantNitro._PINNED_N_DEVS[] === nothing
        # Pin a device count; the accessor now returns the pin, beating the experiment field.
        r1 = EXT.nitro_setup(n_devs = 1)
        @test occursin("pinned via nitro_setup", r1)
        @test ReactantNitro._PINNED_N_DEVS[] == 1
        @test ReactantNitro.n_devs(GateMLP()) == 1   # GateMLP has no n_devs field
        @experiment struct PinnedExp
            n_devs::Int = 7
        end
        @test ReactantNitro.n_devs(PinnedExp()) == 1 # pin beats the field
        # The framework function is the same code path as the tool.
        cfg = ReactantNitro.setup_devices!(n_devs = 1)
        @test cfg.n_devs == 1 && cfg.pinned && cfg.visible == 1 && cfg.backend == "cpu"
        # Backend selection works and errors are loud.
        cfg2 = ReactantNitro.setup_devices!(backend = "cpu")
        @test cfg2.backend == "cpu"
        @test_throws ArgumentError ReactantNitro.setup_devices!(backend = "toaster")
        # Asking for more devices than are visible errors, and names CUDA_VISIBLE_DEVICES.
        @test_throws ErrorException EXT.nitro_setup(n_devs = 99)
        err = try
            ReactantNitro.setup_devices!(n_devs = 99)
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("CUDA_VISIBLE_DEVICES", err)
        # Unpin so later test files (visualize.jl) run under the framework default.
        ReactantNitro._PINNED_N_DEVS[] = nothing
        @test ReactantNitro._PINNED_N_DEVS[] === nothing
    end

    @testset "experiment resolution" begin
        # GATE_SPEC is the qualified form a session host passes (`Main.<module>.<Experiment>`);
        # the unqualified `GateMLP` form is gone here because the @testitem runs in its own module,
        # so nothing user-defined is bound in Main anymore (which is the point of the isolation).
        @test EXT._resolve_experiment(GATE_SPEC) === GateMLP
        @test_throws ErrorException EXT._resolve_experiment("NoSuchModule.GateMLP")
        # Resolves but is not a type: a function, say.
        @test_throws ErrorException EXT._resolve_experiment("Main.sin")
        @test_throws ErrorException EXT._resolve_experiment("")
    end

    @testset "the overrides parser" begin
        @test EXT._parse_overrides(nothing) == Dict{Symbol, Any}()
        @test EXT._parse_overrides("width=8") == Dict(:width => 8)
        d = EXT._parse_overrides("width=8, smoothing=0.05, name=\"x\", on=true")
        @test d == Dict(:width => 8, :smoothing => 0.05, :name => "x", :on => true)
        @test EXT._parse_overrides("labels=[1.0, 2.0, 3.0]")[:labels] == [1.0, 2.0, 3.0]
        # A comma inside a quoted string is not a separator.
        @test EXT._parse_overrides("name=\"a,b\"")[:name] == "a,b"
        # A matrix literal (for a predict batch, say) parses as a matrix.
        m = EXT._parse_overrides("m=[1.0 2.0; 3.0 4.0]")[:m]
        @test m isa Matrix{Float64}
        @test size(m) == (2, 2)
        # A non-literal value is refused, which is the safety property of the parser. A bare
        # identifier is a Symbol literal, accepted here and caught later by the declared-type
        # conversion if it does not fit the field.
        @test_throws ErrorException EXT._parse_overrides("x=sin(1)")
        @test EXT._parse_overrides("x=foo")[:x] === :foo
        @test_throws ErrorException EXT._parse_overrides("justaname")
        @test_throws ErrorException EXT._parse_overrides("=5")
    end

    @testset "kwargs validation" begin
        # Override keys are checked against the experiment's fields, and converted to the declared
        # type, exactly like `from_preset` validates.
        o = EXT._check_overrides(GateMLP, Dict(:width => 8), Pair{Symbol, Any}[])
        @test o == Dict(:width => 8)
        @test_throws ErrorException EXT._check_overrides(
            GateMLP, Dict(:nope => 1), Pair{Symbol, Any}[]
        )
        # A key passed both as a run keyword and in `overrides` is an ambiguity error.
        @test_throws ErrorException EXT._check_overrides(
            GateMLP, Dict(:max_epochs => 3), [:max_epochs => 2]
        )
        # The declared-type conversion: a Float64 override for a GraphConst{Int} field converts.
        # (The dict must be `Dict{Symbol,Any}` as `_parse_overrides` returns; a narrower value type
        # would convert the converted value back on store.)
        @test EXT._check_overrides(GateMLP, Dict{Symbol, Any}(:width => 8.0), Pair{Symbol, Any}[])[:width] === 8
        @test EXT._parse_resume("auto") === :auto
        @test EXT._parse_resume("false") === false
        @test EXT._parse_resume("runs/x/epoch-0003.jld2") == "runs/x/epoch-0003.jld2"
    end

    # KaimonGate's registered-tool list is PRIVATE and has moved: `_SESSION_TOOLS::Ref{Vector}`
    # before the GateSession refactor, the accessor `_session_tools()` after it. Probe both, for
    # the same reason the extension does: pinning a test to one release's internals is how this
    # suite came to fail with `UndefVarError: _SESSION_TOOLS` against a KaimonGate the compat
    # bound admits (1.4.0 under `[compat] KaimonGate = "1.2.0"`), while reporting nothing about
    # whether registration itself still worked.
    gate_tools() = isdefined(KaimonGate, :_session_tools) ? KaimonGate._session_tools() :
        KaimonGate._SESSION_TOOLS[]

    @testset "registration against a live gate" begin
        # A gate needs an interactive session to start by default; `force = true` is the documented
        # override for non-interactive processes, which is what lets the suite exercise the one
        # mechanism the extension exists for.
        KaimonGate.serve(force = true, allow_mirror = false)
        try
            @test EXT._gate_running()
            @test EXT._install_tools() == true
            names = [t.name for t in gate_tools()]
            @test "nitro_train" in names
            @test "nitro_export" in names
            # Kaimon's session boot calls `serve()` with no tools AFTER the extension may have
            # registered; that call must keep the tools rather than wipe them.
            KaimonGate.serve(force = true, allow_mirror = false)
            @test length(gate_tools()) == 10
            @test EXT.reinstall_kaimon_tools() == true
        finally
            KaimonGate.stop()
        end
    end

    @testset "registration is automatic in a non-interactive host" begin
        # The regression this guards. The testset above calls `_install_tools` directly, so it proves
        # the registration MECHANISM works while never entering through `__init__`, which is the only
        # path a real gate uses. That path used to bail on `isinteractive()`, false in the
        # `julia -e '<preamble>'` process every gate host runs, so the retry loop covering the
        # load-before-serve ordering was never armed and a live gate registered nothing, silently.
        #
        # The guard was a property of the process itself, so the honest test is a subprocess that
        # reproduces the host's ordering: KaimonGate first, ReactantNitro second (its `__init__` fires
        # here, with no gate serving yet), `serve` last. The child reports through a file rather than
        # stdout, which a serving gate also writes to.
        out = joinpath(mktempdir(), "probe.txt")
        probe = raw"""
            record(k, v) = open(io -> println(io, k, "=", v), ENV["NITRO_PROBE_OUT"], "a")

            using KaimonGate
            using ReactantNitro          # fires ReactantNitroKaimonGateExt.__init__; no gate is serving yet

            # The parent's `gate_tools()` helper does not exist in here: this is a separate
            # process running a self-contained script. Same probe, restated, for the same reason
            # (KaimonGate's tool list is private and moved from `_SESSION_TOOLS[]` to
            # `_session_tools()`).
            gate_tools() = isdefined(KaimonGate, :_session_tools) ? KaimonGate._session_tools() :
                KaimonGate._SESSION_TOOLS[]

            record("interactive", isinteractive())
            record("at_load", length(gate_tools()))

            # Bounded, and a function rather than a top-level loop: an assignment to a global from
            # inside a loop is soft scope, which does not mean in a script what it means in a REPL.
            function wait_for_tools(tries)         # the extension's own retry window is 30 seconds
                for _ in 1:tries
                    length(gate_tools()) > 0 && break
                    sleep(1.0)
                end
                return length(gate_tools())
            end

            # The gate binds AFTER the extension loaded, as the host's preamble does it.
            KaimonGate.serve(force = true, allow_mirror = false)
            record("after_serve", wait_for_tools(60))
            KaimonGate.stop()
        """
        cmd = addenv(
            `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $probe`,
            "NITRO_PROBE_OUT" => out,
            "CUDA_VISIBLE_DEVICES" => "",
        )
        # The child's stderr is left attached so a failure names itself in the suite's output.
        @test success(pipeline(cmd, stdout = devnull))
        # `isfile` so a child that died before recording anything fails the assertions below rather
        # than erroring out of the testset and taking their diagnosis with it.
        got = Dict{String, String}()
        isfile(out) && for line in eachline(out)
            k, v = split(line, "=", limit = 2)
            got[k] = v
        end
        # If this reads `true` the subprocess was not the process shape the guard was about, and the
        # rest of the testset proves nothing.
        @test get(got, "interactive", "<missing>") == "false"
        @test get(got, "at_load", "<missing>") == "0"
        @test get(got, "after_serve", "<missing>") == "10"
    end

    @testset "a train run's lifecycle" begin
        dir = mktempdir()
        msg = EXT.nitro_train(GATE_SPEC, max_epochs = 2, run_dir = dir)
        id = run_id(msg)
        s = wait_run(id)
        @test s.status == :completed
        @test s.kind == :train
        @test s.experiment == GATE_SPEC
        # The recording logger and phase monitor filled in the live numbers.
        @test s.loss !== nothing
        @test s.val_metrics !== nothing
        @test s.epoch == 2
        @test s.step == 8
        @test s.run_dir == dir
        # Two epochs of checkpoints, the framework's own resume story. The checkpoint filename
        # carries a metric value, so the file is resolved through the manifest rather than reconstructed.
        @test isfile(joinpath(dir, only(e.file for e in read_manifest(dir) if e.epoch == 2)))
        @test occursin("stop_reason=completed", s.result)
        text = EXT.nitro_status(id)
        @test occursin("status=completed", text)
        @test occursin("latest train loss", text)
        @test occursin(dir, text)
        @test occursin(id, EXT.nitro_runs())
        # Re-reading the completed run's Nitro via the registry is how the eval tools reuse it.
        @test s.nitro !== nothing
        # The machine-readable face, for a wait predicate that must not parse prose.
        st = ReactantNitro.run_state(id)
        @test st.status === :completed && st.kind === :train && st.epoch == 2 && st.step == 8
        @test st.run_dir == dir && st.loss !== nothing && st.val_metrics !== nothing
        @test st.elapsed_s >= 0 && occursin("stop_reason=completed", st.result)
        @test_throws ErrorException ReactantNitro.run_state("no-such-run")
    end

    @testset "nitro_logger reports the run's logger, live" begin
        dir = mktempdir()
        msg = EXT.nitro_train(GATE_SPEC, max_epochs = 1, run_dir = dir)
        id = run_id(msg)
        s = wait_run(id)
        @test s.status == :completed

        # GateMLP declares no logger, so the run got the framework's JSON default, wrapped in the
        # recording logger and adopted to the run's directory.
        text = EXT.nitro_logger(id)
        @test occursin("logger=JSONLogger", text)
        @test occursin(joinpath(dir, "metrics.jsonl"), text)

        # The wrapped default really wrote into the run directory.
        @test isfile(joinpath(dir, "metrics.jsonl"))
        ls = [JSON3.read(ln) for ln in eachline(joinpath(dir, "metrics.jsonl"))]
        @test first(ls)["type"] == "params"

        # `nitro_status` carries the same one-liner.
        @test occursin("logger: JSONLogger", EXT.nitro_status(id))
        @test occursin(dir, EXT.nitro_status(id))
    end

    @testset "the recording logger forwards logger info and the default's adoption" begin
        # Setup's adoption reaches a WRAPPED pathless default: the tools wrap `logger(e)` in the
        # recording logger, and setup pins the wrapped default's path through it.
        inner = ReactantNitro.JSONLogger()
        wrapped = EXT.RecordingLogger(EXT.RunState("zz999999", :train, "x"), inner)
        ReactantNitro._adopt_logger!(wrapped, "/tmp/kaimon_adopt")
        @test inner.path == joinpath("/tmp/kaimon_adopt", "metrics.jsonl")
        # `logger_info` forwards to the inner logger's own table, and an absent inner says nothing.
        @test ReactantNitro.logger_info(wrapped) == (; path = inner.path)
        bare = EXT.RecordingLogger(EXT.RunState("zz888888", :train, "x"), nothing)
        @test ReactantNitro.logger_info(bare) == (;)
        @test ReactantNitro.run_id(bare) === nothing
        @test ReactantNitro.run_url(bare) === nothing
    end

    @testset "validate, evaluate, predict, export" begin
        dir = mktempdir()
        train_msg = EXT.nitro_train(GATE_SPEC, max_epochs = 1, run_dir = dir)
        train_id = run_id(train_msg)
        wait_run(train_id)

        # validate reuses the trained Nitro through the registry.
        v_msg = EXT.nitro_validate(run_id = train_id)
        v_id = run_id(v_msg)
        v = wait_run(v_id)
        @test v.status == :completed
        @test occursin("split=val", v.result)
        @test v.val_metrics !== nothing

        # evaluate builds a fresh Nitro from a checkpoint file.
        ckpt = joinpath(dir, only(e.file for e in read_manifest(dir) if e.epoch == 1))
        # `run_dir = dir` keeps this fresh construction (which writes its default logger's file)
        # inside the test's temp directory rather than the suite's working directory.
        e_msg = EXT.nitro_evaluate(
            experiment = GATE_SPEC, split = "test", checkpoint = ckpt, run_dir = dir
        )
        e_id = run_id(e_msg)
        e = wait_run(e_id)
        @test e.status == :completed
        @test occursin("split=test", e.result)

        # predict takes a batch as `name=value` literals; the batch axis is last.
        p_msg = EXT.nitro_predict(
            run_id = train_id,
            inputs = "x=[1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0; 9.0 10.0 11.0 12.0]",
        )
        p_id = run_id(p_msg)
        p = wait_run(p_id)
        @test p.status == :completed
        @test occursin("array(Float32", p.result)
        @test occursin("size 1x4", p.result)
        # The phase monitor is attached to predict runs too, so the completed run's last published
        # event is `Repl` (the `with_repl` bookend, which fires before status flips to `:completed`).
        @test p.phase == "Repl"

        # export through the recording backend, from the trained run.
        EXT._register_backend!("recording", () -> RecordingBundle())
        x_msg = EXT.nitro_export(
            run_id = train_id, dir = dir, name = "gate_v1",
            backend = "recording", batch_sizes = "[1, 4]",
        )
        x_id = run_id(x_msg)
        x = wait_run(x_id)
        @test x.status == :completed
        @test occursin("exported to", x.result)
        # The phase monitor is attached to export runs, and `export_model` publishes
        # `ExportCompiling` around the backend call: the run's recorded phase ends at the `Repl`
        # bookend, proving the status surface saw the export rather than the pre-fix silence.
        @test x.phase == "Repl"
        @test RECORDED[] !== nothing
        @test RECORDED[][:dir] == dir
        @test RECORDED[][:name] == "gate_v1"
        @test RECORDED[][:batch_sizes] == [1, 4]
        # Provenance carries the framework's own entry, so the recording got the real thing.
        @test RECORDED[][:provenance]["framework"] == "ReactantNitro.jl"

        # export from `experiment` + checkpoint constructs a single-device handle automatically.
        x2_msg = EXT.nitro_export(
            experiment = GATE_SPEC, checkpoint = ckpt, run_dir = dir,
            dir = dir, name = "gate_v2", backend = "recording",
        )
        x2_id = run_id(x2_msg)
        x2 = wait_run(x2_id)
        @test x2.status == :completed
        # A multi-device request from `experiment` is refused up front.
        @test_throws ErrorException EXT.nitro_export(
            experiment = GATE_SPEC, n_devs = 4,
            dir = dir, name = "gate_v3", backend = "recording",
        )
    end

    # ── The world-age regression ─────────────────────────────────────────────────────────
    #
    # An export whose backend's methods arrive with the backend must still export. Against the
    # pre-fix extension it fails in a way that is easy to misread: the run reaches `:failed` with
    # "has no `site_provenance` method" and no bundle, and the identical relaunch then succeeds
    # because the session has the method by then. Both halves of the fix are covered:
    # `nitro_export` resolves the backend before it launches, and `_launch_run!` runs the body at
    # the latest world so a method arriving later than the task still dispatches.
    #
    # `provenance_root` is not incidental here. `export_model` calls `site_provenance` only when a
    # root is given, so an export without one never reaches the method the load provides, which is
    # exactly why this hid until somebody asked for a bundle they could trace.
    @testset "a backend whose methods load late still exports" begin
        dir = mktempdir()
        train_id = run_id(EXT.nitro_train(GATE_SPEC, max_epochs = 1, run_dir = dir))
        wait_run(train_id)
        ckpt = joinpath(dir, only(e.file for e in read_manifest(dir) if e.epoch == 1))

        EXT._register_backend!("late", define_late_methods!)
        # Nothing has defined them yet: the erroring generic is what a call would dispatch to.
        @test LATE_DEFINED[] == false
        @test LATE_RECORDED[] === nothing
        @test_throws ErrorException ReactantNitro.site_provenance(LateBundle(), dir)

        x = wait_run(
            run_id(
                EXT.nitro_export(
                    experiment = GATE_SPEC, checkpoint = ckpt, run_dir = dir,
                    dir = dir, name = "late_v1", backend = "late", provenance_root = dir,
                )
            )
        )
        @test x.status == :completed
        @test LATE_DEFINED[] == true
        # The method the load provided is the one that ran, and its dictionary reached the bundle:
        # dispatch did not fall through to the generic, and the site layer is really in there.
        @test LATE_RECORDED[] !== nothing
        @test LATE_RECORDED[][:provenance]["git_commit"] == "late"
        @test LATE_RECORDED[][:provenance]["site_root"] == dir
        # And the framework layer is still under it, so the late method merged rather than replaced.
        @test LATE_RECORDED[][:provenance]["framework"] == "ReactantNitro.jl"
    end

    # ── `data`: the keyword that made an inference-only export expressible ───────────────
    #
    # `Nitro(e; checkpoint = path, data = (;))` is the construction `export_model` prescribes, and
    # this tool once could not express it: `build_data` always ran. A model whose exportable handle
    # is a different build from its trainable one was therefore not exportable through the tool in
    # either direction, which pushes every export back onto a hand-written `export_model` call.
    # `ExportOnly` below is that shape in miniature: `build_data` refuses the handle when the flag
    # is set, and the export hook refuses it when the flag is clear.
    @testset "data: an experiment export skips build_data" begin
        dir = mktempdir()
        EXT._register_backend!("recording", () -> RecordingBundle())

        # A `build_data` that records whether it ran, so "skipped" is asserted and not assumed.
        @test EXT._push_export_data!(Pair{Symbol, Any}[], nothing, nothing, "X") ==
            [:data => (;)]
        @test EXT._push_export_data!(Pair{Symbol, Any}[], "build", nothing, "X") ==
            Pair{Symbol, Any}[]
        @test EXT._push_export_data!(Pair{Symbol, Any}[], "none", nothing, "X") ==
            [:data => (;)]
        # With `run_id` the handle is already built, so the keyword is refused rather than ignored.
        @test_throws ErrorException EXT._push_export_data!(
            Pair{Symbol, Any}[], "none", "abc12345", nothing
        )
        @test EXT._push_export_data!(Pair{Symbol, Any}[], nothing, "abc12345", nothing) ==
            Pair{Symbol, Any}[]
        # A value that is neither is the caller's mistake, named in the caller's answer.
        @test_throws ErrorException EXT._push_export_data!(
            Pair{Symbol, Any}[], "skip", nothing, "X"
        )
        @test_throws ErrorException EXT.nitro_export(
            experiment = GATE_SPEC, dir = dir, name = "bad_data",
            backend = "recording", data = "maybe",
        )

        # End to end, on the pincer shape itself.
        train_id = run_id(EXT.nitro_train(EXPORT_ONLY_SPEC, max_epochs = 1, run_dir = dir))
        wait_run(train_id)
        ckpt = joinpath(dir, only(e.file for e in read_manifest(dir) if e.epoch == 1))

        EXPORT_ONLY_BUILT[] = 0
        x = wait_run(
            run_id(
                EXT.nitro_export(
                    experiment = EXPORT_ONLY_SPEC, checkpoint = ckpt, run_dir = dir,
                    dir = dir, name = "inference_v1", backend = "recording",
                    overrides = "export_inference=true",
                )
            )
        )
        @test x.status == :completed
        # `build_data` never ran, which is why the flag could be set at all.
        @test EXPORT_ONLY_BUILT[] == 0

        # `data = "build"` is the escape hatch, and on THIS experiment it puts the pincer back:
        # `build_data` runs, refuses the inference handle, and the run fails with its own words.
        # Proving the hatch reaches `build_data` matters more here than a passing export would.
        x2 = wait_run(
            run_id(
                EXT.nitro_export(
                    experiment = EXPORT_ONLY_SPEC, checkpoint = ckpt, run_dir = dir,
                    dir = dir, name = "inference_v2", backend = "recording",
                    overrides = "export_inference=true", data = "build",
                )
            )
        )
        @test x2.status == :failed
        @test occursin("cannot train", x2.error)
        @test EXPORT_ONLY_BUILT[] == 1

        # And with the flag clear the export hook refuses the handle: the other jaw, so the test
        # shows there was no way through the tool before `data` existed.
        x3 = wait_run(
            run_id(
                EXT.nitro_export(
                    experiment = EXPORT_ONLY_SPEC, checkpoint = ckpt, run_dir = dir,
                    dir = dir, name = "inference_v3", backend = "recording",
                )
            )
        )
        @test x3.status == :failed
        @test occursin("teacher-forced", x3.error)
    end

    @testset "stop is graceful" begin
        dir = mktempdir()
        msg = EXT.nitro_train(GATE_SPEC, max_epochs = 1000, run_dir = dir)
        id = run_id(msg)
        sleep(2.0)
        reply = EXT.nitro_stop(id)
        @test occursin("stop requested", reply)
        s = wait_run(id)
        @test s.status == :completed
        @test occursin("stop_reason=requested", s.result)
        # The graceful stop still finished an epoch and checkpointed before exiting.
        @test s.epoch >= 1
        @test s.step >= 4
        # Stopping a finished run is a no-op answer, not an error.
        @test occursin("already", EXT.nitro_stop(id))
    end

    @testset "error paths" begin
        @test_throws ErrorException EXT.nitro_status("bogus")
        @test_throws ErrorException EXT.nitro_logger("bogus")
        @test_throws ErrorException EXT.nitro_stop("bogus")
        # A registry entry that never built a `Nitro` has no logger to report.
        lock(EXT.RUNS_LOCK) do
            EXT.RUNS["zz777777"] = EXT.RunState("zz777777", :train, "x")
        end
        @test_throws ErrorException EXT.nitro_logger("zz777777")
        st = EXT.RunState("zzzz", :validate, "")
        @test_throws ErrorException EXT._target_nitro(
            st, "bogus", nothing, nothing, nothing, Pair{Symbol, Any}[], nothing
        )
        @test_throws ErrorException EXT._target_nitro(
            st, "bogus", GATE_SPEC, nothing, nothing, Pair{Symbol, Any}[], nothing
        )
        @test_throws ErrorException EXT._target_nitro(
            st, nothing, nothing, nothing, nothing, Pair{Symbol, Any}[], nothing
        )
        @test_throws ErrorException EXT._resolve_backend("no_such_backend")
        @test_throws ErrorException EXT._parse_batch_sizes("\"one\"")
        @test_throws ErrorException EXT._parse_batch_sizes("[]")
    end

    @testset "the registry trims finished runs" begin
        lock(EXT.RUNS_LOCK) do
            for i in 1:(EXT.MAX_FINISHED_RUNS + 5)
                s = EXT.RunState(string("f", lpad(i, 3, '0')), :train, "x")
                s.status = :completed
                s.started_at = float(i)
                EXT.RUNS[string("f", lpad(i, 3, '0'))] = s
            end
            running = EXT.RunState("keepme", :train, "x")
            EXT.RUNS["keepme"] = running
            EXT._trim_registry_locked!()
            finished = [s for s in values(EXT.RUNS) if s.status !== :running]
            @test length(finished) <= EXT.MAX_FINISHED_RUNS
            # A running run is never trimmed.
            @test haskey(EXT.RUNS, "keepme")
        end
    end

end
