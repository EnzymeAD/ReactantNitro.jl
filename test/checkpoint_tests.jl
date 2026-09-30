# Checkpoint layer tests: checkpointing, resume, and seeding.
#
# The acceptance criterion this file exercises: "A run killed at epoch 3 resumes into an identical
# trajectory; a config change refuses with a diff; a changed permutation refuses; an anchored
# group with a nondeterministic init refuses on its checksum; green."

@testitem "checkpoint" begin
    using Test
    using ReactantNitro
    using ReactantNitro: HostRNG, anchor_checksum, assert_host_record, check_permutation_compatible,
        device_paths, find_latest, from_host, graphconst_fields, is_transient_io,
        read_manifest, sanitize_metric_name, snapshot, to_host, with_io_retry
    using Functors, JLD2, Lux, Optimisers, Random, Reactant, Statistics

    const FC = Float32

    @experiment struct CkptMLP
        "Traced: excluded from the resume config comparison, recorded in `devices` instead."
        scale::Device{Float32} = 1.0f0
        "GraphConst: bakes, enters the cache key, and IS compared on resume."
        tag::GraphConst{Int} = 1
        "GraphConst and structural."
        width::GraphConst{Int} = 6
        "Driver-only: excluded, since raising `max_epochs` on resume is the normal case."
        max_epochs::Host{Int} = 2
    end

    ckpt_chain(w) = Lux.Chain(Lux.Dense(4 => w, tanh), Lux.Dense(w => 2))
    ReactantNitro.build_model(e::CkptMLP, rng) = (m = ckpt_chain(e.width); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::CkptMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(e::CkptMLP, ŷ; y) = e.scale * mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::CkptMLP) = 1.0f-2

    ckpt_batches(n; seed = 5) =
        (
        rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, FC, 4, 8), y = randn(rng, FC, 2, 8)) for _ in 1:n]
    )
    const CK_TRAIN = ckpt_batches(4)
    const CK_VAL = ckpt_batches(2; seed = 6)
    ReactantNitro.build_data(::CkptMLP, dist) = (; train = CK_TRAIN, val = CK_VAL)

    # The checkpoint-naming hook, defined on an experiment of its own so the default-shape tests above
    # keep exercising the framework's own method. `kwargs...` is what makes a method survive a later
    # keyword addition, and the docstring asks for it.
    @experiment struct NamedCkpt
        max_epochs::Host{Int} = 2
    end
    ReactantNitro.build_model(::NamedCkpt, rng) = (m = ckpt_chain(6); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::NamedCkpt, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::NamedCkpt, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.build_data(::NamedCkpt, dist) = (; train = CK_TRAIN, val = CK_VAL)
    ReactantNitro.checkpoint_filename(::NamedCkpt; epoch, kwargs...) =
        "mine-" * lpad(epoch, 4, '0') * ".jld2"

    params_of(n) = [Array(l) for l in Functors.fleaves(parameters(n))]

    # The manifest makes the `file` the ONLY identity a checkpoint has, so a test asks it
    # for the file of an epoch rather than reconstructing a name. That is also what keeps these
    # tests from encoding a metric value, which is a property of the fixture's loss curve.
    epoch_file(dir, epoch) =
        joinpath(dir, only(e.file for e in read_manifest(dir) if e.epoch == epoch))

    # The checkpoint filename's `name` override, used by the retention testsets below. They assert
    # a SET OF FILENAMES, which is a statement about top-K and not about naming, so they pin a short
    # name rather than churning whenever the default shape changes. Doubles as coverage of the
    # override itself, and of a checkpointer built without going through setup.
    short_name(; epoch, kwargs...) = "epoch-" * lpad(epoch, 4, '0') * ".jld2"

    # ── the retry helper ─────────────────────────────────────────────────────────────────

    @testset "`with_io_retry` retries transient I/O and nothing else" begin
        tries = Ref(0)
        got = with_io_retry(; backoff = 0.001) do
            tries[] += 1
            tries[] < 3 && throw(Base.IOError("transient", -5))
            :ok
        end
        @test got === :ok && tries[] == 3

        # A `MethodError` is not going to succeed on the fourth try, and retrying it would turn an
        # instant, legible failure into a slow one with the cause four backoffs back in the log.
        hard = Ref(0)
        @test_throws MethodError with_io_retry(; backoff = 0.001) do
            hard[] += 1
            throw(MethodError(+, ()))
        end
        @test hard[] == 1

        @test_throws Base.IOError with_io_retry(; attempts = 2, backoff = 0.001) do
            throw(Base.IOError("always", -5))
        end
        @test is_transient_io(SystemError("x")) && is_transient_io(EOFError())
        @test !is_transient_io(ArgumentError("x"))
    end

    # ── the record, and the write path ──────────────────────────────────────────────────

    @testset "the record round-trips, carrying every field the table requires" begin
        dir = mktempdir()
        n = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2))

        files = filter(f -> startswith(f, "epoch-"), readdir(dir))
        @test length(files) == 2
        @test !any(endswith(f, ".tmp") for f in readdir(dir))   # written via a renamed temp
        @test isfile(joinpath(dir, "manifest.jld2"))

        rec = load_checkpoint(n.checkpointer, epoch_file(dir, 2))
        @test rec isa CheckpointRecord
        @test rec.format_version == 1 && rec.framework_version isa String
        @test rec.epoch == 2 && rec.step == 8                   # 2 epochs x 4 batches, accum = 1
        @test rec.seed == 42
        @test rec.config == (; tag = 1, width = 6)              # GraphConst fields only
        @test keys(rec.devices) == (:scale,)                   # Devices recorded, not compared
        @test rec.devices.scale === 1.0f0                        # and read back to a HOST value
        @test haskey(rec.metrics, :val_loss)
        @test rec.flat_permutation == n.layout.permutation
        # The per-epoch record is written BEFORE the run knows how it ended, so the outcome
        # reaches the record through the final rewrite of the last epoch's checkpoint.
        @test rec.stop_reason === :completed
        @test load_checkpoint(n.checkpointer, epoch_file(dir, 1)).stop_reason === nothing
        @test rec.logger_state === nothing && rec.logger_type === nothing

        @testset "`opt_state` is stored as HOST values (one normalization point per path)" begin
            # Serializing device leaves would tie a checkpoint to a device configuration, and JLD2 has
            # no reason to round-trip them faithfully.
            for leaf in rec.opt_state
                for v in Functors.fleaves(leaf.state)
                    @test v isa Array || v isa Number
                    @test !(v isa Reactant.AbstractConcreteArray)
                end
                # THE RULE TOO, not only the state. A rule is a struct rather than a container, so the
                # generic host walk does not reach inside one, and the per-step rebuild puts a
                # device scalar in every schedulable field of it on every optimizer step.
                for f in fieldnames(typeof(leaf.rule))
                    v = getfield(leaf.rule, f)
                    @test !(v isa Reactant.RNumber) && !(v isa Reactant.AbstractConcreteArray)
                end
            end
        end

        @testset "the manifest is what `resume = :auto` reads, and it points at the NEWEST" begin
            entries = read_manifest(dir)
            @test length(entries) == 2
            @test Set(e.epoch for e in entries) == Set([1, 2])
            # Newest, not best: resuming from the best checkpoint is not resuming from where you were.
            @test find_latest(n.checkpointer, dir) == epoch_file(dir, 2)
        end
    end

    @testset "migration: a record written before the Device rename loads" begin
        # The fixture was written by the pre-rename framework, whose `CheckpointRecord` carried a
        # `tunables` field where this one carries `devices`. JLD2 reconstructs a struct from the ON-DISK
        # field names, so the old record arrives as a ReconstructedMutable, and `load_checkpoint`
        # performs the rename field by field rather than refusing.
        path = joinpath(@__DIR__, "fixtures", "old-format-record.jld2")
        rec = load_checkpoint(TopKCheckpointer(), path)
        @test rec isa CheckpointRecord
        @test rec.devices == (; scale = 1.0f0)     # the old `tunables` value, now under `devices`
        @test rec.config == (; n_layers = 4)     # and the GraphConst config half is untouched
        @test rec.format_version == 1 && rec.framework_version isa String
        @test rec.seed == 1 && rec.epoch == 0 && rec.step == 0
        @test rec.preset === nothing               # `preset` was ADDED at 4.23; absent reads as nothing

        # AND IT LOADS QUIETLY. JLD2 warns "missing field ... reconstructing" for any record predating a
        # field addition, which reads like breakage at someone resuming a run that is about to work. The
        # migration above is explicit and tested, so the warning is noise, and noise trains people to
        # ignore warnings. Suppressed narrowly: this asserts no Warn-level message escapes the load.
        @test_logs load_checkpoint(TopKCheckpointer(), path)
    end

    @testset "top-K retains the best K in union with the newest" begin
        dir = mktempdir()
        ck = TopKCheckpointer(; k = 2, metric = :val_loss, mode = :min, dir, name = short_name)
        snap = snapshot(Nitro(CkptMLP(); run_dir = mktempdir(), checkpointer = nothing))
        # Deliberately worsening, so the newest is NOT among the best and the union is K+1.
        for (epoch, v) in enumerate([1.0, 2.0, 3.0, 4.0])
            save_checkpoint!(ck, epoch, (; val_loss = v), merge(snap, (; epoch)))
        end
        kept = sort(filter(f -> startswith(f, "epoch-"), readdir(dir)))
        @test kept == ["epoch-0001.jld2", "epoch-0002.jld2", "epoch-0004.jld2"]
        @test length(read_manifest(dir)) == 3

        # And exactly K when the newest IS among the best.
        dir2 = mktempdir()
        ck2 = TopKCheckpointer(;
            k = 2, metric = :val_loss, mode = :min, dir = dir2, name = short_name
        )
        for (epoch, v) in enumerate([4.0, 3.0, 2.0, 1.0])
            save_checkpoint!(ck2, epoch, (; val_loss = v), merge(snap, (; epoch)))
        end
        @test sort(filter(f -> startswith(f, "epoch-"), readdir(dir2))) ==
            ["epoch-0003.jld2", "epoch-0004.jld2"]

        @testset "`mode = :max` ranks the other way" begin
            dir3 = mktempdir()
            ck3 = TopKCheckpointer(; k = 1, metric = :acc, mode = :max, dir = dir3, name = short_name)
            for (epoch, v) in enumerate([0.1, 0.9, 0.5])
                save_checkpoint!(ck3, epoch, (; acc = v), merge(snap, (; epoch)))
            end
            @test sort(filter(f -> startswith(f, "epoch-"), readdir(dir3))) ==
                ["epoch-0002.jld2", "epoch-0003.jld2"]
        end

        @testset "the selection metric is a validated readback" begin
            # It drives retention, which is control flow, and a failed readback returns garbage without
            # raising on this stack.
            @test_throws ErrorException save_checkpoint!(
                ck, 9, (; val_loss = NaN),
                merge(snap, (; epoch = 9))
            )
            @test_throws ErrorException save_checkpoint!(
                ck, 9, (; other = 1.0),
                merge(snap, (; epoch = 9))
            )
        end

        @test save_checkpoint!(nothing, 1, (;), snap) === nothing   # the documented opt-out
    end

    # ── the filename ─────────────────────────────────────────────────────────────────────

    @testset "the filename carries epoch, step, and the selection metric with its value" begin
        cf(; kwargs...) = checkpoint_filename(CkptMLP(); kwargs...)

        @test cf(epoch = 6, step = 4500, metric = :val_loss, score = 0.0416831) ==
            "epoch-0006-step-4500-val_loss=0.0416831.jld2"
        # `%.6g`, the predecessor stack's format, so a score in a name is reproducible with printf.
        @test cf(epoch = 43, step = 32250, metric = :mae, score = -1.234567e-7) ==
            "epoch-0043-step-32250-mae=-1.23457e-07.jld2"
        @test cf(epoch = 12, step = 9000, metric = :dice, score = 1.0) ==
            "epoch-0012-step-9000-dice=1.jld2"
        @test cf(epoch = 1, step = 750, metric = :m, score = 3.14159265e300) ==
            "epoch-0001-step-750-m=3.14159e+300.jld2"

        @testset "`score === nothing` omits the segment rather than filling it" begin
            # What a run with no `val` split produces. No metric segment means no metric was
            # computed, and a token such as `no_score` would reserve a name a real metric could take.
            @test cf(epoch = 6, step = 4500, metric = :val_loss, score = nothing) ==
                "epoch-0006-step-4500.jld2"
        end

        @testset "the epoch stays first and zero-padded, so `ls` order is training order" begin
            names = [cf(epoch = i, step = 10i, metric = :m, score = 1.0 / i) for i in [1, 2, 10, 100]]
            @test names == sort(names)
            @test all(startswith(n, "epoch-0") for n in names[1:3])
        end

        @testset "the metric NAME is sanitized, because a `Symbol` is not a filename" begin
            # The VALUE needs none: the readback check refuses a non-finite selection metric
            # before this is
            # reached, so `%g` output is always within `[0-9.eE+-]`.
            @test sanitize_metric_name(:val_loss) == "val_loss"
            @test sanitize_metric_name(Symbol("a/b")) == "a_b"       # no path separator survives
            @test sanitize_metric_name(Symbol("a*b?c[d]")) == "a_b_c_d_"   # no glob metacharacter
            @test sanitize_metric_name(Symbol("a:b|c")) == "a_b_c"   # nothing Windows-hostile
            @test length(sanitize_metric_name(Symbol("x"^80))) == 40
            @test sanitize_metric_name(Symbol("")) == "metric"
            @test !occursin('/', cf(epoch = 1, step = 1, metric = Symbol("a/b"), score = 1.0))
        end

        @testset "an experiment names its own, and the checkpointer adopts it at setup" begin
            dir = mktempdir()
            n = train!(Nitro(NamedCkpt(); run_dir = dir, max_epochs = 2))
            @test sort(filter(endswith(".jld2"), readdir(dir))) ==
                ["manifest.jld2", "mine-0001.jld2", "mine-0002.jld2"]
            # The manifest is the identity, so resume resolves through the custom name unchanged.
            @test find_latest(n.checkpointer, dir) == joinpath(dir, "mine-0002.jld2")
            @test load_checkpoint(n.checkpointer, epoch_file(dir, 2)).epoch == 2

            @testset "an explicit `name` pins it, exactly as an explicit `dir` does" begin
                dir2 = mktempdir()
                train!(
                    Nitro(
                        NamedCkpt(); run_dir = dir2, max_epochs = 1,
                        checkpointer = TopKCheckpointer(; name = short_name)
                    )
                )
                @test "epoch-0001.jld2" in readdir(dir2)
            end
        end

        @testset "the return is checked, since a hook is a user method" begin
            snap = snapshot(Nitro(CkptMLP(); run_dir = mktempdir(), checkpointer = nothing))
            bad(name) = save_checkpoint!(
                TopKCheckpointer(; dir = mktempdir(), name),
                1, (; val_loss = 1.0), merge(snap, (; epoch = 1))
            )
            @test_throws "not a checkpoint filename" bad((; kwargs...) -> "no-extension")
            @test_throws "not a checkpoint filename" bad((; kwargs...) -> "")
            @test_throws "not a checkpoint filename" bad((; kwargs...) -> 7)
            # A name is a BASENAME inside `run_dir`, which the checkpoint layer makes the one path
            # concept.
            @test_throws "path separator" bad((; kwargs...) -> "a/b.jld2")
            @test_throws "path separator" bad((; kwargs...) -> "../escape.jld2")
            # A checkpointer that never went through setup has no name to call.
            @test_throws "has no `name`" save_checkpoint!(
                TopKCheckpointer(; dir = mktempdir()),
                1, (; val_loss = 1.0), merge(snap, (; epoch = 1))
            )
        end

        @testset "rewriting an epoch under a NEW name deletes the file it displaced" begin
            # The manifest holds one entry per epoch, and the rotation only deletes files that still
            # have an entry, so a changed name would otherwise leave a file invisible to this
            # rotation and to every later one: permanent, not merely untidy. Unreachable under the
            # default name, whose four inputs are identical across the phase system's final
            # rewrite, but `name` is
            # a hook and Revise can change one mid-run.
            dir = mktempdir()
            snap = snapshot(Nitro(CkptMLP(); run_dir = mktempdir(), checkpointer = nothing))
            ck = TopKCheckpointer(; dir, name = short_name)
            save_checkpoint!(ck, 1, (; val_loss = 1.0), merge(snap, (; epoch = 1)))
            @test filter(endswith(".jld2"), readdir(dir)) |> sort ==
                ["epoch-0001.jld2", "manifest.jld2"]

            ck.name = (; epoch, kwargs...) -> "renamed-$(epoch).jld2"
            save_checkpoint!(ck, 1, (; val_loss = 1.0), merge(snap, (; epoch = 1)))
            @test sort(filter(endswith(".jld2"), readdir(dir))) ==
                ["manifest.jld2", "renamed-1.jld2"]
            @test length(read_manifest(dir)) == 1
            @test read_manifest(dir)[1].file == "renamed-1.jld2"
        end

        @testset "a `name` that omits the epoch is \"best only\", and stays one entry" begin
            # The manifest holds one entry per FILE as well as one per epoch, because `file` is the
            # only identity a checkpoint has: an entry naming a file a later write overwrote
            # describes bytes that are gone. Without that, this grows an entry per epoch all naming
            # the same file, and the manifest a resume reads never stops growing.
            dir = mktempdir()
            snap = snapshot(Nitro(CkptMLP(); run_dir = mktempdir(), checkpointer = nothing))
            ck = TopKCheckpointer(; dir, name = (; kwargs...) -> "best.jld2")
            for (epoch, v) in enumerate([3.0, 1.0, 2.0])
                save_checkpoint!(ck, epoch, (; val_loss = v), merge(snap, (; epoch)))
            end
            @test sort(filter(endswith(".jld2"), readdir(dir))) == ["best.jld2", "manifest.jld2"]
            entries = read_manifest(dir)
            @test length(entries) == 1
            @test entries[1].epoch == 3 && entries[1].score == 2.0   # the file's actual contents
            @test find_latest(ck, dir) == joinpath(dir, "best.jld2")
        end
    end

    # ── The checkpoint layer's headline: a killed run resumes into an identical trajectory ──

    @testset "a run stopped at epoch 2 resumes into the SAME trajectory" begin
        uninterrupted = train!(
            Nitro(
                CkptMLP(); run_dir = mktempdir(), max_epochs = 4,
                checkpointer = nothing
            )
        )

        dir = mktempdir()
        first_half = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2))
        @test current_epoch(first_half) == 2 && current_step(first_half) == 8

        # Resuming is OPT IN: a fresh handle over the same directory would start from zero.
        resumed = Nitro(CkptMLP(); run_dir = dir, max_epochs = 4, resume = :auto)
        @test current_epoch(resumed) == 2                        # step and epoch restored, not derived
        @test current_step(resumed) == 8
        train!(resumed)
        @test current_epoch(resumed) == 4 && current_step(resumed) == 16

        for (a, b) in zip(params_of(uninterrupted), params_of(resumed))
            @test a ≈ b rtol = 1.0e-5
        end

        @testset "and `resume = false` forces a fresh run in the same directory" begin
            fresh = Nitro(CkptMLP(); run_dir = dir, max_epochs = 4, resume = false)
            @test current_epoch(fresh) == 0 && current_step(fresh) == 0
        end
    end

    @testset "the seed is restored from the record, and the override is announced" begin
        dir = mktempdir()
        train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1, seed = 11))
        # A silent override turns a seed sweep that forgot to vary `run_dir` into N identical runs.
        n = @test_logs (:warn, r"restoring `seed = 11`") match_mode = :any Nitro(
            CkptMLP();
            run_dir = dir,
            seed = 99,
            resume = :auto
        )
        @test n.seed == 11
    end

    # ── the resume-check refusals ────────────────────────────────────────────────────────

    @testset "a changed GraphConst field refuses with a diff, and a `Device` does not" begin
        dir = mktempdir()
        train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1))

        err = try
            Nitro(CkptMLP(; tag = 2); run_dir = dir, resume = :auto)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("tag: 1 -> 2", err.msg)                   # the DIFF, not just a refusal
        @test occursin("compile cache", err.msg)

        # A `Device` may change freely: it is a traced input and cannot change the graph, and the
        # change stays visible after the fact in the record's `devices`.
        @test (
            @test_logs match_mode = :any Nitro(
                CkptMLP(; scale = 2.0f0); run_dir = dir, resume = :auto
            )
        ).epoch == 1
        # A `Host` field may change: raising `max_epochs` on resume is the normal case.
        @test Nitro(CkptMLP(); run_dir = dir, max_epochs = 9).max_epochs == 9   # raised: no warning
    end

    @testset "a changed permutation refuses, naming the leaf that moved" begin
        dir = mktempdir()
        n = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1))
        rec = load_checkpoint(n.checkpointer, find_latest(n.checkpointer, dir))

        moved = copy(rec.flat_permutation)
        was = moved[1]
        moved[1] = (;
            keypath = was.keypath, group = was.group, offset = was.offset,
            len = was.len, size = (was.size[1] + 1, Base.tail(was.size)...),
        )
        bad = CheckpointRecord(
            rec.format_version, rec.framework_version, rec.ps, rec.st,
            rec.opt_state, moved, rec.step, rec.epoch, rec.seed, rec.config,
            rec.devices, rec.metrics, rec.run_id, rec.run_url, rec.logger_state,
            rec.logger_type, rec.anchor_checksum, rec.stop_reason, rec.preset
        )
        err = try
            check_permutation_compatible(bad, n.layout, "x")
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("scrambles", err.msg)
        @test occursin(string(was.keypath), err.msg)             # names the LEAF, not two int vectors
    end

    # ── the decay anchor, whose NEGATIVE half is the load-bearing one ──────────────────

    const ANCHOR_NOISE = Ref(0.0f0)

    @experiment struct AnchoredExp
        width::GraphConst{Int} = 4
    end
    function ReactantNitro.build_model(e::AnchoredExp, rng)
        m = Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dense(e.width => 2))
        ps, st = Lux.setup(rng, m)
        # The switch that makes this init irreproducible, which is precisely what the resume check
        # refuses on.
        ANCHOR_NOISE[] == 0 && return (m, ps, st)
        return (m, Functors.fmap(x -> x isa AbstractArray ? x .+ ANCHOR_NOISE[] : x, ps), st)
    end
    ReactantNitro.forward(::AnchoredExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::AnchoredExp, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::AnchoredExp) = 1.0f-2
    ReactantNitro.lambda(::AnchoredExp) = 1.0f-2
    ReactantNitro.decay_anchor(::AnchoredExp, ::Val{g}) where {g} = :w0
    ReactantNitro.build_data(::AnchoredExp, dist) = (; train = CK_TRAIN, val = CK_VAL)

    @testset "the anchor checksum: a deterministic init resumes, a nondeterministic one refuses" begin
        dir = mktempdir()
        ANCHOR_NOISE[] = 0.0f0
        n = train!(Nitro(AnchoredExp(); run_dir = dir, max_epochs = 1))
        rec = load_checkpoint(n.checkpointer, find_latest(n.checkpointer, dir))
        @test rec.anchor_checksum !== nothing
        @test all(c -> c isa String, rec.anchor_checksum)        # SHA256 hex, stable across versions

        # The POSITIVE half passes whether or not the check is implemented, which is why the negative
        # half below is the one that matters.
        @test Nitro(AnchoredExp(); run_dir = dir, max_epochs = 2, resume = :auto).epoch == 1

        ANCHOR_NOISE[] = 0.5f0
        err = try
            Nitro(AnchoredExp(); run_dir = dir, max_epochs = 2, resume = :auto)
            nothing
        catch ex
            ex
        end
        ANCHOR_NOISE[] = 0.0f0
        @test err isa ErrorException
        @test occursin("decay anchor", err.msg)
        @test occursin("not reproducible", err.msg)
        # Refuses rather than warning and continuing: continuing would silently regularize toward a
        # DIFFERENT point than the run being resumed, with a plausible loss curve and no error.
        @test occursin("Continuing is not offered", err.msg)
    end

    @testset "an unanchored experiment stores no checksum and checks none" begin
        dir = mktempdir()
        n = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1))
        rec = load_checkpoint(n.checkpointer, find_latest(n.checkpointer, dir))
        @test rec.anchor_checksum === nothing
        @test anchor_checksum(nothing) === nothing
    end

    # ── the killed-run resume test ───────────────────────────────────────────────────────

    mutable struct StatefulLog
        state::Any
        reattached::Any
    end
    ReactantNitro.log_metrics!(::StatefulLog, m; kwargs...) = nothing
    ReactantNitro.log_params!(::StatefulLog, p) = nothing
    ReactantNitro.log_other!(::StatefulLog, k, v) = nothing
    ReactantNitro.finish!(::StatefulLog, s) = nothing
    ReactantNitro.run_id(l::StatefulLog) = "run-7"
    ReactantNitro.run_url(l::StatefulLog) = "https://example.invalid/run-7"
    ReactantNitro.logger_state(l::StatefulLog) = l.state
    ReactantNitro.reattach!(l::StatefulLog, s) = (l.reattached = s; nothing)

    # Claims resumable state and cannot restore it: no `reattach!` method anywhere.
    mutable struct BrokenLog end
    ReactantNitro.log_metrics!(::BrokenLog, m; kwargs...) = nothing
    ReactantNitro.log_params!(::BrokenLog, p) = nothing
    ReactantNitro.log_other!(::BrokenLog, k, v) = nothing
    ReactantNitro.finish!(::BrokenLog, s) = nothing
    ReactantNitro.run_id(::BrokenLog) = nothing
    ReactantNitro.run_url(::BrokenLog) = nothing
    ReactantNitro.logger_state(::BrokenLog) = (; key = "abc")

    # A logger of a DIFFERENT type than `StatefulLog`, for the resume-refusal test below. Defined
    # here rather than borrowed from `lifecycle.jl`: this file must run standalone (the suite's
    # include order is an accident of `runtests.jl`, not a contract) and the refusal check only
    # needs the type NAME to differ. The full verb surface is no-ops so construction can call any
    # of them; none are exercised, since the refusal fires before the run's first log.
    mutable struct OtherLog end
    ReactantNitro.log_metrics!(::OtherLog, m; kwargs...) = nothing
    ReactantNitro.log_params!(::OtherLog, p) = nothing
    ReactantNitro.log_tags!(::OtherLog, t) = nothing
    ReactantNitro.log_other!(::OtherLog, k, v) = nothing
    ReactantNitro.log_confusion!(::OtherLog, m, labels; kwargs...) = nothing
    ReactantNitro.finish!(::OtherLog, s) = nothing
    ReactantNitro.run_id(::OtherLog) = "other-1"
    ReactantNitro.run_url(::OtherLog) = "https://example.invalid/other-1"
    ReactantNitro.logger_info(::OtherLog) = (;)
    ReactantNitro.logger_state(::OtherLog) = nothing
    ReactantNitro.reattach!(::OtherLog, s) = nothing

    # Returns the live backend object rather than plain data, which the logger contract names as
    # the one
    # detail a backend author is likely to get wrong.
    mutable struct HandleLog end
    ReactantNitro.log_metrics!(::HandleLog, m; kwargs...) = nothing
    ReactantNitro.log_params!(::HandleLog, p) = nothing
    ReactantNitro.log_other!(::HandleLog, k, v) = nothing
    ReactantNitro.finish!(::HandleLog, s) = nothing
    ReactantNitro.run_id(::HandleLog) = nothing
    ReactantNitro.run_url(::HandleLog) = nothing
    ReactantNitro.logger_state(l::HandleLog) = l

    @testset "the logger round trip, and the two ways it can be wrong" begin
        @testset "a logger with no resumable state round-trips with NEITHER field stored" begin
            dir = mktempdir()
            n = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1, logger = StatefulLog(nothing, nothing)))
            rec = load_checkpoint(n.checkpointer, find_latest(n.checkpointer, dir))
            @test rec.logger_state === nothing
            @test rec.logger_type === nothing
            @test rec.run_id == "run-7"                          # informational, and kept regardless
            @test rec.run_url == "https://example.invalid/run-7"
            # Reattachment is not attempted, so a ten-line file logger keeps working across a resume.
            lg = StatefulLog(nothing, nothing)
            Nitro(CkptMLP(); run_dir = dir, logger = lg)
            @test lg.reattached === nothing
        end

        @testset "a logger WITH state round-trips and is reattached before the first metric" begin
            dir = mktempdir()
            n = train!(
                Nitro(
                    CkptMLP(); run_dir = dir, max_epochs = 1,
                    logger = StatefulLog((; key = "exp-1"), nothing)
                )
            )
            rec = load_checkpoint(n.checkpointer, find_latest(n.checkpointer, dir))
            @test rec.logger_state == (; key = "exp-1")
            @test rec.logger_type == "StatefulLog"

            fresh = StatefulLog((; key = "exp-1"), nothing)
            Nitro(CkptMLP(); run_dir = dir, logger = fresh, resume = :auto)
            @test fresh.reattached == (; key = "exp-1")
        end

        @testset "a WEIGHTS-ONLY restore may decline the logger; a resume may not" begin
            # The bug two independent model ports found: this check ran on both restore paths, so every
            # Before this, export of a checkpoint written with a logger was impossible. `logger =
            # nothing` was
            # refused, and the only alternative was to pass the real backend and have setup reattach and
            # re-log config into the finished TRAINING experiment from an export.
            dir = mktempdir()
            train!(
                Nitro(
                    CkptMLP(); run_dir = dir, max_epochs = 1,
                    logger = StatefulLog((; key = "exp-2"), nothing)
                )
            )
            latest = find_latest(TopKCheckpointer(; dir), dir)

            # Weights-only (`weights = path`): declining the logger is legal and reattaches nothing.
            n = Nitro(CkptMLP(); run_dir = dir, weights = latest, logger = nothing)
            @test n isa Nitro
            @test current_epoch(n) == 0                     # continues no trajectory

            # A RESUME still refuses, because it continues one history and dropping the state silently
            # would start a second experiment.
            err = try
                Nitro(CkptMLP(); run_dir = dir, resume = latest, logger = nothing)
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("no logger", err.msg)

            # And the TYPE check still applies to a logger that IS passed on the weights-only path, so a
            # mismatched backend cannot be handed another's opaque state.
            err2 = try
                Nitro(CkptMLP(); run_dir = dir, weights = latest, logger = HandleLog())
                nothing
            catch ex
                ex
            end
            @test err2 isa ErrorException
        end

        @testset "state with no `reattach!` fails LOUDLY, where the state is produced" begin
            dir = mktempdir()
            err = try
                train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1, logger = BrokenLog()))
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("reattach!", err.msg)
            @test occursin("REQUIRED", err.msg)
        end

        @testset "`logger_state` returning the live backend object is refused, and named" begin
            dir = mktempdir()
            err = try
                train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1, logger = HandleLog()))
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("plain serializable data", err.msg)
        end

        @testset "resuming into a DIFFERENT logger type refuses, naming both" begin
            dir = mktempdir()
            train!(
                Nitro(
                    CkptMLP(); run_dir = dir, max_epochs = 1,
                    logger = StatefulLog((; key = "exp-2"), nothing)
                )
            )
            err = try
                Nitro(CkptMLP(); run_dir = dir, logger = OtherLog(), resume = :auto)
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("StatefulLog", err.msg) && occursin("OtherLog", err.msg)
        end
    end

    # ── weights-only construction ────────────────────────────────────────────────────────

    @testset "`weights = path` restores weights, not a trajectory" begin
        dir = mktempdir()
        trained = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2))
        path = find_latest(trained.checkpointer, dir)

        served = Nitro(CkptMLP(); data = (; test = CK_VAL), weights = path, run_dir = mktempdir())
        @test params_of(served) == params_of(trained)            # the trained weights, exactly
        @test current_epoch(served) == 0                         # but NOT the trajectory
        @test current_step(served) == 0
        @test served.opt_state === nothing                       # and no moments: step 9 is skipped
        # This is the evaluation construction, and it works with no training data at all.
        @test evaluate(served; split = :test).val_loss isa Real

        # `load_checkpoint` dispatches on the checkpointer, so `checkpointer = nothing` has no
        # loader. Asking to load one anyway is a contradiction, and answering with freshly initialized
        # weights that look plausible is the worst way to resolve it.
        err = try
            Nitro(CkptMLP(); weights = path, checkpointer = nothing, run_dir = mktempdir())
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("has no loader", err.msg)
    end

    @testset "`weights = :latest => :best | :latest` name a checkpoint of the run directory" begin
        dir = mktempdir()
        trained = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 3))
        best = ReactantNitro.selected_checkpoint(trained.checkpointer, dir)
        latest = find_latest(trained.checkpointer, dir)

        # Resolved to the file the manifest names, so the handle, and anything exported from it,
        # records the actual path rather than the symbol.
        nb = Nitro(CkptMLP(); data = (; test = CK_VAL), run_dir = dir, weights = :latest => :best)
        @test nb.checkpoint_source == best.path
        nl = Nitro(CkptMLP(); data = (; test = CK_VAL), run_dir = dir, weights = :latest => :latest)
        @test nl.checkpoint_source == latest
        @test params_of(nl) == params_of(trained)       # the newest record is the final weights

        # Every miss raises: fresh weights in place of the ones asked for would look plausible.
        msg(f) = try
            f(); ""
        catch ex
            ex isa ErrorException ? ex.msg : rethrow()
        end
        @test occursin("found no checkpoint", msg(() -> Nitro(CkptMLP(); run_dir = mktempdir(), weights = :latest => :best)))
        @test occursin("`weights` takes", msg(() -> Nitro(CkptMLP(); run_dir = dir, weights = :newest)))
        @test occursin("has none", msg(() -> Nitro(CkptMLP(); run_dir = dir, weights = :latest => :best, checkpointer = nothing)))
    end

    @testset "`checkpoint = path` still loads, warns it is deprecated, and yields to `weights`" begin
        dir = mktempdir()
        trained = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2))
        path = find_latest(trained.checkpointer, dir)
        old = @test_logs (:warn, r"deprecated") match_mode = :any Nitro(
            CkptMLP(); data = (; test = CK_VAL), checkpoint = path, run_dir = mktempdir()
        )
        @test params_of(old) == params_of(trained)
        @test old.checkpoint_source == path
        @test old.opt_state === nothing                       # still weights-only
        err = try
            Nitro(CkptMLP(); weights = path, checkpoint = path, run_dir = mktempdir()); nothing
        catch ex
            ex
        end
        @test err isa ErrorException && occursin("both were given", err.msg)
    end

    @testset "the checkpointer's own configuration is checked at setup" begin
        @test_throws ErrorException Nitro(
            CkptMLP(); run_dir = mktempdir(),
            checkpointer = TopKCheckpointer(; mode = :lowest)
        )
        @test_throws ErrorException Nitro(
            CkptMLP(); run_dir = mktempdir(),
            checkpointer = TopKCheckpointer(; k = 0)
        )
        # This experiment defines no `metrics`, so `:val_loss` is the only key that can appear.
        err = try
            Nitro(CkptMLP(); run_dir = mktempdir(), checkpointer = TopKCheckpointer(; metric = :acc))
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("acc", err.msg) && occursin("val_loss", err.msg)
    end


    # ── The device-residency fix: a device value in `st`, which is fact 8 recurring ──────

    @experiment struct DropoutCkpt
        max_epochs::Host{Int} = 1
    end
    dropout_chain() = Lux.Chain(Lux.Dense(4 => 6, tanh), Lux.Dropout(0.5f0), Lux.Dense(6 => 2))
    ReactantNitro.build_model(::DropoutCkpt, rng) =
        (m = dropout_chain(); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::DropoutCkpt, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::DropoutCkpt, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.build_data(::DropoutCkpt, dist) = (; train = CK_TRAIN, val = CK_VAL)

    @testset "a stateful RNG in `st` does not reach the record, and resume survives it" begin
        # THE DEVICE-RESIDENCY BLOCKER. `Lux.Dropout`'s state carries a `Reactant.ReactantRNG`
        # whose `seed` is a
        # `ConcretePJRTArray`. `to_host` walked containers only, so the device array went into the
        # record, JLD2 wrote its raw pointer without complaint, and the restore died in a FRESH process
        # on `AssertionError: buffer.buffer !== C_NULL`, naming neither the field nor serialization.
        # Fact 8, in the one place the checkpoint layer's host-ification did not reach.
        dir = mktempdir()
        n = train!(DropoutCkpt(); run_dir = dir)
        @test current_epoch(n) == 1

        @testset "the snapshot is device-free, and the RNG survives as a surrogate" begin
            snap = snapshot(n)
            @test isempty(device_paths(snap.ps, "ps"))
            @test isempty(device_paths(snap.st, "st"))
            @test isempty(device_paths(snap.opt_state, "opt_state"))
            # `ReactantRNG`'s type parameter is constrained to a device array, so there is no
            # host-resident `ReactantRNG` to rebuild and the record has to store a surrogate.
            @test_throws MethodError Reactant.ReactantRNG(UInt64[1, 2], "algo")
            @test snap.st.layer_2.rng isa HostRNG
            @test snap.st.layer_2.rng.seed isa Vector{UInt64}
        end

        @testset "and the surrogate turns back into a device RNG on the way in" begin
            rec = load_checkpoint(n.checkpointer, epoch_file(dir, 1))
            back = from_host(rec.st)
            @test back.layer_2.rng isa Reactant.ReactantRNG
            @test back.layer_2.rng.seed isa Reactant.AbstractConcreteArray
            @test Array(back.layer_2.rng.seed) == rec.st.layer_2.rng.seed
        end

        @testset "resume reads it back rather than dying on a dead pointer" begin
            n2 = Nitro(DropoutCkpt(); run_dir = dir, max_epochs = 2, resume = :auto)
            @test current_epoch(n2) == 1                      # restored, not reset
            @test n2.st.layer_2.rng isa Reactant.ReactantRNG
            n3 = train!(n2)
            @test current_epoch(n3) == 2                      # and it continues
        end
    end

    @testset "CONTROL: `assert_host_record` actually detects a device value" begin
        # Without this the assertion above could pass by never firing. A record carrying a device array
        # must be refused AT WRITE TIME, in the writing process, naming the path.
        live = Reactant.to_rarray(randn(FC, 4))
        snap = (; ps = (; layer_1 = (; weight = live)), st = (;), opt_state = (;), devices = (;))
        err = try
            assert_host_record(snap)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("ps.layer_1.weight", err.msg)          # the PATH, which is the useful part
        @test occursin("HostRNG", err.msg)                    # and what to do about it

        # It reaches inside a struct, which is the shape the hole kept recurring in.
        dev_rng = Reactant.ReactantRNG(Reactant.to_rarray(UInt64[1, 2]), "x")
        @test !isempty(device_paths((; l = (; rng = dev_rng)), "st"))
        # And `to_host` now closes it, which is what the walk reaching structs bought.
        @test isempty(device_paths(to_host((; l = (; rng = dev_rng))), "st"))
    end

    # ── the preset name in the record ───────────────────────────────────────────────────

    ReactantNitro.presets(::Type{CkptMLP}) = (small = (; width = 3),)

    @testset "the preset name reaches the checkpoint record" begin
        dir = mktempdir()
        n = train!(Nitro(CkptMLP, :small; run_dir = dir, max_epochs = 1))
        @test n.preset === :small
        rec = load_checkpoint(n.checkpointer, find_latest(n.checkpointer, dir))
        # This is requirement 5: a RESULT can say which named recipe produced it.
        @test rec.preset === :small

        # A run that named none round-trips as `nothing` rather than refusing, which is also what an
        # older record does, since `preset` was ADDED at 4.23 and FORMAT_VERSION stayed 1.
        dir2 = mktempdir()
        n2 = train!(Nitro(CkptMLP(); run_dir = dir2, max_epochs = 1))
        @test n2.preset === nothing
        @test load_checkpoint(n2.checkpointer, find_latest(n2.checkpointer, dir2)).preset === nothing
    end

    # ── asking a checkpoint what it is, without its weights ─────────────────────────────
    #
    # The record is one JLD2 entry holding one object, four of whose fields are the parameter tree,
    # so there is no way to read `epoch` without materializing all of it. That left every session to
    # invent its own reader, and the invention that costs something is `println(record)`: the default
    # struct `show` walks every field, so a 115 MB checkpoint renders as tens of millions of
    # characters. In a REPL that is a wasted screen. In an agent session the REPL's output IS the
    # transcript, so printing a record to find `epoch` can spend an entire session budget on the
    # parameter dump.
    #
    # Two things have to hold for that to be over: the safe rendering must be the DEFAULT rendering,
    # and there must be a supported way to ask that never gets near the arrays.
    @testset "a record shows its metadata and never its weights" begin
        dir = mktempdir()
        n = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 1, seed = 4321))
        path = find_latest(n.checkpointer, dir)
        rec = load_checkpoint(n.checkpointer, path)

        # The weights ARE in there: this is a summary, not a smaller record.
        @test rec.ps !== nothing

        for s in (sprint(show, rec), sprint(show, MIME"text/plain"(), rec))
            # Nothing array-shaped reaches the output, by the only test that matters: the printed
            # form of an array of floats. A record whose show regressed prints these in the thousands.
            @test !occursin("Float32[", s)
            @test !occursin("Float64[", s)
            # Bounded, and bounded SMALL: the whole point is that it cannot grow with the model.
            @test length(s) < 2_000
        end
        # The one-line form names the fields it withheld rather than pretending they are absent,
        # and still answers the question that made someone open the file.
        short = sprint(show, rec)
        @test occursin("not shown", short)
        @test occursin("epoch $(rec.epoch)", short) && occursin("step $(rec.step)", short)
        # The long form withholds all four and says where the weights actually go.
        long = sprint(show, MIME"text/plain"(), rec)
        @test count("<not shown", long) == 4
        for f in ("ps", "st", "opt_state", "flat_permutation")
            @test occursin(f, long)
        end
        @test occursin("checkpoint_info", long)   # the reader is named where the question is asked

        # `checkpoint_info` answers without handing back anything weight-shaped.
        i = checkpoint_info(path)
        @test i.epoch == rec.epoch && i.step == rec.step && i.seed == 4321
        @test i.format_version == ReactantNitro.FORMAT_VERSION
        @test i.metrics == rec.metrics
        @test !any(hasproperty(i, f) for f in (:ps, :st, :opt_state, :flat_permutation))
        # Small enough to print, which is the property a caller relies on without checking.
        @test length(sprint(show, i)) < 2_000
        @test_throws ErrorException checkpoint_info(joinpath(dir, "nope.jld2"))

        # And the run-directory question needs no record at all: the manifest carries it.
        entries = read_manifest(dir)
        @test !isempty(entries)
        @test any(e.epoch == rec.epoch for e in entries)
        @test all(hasproperty(e, f) for e in entries for f in (:file, :epoch, :score, :stop_reason))
    end

    # ── several runs in one directory ───────────────────────────────────────────────────
    #
    # The default `run_dir` is `runs/<ExperimentTypeName>`, so two fresh runs of one experiment
    # share a directory unless someone thought to vary it. The manifest records which run wrote
    # each entry, and every rule below is a rule about ONE run: retention, top-K, the names
    # `:best` and `:latest`, and `resume = :auto`. `epoch_file` above assumes one entry per epoch,
    # which a shared directory breaks, so these testsets ask by run.
    run_entries(dir, run) = [e for e in read_manifest(dir) if isequal(e.run, run)]
    run_file(dir, run, epoch) =
        joinpath(dir, only(e.file for e in run_entries(dir, run) if e.epoch == epoch))
    run_files(dir, run) = Set(e.file for e in run_entries(dir, run))
    runs_in(dir) = Set(e.run for e in read_manifest(dir))
    run_of(n) = n.checkpointer.run
    # What `retained` promises for one run's entries: the top K by score in union with the newest.
    function expected_kept(entries, k)
        newest = entries[argmax([e.epoch for e in entries])]
        by = sort([e for e in entries if e.score !== nothing]; by = e -> e.score)
        return Set(vcat([e.file for e in first(by, k)], newest.file))
    end
    k1() = TopKCheckpointer(; k = 1)

    @testset "a second fresh run in the directory deletes none of the first run's checkpoints" begin
        dir = mktempdir()
        a = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 4, seed = 1, checkpointer = k1()))
        a_files = run_files(dir, run_of(a))
        @test a_files == expected_kept(run_entries(dir, run_of(a)), 1)
        b = @test_logs (:info, r"already holds") match_mode = :any Nitro(
            CkptMLP(); run_dir = dir, max_epochs = 4, seed = 2, checkpointer = k1()
        )
        train!(b)

        @test run_of(a) isa String && run_of(b) isa String && run_of(a) != run_of(b)
        @test runs_in(dir) == Set([run_of(a), run_of(b)])
        # Every one of A's retained files is still listed and still on disk.
        @test run_files(dir, run_of(a)) == a_files
        @test all(isfile(joinpath(dir, f)) for f in a_files)
        # B retains its OWN newest and its OWN top K, ranked against its own entries only.
        b_entries = run_entries(dir, run_of(b))
        @test maximum(e.epoch for e in b_entries) == 4
        @test run_files(dir, run_of(b)) == expected_kept(b_entries, 1)
        @test all(isfile(joinpath(dir, e.file)) for e in b_entries)
        # And every entry records the metric and mode that produced its score.
        @test all(e.metric === :val_loss && e.mode === :min for e in read_manifest(dir))
        # After `train!`, the handle reports ITS run's selection, not the directory's.
        @test a.best_checkpoint.path in (joinpath(dir, f) for f in a_files)
        @test basename(b.best_checkpoint.path) in run_files(dir, run_of(b))
    end

    @testset "the same seed and config twice: identical names, and neither overwrites the other" begin
        dir = mktempdir()
        a = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2))
        a_files = run_files(dir, run_of(a))
        stamps = Dict(f => mtime(joinpath(dir, f)) for f in a_files)
        a_params = params_of(a)
        b = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2))
        @test params_of(b) == a_params                       # the two runs really are identical

        @test length(read_manifest(dir)) == 4
        @test run_files(dir, run_of(a)) == a_files
        # A's bytes are A's: not rewritten, which the same name would otherwise have done.
        @test all(mtime(joinpath(dir, f)) == stamps[f] for f in a_files)
        b_files = run_files(dir, run_of(b))
        @test isempty(intersect(a_files, b_files))
        @test all(occursin("-run-$(run_of(b))", f) for f in b_files)
        for f in union(a_files, b_files)
            @test load_checkpoint(TopKCheckpointer(), joinpath(dir, f)).epoch in (1, 2)
        end
        # The final rewrite of B's last epoch landed on B's disambiguated file, not on A's.
        @test load_checkpoint(TopKCheckpointer(), run_file(dir, run_of(b), 2)).stop_reason === :completed
        @test load_checkpoint(TopKCheckpointer(), run_file(dir, run_of(a), 2)).stop_reason === :completed
    end

    @testset "`:latest => :best | :latest` and `resume = :auto` mean the most recent run" begin
        dir = mktempdir()
        a = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2, seed = 1))
        b = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 3, seed = 2))
        b_best = ReactantNitro.selected_checkpoint(b.checkpointer, dir; run = run_of(b))
        # With no run named, both lookups answer for the most recent run.
        @test find_latest(TopKCheckpointer(; dir), dir) == run_file(dir, run_of(b), 3)
        @test ReactantNitro.selected_checkpoint(TopKCheckpointer(; dir), dir).path == b_best.path
        # And a named run answers for that run.
        @test find_latest(TopKCheckpointer(; dir), dir; run = run_of(a)) == run_file(dir, run_of(a), 2)

        nl = @test_logs (:info, r"holds 2 runs") match_mode = :any Nitro(
            CkptMLP(); data = (; test = CK_VAL), run_dir = dir, weights = :latest => :latest
        )
        @test nl.checkpoint_source == run_file(dir, run_of(b), 3)
        @test params_of(nl) == params_of(b)
        nb = Nitro(CkptMLP(); data = (; test = CK_VAL), run_dir = dir, weights = :latest => :best)
        @test nb.checkpoint_source == b_best.path

        r = @test_logs (:info, r"holds 2 runs") match_mode = :any Nitro(
            CkptMLP(); run_dir = dir, max_epochs = 4, resume = :auto
        )
        @test r.checkpoint_source == run_file(dir, run_of(b), 3)
        @test current_epoch(r) == 3
        # It CONTINUES B: its later checkpoints join B's run rather than starting a third.
        @test run_of(r) == run_of(b)
        train!(r)
        @test runs_in(dir) == Set([run_of(a), run_of(b)])
        @test maximum(e.epoch for e in run_entries(dir, run_of(b))) == 4
    end

    @testset "a legacy manifest loads, is one run, and a new run never deletes its files" begin
        dir = mktempdir()
        train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 3, seed = 1))
        # Rewrite the manifest in the shape every earlier framework version wrote: four fields.
        Old = @NamedTuple{
            file::String, epoch::Int, score::Union{Float64, Nothing},
            stop_reason::Union{Symbol, Nothing},
        }
        old = Old[Old((e.file, e.epoch, e.score, e.stop_reason)) for e in read_manifest(dir)]
        JLD2.jldsave(joinpath(dir, "manifest.jld2"); entries = old)
        legacy = read_manifest(dir)
        @test length(legacy) == 3
        @test all(e.run === nothing && e.metric === nothing && e.mode === nothing for e in legacy)
        @test all(e.written == 0.0 for e in legacy)
        legacy_files = Set(e.file for e in legacy)

        # A legacy run on its own is still the directory's run.
        @test find_latest(TopKCheckpointer(; dir), dir) == run_file(dir, nothing, 3)
        @test ReactantNitro.selected_checkpoint(TopKCheckpointer(; dir), dir) !== nothing

        b = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 3, seed = 2, checkpointer = k1()))
        @test run_files(dir, nothing) == legacy_files
        @test all(isfile(joinpath(dir, f)) for f in legacy_files)
        # Legacy entries count as the oldest run.
        @test find_latest(TopKCheckpointer(; dir), dir) == run_file(dir, run_of(b), 3)

        @testset "and `resume = :auto` into a legacy-only directory continues that run" begin
            dir2 = mktempdir()
            train!(Nitro(CkptMLP(); run_dir = dir2, max_epochs = 2))
            old2 = Old[Old((e.file, e.epoch, e.score, e.stop_reason)) for e in read_manifest(dir2)]
            JLD2.jldsave(joinpath(dir2, "manifest.jld2"); entries = old2)
            r = Nitro(CkptMLP(); run_dir = dir2, max_epochs = 3, resume = :auto)
            @test current_epoch(r) == 2
            @test run_of(r) === nothing                   # the unnamed run, adopted
            train!(r)
            @test runs_in(dir2) == Set([nothing])
            @test maximum(e.epoch for e in read_manifest(dir2)) == 3
        end
    end

    @testset "entries scored under another metric or mode are kept and never ranked" begin
        dir = mktempdir()
        a = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 3, checkpointer = k1()))
        a_files = run_files(dir, run_of(a))
        # The same run resumed, selecting the other way. Its earlier entries' scores were written
        # under `:min`, which `:max` cannot rank, so they stay.
        r = Nitro(
            CkptMLP(); run_dir = dir, max_epochs = 5, resume = :auto,
            checkpointer = TopKCheckpointer(; k = 1, mode = :max)
        )
        @test run_of(r) == run_of(a)
        train!(r)
        mine = run_entries(dir, run_of(a))
        @test issubset(a_files, Set(e.file for e in mine))
        @test all(isfile(joinpath(dir, f)) for f in a_files)
        maxed = [e for e in mine if e.mode === :max]
        # Ranked among themselves: the newest (epoch 5) and at most one best besides.
        @test 5 in Set(e.epoch for e in maxed) && issubset(Set(e.epoch for e in maxed), Set([4, 5]))
        # Selection ranks only entries under the checkpointer's own metric and mode.
        sel = ReactantNitro.selected_checkpoint(r.checkpointer, dir; run = run_of(r))
        @test sel.epoch in (4, 5)
        @test r.best_checkpoint.epoch in (4, 5)

        # And a FRESH run selecting on another mode leaves both of those alone too.
        before = Set(e.file for e in read_manifest(dir))
        train!(
            Nitro(
                CkptMLP(); run_dir = dir, max_epochs = 2, seed = 3,
                checkpointer = TopKCheckpointer(; k = 1, mode = :max)
            )
        )
        @test issubset(before, Set(e.file for e in read_manifest(dir)))
        @test all(isfile(joinpath(dir, f)) for f in before)
    end

    @testset "resuming from an earlier record branches; resuming from the newest continues" begin
        dir = mktempdir()
        a = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 3))
        a_files = run_files(dir, run_of(a))
        @test length(a_files) == 3
        stamps = Dict(f => mtime(joinpath(dir, f)) for f in a_files)

        # From A's newest record: that IS continuing A.
        cont = Nitro(CkptMLP(); run_dir = dir, max_epochs = 4, resume = run_file(dir, run_of(a), 3))
        @test run_of(cont) == run_of(a)

        # From epoch 1: a new run, since continuing A from there would rewrite A's epochs 2 and 3.
        # Deterministic, so the branch's epochs 2 and 3 carry the SAME names as A's.
        br = @test_logs (:info, r"branch") match_mode = :any Nitro(
            CkptMLP(); run_dir = dir, max_epochs = 3, resume = run_file(dir, run_of(a), 1)
        )
        @test run_of(br) isa String && run_of(br) != run_of(a)
        train!(br)
        @test run_files(dir, run_of(a)) == a_files
        @test all(mtime(joinpath(dir, f)) == stamps[f] for f in a_files)
        br_entries = run_entries(dir, run_of(br))
        @test Set(e.epoch for e in br_entries) == Set([2, 3])
        # The branch records where it came from.
        @test all(e.parent == run_file(dir, run_of(a), 1) for e in br_entries)

        # A file this manifest does not list is a branch too.
        other = mktempdir()
        train!(Nitro(CkptMLP(); run_dir = other, max_epochs = 1))
        ext = Nitro(CkptMLP(); run_dir = dir, max_epochs = 2, resume = find_latest(TopKCheckpointer(; dir = other), other))
        @test !(run_of(ext) in (run_of(a), run_of(br)))

        # `:auto` continues the most recent run, which is now the branch, and keeps its parent.
        r = Nitro(CkptMLP(); run_dir = dir, max_epochs = 4, resume = :auto)
        @test run_of(r) == run_of(br)
        train!(r)
        @test all(e.parent == run_file(dir, run_of(a), 1) for e in run_entries(dir, run_of(br)))
    end

    @testset "another handle as the source: `other => :current | :best | :latest`" begin
        dir = mktempdir()
        a = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 3, seed = 1))
        a_latest = find_latest(a.checkpointer, dir; run = run_of(a))
        a_best = ReactantNitro.selected_checkpoint(a.checkpointer, dir; run = run_of(a))
        # A newer run in A's directory, so "A's run" and "the most recent run" differ.
        b = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2, seed = 2))

        wl = Nitro(CkptMLP(); data = (; test = CK_VAL), run_dir = mktempdir(), weights = a => :latest)
        @test wl.checkpoint_source == a_latest
        @test params_of(wl) == params_of(a)
        @test current_epoch(wl) == 0
        wb = Nitro(CkptMLP(); data = (; test = CK_VAL), run_dir = mktempdir(), weights = a => :best)
        @test wb.checkpoint_source == a_best.path

        # `:current` is the in-memory warm start, exactly as the bare handle is.
        wc = Nitro(CkptMLP(); run_dir = mktempdir(), weights = a => :current)
        @test wc.checkpoint_source === nothing && wc.weights_source.epoch == 3
        @test params_of(wc) == params_of(a)
        wbare = Nitro(CkptMLP(); run_dir = mktempdir(), weights = a)
        @test params_of(wbare) == params_of(a) && wbare.weights_source.epoch == 3

        # A full resume from A's run's newest record, into A's own directory: that continues A.
        ra = Nitro(CkptMLP(); run_dir = dir, max_epochs = 4, resume = a => :latest)
        @test ra.checkpoint_source == a_latest
        @test current_epoch(ra) == 3
        @test run_of(ra) == run_of(a)
        rb = Nitro(CkptMLP(); run_dir = dir, max_epochs = 4, resume = a => :best)
        @test rb.checkpoint_source == a_best.path
        @test current_epoch(rb) == a_best.epoch
        a_best.epoch == 3 || @test run_of(rb) != run_of(a)

        msg(f) = try
            f(); ""
        catch ex
            ex isa ErrorException ? ex.msg : rethrow()
        end
        m = msg(() -> Nitro(CkptMLP(); run_dir = dir, resume = a => :current))
        @test occursin("not supported yet", m) && occursin("weights = ", m)
        m = msg(() -> Nitro(CkptMLP(); run_dir = dir, resume = a))
        @test occursin("not supported yet", m) && occursin("weights = ", m)
        @test occursin(":current", msg(() -> Nitro(CkptMLP(); run_dir = dir, weights = a => :newest)))
        @test occursin(":latest", msg(() -> Nitro(CkptMLP(); run_dir = dir, resume = a => :newest)))
        # A handle with no checkpointer has no records to name.
        bare = Nitro(CkptMLP(); run_dir = mktempdir(), checkpointer = nothing)
        @test occursin("no checkpointer", msg(() -> Nitro(CkptMLP(); run_dir = dir, weights = bare => :best)))
        @test occursin("no checkpointer", msg(() -> Nitro(CkptMLP(); run_dir = dir, resume = bare => :latest)))
        # And a handle that never wrote one has no such checkpoint.
        unrun = Nitro(CkptMLP(); run_dir = mktempdir())
        @test occursin("has no checkpoints", msg(() -> Nitro(CkptMLP(); run_dir = dir, weights = unrun => :latest)))
    end


    # ── `keep_latest = false`: the top K alone, and no write for an epoch outside it ──────
    #
    # The scores are scripted rather than taken from the fixture's loss curve, so which epochs are
    # top K is a property of the test and not of the optimizer. `finalize_metrics` runs once per
    # validation, host-side, and replaces the measured `val_loss` with the script's next value.
    @experiment struct ScriptedCkpt
        max_epochs::Host{Int} = 4
    end
    ReactantNitro.build_model(::ScriptedCkpt, rng) = (m = ckpt_chain(6); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::ScriptedCkpt, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::ScriptedCkpt, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.build_data(::ScriptedCkpt, dist) = (; train = CK_TRAIN, val = CK_VAL)
    const SCRIPT = Ref(Float64[])
    const SCRIPT_AT = Ref(0)
    script!(v) = (SCRIPT[] = Float64.(v); SCRIPT_AT[] = 0; nothing)
    ReactantNitro.finalize_metrics(::ScriptedCkpt, acc, ::Symbol) =
        (SCRIPT_AT[] += 1; (; val_loss = SCRIPT[][SCRIPT_AT[]]))
    drop_latest(k) = TopKCheckpointer(; k, keep_latest = false)
    epochs_of(dir, run) = Set(e.epoch for e in run_entries(dir, run))
    ckpt_files(dir) = Set(filter(f -> endswith(f, ".jld2") && f != "manifest.jld2", readdir(dir)))

    @testset "`keep_latest = false`, driven by hand: the top K only, and no write outside it" begin
        @test TopKCheckpointer().keep_latest === true            # the default is today's rule
        snap = snapshot(Nitro(CkptMLP(); run_dir = mktempdir(), checkpointer = nothing))
        # The name hook is called once per write, and only on the write path, so counting its
        # calls counts the records written, including any a rotation deleted again.
        named = Int[]
        counting(; epoch, kwargs...) = (push!(named, epoch); short_name(; epoch, kwargs...))

        dir = mktempdir()
        ck = TopKCheckpointer(; k = 2, dir, name = counting, keep_latest = false)
        # Worsening after epoch 2, so the default would keep the newest as a K+1th file.
        wrote = [
            save_checkpoint!(ck, epoch, (; val_loss = v), merge(snap, (; epoch)))
                for (epoch, v) in enumerate([2.0, 1.0, 3.0, 4.0])
        ]
        @test wrote == [true, true, false, false]
        @test named == [1, 2]                                    # epochs 3 and 4 were never written
        @test ckpt_files(dir) == Set(["epoch-0001.jld2", "epoch-0002.jld2"])
        @test Set(e.epoch for e in read_manifest(dir)) == Set([1, 2])
        # "Latest" is the newest RETAINED record.
        @test find_latest(ck, dir) == joinpath(dir, "epoch-0002.jld2")

        # An improvement enters, displacing the worst of the K; a tie with the K-th does not.
        @test save_checkpoint!(ck, 5, (; val_loss = 2.0), merge(snap, (; epoch = 5))) === false
        @test save_checkpoint!(ck, 6, (; val_loss = 0.5), merge(snap, (; epoch = 6))) === true
        @test ckpt_files(dir) == Set(["epoch-0002.jld2", "epoch-0006.jld2"])
        @test named == [1, 2, 6]

        # The final `stop_reason` rewrite: an epoch already in the top K is rewritten in place,
        # and one that was declined stays declined.
        fin = merge(snap, (; epoch = 6, stop_reason = :completed))
        @test save_checkpoint!(ck, 6, (; val_loss = 0.5), fin) === true
        @test load_checkpoint(ck, joinpath(dir, "epoch-0006.jld2")).stop_reason === :completed
        @test length(read_manifest(dir)) == 2
        @test save_checkpoint!(ck, 7, (; val_loss = 9.0), merge(fin, (; epoch = 7))) === false
        @test ckpt_files(dir) == Set(["epoch-0002.jld2", "epoch-0006.jld2"])

        @testset "`mode = :max` ranks the other way" begin
            d = mktempdir()
            c = TopKCheckpointer(;
                k = 1, metric = :acc, mode = :max, dir = d, name = short_name,
                keep_latest = false
            )
            got = [
                save_checkpoint!(c, epoch, (; acc = v), merge(snap, (; epoch)))
                    for (epoch, v) in enumerate([0.1, 0.9, 0.5])
            ]
            @test got == [true, true, false]
            @test ckpt_files(d) == Set(["epoch-0002.jld2"])
        end

        @testset "an unscored epoch cannot be kept, and says so" begin
            c = TopKCheckpointer(; k = 1, dir = mktempdir(), name = short_name, keep_latest = false)
            @test_throws "keep_latest = false" save_checkpoint!(c, 1, (;), merge(snap, (; epoch = 1)))
        end
    end

    @testset "the default still keeps the newest, whatever it scored, in a real run" begin
        dir = mktempdir()
        script!([3.0, 1.0, 2.0, 4.0])
        n = train!(Nitro(ScriptedCkpt(); run_dir = dir, checkpointer = TopKCheckpointer(; k = 1)))
        @test epochs_of(dir, run_of(n)) == Set([2, 4])
        @test load_checkpoint(n.checkpointer, run_file(dir, run_of(n), 4)).stop_reason === :completed
    end

    @testset "`keep_latest = false` in a real run: top K on disk, and `:latest` is the best" begin
        dir = mktempdir()
        script!([3.0, 1.0, 2.0, 4.0])
        n = @test_logs (:warn, r"keep_latest = false") match_mode = :any Nitro(
            ScriptedCkpt(); run_dir = dir, checkpointer = drop_latest(1)
        )
        train!(n)
        @test current_epoch(n) == 4
        @test epochs_of(dir, run_of(n)) == Set([2])
        @test ckpt_files(dir) == run_files(dir, run_of(n))      # no file outside the manifest
        best = run_file(dir, run_of(n), 2)
        # The final epoch was not top K, so the final rewrite wrote nothing: the one record is the
        # per-epoch write of epoch 2, which predates the outcome.
        @test load_checkpoint(n.checkpointer, best).stop_reason === nothing
        @test n.best_checkpoint.path == best
        @test find_latest(n.checkpointer, dir) == best

        nl = Nitro(ScriptedCkpt(); data = (; test = CK_VAL), run_dir = dir, weights = :latest => :latest)
        @test nl.checkpoint_source == best

        # `resume = :auto` continues the SAME run from its newest retained record, and says so.
        r = @test_logs (:warn, r"keep_latest = false.*built with `resume = :auto`"s) match_mode = :any Nitro(
            ScriptedCkpt(); run_dir = dir, max_epochs = 5, checkpointer = drop_latest(1),
            resume = :auto
        )
        @test r.checkpoint_source == best
        @test current_epoch(r) == 2
        @test run_of(r) == run_of(n)
        # Epochs 3 to 5 again: 3 improves on epoch 2 and replaces it, 4 and 5 are never written.
        script!([0.5, 5.0, 6.0])
        train!(r)
        @test runs_in(dir) == Set([run_of(n)])
        @test epochs_of(dir, run_of(n)) == Set([3])
        @test ckpt_files(dir) == run_files(dir, run_of(n))

        @testset "a final epoch that IS top K gets the `stop_reason` rewrite" begin
            d = mktempdir()
            script!([3.0, 2.0, 1.0])
            m = train!(Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 3, checkpointer = drop_latest(1)))
            @test epochs_of(d, run_of(m)) == Set([3])
            @test load_checkpoint(m.checkpointer, run_file(d, run_of(m), 3)).stop_reason === :completed
        end
    end

    @testset "`keep_latest = false`: refused with no `val` split, and quiet when not training" begin
        err = try
            Nitro(
                ScriptedCkpt(); data = (; train = CK_TRAIN), run_dir = mktempdir(),
                checkpointer = drop_latest(1)
            )
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("keep_latest = false", err.msg) && occursin("no `val` split", err.msg)
        # A handle that never trains writes nothing, so there is nothing to warn about.
        @test_logs min_level = Base.CoreLogging.Warn Nitro(
            ScriptedCkpt(); data = (; test = CK_VAL), run_dir = mktempdir(),
            checkpointer = drop_latest(1)
        )
    end

    # ── checkpoint sources: `run => checkpoint`, one grammar for `weights` and `resume` ──────
    #
    # Every expectation below is fixed by a script, so which record is best is a fact of the test.
    # Each resolution is checked three ways: the file the handle says it read, the manifest entry
    # that file belongs to, and the parameters the handle actually holds against that record's.
    record_params(path) = [Array(l) for l in Functors.fleaves(load_checkpoint(TopKCheckpointer(), path).ps)]
    host_leaf(x) = x isa AbstractArray ? Array(x) : x
    opt_leaves(os) = [host_leaf(x) for l in os for x in Functors.fleaves(to_host(l.state))]
    errmsg(f) = try
        f(); ""
    catch ex
        ex isa ErrorException ? ex.msg : rethrow()
    end
    ReactantNitro.presets(::Type{ScriptedCkpt}) = (quick = (; max_epochs = 4),)

    # Two runs in one directory. A's best (0.2 at epoch 2) beats B's (0.5 at epoch 1), so
    # `:all => :best` and `:latest => :best` must disagree, and every record is retained (k = 3).
    src_dir = mktempdir()
    script!([3.0, 0.2, 2.0])
    src_a = train!(Nitro(ScriptedCkpt(); run_dir = src_dir, max_epochs = 3, seed = 1))
    script!([0.5, 4.0, 5.0, 6.0])
    src_b = train!(Nitro(ScriptedCkpt(); run_dir = src_dir, max_epochs = 4, seed = 2))
    ida, idb = run_of(src_a), run_of(src_b)

    @testset "every `weights` source form resolves to the exact record" begin
        @test runs_in(src_dir) == Set([ida, idb])
        @test length(run_entries(src_dir, ida)) == 3 && length(run_entries(src_dir, idb)) == 4
        expect = [
            (:latest => :best) => run_file(src_dir, idb, 1),
            (:latest => :latest) => run_file(src_dir, idb, 4),
            (ida => :best) => run_file(src_dir, ida, 2),
            (ida => :latest) => run_file(src_dir, ida, 3),
            (idb => :best) => run_file(src_dir, idb, 1),
            (:all => :best) => run_file(src_dir, ida, 2),
            (src_a => :best) => run_file(src_dir, ida, 2),
            (src_a => :latest) => run_file(src_dir, ida, 3),
            (src_b => :best) => run_file(src_dir, idb, 1),
            run_file(src_dir, ida, 1) => run_file(src_dir, ida, 1),
        ]
        # The records really differ, or matching parameters would prove nothing.
        @test length(unique(record_params(last(p)) for p in expect)) == length(unique(last.(expect)))
        for (src, file) in expect
            n = Nitro(ScriptedCkpt(); data = (; test = CK_VAL), run_dir = src_dir, weights = src)
            @test n.checkpoint_source == file
            @test params_of(n) == record_params(file)
            @test current_epoch(n) == 0 && current_step(n) == 0
        end
        # A handle's newest record is its final weights, and `:current` is them in memory.
        @test record_params(run_file(src_dir, ida, 3)) == params_of(src_a)
        for w in (src_a => :current, src_a)
            n = Nitro(ScriptedCkpt(); data = (; test = CK_VAL), run_dir = mktempdir(), weights = w)
            @test n.checkpoint_source === nothing && n.weights_source.epoch == 3
            @test params_of(n) == params_of(src_a)
        end
        # Another handle's run is read in ITS directory, whatever `run_dir` this handle has.
        far = Nitro(ScriptedCkpt(); data = (; test = CK_VAL), run_dir = mktempdir(), weights = src_a => :best)
        @test far.checkpoint_source == run_file(src_dir, ida, 2)
        # A directory holding two runs says which it read for `:latest`.
        @test_logs (:info, r"holds 2 runs") match_mode = :any Nitro(
            ScriptedCkpt(); data = (; test = CK_VAL), run_dir = src_dir, weights = :latest => :best
        )
    end

    @testset "every `resume` source form resolves, continuing or branching as it must" begin
        cases = [
            # (source, the file, the epoch it restores, whether it continues that run)
            (:latest => :latest, run_file(src_dir, idb, 4), 4, idb),
            (:latest => :best, run_file(src_dir, idb, 1), 1, nothing),
            (ida => :latest, run_file(src_dir, ida, 3), 3, ida),
            (ida => :best, run_file(src_dir, ida, 2), 2, nothing),
            (:all => :best, run_file(src_dir, ida, 2), 2, nothing),
            (src_a => :latest, run_file(src_dir, ida, 3), 3, ida),
            (src_b => :best, run_file(src_dir, idb, 1), 1, nothing),
            (run_file(src_dir, idb, 4), run_file(src_dir, idb, 4), 4, idb),
        ]
        for (src, file, epoch, continues) in cases
            r = Nitro(ScriptedCkpt(); run_dir = src_dir, max_epochs = 6, resume = src)
            rec = load_checkpoint(r.checkpointer, file)
            @test r.checkpoint_source == file
            @test current_epoch(r) == epoch && current_step(r) == rec.step
            @test params_of(r) == record_params(file)
            @test opt_leaves(r.opt_state) == opt_leaves(rec.opt_state)
            if continues === nothing
                # Not the run's newest record: a branch, with a new id and the source as parent.
                @test !(run_of(r) in (ida, idb)) && r.checkpointer.parent == file
            else
                @test run_of(r) == continues
            end
        end
    end

    @testset "sources that are refused say what to write instead" begin
        mk(; kw...) = Nitro(ScriptedCkpt(); run_dir = src_dir, kw...)
        m = errmsg(() -> mk(; weights = :best))
        @test occursin("`weights = :latest => :best`", m) && occursin("`weights = :all => :best`", m)
        @test occursin("`weights = :latest => :latest`", errmsg(() -> mk(; weights = :latest)))
        @test occursin("`resume = :latest => :best`", errmsg(() -> mk(; resume = :best)))
        @test occursin("`resume = :latest => :latest`", errmsg(() -> mk(; resume = :latest)))
        m = errmsg(() -> mk(; weights = :all => :latest))
        @test occursin("`weights = :latest => :latest`", m)
        @test occursin("`resume = :latest => :latest`", errmsg(() -> mk(; resume = :all => :latest)))
        @test occursin("needs a `Nitro` on the left", errmsg(() -> mk(; weights = :all => :current)))
        @test occursin("needs a `Nitro` on the left", errmsg(() -> mk(; weights = :latest => :current)))
        @test occursin("names no run", errmsg(() -> mk(; weights = :recent => :best)))
        m = errmsg(() -> mk(; weights = "nosuchid" => :best))
        @test occursin("`nosuchid`", m) && occursin(src_dir, m)
        @test occursin(ida, m) && occursin(idb, m) && occursin("runs(\"$(src_dir)\")", m)
        m = errmsg(() -> mk(; resume = "nosuchid" => :latest))
        @test occursin("`nosuchid`", m) && occursin(ida, m) && occursin("runs(", m)
        for m in (errmsg(() -> mk(; resume = src_a => :current)), errmsg(() -> mk(; resume = src_a)))
            @test occursin("not supported yet", m) && occursin("weights = other", m)
        end
        # A directory with nothing in it names itself and says it holds nothing.
        empty = mktempdir()
        m = errmsg(() -> Nitro(ScriptedCkpt(); run_dir = empty, resume = :latest => :latest))
        @test occursin("found no checkpoint", m) && occursin(empty, m) && occursin("runs(", m)
    end

    @testset "`resume = :auto` is `:latest => :latest`, and starts fresh with a warning on nothing" begin
        latest = run_file(src_dir, idb, 4)
        rec = load_checkpoint(TopKCheckpointer(), latest)
        said = Regex("`resume = :auto`: continuing `:latest => :latest`, run `$(idb)`, epoch 4")
        r = @test_logs (:info, said) match_mode = :any Nitro(
            ScriptedCkpt(); run_dir = src_dir, max_epochs = 6, resume = :auto
        )
        # Exactly the record, and the same run continued.
        @test r.checkpoint_source == latest
        @test run_of(r) == idb
        @test current_step(r) == rec.step && current_epoch(r) == rec.epoch == 4
        @test params_of(r) == record_params(latest)
        @test opt_leaves(r.opt_state) == opt_leaves(rec.opt_state)
        # And that is not what a fresh optimizer holds, so the comparison above has teeth.
        fresh = Nitro(ScriptedCkpt(); run_dir = mktempdir(), max_epochs = 6)
        @test opt_leaves(fresh.opt_state) != opt_leaves(rec.opt_state)
        # It continues training the same run from there.
        script!([7.0, 8.0])
        train!(r)
        @test runs_in(src_dir) == Set([ida, idb])
        @test maximum(e.epoch for e in run_entries(src_dir, idb)) == 6

        empty = mktempdir()
        f = @test_logs (:warn, r"found nothing to resume") match_mode = :any Nitro(
            ScriptedCkpt(); run_dir = empty, resume = :auto
        )
        @test current_epoch(f) == 0 && f.checkpoint_source === nothing
    end

    # ── `restore_optimizer`: a record's parameters and optimizer state, under a new schedule ──

    @experiment struct ScriptedAdam
        max_epochs::Host{Int} = 2
    end
    ReactantNitro.build_model(::ScriptedAdam, rng) = (m = ckpt_chain(6); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::ScriptedAdam, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::ScriptedAdam, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.build_data(::ScriptedAdam, dist) = (; train = CK_TRAIN, val = CK_VAL)
    ReactantNitro.optimizer(::ScriptedAdam) = Optimisers.Adam

    @testset "`restore_optimizer = true` takes the record's optimizer state and nothing else" begin
        file = run_file(src_dir, ida, 2)
        rec = load_checkpoint(TopKCheckpointer(), file)
        @test rec.step == 8 && rec.epoch == 2
        # `run_dir` is where a run id is looked up; nothing is written, since nothing trains.
        for src in (file, ida => :best, src_a => :best)
            n = Nitro(
                ScriptedCkpt(); run_dir = src_dir, max_epochs = 2, weights = src,
                restore_optimizer = true
            )
            @test n.checkpoint_source == file
            @test params_of(n) == record_params(file)
            @test opt_leaves(n.opt_state) == opt_leaves(rec.opt_state)
            # A new run under a new schedule: counters at zero, the horizon this call's.
            @test current_step(n) == 0 && current_epoch(n) == 0
            @test n.total == 2 * length(CK_TRAIN)
            ReactantNitro.assert_opt_state_device(n.opt_state)
        end
        # Without the flag the optimizer is fresh.
        plain = Nitro(ScriptedCkpt(); run_dir = mktempdir(), max_epochs = 2, weights = file)
        @test opt_leaves(plain.opt_state) != opt_leaves(rec.opt_state)
        @test opt_leaves(plain.opt_state) == opt_leaves(Nitro(ScriptedCkpt(); run_dir = mktempdir()).opt_state)
        # And it trains, as a new run from step 0.
        d = mktempdir()
        n = Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 2, weights = file, restore_optimizer = true)
        script!([1.0, 2.0])
        train!(n)
        @test current_epoch(n) == 2 && current_step(n) == 2 * length(CK_TRAIN)
        @test !(run_of(n) in (ida, idb))

        mk(; kw...) = Nitro(ScriptedCkpt(); run_dir = src_dir, kw...)
        for kw in ((;), (; weights = src_a), (; weights = src_a => :current))
            m = errmsg(() -> mk(; restore_optimizer = true, kw...))
            @test occursin("needs `weights` to name one", m) && occursin("not supported yet", m)
        end
        @test occursin("meaningless with `resume`", errmsg(() -> mk(; resume = :auto, restore_optimizer = true)))
        @test occursin(
            "no `train` split",
            errmsg(() -> mk(; data = (; test = CK_VAL), weights = file, restore_optimizer = true))
        )
        # A record written under another optimizer does not fit, and says which group and rule.
        m = errmsg(() -> Nitro(ScriptedAdam(); run_dir = mktempdir(), weights = file, restore_optimizer = true))
        @test occursin("does not fit", m) && occursin("RAdam", m) && occursin("Adam", m)
    end

    # ── `restore_best`: the handle ends `train!` holding its run's best checkpoint ─────────

    @testset "`restore_best = true` loads the run's best, not its last, and says so" begin
        dir = mktempdir()
        script!([3.0, 1.0, 2.0, 4.0])                      # best is epoch 2, not the last
        n = Nitro(ScriptedCkpt, :quick; run_dir = dir, restore_best = true)
        train!(n)
        id = run_of(n)
        best, last4 = run_file(dir, id, 2), run_file(dir, id, 4)
        @test record_params(best) != record_params(last4)
        @test params_of(n) == record_params(best)
        @test n.checkpoint_source == best
        @test n.restored_best.epoch == 2 && n.restored_best.path == best
        # The trajectory is as training ended.
        @test current_epoch(n) == 4 && current_step(n) == 4 * length(CK_TRAIN)
        @test length(history(n)) == 4 && n.last_metrics.val_loss == 4.0
        # The display says it, and names the run.
        s = sprint(show, MIME"text/plain"(), n)
        @test occursin("restored to the best checkpoint, epoch 2", s)
        @test occursin("the handle holds these weights", s)
        @test occursin("run " * id, s)
        # Export provenance names the file the weights came from.
        @test export_provenance(n)["checkpoint"] == best

        # A second `train!` explains itself and prints the three real states, filled in.
        m = errmsg(() -> train!(n))
        @test occursin("cannot train again", m)
        @test occursin("Nitro(ScriptedCkpt, :quick; run_dir = $(repr(dir)), weights = \"$(id)\" => :best)", m)
        @test occursin("weights = \"$(id)\" => :best, restore_optimizer = true)", m)
        @test occursin("Nitro(ScriptedCkpt, :quick; run_dir = $(repr(dir)), resume = \"$(id)\" => :best)", m)
        @test occursin("weights = n => :best", m)
        # Each suggestion is a working construction of the state it names.
        w = Nitro(ScriptedCkpt, :quick; run_dir = dir, weights = id => :best)
        @test w.checkpoint_source == best && params_of(w) == record_params(best)
        wo = Nitro(ScriptedCkpt, :quick; run_dir = dir, weights = id => :best, restore_optimizer = true)
        @test opt_leaves(wo.opt_state) == opt_leaves(load_checkpoint(TopKCheckpointer(), best).opt_state)
        rs = Nitro(ScriptedCkpt, :quick; run_dir = dir, resume = id => :best)
        @test current_epoch(rs) == 2 && run_of(rs) != id
        @test params_of(Nitro(ScriptedCkpt(); run_dir = dir, weights = n => :best)) == record_params(best)
        # Without a preset the message builds the experiment itself.
        script!([1.0, 2.0])
        np = train!(Nitro(ScriptedCkpt(); run_dir = mktempdir(), max_epochs = 2, restore_best = true))
        @test occursin("Nitro(ScriptedCkpt(); run_dir = ", errmsg(() -> train!(np)))

        @testset "through `train!(e; ...)`, and on an early stop and a requested stop" begin
            script!([2.0, 1.0, 3.0])
            t = train!(ScriptedCkpt(); run_dir = mktempdir(), max_epochs = 3, restore_best = true)
            @test t.restored_best.epoch == 2
            @test params_of(t) == record_params(run_file(t.run_dir, run_of(t), 2))

            script!([1.0, 2.0, 3.0, 4.0])
            es = train!(
                Nitro(
                    ScriptedCkpt(); run_dir = mktempdir(), restore_best = true,
                    early_stop = EarlyStopping(; patience = 1)
                )
            )
            @test es.stop_reason === :early_stop && current_epoch(es) < 4
            @test es.restored_best.epoch == 1
            @test params_of(es) == record_params(run_file(es.run_dir, run_of(es), 1))

            script!([1.0, 2.0, 3.0, 4.0])
            rq = Nitro(ScriptedCkpt(); run_dir = mktempdir(), restore_best = true)
            register_phase_monitor!(
                rq, (ph, step, epoch, info) -> (ph isa Checkpointing && epoch == 2 && request_stop!(rq); nothing)
            )
            train!(rq)
            @test rq.stop_reason === :requested && current_epoch(rq) == 2
            @test rq.restored_best.epoch == 1
            @test params_of(rq) == record_params(run_file(rq.run_dir, run_of(rq), 1))
        end

        @testset "no scored checkpoint: a warning, the final weights, and training may go on" begin
            u = Nitro(
                ScriptedCkpt(); data = (; train = CK_TRAIN), run_dir = mktempdir(), max_epochs = 1,
                restore_best = true
            )
            @test_logs (:warn, r"saved no scored checkpoint") match_mode = :any train!(u)
            @test u.restored_best === nothing && u.checkpoint_source === nothing
            @test params_of(u) == record_params(run_file(u.run_dir, run_of(u), 1))
            u.max_epochs = 2
            train!(u)
            @test current_epoch(u) == 2
        end
    end

    # ── mixed metrics: selection sees the current metric only, and never prunes the others ──

    # Entries under `:acc`, written by hand into a directory as run `accrun01`: an experiment that
    # emits only `val_loss` cannot score them itself, which is exactly the situation to cover.
    acc_name(; epoch, kwargs...) = "acc-" * lpad(epoch, 4, '0') * ".jld2"
    function acc_run!(dir)
        snap = snapshot(Nitro(ScriptedCkpt(); run_dir = mktempdir(), checkpointer = nothing))
        hand = TopKCheckpointer(; k = 1, metric = :acc, mode = :max, dir, name = acc_name)
        hand.run = "accrun01"
        for (epoch, v) in enumerate([0.1, 0.9, 0.5])
            save_checkpoint!(hand, epoch, (; acc = v), merge(snap, (; epoch)))
        end
        return Set(["acc-0002.jld2", "acc-0003.jld2"])     # the best, and the newest
    end

    @testset "other-metric checkpoints are kept, warned about, and never selected" begin
        dir = mktempdir()
        acc_files = acc_run!(dir)
        @test run_files(dir, "accrun01") == acc_files
        # A `val_loss` run in the same directory, retention over the whole directory: the `acc`
        # entries are in its scope, so setup says they are mixed, and training prunes around them.
        script!([3.0, 1.0, 2.0])
        n = @test_logs (:warn, r"retention considers checkpoints scored under other metrics: `acc` \(max\) x2") match_mode = :any Nitro(
            ScriptedCkpt(); run_dir = dir, max_epochs = 3,
            checkpointer = TopKCheckpointer(; k = 1, scope = :dir)
        )
        train!(n)
        @test run_files(dir, "accrun01") == acc_files
        @test all(isfile(joinpath(dir, f)) for f in acc_files)
        @test epochs_of(dir, run_of(n)) == Set([2, 3])
        # `:all => :best` considers them, says so, and picks among `val_loss` entries only.
        w = @test_logs (:warn, r"other metrics: `acc` \(max\) x2.*kept"s) match_mode = :any Nitro(
            ScriptedCkpt(); data = (; test = CK_VAL), run_dir = dir, weights = :all => :best
        )
        @test w.checkpoint_source == run_file(dir, run_of(n), 2)
        # A run with nothing under the current metric cannot answer `:best`, and says what it has.
        m = errmsg(() -> Nitro(ScriptedCkpt(); data = (; test = CK_VAL), run_dir = dir, weights = "accrun01" => :best))
        @test occursin("`val_loss` (min)", m) && occursin("`acc` (max) x2", m)
        @test occursin("Set the checkpointer's `metric`", m) && occursin("checkpoint's path", m)
        # Selecting on `acc` reads them.
        wa = Nitro(
            ScriptedCkpt(); data = (; test = CK_VAL), run_dir = dir, weights = "accrun01" => :best,
            checkpointer = TopKCheckpointer(; metric = :acc, mode = :max)
        )
        @test wa.checkpoint_source == joinpath(dir, "acc-0002.jld2")
        # `runs` reports the mix per run.
        rs = Dict(r.id => r for r in runs(dir))
        @test rs["accrun01"].metric === :acc && rs["accrun01"].best == 0.9
        @test rs[run_of(n)].metric === :val_loss && rs[run_of(n)].best == 1.0
    end

    @testset "a resume under another mode keeps the run's earlier entries through pruning" begin
        dir = mktempdir()
        script!([3.0, 1.0, 2.0])
        a = train!(Nitro(ScriptedCkpt(); run_dir = dir, max_epochs = 3, checkpointer = TopKCheckpointer(; k = 1)))
        min_files = run_files(dir, run_of(a))
        @test epochs_of(dir, run_of(a)) == Set([2, 3])
        r = @test_logs (:warn, r"this run's retention considers checkpoints scored under other metrics: `val_loss` \(min\) x2") match_mode = :any Nitro(
            ScriptedCkpt(); run_dir = dir, max_epochs = 6, resume = :auto,
            checkpointer = TopKCheckpointer(; k = 1, mode = :max)
        )
        @test run_of(r) == run_of(a)
        script!([5.0, 9.0, 7.0])
        train!(r)
        mine = run_entries(dir, run_of(a))
        @test issubset(min_files, Set(e.file for e in mine))
        @test all(isfile(joinpath(dir, f)) for f in min_files)
        @test Set(e.epoch for e in mine if e.mode === :max) == Set([5, 6])
        # Each selection ranks its own pair only.
        mx = @test_logs (:warn, r"other metrics") match_mode = :any Nitro(
            ScriptedCkpt(); data = (; test = CK_VAL), run_dir = dir, weights = :latest => :best,
            checkpointer = TopKCheckpointer(; mode = :max)
        )
        @test mx.checkpoint_source == run_file(dir, run_of(a), 5)
        mn = Nitro(ScriptedCkpt(); data = (; test = CK_VAL), run_dir = dir, weights = :latest => :best)
        @test mn.checkpoint_source == run_file(dir, run_of(a), 2)
    end

    # ── `scope = :dir`: the top K across the whole directory ────────────────────────────

    @testset "`scope = :dir` keeps the directory's top K plus this run's newest" begin
        @test TopKCheckpointer().scope === :run
        @test_throws "must be `:run`" TopKCheckpointer(; scope = :everywhere)
        dir = mktempdir()
        dirk2() = TopKCheckpointer(; k = 2, scope = :dir)
        script!([5.0, 2.0, 6.0, 7.0])
        a = train!(Nitro(ScriptedCkpt(); run_dir = dir, checkpointer = dirk2()))
        @test epochs_of(dir, run_of(a)) == Set([1, 2, 4])      # alone, it is the per-run rule
        a2 = run_file(dir, run_of(a), 2)
        script!([3.0, 1.0, 8.0])
        b = @test_logs (:warn, r"`scope = :dir`.*3 checkpoint\(s\) of 1 other run\(s\)"s) match_mode = :any Nitro(
            ScriptedCkpt(); run_dir = dir, max_epochs = 3, checkpointer = dirk2(), seed = 7
        )
        train!(b)
        # Top 2 across both runs (B's 1.0, A's 2.0) plus B's newest; A's others are gone.
        @test epochs_of(dir, run_of(a)) == Set([2])
        @test epochs_of(dir, run_of(b)) == Set([2, 3])
        @test ckpt_files(dir) == union(run_files(dir, run_of(a)), run_files(dir, run_of(b)))
        @test length(ckpt_files(dir)) == 3
        @test isfile(a2) && record_params(a2) == record_params(run_file(dir, run_of(a), 2))
        @test Set(e.score for e in read_manifest(dir)) == Set([2.0, 1.0, 8.0])

        @testset "under `keep_latest = false`, at most K files in the directory" begin
            d = mktempdir()
            ck() = TopKCheckpointer(; k = 2, scope = :dir, keep_latest = false)
            script!([5.0, 2.0, 6.0])
            a = train!(Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 3, checkpointer = ck()))
            @test epochs_of(d, run_of(a)) == Set([1, 2])
            script!([3.0, 1.0, 8.0])
            b = train!(Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 3, checkpointer = ck(), seed = 7))
            @test epochs_of(d, run_of(a)) == Set([2]) && epochs_of(d, run_of(b)) == Set([2])
            @test length(ckpt_files(d)) == 2 == length(read_manifest(d))
        end

        @testset "other-metric files survive `:dir` pruning" begin
            d = mktempdir()
            acc_files = acc_run!(d)
            ck() = TopKCheckpointer(; k = 1, scope = :dir, keep_latest = false)
            script!([2.0, 1.0, 3.0])
            train!(Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 3, checkpointer = ck()))
            @test issubset(acc_files, ckpt_files(d))
            @test length(setdiff(ckpt_files(d), acc_files)) == 1     # K = 1 for `val_loss`
        end

        @testset "identical names across runs never overwrite" begin
            d = mktempdir()
            ck() = TopKCheckpointer(; k = 3, scope = :dir)
            script!([2.0, 1.0])
            a = train!(Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 2, checkpointer = ck()))
            a_files = run_files(d, run_of(a))
            stamps = Dict(f => mtime(joinpath(d, f)) for f in a_files)
            script!([2.0, 1.0])
            b = train!(Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 2, checkpointer = ck()))
            @test params_of(a) == params_of(b)                    # the same names, epoch for epoch
            @test run_files(d, run_of(a)) == a_files
            @test all(mtime(joinpath(d, f)) == stamps[f] for f in a_files)
            b_files = run_files(d, run_of(b))
            @test !isempty(b_files) && all(occursin("-run-$(run_of(b))", f) for f in b_files)
            @test isempty(intersect(a_files, b_files))
            @test epochs_of(d, run_of(b)) == Set([2])             # B's tie at epoch 1 ranks after A's
        end
    end

    # ── finding run ids ─────────────────────────────────────────────────────────────────

    @testset "`runs(dir)`, `checkpoint_run(n)` and `show(n)` name the runs" begin
        rs = runs(src_dir)
        @test length(rs) == 2
        @test [r.id for r in rs] == [idb, ida]                     # most recent first
        ra = rs[2]
        @test ra.checkpoints == length(run_entries(src_dir, ida)) == 3
        @test ra.epochs == [1, 2, 3]
        @test ra.metric === :val_loss && ra.mode === :min && ra.best == 0.2
        @test ra.parent === nothing
        @test rs[1].checkpoints == length(run_entries(src_dir, idb))
        @test rs[1].written >= ra.written
        line = sprint(show, ra)
        @test occursin("run $(ida)", line) && occursin("3 checkpoints", line) && occursin("0.2", line)
        @test occursin(ida, sprint(show, MIME"text/plain"(), rs))
        @test isempty(runs(mktempdir()))
        @test checkpoint_run(src_a) == ida && checkpoint_run(src_b) == idb
        @test checkpoint_run(Nitro(ScriptedCkpt(); run_dir = mktempdir(), checkpointer = nothing)) === nothing
        s = sprint(show, MIME"text/plain"(), src_a)
        @test occursin("run $(ida)", s)

        # The unnamed run of an old manifest is `"legacy"`, and that id resolves.
        d = mktempdir()
        script!([2.0, 1.0])
        train!(Nitro(ScriptedCkpt(); run_dir = d, max_epochs = 2))
        Old = @NamedTuple{
            file::String, epoch::Int, score::Union{Float64, Nothing},
            stop_reason::Union{Symbol, Nothing},
        }
        JLD2.jldsave(
            joinpath(d, "manifest.jld2");
            entries = Old[Old((e.file, e.epoch, e.score, e.stop_reason)) for e in read_manifest(d)]
        )
        lr = only(runs(d))
        @test lr.id == "legacy" && lr.checkpoints == 2 && lr.written == 0.0
        @test occursin("before runs were recorded", sprint(show, lr))
        wl = Nitro(ScriptedCkpt(); data = (; test = CK_VAL), run_dir = d, weights = "legacy" => :best)
        @test wl.checkpoint_source == run_file(d, nothing, 2)
    end

end
