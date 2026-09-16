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

            # Weights-only (`checkpoint = path`): declining the logger is legal and reattaches nothing.
            n = Nitro(CkptMLP(); run_dir = dir, checkpoint = latest, logger = nothing)
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
                Nitro(CkptMLP(); run_dir = dir, checkpoint = latest, logger = HandleLog())
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

    @testset "`checkpoint = path` restores weights, not a trajectory" begin
        dir = mktempdir()
        trained = train!(Nitro(CkptMLP(); run_dir = dir, max_epochs = 2))
        path = find_latest(trained.checkpointer, dir)

        served = Nitro(CkptMLP(); data = (; test = CK_VAL), checkpoint = path, run_dir = mktempdir())
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
            Nitro(CkptMLP(); checkpoint = path, checkpointer = nothing, run_dir = mktempdir())
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("has no loader", err.msg)
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

end
