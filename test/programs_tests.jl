# Setup sequence and training-program tests.
#
# The acceptance criterion this file exercises: "A fixed-LR loop trains a two-layer MLP for two
# epochs on CPU with **no recompile after step 1**, verified by the cache counter; green".

@testitem "programs" begin
    using Test
    using ReactantNitro
    using ReactantNitro: cache_reset!, cache_stats, flatten, n_groups, to_device_batch
    using Functors, Lux, Optimisers, Random, Reactant, Statistics

    const FT = Float32

    @experiment struct MLP
        "Traced input: a loss weight, varied without recompiling."
        scale::Device{Float32} = 1.0f0
        "Structural: bakes into the graph and enters the cache key."
        width::GraphConst{Int} = 8
        "Driver-only: invisible to the tracer."
        max_epochs::Host{Int} = 2
    end

    ReactantNitro.build_model(e::MLP, rng) =
        Lux.setup(rng, Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dense(e.width => 3))) |>
        ((ps, st),) -> (Lux.Chain(Lux.Dense(4 => e.width, tanh), Lux.Dense(e.width => 3)), ps, st)

    ReactantNitro.forward(::MLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(e::MLP, ŷ; y) = e.scale * mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::MLP) = 1.0f-2

    function mlp_data(; n_batches = 4, bs = 8, seed = 1)
        rng = Random.MersenneTwister(seed)
        return [(; x = randn(rng, FT, 4, bs), y = randn(rng, FT, 3, bs)) for _ in 1:n_batches]
    end
    ReactantNitro.build_data(::MLP, dist) = (; train = mlp_data())

    @testset "the setup sequence" begin
        cache_reset!()
        n = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())

        @testset "the accessors read what setup produced" begin
            @test experiment(n) isa MLP
            @test parameters(n) !== nothing
            @test states(n) !== nothing
            @test current_step(n) == 0
            @test current_epoch(n) == 0
            @test phase(n) isa Starting
            @test binding_report(n) isa String
        end

        @testset "step 5 converted the Device and left everything else alone" begin
            e = experiment(n)
            @test e.scale isa Reactant.RNumber      # Device -> device
            @test e.width === 8                     # GraphConst -> untouched, bakes
            @test e.max_epochs === 2                # Host -> untouched, invisible to the tracer
            @test typeof(e) !== typeof(MLP())       # "e's type changes here"
        end

        @testset "`set_device!` writes a live handle and reuses the compiled program" begin
            n2 = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
            b = first(mlp_data())
            predict(n2, (; x = b.x))                        # compiles `fwd_program`
            m0 = cache_stats().misses
            @test m0 > 0                                    # a control: something really compiled

            set_device!(n2; scale = 2.0f0)
            predict(n2, (; x = b.x))
            # THE PROMISE. A Device is a traced input, excluded from the compile cache's key by
            # construction, so the write cannot have moved the key and the second `predict` must
            # have hit.
            @test cache_stats().misses == m0
            @test device_value(n2, :scale) ≈ 2.0f0

            # A host value of a different numeric type CONVERTS rather than recompiling: it is the device
            # type that must match, since `typeof(compile_view(e))` is `args[1]`'s type at every site.
            set_device!(n2, :scale, 3.0)                    # Float64 in, Float32 field
            predict(n2, (; x = b.x))
            @test cache_stats().misses == m0
            @test device_value(n2, :scale) ≈ 3.0f0
            @test experiment(n2).scale isa Reactant.RNumber  # still device-resident
        end

        @testset "and it refuses exactly what would break that promise" begin
            n3 = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
            # A GraphConst field BAKES and is hashed, so changing it genuinely is a different program.
            @test_throws ErrorException set_device!(n3, :width, 16)
            # A Host field reaches no trace at all, so writing it here would change nothing compiled.
            @test_throws ErrorException set_device!(n3, :max_epochs, 5)
            @test_throws ErrorException set_device!(n3, :not_a_field, 1)
            @test_throws ErrorException set_device!(n3, :scale, "not a number")
            @test_throws ErrorException set_device!(n3)                    # no fields named
            # The refusals are not partial writes: the experiment is untouched.
            @test experiment(n3).width === 8
            @test experiment(n3).max_epochs === 2
            # `device_value` shares the same gate, so the two cannot disagree about what is settable.
            @test_throws ErrorException device_value(n3, :width)
        end

        @testset "step 7's w0 is the parameters as build_model returned them" begin
            @test n.w0 !== nothing
            @test all(Array.(Functors.fleaves(n.w0)) .== Array.(Functors.fleaves(parameters(n))))
        end

        @testset "step 8 is unconditional and step 9 is training only" begin
            @test n_groups(n.layout) == 1
            @test n.opt_state !== nothing           # this Nitro HAS a train split
            eval_only = Nitro(
                MLP(); data = (; test = mlp_data()), checkpointer = nothing,
                run_dir = mktempdir()
            )
            @test eval_only.layout !== nothing      # step 8 STILL RUNS: the resume check needs it
            @test eval_only.opt_state === nothing   # step 9 skipped: the moments would be twice the
            @test eval_only.total === nothing       #   parameter memory on device, for nothing
            @test eval_only.schedules === nothing   # step 11 skipped: `total` is undefined
            @test eval_only.batch_size == 8         # inferred from whichever split exists
        end

        @testset "step 10 resolved routing against the STRIPPED type" begin
            @test keys(n.routing.forward) == (:x,)
            @test keys(n.routing.loss) == (:y,)
            @test n.routing.metrics === nothing     # no method: the framework substitutes
            @test n.schema == (:x, :y)
        end

        @testset "step 11's horizon is an exact division" begin
            @test n.total == 2 * div(4, 1)          # max_epochs * div(steps_per_epoch, accum)
            @test n.accum == 1
        end

        @testset "`build_data` and `derive` see the PRE-conversion experiment" begin
            seen = Ref{Any}(nothing)
            @experiment struct PreConv
                w::Device{Float32} = 2.0f0
                n::Int = 1
            end
            @eval ReactantNitro.build_data(e::PreConv, dist) =
                ($seen[] = e.w; (; train = $(mlp_data)()))
            @eval ReactantNitro.build_model(e::PreConv, rng) =
                Lux.setup(rng, Lux.Dense(4 => 3)) |> ((ps, st),) -> (Lux.Dense(4 => 3), ps, st)
            @eval ReactantNitro.forward(::PreConv, model, ps, st; x) = Lux.apply(model, x, ps, st)
            @eval ReactantNitro.loss(::PreConv, ŷ; y) = mean(abs2, ŷ .- y)
            p = Nitro(PreConv(); checkpointer = nothing, run_dir = mktempdir())
            @test seen[] === 2.0f0                    # a host value, not a device scalar
            @test experiment(p).w isa Reactant.RNumber   # and every later hook sees the converted one
        end
    end

    @testset "fixed-LR loop: two epochs, and NO RECOMPILE AFTER STEP 1" begin
        cache_reset!()
        n = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
        ps0 = deepcopy(Array.(Functors.fleaves(parameters(n))))

        train!(n)

        @test current_epoch(n) == 2
        @test current_step(n) == 8                  # 2 epochs x 4 batches, accum = 1
        @test phase(n) isa Done

        @testset "exactly two programs were compiled, and never again" begin
            # The gradient program and the optimizer program. There is NO FUSED PROGRAM even at
            # accum = 1, so this is 2 rather than 1, deliberately.
            st = cache_stats()
            @test st.misses == 2
            @test st.entries == 2
            # 8 optimizer steps and 8 micro-batches, so 16 lookups, 2 of which missed.
            @test st.hits == 14
        end

        @testset "it actually trained" begin
            ps1 = Array.(Functors.fleaves(parameters(n)))
            @test any(!=(0), reduce(vcat, vec.(ps1)) .- reduce(vcat, vec.(ps0)))
            @test all(isfinite, reduce(vcat, vec.(ps1)))
        end

        @testset "a second train! over a fresh Nitro reuses every compiled program" begin
            before = cache_stats().misses
            n2 = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
            train!(n2)
            @test cache_stats().misses == before    # the cache is module-level
        end

        @testset "changing a Device does not recompile; changing a GraphConst field does" begin
            before = cache_stats().misses
            train!(Nitro(MLP(; scale = 3.0f0); checkpointer = nothing, run_dir = mktempdir()))
            @test cache_stats().misses == before    # a traced input cannot affect the graph

            train!(Nitro(MLP(; width = 16); checkpointer = nothing, run_dir = mktempdir()))
            @test cache_stats().misses == before + 2  # structural: both programs are new
        end
    end

    @testset "a scheduled experiment Device field is LIVE in the loop" begin
        # The regression this exists for: a scheduled Device field was specified, reported by the
        # binding report, and
        # had no caller in the run path, so a user who scheduled an experiment field got a report saying
        # it was scheduled and a run in which it never moved. A framework that vouches for something it
        # is not doing is worse than one that does not offer it.
        #
        # `scale` multiplies the loss (see `ReactantNitro.loss(::MLP, ...)`), so a schedule on it is
        # observable in the gradient rather than only in the field.
        cache_reset!()
        sched = (; device = (; scale = total -> (t -> FT(t / total))))
        n = Nitro(MLP(); schedules = sched, checkpointer = nothing, run_dir = mktempdir())
        train!(n)

        # The horizon is 2 epochs x 4 batches / accum 1 = 8, and the rebuild runs for the UPCOMING step,
        # so the last one built for step 8 and left the field at 8/8.
        @test device_value(n, :scale) ≈ 1.0f0

        @testset "and it did not cost a recompile" begin
            # The whole claim: a Device field is a traced INPUT, so writing a fresh one per step
            # re-enters the same program. Still exactly the gradient program and the optimizer program.
            st = cache_stats()
            @test st.misses == 2
            @test st.entries == 2
        end

        @testset "the ramp reached the parameters, not only the field" begin
            # Against the same run at a constant `scale = 1`. If the per-step rebuild were missing, the
            # scheduled run would have trained at the setup value throughout and these would agree.
            n_const = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
            train!(n_const)
            a = reduce(vcat, vec.(Array.(Functors.fleaves(parameters(n)))))
            b = reduce(vcat, vec.(Array.(Functors.fleaves(parameters(n_const)))))
            @test any(!=(0), a .- b)
            @test all(isfinite, a)
        end
    end

    # ── residency at the program boundary ────────────────────────────────────────────────

    @testset "residency at the program boundary" begin
        cache_reset!()
        n = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())

        ps_before = parameters(n)
        opt_before = n.opt_state
        acc_before = n.g_accum
        G = n_groups(n.layout)

        train!(n)

        @testset "`ps` is a TREE on both sides, and its type is unchanged" begin
            @test typeof(parameters(n)) === typeof(ps_before)
            @test parameters(n) isa NamedTuple                # a tree, not a flat buffer
            @test keys(parameters(n)) == keys(ps_before)
        end

        @testset "`opt_state` and the accumulator are NTuple{G} on both sides" begin
            @test n.opt_state isa NTuple{G, Any}
            @test typeof(n.opt_state) === typeof(opt_before)
            @test n.g_accum isa NTuple{G, Any}
            @test typeof(n.g_accum) === typeof(acc_before)
            @test all(l -> l isa Optimisers.Leaf, n.opt_state)
        end

        @testset "and none of it recompiled on the second step" begin
            # The residency rule drifting would change both signatures and hence the cache key, so
            # a stable type across 8 steps with 2 compiles is the assertion that catches it.
            @test cache_stats().misses == 2
            @test cache_stats().hits > 0
        end

        @testset "the flat side really is flat, and partitions the parameters" begin
            flat = flatten(parameters(n), n.layout)
            @test flat isa NTuple{G, Any}
            @test sum(length, flat) == sum(length, Functors.fleaves(parameters(n)))
        end
    end

    @testset "the Nitro holds no Enzyme shadow" begin
        # `dps` is allocated inside the gradient program on every invocation and never crosses a
        # boundary. A field here is the natural way to write the bug that doubles the gradient.
        @test :dps ∉ fieldnames(Nitro)
        @test !any(f -> occursin("shadow", String(f)), fieldnames(Nitro))
    end

    @testset "a non-finite loss stops the run, naming step and epoch" begin
        @experiment struct Diverge
            n::Int = 1
            max_epochs::Host{Int} = 1
        end
        @eval ReactantNitro.build_model(::Diverge, rng) =
            Lux.setup(rng, Lux.Dense(4 => 3)) |> ((ps, st),) -> (Lux.Dense(4 => 3), ps, st)
        @eval ReactantNitro.forward(::Diverge, model, ps, st; x) = Lux.apply(model, x, ps, st)
        @eval ReactantNitro.loss(::Diverge, ŷ; y) = sum(ŷ) / 0.0f0      # +Inf on the first step
        @eval ReactantNitro.build_data(::Diverge, dist) = (; train = $(mlp_data)())

        n = Nitro(Diverge(); checkpointer = nothing, run_dir = mktempdir())
        err = try
            train!(n)
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("no non-finite rollback by design", err.msg)
        @test occursin("epoch", err.msg)
        @test phase(n) isa Failed          # and the run is left in the Failed terminal state
    end

    @testset "only routed fields are transferred to device" begin
        n = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
        batch = (; x = randn(FT, 4, 8), y = randn(FT, 3, 8), case_id = ["a"])
        b = to_device_batch(batch, n.routing)
        @test keys(b) == (:x, :y)          # `case_id` reaches nobody, so it is never transferred
        @test b.x isa Reactant.AbstractConcreteArray
    end

    # ── the run accessors ───────────────────────────────────────────────────────────────

    @experiment struct KwExp
        width::Int = 8
        max_epochs::Host{Int} = 3
        "Driver-only, so `Host` (unmarked already is): a GraphConst field here recompiles per seed."
        seed::Host{Int} = 7
        "This one bakes, so a GraphConst field is correct for it."
        accum::GraphConst{Int} = 2
    end
    ReactantNitro.build_model(e::KwExp, rng) =
        (Lux.Dense(4 => 3), Lux.setup(rng, Lux.Dense(4 => 3))...)
    ReactantNitro.forward(::KwExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::KwExp, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.checkpointer(::KwExp) = nothing
    ReactantNitro.n_devs(::KwExp) = 1

    kw_data() = [(; x = randn(FT, 4, 8), y = randn(FT, 3, 8)) for _ in 1:4]

    @testset "the run accessors: an experiment carries its own defaults" begin
        @testset "read from a field of the same name" begin
            n = Nitro(KwExp(); data = (; train = kw_data()), run_dir = mktempdir())
            @test n.seed == 7                      # the Host field, not the framework's 42
            @test n.accum == 2                     # the GraphConst field, which correctly bakes
            @test n.max_epochs == 3
        end

        @testset "read from a method on the experiment type" begin
            n = Nitro(KwExp(); data = (; train = kw_data()), run_dir = mktempdir())
            @test n.checkpointer === nothing       # the accessor, not TopKCheckpointer()
            @test seed(KwExp()) == 7
            @test accum(KwExp()) == 2
            @test n_devs(KwExp()) == 1
            @test early_stop(KwExp()) === nothing  # the framework default, no accessor defined
            @test logger(KwExp()) isa JSONLogger   # the shipped JSON default
        end

        @testset "and the keyword replaces either, for one run" begin
            n = Nitro(
                KwExp(); data = (; train = kw_data()), run_dir = mktempdir(),
                seed = 99, accum = 1, max_epochs = 1
            )
            @test n.seed == 99
            @test n.accum == 1
            @test n.max_epochs == 1
            # The experiment is untouched: the override is per run, not a mutation.
            @test seed(KwExp()) == 7
        end

        @testset "overriding the `accum` ACCESSOR is inert on a built handle, and does not recompile" begin
            # `accum` USED to be in the compile cache's hook-world list, on the reasoning that
            # overriding the accessor
            # changes the compiled program without changing a hashed field. That reasoning does not hold:
            # the value is resolved ONCE here, into `nitro.accum`, and every read after that goes to the
            # field, so an accessor override cannot reach a trace without also moving `baked`.
            n = Nitro(KwExp(); data = (; train = kw_data()), run_dir = mktempdir(), max_epochs = 1)
            before = ReactantNitro.hook_worlds(KwExp())

            @eval ReactantNitro.accum(::KwExp) = 4
            @test accum(KwExp()) == 4                              # the accessor moved
            @test n.accum == 2                                     # the handle did not
            @test ReactantNitro.hook_worlds(KwExp()) == before     # and neither did the key

            # The report is what tells the user, since nothing else will: no recompile, no behaviour
            # change, and the loop still groups micro-batches by the stored 2.
            txt = ReactantNitro.fixed_config_report(n; entry = :train)
            # DRIFT ONLY now: the handle's values are `show(nitro)`'s job, so what this report
            # carries is the redefinition and what the handle is still using instead.
            @test !isempty(txt)
            @test occursin("`accum(e)` was redefined", txt)
            @test occursin("still uses 2", txt)
            # And the check does NOT fire for a keyword override, which is what makes it usable: this
            # handle's `run_dir` came from `mktempdir()` and differs from `run_dir(e)` on every run.
            @test !occursin("`run_dir(e)` was redefined", txt)

            @eval ReactantNitro.accum(::KwExp) = 2                 # put it back
        end

        @testset "a driver-only value as a GraphConst field is a setup error naming the fix" begin
            @experiment struct KwBad
                seed::GraphConst{Int} = 7           # driver-only: must stay Host (unmarked)
                n::Int = 1
            end
            err = try
                ReactantNitro.check_driver_fields(KwBad())
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("seed", err.msg)
            @test occursin("seeding rule", err.msg)
            @test occursin("Host", err.msg)
            @test occursin("recompiles per seed", err.msg)

            # `accum` and `gradient_clip_norm` are deliberately NOT checked: both genuinely bake, so a
            # GraphConst field is correct for them rather than a trap.
            @experiment struct D7Fine
                accum::GraphConst{Int} = 2
                gradient_clip_norm::GraphConst{Float32} = 1.0f0
                n::Int = 1
            end
            @test ReactantNitro.check_driver_fields(D7Fine()) === nothing
        end

        @testset "the four keyword-only knobs have no accessor" begin
            # Each names a fact about THIS INVOCATION rather than a property of the experiment.
            for name in (:data, :checkpoint, :resume, :run_ref)
                @test !isdefined(ReactantNitro, name)
            end
            # `data`'s accessor exists under another name.
            @test isdefined(ReactantNitro, :build_data)
        end
    end

    # ── single-host multi-device ─────────────────────────────────────────────────────────
    #
    # SCOPE, honestly. The device-residency fix's lesson applies here more than anywhere: a CPU
    # backend reports ONE
    # device, so nothing below exercises a real mesh, and none of it is evidence that sharding works.
    # What a CPU test CAN establish is the surrounding contract: the default, the guards, and that the
    # single-device path is untouched by the multi-device code. The mesh itself needs >= 2 real devices
    # and is verified there instead.

    @testset "the single-device path is unchanged, and n_devs defaults to what is visible" begin
        # `Reactant.devices()` reports 1 on a CPU backend, which is why the whole suite is unaffected by
        # the default becoming "every visible device" rather than 1.
        @test ReactantNitro.n_devs((;)) == length(Reactant.devices())

        # n_devs == 1 skips the mesh ENTIRELY. A one-device mesh is a no-op Reactant warns about, so
        # `nothing` is the correct answer rather than a degenerate mesh.
        @test ReactantNitro.setup_devices(1) === nothing

        cache_reset!()
        n = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
        @test n.mesh === nothing
        train!(n)
        @test current_epoch(n) == 2
        # The placement helpers are threaded through every transfer now; the single-device path must
        # still compile exactly two programs.
        @test cache_stats().misses == 2
    end

    @testset "`setup_devices` refuses more devices than are visible" begin
        err = try
            ReactantNitro.setup_devices(length(Reactant.devices()) + 1)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("VISIBLE", err.msg)
        @test occursin("CUDA_VISIBLE_DEVICES", err.msg)   # names the knob people actually use

        @test_throws ErrorException ReactantNitro.setup_devices(0)
    end

    @testset "a batch that does not divide across the mesh is a SETUP error" begin
        # The failure this prevents lands inside XLA as a shape complaint naming neither the batch size
        # nor the device count, so the framework checks where both numbers are in scope.
        @test ReactantNitro.check_shardable_batch(32, 1) === nothing    # one device: nothing to divide
        @test ReactantNitro.check_shardable_batch(32, 4) === nothing
        @test ReactantNitro.check_shardable_batch(32, 8) === nothing

        err = try
            ReactantNitro.check_shardable_batch(30, 4)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("30", err.msg) && occursin("4", err.msg)         # BOTH numbers, not just one
        @test occursin("GLOBAL", err.msg)                               # and the semantics that surprise people
    end

    @testset "the placement helpers fall through on one device" begin
        # `mesh === nothing` must make every call site read the same as it did before the mesh existed,
        # which is what lets the sharded and unsharded paths share one line of code.
        x = randn(FT, 4, 8)
        @test ReactantNitro.place_replicated(x, nothing) isa Reactant.AbstractConcreteArray
        @test ReactantNitro.place_batch(x, nothing) isa Reactant.AbstractConcreteArray
        @test Array(ReactantNitro.place_batch(x, nothing)) ≈ x
        @test Array(ReactantNitro.place_replicated(x, nothing)) ≈ x

        # `track_numbers` still reaches `to_rarray`, which is what makes a scheduled scalar a device
        # NUMBER rather than passing through unchanged.
        @test ReactantNitro.place_replicated(1.0f0, nothing; track_numbers = Number) isa Reactant.RNumber

        # The batch axis is the LAST one, matching the data-source contract and Lux's convention.
        @test ReactantNitro._batch_spec(4) === (nothing, nothing, nothing, :data)
        @test ReactantNitro._batch_spec(2) === (nothing, :data)
    end

    @testset "`Nitro(E, name)` records the preset it claimed" begin
        ReactantNitro.presets(::Type{MLP}) = (
            wide = (; width = 16),
            narrow = (; width = 8),
        )
        cache_reset!()
        n = Nitro(MLP, :narrow; checkpointer = nothing, run_dir = mktempdir())
        @test n.preset === :narrow
        @test experiment(n).width == 8
        # The preset is named by the HANDLE, not by the binding report: a preset's values are
        # ordinary struct fields by the time anything reads `e`, so it was never a per-value source,
        # and the handle is where the run's own facts live.
        @test occursin("preset", sprint(show, MIME"text/plain"(), n))
        @test occursin("narrow", sprint(show, MIME"text/plain"(), n))
        @test !occursin("preset", binding_report(n))

        # A Nitro keyword goes to Nitro, EVEN when it is also a field, which is what preserves the
        # three-way collision resolution on names like max_epochs.
        n2 = Nitro(MLP, :wide; max_epochs = 1, checkpointer = nothing, run_dir = mktempdir())
        @test n2.max_epochs == 1
        @test experiment(n2).width == 16

        # And a FIELD goes to the recipe, which is the fix: every override the first real model needed
        # was a field, so the two-name escape hatch had become the common path.
        n4 = Nitro(MLP, :wide; width = 4, max_epochs = 1, checkpointer = nothing, run_dir = mktempdir())
        @test experiment(n4).width == 4        # the keyword beat the preset's 16
        @test n4.preset === :wide              # and the name is still recorded, once
        @test n4.max_epochs == 1               # while the run keyword still went to the run

        # Neither a keyword nor a field is an error naming both sets, rather than a bare MethodError.
        err = try
            Nitro(MLP, :wide; nonsense = 1, checkpointer = nothing, run_dir = mktempdir())
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException && occursin("nonsense", err.msg)

        # A run that named no preset carries `nothing`, and the report stays silent about it.
        n3 = Nitro(MLP(); checkpointer = nothing, run_dir = mktempdir())
        @test n3.preset === nothing
        @test !occursin("preset", binding_report(n3))
    end

end
