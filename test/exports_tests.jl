# Export skeleton tests.
#
# This suite is extended past the acceptance criterion to an executable skeleton: every file in
# every file present, every export declared with its signature, bodies throwing. INTERFACE DRIFT
# BETWEEN LATER ITEMS IS THE FAILURE MODE A PROSE SPEC DOES NOT PREVENT AND A COMPILING SKELETON
# DOES, so this file asserts the export surface directly. A name that moves has to
# move here too.

@testitem "exports" begin
    using Test
    using ReactantNitro
    # Imported, not `using`ed, exactly as the package itself does it: the collision test below
    # needs their export lists, and `using` them here would reintroduce the ambiguity it exists to
    # forbid.
    import Enzyme, Functors, LinearAlgebra, Lux, Optimisers, Random, Reactant, Statistics

    # The export list, transcribed group by group. Keep the grouping: it is how a reader checks
    # this against the framework's own documentation.
    const CONFIGURATION = [
        :Device, :Host, :GraphConst, :device_fields, :host_fields,
        :config_metadata, :compile_view,
        # The export view, exported for the same reason `compile_view` is: it is an overridable
        # accessor, and an experiment whose `Device` fields need different treatment at export
        # overrides it by name.
        :export_view,
        # Named configurations.
        :presets, :from_preset,
        # Accelerator setup: one Julia process initializes XLA once, so the visible-device
        # configuration is process-wide.
        :setup_devices!,
    ]
    const HOOKS = [
        :build_data, :build_model, :forward, :loss,
        :metrics, :train_metrics, :finalize_metrics, :metrics_residency, :derive,
        :param_group, :optimizer, :learning_rate, :lambda, :decay_anchor,
        # The per-LEAF decay exclusion. `default_no_decay` is exported
        # alongside the hook so a user can extend the built-ins rather than replace them.
        :no_decay, :default_no_decay,
        :gradient_clip_norm, :schedules, :nonschedulable, :dispatch_variant,
        :max_epochs,
        # Manual training mode. Defining `train_step` for an experiment type is what selects
        # manual mode; `manual_training(e) = false` declines it while keeping the method;
        # `setup_optimizers` supplies the user-owned optimizer states the closure steps itself;
        # `backward` and `step_optimizer` are the helpers the closure calls inside the step.
        :train_step, :setup_optimizers, :manual_training, :backward, :step_optimizer,
        # The run accessors: each one's `Nitro` keyword defaults to the accessor call, so an
        # experiment declares what it is and a caller passes only what this run changes.
        :seed, :accum, :n_devs, :checkpointer, :early_stop, :logger,
    ]
    const ENTRY_POINTS = [
        :Nitro, :train!, :validate, :evaluate, :predict,
        # Verbs on a LIVE handle. A `Device` is excluded from the compile cache's key
        # by construction, so these are the supported way to sweep a value without
        # rebuilding, and the reason a GraphConst field is refused rather than accepted.
        :set_device!, :device_value,
    ]
    # The prefetch surface. `PrefetchIterator` was exported as an optional helper the
    # user wrapped their loader in; prefetch is now a framework DEFAULT (setup wraps the `train` split at
    # `depth = 1`, `workers = Threads.nthreads(:default)`), so what is public is the override, the one
    # opt-out, and the two-method trait a source EXTENDS to get real concurrency.
    #
    # `batch_at` and `begin_epoch!` are exported for the same reason the hooks are: a model defines methods
    # on them. `check_batch_at`, `NoPrefetch`'s internals, and `prefetch_config` stay internal.
    const DATA = [:PrefetchIterator, :NoPrefetch, :batch_at, :begin_epoch!]
    const OPTIMIZER = [:Decay]
    # `checkpoint_filename` is exported for the same reason the hooks are: a model defines a method
    # on it to name checkpoints differently.
    #
    # `read_manifest` and `checkpoint_info` are the READERS, and they are public because the
    # alternative kept being reinvented: neither is a new capability, both existed as internals, and
    # a caller who could not reach them opened the JLD2 by hand and printed a parameter tree into
    # its own transcript. A run directory's contents come from the manifest, opening no record at
    # all; one record's metadata comes from `checkpoint_info`, which returns none of its weights.
    const CHECKPOINTING = [
        :TopKCheckpointer, :save_checkpoint!, :load_checkpoint, :CheckpointRecord,
        :checkpoint_filename, :read_manifest, :checkpoint_info,
    ]
    const LOGGING = [
        :log_metrics!, :log_params!, :log_tags!, :log_other!, :log_confusion!, :finish!,
        :run_id, :run_url, :logger_state, :reattach!, :backend, :logger_info,
        # The one shipped backend, the JSON default.
        :JSONLogger,
    ]
    const PHASES = [
        :Phase, :Repl, :Starting, :Compiling, :GradCompiling, :OptCompiling, :EvalCompiling, :ExportCompiling,
        :Stepping, :TrainStepping, :EvalStepping, :Checkpointing, :Terminal, :Done, :Failed,
        # Named for what gets built on the registry (a heartbeat, a watchdog, a progress display)
        # rather than for how the function is invoked. It was `register_phase_callback!`.
        :register_phase_monitor!, :unregister_phase_monitor!, :progress_counter,
        :request_stop!, :run_dir,
        :current_step, :current_epoch, :phase, :experiment, :parameters, :states,
        :EarlyStopping, :should_stop,
        # The run's per-epoch series, as data that displays as a table: what a REPL asks a
        # finished handle. Where each value bound is part of `show(nitro)`; there is no
        # accessor returning that table as a String, since a framed String is unreadable.
        :history,
    ]
    const DISTRIBUTION = [:rank, :world_size]
    # Export. The hooks are here rather than in HOOKS because they are optional in a different
    # sense: an experiment that is never exported implements none of them, and the four required hooks stay four.
    const EXPORT = [
        :ExportSpec, :ExportBackend, :ReactantServerBundle,
        :export_inputs, :export_outputs, :export_preprocess, :export_postprocess,
        :export_client_outputs, :export_client_inputs,
        # The model half of provenance, as a hook rather than an argument a caller must remember
        :export_provenance_extra,
        :export_model, :export_provenance, :write_export,
        # The second backend verb: repository state, collected at a root the CALLER names
        :site_provenance,
    ]
    # Visualization. Separate from HOOKS for the same reason EXPORT is: `visualize` and
    # `save_figure` are optional
    # in a different sense, since an experiment that is never rendered implements neither, and the four
    # required hooks stay four. `render` is a driver rather than a hook.
    const VISUALIZATION = [:visualize, :save_figure, :render]

    const ALL_EXPORTS = vcat(
        CONFIGURATION, HOOKS, ENTRY_POINTS, DATA, OPTIMIZER, CHECKPOINTING, LOGGING,
        PHASES, DISTRIBUTION, EXPORT, VISUALIZATION
    )

    @testset "the export surface" begin
        exported = Set(names(ReactantNitro))

        @testset "every name is exported" begin
            for n in ALL_EXPORTS
                @test n in exported
            end
            @test Symbol("@experiment") in exported
        end

        @testset "nothing is exported that this list does not name" begin
            # "Everything not listed there is internal and may change without a breaking release", so an
            # accidental export is a silent public-API commitment.
            extra = setdiff(exported, Set(ALL_EXPORTS))
            delete!(extra, :ReactantNitro)
            delete!(extra, Symbol("@experiment"))
            @test isempty(extra)
        end

        @testset "no export name collides with ReactantServerExport" begin
            # The same class of bug as the `Training`/`Lux.Training` collision below, and with a sharper
            # edge: the export extension makes `using ReactantServerExport` the thing that turns
            # the export
            # path on, so that package is in scope in EVERY session where this surface is usable. A
            # shared name would be ambiguous on the user, always, with no way to avoid it.
            #
            # Written as a literal rather than read from the package, because it is a weak dependency and
            # is not in the test environment. It is that package's whole export list at 0.1.10; if it
            # grows one, this list is what has to be updated alongside.
            theirs = [
                :IOSpec, :write_bundle, :export_bundle, :export_torchscript_bundle,
                :collect_provenance,
            ]
            @test isempty(intersect(ALL_EXPORTS, theirs))
        end

        @testset "no name is listed twice" begin
            # This list names `Nitro` under both Entry points and Phases and control; the
            # module's own list
            # must not, or `export` warns.
            @test length(ALL_EXPORTS) == length(unique(ALL_EXPORTS))
        end
    end

    @testset "the phase tree" begin
        # Three groupings, each with more than one child and a real query behind it. NO SINGLE-CHILD
        # PARENTS, and in particular NO `CompilingFused` leaf: the framework compiles no fused
        # program, so a fused phase could never fire, and a phase that never arrives is a monitor
        # that hangs. The four compile leaves each have a real entry point: the train-side three
        # fire on a cache miss and `ExportCompiling` fires once per `export_model` call.
        @test Repl <: Phase
        @test Starting <: Phase
        @test GradCompiling <: Compiling <: Phase
        @test OptCompiling <: Compiling
        @test EvalCompiling <: Compiling
        @test ExportCompiling <: Compiling
        @test TrainStepping <: Stepping <: Phase
        @test EvalStepping <: Stepping
        @test Checkpointing <: Phase
        @test Done <: Terminal <: Phase
        @test Failed <: Terminal

        leaves = [
            Repl, Starting, GradCompiling, OptCompiling, EvalCompiling, ExportCompiling,
            TrainStepping, EvalStepping,
            Checkpointing, Done, Failed,
        ]
        @test all(isconcretetype, leaves)
        @test !isdefined(ReactantNitro, :CompilingFused)

        @testset "the phase-naming rule: a grouped leaf ends with its parent's name" begin
            # `<Qualifier><Parent>`, so `p isa Compiling` is readable straight off the leaf's name and a
            # leaf added later names itself. `Starting` and `Checkpointing` have no group, and the
            # `Terminal` branch is the documented exception: it names results, not activities.
            for leaf in leaves
                parent = supertype(leaf)
                parent === Phase && continue
                parent === Terminal && continue
                @test endswith(string(nameof(leaf)), string(nameof(parent)))
            end
        end

        @testset "no phase name collides with a package an experiment file `using`s" begin
            # This is the regression test for the collision that forced the rename: the leaf was
            # `Training`, `Lux` exports a `Training` module, and the ambiguity landed on the USER in any
            # file doing `using Lux, ReactantNitro`. Measured empty across the whole stack, and it stays
            # empty because this asserts it.
            phase_names = [nameof(t) for t in vcat(leaves, [Phase, Compiling, Stepping, Terminal])]
            for pkg in (
                    Lux, Optimisers, Reactant, Functors, Enzyme, Random, Statistics, LinearAlgebra,
                    Base, Core,
                )
                clash = intersect(phase_names, names(pkg))
                @test isempty(clash)
            end
        end

        # Users can extend it, which a closed enum would forbid.
        @test_nowarn @eval struct Preprocessing <: Phase end
    end

    @testset "the skeleton is executable: NOTHING is left unbuilt" begin
        # `_unimplemented` used to mark a declared-but-not-yet-built entry point, so a not-yet-built
        # hole and a genuine MethodError-shaped one were distinguishable at a glance. Entries moved
        # out of this list as they were built, and the list shrinking was the point.
        #
        # IT IS NOW EMPTY. Every entry point that once used `_unimplemented` is built, so the loop
        # below runs zero times and the assertion that matters is the emptiness itself.
        UNBUILT = []
        @test isempty(UNBUILT)
        for (f, args) in UNBUILT
            err = try
                f(args...)
                nothing
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("not implemented yet", err.msg)
        end

        # The mechanism still has to WORK, or the emptiness above would be indistinguishable from a
        # broken `_unimplemented` that no longer says anything useful.
        err = try
            ReactantNitro._unimplemented("a thing", 99)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("not implemented yet", err.msg)

        # And the entry that used to be here now does its real job instead of raising.
        @test ReactantNitro.setup_devices(1) === nothing

        # The checkpoint layer built the checkpoint path. What is still asserted here is the
        # documented opt-out and the refusal to read something that is not there, neither of which
        # is a hole.
        @test save_checkpoint!(nothing, 1, (;), (;)) === nothing
        @test load_checkpoint(nothing, "x") === nothing
        @test_throws ErrorException load_checkpoint(TopKCheckpointer(), "no-such-file.jld2")
        # The phase system built the policy; the `::Nothing` no-op was shipped with the export
        # skeleton and still holds.
        @test should_stop(nothing, 1, (; val_loss = 1.0)) === false
        @test should_stop(EarlyStopping(), 1, (; val_loss = 1.0)) === false
    end

    @testset "the shipped contract methods that are pure declaration" begin
        # The distribution stubs, whose whole point is that nothing dispatches on the handle.
        @test rank(nothing) == 0
        @test world_size(nothing) == 1

        # `nothing` is the public "no logging" value, with a method for ALL TEN verbs.
        @test log_metrics!(nothing, (; a = 1); step = 1, epoch = 1, context = "train") === nothing
        @test log_params!(nothing, (; a = 1)) === nothing
        @test log_tags!(nothing, ["a"]) === nothing
        @test log_other!(nothing, "k", "v") === nothing
        @test log_confusion!(nothing, zeros(Int, 2, 2), ["a", "b"]; epoch = 1) === nothing
        @test finish!(nothing, :completed) === nothing
        @test run_id(nothing) === nothing        # the checkpoint layer needs this one
        @test run_url(nothing) === nothing
        @test logger_state(nothing) === nothing
        @test reattach!(nothing, nothing) === nothing
        # `logger_info` is informational, defaulting to an empty table; nothing to say is the
        # whole contract, and the generic default covers any object a ten-line logger could be.
        @test logger_info(nothing) === (;)
        @test logger_info("a logger that defines nothing") === (;)

        # `logger_state` defaults to `nothing`, meaning "I have no resumable state", so a
        # ten-line file logger defines neither it nor `reattach!` and keeps working.
        @test logger_state("a logger that defines nothing") === nothing
        # `backend` is the identity, because the logger a user passes IS their backend object.
        @test backend("an experiment handle") == "an experiment handle"

        # No early stopping is the default, with no branch in the driver.
        @test should_stop(nothing, 1, (; val_loss = 1.0)) === false
        es = EarlyStopping()
        @test es.metric === :val_loss && es.mode === :min && es.patience == 5

        # `checkpointer = nothing` is the documented opt-out, with `::Nothing` no-op methods.
        @test save_checkpoint!(nothing, 1, (;), (;)) === nothing
        @test load_checkpoint(nothing, "anywhere") === nothing
        ck = TopKCheckpointer()
        @test ck.k == 3 && ck.metric === :val_loss && ck.mode === :min
    end

end
