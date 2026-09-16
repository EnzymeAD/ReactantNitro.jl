# Cache tests, plus prefetch-cleanup cache coverage.
#
# The acceptance criterion this file exercises: "green; a two-group OneCycle holds the LR
# ratio; a redefined `forward` misses the cache; the binding report names every source".

@testitem "cache" begin
    using Test
    using ReactantNitro
    using ReactantNitro: CACHE, ResolvedSchedules, binding_report_text, build_layout, cache_key, guard_horizon,
        cache_reset!, cache_stats, compile_cached, effective_lr, hook_worlds,
        level1_chain, method_world, graphconst_field_hash, primary_world_available,
        resolve_hp, resolve_schedules, schedulable_fields, step_aux, step_experiment, to_device,
        assert_graphconst_hashable, compile_view, graphconst_fields,
        to_device_config, to_device_leaf, apply_group,
        closure_drift, fwd_program, grad_program, world_closure, world_closure_available,
        world_closure_staleness, _closure_target
    using Lux, Optimisers, Random, Reactant

    const F4 = Float32

    @experiment struct CacheExp
        "A traced input: excluded from the key by construction."
        scale::Device{Float32} = 1.0f0
        "A GraphConst field: BAKES, so it is in the key."
        n_layers::GraphConst{Int} = 4
        "Driver-only: not in the traced view at all."
        max_epochs::Host{Int} = 40
    end

    @experiment struct SchedExp
        "The EXPERIMENT's lambda: an auxiliary loss weight, not the optimizer's."
        lambda::Device{Float32} = 0.25f0
        aux::Device{Float32} = 1.0f0
        n::Int = 1
    end

    @experiment struct RatioExp
        n::Int = 1
    end
    ReactantNitro.param_group(::RatioExp, ks) = ks[1] === :backbone ? :backbone : :default
    ReactantNitro.learning_rate(::RatioExp) = 1.0f-3
    ReactantNitro.learning_rate(::RatioExp, ::Val{:backbone}) = 1.0f-4

    # A stand-in for ParameterSchedulers' OneCycle, which the framework deliberately does not depend on.
    onecycle(total, maxval; startval = maxval / 25, endval = maxval / 1.0e4) =
        t -> t <= total ÷ 2 ?
        startval + (maxval - startval) * (t - 1) / max(1, total ÷ 2 - 1) :
        maxval + (endval - maxval) * (t - total ÷ 2) / max(1, total - total ÷ 2)

    # ── the key ──────────────────────────────────────────────────────────────────────────

    @testset "the key: what changes it and what must not" begin
        e = CacheExp()
        ev = compile_view(e)
        a = (Reactant.to_rarray(randn(F4, 4, 8)),)

        base = cache_key(identity, ev, a, ())
        @test cache_key(identity, ev, a, ()) == base                       # same input hits

        @testset "a changed GraphConst field misses: it bakes as a trace-time constant" begin
            @test cache_key(identity, compile_view(CacheExp(; n_layers = 5)), a, ()) != base
        end

        @testset "a changed Device HITS: it is a traced input and cannot affect the graph" begin
            @test cache_key(identity, compile_view(CacheExp(; scale = 2.0f0)), a, ()) == base
        end

        @testset "a changed Host field HITS: it is not in the traced view at all" begin
            @test cache_key(identity, compile_view(CacheExp(; max_epochs = 41)), a, ()) == base
        end

        @testset "a rebuilt experiment with FRESH device scalars hits, which every step does" begin
            # This is the reason for keying on GraphConst fields rather than on compile_view(e):
            # keying on the view would hash device scalars that change every step, so EVERY optimizer
            # step would miss and recompile.
            rebuilt = compile_view(CacheExp(; scale = to_device(1.0f0)))
            @test graphconst_field_hash(rebuilt) == graphconst_field_hash(ev)
            @test cache_key(identity, rebuilt, a, ()) == base
        end

        @testset "a Device field may hold a BUFFER SET, not only one value" begin
            # A frozen backbone's weights are PyTorch's "buffers": arrays the graph reads and
            # nothing differentiates. They reach the trace through a Device field, so the
            # conversion has to recurse through the container that holds them. Before this
            # method the conversion threw a MethodError naming the whole tuple type.
            bufs = (randn(F4, 2, 3), randn(F4, 4))
            dev = to_device(bufs)
            @test dev isa Tuple && length(dev) == 2
            @test all(x -> x isa Reactant.AbstractConcreteArray, dev)
            @test Array(dev[1]) == bufs[1] && Array(dev[2]) == bufs[2]

            # Idempotent, like the scalar and array methods: a buffer set already on device
            # passes through leaf by leaf rather than being handed back to `to_rarray`.
            again = to_device(dev)
            @test all(again[i] === dev[i] for i in 1:2)

            # NamedTuple too, since that is how a model names its buffers.
            named = to_device((; w = randn(F4, 2), b = randn(F4, 2)))
            @test named isa NamedTuple && keys(named) == (:w, :b)
            @test all(x -> x isa Reactant.AbstractConcreteArray, values(named))

            # A device field holding a buffer set is still a traced input, so it cannot move
            # the graph: the cache key is unchanged by its VALUE, exactly as a device scalar.
            @test cache_key(identity, compile_view(CacheExp(; scale = 3.0f0)), a, ()) == base
        end

        @testset "argument SHAPES are in the key, not only types" begin
            # Reactant's @generated guard covers types but not shapes, since shape is a runtime field.
            wider = (Reactant.to_rarray(randn(F4, 4, 16)),)
            @test typeof(wider[1]) === typeof(a[1])          # same type...
            @test cache_key(identity, ev, wider, ()) != base  # ...different key
        end

        @testset "function identity is in the key" begin
            @test cache_key(sum, ev, a, ()) != base
        end

        @testset "baked train! keywords are in the key" begin
            @test cache_key(identity, ev, a, (; accum = 1)) != cache_key(identity, ev, a, (; accum = 2))
        end
    end

    # ── method redefinition ──────────────────────────────────────────────────────────────

    @experiment struct RedefExp
        n::Int = 1
    end
    ReactantNitro.forward(::RedefExp, model, ps, st; img) = (img, st)
    ReactantNitro.loss(::RedefExp, outputs; lab) = sum(outputs)

    # Separate from `RedefExp` so the two `@eval`ed accessor redefinitions cannot perturb each other's
    # baseline: this one exists to prove a redefinition does NOT move the worlds.
    @experiment struct WorldFreeExp
        n::Int = 1
    end
    ReactantNitro.forward(::WorldFreeExp, model, ps, st; img) = (img, st)
    ReactantNitro.loss(::WorldFreeExp, outputs; lab) = sum(outputs)

    @testset "a redefined `forward` misses; an unrelated definition does not" begin
        @test primary_world_available()          # a load-bearing internal, still there

        before = hook_worlds(RedefExp())
        global_counter_before = Base.get_world_counter()
        @eval an_unrelated_definition_for_the_cache_test() = 1
        # The global counter DOES move on any definition anywhere, which is why the cache keys on
        # the per-hook method world instead: otherwise the cache would invalidate every time a user defines anything
        # at the REPL, which for a REPL-first design is every few seconds.
        @test Base.get_world_counter() != global_counter_before
        @test hook_worlds(RedefExp()) == before

        @eval ReactantNitro.forward(::RedefExp, model, ps, st; img) = (img .* 2, st)
        after = hook_worlds(RedefExp())
        @test after != before

        # And it is `forward`'s slot that moved, not a blanket invalidation.
        @test count(!=(0), after .- before) == 1

        # A hook with no method contributes 0 rather than raising.
        @test method_world(metrics, Tuple{RedefExp, Any}) == 0
    end

    # ── values already in the key are not in the worlds ─────────────────────────────────

    @testset "`accum` and `gradient_clip_norm` are NOT in the worlds: they cannot catch anything" begin
        # Both are resolved once at `Nitro` construction and read from the stored field thereafter, so the
        # value that reaches a trace is already in the key: `accum` through `baked = (; accum = ...)` and
        # the clip through `Val(nitro.gradient_clip_norm)`, whose value IS its type parameter. A world
        # entry could therefore only fire spuriously, and it DID: revising either accessor on a handle
        # built before the edit used to recompile BOTH programs (495.6 s + 63.1 s) to rebuild an
        # identical `inv_n`, and the recompile was the user's only evidence that Revise had "worked".
        before = hook_worlds(WorldFreeExp())

        @eval ReactantNitro.accum(::WorldFreeExp) = 4
        @test accum(WorldFreeExp()) == 4                    # the accessor really did move
        @test hook_worlds(WorldFreeExp()) == before         # and the key did not

        @eval ReactantNitro.gradient_clip_norm(::WorldFreeExp) = 1.0f0
        @test gradient_clip_norm(WorldFreeExp()) == 1.0f0
        @test hook_worlds(WorldFreeExp()) == before

        # What DOES cover them, so removing the entries lost no coverage. `accum` sits in `baked`, and
        # the clip in the argument's `Val` type, which `cache_key` reads through `map(typeof, args)`.
        ev = compile_view(WorldFreeExp())
        a = (Reactant.to_rarray(randn(F4, 4)),)
        @test cache_key(identity, ev, a, (; accum = 2)) != cache_key(identity, ev, a, (; accum = 4))
        @test cache_key(identity, ev, (a..., Val(0.0f0)), ()) !=
            cache_key(identity, ev, (a..., Val(1.0f0)), ())
    end

    # ── the Nitro is a fixed point ──────────────────────────────────────────────────────

    @experiment struct FrozenExp
        n::Int = 1
    end
    frozen_chain() = Lux.Chain(Lux.Dense(3 => 2))
    ReactantNitro.build_model(::FrozenExp, rng) = (m = frozen_chain(); (m, Lux.setup(rng, m)...))
    ReactantNitro.forward(::FrozenExp, model, ps, st; img) = Lux.apply(model, img, ps, st)
    ReactantNitro.loss(::FrozenExp, out; lab) = sum(abs2, out .- lab)
    frozen_data() = [(; img = randn(F4, 3, 2), lab = randn(F4, 2, 2)) for _ in 1:2]

    # A custom rule, which is the ONLY way to reach the `apply!` worlds: every shipped rule's `apply!`
    # lives in Optimisers and nobody redefines those.
    mutable struct CountingRule <: Optimisers.AbstractRule
        eta::Float32
    end
    Optimisers.init(::CountingRule, x) = nothing
    Optimisers.apply!(r::CountingRule, st, x, dx) = (st, dx .* r.eta)
    ReactantNitro.nonschedulable(::Type{<:CountingRule}) = ()

    @testset "every live-dispatch component of the key is resolved at construction" begin
        n = Nitro(
            FrozenExp(); run_dir = mktempdir(),
            data = (; train = frozen_data(), val = frozen_data())
        )

        @testset "the handle carries them, and they are the three the programs need" begin
            # `graphconst_hash` joined this set: the experiment-derived key component
            # is resolved here for the same reason the worlds are, so every component of the key now
            # derives from state frozen at construction. Manual-mode entries joined later: the
            # mode itself, the closure program's worlds, and `setup_optimizers`' staleness-only world.
            # For an automatic experiment they resolve to the defaults, `false` and `()`, but they are
            # still frozen dispatch, resolved once.
            @test keys(n.frozen) == (
                :worlds_train, :worlds_opt, :worlds_eval,
                :tm_residency, :metrics_residency, :graphconst_hash,
                :manual, :worlds_manual, :worlds_setup,
            )
            @test n.frozen.manual === false
            @test n.frozen.worlds_manual == ()
            @test n.frozen.worlds_setup == ()
            # The metrics-residency defaults, resolved once. `train_metrics` has no method here, so
            # the rule that
            # keeps it on the `:device` program applies and is baked into the answer.
            @test n.frozen.tm_residency === :device
            @test n.frozen.metrics_residency === :host
        end

        @testset "the two `st` modes are resolved separately" begin
            # `forward`'s world is resolved against `typeof(st)`, and `train!` compiles with a
            # train-mode `st` while the eval paths use an eval-mode one. Before this revision `train!`
            # resolved against the UN-MODED `nitro.st`, so a model whose two state types differ and
            # which dispatches `forward` on them keyed on the wrong signature.
            @test n.frozen.worlds_train ==
                ReactantNitro.hook_worlds(
                compile_view(n.e); n.model, n.ps,
                st = Lux.trainmode(n.st)
            )
            @test n.frozen.worlds_eval ==
                ReactantNitro.hook_worlds(
                compile_view(n.e); n.model, n.ps,
                st = Lux.testmode(n.st)
            )
        end

        @testset "a redefined hook does NOT move an existing handle, and says so" begin
            # This reverses an earlier design, and is the whole point of the change. The worlds went
            # into the key because editing `forward` and re-running would SILENTLY run the old
            # program. The program is still the old one, deliberately; what is gone is the silence.
            frozen_before = n.frozen
            @test isempty(ReactantNitro.stale_hooks(n))

            @eval ReactantNitro.loss(::FrozenExp, out; lab) = sum(abs2, out .- lab) * 2
            @test n.frozen === frozen_before                  # the handle did not move
            stale = ReactantNitro.stale_hooks(n)
            @test :worlds_train in stale && :worlds_eval in stale
            @test occursin("hooks were redefined", ReactantNitro.fixed_config_report(n))
            @test occursin("rebuild", ReactantNitro.fixed_config_report(n))

            # And a NEW handle picks the edit up, which is what keeps the cache honest: it exists so
            # a fresh COMPATIBLE handle skips the compile, not so a stale one is reused.
            n2 = Nitro(
                FrozenExp(); run_dir = mktempdir(),
                data = (; train = frozen_data(), val = frozen_data())
            )
            @test n2.frozen.worlds_train != frozen_before.worlds_train
            @test isempty(ReactantNitro.stale_hooks(n2))
        end
    end

    @testset "a revised custom rule's `apply!` reaches the key, and only the optimizer's" begin
        # THE GAP THIS CLOSES: `chains` was a parameter of `hook_worlds` that no call site ever passed,
        # so `_rules_of`'s loop never ran and a revised custom `apply!` silently reused the old
        # optimizer program, which the cache and that docstring both claimed was covered.
        # Construction is the
        # only place a single correct chain set exists, since `rebuild_rules` reconstructs them per step.
        e = FrozenExp()
        ev = compile_view(e)
        r = CountingRule(1.0f-3)

        with_rule = hook_worlds(ev; chains = (r,))
        without = hook_worlds(ev)
        @test length(with_rule) == length(without) + 1        # the loop actually ran
        @test with_rule[1:length(without)] == without

        @eval Optimisers.apply!(r::CountingRule, st, x, dx) = (st, dx .* r.eta .* 2)
        after = hook_worlds(ev; chains = (r,))
        @test after != with_rule                              # the revision moved the rule's slot
        @test after[1:length(without)] == without             # and moved nothing else

        @testset "the rules ride in `worlds_opt` ONLY" begin
            # `opt_program` traces no user hook and `grad_program` contains no `apply!`, so sharing one
            # tuple would make a rule edit re-buy the 495.6 s gradient compile: exactly the spurious
            # expensive recompile that was removed above for `accum` and the clip.
            n = Nitro(
                FrozenExp(); run_dir = mktempdir(),
                data = (; train = frozen_data(), val = frozen_data())
            )
            # The default `RAdam` chain contributes its own `apply!`, so `worlds_opt` is strictly longer
            # than `worlds_train` and agrees with it on every entry they share.
            @test length(n.frozen.worlds_opt) > length(n.frozen.worlds_train)
            @test n.frozen.worlds_opt[1:length(n.frozen.worlds_train)] == n.frozen.worlds_train
        end
    end

    # ── clip cache scope ─────────────────────────────────────────────────────────────────

    @testset "clip cache scope: the clip invalidates the optimizer program only" begin
        e = CacheExp()
        ev = compile_view(e)
        gargs = (Reactant.to_rarray(randn(F4, 8)),)
        oargs = (Reactant.to_rarray(randn(F4, 8)),)

        # `accum` bakes into the GRADIENT program, `gradient_clip_norm` into the OPTIMIZER
        # program, so the key is naturally per program rather than per run.
        gkey(accum) = cache_key(identity, ev, gargs, (; accum))
        okey(clip) = cache_key(sum, ev, oargs, (; gradient_clip_norm = clip))

        @testset "a different clip misses the optimizer program" begin
            @test okey(0.0f0) != okey(1.0f0)
        end

        @testset "and HITS the gradient program, which never sees the clip" begin
            @test gkey(2) == gkey(2)
            # The clip is not in the gradient program's baked set at all, which is what makes a clip
            # sweep re-pay the cheap compile (63.1 s) and reuse the expensive one (495.6 s).
            @test :gradient_clip_norm ∉ keys((; accum = 2))
        end

        @testset "keyword, method, and field resolve identically" begin
            # The clip is a defaulted accessor AND a Nitro keyword. All three routes must produce
            # the same baked value, or the same run configured three ways would compile three programs.
            @experiment struct ClipField
                gradient_clip_norm::Float32 = 1.0f0
                n::Int = 1
            end
            @experiment struct ClipMethod
                n::Int = 1
            end
            @eval ReactantNitro.gradient_clip_norm(::ClipMethod) = 1.0f0

            by_field = gradient_clip_norm(ClipField())
            by_method = gradient_clip_norm(ClipMethod())
            by_keyword = 1.0f0
            @test by_field === by_method === by_keyword
            @test okey(by_field) == okey(by_method) == okey(by_keyword)
        end
    end

    # ── scheduled eta reuse ──────────────────────────────────────────────────────────────

    @testset "a scheduled `eta` reuses the thunk and tracks the schedule" begin
        cache_reset!()
        x0 = randn(F4, 32)
        g = randn(F4, 32) .* 0.1f0

        step1(leaf, x, grad) = apply_group(leaf, x, grad)
        xd, gd = Reactant.to_rarray(copy(x0)), Reactant.to_rarray(g)

        build(eta) = to_device_leaf(Optimisers.setup(Optimisers.Adam(; eta = to_device(F4(eta))), x0))
        l1 = build(1.0f-3)
        thunk = compile_cached(step1, compile_view(CacheExp()), l1, xd, gd)
        @test cache_stats().misses == 1

        # Rebuilding the rule with a fresh device scalar must RE-ENTER THE SAME THUNK: constant and
        # scheduled are the same device slot, and switching between them does not recompile.
        l2 = build(5.0f-3)
        @test typeof(l2) === typeof(l1)
        thunk2 = compile_cached(step1, compile_view(CacheExp()), l2, xd, gd)
        @test thunk2 === thunk
        @test cache_stats().hits == 1
        @test cache_stats().misses == 1

        # And the value is LIVE: a larger eta moves the parameters further.
        _, xa = thunk(l1, xd, gd)
        _, xb = thunk(l2, xd, gd)
        @test Array(xa) != Array(xb)
        @test maximum(abs.(Array(xb) .- x0)) > maximum(abs.(Array(xa) .- x0))
    end

    # ── schedule resolution ──────────────────────────────────────────────────────────────

    @testset "every entry is a factory of the horizon, called once" begin
        e = SchedExp()
        calls = Ref(0)
        given = (; aux = total -> (calls[] += 1; t -> Float32(t / total)))
        r = resolve_schedules(e, given, 100)
        @test calls[] == 1                      # the factory is called ONCE, at setup
        @test r.device.aux(50) ≈ 0.5f0
        @test r.device.aux(100) ≈ 1.0f0
        @test calls[] == 1                      # and not again per step

        # A bare Number normalizes to `_ -> (_ -> value)`, which is what makes constants setup-fixed by
        # construction: only scheduled entries transfer per step.
        rc = resolve_schedules(e, (; aux = 1.0f-8), 100)
        @test rc.device.aux(1) === 1.0f-8
        @test rc.device.aux(999) === 1.0f-8
        @test :aux in rc.constant
        @test !rc.horizon_dependent             # nothing is actually varying
    end

    @testset "a schedule that throws past the horizon holds its final value and warns once" begin
        # What ParameterSchedulers' `Shortened` does at `total + 1`: a BoundsError, at the very end
        # of a run whose per-epoch batch count drifted above what setup read.
        e = SchedExp()
        given = (; aux = total -> (t -> t > total ? throw(BoundsError([1.0], t)) : Float32(t / total)))
        r = resolve_schedules(e, given, 100)
        @test r.device.aux(100) ≈ 1.0f0
        # Past the horizon: the step-100 value, one warning naming the key, the step and the horizon.
        v = @test_logs (:warn, r"schedule `aux` threw at step 101, past its horizon `total = 100`") r.device.aux(101)
        @test v ≈ 1.0f0
        # And silence after that: the run may have a few hundred tail steps.
        @test (@test_logs r.device.aux(102)) ≈ 1.0f0
        @test (@test_logs r.device.aux(150)) ≈ 1.0f0
        # A schedule that ANSWERS past the horizon is left alone: no hold, no warning. A step decay
        # or a cycle does not depend on `total` and must not be frozen by a guard meant for one that does.
        open = resolve_schedules(e, (; aux = _ -> (t -> Float32(t))), 100)
        @test (@test_logs open.device.aux(150)) === 150.0f0
        # An exception INSIDE the horizon is a real error and propagates.
        bad = resolve_schedules(e, (; aux = _ -> (t -> t == 50 ? error("mid-run bug") : 1.0f0)), 100)
        @test_throws ErrorException bad.device.aux(50)
        # A constant is never guarded (and never throws), and an unknown horizon guards nothing.
        rc = resolve_schedules(e, (; aux = 2.0f0), 100)
        @test rc.device.aux(999) === 2.0f0
        @test guard_horizon(identity, 0, :x) === identity
    end

    @testset "keys resolve against two namespaces, and collisions are qualified" begin
        e = SchedExp()
        chain = Optimisers.OptimiserChain(Optimisers.Adam(; eta = 1.0f-3), Decay(1.0f-4))
        @test schedulable_fields((chain,)) == (:eta, :lambda)

        @testset "an unambiguous key stays top level" begin
            r = resolve_schedules(
                e, (; eta = _ -> t -> 1.0f-3, aux = _ -> t -> 1.0f0), 10;
                chains = (chain,)
            )
            @test keys(r.opt) == (:eta,)
            @test keys(r.device) == (:aux,)
        end

        @testset "`lambda` matches BOTH, and the error shows the qualified form for THAT key" begin
            err = try
                resolve_schedules(e, (; lambda = _ -> t -> 1.0f-4), 10; chains = (chain,))
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("ambiguous", err.msg)
            @test occursin("device = (; lambda =", err.msg)   # the user's own key, not a generic one
            @test occursin("opt = (; lambda =", err.msg)
            @test occursin("Renaming the field is never required", err.msg)
        end

        @testset "qualifying resolves it, and the two land in different places" begin
            r = resolve_schedules(
                e, (;
                    device = (; lambda = _ -> t -> 0.5f0),
                    opt = (; lambda = _ -> t -> 1.0f-4),
                ), 10; chains = (chain,)
            )
            @test r.device.lambda(1) === 0.5f0
            @test r.opt.lambda(1) === 1.0f-4
        end

        @testset "a key matching neither names the fix" begin
            err = try
                resolve_schedules(e, (; lr = _ -> t -> 1.0f-3), 10; chains = (chain,))
            catch ex
                ex
            end
            @test occursin("matches nothing", err.msg)
            @test occursin("`eta`, not `lr`", err.msg)
            @test occursin("aux", err.msg)                     # names the available Device fields
        end

        @testset "a qualified key naming nothing in its namespace also errors" begin
            @test_throws ErrorException resolve_schedules(
                e, (; opt = (; nope = _ -> t -> 1)), 10;
                chains = (chain,)
            )
            @test_throws ErrorException resolve_schedules(
                e, (; device = (; nope = _ -> t -> 1)), 10;
                chains = (chain,)
            )
        end

        @testset "a parameter-sized rule field is not schedulable" begin
            # `anchor` and `no_decay_mask` are excluded, so a schedule on either is rejected rather than
            # pushing a parameter-sized buffer to device every optimizer step.
            @test :anchor ∉ schedulable_fields((chain,))
            @test :no_decay_mask ∉ schedulable_fields((chain,))
        end
    end

    @testset "path-bound `opt` keys name a group and bind that group's chain only" begin
        e = RatioExp()
        chain = Optimisers.OptimiserChain(Optimisers.Adam(; eta = 1.0f-3), Decay(1.0f-4))
        chains = (chain, chain)                       # one chain per group, the constructor's shape
        groups = (:default, :backbone)

        r = resolve_schedules(
            e, (; opt = (; backbone = (; eta = _ -> t -> 1.0f-3))), 10;
            chains, groups
        )
        @test keys(r.opt) == (Symbol("backbone.eta"),)
        @test r.opt[Symbol("backbone.eta")](1) === 1.0f-3
        @test r.horizon_dependent

        # bare and path coexist: the bare key stays the all-groups shorthand.
        r2 = resolve_schedules(
            e, (; opt = (; eta = _ -> t -> 2.0f-3, backbone = (; eta = _ -> t -> 1.0f-3))), 10;
            chains, groups
        )
        @test keys(r2.opt) == (:eta, Symbol("backbone.eta"))

        # a constant path value is a constant, exactly like a bare one.
        rc = resolve_schedules(
            e, (; opt = (; backbone = (; eta = 1.0f-3))), 10;
            chains, groups
        )
        @test Symbol("opt.backbone.eta") in rc.constant

        # a path key supplied by the ACCESSOR is sourced as such.
        acc = (; opt = (; backbone = (; eta = _ -> t -> 1.0f-3)))
        ra = resolve_schedules(e, acc, 10; chains, groups, accessor = acc)
        @test ra.source[Symbol("opt.backbone.eta")] === :accessor

        # a bad group name is a setup error naming the offender and the real groups.
        err = try
            resolve_schedules(
                e, (; opt = (; nope = (; eta = _ -> t -> 1.0f-3))), 10;
                chains, groups
            )
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("nope", err.msg) && occursin("backbone", err.msg)

        # the field must be schedulable in THAT group's chain, not merely somewhere: `epsilon` is a
        # field of Adam but `nonschedulable` excludes it.
        err2 = try
            resolve_schedules(
                e, (; opt = (; backbone = (; epsilon = _ -> t -> 1.0f-8))), 10;
                chains, groups
            )
        catch ex
            ex
        end
        @test err2 isa ErrorException && occursin("epsilon", err2.msg)

        # a multi-segment path is refused: automatic groups are flat, so paths are ONE level.
        err3 = try
            resolve_schedules(
                e, (; opt = (; backbone = (; deep = (; eta = _ -> t -> 1.0f-3)))), 10;
                chains, groups
            )
        catch ex
            ex
        end
        @test err3 isa ErrorException && occursin("ONE level", err3.msg)

        # an UNQUALIFIED dotted key cannot be a group path: the `:auto` branch has no groups to
        # resolve against, so it is the ordinary "matches nothing" error.
        err4 = try
            resolve_schedules(
                e, (; Symbol("backbone.eta") => _ -> t -> 1.0f-3), 10;
                chains, groups
            )
        catch ex
            ex
        end
        @test err4 isa ErrorException && occursin("matches nothing", err4.msg)
    end

    @testset "a two-group OneCycle holds the LR ratio at every step" begin
        e = RatioExp()
        total = 40
        sched = onecycle(total, 1.0f-3)
        ratios = Float64[]
        for t in 1:total
            eta_t = sched(t)
            d = effective_lr(e, :default, eta_t)
            b = effective_lr(e, :backbone, eta_t)
            @test d ≈ eta_t                        # η_default(t) == eta_sched(t) EXACTLY
            push!(ratios, b / d)
        end
        @test all(r -> isapprox(r, 0.1; rtol = 1.0e-5), ratios)
        @test length(unique(round.(ratios; digits = 6))) == 1

        # The curve really varies, so the ratio holding is not vacuous.
        @test length(unique(round.([sched(t) for t in 1:total]; digits = 8))) > 10

        # With no schedule at all, eta_sched(t) is learning_rate(e).
        @test effective_lr(e, :default) ≈ 1.0f-3
        @test effective_lr(e, :backbone) ≈ 1.0f-4
    end

    @testset "scheduling `opt.lambda` alongside per-group `lambda` accessors is an error" begin
        @experiment struct LamExp
            n::Int = 1
        end
        @eval ReactantNitro.lambda(::LamExp, ::Val{:backbone}) = 1.0f-3
        chain = Optimisers.OptimiserChain(Optimisers.Adam(), Decay(1.0f-4))
        err = try
            resolve_schedules(
                LamExp(), (; opt = (; lambda = _ -> t -> 1.0f-4)), 10;
                chains = (chain,), groups = (:default, :backbone)
            )
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("backbone", err.msg)
        @test occursin("setup error naming both", err.msg)
        @test occursin("eta", err.msg)          # explains why eta is deliberately not symmetric

        # eta IS exempt: per-group learning_rate accessors define ratios that compose with a schedule.
        @test resolve_schedules(
            RatioExp(), (; opt = (; eta = _ -> t -> 1.0f-3)), 10;
            chains = (chain,), groups = (:default, :backbone)
        ) isa
            ResolvedSchedules
    end

    @testset "a PATH `lambda` key conflicts per-group, not whole-table" begin
        @experiment struct PathLamExp
            n::Int = 1
        end
        @eval ReactantNitro.lambda(::PathLamExp, ::Val{:backbone}) = 1.0f-3
        chain = Optimisers.OptimiserChain(Optimisers.Adam(), Decay(1.0f-4))
        chains = (chain, chain)
        groups = (:default, :backbone)

        # the path names the group, so only THAT group's accessor conflicts.
        err = try
            resolve_schedules(
                PathLamExp(), (; opt = (; backbone = (; lambda = _ -> t -> 1.0f-4))), 10;
                chains, groups
            )
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("opt.backbone.lambda", err.msg)
        @test occursin("backbone", err.msg)
        @test occursin("setup error naming both", err.msg)

        # a path for a group with NO accessor is allowed: the per-group rule has no conflict to name.
        @test resolve_schedules(
            PathLamExp(), (; opt = (; default = (; lambda = _ -> t -> 1.0f-4))), 10;
            chains, groups
        ) isa ResolvedSchedules

        # the bare key keeps its whole-table rule: the backbone accessor conflicts regardless.
        err2 = try
            resolve_schedules(
                PathLamExp(), (; opt = (; lambda = _ -> t -> 1.0f-4)), 10;
                chains, groups
            )
        catch ex
            ex
        end
        @test err2 isa ErrorException && occursin("backbone", err2.msg)
    end

    # ── the per-step rebuild ─────────────────────────────────────────────────────────────

    @testset "step_aux rebuilds the device fields in one pass" begin
        e = SchedExp()
        r = resolve_schedules(e, (; aux = _ -> t -> Float32(t)), 10)
        fields = (; lambda = 0.25f0, aux = 1.0f0)
        out = step_aux(fields, r, 7)
        @test keys(out) == keys(fields)
        @test out.lambda === 0.25f0             # untouched: not scheduled
        @test out.aux isa Reactant.RNumber      # scheduled: converted to a device scalar
        @test Float32(out.aux) == 7.0f0
    end

    @testset "step_aux coerces to the field's own type" begin
        # The mistake this exists for: a missing `f0`, so the schedule returns `Float64` into a
        # `Device{Float32}` field. Uncoerced that produces a `ConcretePJRTNumber{Float64}`, which moves
        # `typeof(compile_view(e))` and recompiles the gradient program on EVERY optimizer step.
        e = SchedExp()
        r = resolve_schedules(e, (; aux = _ -> t -> 0.25 * t), 10)

        # From a host field, which is what `fields` holds before setup converts it.
        out = step_aux((; aux = 1.0f0), r, 4)
        @test typeof(out.aux) === typeof(to_device(1.0f0))
        @test Float32(out.aux) ≈ 1.0f0

        # And from a field that is ALREADY device-resident, which is what the run path passes. This is
        # the case `eltype` cannot serve: `eltype` of a device NUMBER is the device number's own type,
        # not `Float32`, so the coercion keys on `typeof(old)` instead.
        out_d = step_aux((; aux = to_device(1.0f0)), r, 4)
        @test typeof(out_d.aux) === typeof(to_device(1.0f0))
        @test Float32(out_d.aux) ≈ 1.0f0
    end

    @testset "step_experiment puts the rebuild back into the experiment" begin
        e = to_device_config(SchedExp())
        r = resolve_schedules(e, (; aux = _ -> t -> Float32(t)), 10)

        e7 = step_experiment(e, r, 7)
        @test Float32(e7.aux) == 7.0f0
        @test e7.lambda === e.lambda            # untouched: not scheduled
        @test e7.n === e.n                      # untouched: not even a Device field
        # The cache keys on `args[1]`'s type at every trace site. If the rebuild moved it, every
        # step would recompile,
        # which is the failure the assertion inside `step_experiment` exists to refuse rather than pay.
        @test typeof(compile_view(e7)) === typeof(compile_view(e))
        @test graphconst_field_hash(compile_view(e7)) ==
            graphconst_field_hash(compile_view(e))

        @testset "a Float64 schedule into a Device{Float32} field does not move the key" begin
            # The end this all serves: the cache key's `args[1]` type is unmoved, so the gradient
            # program is
            # reused rather than recompiled once per optimizer step.
            rf = resolve_schedules(e, (; aux = _ -> t -> 0.25 * t), 10)
            e4 = step_experiment(e, rf, 4)
            @test typeof(compile_view(e4)) === typeof(compile_view(e))
            @test Float32(e4.aux) ≈ 1.0f0
        end

        @testset "and returns `e` ITSELF when nothing device-side is scheduled" begin
            # The overwhelmingly common case, and why `train!` can call this unconditionally: a run with
            # no device schedule pays one `isempty` per optimizer step and copies nothing.
            @test step_experiment(e, nothing, 7) === e
            @test step_experiment(e, resolve_schedules(e, (;), 10), 7) === e
        end
    end

    # ── the binding report ───────────────────────────────────────────────────────────────

    @testset "the binding report names every source" begin
        e = SchedExp()
        chain = Optimisers.OptimiserChain(Optimisers.Adam(; eta = 1.0f-3), Decay(1.0f-4))
        accessor = (; device = (; lambda = _ -> t -> 0.5f0))
        given = merge(accessor, (; eta = _ -> t -> 1.0f-3))
        r = resolve_schedules(e, given, 40; chains = (chain,), accessor)

        txt = binding_report_text(;
            name = "SchedExp", seed = 42, accum = 2, total = 1240,
            splits = [
                (;
                    name = "train", batches = 46, batch_size = 64, samples = 3001,
                    dropped = 57, short_final = nothing,
                ),
                (;
                    name = "val", batches = 8, batch_size = 64, samples = 500,
                    dropped = 0, short_final = 52,
                ),
            ],
            clip = 1.0, clip_source = :keyword, schedules = r,
            groups = [
                (;
                    name = :default, base_eta = 1.0f-3, ratio = 1.0, anchor = :zero,
                    lambda = 1.0f-4, rule = :RAdam, params = 1234567,
                ),
                (;
                    name = :backbone, base_eta = 1.0f-4, ratio = 0.1, anchor = :w0,
                    lambda = 1.0f-3, rule = :RAdam, params = 23456789,
                ),
            ],
            level2 = [(; group = :backbone, present = (:eta, :beta), absent = (:epsilon,))]
        )

        @testset "every source is named" begin
            @test occursin("[train! keyword]", txt)
            @test occursin("[schedules(e)]", txt)
            # Per-key attribution: `eta` came from the keyword, `device.lambda` from the accessor.
            @test occursin(r"eta.*\[train! keyword\]", txt)
            @test occursin(r"device\.lambda.*\[schedules\(e\)\]", txt)
        end

        @testset "the data block reports drop-last, which no check can catch" begin
            @test occursin("2,944 of 3,001 samples", txt)
            @test occursin("57 dropped by drop-last", txt)
            @test occursin("final batch of 52 padded then sliced", txt)
        end

        @testset "the clip line states its own source, the one place a reader sees which won" begin
            @test occursin("global norm 1.0", txt)
            @test occursin("optimizer program only", txt)
        end

        @testset "the groups block names the anchor per group" begin
            @test occursin("decay toward zero", txt)
            @test occursin("decay toward w0", txt)
            @test occursin("23,456,789 params", txt)
            @test occursin("G = 2", txt)
        end

        @testset "the level 2 block flags a scheduled value the factory did not apply" begin
            @test occursin("NOT present: epsilon", txt)
            @test occursin("the factory did not apply it", txt)
        end

        @testset "a source that cannot report samples omits the parenthetical rather than guessing" begin
            bare = binding_report_text(;
                name = "X", seed = 1, accum = 1, total = 10,
                splits = [
                    (;
                        name = "train", batches = 4, batch_size = 8, samples = nothing,
                        dropped = nothing, short_final = nothing,
                    ),
                ],
                clip = 0, clip_source = :field
            )
            @test occursin("4 batches x 8", bare)
            @test !occursin("samples", bare)
            @test occursin("none (threshold 0)", bare)
            @test occursin("[field on e]", bare)
        end
    end

    @testset "the report names the group for each path-bound `opt` binding" begin
        r = ResolvedSchedules(
            (;),
            (;
                Symbol("backbone.eta") => _ -> t -> 1.0f-3,
                Symbol("default.eta") => _ -> t -> 1.0f-3,
            ),
            Dict(Symbol("opt.backbone.eta") => :accessor, Symbol("opt.default.eta") => :keyword),
            Symbol[], true
        )
        txt = binding_report_text(;
            name = "X", seed = 1, accum = 1, total = 10, clip = 0,
            clip_source = :default, schedules = r,
            groups = [
                (;
                    name = :default, base_eta = 1.0f-3, ratio = 1.0, anchor = :zero,
                    lambda = 1.0f-4, rule = :RAdam, params = 1,
                ),
                (;
                    name = :backbone, base_eta = 1.0f-4, ratio = 0.1, anchor = :zero,
                    lambda = 1.0f-3, rule = :RAdam, params = 2,
                ),
            ]
        )
        @test occursin("opt.backbone.eta", txt)
        @test occursin("group :backbone, per-group ratio applied", txt)
        @test occursin("group :default, per-group ratio applied", txt)
        # per-key source attribution, as for the bare keys
        @test occursin(r"opt\.backbone\.eta.*\[schedules\(e\)\]", txt)
        @test occursin(r"opt\.default\.eta.*\[train! keyword\]", txt)
    end

    # ── metric residency ────────────────────────────────────────────────────────────────

    @experiment struct ResidencyExp
        n::Int = 1
    end
    ReactantNitro.forward(::ResidencyExp, model, ps, st; x) = (x, st)
    ReactantNitro.loss(::ResidencyExp, out; y) = sum(out)
    ReactantNitro.metrics(::ResidencyExp, out; y) = (; acc = (1.0f0, 1))
    ReactantNitro.train_metrics(::ResidencyExp, out; y) = (; d = 1.0f0)

    @experiment struct TracedEval
        n::Int = 1
    end
    ReactantNitro.forward(::TracedEval, model, ps, st; x) = (x, st)
    ReactantNitro.loss(::TracedEval, out; y) = sum(out)
    ReactantNitro.metrics(::TracedEval, out; y) = (; acc = (1.0f0, 1))
    ReactantNitro.train_metrics(::TracedEval, out; y) = (; d = 1.0f0)
    ReactantNitro.metrics_residency(::TracedEval, ::Symbol) = :device

    @experiment struct HostTrain
        n::Int = 1
    end
    ReactantNitro.forward(::HostTrain, model, ps, st; x) = (x, st)
    ReactantNitro.loss(::HostTrain, out; y) = sum(out)
    ReactantNitro.train_metrics(::HostTrain, out; y) = (; d = 1.0f0)
    ReactantNitro.metrics_residency(::HostTrain, ::Symbol) = :host

    @testset "metric residency: host or device, per hook" begin
        @testset "the defaults follow the CADENCE, not a blanket preference" begin
            e = ResidencyExp()
            # Once per epoch: the transfer is noise, and untraceable evaluation code is common.
            @test metrics_residency(e, :metrics) === :host
            # Once per micro-batch: a full output batch per micro-batch would dwarf the scalars.
            @test metrics_residency(e, :train_metrics) === :device
        end

        @testset "both hooks accept both values" begin
            @test metrics_residency(TracedEval(), :metrics) === :device
            @test metrics_residency(TracedEval(), :train_metrics) === :device
            @test metrics_residency(HostTrain(), :metrics) === :host
            @test metrics_residency(HostTrain(), :train_metrics) === :host
        end

        @testset "a value that is neither is rejected, naming both" begin
            @experiment struct BadResidency
                n::Int = 1
            end
            @eval ReactantNitro.metrics_residency(::BadResidency, ::Symbol) = :gpu
            err = try
                ReactantNitro.check_residency(BadResidency(), :metrics)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin(":gpu", err.msg)
            @test occursin(":host", err.msg) && occursin(":device", err.msg)
            # A typo would otherwise select the default silently and MOVE WHERE THE METRIC RUNS.
            @test ReactantNitro.check_residency(ResidencyExp(), :metrics) === :host
        end

        @testset "a HOST metric is not in the compile key, so editing it recompiles nothing" begin
            # This is the point of the mechanism rather than a detail of it.
            before = ReactantNitro.hook_worlds(ResidencyExp())
            @eval ReactantNitro.metrics(::ResidencyExp, out; y) = (; acc = (2.0f0, 1))
            @test ReactantNitro.hook_worlds(ResidencyExp()) == before

            # ...while a TRACED one is, because a cached program built against the previous metric would
            # otherwise keep reporting it.
            tbefore = ReactantNitro.hook_worlds(TracedEval())
            @eval ReactantNitro.metrics(::TracedEval, out; y) = (; acc = (2.0f0, 1))
            @test ReactantNitro.hook_worlds(TracedEval()) != tbefore
        end

        @testset "the same asymmetry holds for `train_metrics`" begin
            # Traced by default, so editing it invalidates the GRADIENT program, the expensive one.
            before = ReactantNitro.hook_worlds(ResidencyExp())
            @eval ReactantNitro.train_metrics(::ResidencyExp, out; y) = (; d = 2.0f0)
            @test ReactantNitro.hook_worlds(ResidencyExp()) != before

            # Put it on the host and the same edit is free.
            hbefore = ReactantNitro.hook_worlds(HostTrain())
            @eval ReactantNitro.train_metrics(::HostTrain, out; y) = (; d = 2.0f0)
            @test ReactantNitro.hook_worlds(HostTrain()) == hbefore
        end

        @testset "residency does not touch routing, which is about fields rather than residency" begin
            bh = (; x = randn(F4, 2, 4), y = randn(F4, 2, 4))
            rh = ReactantNitro.resolve_routing(compile_view(ResidencyExp()), bh)
            rd = ReactantNitro.resolve_routing(compile_view(TracedEval()), bh)
            @test keys(rh.metrics) == keys(rd.metrics) == (:y,)
            @test keys(rh.forward) == keys(rd.forward) == (:x,)
        end
    end

    # ── a GraphConst value must hash by CONTENT ─────────────────────────────────────────

    # A struct with a mutable field, which is the shape a model config naturally takes and the one that
    # breaks. Julia has no `hash` method for it, so `Base.hash` falls through to `hash(objectid(x), h)`
    # and reaches the `Vector` by identity instead of descending into it.
    struct HashSpecBad
        channels::Vector{Int}
        name::Symbol
    end

    # The same struct with the three methods the framework's own error message tells you to write.
    struct HashSpecGood
        channels::Vector{Int}
        name::Symbol
    end
    Base.hash(x::HashSpecGood, h::UInt) = hash(x.channels, hash(x.name, hash(:HashSpecGood, h)))
    Base.:(==)(a::HashSpecGood, b::HashSpecGood) = a.channels == b.channels && a.name == b.name
    Base.isequal(a::HashSpecGood, b::HashSpecGood) =
        isequal(a.channels, b.channels) && isequal(a.name, b.name)

    # All-immutable fields are fine with no methods at all: `===` on such a struct is field-wise.
    struct HashSpecFlat
        channels::Int
        name::Symbol
    end

    @experiment struct HashBadExp
        head::GraphConst{HashSpecBad} = HashSpecBad([4, 8], :dsnt)
    end
    @experiment struct HashGoodExp
        head::GraphConst{HashSpecGood} = HashSpecGood([4, 8], :dsnt)
    end
    @experiment struct HashFlatExp
        head::GraphConst{HashSpecFlat} = HashSpecFlat(4, :dsnt)
    end
    @experiment struct HashVecExp
        sizes::GraphConst{Vector{Int}} = [4, 8]
        label::GraphConst{Symbol} = :dsnt
        n::GraphConst{Int} = 4
    end

    @testset "a GraphConst value must hash and compare by CONTENT" begin
        @testset "USING A VECTOR IS FINE, and this is the case that must not regress" begin
            # The whole point of refusing the struct wrapper is that it does NOT cost you vectors.
            # `hash(::AbstractArray)` is content-based, so a bare `GraphConst{Vector{Int}}` keys
            # correctly and the check must never fire on one.
            @test assert_graphconst_hashable(HashVecExp()) === nothing
            a, b = HashVecExp(), HashVecExp()
            @test a.sizes !== b.sizes                                    # genuinely different objects
            @test graphconst_field_hash(compile_view(a)) == graphconst_field_hash(compile_view(b))
            @test graphconst_field_hash(compile_view(HashVecExp(; sizes = [4, 16]))) !=
                graphconst_field_hash(compile_view(a))
        end

        @testset "an all-immutable struct is fine with no methods defined" begin
            @test assert_graphconst_hashable(HashFlatExp()) === nothing
            @test graphconst_field_hash(compile_view(HashFlatExp())) ==
                graphconst_field_hash(compile_view(HashFlatExp()))
        end

        @testset "a struct holding a Vector is REFUSED, and the message names the field and the fix" begin
            err = try
                assert_graphconst_hashable(HashBadExp())
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("head::", err.msg)              # names the field
            @test occursin("HashSpecBad", err.msg)         # and its type
            @test occursin("resume compatibility check", err.msg)
            @test occursin("objectid", err.msg)            # states the cause
            # and hands back the three methods, spelled for THIS type. The type sits in the
            # @testitem's generated module now (ReTestItems), so the printed signature carries a
            # `Main.var"##..."` prefix; the regexes tolerate any module path before the name.
            @test occursin(r"Base\.hash\(x::.*HashSpecBad, h::UInt\)", err.msg)
            @test occursin(r"Base\.:\(==\)\(a::.*HashSpecBad, b::.*HashSpecBad\)", err.msg)
            @test occursin(r"Base\.isequal\(a::.*HashSpecBad, b::.*HashSpecBad\)", err.msg)
            @test occursin("x.channels", err.msg) && occursin("x.name", err.msg)
        end

        @testset "the refusal is not vacuous: the defect it names is real" begin
            # If these ever start passing, the check has become theatre and can be deleted.
            a, b = HashBadExp(), HashBadExp()
            @test graphconst_field_hash(compile_view(a)) != graphconst_field_hash(compile_view(b))
            @test !isequal(graphconst_fields(a).head, graphconst_fields(b).head)
            # The SILENT direction, and the reason this is an error rather than a warning: mutating the
            # value in place leaves the key untouched, so a stale program would be served.
            c = HashBadExp()
            h = graphconst_field_hash(compile_view(c))
            push!(c.head.channels, 16)
            @test graphconst_field_hash(compile_view(c)) == h
        end

        @testset "with the three methods defined, BOTH directions close" begin
            @test assert_graphconst_hashable(HashGoodExp()) === nothing
            a, b = HashGoodExp(), HashGoodExp()
            # 1. two identical constructions agree, so no spurious recompile and no resume refusal
            @test graphconst_field_hash(compile_view(a)) == graphconst_field_hash(compile_view(b))
            @test isequal(graphconst_fields(a), graphconst_fields(b))
            # 2. an in-place mutation now MOVES the key, which the objectid fallback did not
            c = HashGoodExp()
            h = graphconst_field_hash(compile_view(c))
            push!(c.head.channels, 16)
            @test graphconst_field_hash(compile_view(c)) != h
            # 3. and a genuine config change is still a different program
            @test graphconst_field_hash(compile_view(HashGoodExp(; head = HashSpecGood([4, 16], :dsnt)))) !=
                graphconst_field_hash(compile_view(a))
        end

        @testset "it runs at SETUP, so a bad config cannot reach a compile" begin
            # `Nitro(e)` must refuse before it builds anything, which is what makes the check useful
            # rather than a lint nobody calls.
            err = try
                Nitro(
                    HashBadExp(); data = (; test = [(; x = zeros(Float32, 2, 2))]),
                    checkpointer = nothing, run_dir = mktempdir()
                )
            catch e
                e
            end
            @test err isa ErrorException
            @test occursin("does not hash and compare by CONTENT", err.msg)
        end
    end

    # ── the experiment half of the key is resolved once per handle ─────────────────────

    @testset "`graphconst_hash` is frozen at construction and used by every lookup" begin
        # `FrozenExp`, because it is the fixture in this file that defines `build_model`.
        n = Nitro(
            FrozenExp(); run_dir = mktempdir(),
            data = (; train = frozen_data(), val = frozen_data())
        )

        @testset "the frozen value is the one `graphconst_field_hash` computes" begin
            @test n.frozen.graphconst_hash == graphconst_field_hash(compile_view(n.e))
        end

        @testset "supplying it produces the SAME key as recomputing it" begin
            # Or the hoist would silently partition the cache, which presents as a permanent miss.
            ev = compile_view(n.e)
            args = (ev, randn(F4, 3, 2))
            w = n.frozen.worlds_eval
            @test cache_key(identity, ev, args, (), w) ==
                cache_key(identity, ev, args, (), w; gc_hash = n.frozen.graphconst_hash)
            @test cache_key(identity, ev, args, ()) ==
                cache_key(identity, ev, args, (); gc_hash = n.frozen.graphconst_hash)
        end

        @testset "a WRONG `gc_hash` still moves the key, so it is really in there" begin
            # If this passes when the component is dropped, the test above proves nothing.
            ev = compile_view(n.e)
            args = (ev, randn(F4, 3, 2))
            @test cache_key(identity, ev, args, (); gc_hash = UInt(0)) !=
                cache_key(identity, ev, args, ())
        end

        @testset "the ARGUMENT half is still recomputed, which is the ragged-batch guarantee" begin
            # The whole reason shapes are in the key: Reactant's guard covers types but not shapes, so a
            # batch of a different width must MISS. Hoisting the experiment half must not touch this.
            ev = compile_view(n.e)
            g = n.frozen.graphconst_hash
            k2 = cache_key(identity, ev, (ev, randn(F4, 3, 2)), (); gc_hash = g)
            k8 = cache_key(identity, ev, (ev, randn(F4, 3, 8)), (); gc_hash = g)
            @test k2 != k8
            # and same shape, same key, or every batch would recompile
            @test k2 == cache_key(identity, ev, (ev, randn(F4, 3, 2)), (); gc_hash = g)
        end

        @testset "the invariant it rests on is enforced elsewhere, and still is" begin
            # `set_device!` refuses to move the config hash, which is what makes freezing it safe across
            # The adaptive rebuild. Assert the guarantee rather than the comment claiming it.
            ev = compile_view(n.e)
            @test graphconst_field_hash(ev) == n.frozen.graphconst_hash
            @test graphconst_field_hash(compile_view(n.e)) == n.frozen.graphconst_hash
        end
    end

    # ── world-closure staleness guard ────────────────────────────────────────────────────

    # File-level fixtures, for the same reason the method-redefinition testset `@eval`s module-level
    # methods: `@eval` targets Main, so a helper defined inside a `@testset` is a local closure and a
    # redefinition via `@eval` would miss it entirely.
    wc_helper(g) = g
    wc_step(leaf, x, grad) = apply_group(leaf, x, wc_helper(grad))

    @testset "an existing handle is a fixed point; only a new one recompiles" begin
        cache_reset!()
        x0 = randn(F4, 32)
        g = randn(F4, 32) .* 0.1f0
        xd, gd = Reactant.to_rarray(copy(x0)), Reactant.to_rarray(g)
        # `Descent` (SGD), NOT Adam: Adam is scale-invariant in the gradient, so a helper that scales
        # the gradient would produce identical steps and this value assertion would be vacuous.
        build(eta) = to_device_leaf(Optimisers.setup(Optimisers.Descent(to_device(F4(eta))), x0))
        l1 = build(1.0f-3)
        # A handle stand-in: `compile_cached` only reads `.programs` (and `.frozen` when `gc_hash` is
        # not passed, which is why it is passed explicitly here).
        handle = (; programs = Dict{Any, Any}())

        thunk = compile_cached(
            wc_step, compile_view(CacheExp()), l1, xd, gd;
            nitro = handle, gc_hash = UInt(0)
        )
        @test cache_stats().misses == 1
        @test length(handle.programs) == 1          # the handle now holds its own thunk

        # A downstream redefinition drifts the module entry.
        @eval wc_helper(g) = g .* 3.0f0
        staleness = world_closure_staleness()
        @test length(staleness.poisoned) == 1

        # The EXISTING handle keeps its thunk: same object, no miss, no recompile.
        thunk2 = compile_cached(
            wc_step, compile_view(CacheExp()), l1, xd, gd;
            nitro = handle, gc_hash = UInt(0)
        )
        @test thunk2 === thunk
        @test cache_stats().misses == 1

        # A NEW handle has no memo: it consults the module cache, finds the poisoned entry gone, and
        # recompiles against current dispatch.
        handle2 = (; programs = Dict{Any, Any}())
        thunk3 = compile_cached(
            wc_step, compile_view(CacheExp()), l1, xd, gd;
            nitro = handle2, gc_hash = UInt(0)
        )
        @test thunk3 !== thunk
        @test cache_stats().misses == 2
        # Separate rule builds per program: a thunk MUTATES the leaf it is handed, so running both
        # programs on one build would step the second from the first's result. Same seed, same start.
        _, x_after = thunk3(build(1.0f-3), xd, gd)
        _, x_old = thunk2(build(1.0f-3), xd, gd)
        @test maximum(abs.(Array(x_after) .- x0)) > maximum(abs.(Array(x_old) .- x0))
    end

    @testset "world closure: an unrelated definition stays a hit, a used method misses" begin
        @test world_closure_available()          # a load-bearing internal, still there
        cache_reset!()

        x0 = randn(F4, 32)
        g = randn(F4, 32) .* 0.1f0
        xd, gd = Reactant.to_rarray(copy(x0)), Reactant.to_rarray(g)
        build(eta) = to_device_leaf(Optimisers.setup(Optimisers.Adam(; eta = to_device(F4(eta))), x0))
        l1 = build(1.0f-3)

        thunk = compile_cached(wc_step, compile_view(CacheExp()), l1, xd, gd)
        @test cache_stats().misses == 1
        _, x_before = thunk(l1, xd, gd)

        @testset "an unrelated definition neither poisons nor misses" begin
            @eval wc_unrelated_definition() = 42
            staleness = world_closure_staleness()
            @test isempty(staleness.poisoned)
            @test isempty(staleness.drifted)
            thunk_hit = compile_cached(wc_step, compile_view(CacheExp()), l1, xd, gd)
            @test thunk_hit === thunk
            @test cache_stats().misses == 1
            @test cache_stats().hits == 1
        end

        @testset "a method the program USES poisons, and the next compile misses" begin
            @eval wc_helper(g) = g .* 2.0f0
            staleness = world_closure_staleness()
            @test length(staleness.poisoned) == 1
            @test :wc_helper in staleness.drifted
            thunk_miss = compile_cached(wc_step, compile_view(CacheExp()), l1, xd, gd)
            @test thunk_miss !== thunk
            @test cache_stats().misses == 2
            # and the rebuilt program really runs the NEW method: the parameters move twice as far
            _, x_after = thunk_miss(l1, xd, gd)
            @test maximum(abs.(Array(x_after) .- x0)) > maximum(abs.(Array(x_before) .- x0))
        end
    end

    @testset "grad's entry is guarded by fwd_program's closure, not its own" begin
        # The routing: `grad_program` captures `fwd_program` at the argtypes derivable from grad's own
        # (ev, model, ps, st, batch, router), because plain inference cannot descend into Enzyme's
        # backward pass (its own closure is glue-only).
        fake_router = (;
            forward = ReactantNitro.Router{(:x,)}(), loss = nothing, metrics = nothing,
            train_metrics = nothing,
        )
        fake = (Int(1), Float64(2.0), Char('x'), :sym, Vector{F4}([1.0]), 6, 7, 8, fake_router, 10, 11)
        cf, at = _closure_target(grad_program, fake)
        @test cf === fwd_program
        @test at == Tuple{Int, Float64, Char, Symbol, Vector{F4}, ReactantNitro.Router{(:x,)}}
        cf2, at2 = _closure_target(fwd_program, fake[1:5])
        @test cf2 === fwd_program
        @test at2 == Tuple{Int, Float64, Char, Symbol, Vector{F4}}

        # And the capture path really descends through the kwcall shim: capture `fwd_program` at plain
        # host types (where inference is cheap) and find Lux's `apply`, which `FrozenExp.forward` calls
        # with the keyword batch field `img`. This is exactly the wall a direct `forward` capture hits.
        model = frozen_chain()
        ps, st = Lux.setup(Xoshiro(7), model)
        img = randn(F4, 3, 2)
        at3 = Tuple{
            FrozenExp, typeof(model), typeof(ps), typeof(st), NamedTuple{(:img,), Tuple{typeof(img)}},
            ReactantNitro.Router{(:img,)},
        }
        cl = world_closure(fwd_program, at3)
        @test !isempty(cl)
        @test any(p -> string(p[1].name) == "apply", cl)
        # and a freshly captured closure matches current dispatch, so a clean capture never drifts
        @test !closure_drift(cl).drift
    end

end
