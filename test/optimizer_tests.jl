# Optimizer tests.
#
# The control test matters more than the rest: it deliberately leaves the RAdam step counter on
# the host and asserts the result DIFFERS from the host reference. If it passes when the bug is
# present, this suite cannot see the thing it exists to catch.

@testitem "optimizer" begin
    using Test
    using ReactantNitro
    using ReactantNitro: FlatLayout, ScalarMemo, apply_group, assert_device_state, build_layout,
        build_opt_state, cache_reset!, cache_stats, check_allowlist, check_decay_last,
        check_no_duplicate_fields, clip_by_global_norm, construct_rule, effective_lr, flatten,
        global_grad_norm, host_numbers, level1_chain, memo_to_device, no_decay_masks, resolve_hp,
        reuse_refused, to_device, to_device_leaf, to_device_rule, unflatten
    using Functors, JLD2, Lux, Optimisers, Random, Reactant, Statistics

    const F = Float32

    # ── helpers ─────────────────────────────────────────────────────────────────────────

    # The promotion-policy recipe with the policy dialled down, which is the ONLY thing the control test varies.
    bad_device_leaf(l::Optimisers.Leaf) = Optimisers.Leaf(
        bad_device_rule(l.rule), Reactant.to_rarray(l.state; track_numbers = AbstractFloat), l.frozen
    )
    function bad_device_rule(r)
        T = typeof(r); ns = ReactantNitro.nonschedulable(T)
        return T.name.wrapper(
            map(
                f -> f in ns ? getfield(r, f) :
                    Reactant.to_rarray(getfield(r, f); track_numbers = AbstractFloat),
                fieldnames(T)
            )...
        )
    end

    # One optimizer step over G groups, the shape the optimizer program has inside it.
    function group_step(state::Tuple, flat::Tuple, grads::Tuple)
        both = ntuple(gi -> apply_group(state[gi], flat[gi], grads[gi]), length(state))
        return map(first, both), map(last, both)
    end

    host_run(chain, x0, gs) = begin
        state, flat = (to_leaf(chain, x0),), (copy(x0),)
        for g in gs
            state, flat = group_step(state, flat, (g,))
        end
        (flat[1], state[1])
    end
    to_leaf(chain, x) = Optimisers.setup(chain, x)

    function traced_run(chain, x0, gs; normalize = to_device_leaf)
        state = (normalize(to_leaf(chain, x0)),)
        flat = (Reactant.to_rarray(copy(x0)),)
        gd = [(Reactant.to_rarray(g),) for g in gs]
        f = @compile group_step(state, flat, gd[1])
        types = Any[(typeof(state), typeof(flat))]
        for g in gd
            state, flat = f(state, flat, g)
            push!(types, (typeof(state), typeof(flat)))
        end
        return Array(flat[1]), state[1], types
    end

    # ── comparing a traced result to a host one ─────────────────────────────────────────
    #
    # Bitwise equality is not portable: FMA contraction is a legal per-target choice, so a rule can
    # be bitwise on x86-64 and off by a float on aarch64 with neither wrong. The SIZE of the gap is
    # portable, and ulps rather than an absolute bound because `eps(Float32)` is 1 ulp only in
    # `[1, 2)` (128 ulps at 0.01, under one at 1e6).

    # A float's index in the total ordering, so a difference counts representable values. The
    # sign fixup continues the ordering across zero and maps `-0.0` and `0.0` together.
    _ulp_key(x::F) = (b = Int64(reinterpret(Int32, x)); b >= 0 ? b : Int64(typemin(Int32)) - b)

    function _ulps(a::F, b::F)
        (isnan(a) || isnan(b)) && return a === b ? 0 : typemax(Int64)
        (isinf(a) || isinf(b)) && return a === b ? 0 : typemax(Int64)
        return abs(_ulp_key(a) - _ulp_key(b))
    end

    # Enough to tell a contraction from a real bug on a machine you do not have. `repr`
    # round-trips a Float32 exactly; "agrees to all printed digits" is what made the first report
    # unactionable.
    function ulp_report(name, xd_, xh_)
        # `Decay(anchored)`'s HOST run returns a `ConcreteRArray`, and broadcasting over one traces
        # instead of computing. `Array` on an `Array` is just a copy.
        xd, xh = Array(xd_), Array(xh_)
        gaps = _ulps.(xd, xh)
        i = argmax(gaps)
        return (;
            name, bitwise = xd == xh, max_ulps = gaps[i],
            differing = count(!iszero, gaps), n = length(gaps),
            at = i, traced = repr(xd[i]), host = repr(xh[i]),
            max_absdiff = maximum(abs.(xd .- xh)),
            platform = string(Sys.MACHINE, ", julia ", VERSION),
        )
    end

    # A FEW ULPS, not bitwise. Contraction is a per-target choice, so bitwise equality between a
    # traced run and a host one is not portable; the SIZE of the gap is. Measured,
    # arm64-apple-darwin / julia 1.13:
    #
    #     Descent    1 ulp,  3 of 64 elements   (1 ulp on x86-64 too)
    #     Nesterov   2 ulps, 1 of 64 elements,  max_absdiff 1.16e-10   (bitwise on x86-64)
    #
    # with the other eleven bitwise on both, so aarch64 fuses one site x86-64 does not. The second
    # ulp is CANCELLATION, not accumulation: the element that moved sits at 5.9e-4 where a typical
    # one is near 1, so its ulp is 2048x finer and one rounding costs two. Harder cancellation on a
    # future target costs a few more, hence the headroom. The control below drifts >1e-4, upwards
    # of 800 ulps, so rounding and a wrong result stay two orders of magnitude apart.
    const MAX_CONTRACTION_ULPS = 8

    Random.seed!(0x00C0FFEE)
    const X0 = randn(F, 64)
    const GS = [randn(F, 64) .* 0.1f0 for _ in 1:3]

    @experiment struct OptExp
        lr::Float32 = 1.0f-3
    end
    ReactantNitro.learning_rate(e::OptExp) = e.lr

    @experiment struct TwoGroup
        unused::Int = 0
        max_epochs::Host{Int} = 2
    end
    ReactantNitro.param_group(::TwoGroup, ks) = ks[1] === :backbone ? :backbone : :default
    ReactantNitro.learning_rate(::TwoGroup) = 1.0f-3
    ReactantNitro.learning_rate(::TwoGroup, ::Val{:backbone}) = 1.0f-4
    ReactantNitro.optimizer(::TwoGroup, ::Val{:backbone}) = Optimisers.Adam

    mlp() = Lux.Chain(;
        backbone = Lux.Chain(Lux.Dense(4 => 8, tanh), Lux.Dense(8 => 8, tanh)),
        head = Lux.Dense(8 => 3)
    )

    # The runnable half of the fixture: the automatic loop needs the model/data/loss hooks, which the
    # layout-level tests above never call. Additive, so the unit tests keep their `TwoGroup()`.
    ReactantNitro.build_model(e::TwoGroup, rng) =
        Lux.setup(rng, mlp()) |> ((ps, st),) -> (mlp(), ps, st)
    function two_group_data(; n_batches = 4, bs = 8, seed = 1)
        rng = Random.MersenneTwister(seed)
        return [(; x = randn(rng, F, 4, bs), y = randn(rng, F, 3, bs)) for _ in 1:n_batches]
    end
    ReactantNitro.build_data(::TwoGroup, dist) = (; train = two_group_data())
    ReactantNitro.forward(::TwoGroup, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::TwoGroup, ŷ; y) = mean(abs2, ŷ .- y)

    # ── state normalization ──────────────────────────────────────────────────────────────

    @testset "state normalization: no host Number survives, fresh and after a JLD2 round trip" begin
        leaf = Optimisers.setup(Optimisers.RAdam(; eta = 1.0f-3), X0)

        # The unnormalized state is the failure this test exists to detect, so assert it IS detectable.
        @test !isempty(host_numbers(leaf.state))
        @test any(contains("Int64"), host_numbers(leaf.state))   # RAdam's `t`

        dl = to_device_leaf(leaf)
        @test isempty(host_numbers(dl.state))
        @test assert_device_state(dl.state) === nothing
        @test dl.state[4] isa Reactant.RNumber        # the step counter, the whole point of this test

        # A single to_rarray over the Leaf cannot work: `frozen` is declared ::Bool and is not a type
        # parameter, so the rule and the state MUST convert separately.
        @test_throws Reactant.NoFieldMatchError Reactant.to_rarray(leaf; track_numbers = Number)

        # The resume path. The checkpoint stores HOST values, so the flow is write host, read host,
        # normalize on the way in; the second normalization is what keeps this test's bug from returning through
        # `resume = :auto`, which is the DEFAULT path.
        mktempdir() do dir
            path = joinpath(dir, "state.jld2")
            JLD2.jldsave(path; opt_state = leaf)
            restored = JLD2.load(path, "opt_state")
            @test !isempty(host_numbers(restored.state))          # host on disk, by design
            renorm = to_device_leaf(restored)
            @test isempty(host_numbers(renorm.state))
            @test assert_device_state(renorm.state) === nothing
        end

        # The assertion's message has to name the leaked path, or it cannot be acted on.
        err = try
            assert_device_state(leaf.state)
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("BAKES as a trace-time constant", err.msg) && occursin("Int64", err.msg)
    end

    # ── the control test ─────────────────────────────────────────────────────────────────

    @testset "CONTROL: leaving `t` host must produce a DIFFERENT result" begin
        rule = Optimisers.RAdam(; eta = 1.0f-3)
        xh, lh = host_run(rule, X0, GS)
        xgood, lgood, _ = traced_run(rule, X0, GS)
        xbad, lbad, _ = traced_run(rule, X0, GS; normalize = bad_device_leaf)

        # The counter advances on the host and through the correctly normalized program...
        @test lh.state[4] == 4
        @test Int(lgood.state[4]) == 4
        # ...and FREEZES when it is left on the host, baked as a trace-time constant.
        @test lbad.state[4] isa Int
        @test lbad.state[4] == 2

        # Both sides in ULPS, which is what makes the contrast the point: correct normalization
        # lands within contraction distance of the host, the control is orders of magnitude out.
        good = ulp_report("normalized", xgood, xh)
        bad = ulp_report("host-t", xbad, xh)
        good.bitwise || @info "optimizer: traced and host differ" good...
        @test good.max_ulps <= MAX_CONTRACTION_ULPS
        @test bad.max_ulps > 100                  # the control's assertion
        @test maximum(abs.(xbad .- xh)) > 1.0e-4  # and in absolute terms too
    end

    # ── per-rule admission ───────────────────────────────────────────────────────────────

    @testset "per-rule admission: traces, type fixed point, bitwise vs host over 3 steps" begin
        admitted = [
            ("Descent", Optimisers.Descent(F(0.1))),
            ("Momentum", Optimisers.Momentum(F(0.01), F(0.9))),
            ("Nesterov", Optimisers.Nesterov(F(0.01), F(0.9))),
            ("Adam", Optimisers.Adam(; eta = F(1.0e-3))),
            ("AdamW", Optimisers.AdamW(; eta = F(1.0e-3))),
            ("RAdam", Optimisers.RAdam(; eta = F(1.0e-3))),
            ("WeightDecay", Optimisers.WeightDecay(F(1.0e-4))),
            ("ClipGrad", Optimisers.ClipGrad(F(1.0))),
            ("ClipNorm(throw=false)", Optimisers.ClipNorm(F(1.0), 2; throw = false)),
            ("Decay(zero)", Decay(to_device(F(1.0e-4)))),
            ("Decay(anchored)", Decay(to_device(F(1.0e-4)), Reactant.to_rarray(copy(X0)))),
            (
                "Chain(RAdam, Decay)", Optimisers.OptimiserChain(
                    Optimisers.RAdam(; eta = F(1.0e-3)),
                    Decay(to_device(F(1.0e-4)))
                ),
            ),
            (
                "Chain(ClipNorm, Adam)", Optimisers.OptimiserChain(
                    Optimisers.ClipNorm(F(1.0), 2; throw = false), Optimisers.Adam(; eta = F(1.0e-3))
                ),
            ),
        ]
        reports = NamedTuple[]
        for (name, chain) in admitted
            @testset "$name" begin
                @test check_allowlist(chain) === nothing
                xh, _ = host_run(chain, X0, GS)
                xd, _, types = traced_run(chain, X0, GS)
                @test allequal(types)                     # type fixed point across the boundary
                rep = ulp_report(name, xd, xh)
                push!(reports, rep)
                # Before the assertion and on any disagreement, not just a failing one: `@test`
                # reports the comparison, never the numbers, and the numbers are the diagnosis.
                # Two ulps on one element is contraction; a wide gap on every one is a broken
                # update.
                rep.bitwise || @info "optimizer: traced and host differ" rep...
                @test rep.max_ulps <= MAX_CONTRACTION_ULPS
            end
        end
        # Always emitted: which rules are bitwise HERE is what a new target's first failure needs,
        # and it is invisible if only logged on failure.
        @info "optimizer: bitwise agreement by rule" platform = string(Sys.MACHINE) bitwise =
            [r.name for r in reports if r.bitwise] contracted =
            [(r.name, r.max_ulps) for r in reports if !r.bitwise]
    end

    @testset "addendum: Descent is 1 ulp, and exactly XLA's fused multiply-add" begin
        # The per-rule table records Descent at maxdiff 0.0; measured here it is 1 ulp, because `x - eta*g` is
        # a multiply immediately consumed by a subtract and XLA contracts it into an FMA, which rounds
        # once where the host rounds twice. Asserted EXACTLY against the FMA reference rather than with a
        # tolerance, so this still fails loudly if XLA's behavior changes, which is what this test is for.
        eta = F(0.1)
        xd, _, _ = traced_run(Optimisers.Descent(eta), X0, GS[1:1])
        @test xd == fma.(-eta, GS[1], X0)
        @test xd != X0 .- eta .* GS[1]
        @test maximum(abs.(xd .- (X0 .- eta .* GS[1]))) <= eps(F)

        # The diagnosis, as two falsifiable predictions: an exact-power-of-two eta cannot double-round,
        # and the difference does not accumulate.
        for p2 in F[0.25, 0.5, 0.125]
            xh2, _ = host_run(Optimisers.Descent(p2), X0, GS)
            xd2, _, _ = traced_run(Optimisers.Descent(p2), X0, GS)
            @test xd2 == xh2
        end
        xh3, _ = host_run(Optimisers.Descent(eta), X0, GS)
        xd3, _, _ = traced_run(Optimisers.Descent(eta), X0, GS)
        @test maximum(abs.(xd3 .- xh3)) <= eps(F)
    end

    # ── promotion policy and ClipNorm setup ──────────────────────────────────────────────

    @testset "promotion policy: nonschedulable stays host, everything else goes to device" begin
        # The ClipNorm case is load-bearing and must be constructed BARE. Written against a Float32
        # omega the two fields differ in type, which is a case the hazard does not arise in, so the
        # test would pass vacuously.
        cn = Optimisers.ClipNorm()
        @test typeof(cn.omega) === typeof(cn.p) === Float64      # the premise of the whole argument
        cnd = to_device_rule(cn)
        @test cnd.omega isa Reactant.RNumber                      # schedulable: promoted
        @test cnd.p isa Float64                                   # structural: MUST stay host
        @test cnd.throw isa Bool

        # Assert the RESOLVED trait per allowlist entry rather than assuming the declarations compose.
        resolved = Dict(
            Optimisers.Descent => (), Optimisers.Momentum => (), Optimisers.Nesterov => (),
            Optimisers.Adam => (:beta, :epsilon), Optimisers.RAdam => (:beta, :epsilon),
            Optimisers.AdamW => (:beta, :epsilon, :couple),
            Optimisers.WeightDecay => (), Optimisers.ClipGrad => (),
            Optimisers.ClipNorm => (:p, :throw), Decay => (:anchor, :no_decay_mask)
        )
        for (R, want) in resolved
            @test ReactantNitro.nonschedulable(R) == want
        end

        # The general property, over every allowlist entry: promoted iff not in the opt-out.
        for rule in (
                Optimisers.Descent(), Optimisers.Momentum(), Optimisers.Nesterov(),
                Optimisers.Adam(), Optimisers.RAdam(), Optimisers.AdamW(),
                Optimisers.WeightDecay(), Optimisers.ClipGrad(),
                Optimisers.ClipNorm(1.0, 2; throw = false),
            )
            T = typeof(rule)
            ns = ReactantNitro.nonschedulable(T)
            d = to_device_rule(rule)
            for f in fieldnames(T)
                v = getfield(d, f)
                if f in ns
                    @test !(v isa Reactant.RNumber)
                else
                    @test v isa Reactant.RNumber
                end
            end
        end

        # A Union method would be shadowed by any more specific one, silently, so the schedule contract requires one
        # method per concrete rule. AdamW resolving to (:couple,) alone is exactly that failure.
        @test :beta in ReactantNitro.nonschedulable(Optimisers.AdamW)
    end

    @testset "ClipNorm with the default throw = true is rejected at setup, naming the flag" begin
        err = try
            check_allowlist(Optimisers.ClipNorm(1.0, 2))
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("throw", err.msg)
        @test occursin("throw = false", err.msg)
        @test check_allowlist(Optimisers.ClipNorm(1.0, 2; throw = false)) === nothing

        # The rejection is not precautionary: the unrejected form really does fail to trace, and with a
        # message that names neither the rule nor the flag.
        @test_throws Exception traced_run(Optimisers.ClipNorm(F(1.0), 2), X0, GS[1:1])

        # AccumGrad traces and is WRONG, so it is off the list for a different reason.
        @test_throws ErrorException check_allowlist(Optimisers.AccumGrad(2))
        # And a rule nobody has measured is rejected rather than assumed.
        @test_throws ErrorException check_allowlist(Optimisers.OAdam())
    end

    # ── the flat layout ──────────────────────────────────────────────────────────────────

    @testset "the flat map: a tree round-trips unchanged" begin
        rng = Random.MersenneTwister(0)
        ps, _ = Lux.setup(rng, mlp())
        e = OptExp()
        layout = build_layout(e, ps)

        @test layout.groups == (:default,)
        @test length(layout.permutation) == 6
        @test sum(r.len for r in layout.permutation) == layout.lengths[1]

        flat = flatten(ps, layout)
        @test flat isa NTuple{1, Any}
        @test length(flat[1]) == 139

        back = unflatten(flat, ps, layout)
        @test typeof(back) === typeof(ps)
        @test all(Functors.fleaves(back) .== Functors.fleaves(ps))

        # Leaf order is Functors traversal order, depth-first and deterministic, which is what makes the
        # layout reproducible across sessions rather than a property of the framework.
        @test [r.keypath for r in layout.permutation] == ReactantNitro.leaf_keypaths(ps)
        @test build_layout(e, ps).permutation == layout.permutation

        # Offsets partition the group buffer exactly: contiguous, no gaps, no overlap.
        offs = [(r.offset, r.len) for r in layout.permutation]
        @test first(offs)[1] == 0
        @test all(offs[i][1] == offs[i - 1][1] + offs[i - 1][2] for i in 2:length(offs))
    end

    @testset "param groups: contiguous ranges, stable order, :default first" begin
        rng = Random.MersenneTwister(0)
        ps, _ = Lux.setup(rng, mlp())
        layout = build_layout(TwoGroup(), ps)

        @test layout.groups == (:default, :backbone)     # :default is group 1
        @test [r.group for r in layout.permutation] == [
            :default, :default, :backbone, :backbone,
            :backbone, :backbone,
        ]
        # Within a group, leaves keep traversal order: the sort is STABLE, which is what makes the
        # layout deterministic.
        bb = [r.keypath for r in layout.permutation if r.group === :backbone]
        @test bb == [kp for kp in ReactantNitro.leaf_keypaths(ps) if kp[1] === :backbone]

        @test layout.lengths == (27, 112)                # head 24+3, backbone 32+8+64+8
        flat = flatten(ps, layout)
        @test length.(flat) == layout.lengths
        @test typeof(unflatten(flat, ps, layout)) === typeof(ps)

        # The default exclusion is structural: every 1-D leaf (biases, norm affines) is masked out.
        masks = no_decay_masks(TwoGroup(), ps, layout)
        @test length.(masks) == layout.lengths
        @test all(m -> all(x -> x in (0.0f0, 1.0f0), m), masks)
        @test sum(sum, masks) == 32 + 64 + 24            # the three weight matrices, no biases
    end

    # ── the `no_decay` hook ──────────────────────────────────────────────────────────────

    @experiment struct AddExclusion
        n::Int = 1
    end
    # ADD to the defaults, which is the case the exported `default_no_decay` exists for.
    ReactantNitro.no_decay(::AddExclusion, ks, x) = default_no_decay(ks, x) || (:head in ks)

    @experiment struct DecayEverything
        n::Int = 1
    end
    # DISABLE the defaults entirely.
    ReactantNitro.no_decay(::DecayEverything, ks, x) = false

    @experiment struct OnlyHeadExcluded
        n::Int = 1
    end
    # REPLACE them: 1-D leaves are decayed again, and only the head's weight matrix is excluded. This is
    # a shape real fine-tuning setups ask for, and it is not expressible through `param_group` without
    # also splitting the learning-rate ratio.
    ReactantNitro.no_decay(::OnlyHeadExcluded, ks, x) = (:head in ks) && (:weight in ks)

    @experiment struct ShapeOnly
        n::Int = 1
    end
    # A rule keyed on the LEAF rather than the keypath, which is only possible because the hook is
    # handed the array. Excludes every matrix and decays every vector, the inverse of the default.
    ReactantNitro.no_decay(::ShapeOnly, ks, x) = ndims(x) == 2

    @testset "`no_decay` is per LEAF, and the default is what the framework always did" begin
        rng = Random.MersenneTwister(0)
        ps, _ = Lux.setup(rng, mlp())
        layout = build_layout(OptExp(), ps)
        total = sum(layout.lengths)

        # `mlp()` is backbone = Chain(Dense(4=>8), Dense(8=>8)), head = Dense(8=>3).
        weights = 32 + 64 + 24
        biases = 8 + 8 + 3
        @test total == weights + biases == 139

        decayed(e) = sum(sum, no_decay_masks(e, ps, layout))   # the mask is 1 where DECAYED

        @testset "the default is `ndims == 1`, unchanged, so no run's numerics move" begin
            @test default_no_decay((:head, :weight), zeros(Float32, 3, 8)) === false
            @test default_no_decay((:head, :bias), zeros(Float32, 3)) === true
            # A bare experiment defines no method and gets exactly the old hardcoded behavior.
            @test decayed(OptExp()) == weights
        end

        @testset "ADD: the defaults survive and the extra rule is unioned in" begin
            # Every bias still excluded, and now the head entirely, so only backbone weights decay.
            @test decayed(AddExclusion()) == weights - 24
        end

        @testset "DISABLE: `false` decays everything, biases and norm affines included" begin
            m = no_decay_masks(DecayEverything(), ps, layout)
            @test sum(sum, m) == total
            @test all(x -> x == 1.0f0, reduce(vcat, collect.(m)))
        end

        @testset "REPLACE: the built-ins are gone, only the stated rule applies" begin
            # Every 1-D leaf is decayed again; only the head's weight matrix is excluded.
            @test decayed(OnlyHeadExcluded()) == total - 24
        end

        @testset "the hook sees the LEAF, so a rule can key on shape rather than name" begin
            # The exact inverse of the default, which no keypath-only predicate could express.
            @test decayed(ShapeOnly()) == biases
        end
    end

    # ── per-group slices ─────────────────────────────────────────────────────────────────

    @testset "per-group slices: two groups, different rules and hyperparameters, one program" begin
        rng = Random.MersenneTwister(0)
        ps, _ = Lux.setup(rng, mlp())
        e = TwoGroup()
        layout = build_layout(e, ps)
        @test ReactantNitro.n_groups(layout) == 2

        # Per-group accessors define RATIOS; the backbone stays a tenth of the default.
        @test effective_lr(e, :default) ≈ 1.0f-3
        @test effective_lr(e, :backbone) ≈ 1.0f-4
        @test effective_lr(e, :backbone) / effective_lr(e, :default) ≈ 0.1 rtol = 1.0e-6

        hp1 = resolve_hp(e, layout, 1)
        hp2 = resolve_hp(e, layout, 2)
        @test hp1.eta isa Reactant.RNumber
        @test hp1.lambda === nothing                     # lambda defaults to 0 -> NO Decay in the chain
        chain1 = level1_chain(e, :default, hp1)
        chain2 = level1_chain(e, :backbone, hp2)
        @test chain1 isa Optimisers.RAdam                # Level 0 default
        @test chain2 isa Optimisers.Adam                 # Level 1 per group

        flat = map(b -> Reactant.to_rarray(collect(F, b)), flatten(ps, layout))
        hostflat = map(collect, flatten(ps, layout))
        state = (
            to_device_leaf(Optimisers.setup(chain1, flat[1])),
            to_device_leaf(Optimisers.setup(chain2, flat[2])),
        )
        hstate = (Optimisers.setup(chain1, hostflat[1]), Optimisers.setup(chain2, hostflat[2]))

        # The collection is an NTuple, never a Vector: with two different rules a Vector infers as
        # Vector{Optimisers.Leaf}, which is abstractly typed, and the thunk guard rejects the 2nd call.
        @test state isa Tuple
        @test isconcretetype(typeof(state))
        @test !isconcretetype(eltype([state...]))

        gs = (randn(F, layout.lengths[1]) .* 0.1f0, randn(F, layout.lengths[2]) .* 0.1f0)
        gd = map(Reactant.to_rarray, gs)
        f = @compile group_step(state, flat, gd)
        t0 = typeof(state)
        for _ in 1:3
            state, flat = f(state, flat, gd)
            hstate, hostflat = group_step(hstate, hostflat, gs)
        end
        @test typeof(state) === t0                       # fixed point
        # Three optimizer steps on device against three on host, so the same contraction bound
        # applies here as to the per-rule table; exact equality would break on the first target
        # that fuses a site this chain's rules currently do not.
        for gi in 1:2
            rep = ulp_report("group $gi", flat[gi], hostflat[gi])
            rep.bitwise || @info "optimizer: traced and host differ" rep...
            @test rep.max_ulps <= MAX_CONTRACTION_ULPS
        end

        # And build_opt_state assembles exactly that, with the promotion-policy assertion run on the result.
        st = build_opt_state(e, map(b -> Reactant.to_rarray(collect(F, b)), flatten(ps, layout)), layout)
        @test st isa NTuple{2, Any}
        @test all(gi -> isempty(host_numbers(st[gi].state)), 1:2)
    end

    # ── path-bound `opt` schedules: the automatic loop ──────────────────────────────────

    @testset "path-bound `opt` schedules: per-group curves in the automatic loop" begin
        # A path key (`opt.backbone.eta`) names a group and binds that group's chain only, with the
        # per-group ratio STILL applied; a group with no path key falls back to the bare key, then to
        # its base rate. `rebuild_rules` looks each group's eta up per group, so the two curves can
        # differ, and the rebuild is type-preserving, so one program serves every step (the schedule-rebuild test's
        # property, asserted by the cache counter below).
        ps_moved(a, b) = maximum(abs, vec(Array(a)) .- vec(Array(b)))
        moved(n) = flatten(parameters(n), n.layout)
        rule_eta(n, gi) = Float32(n.opt_state[gi].rule.eta)

        # backbone only, at a tenth of the base: the path binds :backbone's chain; :default has no
        # path key and no bare key, so it falls back to its base rate.
        cache_reset!()
        n = Nitro(
            TwoGroup(); schedules = (; opt = (; backbone = (; eta = _ -> _ -> 1.0f-5))),
            checkpointer = nothing, run_dir = mktempdir()
        )
        @test keys(n.schedules.opt) == (Symbol("backbone.eta"),)
        @test occursin("group :backbone", sprint(show, MIME"text/plain"(), n))
        d0 = moved(n)[1]
        b0 = moved(n)[2]
        train!(n)
        # The rule eta the run actually stepped with: :backbone's chain at 1e-5 x its ratio 0.1, and
        # :default at its unscheduled base 1e-3. Setup would have left both at their base rates, so
        # these values prove the schedule was applied at runtime, per group.
        @test rule_eta(n, 1) ≈ 1.0f-3 rtol = 1.0e-5      # :default, no path, no bare key -> base
        @test rule_eta(n, 2) ≈ 1.0f-6 rtol = 1.0e-5      # :backbone, 1e-5 x (1e-4 / 1e-3)
        # The schedule-rebuild test's property: the rebuilt rules re-entered the same programs, so nothing recompiled.
        @test cache_stats().misses == 2
        # Liveness through the parameters, not only the rule fields: at 1e-6 against 1e-3 the backbone
        # must move a fraction of the default group.
        @test ps_moved(moved(n)[2], b0) < ps_moved(moved(n)[1], d0) / 5

        # bare + path: the path OVERRIDES the bare key for its group, and the ratio is still applied
        # to the path value. :default takes the bare 3e-3 (ratio 1.0); :backbone takes ITS path 1e-2
        # x 0.1 = 1e-3, NOT the bare 3e-3 x 0.1 = 3e-4.
        cache_reset!()
        nb = Nitro(
            TwoGroup(); schedules = (;
                opt = (;
                    eta = _ -> _ -> 3.0f-3, backbone = (; eta = _ -> _ -> 1.0f-2),
                ),
            ),
            checkpointer = nothing, run_dir = mktempdir()
        )
        @test keys(nb.schedules.opt) == (:eta, Symbol("backbone.eta"))
        train!(nb)
        @test rule_eta(nb, 1) ≈ 3.0f-3 rtol = 1.0e-5    # bare key, ratio 1.0
        @test rule_eta(nb, 2) ≈ 1.0f-3 rtol = 1.0e-5    # path value 1e-2 x ratio 0.1, not the bare
        @test cache_stats().misses == 2

        # a bad group name is a SETUP error naming the offender.
        err = try
            Nitro(
                TwoGroup(); schedules = (; opt = (; nope = (; eta = _ -> _ -> 1.0f-3))),
                checkpointer = nothing, run_dir = mktempdir()
            )
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException && occursin("nope", err.msg)

        # a field that is not schedulable in THAT group's chain is refused, as for a bare key.
        err2 = try
            Nitro(
                TwoGroup(); schedules = (; opt = (; backbone = (; epsilon = _ -> _ -> 1.0f-8))),
                checkpointer = nothing, run_dir = mktempdir()
            )
            nothing
        catch ex
            ex
        end
        @test err2 isa ErrorException && occursin("epsilon", err2.msg)
    end

    # ── the composition checks ───────────────────────────────────────────────────────────

    @testset "a base rule declaring `lambda` is rejected at Level 1, naming the fix" begin
        @experiment struct AdamWExp
            unused::Int = 0
        end
        ReactantNitro.optimizer(::AdamWExp) = Optimisers.AdamW
        e = AdamWExp()
        layout = build_layout(e, (; w = randn(F, 2, 2)))
        err = try
            level1_chain(e, :default, resolve_hp(e, layout, 1))
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("AdamW", err.msg) && occursin("lambda", err.msg)
        @test occursin("TWICE", err.msg)
        @test occursin("Adam", err.msg)                  # points at the equivalent composition

        # Level 2 may still use AdamW: there the user owns the chain and the framework composes no tail.
        @test check_allowlist(Optimisers.AdamW()) === nothing
    end

    @testset "Decay must be last in the chain" begin
        d = Decay(to_device(F(1.0e-4)))
        @test check_decay_last(Optimisers.OptimiserChain(Optimisers.Adam(), d)) === nothing
        err = try
            check_decay_last(Optimisers.OptimiserChain(d, Optimisers.Adam()))
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("LAST", err.msg) && occursin("coupled L2", err.msg)
    end

    @testset "two rules in one chain declaring the same schedulable field" begin
        @test check_no_duplicate_fields(
            Optimisers.OptimiserChain(Optimisers.RAdam(), Decay(1.0f-4))
        ) === nothing
        err = try
            check_no_duplicate_fields(Optimisers.OptimiserChain(Optimisers.Adam(), Optimisers.Descent()))
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("eta", err.msg)                   # both declare `eta`
        @test occursin("Adam", err.msg) && occursin("Descent", err.msg)
    end

    @testset "the framework constructs Level 0 and 1 rules by field name" begin
        hp = (; eta = to_device(F(2.0e-3)), lambda = nothing, anchor = nothing, no_decay_mask = true)
        r = construct_rule(Optimisers.RAdam, hp)
        @test r isa Optimisers.RAdam
        @test r.eta isa Reactant.RNumber                 # supplied by hp
        @test r.beta == Optimisers.RAdam().beta          # not in hp: the rule's own default

        # The eltype of the carrier is load-bearing. A Float64 eta on a Float32 buffer promotes the
        # PARAMETERS to Float64, and the thunk guard then rejects the second call with a message about
        # argument types rather than about precision.
        layout = build_layout(OptExp(), (; w = randn(F, 4)))
        @test resolve_hp(OptExp(), layout, 1).eta isa Reactant.RNumber{F}
    end

    # ── clipping ─────────────────────────────────────────────────────────────────────────

    @testset "clipping" begin
        g = (randn(F, 32) .* 3, randn(F, 16) .* 3)
        n = sqrt(sum(sum(abs2, gi) for gi in g))
        @test global_grad_norm(g) ≈ n

        @testset "a gradient over the threshold comes out at exactly the threshold" begin
            thr = F(0.5)
            @test n > thr
            clipped = clip_by_global_norm(g, thr)
            @test global_grad_norm(clipped) ≈ thr rtol = 1.0e-6
            # Direction is preserved: it is a scale, not a projection.
            @test clipped[1] ./ g[1] ≈ fill(thr / n, 32) rtol = 1.0e-5
        end

        @testset "a gradient under the threshold is untouched" begin
            big = F(1.0e6)
            @test all(clip_by_global_norm(g, big) .== g)
        end

        @testset "threshold 0 means OFF, and emits no ops at all" begin
            # A HOST branch, so the accumulator comes back identically, not scaled to zero norm. With
            # the threshold device-resident, `0` would scale the gradient to zero and `0/0` on an
            # all-zero gradient would silently yield NaN.
            @test clip_by_global_norm(g, 0) === g
            @test clip_by_global_norm(g, F(0)) === g

            # And the compiled program is identical to one from a run that never configured clipping.
            gd = map(Reactant.to_rarray, g)
            clipped_off(x) = clip_by_global_norm(x, 0.0f0)
            never_clipped(x) = x
            body(m) = join(split(string(m), "\n")[2:end], "\n")   # drop the module NAME line, which
            # Reactant derives from the function
            hlo_off = body(Reactant.@code_hlo optimize = false clipped_off(gd))
            hlo_none = body(Reactant.@code_hlo optimize = false never_clipped(gd))
            @test hlo_off == hlo_none
            @test !occursin("sqrt", hlo_off)
            @test !occursin("divide", hlo_off)

            clipped_on(x) = clip_by_global_norm(x, 1.0f0)
            hlo_on = body(Reactant.@code_hlo optimize = false clipped_on(gd))
            @test hlo_on != hlo_none
            @test occursin("sqrt", hlo_on)               # the norm really is being computed
        end

        @testset "it traces, and matches the host path bitwise" begin
            gd = map(Reactant.to_rarray, g)
            clip1(x) = clip_by_global_norm(x, 0.5f0)
            f = @compile clip1(gd)
            out = f(gd)
            @test typeof(out) === typeof(gd)             # fixed point
            @test all(Array.(out) .== clip_by_global_norm(g, 0.5f0))
        end

        # This test's middle claim, that `accum = 2` clips ONCE on the accumulated gradient rather
        # than per micro-batch, needs the gradient accumulation loop. It is the Lightning-parity
        # claim and the one an implementer is most likely to get wrong, so it is tracked here rather
        # than quietly dropped.
    end

    # ── The :w0 anchor path, which the first GPU run found broken ───────────────────────

    @experiment struct AnchorTwoGroup
        n::GraphConst{Int} = 1
    end
    ReactantNitro.param_group(::AnchorTwoGroup, ks) = ks[1] === :backbone ? :backbone : :default
    ReactantNitro.decay_anchor(::AnchorTwoGroup, ::Val{:backbone}) = :w0
    ReactantNitro.lambda(::AnchorTwoGroup, ::Val{g}) where {g} = 1.0f-4

    @testset "the :w0 anchor is built from HOST values and placed, never concatenated on device" begin
        # The GPU-only bug the gate run hit: `w0` is device-resident by the time `decay_anchors` runs,
        # and `_concat_group`'s multi-leaf branch is `vcat(map(vec, ...)...)`, which scalar-indexes.
        # Reactant refuses that on a GPU and the run dies during `Nitro` construction, before step 1.
        #
        # NO CPU TEST CAN CATCH THE SCALAR INDEXING ITSELF: on CPU it is legal and merely slow, so the
        # suite was green while doing something pathological. What this test does instead is pin the two
        # preconditions that made it reachable, so the path cannot silently stop being exercised, and
        # assert the values are right.
        rng = Random.MersenneTwister(0)
        ps, _ = Lux.setup(rng, mlp())
        e = AnchorTwoGroup()
        layout = build_layout(e, ps)

        @testset "the preconditions: two groups, a non-:zero anchor, and a MULTI-LEAF group" begin
            # Both were needed for the bug and either one missing makes the test vacuous. `decay_anchors`
            # returns `nothing` early with no non-:zero anchor, and the single-leaf branch of
            # `_concat_group` takes `vec` rather than `vcat`, so it never reaches the failing code.
            @test ReactantNitro.n_groups(layout) == 2
            @test decay_anchor(e, Val(:backbone)) === :w0
            @test all(length(layout.group_rows[gi]) > 1 for gi in 1:2)
        end

        w0 = Reactant.to_rarray(ps)                       # device, exactly as setup leaves it
        anchors = ReactantNitro.decay_anchors(e, w0, layout)

        @test anchors !== nothing
        @test anchors[1] === nothing                      # :default is :zero, so no buffer at all
        @test anchors[2] isa Reactant.AbstractConcreteArray

        @testset "and it equals the host flatten of the same group, elementwise" begin
            expected = ReactantNitro.flatten(ps, layout)[2]
            @test Array(anchors[2]) == expected
            @test length(anchors[2]) == layout.lengths[2]
        end
    end

    @testset "`flatten` accepts DEVICE-resident leaves, which is what a GPU run needs" begin
        # The defect the first gate run hit, and it was not the anchor path: `flat = flatten(ps, layout)`
        # in `Nitro` construction runs on `ps` AFTER `to_rarray`, so every multi-leaf group took
        # `_concat_group`'s `vcat(map(vec, ...)...)` branch. `vec` of a `ConcretePJRTArray` is a
        # `ReshapedArray`, which misses Reactant's `vcat` methods, lands in Base's generic `typed_vcat`,
        # and fills the destination with `setindex!` elementwise. Reactant refuses that on a GPU. THE
        # FLAT PARAMETER LAYOUT COULD NOT BE BUILT ON A GPU, so nothing here had ever run on one.
        #
        # BE HONEST ABOUT WHAT THIS TEST CAN DO: the scalar indexing itself is legal on CPU, so no CPU
        # test can reproduce the failure. What it pins is that the device path is EXERCISED and returns
        # the right values as a real device array, so the branch cannot silently rot; the GPU-only half
        # is covered by the gate run.
        rng = Random.MersenneTwister(0)
        ps, _ = Lux.setup(rng, mlp())
        layout = build_layout(TwoGroup(), ps)
        @test all(length(layout.group_rows[gi]) > 1 for gi in 1:2)   # or the vcat branch is never taken

        host = flatten(ps, layout)
        dev = flatten(Reactant.to_rarray(ps), layout)

        @test length.(dev) == layout.lengths
        for gi in 1:2
            @test dev[gi] isa Reactant.AbstractConcreteArray    # a real buffer, never a lazy view
            @test Array(dev[gi]) == host[gi]                    # and elementwise identical to the host
        end
    end

    # ── the scheduled-scalar memo ────────────────────────────────────────────────────────

    @testset "the scheduled-scalar memo" begin
        # BE HONEST ABOUT WHAT THIS SUITE CAN DO. The memo exists because uploading a constant
        # `eta` every optimizer step is waste, and it is SAFE because the optimizer program does not
        # donate the buffer the scalar lives in. Donation is a device-backend property: the same probe
        # that returns `Bool[0, 1, 1, 0, 0, 0, 0]` on CUDA returns an all-false mask on CPU, for pure
        # consumption included. So nothing below is evidence that reuse is safe. What these pin is the
        # BOOKKEEPING: which values are treated as the same value, that a donated scalar is refused, and
        # that `resolve_hp` returns identical numbers with the memo and without. The safety measurement
        # lives in `resolve_hp`'s docstring and has to be re-run on a Reactant bump.

        @testset "`nothing` is the unmemoized path, and is what `setup` and the unit tests take" begin
            a = memo_to_device(nothing, (1, :eta), 1.0f-3)
            b = memo_to_device(nothing, (1, :eta), 1.0f-3)
            @test a isa Reactant.RNumber
            @test Float32(a) == 1.0f-3
            @test a !== b                        # no memo means no reuse, which is the old behaviour
        end

        @testset "an unchanged host value returns the SAME device object" begin
            memo = ScalarMemo()
            a = memo_to_device(memo, (1, :eta), 1.0f-3)
            @test Float32(a) == 1.0f-3
            for _ in 1:10
                @test memo_to_device(memo, (1, :eta), 1.0f-3) === a
            end
            @test length(memo.slots) == 1
        end

        @testset "a changed value uploads again, and the slot is REPLACED rather than grown" begin
            # A scheduled `eta` differs on most steps. A keyed cache would grow without bound and never
            # hit; one slot per key makes that case cost exactly what it costs today.
            memo = ScalarMemo()
            prev = memo_to_device(memo, (1, :eta), 1.0f-3)
            for (i, v) in enumerate(Float32[9.0f-4, 8.0f-4, 7.0f-4, 6.0f-4])
                cur = memo_to_device(memo, (1, :eta), v)
                @test cur !== prev
                @test Float32(cur) == v
                @test length(memo.slots) == 1     # one slot, whatever the schedule does
                prev = cur
            end
            # And returning to an earlier value is a MISS, not a hit: the slot holds the last one only.
            @test memo_to_device(memo, (1, :eta), 1.0f-3) !== prev
        end

        @testset "`===` is the comparison, so no two different numbers share a slot" begin
            memo = ScalarMemo()
            z = memo_to_device(memo, (1, :eta), 0.0f0)
            @test memo_to_device(memo, (1, :eta), -0.0f0) !== z    # `==` would have conflated these
            @test signbit(Float32(memo_to_device(memo, (1, :eta), -0.0f0)))

            # And a `Float64` spelling of one number is a different value, which is why the memo needs
            # no separate element-type key. `resolve_hp` converts to `elt` before it gets here, and the
            # conversion is load-bearing: an `RAdam` defaulting `eta` to `Float64` is what makes a
            # `Float32` run die on step 2 with a message about argument types.
            memo2 = ScalarMemo()
            f32 = memo_to_device(memo2, (1, :eta), 1.0f-3)
            @test memo_to_device(memo2, (1, :eta), 1.0e-3) !== f32
        end

        @testset "a DONATED scalar is refused, which is the whole point of the guard" begin
            # Reuse after donation is the freed-buffer hazard, and it presents as a WRONG LEARNING
            # RATE rather than a crash. `mark_donated!` lets the refusal be tested without a device that
            # actually donates, which is the one piece of this a CPU suite can legitimately pin.
            memo = ScalarMemo()
            a = memo_to_device(memo, (1, :eta), 1.0f-3)
            @test !reuse_refused(a)
            @test memo_to_device(memo, (1, :eta), 1.0f-3) === a

            Reactant.mark_donated!(a)
            @test reuse_refused(a)
            b = memo_to_device(memo, (1, :eta), 1.0f-3)
            @test b !== a                         # same host value, and it still uploads again
            @test Float32(b) == 1.0f-3
            @test !reuse_refused(b)
        end

        @testset "a value with no `donated` field to read is refused too" begin
            # The safe direction is one wasted upload. An IFRT or future concrete type this was never
            # measured against must fall back to uploading rather than guess that reuse is fine.
            @test reuse_refused(1.0f-3)
            @test reuse_refused(nothing)
        end

        @testset "groups and keys do not share a slot" begin
            memo = ScalarMemo()
            e1 = memo_to_device(memo, (1, :eta), 1.0f-3)
            e2 = memo_to_device(memo, (2, :eta), 1.0f-3)
            l1 = memo_to_device(memo, (1, :lambda), 1.0f-3)
            @test e1 !== e2                       # same value, different group, different slot
            @test e1 !== l1                       # same value, different key, different slot
            @test length(memo.slots) == 3
            @test memo_to_device(memo, (1, :eta), 1.0f-3) === e1
            @test memo_to_device(memo, (2, :eta), 1.0f-3) === e2
        end

        @testset "`resolve_hp` returns the same numbers with a memo as without" begin
            rng = Random.MersenneTwister(0)
            ps, _ = Lux.setup(rng, mlp())
            e = TwoGroup()
            layout = build_layout(e, ps)
            memo = ScalarMemo()

            for gi in 1:2
                plain = resolve_hp(e, layout, gi)
                memoed = resolve_hp(e, layout, gi; memo)
                @test Float32(memoed.eta) == Float32(plain.eta)
                @test memoed.lambda === plain.lambda === nothing    # lambda defaults to 0
                @test memoed.anchor === plain.anchor
                @test memoed.no_decay_mask === plain.no_decay_mask
            end

            # The second resolve at the same step reuses, and the two groups' rates really do differ,
            # so a memo collapsing them would be caught here rather than as silently equal learning.
            hp1 = resolve_hp(e, layout, 1; memo)
            hp2 = resolve_hp(e, layout, 2; memo)
            @test hp1.eta === resolve_hp(e, layout, 1; memo).eta
            @test Float32(hp1.eta) != Float32(hp2.eta)
            @test Float32(hp2.eta) / Float32(hp1.eta) ≈ 0.1f0 rtol = 1.0e-6
        end

        @testset "a rebuilt rule carrying a REUSED scalar re-enters the same compiled thunk" begin
            # The schedule-rebuild test's property is what makes per-step rebuild free, and the memo
            # must not break it. A reused `ConcretePJRTNumber` has the type a fresh one has, so the cache key cannot move.
            memo = ScalarMemo()
            x0 = Reactant.to_rarray(zeros(F, 8))
            g0 = Reactant.to_rarray(fill(1.0f-2, 8))

            state = (
                Optimisers.Leaf(
                    Optimisers.Descent(memo_to_device(memo, (1, :eta), 1.0f-3)), nothing, false
                ),
            )
            t0 = typeof(state)
            f = @compile group_step(state, (x0,), (g0,))
            flat = (x0,)
            for _ in 1:3
                state = (
                    Optimisers.Leaf(
                        Optimisers.Descent(memo_to_device(memo, (1, :eta), 1.0f-3)), nothing, false,
                    ),
                )
                @test typeof(state) === t0        # or the thunk guard would reject the next call
                state, flat = f(state, flat, (g0,))
            end
            # `Descent` moves `eta * g` per step: 1e-3 * 1e-2 = 1e-5, three times.
            @test Array(flat[1]) ≈ fill(-3.0f-5, 8) rtol = 1.0f-5
            @test length(memo.slots) == 1         # one upload for the whole loop
        end
    end

end
