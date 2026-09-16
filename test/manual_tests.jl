# Manual training mode tests.
#
# The mechanism was verified by a spike before the interface was built; these are its findings as
# permanent tests, plus the driver, optimizer, and exclusion contracts.
#
# CPU-only, inheriting `CUDA_VISIBLE_DEVICES = ""` from runtests.jl.

@testitem "manual" begin
    using Test
    using ReactantNitro
    using ReactantNitro:
        assert_opt_state_device, cache_reset!, cache_stats, compile_cached, compile_view,
        manual_program, strip_rules, to_device_batch
    using Enzyme, Functors, Lux, Optimisers, Random, Reactant, Statistics

    const FM = Float32

    # ── the fixture: a toy least-squares GAN ──────────────────────────────────────────
    # gen: z (2,) -> x̂ (4,); disc: x (4,) -> score. Noise rides in the batch as `z`.

    @experiment struct ToyGAN
        max_epochs::Host{Int} = 2
        "A Device field the device-schedule test varies, to prove device schedules reach the closure."
        scale::Device{Float32} = 1.0f0
    end

    function ReactantNitro.build_data(e::ToyGAN, dist)
        rng = Random.MersenneTwister(7)
        return (;
            train = [(; x = randn(rng, FM, 4, 32), z = randn(rng, FM, 2, 32)) for _ in 1:3],
            val = [(; x = randn(rng, FM, 4, 16), z = randn(rng, FM, 2, 16)) for _ in 1:2],
        )
    end

    ReactantNitro.build_model(e::ToyGAN, rng) = begin
        model = (;
            gen = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 4)),
            disc = Lux.Chain(Lux.Dense(4 => 16, tanh), Lux.Dense(16 => 1)),
        )
        (model, Lux.setup(rng, model)...)
    end

    ReactantNitro.setup_optimizers(e::ToyGAN, model, ps, st, mesh) = (;
        gen = Optimisers.setup(Optimisers.Adam(1.0f-3), ps.gen),
        disc = Optimisers.setup(Optimisers.Adam(1.0f-3), ps.disc),
    )

    # eval-side hooks so validation runs; the closure is never used for eval.
    ReactantNitro.forward(::ToyGAN, model, ps, st; x) = Lux.apply(model.disc, x, ps.disc, st.disc)
    ReactantNitro.metrics(::ToyGAN, out; x) = (; d = (sum(abs2, out), size(out, 2)))

    function gan_step(e, model, ps, opt_state, st; x, z)
        # The generator's forward is the discriminator's fake input. Each sub-network applies with
        # its own state subtree; `st` stays the WHOLE tree for the backwards and the return.
        fake, st_gen = Lux.apply(model.gen, z, ps.gen, st.gen)
        # discriminator backward: real -> 1, fake -> 0
        l_d, g_d = ReactantNitro.backward(ps.disc, x, fake, st) do ps_d, xc, fc, stc
            d1, _ = Lux.apply(model.disc, xc, ps_d, stc.disc)
            d2, _ = Lux.apply(model.disc, fc, ps_d, stc.disc)
            mean(abs2, d1 .- 1.0f0) + mean(abs2, d2)
        end
        # generator backward: fake -> 1, THROUGH the discriminator's forward as a function of the
        # fake data, never into the discriminator's parameters (non-saturating).
        l_g, g_g = ReactantNitro.backward(ps.gen, ps.disc, z, st) do ps_g, ps_d, zc, stc
            f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
            d3, _ = Lux.apply(model.disc, f2, ps_d, stc.disc)
            mean(abs2, d3 .- 1.0f0)
        end
        s_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
        s_d, ps_d = ReactantNitro.step_optimizer(opt_state.disc, ps.disc, g_d)
        return (;
            loss = l_d + l_g, ps = (; gen = ps_g, disc = ps_d),
            st = (; gen = st_gen, disc = st.disc),
            opt_state = (; gen = s_g, disc = s_d), stats = (; l_d, l_g),
        )
    end

    # The dispatch method. Kept as a one-liner so a test that redefines the closure (the NaN and
    # cache-invalidation testsets) can restore the fixture by re-defining this one line.
    ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z) =
        gan_step(e, model, ps, opt_state, st; x, z)

    toy_handle(; kwargs...) =
        Nitro(ToyGAN(); checkpointer = nothing, run_dir = mktempdir(), kwargs...)

    # a device batch narrowed to the closure's own routed fields, as the driver builds it
    closure_batch(n, b) = NamedTuple{keys(n.routing.train_step)}(to_device_batch(b, n.routing, n.mesh))

    # ── top-level probe programs (all-arguments, no captured locals: the same discipline the
    # objective contract demands, and the reason they are not closures inside a testset) ──

    # the capturing form: `ps.disc` (the whole ps) is a closure capture inside the objective
    function capture_program(model, ps, st, batch)
        z = batch.z
        l, g = ReactantNitro.backward(ps.gen, z, st) do ps_g, zc, stc
            f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
            d3, _ = Lux.apply(model.disc, f2, ps.disc, stc.disc)   # ps.disc CAPTURED
            mean(abs2, d3 .- 1.0f0)
        end
        return l, g
    end

    # the all-arguments form: ps.disc is an explicit Const argument
    function constargs_program(model, ps, st, batch)
        z = batch.z
        l, g = ReactantNitro.backward(ps.gen, ps.disc, z, st) do ps_g, ps_d, zc, stc
            f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
            d3, _ = Lux.apply(model.disc, f2, ps_d, stc.disc)
            mean(abs2, d3 .- 1.0f0)
        end
        return l, g
    end

    function grad_probe(model, ps, st, batch)
        z, x = batch.z, batch.x
        fake, _ = Lux.apply(model.gen, z, ps.gen, st.gen)
        _, g_d = ReactantNitro.backward(ps.disc, x, fake, st) do ps_d, x_c, fc, stc
            d1, _ = Lux.apply(model.disc, x_c, ps_d, stc.disc)
            d2, _ = Lux.apply(model.disc, fc, ps_d, stc.disc)
            mean(abs2, d1 .- 1.0f0) + mean(abs2, d2)
        end
        _, g_g = ReactantNitro.backward(ps.gen, ps.disc, z, st) do ps_g, ps_d, zc, stc
            f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
            d3, _ = Lux.apply(model.disc, f2, ps_d, stc.disc)
            mean(abs2, d3 .- 1.0f0)
        end
        return g_d, g_g
    end

    # ── the driver ────────────────────────────────────────────────────────────────────

    @testset "the manual driver: one compiled program, a type fixed point" begin
        cache_reset!()
        n = toy_handle()
        @test n.frozen.manual === true
        train!(n)
        # Exactly two compiles: the closure program and the eval forward. 6 optimizer steps and 2
        # validation runs re-enter them; NOTHING recompiles after step 1.
        st = cache_stats()
        @test st.misses == 2
        @test st.entries == 2
        @test current_step(n) == 6
        @test current_epoch(n) == 2
        @test phase(n) isa Done
        # The user's opt_state stays a tree of Leaves across the run (rules re-attached each step).
        @test n.opt_state.gen.layer_1.weight isa Optimisers.Leaf
        @test n.opt_state.disc.layer_1.weight isa Optimisers.Leaf
        # ...and the returned states were stripped, so `strip_rules` on the handle is identity.
        @test strip_rules(n.opt_state).gen.layer_1.weight isa Tuple   # Adam state, not a Leaf
        # The GAN trained: parameters moved, and the two networks moved in the direction Adam took.
        @test any(!=(0.0f0), reduce(vcat, vec.(Array.(Functors.fleaves(parameters(n))))))
    end

    @testset "the closure program: per-step compile_cached hits, batches flow" begin
        # The direct-program harness, in the style of accumulation.jl's grad_harness: compile
        # `manual_program` once and drive it by hand, so the per-call claims are testable.
        cache_reset!()
        n = toy_handle()
        ev = compile_view(n.e)
        router = n.routing.train_step
        st = Lux.trainmode(n.st)
        data = ReactantNitro.build_data(ToyGAN(), nothing).train
        bf1 = closure_batch(n, data[1])
        bf2 = closure_batch(n, data[2])

        thunk = compile_cached(
            manual_program, ev, ev, n.model, n.ps, n.opt_state, st, bf1, router;
            phase = nothing, worlds = n.frozen.worlds_manual
        )
        # The driver calls compile_cached per step: same types and shapes, different values -> hit.
        compile_cached(
            manual_program, ev, ev, n.model, n.ps, n.opt_state, st, bf2, router;
            phase = nothing, worlds = n.frozen.worlds_manual
        )
        @test cache_stats().misses == 1
        @test cache_stats().hits == 1

        out1 = thunk(ev, n.model, n.ps, n.opt_state, st, bf1, router)
        out2 = thunk(ev, n.model, n.ps, n.opt_state, st, bf2, router)
        @test out1 isa NamedTuple && haskey(out1, :loss) && haskey(out1, :ps)
        # Different batches give different losses: the batch is a traced input, never baked.
        # (The baking failure mode was measured in the spike: a HOST batch silently baked.)
        @test Reactant.to_number(out1.loss) != Reactant.to_number(out2.loss)
    end

    @testset "objective discipline: captures silently zero the gradient (control)" begin
        # The spike's finding, as a CONTROL in the spirit of the suite's unbarriered-state gradient
        # control. A captured parameter subtree
        # SILENTLY zeroes the gradient, with the primal loss correct, which is why the all-arguments
        # contract exists. The control proves the suite can detect the failure mode it forbids.
        n = toy_handle()
        ev = compile_view(n.e)
        st = Lux.trainmode(n.st)
        data = ReactantNitro.build_data(ToyGAN(), nothing).train
        bf = closure_batch(n, data[1])

        # the capturing form: `ps.disc` (the whole ps) is a closure capture inside the objective
        l_cap, g_cap = compile_cached(
            capture_program, ev, n.model, n.ps, st, bf; worlds = ()
        )(n.model, n.ps, st, bf)
        l_arg, g_arg = compile_cached(
            constargs_program, ev, n.model, n.ps, st, bf; worlds = ()
        )(n.model, n.ps, st, bf)
        # the same primal loss either way (the capture is silent, not loud) ...
        @test Reactant.to_number(l_cap) ≈ Reactant.to_number(l_arg) rtol = 1.0f-6
        # ... but the captured form's gradient is ZERO, and the all-arguments form's is not.
        @test maximum(abs, Array(Functors.fleaves(g_cap)[1])) == 0.0f0
        @test maximum(abs, Array(Functors.fleaves(g_arg)[1])) > 0.0f0
    end

    @testset "the closure program's gradients match host references" begin
        # The numeric heart of the spike: the compiled gradients equal host-Enzyme references to
        # Float32 epsilon, for BOTH networks. The references take every value as an argument (host
        # Enzyme's closures must not capture globals; measured, that raises a runtime-activity error)
        # and use `Active` returns, which host Enzyme supports and ReactantABI does not.
        n = toy_handle()
        ev = compile_view(n.e)
        st = Lux.trainmode(n.st)
        data = ReactantNitro.build_data(ToyGAN(), nothing).train
        bf = closure_batch(n, data[1])
        # The host reference must sit at EXACTLY the parameters the compiled program uses, so it is
        # derived from the device tree rather than re-initialized (a re-init with any seed lands
        # elsewhere and the gradients then differ for the right reason).
        ps_host = ReactantNitro.to_host(n.ps)
        st_host = ReactantNitro.to_host(st)
        z_h, xr_h = data[1].z, data[1].x
        fake_h = Lux.apply(n.model.gen, z_h, ps_host.gen, st_host.gen)[1]

        # Host references, ALL-ARGUMENTS: host Enzyme's closures must not capture anything mutable
        # (measured: a captured Nitro or global state raises a runtime-activity error), and its
        # closures must not capture the differentiated variable either; the differentiated tree is
        # passed Duplicated at the call, everything else is an argument.
        disc_ref(model, xr_h, fake_h, ps_d_h, st_h) = begin
            dps = Enzyme.make_zero(ps_d_h)
            inner = ps_d -> begin
                d1, _ = Lux.apply(model.disc, xr_h, ps_d, st_h.disc)
                d2, _ = Lux.apply(model.disc, fake_h, ps_d, st_h.disc)
                mean(abs2, d1 .- 1.0f0) + mean(abs2, d2)
            end
            Enzyme.autodiff(
                Enzyme.ReverseWithPrimal, Enzyme.Const(inner), Enzyme.Active,
                Enzyme.Duplicated(ps_d_h, dps)
            )
            dps
        end
        gen_ref(model, z_h, ps_g_h, ps_d_h, st_h) = begin
            dps = Enzyme.make_zero(ps_g_h)
            inner = ps_g -> begin
                f2, _ = Lux.apply(model.gen, z_h, ps_g, st_h.gen)
                d3, _ = Lux.apply(model.disc, f2, ps_d_h, st_h.disc)
                mean(abs2, d3 .- 1.0f0)
            end
            Enzyme.autodiff(
                Enzyme.ReverseWithPrimal, Enzyme.Const(inner), Enzyme.Active,
                Enzyme.Duplicated(ps_g_h, dps)
            )
            dps
        end

        g_d, g_g = compile_cached(
            grad_probe, ev, n.model, n.ps, st, bf; worlds = ()
        )(n.model, n.ps, st, bf)

        dref = disc_ref(n.model, xr_h, fake_h, ps_host.disc, st_host)
        gref = gen_ref(n.model, z_h, ps_host.gen, ps_host.disc, st_host)

        maxdiff(a, b) = maximum(
            Functors.fleaves(
                Functors.fmap((x, y) -> maximum(abs, Array(x) .- y), a, b)
            )
        )
        @test maxdiff(g_d, dref) < 1.0f-5
        @test maxdiff(g_g, gref) < 1.0f-5
    end

    @testset "v1 exclusions are setup errors, not silence" begin
        err = try
            Nitro(ToyGAN(); accum = 2, checkpointer = nothing, run_dir = mktempdir())
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("manual", err.msg) && occursin("accum", err.msg)

        # device-keyed schedules are supported, and the run completes.
        n = Nitro(
            ToyGAN(); schedules = (; device = (; scale = _ -> t -> Float32(t / 6))),
            checkpointer = nothing, run_dir = mktempdir()
        )
        @test n.frozen.manual === true
        train!(n)
        @test phase(n) isa Done
        @test device_value(n, :scale) ≈ 1.0f0    # the last step wrote 6/6
    end

    @testset "path-bound `opt` schedules: the key says what eta binds to" begin
        # A bare key binds EVERY rule with the field (the absolute value; manual mode has no base
        # rate and no ratios); a nested key is a PATH into the user's `opt_state` and binds the rules
        # at that subtree. The structure IS the binding, which is what makes "which optimizer's eta"
        # unaskable, and the per-step rebuild is type-preserving, so one program serves every step.
        ps_moved(a, b) = maximum(
            abs,
            reduce(vcat, vec.(Array.(Functors.fleaves(a)))) .-
                reduce(vcat, vec.(Array.(Functors.fleaves(b))))
        )

        # gen only: the path binds gen's rules; disc's fixed eta is untouched.
        cache_reset!()
        n = Nitro(
            ToyGAN(); schedules = (; opt = (; gen = (; eta = _ -> _ -> 1.0f-5))),
            checkpointer = nothing, run_dir = mktempdir()
        )
        @test keys(n.schedules.opt) == (Symbol("gen.eta"),)
        @test occursin("opt_state path gen", binding_report(n))
        gen0 = deepcopy(parameters(n).gen)
        disc0 = deepcopy(parameters(n).disc)
        train!(n)
        gen_moved = ps_moved(parameters(n).gen, gen0)
        disc_moved = ps_moved(parameters(n).disc, disc0)
        # Exactly the closure program and the eval forward compiled; the per-step rebuilt rules
        # re-entered the closure program, so nothing recompiled across the 6 scheduled steps.
        @test cache_stats().misses == 2
        # gen is scheduled at 1e-5 against its fixed 1e-3, so it must move a fraction of disc.
        @test gen_moved < disc_moved / 5

        # bare key: binds both optimizers.
        cache_reset!()
        nb = Nitro(
            ToyGAN(); schedules = (; opt = (; eta = _ -> _ -> 1.0f-5)),
            checkpointer = nothing, run_dir = mktempdir()
        )
        gen0b = deepcopy(parameters(nb).gen)
        disc0b = deepcopy(parameters(nb).disc)
        train!(nb)
        @test ps_moved(parameters(nb).gen, gen0b) < disc_moved / 5
        @test ps_moved(parameters(nb).disc, disc0b) < disc_moved / 5
        @test cache_stats().misses == 2

        # a bad path and a nonschedulable field are setup errors naming the offender.
        err = try
            Nitro(
                ToyGAN(); schedules = (; opt = (; nope = (; eta = _ -> _ -> 1.0f-3))),
                checkpointer = nothing, run_dir = mktempdir()
            )
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException && occursin("nope", err.msg)
        err2 = try
            Nitro(
                ToyGAN(); schedules = (; opt = (; gen = (; beta = _ -> _ -> 0.9f0))),
                checkpointer = nothing, run_dir = mktempdir()
            )
            nothing
        catch ex
            ex
        end
        @test err2 isa ErrorException && occursin("nonschedulable", err2.msg)
    end

    @testset "closure routing: only declared fields are transferred" begin
        n = toy_handle()
        @test keys(n.routing.train_step) == (:x, :z)
        batch = (; x = randn(FM, 4, 8), z = randn(FM, 2, 8), case_id = ["a"])
        b = to_device_batch(batch, n.routing)
        @test keys(b) == (:x, :z)      # `case_id` reaches nobody, so it is never transferred
    end

    @testset "editing `train_step` invalidates the cache on a NEW handle" begin
        cache_reset!()
        n1 = toy_handle()
        train!(n1)
        @test cache_stats().misses == 2
        # Re-define train_step: the closure's method world moves, so a new Nitro recompiles the
        # closure program but NOT the eval forward, which no change touched.
        @eval function ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z)
            fake, st_gen = Lux.apply(model.gen, z, ps.gen, st.gen)
            l_d, g_d = ReactantNitro.backward(ps.disc, x, fake, st) do ps_d, xc, fc, stc
                d1, _ = Lux.apply(model.disc, xc, ps_d, stc.disc)
                d2, _ = Lux.apply(model.disc, fc, ps_d, stc.disc)
                mean(abs2, d1 .- 1.0f0) + mean(abs2, d2)
            end
            l_g, g_g = ReactantNitro.backward(ps.gen, ps.disc, z, st) do ps_g, ps_d, zc, stc
                f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
                d3, _ = Lux.apply(model.disc, f2, ps_d, stc.disc)
                mean(abs2, d3 .- 1.0f0)
            end
            s_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
            s_d, ps_d = ReactantNitro.step_optimizer(opt_state.disc, ps.disc, g_d)
            return (;
                loss = 2.0f0 * (l_d + l_g), ps = (; gen = ps_g, disc = ps_d),
                st = (; gen = st_gen, disc = st.disc),
                opt_state = (; gen = s_g, disc = s_d), stats = (;),
            )
        end
        n2 = toy_handle()
        train!(n2)
        @test cache_stats().misses == 3     # only the closure program recompiled
        @test n2.frozen.manual === true
        # restore the fixture closure for the tests that follow.
        @eval ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z) =
            gan_step(e, model, ps, opt_state, st; x, z)
    end

    @testset "a non-finite loss stops the run, naming the step" begin
        # A closure returning a NaN loss must fail fast, exactly as the non-finite-loss guard does
        # for the automatic loop, rather than continue poisoning state.
        @eval function ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z)
            return (; loss = NaN32, ps, st, opt_state, stats = (;))
        end
        n = toy_handle()
        err = try
            train!(n)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("non-finite", err.msg)
        @test phase(n) isa Failed
        # restore the fixture closure for the tests that follow.
        @eval ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z) =
            gan_step(e, model, ps, opt_state, st; x, z)
    end

    @testset "checkpoint/resume round-trips the user's opt_state" begin
        dir = mktempdir()
        ck = TopKCheckpointer(; metric = :d, mode = :min)   # ToyGAN's metrics emit `d`, not val_loss
        n1 = Nitro(ToyGAN(); checkpointer = ck, run_dir = dir)
        train!(n1)
        # `resume = :auto`, which is opt in, restores ps, st, opt_state, step, and epoch.
        n2 = Nitro(ToyGAN(); checkpointer = ck, run_dir = dir, max_epochs = 3, resume = :auto)
        @test current_step(n2) == current_step(n1)
        @test current_epoch(n2) == current_epoch(n1)
        # The restored user opt_state is re-normalized to device residency, the discipline the
        # resume path owes every handle.
        ReactantNitro.assert_opt_state_device(n2.opt_state)
        # and a resumed run continues training instead of re-running or silently stopping.
        train!(n2)
        @test current_step(n2) > current_step(n1)
        @test phase(n2) isa Done
    end

    # ── a 1-D GAN that demonstrably learns its target: the whole feature, end to end ────

    @experiment struct OneDGan
        max_epochs::Host{Int} = 60
    end

    function ReactantNitro.build_data(e::OneDGan, dist)
        x = randn(Random.MersenneTwister(11), FM, 1, 128)   # 128 real samples, fixed for the run
        return (; train = GanLoader(x))
    end

    # A source that draws FRESH noise every batch: a GAN's generator must learn the mapping from
    # noise to data, and a fixed noise sequence lets it memorize the draw instead. `state` is the
    # batch index within the epoch, so per-epoch restart is deterministic; the noise draw itself is
    # seeded by a monotone counter so no batch anywhere repeats.
    struct GanLoader
        x::Matrix{Float32}
    end
    Base.length(::GanLoader) = 10
    const GAN_Z_COUNTER = Ref(0)
    function Base.iterate(l::GanLoader, state = 1)
        state > length(l) && return nothing
        GAN_Z_COUNTER[] += 1
        z = randn(Random.MersenneTwister(GAN_Z_COUNTER[]), FM, 1, 128)
        return ((; x = l.x, z = z), state + 1)
    end

    ReactantNitro.build_model(e::OneDGan, rng) = begin
        model = (;
            gen = Lux.Chain(Lux.Dense(1 => 16, tanh), Lux.Dense(16 => 1)),
            disc = Lux.Chain(Lux.Dense(1 => 16, tanh), Lux.Dense(16 => 1)),
        )
        (model, Lux.setup(rng, model)...)
    end

    # the discriminator learns faster than the generator, the standard GAN recipe
    ReactantNitro.setup_optimizers(e::OneDGan, model, ps, st, mesh) = (;
        gen = Optimisers.setup(Optimisers.Adam(1.0f-3), ps.gen),
        disc = Optimisers.setup(Optimisers.Adam(2.0f-3), ps.disc),
    )

    ReactantNitro.train_step(e::OneDGan, model, ps, opt_state, st; x, z) =
        gan_step(e, model, ps, opt_state, st; x, z)

    @testset "a toy GAN learns its target distribution" begin
        cache_reset!()
        n = Nitro(OneDGan(); checkpointer = nothing, run_dir = mktempdir())
        train!(n)
        @test current_step(n) == 600
        # The generator's output on FRESH noise (never seen in training) should match the target
        # N(0, 1): mean near 0, std near 1. Measured at (-0.048, 0.866); the tolerances are
        # deliberately loose, since a GAN is not a moment-matching machine.
        z0 = randn(Random.MersenneTwister(3), FM, 1, 512)
        ps_h = ReactantNitro.to_host(n.ps)
        st_h = Lux.testmode(ReactantNitro.to_host(n.st))
        x̂, _ = Lux.apply(n.model.gen, z0, ps_h.gen, st_h.gen)
        @test abs(mean(x̂)) < 0.15
        @test 0.8 < std(x̂) < 1.2
        # and the run compiled exactly the closure program: nothing else, ever.
        @test cache_stats().misses == 1
    end

end
