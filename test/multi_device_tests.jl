# Multi-device training: one process, `n_devs = 2`, XLA SPMD over a `Reactant.Sharding.Mesh`.
#
# Two CPU devices need `--xla_force_host_platform_device_count=2` in `XLA_FLAGS` before Reactant
# starts its client, so each item runs its body in a subprocess. The child reports `key=value`
# lines through a file and its stderr stays attached, so a failure names itself in the suite's
# output. A failing part records its error's first line instead of `ok`.
#
# Each item compares `n_devs = 2` against `n_devs = 1` on one seed, because the device count is
# throughput only: the same run, split differently.

@testsetup module MultiDevice

export run_on_two_devices

const PRELUDE = raw"""
    record(k, v) = open(io -> println(io, k, "=", v), ENV["NITRO_PROBE_OUT"], "a")
    firstline(err) = replace(first(split(sprint(showerror, err), '\n')), '=' => ':')

    using ReactantNitro, Lux, Random, Reactant, Statistics, Functors

    record("devices", length(Reactant.devices()))

    # Each device's own copy of a replicated array, read back separately.
    replicas(x) = [
        Array(Reactant.ConcretePJRTArray{eltype(x), ndims(x), 1}(
            (d,), size(x), Reactant.Sharding.NoShardInfo()
        )) for d in x.data
    ]
    agree(x) = (r = replicas(x); length(r) == 2 && r[1] == r[2])
    flat(t) = reduce(vcat, vec.(Array.(Functors.fleaves(t))))
    maxdiff(a, b) = maximum(abs, Float64.(a) .- Float64.(b))
    images(n; seed) = (rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, Float32, 5, 5, 2, 8), y = randn(rng, Float32, 2, 8)) for _ in 1:n])
    vectors(n; seed) = (rng = Random.MersenneTwister(seed);
        [(; x = randn(rng, Float32, 4, 8), y = randn(rng, Float32, 2, 8)) for _ in 1:n])
"""

"""
    run_on_two_devices(probe) -> (success, Dict{String, String})

Run `probe` after the prelude in a child with two forced CPU devices; return its records.
"""
function run_on_two_devices(probe::AbstractString)
    out = joinpath(mktempdir(), "probe.txt")
    flags = strip(get(ENV, "XLA_FLAGS", "") * " --xla_force_host_platform_device_count=2")
    code = PRELUDE * probe
    cmd = addenv(
        `$(Base.julia_cmd()) --project=$(Base.active_project()) --startup-file=no -e $code`,
        "NITRO_PROBE_OUT" => out,
        "XLA_FLAGS" => flags,
        "CUDA_VISIBLE_DEVICES" => "",
    )
    ok = success(pipeline(cmd, stdout = devnull))
    got = Dict{String, String}()
    isfile(out) && for line in eachline(out)
        k, v = split(line, "=", limit = 2)
        got[k] = v
    end
    for (k, v) in got
        k in ("dropout", "batchnorm") && v != "ok" && @info "multi_device: the `$k` part reported" v
    end
    return ok, got
end

end

# A per-element `Dropout` on the input feeding a conv and a `GroupNorm`. `forward` also returns the
# dropout keep-mask and a host `train_metrics` collects it per micro-batch, so the masks themselves
# are compared.
@testitem "multi_device dropout" setup = [MultiDevice] timeout = 1200 begin
    using Test

    ok, got = run_on_two_devices(
        raw"""
        @experiment struct DropExp
            max_epochs::Int = 1
        end
        ReactantNitro.build_model(::DropExp, rng) = (
            m = Chain(;
                drop = Dropout(0.5f0),
                body = Chain(
                    Conv((3, 3), 2 => 4, tanh; pad = 1), GroupNorm(4, 2),
                    GlobalMeanPool(), FlattenLayer(), Dense(4 => 2)
                ),
            );
            (m, Lux.setup(rng, m)...)
        )
        function ReactantNitro.forward(::DropExp, model, ps, st; x)
            h, s1 = Lux.apply(model.layers.drop, x, ps.drop, st.drop)
            y, s2 = Lux.apply(model.layers.body, h, ps.body, st.body)
            return (; y, keep = h .!= 0), (; drop = s1, body = s2)
        end
        ReactantNitro.loss(::DropExp, out; y) = mean(abs2, out.y .- y)
        ReactantNitro.metrics_residency(::DropExp, ::Symbol) = :host
        const MASKS = Ref(Any[])
        ReactantNitro.train_metrics(::DropExp, out) =
            (push!(MASKS[], copy(out.keep)); (; keep = mean(out.keep)))

        const DATA = (; train = images(4; seed = 3), val = images(2; seed = 4))

        # Four micro-batches per epoch at `accum = 2`, so two optimizer steps. Only the resume
        # registers a monitor, so the state round trip is isolated from `fire_monitors`.
        function run!(; n_devs, epochs = 2, dir = mktempdir(), resume = false, philox = false,
                monitor = nothing)
            MASKS[] = Any[]
            n = Nitro(DropExp(); run_dir = dir, n_devs, accum = 2, max_epochs = epochs, seed = 5,
                resume, data = DATA)
            philox && (n.st = to_philox(n.st))
            monitor === nothing ||
                register_phase_monitor!(n, (ph, s, ep, info) -> (monitor[] += 1; nothing))
            train!(n)
            return n, MASKS[]
        end
        distinct(ms) = length(unique(ms)) == length(ms)
        to_philox(r::Reactant.ReactantRNG) = Reactant.ReactantRNG(r.seed, "PHILOX")
        to_philox(x::NamedTuple) = map(to_philox, x)
        to_philox(x) = x

        try
            d1, md1 = run!(; n_devs = 1)
            record("d1_algorithm", d1.st.drop.rng.algorithm)
            record("d1_masks", length(md1))
            record("d1_distinct", distinct(md1))

            p1, mp1 = run!(; n_devs = 1, philox = true)
            m2, mm2 = run!(; n_devs = 2)
            rng = m2.st.drop.rng
            record("m2_algorithm", rng.algorithm)
            record("m2_rng_devices", length(rng.seed.data))
            record("m2_rng_replicas_agree", agree(rng.seed))
            record("m2_masks", length(mm2))
            record("m2_distinct", distinct(mm2))
            # One global mask, not one per-device mask repeated on each shard.
            record("m2_shards_differ", all(m -> m[:, :, :, 1:4] != m[:, :, :, 5:8], mm2))
            record("m2_masks_equal_p1", mm2 == mp1)
            record("m2_seed_equal_p1", Array(rng.seed) == Array(p1.st.drop.rng.seed))
            record("m2_loss_rdiff", abs(m2.history[end].loss - p1.history[end].loss) /
                abs(p1.history[end].loss))
            record("m2_ps_maxdiff", maxdiff(flat(parameters(m2)), flat(parameters(p1))))
            record("m2_ps_replicas_agree", all(agree, Functors.fleaves(parameters(m2))))

            # One epoch, a checkpoint, then the second epoch resumed onto the mesh: the same masks
            # and weights as the uninterrupted run.
            dir = mktempdir()
            r1, mr1 = run!(; n_devs = 2, epochs = 1, dir)
            misses = ReactantNitro.cache_stats().misses
            calls = Ref(0)
            r2, mr2 = run!(; n_devs = 2, epochs = 2, dir, resume = :auto, monitor = calls)
            record("resume_monitor_calls", calls[])
            record("resume_recompiles", ReactantNitro.cache_stats().misses - misses)
            record("resume_algorithm", r2.st.drop.rng.algorithm)
            record("resume_steps", r2.step)
            record("resume_masks_equal", vcat(mr1, mr2) == mm2)
            record("resume_ps_maxdiff", maxdiff(flat(parameters(r2)), flat(parameters(m2))))
            record("dropout", "ok")
        catch err
            record("dropout", firstline(err))
        end
        """
    )
    num(k) = parse(Float64, get(got, k, "NaN"))
    @test ok
    @test parse(Int, get(got, "devices", "0")) >= 2
    @test get(got, "dropout", "<missing>") == "ok"

    @testset "the mask advances every micro-batch on one device" begin
        @test get(got, "d1_algorithm", "") == "DEFAULT"
        @test get(got, "d1_masks", "") == "8"
        @test get(got, "d1_distinct", "") == "true"
    end

    @testset "on the mesh: PHILOX, replicated, advancing, and one global mask" begin
        @test get(got, "m2_algorithm", "") == "PHILOX"
        @test get(got, "m2_rng_devices", "") == "2"
        @test get(got, "m2_rng_replicas_agree", "") == "true"
        @test get(got, "m2_masks", "") == "8"
        @test get(got, "m2_distinct", "") == "true"
        @test get(got, "m2_shards_differ", "") == "true"
        # Partitioned PHILOX is single-device PHILOX: the same masks, so the same run.
        @test get(got, "m2_masks_equal_p1", "") == "true"
        @test get(got, "m2_seed_equal_p1", "") == "true"
        @test num("m2_loss_rdiff") < 1.0e-5
        @test num("m2_ps_maxdiff") < 1.0e-5
        @test get(got, "m2_ps_replicas_agree", "") == "true"
    end

    @testset "a checkpoint resume onto the mesh continues the mask stream" begin
        @test parse(Int, get(got, "resume_monitor_calls", "0")) > 0
        @test get(got, "resume_recompiles", "") == "0"
        @test get(got, "resume_algorithm", "") == "PHILOX"
        @test get(got, "resume_steps", "") == "4"
        @test get(got, "resume_masks_equal", "") == "true"
        @test num("resume_ps_maxdiff") < 1.0e-6
    end
end

# The batch is sharded and BatchNorm reduces over it, so `n_devs = 2` must compute the GLOBAL batch
# statistics, in training and in the running statistics it returns.
@testitem "multi_device batchnorm" setup = [MultiDevice] timeout = 1200 begin
    using Test

    ok, got = run_on_two_devices(
        raw"""
        @experiment struct BNExp
            max_epochs::Int = 1
        end
        ReactantNitro.build_model(::BNExp, rng) = (
            m = Chain(Dense(4 => 8), BatchNorm(8, tanh), Dense(8 => 2));
            (m, Lux.setup(rng, m)...)
        )
        ReactantNitro.forward(::BNExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
        ReactantNitro.loss(::BNExp, ŷ; y) = mean(abs2, ŷ .- y)
        ReactantNitro.metrics_residency(::BNExp, ::Symbol) = :host
        const OUTS = Ref(Any[])
        ReactantNitro.train_metrics(::BNExp, ŷ) = (push!(OUTS[], copy(ŷ)); (; m = mean(ŷ)))

        const DATA = (; train = vectors(4; seed = 3), val = vectors(2; seed = 4))
        const PROBE = vectors(1; seed = 6)[1]

        function run!(; n_devs, monitor = Ref(0))
            OUTS[] = Any[]
            n = Nitro(BNExp(); run_dir = mktempdir(), n_devs, accum = 2, seed = 5, data = DATA)
            register_phase_monitor!(n, (ph, s, ep, info) -> (monitor[] += 1; nothing))
            train!(n)
            return n, OUTS[]
        end

        try
            s1, o1 = run!(; n_devs = 1)
            calls = Ref(0)
            s2, o2 = run!(; n_devs = 2, monitor = calls)
            bn1, bn2 = s1.st.layer_2, s2.st.layer_2
            record("bn_monitor_calls", calls[])
            record("bn_steps", s2.step)
            record("bn_train_out_maxdiff", maximum(maxdiff.(o1, o2)))
            record("bn_loss_rdiff", abs(s1.history[end].loss - s2.history[end].loss) /
                abs(s1.history[end].loss))
            record("bn_grad_maxdiff", maxdiff(flat(s1.g_accum), flat(s2.g_accum)))
            record("bn_ps_maxdiff", maxdiff(flat(parameters(s1)), flat(parameters(s2))))
            record("bn_mean_maxdiff", maxdiff(Array(bn1.running_mean), Array(bn2.running_mean)))
            record("bn_var_maxdiff", maxdiff(Array(bn1.running_var), Array(bn2.running_var)))
            record("bn_stats_moved", any(!iszero, Array(bn2.running_mean)))
            record("bn_stats_replicas_agree", agree(bn2.running_mean) && agree(bn2.running_var))
            record("bn_val_rdiff", abs(s1.last_metrics.val_loss - s2.last_metrics.val_loss) /
                abs(s1.last_metrics.val_loss))
            record("bn_eval_out_maxdiff", maxdiff(predict(s1, PROBE), predict(s2, PROBE)))
            record("batchnorm", "ok")
        catch err
            record("batchnorm", firstline(err))
        end
        """
    )
    num(k) = parse(Float64, get(got, k, "NaN"))
    @test ok
    @test parse(Int, get(got, "devices", "0")) >= 2
    @test get(got, "batchnorm", "<missing>") == "ok"
    @test parse(Int, get(got, "bn_monitor_calls", "0")) > 0
    @test get(got, "bn_steps", "") == "2"
    @test num("bn_train_out_maxdiff") < 1.0e-5
    @test num("bn_loss_rdiff") < 1.0e-5
    @test num("bn_grad_maxdiff") < 1.0e-5
    @test num("bn_ps_maxdiff") < 1.0e-5
    @test get(got, "bn_stats_moved", "") == "true"
    @test num("bn_mean_maxdiff") < 1.0e-5
    @test num("bn_var_maxdiff") < 1.0e-5
    @test get(got, "bn_stats_replicas_agree", "") == "true"
    @test num("bn_val_rdiff") < 1.0e-5
    @test num("bn_eval_out_maxdiff") < 1.0e-5
end
