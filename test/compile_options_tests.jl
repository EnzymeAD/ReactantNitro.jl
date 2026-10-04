# The `compile_options` keyword of `Nitro`: validated at construction, forwarded to every program's
# compile, and part of the compile-cache key.

@testitem "compile options" begin
    using Test
    using ReactantNitro
    using ReactantNitro: cache_reset!, cache_stats, check_compile_options
    using Functors, Lux, Optimisers, Random, Reactant, Statistics

    const FA = Float32

    @experiment struct OptsExp
        max_epochs::Host{Int} = 1
    end
    ReactantNitro.build_model(::OptsExp, rng) =
        (Lux.Dense(4 => 4), Lux.setup(rng, Lux.Dense(4 => 4))...)
    ReactantNitro.forward(::OptsExp, model, ps, st; x) = Lux.apply(model, x, ps, st)
    ReactantNitro.loss(::OptsExp, ŷ; y) = mean(abs2, ŷ .- y)
    ReactantNitro.learning_rate(::OptsExp) = 1.0f-2
    ReactantNitro.optimizer(::OptsExp) = Optimisers.Adam

    const RNG = Random.MersenneTwister(3)
    batches() = [(; x = randn(RNG, FA, 4, 8), y = randn(RNG, FA, 4, 8)) for _ in 1:4]
    const DATA = batches()
    nitro(; kw...) = Nitro(
        OptsExp(); data = (; train = DATA), checkpointer = nothing,
        run_dir = mktempdir(), kw...
    )
    params_of(n) = reduce(vcat, vec.(Array.(Functors.fleaves(parameters(n)))))

    @testset "validation at construction" begin
        @test check_compile_options((;)) == (;)
        @test check_compile_options((; cudnn_hlo_optimize = true)) == (; cudnn_hlo_optimize = true)
        co = Reactant.CompileOptions(; transpose_propagate = :down)
        @test check_compile_options(co) == (; compile_options = co)

        err = try
            check_compile_options((; not_an_option = 1))
        catch ex
            ex
        end
        @test err isa ErrorException && occursin("not_an_option", err.msg)
        # A real `CompileOptions` field that `compile` does not take as a keyword: refused here,
        # where it would otherwise fail inside the first compile, and accepted as a struct.
        err = try
            check_compile_options((; disable_slice_to_batch_passes = false))
        catch ex
            ex
        end
        @test err isa ErrorException && occursin("CompileOptions", err.msg)
        @test check_compile_options(Reactant.CompileOptions(; disable_slice_to_batch_passes = false)) isa NamedTuple
        @test_throws ErrorException check_compile_options(1)

        # Donation is what keeps the in-place accumulator and parameters from leaking.
        for bad in (
                (; donated_args = :none),
                Reactant.CompileOptions(; donated_args = :none),
                (; compile_options = Reactant.CompileOptions(; donated_args = :none)),
            )
            err = try
                check_compile_options(bad)
            catch ex
                ex
            end
            @test err isa ErrorException && occursin("donated_args", err.msg)
        end
        # Refused before the setup sequence runs.
        @test_throws ErrorException nitro(; compile_options = (; donated_args = :none))
    end

    @testset "the options reach the compiler" begin
        # `assert_nonallocating` makes Reactant refuse a program that returns a buffer it was not
        # donated; the gradient program returns a fresh loss, so the compile must fail, which it
        # can only do if the option arrived.
        err = try
            train!(nitro(; compile_options = (; assert_nonallocating = true)))
        catch ex
            ex
        end
        @test err isa Exception
        @test occursin("preserved_args", sprint(showerror, err))
    end

    @testset "a CompileOptions value carries the pass switches through a training run" begin
        co = Reactant.CompileOptions(;
            disable_slice_to_batch_passes = false,
            disable_structured_tensors_detection_passes = false
        )
        @test train!(nitro(; compile_options = co)) isa Nitro
    end

    @testset "part of the compile-cache key, and numerically neutral here" begin
        cache_reset!()
        a = params_of(train!(nitro()))
        base = cache_stats().misses
        @test base == 2                                        # the gradient and optimizer programs

        b = params_of(train!(nitro(; compile_options = (; transpose_propagate = :down))))
        @test cache_stats().misses == 2base                    # its own programs, not the default ones

        c = params_of(train!(nitro()))
        @test cache_stats().misses == 2base                    # the default programs are still served
        @test a == c
        @test a ≈ b rtol = 1.0e-5
    end
end
