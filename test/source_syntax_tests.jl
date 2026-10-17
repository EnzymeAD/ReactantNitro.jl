# The string form of a checkpoint source: what the gate tools pass. It must mean exactly what the
# Julia value means, so the central check is a round trip through `show`.

@testitem "source syntax" begin
    using Test
    using ReactantNitro
    using ReactantNitro: parse_source, parse_restore_best, syntax_value, SOURCE_SYNTAX

    const SELECTORS = [
        Best(3), Best(1), Best(5; among = Last(20)), Best(2; among = Epochs(3:9)),
        Last(10), Last(1), Epochs(3:5), Epochs(2:2:10), Epochs([3, 5, 8]), Every(5),
    ]
    const AVERAGES = [
        Average(Last(10)), Average(Best(5; among = Last(20))), Average(Epochs([3, 5])),
        Average(Every(2); state = :best), Average(Best(3); state = :latest),
    ]

    @testset "every selector type is registered, and every value round-trips through `show`" begin
        exported = [getfield(ReactantNitro, n) for n in names(ReactantNitro)]
        kinds = [nameof(T) for T in exported if T isa Type && T <: Selector && T !== Selector]
        @test !isempty(kinds) && Set(kinds) ⊆ Set(keys(SOURCE_SYNTAX))
        for v in vcat(SELECTORS, AVERAGES)
            @test syntax_value(Meta.parse(repr(v)), "test") == v
        end
        for a in AVERAGES
            @test parse_source("latest => " * repr(a)) == (:latest => a)
            @test parse_source("a1b2c3d4 => " * repr(a)) == ("a1b2c3d4" => a)
        end
    end

    @testset "the forms the tools already took, unchanged" begin
        @test parse_source("latest => best") == (:latest => :best)
        @test parse_source(":latest => :best") == (:latest => :best)
        @test parse_source("all=>best") == (:all => :best)
        @test parse_source("latest => Best(1)") == (:latest => Best(1))
        @test parse_source("runs/x/epoch-0003.jld2") == "runs/x/epoch-0003.jld2"
        @test parse_source("auto"; kw = :resume) === :auto
        @test parse_source("false"; kw = :resume) === false
        @test parse_source("a1b2c3d4 => latest"; kw = :resume) == ("a1b2c3d4" => :latest)
    end

    @testset "nothing outside the whitelist is evaluated or accepted" begin
        for bad in (
                "latest => run(`touch pwned`)", "latest => Average(Base.exit(1))",
                "latest => Average(Last(1.5))", "latest => Average(Last(\"3\"))",
                "latest => Average(Last(n))", "latest => eval(:(1 + 1))",
            )
            @test_throws ErrorException parse_source(bad)
        end
        @test_throws "is not allowed in a source string" parse_source("latest => run(`touch pwned`)")
        @test_throws "does not match any form" parse_source("latest => Average(Last(3); color = :red)")
        @test_throws "not supported yet" parse_source("latest => Average(Last(3); weight = :ema)")
        @test_throws "does not parse" parse_source("latest => Average(Last(3)")
        # The keyword's own refusals, unchanged.
        @test_throws "Average them" parse_source("latest => Last(3)")
        @test_throws "a resume continues one record" parse_source("latest => Average(Last(3))"; kw = :resume)
        @test_throws "across runs" parse_source("all => Average(Last(3))")
    end

    @testset "`restore_best`" begin
        @test parse_restore_best(true) === true
        @test parse_restore_best("false") === false
        @test parse_restore_best("Average(Last(10))") == Average(Last(10))
        @test_throws "or an `Average`" parse_restore_best("Last(10)")
    end
end
