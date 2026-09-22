# `release!`: the source trait's resource hook, and the one place the framework calls it.
#
# What can go wrong is silent: a source released twice closes a server under a second loader, a
# source never released leaves a server running after every export. So every assertion is a count.
@testitem "release" begin
    using Test
    using ReactantNitro
    using ReactantNitro: Done, Failed, NoPrefetch, Terminal, publish_phase, release!
    using NitroTestKit

    mutable struct ReleasableSource
        batches::Vector{Any}
        released::Int
        fail_at::Int                # 0 for never
        raise_on_release::Bool
    end
    ReleasableSource(b; fail_at = 0, raise_on_release = false) =
        ReleasableSource(collect(b), 0, fail_at, raise_on_release)

    Base.length(s::ReleasableSource) = length(s.batches)
    function Base.iterate(s::ReleasableSource, i::Int = 1)
        i > length(s.batches) && return nothing
        i == s.fail_at && error("ReactantNitro test: the source threw at batch $i")
        return (s.batches[i], i + 1)
    end
    function ReactantNitro.release!(s::ReleasableSource)
        s.released += 1
        s.raise_on_release && error("ReactantNitro test: release! threw")
        return nothing
    end

    # The val split sits under `NoPrefetch`, so the unwrap before `release!` is exercised too.
    function mk(; train_kw = (;), val_kw = (;), kw...)
        tr = ReleasableSource(KIT_TRAIN; train_kw...)
        va = ReleasableSource(KIT_VAL; val_kw...)
        n = kit_nitro(KitMLP(); data = (; train = tr, val = NoPrefetch(va)), kw...)
        return n, tr, va
    end

    @testset "the default releases nothing and says so" begin
        @test release!([1, 2]) === nothing
        @test release!((; x = 1)) === nothing
    end

    @testset "`Done` releases every split exactly once, after the monitors" begin
        n, tr, va = mk()
        seen_at_terminal = Ref(-1)
        register_phase_monitor!(
            n, (ph, _step, _epoch, _info) -> begin
                ph isa Terminal && (seen_at_terminal[] = tr.released)
                nothing
            end
        )
        train!(n)
        @test phase(n) isa Done
        @test tr.released == 1
        @test va.released == 1
        # The monitor saw the source still open.
        @test seen_at_terminal[] == 0
        # Neither an explicit call nor a second `Done` releases again.
        release!(n)
        @test tr.released == 1
        train!(n)
        @test tr.released == 1
        @test va.released == 1
    end

    @testset "`Failed` releases too, and the error still surfaces" begin
        n, tr, va = mk(; train_kw = (; fail_at = 2))
        # The single-producer path surfaces the source's error through its `Channel`, wrapped.
        @test_throws Exception train!(n)
        @test phase(n) isa Failed
        @test tr.released == 1
        @test va.released == 1
    end

    @testset "an explicit `release!(nitro)` before any Terminal, then `Done` is a no-op" begin
        n, tr, va = mk()
        release!(n)
        @test tr.released == 1
        @test va.released == 1
        publish_phase(n, Done())
        @test tr.released == 1
        @test va.released == 1
    end

    @testset "a handle-level `publish_phase(Done)` releases, which is the export path" begin
        n, tr, va = mk()
        publish_phase(n, Done())
        @test tr.released == 1
        @test va.released == 1
    end

    @testset "a throwing `release!` is warned about and takes nothing with it" begin
        n, tr, va = mk(; train_kw = (; raise_on_release = true))
        @test_logs (:warn, r"release!") match_mode = :any train!(n)
        @test phase(n) isa Done
        @test tr.released == 1
        @test va.released == 1
    end
end
