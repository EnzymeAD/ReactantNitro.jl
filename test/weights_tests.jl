# Warm starts: `Nitro(e; weights = other)` takes another handle's weights as this run's initial
# ones, and `w0` says what a `:w0` decay anchor means for such a run.
@testitem "weights" begin
    using Test
    using ReactantNitro
    using ReactantNitro: to_host
    using Lux, Random, Statistics

    # The shared toy model from the test kit: decoupled decay anchored at `:w0`, so the anchor
    # choice is observable through the checksum the handle carries, and a precompile workload that
    # already warm-started a handle from it once.
    using NitroTestKit

    mk(e = KitMLP(); kw...) = kit_nitro(e; checkpointer = nothing, kw...)
    caught(f) = try
        f()
        nothing
    catch err
        err
    end

    n1 = train!(mk(; max_epochs = 2))
    fresh = mk()
    # Training moved the weights, so a transfer is observable against a fresh init at the same seed.
    @test to_host(fresh.ps) != to_host(n1.ps)

    @testset "the transferred weights are the initial ones, and nothing else carries over" begin
        n2 = mk(; weights = n1, data = n1.data)
        @test to_host(n2.ps) == to_host(n1.ps)
        @test to_host(n2.st) == to_host(n1.st)
        @test current_epoch(n2) == 0
        @test current_step(n2) == 0
        @test n2.checkpoint_source === nothing
        @test n2.weights_source.experiment === :KitMLP
        @test n2.weights_source.epoch == 2
        @test n2.weights_source.run_dir == n1.run_dir
        # The display says where the weights came from, before and after training.
        @test occursin("from Nitro(KitMLP) epoch 2", sprint(show, MIME"text/plain"(), n2))
        train!(n2)
        @test current_epoch(n2) == 1
        @test occursin("trained here, from Nitro(KitMLP)", sprint(show, MIME"text/plain"(), n2))
        # The source handle is untouched by the transfer.
        @test current_epoch(n1) == 2
        # The provenance an export would stamp names the source too.
        prov = export_provenance(n2)
        @test prov["weights_from"]["epoch"] == 2
        @test prov["weights_from"]["experiment"] == "KitMLP"
        @test !haskey(prov, "checkpoint")
    end

    @testset "`w0` selects the anchor" begin
        d = mk(; weights = n1, data = n1.data)                  # the default: build_model's init
        @test to_host(d.w0) == to_host(fresh.ps)
        @test to_host(d.w0) != to_host(n1.ps)
        w = mk(; weights = n1, data = n1.data, w0 = :weights)   # L2-SP toward the transferred
        @test to_host(w.w0) == to_host(n1.ps)
        tree = to_host(fresh.ps)
        t = mk(; weights = n1, data = n1.data, w0 = tree)       # an explicit tree
        @test to_host(t.w0) == tree
        # `decay_anchor = :w0`, so the anchor checksum follows the choice.
        @test d.anchor_checksum !== nothing
        @test w.anchor_checksum != d.anchor_checksum
        @test t.anchor_checksum == d.anchor_checksum
        train!(w)
        @test current_epoch(w) == 1
    end

    @testset "refusals name the problem" begin
        err = caught(() -> mk(; weights = n1, resume = :auto))
        @test err isa ErrorException
        @test occursin("Two sources", err.msg)
        err = caught(() -> mk(; weights = n1, checkpoint = "runs/x/latest.jld2"))
        @test err isa ErrorException
        @test occursin("checkpoint", err.msg)
        err = caught(() -> mk(; w0 = :weights))
        @test err isa ErrorException
        @test occursin("no `weights = `", err.msg)
        err = caught(() -> mk(; weights = "runs/x/latest.jld2"))
        @test err isa ErrorException
        @test occursin("takes a `Nitro`", err.msg)
        err = caught(() -> mk(; weights = n1, w0 = :nope))
        @test err isa ErrorException
        @test occursin(":build_model", err.msg)
        # A model the weights do not fit: refused with the differing leaves named.
        err = caught(() -> mk(KitMLP(; width = 7); weights = n1, data = n1.data))
        @test err isa ErrorException
        @test occursin("does not match", err.msg)
        @test occursin("layer_1", err.msg)
        @test occursin("(7, 4)", err.msg)
        # And the same for a `w0` tree.
        wide = to_host(mk(KitMLP(; width = 7)).ps)
        err = caught(() -> mk(; weights = n1, data = n1.data, w0 = wide))
        @test err isa ErrorException
        @test occursin("`w0 = ` tree", err.msg)
    end
end
