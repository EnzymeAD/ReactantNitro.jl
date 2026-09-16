# Data routing tests: one experiment per routing shape, plus routing coverage.
#
# The training-requirements test is here (both halves). The eval pad-and-slice tests live in the
# eval test file.

@testitem "data" begin
    using Test
    using ReactantNitro
    using ReactantNitro: KWARG_SINK, Router, batch_size_of, call_hook, check_batch_schema,
        check_data_source, check_epoch_length, check_train_batch_shape,
        check_train_divisibility, declared, has_sink, resolve_routing, route_keys,
        routed_fields, validate_batch

    # ── experiments, one per routing shape the routing rules cover ──────────────────────

    @experiment struct ThreeHook
        n::Int = 1
    end
    ReactantNitro.forward(::ThreeHook, model, ps, st; img) = (img, st)
    ReactantNitro.loss(::ThreeHook, outputs; lab, weight) = sum(outputs)
    ReactantNitro.metrics(::ThreeHook, outputs; lab) = (; acc = (1.0f0, 1))

    @experiment struct OptionalKw
        n::Int = 1
    end
    ReactantNitro.forward(::OptionalKw, model, ps, st; img, mask = nothing) =
        (mask === nothing ? img : img .* mask, st)
    ReactantNitro.loss(::OptionalKw, outputs; lab) = sum(outputs)

    @experiment struct RequiredKw
        n::Int = 1
    end
    ReactantNitro.forward(::RequiredKw, model, ps, st; img, mask) = (img .* mask, st)
    ReactantNitro.loss(::RequiredKw, outputs; lab) = sum(outputs)

    @experiment struct SinkHook
        n::Int = 1
    end
    ReactantNitro.forward(::SinkHook, model, ps, st; kwargs...) = (first(values(kwargs)), st)
    ReactantNitro.loss(::SinkHook, outputs; kwargs...) = sum(outputs)

    const BATCH = (; img = randn(Float32, 4, 8), lab = rand(1:3, 8), weight = randn(Float32, 8))

    # ── routing ─────────────────────────────────────────────────────────────────────────

    @testset "routing resolves three hooks against a three-field batch" begin
        e = compile_view(ThreeHook())
        r = resolve_routing(e, BATCH)

        @test keys(r.forward) == (:img,)
        @test keys(r.loss) == (:lab, :weight)
        @test keys(r.metrics) == (:lab,)
        @test r.train_metrics === nothing          # no method: the `(;)` default applies

        # The Router selects by TYPE PARAMETER, so the selection folds at trace time.
        @test r.forward(BATCH) === (; img = BATCH.img)
        @test r.loss(BATCH) === (; lab = BATCH.lab, weight = BATCH.weight)
        @test Router{(:img,)}() === Router{(:img,)}()

        # Rule 1: a field no hook declares reaches nobody, and that is NOT an error.
        withid = merge(BATCH, (; case_id = ["a", "b"]))
        r2 = resolve_routing(e, withid)
        @test :case_id ∉ routed_fields(r2)
        @test routed_fields(r2) == (:img, :lab, :weight)
    end

    @testset "`declared` reads the method's keywords, and cannot see defaults" begin
        # The single fact rule 2 is built on: a REQUIRED keyword and a DEFAULTED one are
        # indistinguishable here, because kwarg_decl returns names only.
        req = declared(forward, Tuple{RequiredKw, Any, Any, Any})
        opt = declared(forward, Tuple{OptionalKw, Any, Any, Any})
        @test req == [:img, :mask]
        @test opt == [:img, :mask]
        @test req == opt

        # A sink is detected by the literal trailing symbol.
        sink = declared(forward, Tuple{SinkHook, Any, Any, Any})
        @test last(sink) === KWARG_SINK
        @test KWARG_SINK === Symbol("kwargs...")
        @test has_sink(sink)
        @test !has_sink(req)

        # A hook with no method for this type reports `nothing` rather than raising: `metrics` and
        # `train_metrics` legitimately have none.
        @test declared(metrics, Tuple{OptionalKw, Any}) === nothing
        @test declared(train_metrics, Tuple{ThreeHook, Any}) === nothing
    end

    @testset "absent keywords are Julia's decision, not the framework's" begin
        b = (; img = randn(Float32, 4, 8), lab = rand(1:3, 8))   # no `mask`

        @testset "a DEFAULTED field absent from the batch runs and takes the default" begin
            e = compile_view(OptionalKw())
            r = resolve_routing(e, b)
            @test keys(r.forward) == (:img,)          # the intersection, not [:img, :mask]
            out, _ = call_hook(forward, :forward, r.forward, b, e, nothing, nothing, nothing)
            @test out == b.img                        # mask defaulted to nothing
        end

        @testset "a REQUIRED field absent from the batch raises, naming hook, keyword, and fields" begin
            e = compile_view(RequiredKw())
            r = resolve_routing(e, b)
            @test keys(r.forward) == (:img,)          # routing is identical: it CANNOT tell them apart
            err = try
                call_hook(forward, :forward, r.forward, b, e, nothing, nothing, nothing)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("forward", err.msg)
            @test occursin("mask", err.msg)
            @test occursin("img", err.msg)            # the batch's fields
            @test occursin("optional batch field", err.msg)
        end

        # This is the pair the original acceptance criterion got wrong: it asked for a
        # declared-but-absent keyword to "raise at setup", which the routing rules say the
        # framework CANNOT do, and the epoch-schema check agrees. Amended after the first draft.
    end

    @testset "a `kwargs...` hook receives the whole batch, unchecked" begin
        e = compile_view(SinkHook())
        r = resolve_routing(e, BATCH)
        @test keys(r.forward) == keys(BATCH)
        @test keys(r.loss) == keys(BATCH)
        @test r.forward(BATCH) === BATCH
    end

    @testset "a batch whose field names change mid-epoch errors" begin
        expected = keys(BATCH)
        @test check_batch_schema(BATCH, expected, :train, 1) === nothing

        err = try
            check_batch_schema((; img = BATCH.img, lab = BATCH.lab), expected, :train, 7)
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("batch 7", err.msg)
        @test occursin("weight", err.msg)            # names what went missing
        @test occursin("train", err.msg)

        err2 = try
            check_batch_schema(merge(BATCH, (; extra = randn(Float32, 8))), expected, :val, 2)
        catch ex
            ex
        end
        @test occursin("extra", err2.msg)            # and what appeared
    end

    @testset "batch validation: abstract element types and host scalars" begin
        e = compile_view(ThreeHook())
        r = resolve_routing(e, BATCH)
        @test validate_batch(BATCH, r) === nothing

        bad = merge(BATCH, (; img = NamedTuple[(; a = 1), (; a = 2)]))
        rbad = resolve_routing(e, bad)
        err = try
            validate_batch(bad, rbad)
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("img", err.msg)
        @test occursin("abstract element type", err.msg)
        @test occursin("NamedTuple", err.msg)

        scalar = merge(BATCH, (; weight = 3))
        err2 = try
            validate_batch(scalar, resolve_routing(e, scalar))
        catch ex
            ex
        end
        @test err2 isa ErrorException
        @test occursin("weight", err2.msg)
        @test occursin("not an array", err2.msg)

        # Only ROUTED fields are validated, which is what makes rule 1's `case_id` example legal: a
        # Vector{String} of case identifiers is neither device-able nor ever looked at.
        withid = merge(BATCH, (; case_id = ["a", "b"]))
        @test validate_batch(withid, resolve_routing(e, withid)) === nothing
    end

    # ── batch size ──────────────────────────────────────────────────────────────────────

    @testset "batch size is inferred from the first batch, never read from config" begin
        e = compile_view(ThreeHook())
        r = resolve_routing(e, BATCH)
        @test batch_size_of(BATCH, r) == 8

        # A disagreement on the last dimension is an error here rather than a wrong slice later.
        ragged = merge(BATCH, (; lab = rand(1:3, 7)))
        err = try
            batch_size_of(ragged, resolve_routing(e, ragged))
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("ambiguous", err.msg)

        # An unrouted field of a different width is fine, because it reaches nobody.
        withid = merge(BATCH, (; case_id = ["a"]))
        @test batch_size_of(withid, resolve_routing(e, withid)) == 8
    end

    # ── the data-source contract ─────────────────────────────────────────────────────────

    @testset "`length` is required, and counts batches" begin
        batches = [BATCH for _ in 1:4]
        @test check_data_source(batches, :train) == 4

        struct NoLength end
        Base.iterate(::NoLength, s = 1) = s > 3 ? nothing : (BATCH, s + 1)
        err = try
            check_data_source(NoLength(), :train)
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("length", err.msg) && occursin("train", err.msg)
        @test occursin("BATCHES", err.msg)
    end

    @testset "a non-restartable source is caught at the end of the first epoch" begin
        @test check_epoch_length(4, 4, :train) === nothing

        # Setup drew one batch for the schema and discarded it, so a one-shot source is short by
        # EXACTLY one. That signature is recognizable, and the error says so.
        err = try
            check_epoch_length(3, 4, :train)
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("NOT RESTARTABLE", err.msg)
        @test occursin("Channel", err.msg)

        # A variable-length loader is fine with constant hyperparameters; only the combination with a
        # horizon-dependent schedule is rejected.
        @test check_epoch_length(2, 4, :train; horizon_dependent = false) === nothing
        err2 = try
            check_epoch_length(2, 4, :train; horizon_dependent = true)
        catch ex
            ex
        end
        @test occursin("horizon", err2.msg)
    end

    # ── two training requirements, checked twice ─────────────────────────────────────────

    @testset "two training requirements, checked in the two places they can be" begin
        @testset "first half, at setup: length(train) % accum == 0" begin
            @test check_train_divisibility(46, 1) === nothing
            @test check_train_divisibility(46, 2) === nothing
            @test check_train_divisibility(45, 3) === nothing

            err = try
                check_train_divisibility(45, 2, :train)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("45", err.msg)                 # the batch count
            @test occursin("accum = 2", err.msg)
            @test occursin("train", err.msg)              # the split
            @test occursin("44", err.msg) && occursin("46", err.msg)   # the nearest workable counts

            err2 = try
                check_train_divisibility(10, 4)
            catch ex
                ex
            end
            @test occursin("8", err2.msg) && occursin("12", err2.msg)
        end

        @testset "second half, at that batch: size(leaf)[end] == batch_size" begin
            e = compile_view(ThreeHook())
            r = resolve_routing(e, BATCH)
            @test check_train_batch_shape(BATCH, 8, :train, 1, r) === nothing

            short = (; img = randn(Float32, 4, 5), lab = rand(1:3, 5), weight = randn(Float32, 5))
            err = try
                check_train_batch_shape(short, 8, :train, 46, r)
            catch ex
                ex
            end
            @test err isa ErrorException
            @test occursin("batch 46", err.msg)           # the batch index
            @test occursin("train", err.msg)              # the split
            @test occursin("5", err.msg) && occursin("8", err.msg)     # the two sizes
            @test occursin("drop_last", err.msg)          # what to do about it

            # It is checked in the loop rather than at setup for the reason the data-source contract
            # gives: the framework can see a batch it was handed and cannot see samples a loader
            # dropped before handing anything over. The setup check above sees batch COUNTS; only
            # this one sees shapes.
        end
    end

end
