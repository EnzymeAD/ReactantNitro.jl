# Hook map tests.
#
# PROTOTYPE. The acceptance criterion: "an experiment with NO hook methods at all trains from a
# map of hook values, two maps that differ in one hook compile separately and share the rest, and
# a hook that captures is refused."
#
# THE CAPTURE GUARD IS THE POINT. Everything else here would be true of any plausible
# implementation; the guard is what keeps the map from being a hole in the compile-cache key, and
# it rests on a Base internal, so it gets the same treatment as `primary_world_available`: a test
# that fails loudly if the predicate stops meaning what it is used for.

@testitem "hooks" begin
    using Test
    using ReactantNitro
    using ReactantNitro: @hooks, check_hooks, check_hook_grouping, hook_fn, hook_fns,
        hook_predicate_available, cache_stats
    using Lux, MLUtils, Random, Reactant

    # ── The Base internal the guard rests on ────────────────────────────────────────
    # Keyword lowering wraps a hook in a function that HOLDS the inner body function, so the naive
    # `isempty(fieldnames(...))` check refuses every routed hook. `issingletontype` is what
    # separates that artifact from a real capture, and this asserts it still does.
    @test hook_predicate_available()

    # ── @hooks builds values, not methods ───────────────────────────────────────────
    # Counted around the macro, not `isdefined`: `forward` is EXPORTED, so it is defined in any
    # module that does `using ReactantNitro` and `isdefined` would be true however the macro
    # behaved. What must hold is that no METHOD was added.
    n_fwd = length(methods(ReactantNitro.forward))
    n_loss = length(methods(ReactantNitro.loss))
    base = @hooks begin
        forward(e, model, ps, st; x) = Lux.apply(model, x, ps, st)
        loss(e, pred; y) = sum(abs2, pred .- y) / size(y, 2)
    end
    @test base isa NamedTuple
    @test keys(base) == (:forward, :loss)
    @test all(f -> Base.issingletontype(typeof(f)), base)
    # Nothing was added to a method table: the names are local to the macro's `let`.
    @test length(methods(ReactantNitro.forward)) == n_fwd
    @test length(methods(ReactantNitro.loss)) == n_loss
    @test base.forward !== ReactantNitro.forward

    # ── merge: a second map that coexists with the first ────────────────────────────
    tweaked = merge(
        base, @hooks begin
            loss(e, pred; y) = sum(abs, pred .- y) / size(y, 2)
        end
    )
    @test typeof(base.loss) !== typeof(tweaked.loss)   # different program, different key
    @test typeof(base.forward) === typeof(tweaked.forward)  # shared, so its program is reused
    @test keys(tweaked) == (:forward, :loss)

    # ── check_hooks ─────────────────────────────────────────────────────────────────
    @test check_hooks(base) === base
    @test_throws ErrorException check_hooks((; loss = 1))          # not a function
    @test_throws ErrorException check_hooks("not a namedtuple")
    w = 2.0f0
    v = Float32[1.0, 2.0]
    @test_throws ErrorException check_hooks(
        (;
            loss = let w = w
                (e, pred; y) -> sum(abs2, pred .- y) * w
            end,
        )
    )
    @test_throws ErrorException check_hooks(
        (;
            loss = let v = v
                (e, pred; y) -> sum(abs2, pred .- v)
            end,
        )
    )
    # The message must name the capture, since "put it on the experiment" is only actionable if
    # the user is told what "it" is.
    msg = try
        check_hooks(
            (;
                loss = let w = w
                    (e, pred; y) -> w
                end,
            )
        )
    catch err
        sprint(showerror, err)
    end
    @test occursin("CAPTURES", msg) && occursin("Float32", msg) && occursin("loss", msg)

    # ── the routing guard ───────────────────────────────────────────────────────────
    # A routing predating the map, or none at all, must resolve to the methods.
    @test hook_fns(nothing) === (;)
    @test hook_fns((; forward = nothing, loss = nothing)) === (;)
    @test hook_fns((; fns = base)) === base
    @test hook_fn(base, :loss, ReactantNitro.loss) === base.loss
    @test hook_fn((;), :loss, ReactantNitro.loss) === ReactantNitro.loss

    # ── End to end, with NO hook method defined for this experiment ─────────────────
    @experiment struct MapOnly
        width::GraphConst{Int} = 8
        max_epochs::Int = 1
    end
    @test !hasmethod(ReactantNitro.forward, Tuple{MapOnly, Any, Any, Any})
    @test !hasmethod(ReactantNitro.loss, Tuple{MapOnly, Any})

    # Two groups, because one is now refused: the host hooks are free to edit and the traced
    # ones are not, and a group is only as cheap as its most expensive member.
    host_h = @hooks begin
        build_model(e, rng) = begin
            m = Chain(Dense(4 => e.width, relu), Dense(e.width => 1))
            (m, Lux.setup(rng, m)...)
        end
        build_data(e, dist) = begin
            part(n) = (; x = randn(Float32, 4, n), y = randn(Float32, 1, n))
            (;
                train = DataLoader(part(64); batchsize = 16, shuffle = false, partial = false),
                val = DataLoader(part(16); batchsize = 16),
            )
        end
    end
    traced_h = @hooks begin
        forward(e, model, ps, st; x) = Lux.apply(model, x, ps, st)
        loss(e, pred; y) = sum(abs2, pred .- y) / size(y, 2)
    end
    full = merge(host_h, traced_h)

    dir = mktempdir()
    n = Nitro(MapOnly(); hooks = full, run_dir = dir, checkpointer = nothing, logger = nothing)
    train!(n)
    @test n.epoch == 1
    @test n.last_metrics.val_loss isa Real
    # Routing resolved from the MAP's keyword declarations, not from any method.
    @test keys(n.routing.forward) == (:x,)
    @test keys(n.routing.loss) == (:y,)
    @test n.routing.fns.forward === traced_h.forward

    # Rebuilding from the same map hits the cache: identity is the type, and the type is stable.
    before = cache_stats().misses
    n2 = Nitro(MapOnly(); hooks = full, run_dir = dir, checkpointer = nothing, logger = nothing)
    train!(n2)
    @test cache_stats().misses == before

    # ── The grouping rules ──────────────────────────────────────────────────────────
    # A group is the unit of invalidation, so mixing a free hook with an expensive one makes the
    # free one expensive. Measured: `build_data` edited alone costs 0 recompiles, and the same
    # edit grouped beside `forward`/`loss` costs 3. That is what this refuses.
    @test check_hook_grouping([:forward, :loss]) === nothing
    @test check_hook_grouping([:build_model, :build_data, :finalize_metrics]) === nothing
    @test_throws ErrorException check_hook_grouping([:forward, :build_data])
    @test_throws ErrorException check_hook_grouping([:loss, :build_model])
    @test_throws ErrorException check_hook_grouping([:forward, :finalize_metrics])
    # The message has to name BOTH sides, since "split them" is only actionable if you are told
    # which hook to move.
    gmsg = try
        check_hook_grouping([:forward, :build_data])
    catch err
        sprint(showerror, err)
    end
    @test occursin("`forward`", gmsg) && occursin("`build_data`", gmsg)

    # A name the map does not carry would sit unread while the method served every call.
    @test_throws ErrorException check_hook_grouping([:optimizer])
    umsg = try
        check_hook_grouping([:optimizer])
    catch err
        sprint(showerror, err)
    end
    @test occursin("optimizer", umsg)

    # `metrics`/`train_metrics` are residency-decided at run time, so the grouping is reported as
    # an assumption rather than checked. Expansion is when the macro warns, so the expansion has
    # to happen inside the log capture.
    @test_logs (:warn,) macroexpand(
        @__MODULE__, :(
            ReactantNitro.@hooks begin
                metrics(e, pred; y) = (; n = (1, 1))
            end
        )
    )
    @test_logs (:warn,) macroexpand(
        @__MODULE__, :(
            ReactantNitro.@hooks begin
                forward(e, model, ps, st; x) = nothing
                metrics(e, pred; y) = (; n = (1, 1))
            end
        )
    )
    # And a group with no residency hook says nothing at all.
    @test_logs macroexpand(
        @__MODULE__, :(
            ReactantNitro.@hooks begin
                loss(e, pred; y) = 0.0f0
            end
        )
    )
end
