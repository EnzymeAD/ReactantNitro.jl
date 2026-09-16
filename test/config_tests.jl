# Config metadata tests.
#
# The acceptance criterion this file exercises: "config_metadata(MyExp) returns the full record for
# a three-field struct; device_fields/host_fields round-trip; compile_view strips Host". The rest of
# this file covers the mechanics that are easy to get wrong, chiefly that the DECLARED type
# survives device conversion (which changes the struct's type) and that the StrippedHost sentinel
# is loud exactly where it should be and quiet everywhere else.

@testitem "config" begin
    using Test
    using ReactantNitro
    using ReactantNitro: StrippedHost, NO_DEFAULT, _field, graphconst_field_hash

    # The macro's own documented example, verbatim, so the docstring mechanics are exercised
    # exactly as documented.
    # `max_epochs` is deliberately UNMARKED: Host is the default, and the docstring says so.
    @experiment struct MyExp
        "Weight of the auxiliary heatmap loss relative to the primary term."
        aux_weight::Device{Float32} = 0.25f0

        "Number of decoder blocks. Structural: changes the compiled graph."
        n_layers::GraphConst{Int} = 4

        "Epochs to train for. Driver-only: never read inside a traced function."
        max_epochs::Int = 40
    end

    abstract type AbstractExp end

    @experiment struct Undocumented <: AbstractExp
        scale::Device{Float64}          # required: no default
        name::String = "x"
    end

    @experiment struct NoMarkers
        a::Int = 1
    end

    # An experiment with NO Host fields: every field is Device or GraphConst, so compile_view passes it
    # through unchanged.
    @experiment struct NoHostFields
        w::Device{Float32} = 1.0f0
        n::GraphConst{Int} = 2
    end

    # The macro stays optional. A user must be able to hand-write the struct and define the three
    # functions themselves, so every fallback below has to be total.
    struct HandWritten{A, B}
        w::A
        dataset::B
        n::Int
    end
    ReactantNitro.device_fields(::Type{<:HandWritten}) = (:w,)
    ReactantNitro.host_fields(::Type{<:HandWritten}) = (:dataset,)

    @testset "the generated interface" begin
        @testset "config_metadata: the full record" begin
            m = config_metadata(MyExp)
            @test keys(m) == (:aux_weight, :n_layers, :max_epochs)   # declaration order

            @test m.aux_weight.type === Float32
            @test m.aux_weight.default === 0.25f0
            @test m.aux_weight.kind === :device
            @test m.aux_weight.doc ==
                "Weight of the auxiliary heatmap loss relative to the primary term."

            @test m.n_layers.type === Int
            @test m.n_layers.default === 4
            @test m.n_layers.kind === :graphconst

            @test m.max_epochs.type === Int
            @test m.max_epochs.default === 40
            @test m.max_epochs.kind === :host

            # `line` points at the source so a validation error can too, and the three are consecutive
            # declarations in this file.
            @test m.n_layers.line == m.aux_weight.line + 3
            @test m.max_epochs.line == m.n_layers.line + 3

            # A field with no default reports NO_DEFAULT rather than `nothing`, which is a legal default.
            u = config_metadata(Undocumented)
            @test u.scale.default === NO_DEFAULT
            @test u.scale.doc === nothing
            @test u.name.default == "x"

            # An experiment with no fields at all still answers.
            @test config_metadata(NoMarkers) isa NamedTuple
        end

        @testset "device_fields / host_fields round-trip" begin
            @test device_fields(MyExp) == (:aux_weight,)
            @test host_fields(MyExp) == (:max_epochs,)

            # Every field is in exactly one category, and the categories reconstruct the field list.
            # GraphConst is the complement of device ∪ host.
            e = MyExp()
            T = typeof(e)
            gc = Tuple(setdiff(fieldnames(T), device_fields(T), host_fields(T)))
            @test gc == (:n_layers,)
            @test sort(collect((device_fields(T)..., gc..., host_fields(T)...))) ==
                sort(collect(fieldnames(T)))

            # The traits answer for the UnionAll and for every instantiation, which is what lets them be
            # read after setup has changed the experiment's type.
            @test device_fields(typeof(e)) == device_fields(MyExp)
            @test host_fields(typeof(e)) == host_fields(MyExp)

            @test device_fields(NoMarkers) == ()
            @test host_fields(NoMarkers) == (:a,)      # unmarked is Host: the default

            # The fallbacks are total, so framework code may call them on anything.
            @test device_fields(Int) == ()
            @test host_fields(Int) == ()
        end

        @testset "the generated constructor" begin
            e = MyExp()
            @test e.aux_weight === 0.25f0
            @test e.n_layers === 4
            @test e.max_epochs === 40

            e2 = MyExp(; n_layers = 6)
            @test e2.n_layers === 6
            @test e2.aux_weight === 0.25f0

            # A field with no default is a required keyword, so omitting it raises naming it.
            @test_throws UndefKeywordError Undocumented()
            @test Undocumented(; scale = 2.0).scale === 2.0

            # A supertype is preserved.
            @test Undocumented(; scale = 1.0) isa AbstractExp

            # A GraphConst field keeps its declared concrete type, so it converts exactly as @kwdef
            # would; a Device or Host field does not convert, since what it holds changes.
            @test MyExp(; n_layers = 0x03).n_layers === 3
        end

        @testset "@experiment rejects what it cannot express" begin
            # Parametric declarations collide with the type parameters the macro generates.
            @test_throws LoadError @eval @experiment struct Bad1{T}
                x::T = 1
            end
            # A marker needs its type parameter.
            @test_throws LoadError @eval @experiment struct Bad2
                x::Device = 1
            end
            # One field is exactly one of the three categories.
            @test_throws LoadError @eval @experiment struct Bad3
                x::Device{Host{Int}} = 1
            end
            # Comments are invisible to a macro, so a type annotation is not optional.
            @test_throws LoadError @eval @experiment struct Bad4
                x = 1
            end
            @test_throws LoadError @eval @experiment struct Bad5
                x::Int = 1
                x::Int = 2
            end
        end

        @testset "the generated docstring" begin
            s = string(Base.Docs.doc(MyExp))
            @test occursin("aux_weight", s)
            @test occursin("Weight of the auxiliary heatmap loss", s)
            @test occursin("traced input", s)          # the Device kind note
            @test occursin("stripped from the trace", s)   # the Host kind note
            @test occursin("baked constant", s)        # the GraphConst kind note
            @test occursin("the default", s)           # Host is the default
        end
    end

    @testset "compile_view and the Host guard" begin
        e = MyExp()
        ev = compile_view(e)

        @testset "it strips Host and nothing else" begin
            @test ev.max_epochs === StrippedHost{:max_epochs}()
            @test ev.aux_weight === e.aux_weight     # a Device is a traced INPUT: it stays
            @test ev.n_layers === e.n_layers         # a GraphConst field bakes: it stays
            @test typeof(ev) <: MyExp
            @test fieldnames(typeof(ev)) == fieldnames(typeof(e))
        end

        @testset "it is type-stable, since setup rebuilds `e` every optimizer step" begin
            @test (@inferred compile_view(e)) === ev
        end

        @testset "an experiment with no Host fields is passed through unchanged" begin
            nh = NoHostFields()
            @test compile_view(nh) === nh
        end

        @testset "the fallback covers a hand-written experiment (the macro's escape hatch)" begin
            h = HandWritten(1.0f0, collect(1:1000), 7)
            hv = compile_view(h)
            @test hv.dataset === StrippedHost{:dataset}()
            @test hv.w === h.w
            @test hv.n === h.n

            # config_metadata's fallback is total, so framework code never MethodErrors on one of these.
            m = config_metadata(typeof(h))
            @test m.w.kind === :device
            @test m.dataset.kind === :host
            @test m.n.kind === :graphconst
            @test m.n.default === NO_DEFAULT
        end

        @testset "the sentinel is loud exactly where it is documented to be" begin
            s = ev.max_epochs
            # Loud: arithmetic, property access, and conversion. These are the cases that matter,
            # because they are how traced code would USE a Host value.
            @test_throws Exception s + 1
            @test_throws Exception s * 2
            @test_throws Exception s.anything
            @test_throws Exception convert(Int, s)
            @test_throws Exception Int(s)

            # Quiet, and documented as such rather than overstating the guard: four of the five
            # escape routes that disqualified `nothing` remain open. Asserting them keeps the
            # documentation honest.
            @test s === StrippedHost{:max_epochs}()
            @test s == StrippedHost{:max_epochs}()
            @test hash(s) isa UInt
            @test !isnothing(s)
            @test identity(s) === s                      # any ::Any signature accepts it
            @test (; a = s).a === s                       # it survives in a returned NamedTuple

            # The field name is at the TYPE level, which is what lets an error message name the field
            # without the sentinel carrying any data.
            @test StrippedHost{:max_epochs} !== StrippedHost{:other}
            @test occursin("max_epochs", sprint(show, s))
        end
    end

    @testset "_field: a Host field and a method are interchangeable" begin
        e = MyExp()
        @test _field(e, :max_epochs, 1) === 40
        @test _field(e, :not_a_field, 1) === 1
        @test _field(NoMarkers(), :max_epochs, 1) === 1

        # Accessors run host-side against the REAL `e`, never the stripped view. Reading a Host field off
        # the view yields the sentinel, which is the whole point of the split.
        @test _field(compile_view(e), :max_epochs, 1) === StrippedHost{:max_epochs}()
    end

    # ── The StrippedHost message, and the docstring merge ───────────────────────────────

    @experiment struct ForgotMarker
        sz::Int = 512                      # unmarked, so Host, and read under trace by mistake
    end

    @testset "reading an unmarked field under trace says WHAT TO DO, not just that it failed" begin
        # Under the old scheme this sentinel defined no methods and you got
        # `MethodError: no method matching *(::StrippedHost{:sz}, ::Float32)`, which names the field only
        # by accident and offers no fix. Since `Host` became the DEFAULT, forgetting `GraphConst` is the
        # common mistake rather than an exotic one, so the message has to carry the remedy.
        ev = compile_view(ForgotMarker())
        err = try
            ev.sz * 2.0f0
        catch ex
            ex
        end
        @test err isa ErrorException

        @testset "it names the field, the cause, and BOTH fixes with their consequences" begin
            @test occursin("`e.sz`", err.msg)
            @test occursin("is not\nmarked", err.msg) || occursin("not marked", err.msg)
            # Both, because the framework cannot know which was meant; naming one would be advice half
            # the time.
            @test occursin("sz::GraphConst{T}", err.msg) && occursin("RECOMPILES", err.msg)
            @test occursin("sz::Device{T}", err.msg) && occursin("does NOT recompile", err.msg)
            # And the third answer, which is the commonest: it did not belong under trace at all.
            @test occursin("read it host-side instead", err.msg)
        end

        @testset "every high-traffic way of USING one routes to that message" begin
            for f in (
                    s -> s * 1.0f0, s -> 1.0f0 * s, s -> s + 1, s -> s / 2, s -> s.field,
                    s -> s[1], s -> length(s), s -> collect(s), s -> s .+ 1,
                )
                e2 = try
                    f(compile_view(ForgotMarker()).sz); nothing
                catch ex
                    ex
                end
                @test e2 isa ErrorException && occursin("not\nmarked", e2.msg * "\n")
            end
        end

        @testset "detection is NOT widened, and the docstring says so" begin
            # Merely storing or comparing the sentinel still escapes, exactly as before. Asserted so the
            # honest limitation stays honest rather than drifting into an implied guarantee.
            s = compile_view(ForgotMarker()).sz
            @test s === StrippedHost{:sz}()
            @test (; a = s).a === s
            @test !isnothing(s)
        end
    end

    """
    A documented experiment.

    Second paragraph of the user's prose.
    """
    @experiment struct DocMerged
        sz::GraphConst{Int} = 8
        w::Device{Float32} = 1.0f0
        epochs::Int = 3
    end

    @experiment struct DocPlain
        sz::GraphConst{Int} = 8
    end

    @testset "a docstring on @experiment MERGES with the generated marker table" begin
        # Two bugs in one: the macro's block could not be documented at all (`cannot document the
        # following expression`), and once `Base.@__doc__` fixed that, the macro's own generated table
        # OVERWROTE the user's prose. Merging is the user's call: the prose is intent, the table records
        # which fields bake, cross as inputs, or are invisible, and that half a user cannot write.
        d = string(@doc DocMerged)
        @test occursin("A documented experiment.", d)          # the prose survived
        @test occursin("Second paragraph", d)
        @test occursin("GraphConst", d) && occursin("Device", d) && occursin("Host", d)
        # Order matters: intent above, mechanism below.
        @test findfirst("A documented experiment.", d).start < findfirst("Field", d).start

        @testset "and an undocumented one still gets the table alone" begin
            p = string(@doc DocPlain)
            @test occursin("GraphConst", p)
            @test !occursin("A documented experiment.", p)
        end

        @testset "the table names each marker, so it is actionable" begin
            # A reader deciding a field is in the wrong category needs the word they would type.
            @test occursin("`GraphConst`", d) && occursin("`Device`", d) && occursin("`Host`", d)
        end
    end

    @testset "merging a docstring emits no 'Replacing docs' warning" begin
        # Found by the first real consumer: every precompile of a package with a documented experiment
        # printed `Warning: Replacing docs for ... :: Union{}`. The merge was correct and the warning
        # said the opposite, so a user would reasonably conclude their prose had been thrown away, which
        # is the exact silent-discard outcome merging exists to avoid.
        #
        # It has to be evaluated in a NON-active module to reproduce: `Base.Docs.doc!` suppresses the
        # warning inside `Base.active_module()`, which is `Main` under the test runner, so a probe in
        # Main passes vacuously and proves nothing.
        probe = Module(:DocWarnProbe)
        Base.eval(probe, :(using ReactantNitro))
        src = """
        \"\"\"
        User prose that must survive.
        \"\"\"
        @experiment struct Warned
            a::GraphConst{Int} = 1
        end
        """
        @test_nowarn Base.eval(probe, Meta.parse(src))

        @testset "and the merge still actually happened in that module" begin
            d = string(Base.eval(probe, :(@doc Warned)))
            @test occursin("User prose that must survive.", d)
            @test occursin("GraphConst", d)
        end
    end

    # ── named configurations ─────────────────────────────────────────────────────────────

    @experiment struct PresetExp
        n_classes::GraphConst{Int} = 8
        scale::Device{Float32} = 1.0f0
        rotate_deg::Float64 = 7.5
        max_epochs::Int = 50
    end

    ReactantNitro.presets(::Type{PresetExp}) = (
        reference_v1 = (; n_classes = 8, rotate_deg = 7.5),
        current = (; n_classes = 10, rotate_deg = 12.0, max_epochs = 80),
        typo = (; rotate_degrees = 12.0),          # the mistake requirement 3 exists for
        not_a_table = 42,
    )

    @experiment struct NoPresetExp
        n::Int = 1
    end

    @testset "presets: the table is data, and partial" begin
        @test presets(NoPresetExp) === (;)          # a model declaring none loses nothing
        @test keys(presets(PresetExp))[1] === :reference_v1

        e = from_preset(PresetExp, :current)
        @test e.n_classes == 10
        @test e.rotate_deg == 12.0
        @test e.max_epochs == 80
        @test e.scale isa Float32                   # untouched by the preset, still the struct default

        # PARTIAL: reference_v1 names two fields and everything else falls through.
        r = from_preset(PresetExp, :reference_v1)
        @test r.n_classes == 8 && r.rotate_deg == 7.5
        @test r.max_epochs == 50                    # the struct default, not `current`'s 80
    end

    @testset "explicit overrides beat the preset" begin
        e = from_preset(PresetExp, :current; max_epochs = 40, rotate_deg = 1.0)
        @test e.max_epochs == 40
        @test e.rotate_deg == 1.0
        @test e.n_classes == 10                     # not overridden, so the preset still wins
    end

    @testset "validation, which is the correctness argument" begin
        # An unknown KEY is the whole point: splatted, it would be a silent no-op and the recipe would
        # quietly not mean what it says.
        err = try
            from_preset(PresetExp, :typo)
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("rotate_degrees", err.msg)   # names the offending key
        @test occursin("rotate_deg", err.msg)       # and the valid set

        # An unknown PRESET names the ones that exist.
        err2 = try
            from_preset(PresetExp, :nope)
            nothing
        catch ex
            ex
        end
        @test err2 isa ErrorException
        @test occursin("reference_v1", err2.msg) && occursin("current", err2.msg)

        # A model with no table at all says so rather than listing an empty set.
        err3 = try
            from_preset(NoPresetExp, :anything)
            nothing
        catch ex
            ex
        end
        @test err3 isa ErrorException
        @test occursin("no `presets` method", err3.msg)

        # A preset that is not a NamedTuple is a setup error, not a splat failure.
        @test_throws ErrorException from_preset(PresetExp, :not_a_table)
    end

    @testset "marker semantics are untouched, which is the property worth having" begin
        # A preset sets field VALUES, so the compile key follows the markers exactly as always: two
        # presets differing only in a Host field share a graph, one differing in GraphConst does not.
        a = from_preset(PresetExp, :reference_v1)
        b = from_preset(PresetExp, :reference_v1; max_epochs = 999)   # Host only
        c = from_preset(PresetExp, :current)                          # moves n_classes, a GraphConst
        @test graphconst_field_hash(compile_view(a)) == graphconst_field_hash(compile_view(b))
        @test graphconst_field_hash(compile_view(a)) != graphconst_field_hash(compile_view(c))
    end

    @testset "`Nitro(E, name)` splits keywords by which set they belong to" begin
        # The rule in one line: a `Nitro` keyword goes to `Nitro`, anything else must be a field and
        # goes to the recipe. Reported from the first real use, where every override the model needed
        # was a FIELD, which made the two-name escape hatch the common path instead of the exception.
        nkw = ReactantNitro._nitro_keywords()
        @test :max_epochs in nkw && :run_dir in nkw && :preset in nkw
        @test !(:rotate_deg in nkw)                 # a field, not a run keyword

        # DERIVED from the declaration, never hardcoded, so it cannot go stale as keywords are added.
        # `_build_nitro` and NOT `Nitro`: the public constructor is a wrapper that moves the build
        # off the interactive thread and forwards a bare `kwargs...`, and the accessor defaults
        # deliberately live on the function that reads `e`, so `_build_nitro` is the only place the
        # run keywords are named. Reading `which(Nitro, Tuple{Any})` returned just the sink, which
        # left every keyword above unclassified.
        @test nkw == Base.kwarg_decl(which(ReactantNitro._build_nitro, Tuple{Any}))
        @test !ReactantNitro.has_sink(nkw)   # and it is the DECLARING method, not another wrapper
    end

    @testset "the split's error names both valid sets" begin
        err = try
            ReactantNitro._split_preset_kwargs(PresetExp, :current, (; nonsense = 1))
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("nonsense", err.msg)
        @test occursin("max_epochs", err.msg)       # the Nitro keywords
        @test occursin("rotate_deg", err.msg)       # and the fields
    end

    @testset "the recorded form REFUSES `data`, because it would train on the wrong thing" begin
        # `data` is the one invocation keyword that SKIPS a setup step: supplying it skips
        # `build_data`, and in this form the framework builds the experiment, so the caller's
        # collection came from a different instance. Anything `build_data` populates on the
        # experiment stays empty here and `derive` reads it without complaining. Measured on the
        # first real preset table as a 10-class weighted loss over a zero-length weight vector.
        err = try
            ReactantNitro._split_preset_kwargs(PresetExp, :current, (; data = (; train = [1])))
            nothing
        catch ex
            ex
        end
        @test err isa ErrorException
        @test occursin("build_data", err.msg)
        @test occursin("from_preset(PresetExp, :current)", err.msg)   # the fix, copy-pasteable

        # AND IT CARRIES THE FIELD OVERRIDES. Anyone reaching the recorded form is disproportionately
        # likely to have them, since keyword splitting is the main reason to use it, so a suggestion
        # that dropped them would be copy-pasteable and WRONG: silently building the real backbone
        # instead of a stub is a multi-minute compile of the wrong model, not an obvious failure.
        err2 = try
            ReactantNitro._split_preset_kwargs(
                PresetExp, :current, (; rotate_deg = 99.0, data = (; train = [1]))
            )
            nothing
        catch ex
            ex
        end
        @test err2 isa ErrorException
        @test occursin("from_preset(PresetExp, :current; rotate_deg = 99.0)", err2.msg)

        # A long value is elided rather than pasted as a screenful.
        err3 = try
            ReactantNitro._split_preset_kwargs(
                PresetExp, :current, (;
                    rotate_deg = 1.0, scale = collect(1.0f0:40.0f0),
                    data = (; train = [1]),
                )
            )
            nothing
        catch ex
            ex
        end
        @test occursin("<Array>", err3.msg)      # `nameof(typeof(::Vector))` is `Array`

        # The other three invocation-only keywords skip nothing and stay legal: `checkpoint` and
        # `resume` feed the record lookup, `run_ref` is an output channel.
        for k in (:checkpoint, :resume, :run_ref)
            nkw, fkw = ReactantNitro._split_preset_kwargs(PresetExp, :current, (; k => nothing))
            @test first(first(nkw)) === k
            @test isempty(fkw)
        end
    end

    # ── showing an experiment, without showing a device buffer ──────────────────────────
    #
    # `Device{T}` takes any `T`, and the read-only buffer case puts WEIGHTS in one: a
    # `Device{NamedTuple}` or `Device{Tuple}` of arrays that an `hlo_call` reads. Under Julia's
    # default struct `show` that config prints element by element, so `display(e)` dumps a model.
    # Measured before this existed: 51,635 characters for one 64x64 and two 16x16 buffers.
    @testset "an experiment shows its values and never a device buffer" begin
        @experiment struct BufExp
            width::GraphConst{Int} = 8
            smoothing::Device{Float32} = 0.05f0
            buffers::Device{NamedTuple{(:w, :b), Tuple{Matrix{Float32}, Vector{Float32}}}} =
                (w = zeros(Float32, 64, 64), b = zeros(Float32, 64))
            pair::Device{Tuple{Matrix{Float32}, Matrix{Float32}}} =
                (zeros(Float32, 16, 16), ones(Float32, 16, 16))
            max_epochs::Int = 1
        end

        e = BufExp()
        for s in (sprint(show, e), sprint(show, MIME"text/plain"(), e))
            # The test that catches a regression to the struct default: the printed form of an
            # array of floats. 4,096 zeros print as `0.0, 0.0, ...` and nothing else here does.
            @test !occursin("Float32[", s)
            @test !occursin("0.0, 0.0", s)
            # Bounded by the FIELD COUNT, never by the model. Five fields cannot reach this.
            @test length(s) < 2_000
            # A buffer is named by its shapes, which is what makes the summary useful rather than
            # merely short: both members of the NamedTuple and both of the Tuple.
            @test occursin("size 64x64", s) && occursin("size 64", s)
            @test count("size 16x16", s) == 2
            # Values, not just names: a config table that withheld its scalars would be useless.
            @test occursin("0.05", s) && occursin("8", s)
        end

        # The long form carries the marker per field, which is the column a reader acts on.
        long = sprint(show, MIME"text/plain"(), e)
        @test occursin("GraphConst", long) && occursin("Device", long) && occursin("Host", long)
        for f in fieldnames(BufExp)
            @test occursin(string(f), long)
        end

        # The short form stays one line, because that is where a nested display puts it.
        @test count("\n", sprint(show, e)) == 0

        # An experiment with no Device field at all is the common case and must not regress into
        # something less readable than the struct default was.
        @experiment struct PlainExp
            n::GraphConst{Int} = 3
        end
        @test occursin("n", sprint(show, PlainExp())) && occursin("3", sprint(show, PlainExp()))
    end

end
