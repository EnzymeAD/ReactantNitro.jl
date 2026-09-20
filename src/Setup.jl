# Setup.jl
#
# The setup sequence, the `Nitro` constructor, and the accelerator and distribution surface.
#
# THE SETUP SEQUENCE IS THE SINGLE AUTHORITY ON ORDERING. Every edge has a reason, and the reason is
# why the order cannot be permuted. The sequence is written out in `Nitro`'s implementation below
# rather than paraphrased somewhere else, because a description kept apart from the code it orders
# drifts from it.

"""
    Nitro(e; seed, resume, run_dir, data, n_devs, checkpoint, accum, max_epochs, schedules,
             gradient_clip_norm, logger, checkpointer, early_stop, run_ref, weights, w0) -> Nitro

Run the setup sequence and return the resulting handle. **No training.**

**The signature below is the single authority for these defaults and this docstring deliberately
does not restate them**, because `train!(nitro)` takes no keywords and `train!(e; kwargs...)`
forwards here, so a second list is a second authority with nothing keeping the two equal. It had
already drifted once: this docstring carried a stale table giving `seed`, `logger`, `checkpointer`,
and five more as literal values and saying **three** keywords default to an accessor, while the
signature below it defaults **ten** to one. The values live in the signature; the shape of the rule
lives here.

**Ten keywords default to an accessor of the same name**, which is how the "defaulted accessor" and
"value passed to `train!`" mechanisms compose: the keyword replaces the accessor's
value for one run, and omitting it falls through to the experiment's own. **Four stay keyword-only**,
`data`, `checkpoint`, `resume`, and `run_ref`, each naming a fact about this invocation rather than a
property of the experiment. The per-group accessors are outside the mechanism entirely, having no
sensible keyword form.

**`early_stop` defaults to `nothing`, so a bare experiment does not early-stop.** A framework that
truncates your run by default is surprising, and any patience value would be a guess.
**`max_epochs` defaults to `1`**, since a bare experiment has no such field; stated because a first
run finishing after one epoch reads as a bug otherwise.

## The fresh sequence

| # | Step | Why here |
| --- | --- | --- |
| 1 | Validate config: markers, optimizer allowlist membership, mutually exclusive decay settings | Cheapest failures first, before any allocation |
| 2 | Seed the global rng from `seed` | Must precede anything that draws, i.e. steps 3 and 6 |
| 3 | `build_data(e, dist)` | `derive` needs data; the cache is created after it |
| 4 | `derive(e, data)`, merge into `e` | Needs data; must precede conversion, since it returns host values, and must precede `build_model`, since a derived structural value changes the architecture |
| 4.5 | `setup_devices` | Must precede step 5, because conversion places values on the mesh |
| 5 | Convert `Device` fields to device, rebuild `e`; **`e`'s type changes here** | Must follow every hook that produces config, must precede every hook that is traced |
| 6 | `build_model(e, rng)` | Sees the post-conversion `e`. Pretrained loading happens inside it |
| 7 | Capture `w0`, the parameters exactly as `build_model` returned them | Must be immediately after, before any restore or step |
| 8 | Param groups, the flat permutation, the per-group ranges | Needs `ps`. **Unconditional**, including for an evaluation `Nitro`: it is host-side bookkeeping, and the resume path validates the restored permutation against it |
| 9 | **Training only.** Flatten `ps`; build optimizer state and **normalize it to device residency** | Needs the layout; normalization is what makes tracing correct |
| 10 | Create the compile cache; resolve batch routing, **always against `typeof(compile_view(e))`** | Needs the batch schema and the hooks. The stripped type is what the trace site actually calls with |
| 11 | **Training only.** Check `length(train) % accum == 0`; resolve the horizon `total = max_epochs * div(steps_per_epoch, accum)` and call each schedule factory once with it | Needs `length(train)`, `accum`, and `max_epochs`. The check precedes the horizon because an exact `div` depends on it |
| 12 | Construct the logger, log parameters and the seed | Before the first compile, which can crash |
| 13 | First trace and compile, against `compile_view(e)` | Everything above is a prerequisite. **The tracer must never see the full `e`** |

**Which experiment do hooks see?** `build_data` and `derive` see the **pre-conversion** experiment,
where `Device` fields hold plain host values. Every other hook, including `build_model`, `forward`,
`loss`, `metrics`, `optimizer`, and every accessor, sees the **post-conversion** one. Two hooks, one
boundary.

## Resume inserts three steps and changes nothing else

  * **Between 1 and 2:** locate the checkpoint (`:auto` finds `latest` in `run_dir`), read its
    record, run the config compatibility check, and **restore `seed` from the record, overriding
    any `seed = ...` and warning when they differ**. A silent override turns a seed sweep that forgot
    to vary `run_dir` into N identical runs. Restoring is what makes `w0` reproducible, so the
    override is right and only its silence was wrong.
  * **After 7:** restore `ps`, `st`, `opt_state`, `step`, and `epoch`. **`w0` is the value freshly
    captured at step 7, verified against the record's `anchor_checksum`**, which is why step 6's
    rebuild happens before the restore in spite of being thrown away.
  * **After 8:** verify the restored flat permutation matches the one just computed, and refuse with
    a diff if not. A silently different permutation scrambles `opt_state` against `ps`.
  * **After 9: re-normalize the restored `opt_state` and re-run the no-host-`Number` assertion.**
    Step 9's normalization applies to state the framework *constructed*; a restore does not
    construct, so without this the resumed run reacquires the frozen-step-counter bug in full.
    **That is the single most dangerous omission this framework could have**, because the symptom is
    a plausible loss curve and no error.

## A warm start inserts one step

`weights = other::Nitro` takes `ps` and `st` from a handle in this process and otherwise runs the
fresh sequence: `derive` runs against THIS run's data, the optimizer state is fresh, and `step` and
`epoch` are zero. It is the in-memory sibling of `checkpoint = path`, for the REPL case where the
trained handle is right there. The transfer happens **after 7**, through host memory, so the source
may sit on a different mesh; the trees must match leaf for leaf in keypath and size, and a mismatch
is refused with a diff. `weights` together with `checkpoint` or `resume` is an error: two sources
for one thing.

`w0` says what `decay_anchor = :w0` anchors to for such a run. `:build_model`, the default, keeps
today's meaning, the freshly initialized parameters. `:weights` anchors to the transferred ones,
which is the L2-SP fine-tuning setup: initialize from A, anchor to A. A parameter tree anchors to
that tree: initialize from A, anchor to B. `:weights` without `weights` is an error.

## The no-train variant

A `Nitro` built without a `train` split **skips steps 9 and 11**, since both are training-only and
`total` is undefined with no training split. **Step 8 still runs.** Batch routing and `batch_size`
fall back to the first batch of whichever split exists; with no split at all, routing is **deferred
to the first `predict` call**, which supplies a batch, and `batch_size` is the width of that batch,
with no padding needed because the caller supplied it whole. That is the one case where the
ordering above does not fully apply, and it is why routing is described as resolved once per run
rather than once at setup.
"""
# `Starting` BEFORE the build, through the module-level monitors, because the handle this would
# publish through is what the build is for. `build_data` is inside `_build_nitro`, and on this stack
# it can start and compile a data server, so this is the longest undeclared stretch there was.
Nitro(e; kwargs...) = _off_interactive() do
    publish_phase(Starting())
    return _build_nitro(e; kwargs...)
end

# The body of `Nitro(e)`, split out so the public constructor can move it off the interactive
# thread. Construction blocks its calling thread for 45 to 95 s (device init, `build_model`, the
# device conversion), and when the caller is the main task under `julia -t N,1` that thread is the
# interactive one, which starves every `:interactive` task in the process: an external supervisor's
# session heartbeat among them, so the supervisor reclaimed the session and killed the run before
# its first step. `_off_interactive` (Phases.jl) carries the measurement and applies `with_repl`'s
# spawn policy, so single-threaded sessions and worker-thread callers run this inline exactly as
# before. The keyword defaults stay HERE, on the function that reads `e`, so `Nitro(e; kwargs...)`
# forwards only what the caller passed and the accessor defaults keep a single home.
function _build_nitro(
        e;
        # Ten keywords default to an accessor of the same name, so an experiment
        # declares what it IS and a caller passes only what this run changes.
        seed::Integer = ReactantNitro.seed(e),
        run_dir::AbstractString = ReactantNitro.run_dir(e),
        n_devs::Integer = ReactantNitro.n_devs(e),
        accum::Integer = ReactantNitro.accum(e),
        max_epochs::Integer = ReactantNitro.max_epochs(e),
        schedules = ReactantNitro.schedules(e),
        gradient_clip_norm = ReactantNitro.gradient_clip_norm(e),
        logger = ReactantNitro.logger(e),
        checkpointer = ReactantNitro.checkpointer(e),
        early_stop = ReactantNitro.early_stop(e),
        # Not an accessor: a preset name is a fact about how THIS handle was built, so there
        # is nothing on `e` to read it from. `Nitro(E, :name)` below sets it for you.
        preset::Union{Symbol, Nothing} = nothing,
        # Four stay keyword-only, because each names a fact about THIS INVOCATION rather than
        # a property of the experiment. `data`'s accessor exists and is called `build_data`.
        resume = false,
        data = nothing,
        checkpoint = nothing,
        run_ref = nothing,
        # A warm start: the initial weights from another handle in this process, and what `:w0`
        # anchoring means for a run that starts from them. Keyword-only for the same reason as the
        # four above: both are facts about THIS construction.
        weights = nothing,
        w0 = :build_model,
        # PROTOTYPE: hooks supplied as VALUES, shadowing the method for any name present. See
        # Hooks.jl for why an entry may not capture and why that keeps the cache key sound.
        hooks = (;)
    )
    check_hooks(hooks)
    check_weights_kwargs(weights, w0; checkpoint, resume)

    # ── before step 2: locate a checkpoint (the resume path) ───────────────────────
    # `run_dir` is the one path concept in the framework, so a checkpointer constructed without an
    # explicit `dir` adopts it here. An explicit `dir` pins it and is never overwritten.
    checkpointer isa TopKCheckpointer && checkpointer.dir === nothing &&
        (checkpointer.dir = String(run_dir))
    # The checkpoint filename, adopted under the same rule and in the same place: a checkpointer
    # with no `name` binds the experiment's `checkpoint_filename` hook, and an explicit `name` pins
    # it.
    #
    # BOUND HERE RATHER THAN PASSED TO `save_checkpoint!`, which never receives `e`. Widening that
    # four-argument contract would hand the object full of DEVICE-resident leaves to the function
    # whose job is serialization, which is the failure the record's host discipline and
    # `assert_host_record` exist to prevent; the assertion walks the snapshot and cannot walk `e`.
    #
    # The closure calls the GENERIC FUNCTION, so dispatch happens at write time and a revised
    # `checkpoint_filename` takes effect on the next checkpoint of a live run. That is deliberately
    # unlike a revised `checkpointer` accessor, which `_PROBED` below records as undetected because
    # an accessor that CONSTRUCTS is called once, here.
    checkpointer isa TopKCheckpointer && checkpointer.name === nothing &&
        (checkpointer.name = (; kwargs...) -> ReactantNitro.checkpoint_filename(e; kwargs...))
    # The default logger adopts the run's resolved directory exactly like the checkpointer
    # above, and through the same channel: `_adopt_logger!` is the logger-side half, so a wrapper
    # around the default (the gate extension's RecordingLogger) can forward the pin and the
    # wrapped default still lands in the run's directory. A user's own logger is untouched by it.
    logger = _adopt_logger!(logger, run_dir)

    record, source = nothing, nothing
    if checkpoint !== nothing
        # `load_checkpoint` dispatches on the CHECKPOINTER rather than on the path, so a
        # non-file backend can implement it at all. The corollary is that `checkpointer = nothing`
        # has no loader, and its `::Nothing` method returns `nothing` rather than raising. An
        # explicit `checkpoint = path` therefore has to be checked: without this, a serving
        # construction that quite reasonably disables checkpoint WRITING would load nothing, train
        # nothing, and answer with freshly initialized weights that look plausible.
        checkpointer === nothing && error(
            """
            ReactantNitro: `checkpoint = $(repr(checkpoint))` asks for a checkpoint to be loaded and
            `checkpointer = nothing` has no loader, since `load_checkpoint` dispatches on the
            checkpointer rather than on the path.
            Pass the checkpointer that wrote it, or leave the keyword off to take the default
            `TopKCheckpointer`, which reads the format the framework writes. Disabling checkpointing
            is about WRITING; nothing is written by a `Nitro` that is never trained."""
        )
        record, source = load_checkpoint(checkpointer, checkpoint), checkpoint
    elseif resume === :auto
        found = find_latest(checkpointer, run_dir)
        # No announcement: `:auto` is opt in, so a restore here is what the caller asked for.
        found === nothing || ((record, source) = (load_checkpoint(checkpointer, found), found))
    elseif resume !== false
        record, source = load_checkpoint(checkpointer, resume), resume
    end
    # There are TWO restores with one compatibility check, and they differ in exactly one place:
    # `checkpoint = path` restores the weights and the derived `Device` values FROM the record,
    # because an evaluation process may not have the training data to recompute them from, while
    # `resume` recomputes derived values through `derive` and continues the trajectory.
    weights_only = record !== nothing && checkpoint !== nothing

    # ── 1. validate config, cheapest failures first ────────────────────────────────
    validate_config(e; accum, gradient_clip_norm)

    # Manual mode, selected by the presence of a `train_step` method for this experiment type (or
    # the `manual_training(e) = false` override declining it). Resolved ONCE, here, and frozen into
    # the handle: the driver a `Nitro` uses is fixed at construction, so flipping this on an
    # existing handle is inert and reported by `fixed_config_report` rather than applied silently.
    manual = manual_training(e)
    if manual && accum > 1
        error(
            """
            ReactantNitro: manual mode (`train_step` defined) does not support `accum > 1`.
            The closure is called once per optimizer step on one batch;
            accumulation is the automatic loop's mechanism, and the closure owns its own multi-batch
            work instead: loop inside `train_step` if you want several updates per call. Set
            `accum = 1`."""
        )
    end

    # ── 2. seed, before anything that draws ────────────────────────────────────────
    # The seed is RESTORED from the record, overriding any `seed = ...`, and the override is
    # announced. A silent one turns a seed sweep that forgot to vary `run_dir` into N identical
    # runs. Restoring is what makes `w0` reproducible, which
    # the anchor checksum then depends on, so the override is right and only silence was wrong.
    if record !== nothing && record.seed != seed
        @warn "ReactantNitro: restoring `seed = $(record.seed)` from the checkpoint, overriding the \
               requested `seed = $seed`. A rebuild must reproduce the same initialization: the \
               resume check verifies the decay anchor against a freshly captured `w0`."
        seed = record.seed
    end
    Random.seed!(seed)
    rng = Random.default_rng()

    # ── 3. data. `build_data` sees the PRE-conversion experiment ───────────────────
    collection = data === nothing ? hook_fn(hooks, :build_data, build_data)(e, nothing) : data
    collection isa NamedTuple || error("ReactantNitro: `build_data` must return a NAMED collection, \
        `(; train, val)` or `(; train, val, test)`, and returned a `$(typeof(collection))`. Named and \
        extensible, so `evaluate` with no `test` supplied is a clear error rather than a positional \
        mistake.")
    for name in keys(collection)
        # Through the wrapper: a `PrefetchIterator` is a declaration carrying a source and a
        # depth, so the contract belongs to what it wraps. It forwards `length` anyway, and checking
        # the source is what makes the error name the loader the user actually wrote.
        check_data_source(prefetch_source(getproperty(collection, name)), name)
    end
    training = haskey(collection, :train)

    # ── 4. derive, still pre-conversion, so it returns host values ─────────────────
    # The one place the two restores differ: a `checkpoint = path` construction takes the derived
    # `Device` values from the record instead of recomputing them, because an evaluation process may
    # have no training data to derive from. A resume recomputes them, and the config comparison
    # excludes derived values for exactly that reason.
    e = weights_only ? merge_derived(e, restored_devices(record, e)) :
        merge_derived(e, derive(e, collection))

    # Every `GraphConst` value must hash and compare by CONTENT, checked BEFORE the comparison
    # below rather than after. `check_config_compatible` compares with `isequal`, so a field that
    # answers by identity fails it every time and prints a diff whose two sides are identical; that
    # is the confusing symptom, not the cause. Asserting first turns it into an error that names the
    # field and the fix. Here for the same reason the config check is here: `derive` has merged, so
    # a derived GraphConst field is finally in the struct to check.
    assert_graphconst_hashable(e)

    # The resume config check, run HERE rather than between steps 1 and 2 where the sequence above
    # puts it, because the field set is not final until `derive` has merged: a derived GraphConst
    # field changing is one of the cases that check makes an error, and before step 4 it is not yet
    # in the struct to compare.
    record === nothing || check_config_compatible(record, e, source)

    # ── 4.5 devices, before conversion places values on the mesh ───────────────────
    mesh = setup_devices(n_devs)

    # ── 5. convert Device fields. `e`'s TYPE CHANGES HERE ─────────────────────────
    e = to_device_config(e, mesh)
    ev = compile_view(e)

    # ── 6, 7. build the model, then capture w0 immediately ─────────────────────────
    model, ps, st = hook_fn(hooks, :build_model, build_model)(e, rng)
    # Replicated on a mesh, plain placement on one device. Parameters and layer state exist
    # identically on every device; only the batch is sharded.
    ps = place_replicated(ps, mesh)
    st = place_replicated(st, mesh)
    w0_tree = deepcopy(ps)            # step 7: before any restore or step

    # ── after 7: restore the weights. `w0` above is the FRESH capture, deliberately ─
    # Step 6's rebuild happens before the restore in spite of being thrown away, because the anchor
    # checksum is verified against a freshly captured `w0`. That is what makes storing a checksum
    # instead of the anchor arrays sound.
    if record !== nothing
        # `from_host` FIRST, then place. A record may hold a surrogate for a value whose
        # type forbids host contents, `HostRNG` for `Reactant.ReactantRNG` being the worked case, and
        # `to_rarray` has no way to know what that surrogate stood for.
        ps = place_replicated(from_host(to_host(record.ps)), mesh)
        st = place_replicated(from_host(to_host(record.st)), mesh)
    end

    # ── after 7: a warm start from another handle ──────────────────────────────────
    # Through host memory, exactly as the record path above: the source may live on another mesh,
    # and `to_host` is the one walker that knows every device leaf. The structural check runs
    # against the FRESH layout, because that is the model this experiment builds; the transferred
    # tree has to fit it, not the other way round.
    if weights !== nothing
        check_weights_compatible(weights, build_layout(e, ps))
        ps = place_replicated(from_host(to_host(weights.ps)), mesh)
        st = place_replicated(from_host(to_host(weights.st)), mesh)
    end
    w0_tree = resolve_w0(w0, w0_tree, ps, e, mesh)

    # ── 8. the flat layout. UNCONDITIONAL, including for an evaluation Nitro ───────
    layout = build_layout(e, ps)
    anchors = decay_anchors(e, w0_tree, layout, mesh)
    checksums = anchor_checksums(anchors)

    # ── after 8: the permutation and the decay anchor, both refusals ───────────────
    if record !== nothing
        check_permutation_compatible(record, layout, source)
        check_anchors_compatible(record, layout, anchors, source)
        # `weights_only` continues no run, so declining the logger drops no history. Without
        # this, an export or an offline evaluation of any logger-written checkpoint had no legal way
        # to avoid writing to the training run's experiment.
        check_logger_compatible(record, logger, source; weights_only)
    end

    # ── 10. routing, resolved against typeof(compile_view(e)) ──────────────────────
    # The fallback table: the first batch of `train`, else of whichever split exists, else NOTHING,
    # and routing defers to the first `predict` call, which supplies a batch. The three products
    # stay `nothing` in that last case and `predict_routing!` fills them. `opt_state` is declared
    # here for manual mode, which builds it in this block (see below).
    routing, batch_size, schema = (nothing, nothing, nothing)
    opt_state = nothing
    if !isempty(keys(collection))
        schema_split = training ? :train : first(keys(collection))
        # `prefetch_source` before `first`. The wrapper's own `iterate` is a passthrough, so
        # this is the same batch either way; going through the source says why it is the same batch
        # rather than relying on that.
        probe = first(prefetch_source(getproperty(collection, schema_split)))
        routing = resolve_routing(ev, probe; model, ps, st, hooks)
        if training && manual
            # Manual mode builds its optimizer states HERE, before `batch_size_of`, because the
            # closure's router must be part of the routing the batch size is inferred from: a field
            # only the closure reads (the GAN's noise, say) declares nothing to the other hooks, so
            # without this the batch size would be uninferable. `setup_optimizers` is the step 9
            # replacement; the framework normalizes the result to device residency through the
            # `to_device_leaf` walk and asserts the no-host-`Number` property on every leaf's STATE
            # (not its rule: nonschedulable rule fields legitimately stay host).
            opt_state = setup_optimizers(e, model, ps, st, mesh)
            opt_state = to_device_leaf(opt_state; mesh)
            assert_opt_state_device(opt_state)
            # Restore re-normalization, exactly as the automatic path does at step 9: a restore
            # constructs nothing, so without this the resumed run reacquires the frozen-step-counter
            # bug in full. Weights-only constructions deliberately keep a fresh optimizer, matching
            # the automatic path.
            if record !== nothing && !weights_only
                opt_state = to_device_leaf(record.opt_state; mesh)
                assert_opt_state_device(opt_state)
            end
            ks = route_keys(
                declared(train_step, Tuple{typeof(ev), typeof(model), typeof(ps), typeof(opt_state), typeof(st)}),
                keys(probe), :train_step
            )
            routing = ks === nothing ? routing : merge(routing, (; train_step = Router{ks}()))
        end
        validate_batch(probe, routing)
        batch_size = batch_size_of(probe, routing)
        schema = keys(probe)
        # The batch is sharded on the `:data` axis, so it has to divide across the mesh.
        # Checked here rather than at the first transfer because here it is a legible setup error
        # and there it is an XLA shape complaint that names neither number.
        check_shardable_batch(batch_size, n_devs)
    end

    # ── after 10. prefetch, applied by the framework rather than requested ──────────
    #
    # HERE, and not at step 3, deliberately. Everything above that inspects the collection
    # STRUCTURALLY has now run against exactly what `build_data` returned: `check_data_source` names
    # the user's own loader, `derive(e, collection)` sees the splits it built, and the schema probe
    # draws its batch through `prefetch_source`. Wrapping earlier would put a framework wrapper in
    # front of a `derive` that reaches into a split, which is a break nothing in the contract forbids.
    #
    # Only `train` is wrapped, and the reason is that `run_eval` iterates its split directly and
    # never enters `batch_stream`, so a wrapped eval split would report a worker count nothing uses.
    collection = auto_prefetch(collection)
    # AFTER the wrap, because the question is about the resolved pipeline and not about the source:
    # a split the user declined with `NoPrefetch` runs its data path inline, consuming each batch
    # before the next is built, so a loader option that a producer running ahead would break is
    # perfectly safe there.
    # EVERY split, including the ones that declined prefetch. Some of what a source type can get
    # wrong depends on the resolved path and some does not: a training loader that keeps its partial
    # final batch is wrong whether or not anything reads ahead, and `cfg` is passed so the hook can
    # tell the two kinds apart rather than the caller guessing for it.
    for nm in keys(collection)
        split = getproperty(collection, nm)
        check_source_options(prefetch_source(split), nm, prefetch_config(split))
    end
    training && warn_no_concurrency(collection.train)

    # The early-stopping setup checks, here because they need both the split collection and the
    # resolved routing: a policy with no `val` split to read, or naming a metric an
    # experiment defining no `metrics` cannot emit, is a configuration error the run would otherwise
    # discover an epoch in, or never.
    check_early_stop(early_stop, collection, routing)
    check_checkpointer(checkpointer, collection, routing)

    # ── 9, 11. training only ───────────────────────────────────────────────────────
    # Manual mode flattens nothing. `flat` feeds `build_opt_state`, `masks` feed
    # `rebuild_rules`, and `g_accum` feeds the automatic gradient program; the closure owns all of
    # those, so `()` / `nothing` is correct rather than a placeholder. `opt_state` was already
    # built for manual mode in the step-10 block (the closure's router needed its type); it is
    # declared `nothing` here for the automatic path.
    flat = manual ? () : flatten(ps, layout, mesh)
    resolved, total = nothing, nothing
    # Hoisted out of the `if` so the freeze below can see it. `()` for a no-train `Nitro` is
    # correct rather than a placeholder: with no optimizer program to compile there is no rule whose
    # `apply!` could go stale. Manual mode keeps `()` too: the closure's own rules are the user's,
    # and their `apply!` worlds enter the compile key through `worlds_manual` instead.
    chains = ()
    # The per-leaf decay exclusion, resolved ONCE here rather than at every `train!` entry.
    # `no_decay` is a user hook now, so recomputing it later would let a revision take effect on an
    # existing handle and contradict the handle's fixed point. Training-only, because
    # `rebuild_rules` is the only consumer and a serving `Nitro` should not pay G parameter-sized
    # device buffers for something it will never read. Skipped in manual mode, whose rules are the
    # user's own.
    masks = training && !manual ? map(m -> place_replicated(m, mesh), no_decay_masks(e, ps, layout)) : nothing
    if training
        n_batches = length(collection.train)
        check_train_divisibility(n_batches, accum)
        total = max_epochs * div(n_batches, accum)
        if manual
            # `opt` schedules are PATH-BOUND into the user's `opt_state` (a bare key binds
            # to every rule with that field; a nested key is a path into the tree). Resolved and
            # validated here against the opt_state step 9 built, applied by the driver's per-step
            # rebuild (`rebuild_scheduled_rules`) between closure calls. Device-keyed schedules
            # resolve exactly as in the automatic loop.
            resolved = resolve_manual_schedules(
                e, schedules, total;
                opt_state, accessor = ReactantNitro.schedules(e)
            )
        else
            opt_state = build_opt_state(e, flat, layout; anchors, masks, mesh)
            # ── after 9: RE-NORMALIZE THE RESTORED `opt_state`. ────────────────────────
            # Omitting this is the single most dangerous omission this framework could have. Step
            # 9's normalization applies to state the framework CONSTRUCTED; a restore constructs
            # nothing, so without this the resumed run reacquires the frozen-step-counter bug in
            # full, on `resume = :auto`. The symptom is a plausible loss
            # curve and no error: RAdam's `t` stays host, never advances under trace, and the run
            # silently trains at the wrong point of its own bias correction.
            #
            # A weights-only construction deliberately does NOT restore the moments: it means
            # "trained weights, no training this process", and a fresh optimizer is what a new run
            # starting from those weights wants.
            if record !== nothing && !weights_only
                opt_state = to_device_leaf(record.opt_state; mesh)
                for gi in 1:n_groups(layout)
                    assert_device_state(
                        opt_state[gi].state;
                        context = "RESTORED optimizer state for group \
                                                           $(repr(layout.groups[gi]))"
                    )
                end
            end
            chains = map(
                gi -> build_chain(e, layout.groups[gi], resolve_hp(e, layout, gi)),
                ntuple(identity, n_groups(layout))
            )
            # NOTE for the freeze below: `rebuild_rules` reconstructs these every optimizer step with
            # fresh device scalars, but only the VALUES move; the rule TYPES, and therefore the `apply!`
            # methods, are these. So resolving the rule worlds here covers every step of the run.
            resolved = resolve_schedules(
                e, schedules, total;
                chains, accessor = ReactantNitro.schedules(e),
                groups = layout.groups
            )
        end
    end

    # ── 12. the logger, BEFORE the first compile, which can crash ──────────────────
    # Reattachment happens before the run starts and before any metric is logged, so a
    # backend that supports it continues ONE history rather than opening a second experiment. The
    # state is opaque, which is what keeps backend-specific code out of the framework.
    record === nothing || record.logger_state === nothing ||
        reattach!(logger, record.logger_state)
    # The preset name is logged alongside the config it produced, so a result can be grouped by
    # which named recipe it claimed. `nothing` for a run that named none, and omitted rather than
    # logged as "nothing", since a backend showing `preset = nothing` on every ordinary run is noise.
    # The resolved prefetch settings are hyperparameters here, not system details: `workers`
    # derives from `Threads.nthreads(:default)`, so two runs of byte-identical code at different `-t`
    # get a different partition of each epoch into accumulation groups. A reader comparing two runs
    # needs the number, and the process's thread count is already logged separately by the backend.
    pf = training ? prefetch_config(collection.train) :
        (; device_batches = 0, host_batches = 0, workers = 0, ordered = true)
    cfg = config_params(
        e; seed, accum, max_epochs, gradient_clip_norm,
        prefetch_workers = pf.workers, prefetch_device_batches = pf.device_batches,
        prefetch_host_batches = pf.host_batches, prefetch_ordered = pf.ordered
    )
    log_params!(logger, preset === nothing ? cfg : merge(cfg, (; preset)))

    # `step` is RESTORED, never derived. Deriving it as `epoch * opt_steps_per_epoch` is
    # silently wrong whenever steps-per-epoch changed, and restoring it puts every stateless
    # schedule back exactly, since those are pure functions of `(step, total)`.
    step0, epoch0 = (record === nothing || weights_only) ? (0, 0) : (record.step, record.epoch)

    # With the stop reason stored, a resume into a finished run SAYS SO instead of exiting silently.
    # A run that early-stopped and is resumed with `:auto` would immediately re-satisfy its own
    # condition, and one that completed has nothing left to do; either way the surprising thing is a
    # `train!` that returns instantly, so the framework names the cause rather than leaving it to be
    # inferred.
    if !weights_only && record !== nothing && record.stop_reason !== nothing &&
            (record.stop_reason !== :error && epoch0 >= max_epochs)
        @warn "ReactantNitro: the checkpoint being resumed ended at epoch $epoch0 with \
               `stop_reason = $(repr(record.stop_reason))`, and `max_epochs` is $max_epochs, so \
               `train!` will return without running an epoch. Raise `max_epochs` \
               to continue it, or pass `resume = false` to start over."
    end

    nitro = Nitro(
        e, model, ps, st, w0_tree, layout, opt_state, collection, routing, schema, mesh,
        Dict{Any, Any}(), resolved, total, batch_size, logger, nothing, checksums, preset,
        source === nothing ? nothing : String(source),
        # The provenance half: the record is in scope here for the compatibility check above, so the
        # training run's identity costs nothing to keep and cannot be recovered later, since the
        # weights-only restore discards the record on the next line's worth of scope.
        record === nothing ? nothing : record.run_id,
        record === nothing ? nothing : record.run_url,
        String(run_dir), Int(seed), Int(accum), Int(max_epochs), gradient_clip_norm,
        checkpointer, early_stop, probe_accessors(e),
        frozen_dispatch(e, model, ps, st, routing, chains; manual, opt_state),
        (; masks, anchors),
        map(zero, flat), RegisteredMonitor[], Int(step0), Int(epoch0), Starting(),
        false, nothing,
        # No metrics and no elapsed time yet: a fresh handle has run nothing, including one
        # restored from a checkpoint, whose recorded metrics belong to the process that wrote it.
        (;), nothing, nothing,
        weights === nothing ? nothing : weights_origin(weights),
        NamedTuple[]
    )
    # TWO BUILDS from one set of pieces. The text goes to `log_other!`, so the run's record says
    # where every value bound whether or not anyone displayed the handle. The sections are what
    # the handle DISPLAYS, appended to its own bands by `show`, so a reader sees one table instead
    # of a summary and a report that each drew their own boxes. Nothing on the handle returns the
    # text: a String of a framed table is unreadable at a REPL, and `show(nitro)` is the display.
    #
    # NOTHING IS PRINTED HERE, and the binding report used to be `@info`-ed at exactly this point.
    # It stopped being a separate artifact when it became part of `show(nitro)`: a constructor
    # that also printed its own return value would display the handle twice in the REPL, which is
    # where most of these are built. A script that wants it asks, with `display(nitro)` or
    # `@info sprint(show, MIME"text/plain"(), nitro)`, and the run's record has it either way
    # through `log_other!` on the next line.
    nitro.sections = build_binding_sections(nitro)
    log_other!(logger, "binding_report", build_binding_report(nitro))
    run_ref === nothing || (run_ref[] = nitro)
    # The per-run monitor copy, taken HERE and not only at `train!`. `train!` calls it again,
    # which is safe because it is idempotent, and this call is what makes the eval constructions
    # work: a
    # handle that is validated, predicted from, or rendered never reaches `train!`, so without this
    # its module-level monitors are never adopted and NONE of its transitions are published --
    # including the eval compile, which is the longest thing such a handle does and the one a
    # watchdog most needs to be told about.
    adopt_monitors!(nitro)
    # NO `Repl` here, deliberately: a constructor that reported itself idle would hand the card back
    # between the build and the first verb. `Starting` IS declared, at the top of `Nitro` above and
    # through the module-level monitors, because the handle these lines are still assembling is the
    # thing a handle-based publish would need. What this comment used to say, that the window is
    # "covered by whatever the monitor beat when the process started", was true only while the
    # ambient driver re-declared an idle budget every ten seconds. Under a supervisor that declares
    # idle once and lets it decay, an undeclared window is charged against whatever is LEFT of it.
    return nitro
end

"""
    ReactantNitro.check_weights_kwargs(weights, w0; checkpoint, resume) -> nothing

The warm-start keywords, validated before anything is built: `weights` is a `Nitro` or `nothing`
and names the only weight source; `w0` is `:build_model`, `:weights`, or a parameter tree, and
`:weights` needs a `weights` to point at.
"""
function check_weights_kwargs(weights, w0; checkpoint, resume)
    if weights !== nothing
        weights isa Nitro || error(
            "ReactantNitro: `weights` takes a `Nitro` whose `ps` and `st` become this run's \
             initial weights, and was given a `$(typeof(weights))`. For a checkpoint file use \
             `checkpoint = path`."
        )
        (checkpoint !== nothing || resume !== false) && error(
            "ReactantNitro: `weights = ` names a handle to take the initial weights from, and \
             `$(checkpoint !== nothing ? "checkpoint" : "resume")` names a file to restore them \
             from. Two sources for one set of weights; pass one."
        )
    end
    w0 === :build_model || w0 === :weights || w0 isa NamedTuple || error(
        "ReactantNitro: `w0` is `:build_model` (the freshly initialized parameters, the default), \
         `:weights` (the parameters transferred through `weights = `), or a parameter tree to \
         anchor `decay_anchor = :w0` groups to; got `$(repr(w0))`."
    )
    w0 === :weights && weights === nothing && error(
        "ReactantNitro: `w0 = :weights` anchors to the transferred weights, and no `weights = ` \
         was given to transfer them from."
    )
    return nothing
end

# What `show` and the export provenance say about a warm start: the handle it came from, by the
# facts a reader can chase. Taken at construction, because the source handle may train on
# afterwards and this run's weights are the ones it had THEN.
weights_origin(src::Nitro) = (;
    experiment = nameof(typeof(src.e)), epoch = src.epoch, step = src.step,
    run_dir = src.run_dir,
)

"""
    ReactantNitro.check_weights_compatible(source::Nitro, layout::FlatLayout) -> nothing

Refuse a warm start whose parameter tree does not fit this model, with a diff by keypath. Only
the keypaths and sizes are compared: the group a leaf belongs to is this experiment's
`param_group`'s business and may legitimately differ from the source's.
"""
function check_weights_compatible(source::Nitro, layout::FlatLayout)
    return _check_tree_fits(source.layout.permutation, layout.permutation) do diff
        """
        ReactantNitro: `weights = ` names a Nitro($(nameof(typeof(source.e)))) whose parameter tree
        does not match the model this experiment builds:
        $diff
        A warm start copies leaf for leaf, so the two trees must agree on every keypath and size.
        This usually means the architecture fields differ between the two experiments."""
    end
end

# The shared diff: two permutations compared on keypath and size, and `msg(diff)` composes the
# error around the lines. Six lines then an ellipsis, as the resume check prints.
function _check_tree_fits(msg, was, now)
    same = length(was) == length(now) &&
        all(a.keypath == b.keypath && a.size == b.size for (a, b) in zip(was, now))
    same && return nothing
    diff = String[]
    length(was) == length(now) || push!(
        diff, "  the source has $(length(was)) parameter leaves and this model has $(length(now))"
    )
    for (a, b) in zip(was, now)
        (a.keypath == b.keypath && a.size == b.size) && continue
        push!(diff, "  $(a.keypath) size $(a.size) -> $(b.keypath) size $(b.size)")
        length(diff) >= 6 && (push!(diff, "  ..."); break)
    end
    error(msg(join(diff, "\n")))
end

"""
    ReactantNitro.resolve_w0(w0, fresh, ps, e, mesh) -> tree

The anchor `decay_anchor = :w0` groups decay toward, per the `w0` keyword: `fresh` is step 7's
capture, `ps` the parameters after any transfer, and a tree is placed and checked against `ps`.
"""
function resolve_w0(w0, fresh, ps, e, mesh)
    w0 === :build_model && return fresh
    w0 === :weights && return deepcopy(ps)
    tree = place_replicated(from_host(to_host(w0)), mesh)
    _check_tree_fits(build_layout(e, tree).permutation, build_layout(e, ps).permutation) do diff
        """
        ReactantNitro: the `w0 = ` tree does not match this model's parameters:
        $diff
        A `:w0` anchor is subtracted leaf for leaf, so the tree must agree with `ps` on every
        keypath and size."""
    end
    return tree
end

"""
    ReactantNitro.warn_no_concurrency(train_split) -> nothing

One `@warn` at setup when the **training** split ends up with no host-side concurrency, naming what to
implement.

**This is the change that addresses the footgun rather than the symptom.** The regression this whole
mechanism exists for was not a wrong number, it was the absence of one: a model package's comment
asserted the framework handled prefetching, nothing contradicted it, and the run's first ten lines said
nothing either way for weeks while it trained 4.2x slow. A run that has no concurrency is now told so
before it compiles anything.

A **warning rather than an error**, in all three cases. A single producer trains correctly, some
sources genuinely cannot be indexed, and `NoPrefetch` is a legitimate choice; erroring would refuse to
run a correct configuration on a performance opinion. The binding report carries the same fact
without the severity, for the runs where it is expected.

**Three of the six resolved paths say nothing here, and their silence is deliberate.** `:fanout` and
`:fanout_unordered` are the good cases. `:materialized` is a `Vector` whose batches `build_data`
already built: producing one is a pointer load, so there is no host work for N producers to spread,
and warning about a several-fold slowdown that cannot occur would be the false positive that teaches
people to ignore the real one.
"""
function warn_no_concurrency(train_split)
    cfg = prefetch_config(train_split)
    src = prefetch_source(train_split)
    if cfg.path === :single_no_trait
        @warn """
        ReactantNitro: the `train` split runs ONE producer task and leaves \
        $(prefetch_workers(train_split)) workers idle, because `$(typeof(src))` implements \
        neither half of the index-addressable trait.
        Depth is lookahead, not concurrency: a producer slower per batch than the device is per \
        step starves it at any depth. Watch `data_wait_frac` in the per-epoch metrics.
        Implement BOTH `ReactantNitro.batch_at(src, i)`, where `i` is a BATCH index, and \
        `ReactantNitro.begin_epoch!(src)`. With only `batch_at`, every epoch after the first \
        replays the first epoch's plan."""
    elseif cfg.path === :inline
        @warn """
        ReactantNitro: the `train` split is wrapped in `NoPrefetch`, so its entire host data path
        runs INLINE on the training task and overlaps nothing. That is what this
        marker means and it is presumably deliberate; watch `data_wait_frac` in the per-epoch
        metrics to see what it costs."""
    elseif cfg.path === :single
        @warn """
        ReactantNitro: the `train` split runs its host data path in ONE producer task, because
        `workers = 1` was requested explicitly. Depth is lookahead, not
        concurrency; watch `data_wait_frac` in the per-epoch metrics."""
    end
    return nothing
end

"""
    ReactantNitro.check_driver_fields(e) -> nothing

A driver-only value declared as a **`GraphConst`** field is a setup error naming the fix.

The run accessors read a field of their own name through `_field`, which is what lets an experiment
set its own defaults instead of a caller retyping them. That invites declaring `seed` or
`max_epochs` as an ordinary field, and a `GraphConst` field **bakes as a trace-time constant and
enters the compile cache key**. For `seed` that is exactly what the seeding rule forbids, and the
cost is not abstract: a seed sweep is supposed to share one compiled program, and with a
`GraphConst` `seed` field every seed recompiles.

Unmarked is already right: a field that is neither `Device` nor `GraphConst` is [`Host`](@ref) by
default, which is exactly what a driver knob wants. The error says so. `accum` and
`gradient_clip_norm` are deliberately absent from this check: both genuinely bake, so a `GraphConst`
field is correct for them.
"""
const DRIVER_ONLY_FIELDS = (
    :seed, :max_epochs, :run_dir, :n_devs, :logger, :checkpointer,
    :early_stop,
)

function check_driver_fields(e)
    T = typeof(e)
    df, hf = device_fields(T), host_fields(T)
    offenders = [
        f for f in fieldnames(T)
            if f in DRIVER_ONLY_FIELDS && !(f in df) && !(f in hf)
    ]
    isempty(offenders) && return nothing
    error(
        """
        ReactantNitro: $(join(("`$f`" for f in offenders), ", ")) $(
            length(offenders) == 1 ?
                "is declared as a GraphConst field" : "are declared as GraphConst fields"
        ) on $(nameof(T)), and
        $(length(offenders) == 1 ? "it is" : "they are") driver-only.
        A GraphConst field BAKES as a trace-time constant and enters the compile cache key, so
        every distinct value compiles a fresh program. For `seed` that is precisely what the
        seeding rule rules out: a seed sweep is meant to share one compiled program, and a
        `GraphConst` `seed` field recompiles per seed.
        Leave $(length(offenders) == 1 ? "it" : "them") unmarked: an unmarked field is `Host` by
        default, which is what that category is for, or write the marker explicitly as
        $(join(("$f::Host{...}" for f in offenders), ", "))."""
    )
end

"""
    ReactantNitro.config_params(e; kwargs...) -> NamedTuple

The flat hyperparameter table setup step 12 logs, built from the experiment's **actual configured
values** plus the run keywords that are hyperparameters in their own right.

**It logs values, not declaration metadata.** `config_metadata` is the source of the logged
hyperparameter table in the sense that it supplies the field list and the kinds; logging the record
itself would put each field's *declared default* in the table, so a sweep
varying `width` would report the same `width` for every run. That is silent and plausible-looking,
which is the failure mode this framework spends most of its effort on.

Three rules, each with a reason:

  * **`Device` fields are read back to host values.** By step 12 they are device scalars, and a
    logger backend should receive a number rather than a `ConcretePJRTNumber`.
  * **Anything that is not scalar-ish is logged as a descriptor**, not by value. A derived
    `class_weights` vector is a legitimate `Device`, and splatting it into a flat table would
    defeat the logging contract's "keep flat parameter key style".
  * **`Host` fields are included only when scalar-ish.** A `Host` field's whole second job is
    letting an experiment carry a materialized dataset, and a dataset has no business in a
    hyperparameter table.

`seed` is here rather than in `log_other!` because the seed is a hyperparameter.
"""
function config_params(e; kwargs...)
    T = typeof(e)
    hf = host_fields(T)
    names, vals = Symbol[], Any[]
    for f in fieldnames(T)
        v = param_value(getfield(e, f))
        (f in hf && !(v isa Union{Number, Symbol, AbstractString, Bool})) && continue
        push!(names, f)
        push!(vals, v)
    end
    return merge(NamedTuple{Tuple(names)}(Tuple(vals)), NamedTuple(kwargs))
end

param_value(x::Reactant.RNumber) = Reactant.to_number(x)
param_value(x::Union{Number, Symbol, AbstractString, Bool}) = x
param_value(x::AbstractArray) = string(nameof(typeof(x)), size(x))
param_value(x::StrippedHost) = x
param_value(x) = string(nameof(typeof(x)))

"""
    ReactantNitro.default_run_dir(e) -> String

`joinpath("runs", string(nameof(typeof(e))))`, relative to the working directory. It is the
one path concept in the framework: checkpoints, the manifest, and `resume = :auto` all resolve
against it.
"""
default_run_dir(e) = joinpath("runs", string(nameof(typeof(e))))

"""
    ReactantNitro.decay_anchors(e, w0, layout) -> NTuple{G} or nothing

The per-group decay anchor, resolved to a flat device slice per group, or `nothing` for a group
decaying toward zero. `nothing` is a TYPE-level distinction, so a `:zero` group's chain emits no
subtraction ops and carries no parameter-sized zero buffer.

`w0` is the parameters exactly as `build_model` returned them at setup step 7, which is why `:w0`
anchoring is a consequence of initialization rather than an independent knob.
"""
function decay_anchors(e, w0, layout::FlatLayout{G}, mesh = nothing) where {G}
    any(g -> decay_anchor(e, Val(g)) !== :zero, layout.groups) || return nothing
    flat_w0 = flatten(w0, layout, mesh)
    return ntuple(Val(G)) do gi
        a = decay_anchor(e, Val(layout.groups[gi]))
        a === :zero ? nothing :
            a === :w0 ? flat_w0[gi] :
            a isa AbstractArray ? place_replicated(a, mesh) :
            error("ReactantNitro: `decay_anchor(e, Val($(repr(layout.groups[gi]))))` returned \
               `$(repr(a))`. It accepts `:zero`, `:w0`, or an explicit array aligned with that \
               group's slice.")
    end
end

"""
    ReactantNitro.validate_config(e; kwargs...) -> nothing

Setup step 1, cheapest failures first, before any allocation: marker legality, optimizer allowlist
membership, mutually exclusive decay settings, `learning_rate(e) != 0`, and the parameter-group name
checks.
"""
function validate_config(e; accum = 1, gradient_clip_norm = 0.0f0)
    accum >= 1 || error("ReactantNitro: `accum` must be at least 1, and is $accum.")
    check_driver_fields(e)
    lr = learning_rate(e)
    iszero(lr) && error(
        """
        ReactantNitro: `learning_rate(e)` is 0, which is a setup error. Per-group accessors
        define RATIOS against it, so a zero base makes every group's ratio undefined."""
    )
    gradient_clip_norm isa Real || error("ReactantNitro: `gradient_clip_norm` must be a real host \
        number and is a `$(typeof(gradient_clip_norm))`. It is a TRACE-TIME HOST CONSTANT, not a \
        `Device` and not schedulable.")
    # Both residency hooks, at setup rather than at the first metric call, because a typo would
    # otherwise silently select the default and move where the metric runs. This is the cheapest
    # possible failure and step 1 is where those belong.
    check_residency(e, :metrics)
    check_residency(e, :train_metrics)
    return nothing
end

"""
    Nitro(E::Type, preset::Symbol; kwargs...) -> Nitro

The recorded preset form: build the experiment from [`from_preset`](@ref)`(E, preset)` and **record
which named recipe this run claimed**, so the name reaches the checkpoint record and `log_params!`.

```julia
n = Nitro(MyExp, :baseline; run_dir = "runs/baseline")
```

**Keywords are split by which set they belong to**, in one rule: **a `Nitro` keyword goes to
`Nitro`; anything else must be a field of `E` and goes to the recipe.** Anything in neither is an
error naming both valid sets.

```julia
n = Nitro(MyExp, :baseline; max_epochs = 40, aug_rotate_deg = 9.0)
#                              ^ run keyword    ^ field, so it overrides the recipe
```

**A name that is both keeps going to `Nitro`**, which preserves the collision resolution exactly:
`max_epochs`, `seed`, `run_dir`, `accum`, `n_devs`, and `gradient_clip_norm` are struct fields,
legal preset keys, AND run keywords, and here they are unambiguously the **run** keyword because
this is a `Nitro` call. The preset acts at construction and the run keyword acts at run level, which
is the layering. Nothing that resolved to the run level before this split moves.

**The keyword set is derived from this constructor's own declaration**, so it cannot go stale as
keywords are added.

Overriding a field used to require going through `from_preset` and naming the preset a second time:

```julia
n = Nitro(from_preset(MyExp, :baseline; aug_rotate_deg = 9.0); preset = :baseline)
```

**Prefer the form at the top.** That one still works and is what this is shorthand for, but it names
the preset twice and **nothing checks the two agree**, so passing `preset = :variant` there records
a name the experiment was never built from, silently, and the recorded name is the whole point of
recording one. Reported from the first real use, where every override the model needed was a field
rather than a keyword, which made the two-name form the common path instead of the escape hatch.

**Recording a name for a modified experiment is still accepted**, deliberately: a preset's values are
struct fields by the time anything sees them, so per-value provenance would be a claim the type
system cannot back. What the split removes is having to say the name twice to get it.
"""
function Nitro(E::Type, preset::Symbol; kwargs...)
    nitro_kw, field_kw = _split_preset_kwargs(E, preset, kwargs)
    return Nitro(from_preset(E, preset; field_kw...); preset, nitro_kw...)
end

"""
    ReactantNitro._split_preset_kwargs(E, preset, kwargs) -> (nitro_kw, field_kw)

The preset routing rule, extracted so it is testable without constructing a `Nitro`: **a `Nitro`
keyword goes to `Nitro`, anything else must be a field of `E` and goes to the recipe.** A name that
is both goes to `Nitro`, which preserves the collision resolution exactly.
"""
function _split_preset_kwargs(E::Type, preset::Symbol, kwargs)
    nkw, fields = _nitro_keywords(), fieldnames(E)
    nitro_kw, field_kw = Pair{Symbol, Any}[], Pair{Symbol, Any}[]
    saw_data = false
    for (k, v) in pairs(kwargs)
        k === :preset && error(
            "ReactantNitro: `preset` is given positionally to `Nitro(E, name)`, so passing it \
             again as a keyword would be two answers to one question."
        )
        if k === :data
            # Deferred to AFTER the loop, deliberately: the error suggests a `from_preset` call and
            # that suggestion is only correct once every FIELD override has been classified. Erroring
            # here would print a fix that silently drops them.
            saw_data = true
        elseif k in nkw
            push!(nitro_kw, k => v)          # a run keyword, even when it is also a field
        elseif k in fields
            push!(field_kw, k => v)          # a field, so the recipe carries it
        else
            error(
                """
                ReactantNitro: `$k` is neither a `Nitro` keyword nor a field of $(nameof(E)).
                `Nitro` keywords: $(join(map(string, nkw), ", ")).
                $(nameof(E))'s fields: $(join(map(string, fields), ", "))."""
            )
        end
    end
    saw_data && _refuse_data(E, preset, field_kw)
    return nitro_kw, field_kw
end

# REFUSED because it silently trains on the wrong thing. `data` is the one invocation keyword that
# SKIPS a setup step: step 3 runs `build_data(e, ...)` only when no collection was supplied. In
# this form the framework builds `e` itself, so the caller's collection came from a DIFFERENT
# instance, and anything `build_data` populates on the experiment it is handed stays empty on the one
# actually being constructed. `derive` then reads that empty state and returns something shaped like
# an answer. Measured on the first real preset table: a 10-class weighted loss over a zero-length
# class-weight vector, with no error.
#
# `checkpoint`, `resume`, and `run_ref` are NOT refused: they feed the record lookup and the caller's
# output channel, and skip nothing.
#
# THE SUGGESTION CARRIES THE FIELD OVERRIDES, which is the whole reason this is a separate function.
# Anyone reaching the recorded form is disproportionately likely to HAVE field overrides, since
# keyword splitting is the main reason to use it, so a suggestion that dropped them would be
# copy-pasteable and wrong: a dropped `backbone_kind = :stub` builds the real backbone instead, which
# is a multi-minute compile of the wrong model rather than an obvious failure.
@noinline function _refuse_data(E::Type, preset::Symbol, field_kw)
    ov = isempty(field_kw) ? "" :
        "; " * join(("$k = $(_short_repr(v))" for (k, v) in field_kw), ", ")
    return error(
        """
        ReactantNitro: `Nitro(E, preset; data = ...)` is refused.

        Supplying `data` skips `build_data` at setup step 3, but this form builds the experiment for
        you, so your collection came from a DIFFERENT instance than the one being constructed.
        Anything `build_data` populates on the experiment stays empty here, and `derive` reads it
        without complaining.

        If you are supplying the data you own the experiment, so build it explicitly:

            Nitro(from_preset($(nameof(E)), $(repr(preset))$ov); data = ..., preset = $(repr(preset)))

        which is the one place the two-name form is the right spelling rather than a leftover."""
    )
end

# Field overrides are hyperparameters in the ordinary case, but nothing stops one being an array, and
# a suggestion is useless if it is a screenful.
_short_repr(v) = (s = repr(v); length(s) <= 40 ? s : "<$(nameof(typeof(v)))>")

# DERIVED from the constructor's own declaration, never hardcoded. A literal list would be a second
# place to keep in agreement with the signature above, and going stale silently is the shape of
# defect this framework has paid for repeatedly. `Base.kwarg_decl` is the same mechanism batch
# routing uses to route hooks by their declared keywords.
#
# READ OFF `_build_nitro`, NOT `Nitro`. The public `Nitro(e; kwargs...)` is a thin wrapper that moves
# the build off the interactive thread, so its OWN declaration is a bare sink and `Base.kwarg_decl`
# reports exactly `[KWARG_SINK]` for it. The keyword defaults deliberately live on `_build_nitro`,
# the function that reads `e`, so the accessor defaults keep a single home; that makes
# `_build_nitro` the only place the run keywords are actually named. While this pointed at `Nitro`
# the set was empty of real names, so `_split_preset_kwargs` classified EVERY keyword as unknown and
# the whole of the preset routing failed: `Nitro(E, :name; run_dir = ...)` reported `run_dir` as
# neither a `Nitro` keyword nor a field of `E`.
#
# The sink check is the guard the paragraph above only promised. Deriving the list stops it going
# stale as keywords are ADDED, but it cannot notice the build body being split out from under it a
# second time; this can, and says so at the first `Nitro(E, name)` call rather than misrouting.
function _nitro_keywords()
    kw = Base.kwarg_decl(which(_build_nitro, Tuple{Any}))
    has_sink(kw) && error(
        "ReactantNitro: `_nitro_keywords` resolved to a method whose declaration is a keyword sink, \
         so the preset keyword routing has no names to route by. It must name the method \
         that DECLARES the run keywords, which is `_build_nitro`, not a wrapper forwarding them."
    )
    return kw
end

"""
    ReactantNitro.setup_devices(n_devs) -> mesh

Setup step 4.5. Build a `Reactant.Sharding.Mesh` over the requested local devices, sharding the
batch on the `:data` axis and replicating parameters, optimizer state, and layer state.
**`n_devs == 1` skips the mesh entirely.**

Single node, one or more devices, one process, is the whole of the distribution scope here.
Multi-node and multi-process are out, and the distribution stubs at the bottom of this file are how
the interfaces stay non-breaking if that changes.
**There is no MPI and no NCCL here**: XLA distributes by partitioning one compiled program over a
mesh, which is a different mechanism from traditional data-parallel collectives and does not compose
with them.

**`n_devs == 1` skips the mesh entirely**, and that is not merely an optimization: a one-device
`Sharding.Mesh` is a no-op that Reactant warns about ("single device mesh is not well supported").

**The mesh is 1-D with a single `:data` axis**, because the whole of the parallelism here is data
parallel. The batch is sharded along it and everything else is replicated (see
[`place_replicated`](@ref) and [`place_batch`](@ref)).

**`n_devs` counts VISIBLE devices**, so `CUDA_VISIBLE_DEVICES` is the supported way to restrict a
run, and asking for more than are visible is a setup error rather than a silent truncation.
"""
function setup_devices(n_devs::Integer)
    n_devs >= 1 || error("ReactantNitro: `n_devs` must be at least 1, and is $n_devs.")
    n_devs == 1 && return nothing            # one device skips the mesh entirely
    devs = Reactant.devices()
    n_devs <= length(devs) || error(
        """
        ReactantNitro: `n_devs = $n_devs` but Reactant sees $(length(devs)) device(s).
        `n_devs` counts LOCAL, VISIBLE devices and the default is all of them, so this is either a
        hand-set value that is too large or a `CUDA_VISIBLE_DEVICES` narrower than you meant.
        Note the backend matters: with no GPU visible, Reactant reports a single CPU device."""
    )
    return Reactant.Sharding.Mesh(reshape(collect(devs)[1:n_devs], n_devs), (:data,))
end

# ── Process-level accelerator configuration ────────────────────────────────────────────
#
# The one code path for "choose the accelerator for this session". A REPL session calls
# `setup_devices!` directly; the Kaimon tool `nitro_setup` is a thin wrapper over the same
# function, so the two workflows cannot drift apart. The state it writes is the `_PINNED_N_DEVS`
# pin in Interface.jl, read by the `n_devs` accessor.

"""
    ReactantNitro.setup_devices!(; backend = nothing, n_devs = nothing) -> NamedTuple

Configure this process's accelerator for every subsequent run and report what is now in effect,
as `(; backend, visible, n_devs, pinned)`. **The Kaimon tool `nitro_setup` is exactly this
function**, so a REPL session and a Kaimon-hosted session configure identically.

Calling it is **optional**. A session that never calls it runs on Reactant's default backend with
`n_devs` = every visible device, `length(Reactant.devices())`, which is what a CPU machine (one
visible device) gets automatically. This function exists to be explicit and to fail fast.

`backend` passes through to `Reactant.set_default_backend` and names a Reactant backend:
`"cpu"`, `"gpu"` (whichever of CUDA/ROCm is available), `"cuda"`, `"rocm"`, `"tpu"`. Omit it to
leave Reactant's default: the highest-priority working backend, chosen once per process at the
first device access (GPU where one is visible, else CPU).

`n_devs` pins the device count for this process, validated eagerly against the VISIBLE devices
through the same check a `Nitro` construction runs, one session earlier. **The pin replaces
the default and beats an experiment's declared `n_devs` field**; an explicit `n_devs` keyword on
`Nitro`/`train!` still wins for that one run.

**One process, one XLA.** A Julia process initializes its XLA/PJRT client once; `n_devs` never
creates clients or processes. It slices the already-visible device set to build the mesh, so to
restrict which GPUs a session sees, set `CUDA_VISIBLE_DEVICES` **before starting the process**:
visibility is fixed at the first client access. Asking for more devices than are visible is an
error, and the error names `CUDA_VISIBLE_DEVICES`.

The batch size is GLOBAL and gets split across the mesh, so adding devices buys throughput and
does not change the effective batch.

Calling with no arguments changes nothing and reports the current configuration: the backend in
use, how many devices are visible, and the `n_devs` the next run will use.

```julia
ReactantNitro.setup_devices!()                      # report
ReactantNitro.setup_devices!(backend = "cpu")       # run everything on CPU
ReactantNitro.setup_devices!(backend = "cuda", n_devs = 2)  # two of the visible CUDA devices
```
"""
function setup_devices!(; backend = nothing, n_devs = nothing)
    # 1. Backend: pass through to Reactant's process-global default, with a friendlier error
    #    than the KeyError an unknown name would otherwise surface.
    if backend !== nothing
        backend isa AbstractString || throw(
            ArgumentError(
                "ReactantNitro: `backend` must be a backend name string, got $(repr(backend))."
            )
        )
        try
            Reactant.set_default_backend(backend)
        catch err
            throw(
                ArgumentError(
                    "ReactantNitro: `set_default_backend(\"$(backend)\")` failed: " *
                        sprint(showerror, err) *
                        " Known backend names: \"cpu\", \"gpu\", \"cuda\", \"rocm\", \"tpu\" " *
                        "(and \"tt\" where supported)."
                )
            )
        end
    end
    # 2. Device count: validate through `setup_devices`, so the n_devs contract has a single
    #    authority (≥ 1, ≤ visible, and an error that names CUDA_VISIBLE_DEVICES), then pin.
    if n_devs !== nothing
        setup_devices(n_devs)
        _PINNED_N_DEVS[] = n_devs
    end
    # 3. Report what is now in effect.
    devs = Reactant.devices()
    pinned = _PINNED_N_DEVS[]
    return (;
        backend = Reactant.XLA.platform_name(Reactant.XLA.default_backend()),
        visible = length(devs),
        n_devs = pinned === nothing ? length(devs) : pinned,
        pinned = pinned !== nothing,
    )
end

"""
    ReactantNitro.place_replicated(x, mesh) -> x_device
    ReactantNitro.place_batch(x, mesh) -> x_device

The two placements. **Everything is replicated except the batch**, which is sharded on its last
dimension along the mesh's `:data` axis.

`mesh === nothing` is the single-device path and both fall through to a bare `to_rarray`, so every
call site reads the same on one device and on several.

**Why the last dimension.** The batch contract puts the sample axis last, which is also Lux's
convention, so the batch dimension is `ndims(x)` for every routed field. A field whose sample axis
is somewhere else cannot be sharded correctly by this rule, which is one more reason routing
transfers only the fields a hook actually declares.

**What replication buys.** Parameters, optimizer state, layer state, the decay masks and anchors, the
accumulator, and the scalar carriers all exist identically on every device, so the compiled program
partitions cleanly over the batch axis alone. That is the whole of data parallelism: the GPU count is
**throughput, not a batch multiplier**. A run at `batch_size = 32` on four devices puts 8 samples on
each, and its numerics are those of a 32-sample batch, not a 128-sample one.
"""
place_replicated(x, ::Nothing; kwargs...) = Reactant.to_rarray(x; kwargs...)
place_replicated(x, mesh; kwargs...) =
    Reactant.to_rarray(x; sharding = Reactant.Sharding.Replicated(mesh), kwargs...)

place_batch(x, ::Nothing) = Reactant.to_rarray(x)
place_batch(x, mesh) = Reactant.to_rarray(
    x; sharding = Reactant.Sharding.NamedSharding(mesh, _batch_spec(ndims(x)))
)

# The batch axis is `:data`; every other axis is replicated. `nothing` in a partition spec is
# Reactant's spelling for "not partitioned along any mesh axis".
@inline _batch_spec(n::Integer) = ntuple(i -> i == n ? :data : nothing, n)

"""
    ReactantNitro.check_shardable_batch(batch_size, n_devs) -> nothing

The sharding divisibility requirement, checked at setup where it is cheap.

A batch sharded on the `:data` axis has to divide evenly across the mesh. XLA will pad or refuse
otherwise, and neither failure names the batch size, so the framework checks first and says which
two numbers disagree. This is the multi-device sibling of the `length(train) % accum == 0` check.
"""
function check_shardable_batch(batch_size::Integer, n_devs::Integer)
    n_devs == 1 && return nothing
    batch_size % n_devs == 0 || error(
        """
        ReactantNitro: batch size $batch_size does not divide across $n_devs devices.
        The batch is sharded on the `:data` mesh axis, so each device takes
        $batch_size / $n_devs = $(batch_size / n_devs) samples, which has to be a whole number.
        Pick a batch size divisible by $n_devs, or set `n_devs` to a divisor of $batch_size.
        Note the batch size is the GLOBAL one: the devices split it rather than each taking a full
        copy, so adding devices buys throughput and does not change the effective batch."""
    )
    return nothing
end

"""
    ReactantNitro.to_device_config(e) -> e_converted

Setup step 5: walk the experiment, convert every [`Device`](@ref) leaf through
[`to_device`](@ref), and rebuild. **This changes the experiment's type**, which is fine because it
precedes any compile.

Device conversion happens in exactly three places and only one of them recurs: here at setup,
per optimizer step for the **scheduled** entries only, and nowhere else. **Nothing converts
inside the traced step.** A setup-fixed entry, scalar or 10,000-element vector, is converted once and
reused by reference; because a constant is a `Number` rather than a `Callable`, constants are
setup-fixed by construction, so the per-step transfer count equals the number of quantities actually
being varied.
"""
function to_device_config(e, mesh = nothing)
    T = typeof(e)
    df = device_fields(T)
    isempty(df) && return e
    return Base.typename(T).wrapper(
        map(fieldnames(T)) do f
            v = getfield(e, f)
            f in df ? to_device(v; mesh) : v
        end...
    )
end

"""
    device_value(nitro, name) -> value

The **host** value of a [`Device`](@ref) field of the live experiment. `nitro.e` holds device
values after setup step 5, so reading a field directly hands back a `ConcretePJRTNumber` or
`ConcretePJRTArray`; this transfers it back. The counterpart to [`set_device!`](@ref), and the
supported way for host-side code to read a value it is also writing.

Errors on a field that is not `Device`, because a [`Host`](@ref) or [`GraphConst`](@ref) field is
already a host value and `getfield` is the right way to read it.
"""
function device_value(nitro::Nitro, name::Symbol)
    e = nitro.e
    _check_device(e, name, "device_value")
    return to_host(getfield(e, name))
end

"""
    set_device!(nitro, name, value) -> nitro
    set_device!(nitro; name = value, ...) -> nitro

Write a new value into a [`Device`](@ref) field of a live handle. **Provably does not recompile**,
which is the whole point: it is how a device sweep or a changed inference threshold reuses the
programs already in the compile cache instead of rebuilding a `Nitro`.

The guarantee is structural rather than hopeful. The cache key skips `device_fields` when hashing,
and [`graphconst_field_hash`](@ref) seeds on `Base.typename(T)` rather than `hash(T)` precisely so
that re-parameterizing an experiment for device residency does not move the hash, so a new device
value in a `Device` slot leaves every key component untouched. This function asserts that before it
commits,
and the assertion is not decoration: `typeof(compile_view(e))` is `args[1]`'s type at every trace
site, so a value whose device type differs by even an element type would move the key and buy a
recompile silently.

Two things are refused, both because they would break that guarantee rather than out of caution:

  * **A field that is not `Device`.** A [`GraphConst`](@ref) field bakes as a trace-time constant and
    is hashed, so changing it genuinely is a different program; a `Host` field reaches no trace at
    all. The error names which case it is and points at the rebuild, which is cheap because the
    cache is module-level and a fresh `Nitro` hits every entry the change does not invalidate.
  * **A different type or size.** Element type would move `typeof(ev)` and therefore the key. Size
    would not, since `_shape` of a struct is `nothing`, which is worse: the key would still match
    and the shape mismatch would surface inside XLA at call time, exactly the gap the key lists
    shapes for.

**A scheduled field is also refused**, because the per-optimizer-step rebuild rewrites the
scheduled entries from the schedule, so a value written here would survive until the next optimizer
step and no longer. Silently losing a write one step later is worse than refusing it.

```julia
n = Nitro(e; checkpoint = "runs/MyExp/best.jld2")   # weights-only, skips `derive`
for thr in (0.3f0, 0.5f0, 0.7f0)
    set_device!(n; threshold = thr)
    out = predict(n, batch)                         # no compile, any iteration
end
```

See also [`device_value`](@ref) to read one back, and [`Device`](@ref) for what the marker means.
"""
function set_device!(nitro::Nitro, name::Symbol, value)
    e = nitro.e
    T = typeof(e)
    _check_device(e, name, "set_device!")

    # Refused because the schedule OWNS the field: `step_experiment` rebuilds it at the top of every
    # optimizer step, so a write here would survive exactly until the next step and
    # then vanish, which is worse than a refusal because the value visibly took and then silently
    # did not.
    sched = nitro.schedules
    if sched !== nothing && haskey(sched.device, name)
        error("ReactantNitro: `$name` is scheduled, so the per-step rebuild owns \
            its value and a write here would be overwritten once that rebuild runs. Drop `$name` \
            from `schedules` to set it directly, or change the schedule instead.")
    end

    old = getfield(e, name)
    host_old = to_host(old)
    H = typeof(host_old)
    host_new = try
        convert(H, value)
    catch
        error("ReactantNitro: `set_device!` cannot convert a `$(typeof(value))` to `$H`, which is \
            `$name`'s host type. The type must match exactly: a `Device` field carries its own type \
            PARAMETER, so `typeof(compile_view(e))` is part of the compile key through `args[1]`, \
            and a changed element type would recompile rather than reuse the program.")
    end
    if host_old isa AbstractArray && size(host_new) != size(host_old)
        error("ReactantNitro: `set_device!` was handed a value of size $(size(host_new)) for \
            `$name`, which holds $(size(host_old)). Size is NOT in the compile key, so this \
            would not recompile: it would hit the existing program and fail inside XLA on the \
            shape mismatch. \
            Rebuild the `Nitro` to change a Device's shape.")
    end

    ev_before = compile_view(e)
    e_new = Base.typename(T).wrapper(
        map(
            f -> f === name ? to_device(host_new; mesh = nitro.mesh) :
                getfield(e, f), fieldnames(T)
        )...
    )
    ev_after = compile_view(e_new)
    # The cache key's promise, checked rather than trusted. Both halves have failed in review: the
    # type through a widened element type, the hash through a `Device` that was not declared in
    # `device_fields`.
    typeof(ev_after) === typeof(ev_before) || error("ReactantNitro internal: `set_device!($name)` \
        moved `typeof(compile_view(e))` from $(typeof(ev_before)) to $(typeof(ev_after)), which is \
        `args[1]`'s type at every trace site and therefore part of the compile key. Refusing \
        rather than recompiling silently; please report this.")
    graphconst_field_hash(ev_after) == graphconst_field_hash(ev_before) ||
        error("ReactantNitro internal: `set_device!($name)` moved `graphconst_field_hash`, so \
            `$name` is being hashed into the compile key despite `device_fields($T)` listing it. \
            Refusing rather than recompiling silently; please report this.")

    nitro.e = e_new
    return nitro
end

function set_device!(nitro::Nitro; kwargs...)
    isempty(kwargs) && error("ReactantNitro: `set_device!(nitro; name = value, ...)` needs at \
        least one field to set.")
    for (k, v) in pairs(kwargs)
        set_device!(nitro, k, v)
    end
    return nitro
end

# Shared by the getter and the setter so the two cannot disagree about what is settable.
function _check_device(e, name::Symbol, fn::AbstractString)
    T = typeof(e)
    hasfield(T, name) || error("ReactantNitro: `$name` is not a field of $(nameof(T)). Its fields \
        are $(fieldnames(T)).")
    name in device_fields(T) && return nothing
    if name in host_fields(T)
        error("ReactantNitro: `$name` is a `Host` field, which `compile_view` strips and no trace \
            ever sees, so `$fn` would change nothing about any compiled program. Host fields \
            are read host-side from the stored experiment; rebuild the `Nitro` to change one.")
    end
    error("ReactantNitro: `$name` is a `GraphConst` field, which bakes as a trace-time constant and is \
        hashed into the compile key, so changing it IS a different program and `$fn` cannot do it \
        without recompiling. Rebuild instead: `Nitro(e2; data = nitro.data)`. That is cheaper than \
        it sounds, because the compile cache is MODULE-LEVEL, so the new handle reuses every \
        compiled program the change does not invalidate. Mark `$name` `Device` if you meant it to \
        be settable at runtime.")
end

"""
    ReactantNitro.CONFIG_REPORT

Whether the entry points print [`fixed_config_report`](@ref). `true` by default;
`ReactantNitro.set_config_report!(false)` silences it for a driver that calls `validate` in a loop and
does not want the banner each time.

Not a `Nitro` keyword and not an `@experiment` field, because it is a property of the session's
console rather than of the run, and because `train!(nitro)` deliberately takes no keywords.
"""
const CONFIG_REPORT = Ref(true)

"Set whether the entry points print [`fixed_config_report`](@ref). Returns the previous value."
function set_config_report!(on::Bool)
    prev = CONFIG_REPORT[]
    CONFIG_REPORT[] = on
    return prev
end

# The scalar run knobs whose accessors are PURE and therefore safe to re-probe for the divergence
# check. `logger`, `checkpointer`, `early_stop`, and `schedules` are deliberately absent: their
# accessors are CONSTRUCTORS, and README's logger section states the framework "calls this exactly
# ONCE, at setup, which is what makes it safe for an accessor to have a side effect: opening a file
# here, or registering a run with a hosted tracker". Probing one would fire that side effect on every
# entry-point call, and comparing the result would false-positive anyway, since `EarlyStopping` and
# `TopKCheckpointer` are mutable and compare by identity. Consequence, documented rather than fixed:
# a revised `early_stop`/`checkpointer`/`logger` CONSTRUCTOR is not detected here.
const _PROBED = (:seed, :accum, :max_epochs, :gradient_clip_norm, :run_dir)

"""
    ReactantNitro.frozen_dispatch(ev, model, ps, st, routing, chains) -> NamedTuple

Every component of the compile-cache key that comes from **live method dispatch**, resolved once,
here.

Before this, `hook_worlds` and both `metrics_residency` reads ran at each entry-point call, so an
existing handle could observe a redefinition and silently recompile between two `train!` calls. An
earlier change removed the two entries that could only fire spuriously; these are the ones that
could fire *correctly*, and freezing them is the other half of the same argument. With them stored,
every component of the key derives from stored state, and **the programs a `Nitro` uses are
determined at construction**. What replaces the recompile is [`fixed_config_report`](@ref)'s hook
section, which says the handle is stale and names the rebuild.

**Three world tuples rather than one, because the programs have different dependency sets.**

| Entry | Covers | Consumed by |
| --- | --- | --- |
| `worlds_train` | the hooks, resolved against a **train-mode** `st` | `grad_program` |
| `worlds_opt` | the same, plus every rule's `apply!` | `opt_program` |
| `worlds_eval` | the hooks, resolved against an **eval-mode** `st` | `fwd_program`, `eval_metric_program` |
| `graphconst_hash` | the experiment-derived key component | every `compile_cached`, as `gc_hash` |

**The split is what lets the rules in at all.** `chains` was a parameter of [`hook_worlds`](@ref)
that no call site ever passed, so `_rules_of`'s loop never ran and a revised custom
`Optimisers.apply!` silently reused the old optimizer program, which the cache contract and that
docstring both claimed was covered. Wiring it needs a single correct chain set, and construction is
the only place one exists: `rebuild_rules` reconstructs chains per step, so there is nothing to hand
a per-call resolution. Putting the rules in `worlds_opt` **only** keeps a rule edit off the 495.6 s
gradient program, since `opt_program` traces no user hook and a hook edit cannot change it either.
Sharing one tuple would re-buy exactly the spurious expensive recompile that was removed earlier.

**The two `st` modes are not pedantry.** `forward`'s world is resolved against `typeof(st)`, and
`train!` used to resolve it against the un-moded `nitro.st` while compiling with
`Lux.trainmode(st)`. For a model whose train and eval state types differ and which dispatches
`forward` on them, that resolved the wrong signature. Freezing forces both modes to be named.
"""
function frozen_dispatch(e, model, ps, st, routing, chains; manual = false, opt_state = nothing)
    # `ev` for the worlds, because that is what every trace site passes and what `hook_worlds`
    # requires; the FULL `e` for the residencies, because that is what the call sites this replaces
    # passed. The two agree for any experiment declaring `metrics_residency` on its own type, since
    # `compile_view` rebuilds the same struct with different parameters, but reproducing each read
    # exactly is how this stays a pure relocation rather than a behavior change smuggled in with one.
    ev = compile_view(e)
    st_train, st_eval = Lux.trainmode(st), Lux.testmode(st)
    return (;
        worlds_train = hook_worlds(ev; model, ps, st = st_train),
        worlds_opt = hook_worlds(ev; model, ps, st = st_train, chains),
        worlds_eval = hook_worlds(ev; model, ps, st = st_eval),
        # The two metric residencies, read once for the same reason. An experiment with no
        # `train_metrics` method stays on the `:device` program whatever the accessor says,
        # since there is nothing to move to the host when the hook does not exist.
        tm_residency = routing === nothing || routing.train_metrics === nothing ? :device :
            check_residency(e, :train_metrics),
        metrics_residency = check_residency(e, :metrics),
        # The experiment-derived key component, resolved ONCE here, which is what `cache_key`'s
        # docstring has always claimed and what the implementation did not do: it recomputed the
        # hash on every `compile_cached`, and that runs per micro-batch.
        #
        # Safe for a whole run, and the invariant is already enforced rather than newly assumed.
        # `ev` is not constant across `train!`: the adaptive path rebuilds it when a `Device`
        # field updates. But `set_device!` ERRORS if that rebuild moves `graphconst_field_hash`, so
        # "the config hash cannot change within a run" is a property this framework already
        # guarantees, and hoisting the value is reading that guarantee rather than trusting it.
        #
        # ONLY the experiment half is hoisted. Argument types and SHAPES are still computed on every
        # call, and must be: Reactant's `@generated` guard covers types but not shapes, which is
        # exactly why a ragged batch passes the guard and then fails inside XLA. Caching past the
        # shape check would reintroduce that.
        #
        # Measured: this is worth about 0.04 ms per epoch at 550 lookups and is NOT a
        # performance change. It earns its place by making the code match its own contract, and by
        # bounding a real footgun: a `GraphConst` holding a 1e6-element array hashes in 624 us, so
        # rehashing it per micro-batch costs about 343 ms per epoch, and now it is hashed once.
        graphconst_hash = graphconst_field_hash(ev),
        # Manual mode's frozen-dispatch entries. `manual` is the resolved mode itself, so
        # flipping `manual_training(e)` on an existing handle is reported by `stale_hooks` rather
        # than applied. `worlds_manual` is the closure program's key component: the hooks (a manual
        # closure may call them) plus the closure's OWN method world, so editing `train_step`
        # invalidates the cache on a new `Nitro`. `worlds_setup` is staleness-only: `setup_optimizers`
        # is not traced by the closure, so its world does NOT belong in the compile key (a
        # redefinition would otherwise re-buy the closure compile for a byte-identical program), but
        # redefining it on an existing handle should still be reported rather than silently inert.
        manual = manual,
        worlds_manual = manual ? (
                hook_worlds(ev; model, ps, st = st_train)...,
                method_world(train_step, Tuple{typeof(ev), typeof(model), typeof(ps), typeof(opt_state), typeof(st)}),
            ) : (),
        worlds_setup = manual ? (method_world(setup_optimizers, Tuple{typeof(ev), typeof(model), typeof(ps), typeof(st), Any}),) : (),
    )
end

"""
    ReactantNitro.stale_hooks(nitro) -> Vector{Symbol}

Which frozen world tuples no longer match live dispatch, as the names
[`fixed_config_report`](@ref) prints. Empty is the normal case.

This is the **whole** safety story for freezing the dispatch state, and it is why freezing is not
simply a regression. Method worlds went into the key because "editing `forward` and calling
`train!` again would hit the cache and silently run the old program", and the word carrying the
weight is *silently*. Freezing brings the stale program back and this removes the silence: the
handle reports that its hooks were redefined and names the rebuild, at the cost of a few `which`
calls per entry-point call, against a step that costs orders of magnitude more.

A **new** `Nitro` is unaffected and always resolves current dispatch, which is what keeps the cache
doing its actual job: letting a fresh, genuinely compatible handle skip the compile.
"""
function stale_hooks(nitro::Nitro)
    f = nitro.frozen
    f === nothing && return Symbol[]
    live = frozen_dispatch(
        nitro.e, nitro.model, nitro.ps, nitro.st, nitro.routing,
        frozen_chains(nitro); manual = get(f, :manual, false),
        opt_state = nitro.opt_state
    )
    return Symbol[k for k in keys(f) if getproperty(live, k) != getproperty(f, k)]
end

# The chains whose `apply!` worlds went into `worlds_opt`. Rebuilt rather than stored, since only
# their TYPES matter here and `rebuild_rules` preserves those; storing the construction-time chains
# would pin their device scalars alive for the run's whole life for no benefit.
# Manual mode has no framework chains: the user's rules are never rebuilt by the framework, so
# `()` is correct, and walking the user's opt_state tree for leaves would fail on its nesting.
frozen_chains(nitro::Nitro) =
    get(nitro.frozen, :manual, false) ? () :
    nitro.opt_state === nothing ? () : map(l -> l.rule, nitro.opt_state)

"""
    ReactantNitro.probe_accessors(e) -> NamedTuple

What each of `_PROBED`'s accessors returns right now. Captured once at construction into
`nitro.accessors_at_setup` so [`fixed_config_report`](@ref) can compare accessor-THEN against
accessor-NOW, which is the only comparison that separates a redefinition from a keyword override.
An accessor that throws records `nothing` and is simply not probed later.
"""
probe_accessors(e) = NamedTuple{_PROBED}(
    map(_PROBED) do f
        try
            getfield(ReactantNitro, f)(e)
        catch
            nothing
        end
    end
)

"""
    ReactantNitro.fixed_config_report(nitro; entry = :train) -> String

**What has been redefined since this handle froze it**, and therefore is not in effect. Empty when
nothing has, which is the usual case and is why the entry points are silent about it.

It does NOT list the handle's values. `show(nitro)` is the view of what a handle holds, and the
binding report is the view of where each configured value came from; this is the third question,
and the only one whose answer nothing else can supply. Overlapping the three was how the top of
every `train!` came to repeat the seed and the run directory twice before the run started.

Setup resolves the ten accessor-defaulted keywords once, into the `Nitro`'s own fields, and every
read after that goes to the field. So revising `accum(::MyExp)` or `max_epochs(::MyExp)` and calling
`train!` on an existing handle changes nothing, and it once also triggered a full recompile through
[`hook_worlds`](@ref), which made the wasted compile read as confirmation that the edit had landed.
Removing that entry fixed the cost and made the silence total, so this report is what tells the user
instead.

**The handle always wins.** This never applies a divergence, because `accum` and `max_epochs` are
entangled with the schedule horizon `total = max_epochs * div(n_batches, accum)` and with
`check_train_divisibility`, so applying one in isolation would leave the schedules resolved against
a horizon that no longer exists. The fix is a rebuild, which is cheap: the cache is module-level, so
`Nitro(e; data = nitro.data, max_epochs = 60)` reuses every compiled program.

`entry` selects the relevant set, since `validate` and `predict` read neither `accum` nor the clip.
Only pure scalar accessors are probed; see `_PROBED` for which and why.
"""
function fixed_config_report(nitro::Nitro; entry::Symbol = :train)
    e = nitro.e
    io = IOBuffer()
    # DRIFT ONLY. This used to open with a table of the handle's fixed values, which is now what
    # `show(nitro)` is for: seed, accum, max_epochs, total, clip and run_dir were all in both, and
    # printing them again at the top of every `train!`, `validate` and `predict` was noise a reader
    # learned to scroll past. What is left is the half nothing else can tell you: which accessors
    # and hooks were redefined AFTER this handle froze them, and are therefore not in effect.
    #
    # Empty when nothing drifted, which is the common case, and `report_fixed_config` prints
    # nothing at all then. Silence is the correct output for "everything is as you left it".
    then = nitro.accessors_at_setup
    for f in _PROBED
        entry === :train || f in (:seed, :run_dir) || continue
        was = then === nothing ? nothing : get(then, f, nothing)
        was === nothing && continue                  # not probeable at setup, so not comparable now
        now = try
            getfield(ReactantNitro, f)(e)
        catch
            continue          # an accessor that throws is the user's business, not this report's
        end
        # accessor-THEN vs accessor-NOW. Comparing against the STORED field instead would flag every
        # keyword override, since a keyword is indistinguishable from its accessor default by the time
        # the constructor body runs (see `clip_source`, and `accessors_at_setup`'s comment).
        now == was && continue
        # One string, deliberately: a `\` continuation keeps the next line's indentation, which is
        # tolerable in an error message and looks broken in a formatted report.
        println(
            io, "  ! `$f(e)` was redefined: it returned $was at construction and returns $now ",
            "now. This handle still uses $(getfield(nitro, f)); rebuild to pick up $now."
        )
    end
    # The other half of the same job. The accessor lines above cover CONFIGURATION that this handle
    # fixed; this covers CODE. Both exist because the handle is frozen and neither the stale value
    # nor the stale program may be silent.
    stale = stale_hooks(nitro)
    isempty(stale) || println(
        io, "  ! hooks were redefined after this `Nitro` was built ",
        "($(join(stale, ", "))). It will keep running the programs it was ",
        "built with; rebuild the `Nitro` to pick up the new code. The compile ",
        "cache is module-level, so a rebuild recompiles only what actually ",
        "changed."
    )
    # The third half: the world-closure guard poisons module entries so NEW handles
    # recompile, and this handle is told when its OWN programs were among them. `programs` is the
    # handle-local thunk store (the frozen contract made literal), so its keys ARE this handle's
    # programs, and an intersection with the poisoned set is precise: only a handle that actually
    # holds a now-poisoned program hears about it.
    staleness = LAST_WORLD_STALENESS[]
    mine = isempty(staleness.poisoned) ? nothing :
        findfirst(p -> haskey(nitro.programs, p), staleness.poisoned)
    mine === nothing || println(
        io, "  ! this handle's compiled programs had dependency methods redefined ",
        "($(join(unique(staleness.drifted), ", "))). This handle keeps its programs, now stale; ",
        "rebuild the `Nitro` to recompile against current dispatch. New `Nitro`s already ",
        "recompile."
    )
    body = String(take!(io))
    isempty(body) && return ""
    # The title is built here rather than up front so that "nothing drifted" can be the empty
    # string: a header with no findings under it reads as a finding nobody wrote down.
    title = "ReactantNitro: $(nameof(typeof(e))) has values fixed at construction that have \
             since been redefined. This handle keeps what it froze; rebuild the `Nitro` to pick \
             up the new ones, and see `show(nitro)` for what it is currently holding."
    return title * "\n" * body
end

# Printed rather than `@info`-ed: it is a report, not a diagnostic, and `@info`'s gutter would put
# a `|` down the left of every line of it. And printed only when there is something to say, which
# is why this is not simply `print`. Unlike the binding report, this one has no display of its own
# to ride on: `show(nitro)` states what the handle IS HOLDING, and a value redefined since it
# froze is by definition not that.
function report_fixed_config(nitro::Nitro, entry::Symbol)
    CONFIG_REPORT[] || return nothing
    txt = fixed_config_report(nitro; entry)
    isempty(txt) || print(stdout, txt)
    return nothing
end

"""
    ReactantNitro.merge_derived(e, derived) -> e

Setup step 4: merge [`derive`](@ref)'s result into the experiment. A derived [`Device`](@ref)
becomes a device value excluded from the cache key; a derived `GraphConst` field becomes a baked
constant included in it, and the hook does not need to know the difference.

**A large derived value should be `Device`, not `GraphConst`**, because a `GraphConst` array field is
both baked as a trace-time constant and walked by the tracer, while a `Device` is converted once
and walked as one buffer.
"""
function merge_derived(e, derived::NamedTuple)
    isempty(derived) && return e
    T = typeof(e)
    for k in keys(derived)
        hasfield(T, k) || error("ReactantNitro: `derive` returned `$k`, which is not a field of \
            $(nameof(T)). Derived values are MERGED into the experiment, so every key must name an \
            existing field; a user who already knows a value omits that key and \
            sets it in config instead.")
    end
    return Base.typename(T).wrapper(map(f -> get(derived, f, getfield(e, f)), fieldnames(T))...)
end

# ── Distribution stubs, kept non-breaking ───────────────────────────────────────────
#
# Interfaces that would eventually need a distribution handle keep the parameter now, untyped and
# documented as unused. Nothing dispatches on it, so there is no type to own or version, and adding a
# real handle later does not break consumers who wrote `dist` untyped.
#
# Do NOT extend this to the logger. Rank gating is the driver's job; putting rank in the logger
# contract would force every backend to reimplement the same guard, and one that forgets duplicates
# every metric per rank. Rank reaches phase monitors as `info.is_rank0`, documented as always
# `true` today so nobody writes an untested rank-conditional branch.

"""
    rank(dist) -> Int

This process's rank. `dist` is always `nothing` today and `rank(nothing) == 0`.
"""
rank(::Nothing) = 0

"""
    world_size(dist) -> Int

The number of processes. `dist` is always `nothing` today and `world_size(nothing) == 1`.
"""
world_size(::Nothing) = 1
