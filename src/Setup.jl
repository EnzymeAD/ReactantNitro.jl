# Setup.jl
#
# The setup sequence, the `Nitro` constructor, and the accelerator and distribution surface. The
# sequence is written out in `_build_nitro` itself, since a description kept apart from the code
# it orders drifts from it.

"""
    Nitro(e; seed, resume, run_dir, data, n_devs, checkpoint, accum, max_epochs, schedules,
             gradient_clip_norm, logger, checkpointer, early_stop, run_ref, weights, w0) -> Nitro

Run the setup sequence and return the handle. No training.

Ten keywords default to an accessor of the same name (the signature of `_build_nitro` is the
authority for the defaults), so the keyword replaces the accessor's value for one run and omitting
it falls through to the experiment's own. `data`, `checkpoint`, `resume` and `run_ref` are
keyword-only, each naming a fact about this invocation. `early_stop` defaults to `nothing` and
`max_epochs` to `1`.

## The fresh sequence

| # | Step | Why here |
| --- | --- | --- |
| 1 | Validate config: markers, optimizer allowlist, decay settings | Cheapest failures first |
| 2 | Seed the global rng | Before anything that draws (3 and 6) |
| 3 | `build_data(e, dist)` | `derive` needs data |
| 4 | `derive(e, data)`, merged into `e` | Returns host values, so before conversion; may change the architecture, so before `build_model` |
| 4.5 | `setup_devices` | Conversion places values on the mesh |
| 5 | Convert `Device` fields to device; **`e`'s type changes here** | After every hook that produces config, before every hook that is traced |
| 6 | `build_model(e, rng)` | Sees the post-conversion `e` |
| 7 | Capture `w0` | Immediately after, before any restore or step |
| 8 | Param groups, the flat permutation, the per-group ranges | Needs `ps`; unconditional, since the resume path validates against it |
| 9 | **Training only.** Flatten `ps`, build optimizer state, normalize it to device residency | Needs the layout |
| 10 | Resolve routing against `typeof(compile_view(e))` from the first batch | Needs the batch schema and the hooks |
| 11 | **Training only.** Check `length(train) % accum == 0`; resolve the horizon and call each schedule factory once | Needs `length(train)`, `accum`, `max_epochs` |
| 12 | Construct the logger, log parameters and the seed | Before the first compile, which can crash |

Nothing is compiled here. Each program is traced and compiled, against `compile_view(e)`, by the
first verb that runs it.

`build_data` and `derive` see the pre-conversion experiment, where `Device` fields hold host values;
every other hook sees the post-conversion one.

**Resume** adds three steps: between 1 and 2, locate the checkpoint (`:auto` finds `latest` in
`run_dir`), check config compatibility, and restore `seed` from the record with a warning if it
differs (a silent override turns a seed sweep that forgot to vary `run_dir` into N identical runs);
after 7, restore `ps`, `st`, `opt_state`, `step` and `epoch`, with `w0` the fresh capture verified
against the record's `anchor_checksum`; after 8, refuse a flat permutation that differs; after 9,
re-normalize the restored `opt_state` to device residency, without which a resumed run trains at
the wrong point of its optimizer's bias correction with no error.

**A warm start**, `weights = other::Nitro`, takes `ps` and `st` from a handle in this process
after step 7, through host memory, and otherwise runs the fresh sequence. The trees must match leaf
for leaf. `w0` says what `decay_anchor = :w0` anchors to: `:build_model` (the default), `:weights`
(L2-SP fine-tuning), or a parameter tree.

**Without a `train` split** steps 9 and 11 are skipped. Routing falls back to the first batch of
whichever split exists; with no split at all it is deferred to the first `predict` call.

**Progress.** The sequence is reported as one `"setup"` stretch of unknown length, with the current
step as its phase (`setup [building data]`), closing as `setup done`. See
[`progress_reporter!`](@ref).
"""
# `Starting` is published before the build, through the module-level monitors, because `build_data`
# can start and compile a data server and this was the longest undeclared stretch there was.
Nitro(e; kwargs...) = _off_interactive() do
    publish_phase(Starting())
    return try
        n = with_progress_stretch(() -> _build_nitro(e; kwargs...), "setup", 0, 0, 0)
        progress_done!("setup done")
        n
    catch
        progress_done!("setup failed")
        rethrow()
    end
end

# Split out so the public constructor can move it off the interactive thread: construction blocks
# for a minute or more, and on the interactive thread that starves an external supervisor's
# heartbeat. The keyword defaults live here, on the function that reads `e`.
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
        # A preset name is a fact about how this handle was built; `Nitro(E, :name)` sets it.
        preset::Union{Symbol, Nothing} = nothing,
        # Four stay keyword-only, because each names a fact about THIS INVOCATION rather than
        # a property of the experiment. `data`'s accessor exists and is called `build_data`.
        resume = false,
        data = nothing,
        checkpoint = nothing,
        run_ref = nothing,
        # A warm start from another handle, and what `:w0` anchoring means for it.
        weights = nothing,
        w0 = :build_model,
        # PROTOTYPE: hooks supplied as values shadow the method of the same name (Hooks.jl).
        hooks = (;)
    )
    check_hooks(hooks)
    check_weights_kwargs(weights, w0; checkpoint, resume)

    # ── before step 2: locate a checkpoint (the resume path) ───────────────────────
    # `run_dir` is the one path concept in the framework, so a checkpointer constructed without an
    # explicit `dir` adopts it here. An explicit `dir` pins it and is never overwritten.
    checkpointer isa TopKCheckpointer && checkpointer.dir === nothing &&
        (checkpointer.dir = String(run_dir))
    # The filename hook is bound here too, as a closure calling the generic function, so
    # `save_checkpoint!` never receives `e` (an object full of device leaves handed to the
    # serializer) and a revised `checkpoint_filename` takes effect on the next write.
    checkpointer isa TopKCheckpointer && checkpointer.name === nothing &&
        (checkpointer.name = (; kwargs...) -> ReactantNitro.checkpoint_filename(e; kwargs...))
    # The default logger adopts the run directory the same way; a wrapper around it (the gate
    # extension's RecordingLogger) forwards the pin, and a user's own logger is untouched.
    logger = _adopt_logger!(logger, run_dir)

    (checkpoint !== nothing || resume !== false) && progress_phase!("loading checkpoint")
    record, source = nothing, nothing
    if checkpoint !== nothing
        # `load_checkpoint` dispatches on the checkpointer, so `checkpointer = nothing` has no
        # loader and an explicit `checkpoint = path` would otherwise load nothing and answer with
        # fresh weights.
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
    # Two restores, one compatibility check: `checkpoint = path` takes the derived `Device` values
    # from the record (an evaluation process may lack the training data), `resume` recomputes them.
    weights_only = record !== nothing && checkpoint !== nothing

    # ── 1. validate config, cheapest failures first ────────────────────────────────
    validate_config(e; accum, gradient_clip_norm)

    # Manual mode is selected by a `train_step` method and frozen into the handle; flipping it on
    # an existing handle is reported by `fixed_config_report` rather than applied.
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
    # The seed is restored from the record, with a warning: restoring is what makes `w0`
    # reproducible, and a silent override would make a seed sweep N identical runs.
    if record !== nothing && record.seed != seed
        @warn "ReactantNitro: restoring `seed = $(record.seed)` from the checkpoint, overriding the \
               requested `seed = $seed`. A rebuild must reproduce the same initialization: the \
               resume check verifies the decay anchor against a freshly captured `w0`."
        seed = record.seed
    end
    Random.seed!(seed)
    rng = Random.default_rng()

    # ── 3. data. `build_data` sees the PRE-conversion experiment ───────────────────
    progress_phase!("building data")
    collection = data === nothing ? hook_fn(hooks, :build_data, build_data)(e, nothing) : data
    collection isa NamedTuple || error("ReactantNitro: `build_data` must return a NAMED collection, \
        `(; train, val)` or `(; train, val, test)`, and returned a `$(typeof(collection))`. Named and \
        extensible, so `evaluate` with no `test` supplied is a clear error rather than a positional \
        mistake.")
    for name in keys(collection)
        # Through the wrapper, so the error names the loader the user wrote.
        check_data_source(prefetch_source(getproperty(collection, name)), name)
    end
    training = haskey(collection, :train)

    # ── 4. derive, still pre-conversion, so it returns host values ─────────────────
    # `checkpoint = path` takes the derived values from the record; `resume` recomputes them, and
    # the config comparison excludes derived values for that reason.
    progress_phase!("deriving")
    e = weights_only ? merge_derived(e, restored_devices(record, e)) :
        merge_derived(e, derive(e, collection))

    # `GraphConst` values must hash and compare by content. Asserted before the comparison below,
    # which would otherwise print a diff whose two sides look identical. Here because `derive` has
    # merged, so derived fields are finally in the struct.
    assert_graphconst_hashable(e)

    # Here rather than between steps 1 and 2 because the field set is not final until `derive`
    # has merged.
    record === nothing || check_config_compatible(record, e, source)

    # ── 4.5 devices, before conversion places values on the mesh ───────────────────
    progress_phase!("setting up devices")
    mesh = setup_devices(n_devs)

    # ── 5. convert Device fields. `e`'s TYPE CHANGES HERE ─────────────────────────
    progress_phase!("building model")
    e = to_device_config(e, mesh)
    ev = compile_view(e)

    # ── 6, 7. build the model, then capture w0 immediately ─────────────────────────
    model, ps, st = hook_fn(hooks, :build_model, build_model)(e, rng)
    # Parameters and layer state are replicated on a mesh; only the batch is sharded.
    ps = place_replicated(ps, mesh)
    st = place_replicated(st, mesh)
    w0_tree = deepcopy(ps)            # step 7: before any restore or step

    # Step 6's rebuild happens before the restore, in spite of being thrown away, because the
    # anchor checksum is verified against a fresh `w0`.
    if record !== nothing
        # `from_host` first: a record may hold a surrogate (`HostRNG`) that `to_rarray` cannot
        # recognise.
        ps = place_replicated(from_host(to_host(record.ps)), mesh)
        st = place_replicated(from_host(to_host(record.st)), mesh)
    end

    # Through host memory, so the source may live on another mesh. Checked against the fresh
    # layout: the transferred tree has to fit this model.
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
        # `weights_only` continues no run, so declining the logger drops no history.
        check_logger_compatible(record, logger, source; weights_only)
    end

    # ── 10. routing, resolved against typeof(compile_view(e)) ──────────────────────
    # The first batch of `train`, else of whichever split exists, else nothing, in which case
    # routing defers to the first `predict` call. `opt_state` is declared here for manual mode.
    routing, batch_size, schema = (nothing, nothing, nothing)
    opt_state = nothing
    if !isempty(keys(collection))
        schema_split = training ? :train : first(keys(collection))
        # Through the source; the wrapper's `iterate` is a passthrough so it is the same batch.
        progress_phase!("reading first batch")
        probe = first(prefetch_source(getproperty(collection, schema_split)))
        routing = resolve_routing(ev, probe; model, ps, st, hooks)
        if training && manual
            # Manual mode builds its optimizer state here, before `batch_size_of`, because the
            # closure's router must be part of the routing the batch size is inferred from. The
            # result is normalized to device residency and asserted on every leaf's STATE.
            opt_state = setup_optimizers(e, model, ps, st, mesh)
            opt_state = to_device_leaf(opt_state; mesh)
            assert_opt_state_device(opt_state)
            # A restore constructs nothing, so it is re-normalized as the automatic path does at
            # step 9. Weights-only constructions keep a fresh optimizer.
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
        # A legible setup error here rather than an XLA shape complaint at the first transfer.
        check_shardable_batch(batch_size, n_devs)
    end

    # ── after 10. prefetch, applied by the framework rather than requested ──────────
    #
    # Wrapped here, after every structural check has seen the collection exactly as `build_data`
    # returned it. Every split is wrapped, eval ones included (they stream through `eval_stream`).
    collection = auto_prefetch(collection)
    # Source-option checks run on the resolved pipeline, for every split: some options are only
    # wrong when a producer runs ahead, others (a training loader keeping its partial final batch)
    # are wrong regardless, and `cfg` lets the hook tell the two apart.
    for nm in keys(collection)
        split = getproperty(collection, nm)
        check_source_options(prefetch_source(split), nm, prefetch_config(split))
    end
    training && warn_no_concurrency(collection.train)

    # Here because both need the split collection and the resolved routing: a policy with no `val`
    # split, or naming a metric the experiment cannot emit, is a configuration error.
    check_early_stop(early_stop, collection, routing)
    check_checkpointer(checkpointer, collection, routing)

    # ── 9, 11. training only ───────────────────────────────────────────────────────
    # Manual mode flattens nothing: the closure owns the optimizer state, the masks and the
    # accumulator, so `()` / `nothing` is correct rather than a placeholder.
    training && progress_phase!("building optimizer")
    flat = manual ? () : flatten(ps, layout, mesh)
    resolved, total = nothing, nothing
    # Hoisted so the freeze below can see it. `()` for a no-train `Nitro` and for manual mode,
    # whose rules enter the compile key through `worlds_manual` instead.
    chains = ()
    # Resolved once, here: `no_decay` is a user hook, and recomputing it at `train!` would let a
    # revision take effect on a frozen handle. Training-only and skipped in manual mode.
    masks = training && !manual ? map(m -> place_replicated(m, mesh), no_decay_masks(e, ps, layout)) : nothing
    if training
        n_batches = length(collection.train)
        check_train_divisibility(n_batches, accum)
        total = max_epochs * div(n_batches, accum)
        if manual
            # `opt` schedules are path-bound into the user's `opt_state` and applied by the driver's
            # per-step rebuild; device-keyed schedules resolve as in the automatic loop.
            resolved = resolve_manual_schedules(
                e, schedules, total;
                opt_state, accessor = ReactantNitro.schedules(e)
            )
        else
            opt_state = build_opt_state(e, flat, layout; anchors, masks, mesh)
            # Re-normalize a RESTORED `opt_state`. Step 9's normalization applies to state the
            # framework constructed; a restore constructs nothing, and without this RAdam's `t`
            # stays host, never advances under trace, and the run silently trains at the wrong
            # point of its bias correction. A weights-only construction keeps a fresh optimizer.
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
            # `rebuild_rules` moves only the VALUES each step; the rule types, and so the `apply!`
            # methods, are these, so resolving the rule worlds here covers the whole run.
            resolved = resolve_schedules(
                e, schedules, total;
                chains, accessor = ReactantNitro.schedules(e),
                groups = layout.groups
            )
        end
    end

    # ── 12. the logger, BEFORE the first compile, which can crash ──────────────────
    # Reattachment before any metric is logged, so a backend continues one history rather than
    # opening a second experiment.
    progress_phase!("starting logger")
    record === nothing || record.logger_state === nothing ||
        reattach!(logger, record.logger_state)
    # The preset name is logged with the config it produced, omitted when none. The resolved
    # prefetch settings are hyperparameters: `workers` derives from `-t`, and two runs of identical
    # code at different `-t` get a different partition of each epoch into accumulation groups.
    pf = training ? prefetch_config(collection.train) :
        (; device_batches = 0, host_batches = 0, workers = 0, ordered = true)
    cfg = config_params(
        e; seed, accum, max_epochs, gradient_clip_norm,
        prefetch_workers = pf.workers, prefetch_device_batches = pf.device_batches,
        prefetch_host_batches = pf.host_batches, prefetch_ordered = pf.ordered
    )
    log_params!(logger, preset === nothing ? cfg : merge(cfg, (; preset)))

    # `step` is restored, never derived: deriving it from `epoch` is wrong whenever steps per epoch
    # changed, and restoring it puts every stateless schedule back exactly.
    step0, epoch0 = (record === nothing || weights_only) ? (0, 0) : (record.step, record.epoch)

    # A resume into a finished run says so rather than returning instantly.
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
        # The training run's identity, kept because it cannot be recovered later.
        record === nothing ? nothing : record.run_id,
        record === nothing ? nothing : record.run_url,
        String(run_dir), Int(seed), Int(accum), Int(max_epochs), gradient_clip_norm,
        checkpointer, early_stop, probe_accessors(e),
        frozen_dispatch(e, model, ps, st, routing, chains; manual, opt_state),
        (; masks, anchors),
        map(zero, flat), RegisteredMonitor[], Int(step0), Int(epoch0), Starting(),
        false, nothing,
        # No metrics and no elapsed time yet, even for a restored handle.
        (;), nothing, nothing,
        weights === nothing ? nothing : weights_origin(weights),
        NamedTuple[],
        false
    )
    # Two builds from one set of pieces: the text goes to `log_other!` so the record says where
    # every value bound, and the sections are what `show(nitro)` appends to the handle's display.
    # Nothing is printed here; a constructor that printed its return value would display the
    # handle twice in the REPL.
    nitro.sections = build_binding_sections(nitro)
    log_other!(logger, "binding_report", build_binding_report(nitro))
    run_ref === nothing || (run_ref[] = nitro)
    # Adopted here as well as at `train!` (idempotent), because a handle that is only validated,
    # predicted from or rendered never reaches `train!`, and its eval compile is what a watchdog
    # most needs to hear about.
    adopt_monitors!(nitro)
    # No `Repl` here: a constructor that reported itself idle would hand the card back between the
    # build and the first verb. `Starting` was declared at the top of `Nitro`.
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

# What `show` and the export provenance say about a warm start. Taken at construction, since the
# source handle may train on afterwards.
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

One `@warn` at setup when the training split resolved to no host-side concurrency, naming what to
implement. It exists because a model once trained 4x slow for weeks with nothing saying so.

A warning, not an error: a single producer trains correctly, some sources cannot be indexed, and
`NoPrefetch` is a legitimate choice. `:fanout`, `:fanout_unordered` and `:materialized` (a `Vector`
of already-built batches, so no host work to spread) say nothing.
"""
function warn_no_concurrency(train_split)
    cfg = prefetch_config(train_split)
    src = prefetch_source(train_split)
    if cfg.path === :single_no_trait
        @warn """
        ReactantNitro: the `train` split runs ONE producer task and leaves \
        $(prefetch_workers(train_split)) workers idle, because `$(typeof(src))` implements \
        neither half of the index-addressable trait. Buffering is lookahead, not concurrency: a \
        slow producer starves the device at any `device_batches`; watch `data_wait_frac`. \
        Implement BOTH `ReactantNitro.batch_at(src, i)` (`i` is a BATCH index) and \
        `ReactantNitro.begin_epoch!(src)`."""
    elseif cfg.path === :inline
        @warn """
        ReactantNitro: the `train` split is `NoPrefetch`, so its host data path runs inline on the \
        training task and overlaps nothing; watch `data_wait_frac` to see what it costs."""
    elseif cfg.path === :single
        @warn """
        ReactantNitro: the `train` split runs ONE producer task because `workers = 1` was \
        requested; watch `data_wait_frac`."""
    end
    return nothing
end

"""
    ReactantNitro.check_driver_fields(e) -> nothing

A driver-only value (`seed`, `max_epochs`, `run_dir`, ...) declared as a `GraphConst` field is a
setup error. A `GraphConst` bakes as a trace-time constant and enters the compile cache key, so a
`GraphConst` `seed` recompiles per seed and defeats the sweep the seeding rule exists for. Unmarked
is right: an unmarked field is [`Host`](@ref). `accum` and `gradient_clip_norm` genuinely bake and
are not checked.
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

The flat hyperparameter table setup step 12 logs: the experiment's actual configured values plus
the run keywords that are hyperparameters in their own right. Values, not declaration metadata,
since logging the defaults would report the same `width` for every run of a sweep.

`Device` fields are read back to host numbers; anything not scalar-ish is logged as a descriptor
rather than splatted into the flat table; `Host` fields are included only when scalar-ish, because
a `Host` field may be a materialized dataset. `seed` is here because it is a hyperparameter.
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

The per-group decay anchor as a flat device slice, or `nothing` for a group decaying toward zero.
`nothing` is type-level, so a `:zero` group's chain emits no subtraction and carries no buffer.
`w0` is the parameters as `build_model` returned them at step 7.
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
    # Both residency hooks here: a typo would otherwise silently select the default.
    check_residency(e, :metrics)
    check_residency(e, :train_metrics)
    return nothing
end

"""
    Nitro(E::Type, preset::Symbol; kwargs...) -> Nitro

Build the experiment from [`from_preset`](@ref)`(E, preset)` and record which named recipe the run
claimed, so the name reaches the checkpoint record and `log_params!`.

```julia
n = Nitro(MyExp, :baseline; max_epochs = 40, aug_rotate_deg = 9.0)
#                              ^ run keyword    ^ field, so it overrides the recipe
```

Keywords are split by one rule: a `Nitro` keyword goes to `Nitro`; anything else must be a field
of `E` and goes to the recipe; a name that is both (`max_epochs`, `seed`, `run_dir`, `accum`,
`n_devs`, `gradient_clip_norm`) is the run keyword. Anything in neither set is an error naming
both. The keyword set is derived from the constructor's declaration, so it cannot go stale.

The longer spelling, `Nitro(from_preset(MyExp, :baseline; ...); preset = :baseline)`, still works
and is what this is shorthand for, but it names the preset twice and nothing checks the two agree.
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
            # Deferred past the loop so the error's suggested `from_preset` call carries every
            # field override.
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

# Refused because it silently trains on the wrong thing: `data` skips `build_data`, but this form
# builds `e` itself, so the caller's collection came from a different instance and anything
# `build_data` populates on the experiment stays empty for `derive` to read. (`checkpoint`,
# `resume` and `run_ref` skip nothing and are not refused.) The suggestion carries the field
# overrides, since a copy-pasteable fix that dropped them would build the wrong model.
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

# A field override may be an array, and a suggestion is useless as a screenful.
_short_repr(v) = (s = repr(v); length(s) <= 40 ? s : "<$(nameof(typeof(v)))>")

# Derived from the constructor's own declaration rather than a literal list that could go stale.
# Read off `_build_nitro`, not `Nitro`: the public wrapper's declaration is a bare keyword sink,
# and while this pointed at it every preset keyword was classified as unknown. The sink check is
# what catches the body being split out from under this a second time.
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

Setup step 4.5: a 1-D `Reactant.Sharding.Mesh` with a single `:data` axis over the requested local
devices, sharding the batch along it and replicating everything else (see
[`place_replicated`](@ref) and [`place_batch`](@ref)). `n_devs == 1` skips the mesh entirely, since
a one-device mesh is a no-op Reactant warns about.

The scope is single node, one process, one or more devices; XLA partitions one compiled program
over the mesh, which does not compose with MPI or NCCL. `n_devs` counts VISIBLE devices, so
`CUDA_VISIBLE_DEVICES` is the supported way to restrict a run, and asking for more is an error.
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
# One code path for choosing the session's accelerator: a REPL calls `setup_devices!`, the Kaimon
# tool `nitro_setup` wraps the same function. It writes `_PINNED_N_DEVS` (Interface.jl).

"""
    ReactantNitro.setup_devices!(; backend = nothing, n_devs = nothing) -> NamedTuple

Configure this process's accelerator for every subsequent run and report what is in effect, as
`(; backend, visible, n_devs, pinned)`. The Kaimon tool `nitro_setup` is exactly this function.

Optional: a session that never calls it runs on Reactant's default backend with `n_devs` equal to
every visible device. `backend` passes through to `Reactant.set_default_backend` (`"cpu"`,
`"gpu"`, `"cuda"`, `"rocm"`, `"tpu"`). `n_devs` pins the device count for the process, validated
against the visible devices; the pin beats an experiment's declared `n_devs`, and an explicit
`n_devs` keyword on `Nitro` still wins for that one run.

One process, one XLA: the client is initialized once and `n_devs` only slices the visible device
set, so to restrict which GPUs a session sees, set `CUDA_VISIBLE_DEVICES` before starting the
process. The batch size is global and is split across the mesh, so adding devices buys throughput
without changing the effective batch. Called with no arguments, this only reports.

```julia
ReactantNitro.setup_devices!()                      # report
ReactantNitro.setup_devices!(backend = "cpu")       # run everything on CPU
ReactantNitro.setup_devices!(backend = "cuda", n_devs = 2)  # two of the visible CUDA devices
```
"""
function setup_devices!(; backend = nothing, n_devs = nothing)
    # A friendlier error than the KeyError an unknown backend name would surface.
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
    # Validated through `setup_devices`, so the `n_devs` contract has one authority.
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

The two placements: everything is replicated except the batch, which is sharded on its last
dimension along the `:data` axis. The sample axis is last by the batch contract, which is why
routing transfers only the fields a hook declares. `mesh === nothing` is the single-device path
and both fall through to `to_rarray`.

The device count is throughput, not a batch multiplier: `batch_size = 32` on four devices puts 8
samples on each, with the numerics of a 32-sample batch.
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

A batch sharded on the `:data` axis must divide evenly across the mesh. XLA would pad or refuse
without naming the batch size, so the framework checks first.
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

Setup step 5: convert every [`Device`](@ref) leaf through [`to_device`](@ref) and rebuild the
experiment, which changes its type. Conversion happens here and, per optimizer step, for the
scheduled entries only; nothing converts inside the traced step, so a setup-fixed entry of any size
is converted once and reused by reference.
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

The host value of a [`Device`](@ref) field of the live experiment, which holds device values after
setup step 5. The counterpart to [`set_device!`](@ref). Errors on a field that is not `Device`,
since a `Host` or `GraphConst` field is already a host value.
"""
function device_value(nitro::Nitro, name::Symbol)
    e = nitro.e
    _check_device(e, name, "device_value")
    return to_host(getfield(e, name))
end

"""
    set_device!(nitro, name, value) -> nitro
    set_device!(nitro; name = value, ...) -> nitro

Write a new value into a [`Device`](@ref) field of a live handle without recompiling: this is how a
device sweep or a changed inference threshold reuses the programs already in the compile cache.

The guarantee is structural. The cache key skips `device_fields` when hashing and seeds on the
type name rather than the parameterized type, so a new device value leaves every key component
untouched; the function asserts both before committing. Refused: a field that is not `Device` (a
`GraphConst` is genuinely a different program, a `Host` field reaches no trace), a value of a
different element type or size (the type would move the key; the size would not and would fail
inside XLA instead), and a scheduled field (the per-step rebuild would overwrite the write).

```julia
n = Nitro(e; checkpoint = "runs/MyExp/best.jld2")   # weights-only, skips `derive`
for thr in (0.3f0, 0.5f0, 0.7f0)
    set_device!(n; threshold = thr)
    out = predict(n, batch)                         # no compile, any iteration
end
```
"""
function set_device!(nitro::Nitro, name::Symbol, value)
    e = nitro.e
    T = typeof(e)
    _check_device(e, name, "set_device!")

    # The schedule owns a scheduled field: a write here would vanish at the next optimizer step.
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
    # The cache key's promise, checked rather than trusted; both halves have failed in review.
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
`set_config_report!(false)` silences it for a driver calling `validate` in a loop. A property of the
session's console rather than of the run, so not a `Nitro` keyword.
"""
const CONFIG_REPORT = Ref(true)

"Set whether the entry points print [`fixed_config_report`](@ref). Returns the previous value."
function set_config_report!(on::Bool)
    prev = CONFIG_REPORT[]
    CONFIG_REPORT[] = on
    return prev
end

# The run knobs whose accessors are pure and safe to re-probe. `logger`, `checkpointer`,
# `early_stop` and `schedules` are constructors called exactly once at setup, so probing them would
# fire their side effects and compare by identity anyway; a revised one is not detected here.
const _PROBED = (:seed, :accum, :max_epochs, :gradient_clip_norm, :run_dir)

"""
    ReactantNitro.frozen_dispatch(ev, model, ps, st, routing, chains) -> NamedTuple

Every component of the compile-cache key that comes from live method dispatch, resolved once at
construction, so the programs a `Nitro` uses are determined when it is built and a redefinition
cannot make an existing handle silently recompile between two `train!` calls.
[`fixed_config_report`](@ref) is what tells the user the handle is stale instead.

| Entry | Covers | Consumed by |
| --- | --- | --- |
| `worlds_train` | the hooks, resolved against a train-mode `st` | `grad_program` |
| `worlds_opt` | the same, plus every rule's `apply!` | `opt_program` |
| `worlds_eval` | the hooks, resolved against an eval-mode `st` | `fwd_program`, `eval_metric_program` |
| `graphconst_hash` | the experiment-derived key component | every `compile_cached`, as `gc_hash` |

The rules go in `worlds_opt` only, so a rule edit stays off the expensive gradient program. The two
`st` modes matter for a model whose train and eval state types differ and which dispatches
`forward` on them.
"""
function frozen_dispatch(e, model, ps, st, routing, chains; manual = false, opt_state = nothing)
    # `ev` for the worlds, since that is what every trace site passes; the full `e` for the
    # residencies, since that is what the call sites this replaced passed.
    ev = compile_view(e)
    st_train, st_eval = Lux.trainmode(st), Lux.testmode(st)
    return (;
        worlds_train = hook_worlds(ev; model, ps, st = st_train),
        worlds_opt = hook_worlds(ev; model, ps, st = st_train, chains),
        worlds_eval = hook_worlds(ev; model, ps, st = st_eval),
        # An experiment with no `train_metrics` stays on the `:device` program whatever the
        # accessor says.
        tm_residency = routing === nothing || routing.train_metrics === nothing ? :device :
            check_residency(e, :train_metrics),
        metrics_residency = check_residency(e, :metrics),
        # Hoisted once: `set_device!` errors if a rebuild moves this hash, so it cannot change
        # within a run. Argument types and shapes are still computed on every call, since
        # Reactant's guard covers types but not shapes. Worth little per lookup, but a large
        # `GraphConst` array hashed per micro-batch was hundreds of ms per epoch.
        graphconst_hash = graphconst_field_hash(ev),
        # Manual mode: `worlds_manual` is the closure program's key component (the hooks plus the
        # closure's own method world). `worlds_setup` is staleness-only, since `setup_optimizers`
        # is not traced by the closure and does not belong in the compile key.
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

Which frozen world tuples no longer match live dispatch, as [`fixed_config_report`](@ref) prints
them. Empty is the normal case. This is the whole safety story for freezing dispatch: the handle
keeps its programs, and this removes the silence about it, at the cost of a few `which` calls per
entry-point call. A new `Nitro` always resolves current dispatch.
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

# Rebuilt rather than stored, since only the rule TYPES matter and storing the construction-time
# chains would pin their device scalars alive. Manual mode has no framework chains.
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

What has been redefined since this handle froze it, and is therefore not in effect. Empty when
nothing has, which is why the entry points are silent about it. It does not list the handle's
values; `show(nitro)` does that.

Setup resolves the accessor-defaulted keywords once into the handle's fields, so revising
`accum(::MyExp)` and calling `train!` on an existing handle changes nothing, and this report is what
says so. The handle always wins: `accum` and `max_epochs` are entangled with the schedule horizon,
so applying one in isolation would leave the schedules resolved against a horizon that no longer
exists. The fix is a rebuild, which reuses every compiled program the change does not invalidate.
`entry` selects the relevant set, since `validate` and `predict` read neither `accum` nor the clip.
"""
function fixed_config_report(nitro::Nitro; entry::Symbol = :train)
    e = nitro.e
    io = IOBuffer()
    # Drift only. The handle's fixed values are `show(nitro)`'s job; printing them at the top of
    # every entry point was noise. Empty when nothing drifted, and then nothing is printed.
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
        # Accessor-THEN against accessor-NOW; the stored field would flag every keyword override.
        now == was && continue
        # One string: a `\` continuation keeps the next line's indentation.
        println(
            io, "  ! `$f(e)` was redefined: it returned $was at construction and returns $now ",
            "now. This handle still uses $(getfield(nitro, f)); rebuild to pick up $now."
        )
    end
    # The accessor lines cover configuration; this covers code.
    stale = stale_hooks(nitro)
    isempty(stale) || println(
        io, "  ! hooks were redefined after this `Nitro` was built ",
        "($(join(stale, ", "))). It will keep running the programs it was ",
        "built with; rebuild the `Nitro` to pick up the new code. The compile ",
        "cache is module-level, so a rebuild recompiles only what actually ",
        "changed."
    )
    # The world-closure guard poisons module entries so new handles recompile; this tells the
    # handle when its own programs were among them.
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
    # Built last so "nothing drifted" can be the empty string.
    title = "ReactantNitro: $(nameof(typeof(e))) has values fixed at construction that have \
             since been redefined. This handle keeps what it froze; rebuild the `Nitro` to pick \
             up the new ones, and see `show(nitro)` for what it is currently holding."
    return title * "\n" * body
end

# Printed rather than `@info`-ed (a report, not a diagnostic), and only when there is something to
# say.
function report_fixed_config(nitro::Nitro, entry::Symbol)
    CONFIG_REPORT[] || return nothing
    txt = fixed_config_report(nitro; entry)
    isempty(txt) || print(stdout, txt)
    return nothing
end

"""
    ReactantNitro.merge_derived(e, derived) -> e

Setup step 4: merge [`derive`](@ref)'s result into the experiment. A derived [`Device`](@ref)
becomes a device value excluded from the cache key; a derived `GraphConst` becomes a baked constant
included in it. A large derived value should be `Device`, since a `GraphConst` array is walked by
the tracer.
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
# Interfaces that would need a distribution handle keep the parameter now, untyped and unused, so a
# real handle later breaks nobody. Rank gating is the driver's job, never the logger's; it reaches
# phase monitors as `info.is_rank0`, always `true` today.

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
