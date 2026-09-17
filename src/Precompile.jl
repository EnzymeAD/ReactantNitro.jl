# ── Precompiling the checkpoint write ────────────────────────────────────────────────
#
# MEASURED, not guessed. On the README quick start the first checkpoint took 6.8 seconds and the
# second 2.7, against 7 ms and 30 ms for the third and fourth. It is not I/O and it is not the
# model: it is JLD2 specializing its serialization path on the concrete types inside a record, paid
# once per process across the first two writes. A run's first epoch therefore ends in a stall that
# looks like a hang and is really a compile.
#
# The write is the ONLY thing exercised here, and that is deliberate on two counts. It is the one
# hot path made entirely of ordinary host values, so a workload can drive it without an XLA client,
# without a device, and without compiling a single Reactant program, which is what keeps package
# precompilation from becoming the GPU-hours exercise the docs build refuses to be. And it is the
# one whose cost a user meets at the worst possible moment, an hour into a run rather than at
# `using`.
#
# **The shapes are representative rather than exhaustive.** `CheckpointRecord`'s parameter fields
# are `::Any`, so JLD2 dispatches on what is inside them and a record holding some other model's
# tree still specializes on that tree. What this covers is the generic machinery every write goes
# through, which is the bulk of it; the residue is one model-shaped specialization rather than the
# whole serializer.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # A two-layer Lux-shaped parameter tree: `NamedTuple` of `NamedTuple`s of `Array`s, which is
    # the shape every `build_model` on this stack produces and the one JLD2 has to learn.
    ps = (;
        layer_1 = (; weight = zeros(Float32, 4, 3), bias = zeros(Float32, 4)),
        layer_2 = (; weight = zeros(Float32, 2, 4), bias = zeros(Float32, 2)),
    )
    st = (; layer_1 = NamedTuple(), layer_2 = NamedTuple())
    # `NTuple{G}` of flat host buffers, which is what the optimizer state crosses the program
    # boundary as and what `snapshot` converts to host.
    opt_state = (zeros(Float32, 26),)
    snap = (;
        ps, st, opt_state,
        flat_permutation = collect(1:26),
        step = 1, epoch = 1, seed = 42,
        config = (; width = 4), devices = (; lambda = 1.0f-4),
        run_id = nothing, run_url = nothing,
        logger_state = nothing, logger_type = nothing,
        anchor_checksum = nothing, stop_reason = nothing, preset = nothing,
    )

    @compile_workload begin
        dir = mktempdir()
        try
            # `name` is what setup binds from the experiment's `checkpoint_filename`; the default
            # method is the one nearly every run uses, so it is the one worth specializing.
            ckpt = TopKCheckpointer(;
                k = 1, metric = :val_loss, mode = :min, dir,
                name = (; kwargs...) -> checkpoint_filename(nothing; kwargs...)
            )
            # Two writes rather than one: the second is what exercises retention, which reads the
            # manifest back and deletes a displaced file, and the measurement showed the second
            # write carrying its own seconds of specialization.
            save_checkpoint!(ckpt, 1, (; val_loss = 1.0f0), snap)
            save_checkpoint!(ckpt, 2, (; val_loss = 0.5f0), merge(snap, (; epoch = 2, step = 2)))
            # The read side, which a `resume` pays on its way in rather than on its way out.
            entries = read_manifest(dir)
            isempty(entries) || load_checkpoint(ckpt, joinpath(dir, first(entries).file))
        finally
            rm(dir; force = true, recursive = true)
        end
    end
end
