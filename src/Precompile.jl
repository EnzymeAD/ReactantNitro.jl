# ── Precompiling the checkpoint write ────────────────────────────────────────────────
#
# Measured: the first checkpoint took 6.8 s and the second 2.7, against milliseconds afterwards.
# It is JLD2 specializing its serialization path on the concrete types inside a record, so a run's
# first epoch ends in a stall that looks like a hang. The write is the only thing exercised here:
# it is the one hot path made of ordinary host values, so the workload needs no XLA client and
# compiles no Reactant program. `CheckpointRecord`'s parameter fields are `::Any`, so this covers
# the generic machinery and leaves one model-shaped specialization.
using PrecompileTools: @setup_workload, @compile_workload

@setup_workload begin
    # A two-layer Lux-shaped parameter tree, the shape every `build_model` produces.
    ps = (;
        layer_1 = (; weight = zeros(Float32, 4, 3), bias = zeros(Float32, 4)),
        layer_2 = (; weight = zeros(Float32, 2, 4), bias = zeros(Float32, 2)),
    )
    st = (; layer_1 = NamedTuple(), layer_2 = NamedTuple())
    # `NTuple{G}` of flat host buffers, what `snapshot` converts the optimizer state to.
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
            # The default filename method, which nearly every run uses.
            ckpt = TopKCheckpointer(;
                k = 1, metric = :val_loss, mode = :min, dir,
                name = (; kwargs...) -> checkpoint_filename(nothing; kwargs...)
            )
            # Two writes: the second exercises retention, and carried its own seconds.
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
