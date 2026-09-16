"""
    ReactantNitro

A general Julia training framework on Reactant.jl + Lux.jl.

The central object is a [`Nitro`](@ref): `Nitro(e)` runs the setup sequence and nothing else, so
`validate`, `evaluate`, and `predict` work with no training anywhere in the process. `train!` is a
separate verb. A *run* is what happens when you `train!` a `Nitro`, which is why the `run_*`
accessors keep that name while the object does not.
"""
module ReactantNitro

# ── Dependencies are imported, never `using`ed, for two reasons. ────────────────────
#
# Every snippet in this package is written qualified (`Optimisers.Leaf`, `Reactant.to_rarray`,
# `Enzyme.make_zero`, `Lux.trainmode`), and unqualified `using` collides with names this package
# owns: `LinearAlgebra.rank` against the distribution stub is the concrete case.
#
# The phase leaf USED to be the other one: it was `Training`, which collides with `Lux.Training`,
# and the collision landed on the user rather than on this file, since every experiment file does
# `using Lux, ReactantNitro`. It is now `TrainStepping`, and the naming rule that fixed it is that a
# phase leaf may not take a name any of the packages below already exports. The measured collision
# set across Lux, Optimisers, Reactant, Functors, Enzyme, Random, Statistics, LinearAlgebra, Base,
# and Core is now EMPTY: `test/exports.jl` asserts that, so a future leaf cannot quietly
# reintroduce one.

import Enzyme
import Functors
import JLD2
import JSON3
import LinearAlgebra
import Lux
import Optimisers
# Checkpoint filenames use `%.6g` for the metric value, the same format an earlier stack used,
# so a score in a name is reproducible by anyone with `printf`.
import Printf
import Random
import Reactant
import SHA
import Statistics

"""
    ReactantNitro._unimplemented(what, detail)

Uniform body for a declared-but-unbuilt entry point. The message says the entry point is not built
yet, so a `MethodError`-shaped hole and a not-yet-built one are distinguishable at a glance;
`detail` is whatever identifies the missing piece to whoever is building it.
"""
@noinline _unimplemented(what, detail) = error(
    "ReactantNitro: `$what` is declared but not implemented yet (not yet built: $detail)."
)

# ── Load order matters only where a macro or a type must exist before use. ──────────

include("Config.jl")      # the experiment macro and the field categories
include("Phases.jl")      # the phase system and the phase registry
include("Runs.jl")        # the run registry's machine-readable face; methods live in the KaimonGate extension
include("Interface.jl")   # the user hooks: signatures and defaults
include("Batch.jl")       # batch routing
include("Data.jl")        # the data contract and the dataloaders
include("Decay.jl")       # weight decay
include("Optimizer.jl")   # optimizer construction and parameter groups
include("Schedules.jl")   # schedules
include("Cache.jl")       # the compiled-program cache
include("Setup.jl")       # the setup sequence
include("Train.jl")       # the training loop
include("Logging.jl")     # the logger contract
include("JSONLog.jl")     # the shipped JSON default logger
include("EarlyStop.jl")   # early stopping
include("IORetry.jl")     # retrying I/O
include("Checkpoint.jl")  # checkpointing and resume
# Visualization needs `predict` and the batch routers, so it follows `Train.jl`; it is placed here
# rather than immediately after it because nothing between the two depends on it.
include("Visualize.jl")   # rendering
# Export is last because it consumes almost everything above it: `Nitro`, the routers,
# `check_output_batch_dim`, `host_tree`, and `config_params`.
include("Export.jl")      # export to a served bundle

# ── Exports: this list is the public surface. Everything not listed here is ─────────
# internal and may change without a breaking release.

# Configuration
export @experiment, Device, Host, GraphConst, device_fields, host_fields, config_metadata, compile_view, export_view
# Named configurations. The table belongs to the model; the mechanism lives here.
export presets, from_preset

# User hooks. Four are required, the rest have defaults.
export build_data, build_model, forward, loss
export metrics, train_metrics, finalize_metrics, metrics_residency, derive
export param_group, optimizer, learning_rate, lambda, decay_anchor, gradient_clip_norm
# Manual training mode. Defining `train_step` for an experiment type is what selects
# manual mode; `manual_training(e) = false` declines it while keeping the method;
# `setup_optimizers` supplies the user-owned optimizer states the closure steps itself;
# `backward` and `step_optimizer` are the helpers the closure calls inside the step.
export train_step, setup_optimizers, manual_training, backward, step_optimizer
# Per-LEAF decay exclusion. `default_no_decay` is exported alongside the hook so that
# `no_decay` can be extended rather than replaced, which is the common case.
export no_decay, default_no_decay
export schedules, nonschedulable, dispatch_variant, max_epochs
# The run accessors: each one's `Nitro` keyword defaults to the accessor call.
export seed, accum, n_devs, checkpointer, early_stop, logger

# Asking a run's artifacts what they are, and PUBLIC because the alternative kept being
# reinvented: `read_manifest(dir)` lists a run's retained checkpoints with their epochs and scores
# out of the run's own small manifest, opening no record at all; `checkpoint_info(path)` is the one
# supported way to read a single record's metadata, and returns none of its weights. Neither is a
# new capability, which is the point. Both existed and neither was reachable without knowing the
# internals, so sessions opened the JLD2 by hand, guessed at the API, and printed parameter trees
# into their own transcripts.
export read_manifest, checkpoint_info

# Accelerator configuration: the shared code path for choosing the backend and pinning
# the device count. The Kaimon tool `nitro_setup` is a thin wrapper over it, so a REPL session
# and a Kaimon session configure identically.
export setup_devices!

# Entry points. `Nitro(e)` is the constructor and runs setup only.
export Nitro, train!, validate, evaluate, predict
# Live-handle Device access. Verbs on an existing `Nitro`, and the supported way to sweep a value
# without rebuilding: a `Device` field is excluded from the compile cache's key by construction, so
# writing one provably reuses the compiled programs.
export set_device!, device_value

# Data. Prefetch is a framework DEFAULT rather than an opt-in helper: setup wraps the `train`
# split in a `PrefetchIterator` at `depth = 1` and `workers = Threads.nthreads(:default)`, so a model
# package names none of it. What stays public is the way to override those defaults, the one way to
# decline them, and the two-method trait a source implements to get real concurrency.
#
# `batch_at` and `begin_epoch!` are exported because a model EXTENDS them, which is the same reason
# the hooks above are exported. `check_batch_at` is deliberately NOT: it is a test helper, and
# `ReactantNitro.check_batch_at` at its one call site per package is clearer than a bare name.
export PrefetchIterator, NoPrefetch
export batch_at, begin_epoch!

# Optimizer. `Decay` only: NO `WeightDecay` alias, which would collide with
# `Optimisers.WeightDecay` on the same concept.
export Decay

# Checkpointing. `checkpoint_filename` is exported for the same reason the hooks above
# are: a model package EXTENDS it.
export TopKCheckpointer, save_checkpoint!, load_checkpoint, CheckpointRecord
export checkpoint_filename

# Logging. The contract, plus the one shipped backend: the JSON default, which is what
# an experiment that declares no logger gets. `nothing` stays the documented opt-out.
export log_metrics!, log_params!, log_tags!, log_other!, log_confusion!, finish!
export run_id, run_url, logger_state, reattach!, backend, logger_info
export JSONLogger

# Phases and control
export Phase, Repl, Starting, Compiling, GradCompiling, OptCompiling, EvalCompiling, ExportCompiling
export Stepping, TrainStepping, EvalStepping, Checkpointing, Terminal, Done, Failed
export register_phase_monitor!, unregister_phase_monitor!, progress_counter
export request_stop!, run_dir, current_step, current_epoch, phase
export experiment, parameters, states, binding_report, EarlyStopping, should_stop

# Export. The hooks, the spec carrier, the entry point, and the backend seam.
#
# NOTE the names that are deliberately NOT here. ReactantServerExport exports `export_bundle`,
# `write_bundle`, `IOSpec` and `collect_provenance`, and its extension guarantees that package is in
# scope whenever this surface is usable, so every one of those four would collide on the user rather
# than on us. `export_model`, `write_export` and `ExportSpec` are the same three concepts under names
# that cannot clash. `test/exports.jl` checks the collision set.
export ExportSpec, ExportBackend, ReactantServerBundle
export export_inputs, export_outputs, export_preprocess, export_postprocess
export export_client_outputs, export_client_inputs, export_provenance_extra
export export_model, export_provenance, write_export, site_provenance

# Visualization. `visualize` and `save_figure` are hooks with NO default method; `render` is the
# driver. The framework ships no plotting backend, exactly as logging ships exactly one backend:
# the JSON default, so a bare run leaves a record.
export visualize, save_figure, render

# Distribution stubs
export rank, world_size

end # module
