"""
    ReactantNitro

A training framework for Lux models compiled with Reactant.

The central object is a [`Nitro`](@ref). `Nitro(e)` runs setup only, so `validate`, `evaluate` and
`predict` work without training; [`train!`](@ref) trains it. A *run* is one `train!` of a `Nitro`,
which is what the `run_*` accessors are named for.
"""
module ReactantNitro

# Dependencies are imported, never `using`ed: every snippet is written qualified, and unqualified
# `using` collides with names this package owns (`LinearAlgebra.rank`). A phase leaf may not take a
# name any of these packages exports (`Training` once collided with `Lux.Training`); the exports
# test asserts the collision set is empty.

import Dates
import Enzyme
import Functors
import JLD2
import JSON
import LinearAlgebra
import Logging
import Lux
import Optimisers
# Checkpoint filenames use `%.6g` for the metric value, reproducible by anyone with `printf`.
import Preferences
import PrettyTables
import Printf
import ProgressLogging
import ProgressMeter
import Random
import Reactant
import SHA
import Statistics
import Tables
import UUIDs

"""
    ReactantNitro._unimplemented(what, detail)

The error body for an entry point that is declared but not built yet. The message says so, which
tells it apart from a `MethodError`; `detail` names the missing piece.
"""
@noinline _unimplemented(what, detail) = error(
    "ReactantNitro: `$what` is declared but not implemented yet (not yet built: $detail)."
)

# ── Load order matters only where a macro or a type must exist before use. ──────────

include("Config.jl")      # the experiment macro and the field categories
include("Phases.jl")      # the phase system and the phase registry
include("Runs.jl")        # the run registry's machine-readable face; methods live in the KaimonGate extension
include("Interface.jl")   # the user hooks: signatures and defaults
include("Hooks.jl")       # PROTOTYPE: hooks supplied as a map of values
include("Batch.jl")       # batch routing
include("Data.jl")        # the data contract and the dataloaders
include("Decay.jl")       # weight decay
include("Optimizer.jl")   # optimizer construction and parameter groups
include("GradRules.jl")   # gradient rules: SAM and the default single gradient
include("EMA.jl")         # the exponential moving average of the weights
include("Schedules.jl")   # schedules
include("TraceHook.jl")   # opt-in precise invalidation: records what each trace went through
include("Cache.jl")       # the compiled-program cache
include("Setup.jl")       # the setup sequence
include("Train.jl")       # the training loop
include("Logging.jl")     # the logger contract
include("JSONLog.jl")     # the shipped JSON default logger
include("EarlyStop.jl")   # early stopping
include("IORetry.jl")     # retrying I/O
include("Checkpoint.jl")  # checkpointing and resume
include("SourceSyntax.jl") # the string form of a checkpoint source, for the gate tools
include("History.jl")     # the per-epoch metric history a handle keeps, and its table
include("Render.jl")      # the framed table renderer, text and HTML
# After `Train.jl`, since visualization needs `predict` and the batch routers.
include("Visualize.jl")   # rendering
# Last, since export consumes almost everything above it.
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
# Gradient rules: `grad_rule` picks one; `perturbation` is what a new `PerturbRule` defines.
export grad_rule, GradRule, SingleGrad, PerturbRule, SAM, perturbation
# Manual training mode: defining `train_step` selects it, `manual_training(e) = false` declines it,
# `setup_optimizers` supplies the user-owned optimizer states, `backward` and `step_optimizer` are
# the helpers the closure calls.
export train_step, setup_optimizers, manual_training, backward, step_optimizer
# Per-leaf decay exclusion; `default_no_decay` is exported so `no_decay` can extend it.
export no_decay, default_no_decay
# What a perturbation rule such as SAM leaves alone; independent of `no_decay`.
export no_perturb
# The exponential moving average of the weights.
export EMA, ema
export schedules, nonschedulable, dispatch_variant, max_epochs
# The run accessors: each one's `Nitro` keyword defaults to the accessor call.
export seed, accum, n_devs, checkpointer, early_stop, logger, check_divergence, autolog_schedules,
    log_every_n_steps

# Asking a run's artifacts what they are: `read_manifest(dir)` lists retained checkpoints from the
# manifest without opening a record, and `checkpoint_info(path)` reads one record's metadata with
# none of its weights. Public because sessions otherwise opened the JLD2 by hand.
export read_manifest, checkpoint_info
# Which runs a run directory holds, and which one a handle writes: the ids a `"<id>" => :best`
# checkpoint source names.
export runs, checkpoint_run

# Accelerator configuration; the Kaimon tool `nitro_setup` is a thin wrapper over it.
export setup_devices!

# Entry points. `Nitro(e)` is the constructor and runs setup only.
export Nitro, train!, validate, evaluate, predict
# Live-handle Device access: a `Device` field is excluded from the cache key, so writing one
# provably reuses the compiled programs.
export set_device!, device_value

# Data. Prefetch is a framework default (setup wraps every split), so what is public is the
# override, the opt-out, and the two-method trait a model extends for real concurrency.
# `check_batch_at` is a test helper and stays qualified.
export PrefetchIterator, NoPrefetch
export batch_at, begin_epoch!

# Optimizer. `Decay` only: NO `WeightDecay` alias, which would collide with
# `Optimisers.WeightDecay` on the same concept.
export Decay

# Checkpointing. `checkpoint_filename` is exported for the same reason the hooks above
# are: a model package EXTENDS it.
export Checkpointer, TopKCheckpointer, save_checkpoint!, load_checkpoint, CheckpointRecord
# Selectors: which checkpoints `Checkpointer(; keep)` retains and a `weights` source picks.
export Selector, Best, Last, Epochs, Every, select, admits, Average, average_checkpoints
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
# Called from a hook to say what it is doing; the reporter draws it beside the phase.
export progress_note!, with_progress_note
export request_stop!, run_dir, current_step, current_epoch, phase
export experiment, parameters, states, EarlyStopping, should_stop
# The per-epoch series a handle keeps, as data that also displays as a table.
export history

# Export. ReactantServerExport exports `export_bundle`, `write_bundle`, `IOSpec` and
# `collect_provenance`, and is in scope whenever this surface is usable, so the three concepts
# here take names that cannot clash; the exports test checks the collision set.
export ExportSpec, ExportBackend, ReactantServerBundle, TFSavedModel
export export_inputs, export_outputs, export_preprocess, export_postprocess
export export_client_outputs, export_client_inputs, export_provenance_extra
export export_model, export_provenance, write_export, site_provenance

# Visualization. `visualize` and `save_figure` are hooks with no default method; `render` is the
# driver. The framework ships no plotting backend.
export visualize, save_figure, render

# Distribution stubs
export rank, world_size

# Installed here, not at the definition: a `Ref` filled during precompilation is filled in the
# precompiling process only. Nothing is drawn by this; the reporter decides per stretch of work.
function __init__()
    progress_reporter!(default_progress_reporter)
    # On by default: tracing cost is within noise and nothing is invalidated, and without it a
    # method that dispatches only on traced types can be redefined under a cached program
    # silently. Set the `precise_invalidation` preference to `false` to fall back to the inference
    # closure. Must run before anything is traced (see `TraceHook`).
    Preferences.load_preference(@__MODULE__, "precise_invalidation", true) && TraceHook.install!()
    return nothing
end

# LAST, after every name it drives is defined. The workload writes checkpoints, which is measurably
# the slowest first-call path in a run.
include("Precompile.jl")

end # module
