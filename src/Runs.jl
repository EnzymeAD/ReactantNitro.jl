# Runs.jl
#
# `run_state` is the machine-readable sibling of the `nitro_status` gate tool. The tool renders a
# run for a person (and for the model reading a tool result); this returns the same facts as a
# NamedTuple so that CODE can decide on them: a wait predicate (`run_state(id).epoch >= 3`), a
# harness that drives several runs, a test. Anything that had to regex `nitro_status` text was one
# format change away from breaking, and a predicate that parses prose is not a predicate.
#
# The run registry lives in the KaimonGate extension, because a "run" in this sense is a thing a
# session started through `nitro_train` and friends. The function exists here, with a method that
# says so, so that it can be named in a session before the extension has loaded and so the
# extension has a function to add a method to; a Julia extension can only extend what already
# exists. Same arrangement as `register_phase_monitor!` for the monitors.

"""
    run_state(run_id) -> NamedTuple

The current state of a run started in this session through a `nitro_*` gate tool, as data:

- `id`, `kind` (`:train`, `:validate`, `:evaluate`, `:predict`, `:export`), `experiment`
- `status`: `:running`, `:completed` or `:failed`
- `phase`: the latest phase name (`"TrainStepping"`, ...), `""` before the first
- `epoch`, `step`: `nothing` until the loop has begun
- `loss`: the latest train loss, `nothing` until one was logged
- `val_metrics`: the latest validation metrics NamedTuple, or `nothing`
- `run_dir`, `error`, `result`: as `nitro_status` shows them
- `elapsed_s`: seconds since the run started, to its finish once it has one

Not exported, because the export surface is pinned: call it as `ReactantNitro.run_state`. Throws
for an unknown id, and throws when no KaimonGate extension is loaded, since without it there is no
registry to ask. The intended use is a wait predicate evaluated by something that polls for
you: `run_state("7928bb0e").status !== :running`, `run_state("7928bb0e").epoch >= 3`.
"""
function run_state(run_id)
    error(
        "ReactantNitro.run_state: the run registry lives in the KaimonGate extension, and it is not \
         loaded in this process. In a Kaimon-hosted session it loads with `using KaimonGate`; the \
         `nitro_*` tools create the runs this reports on."
    )
end
