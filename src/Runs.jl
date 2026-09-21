# Runs.jl
#
# `run_state` is the machine-readable sibling of the `nitro_status` gate tool, so code (a wait
# predicate, a harness, a test) can decide on a run's facts without parsing prose. The registry
# lives in the KaimonGate extension; the function exists here so the extension has something to
# add a method to.

"""
    run_state(run_id) -> NamedTuple

The current state of a run started through a `nitro_*` gate tool, as data: `id`, `kind`,
`experiment`, `status` (`:running`, `:completed`, `:failed`), `phase`, `epoch`, `step`, `loss`,
`val_metrics`, `run_dir`, `error`, `result`, and `elapsed_s`. Not exported; call it as
`ReactantNitro.run_state`. Throws for an unknown id and when no KaimonGate extension is loaded.
The intended use is a wait predicate: `run_state("7928bb0e").status !== :running`.
"""
function run_state(run_id)
    error(
        "ReactantNitro.run_state: the run registry lives in the KaimonGate extension, and it is not \
         loaded in this process. In a Kaimon-hosted session it loads with `using KaimonGate`; the \
         `nitro_*` tools create the runs this reports on."
    )
end
