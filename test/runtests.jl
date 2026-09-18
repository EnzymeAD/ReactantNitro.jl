ENV["CUDA_VISIBLE_DEVICES"] = ""

using ReTestItems
using ReactantNitro

nworkers = parse(
    Int, get(
        ENV, "RETESTITEMS_NWORKERS",
        string(clamp(max(1, Sys.CPU_THREADS ÷ 2), 1, 18))
    )
)
worker_init = nworkers > 0 ? (worker_init_expr = :(ENV["CUDA_VISIBLE_DEVICES"] = ""),) : (;)

runtests(
    ReactantNitro;
    nworkers,
    worker_init...,
    retries = parse(Int, get(ENV, "RETESTITEMS_RETRIES", "2")),
    validate_paths = true,
)
