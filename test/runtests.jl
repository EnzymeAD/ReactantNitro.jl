# ReactantNitro test suite.
#
# Each acceptance test carries an ID from the suite's own numbering. The mutating-state tests
# (lifecycle, checkpoint, config, data, eval, defaults, and the export skeleton) were built first,
# followed by the optimizer, accumulation, data, and cache tests.
#
# CPU autodiff WORKS through Reactant's CPU plugin (58/58 nonzero gradients), so almost everything
# here runs in CI without a GPU. Do not inherit a `_HAS_GPU` skip gate from an earlier framework
# without re-scoping it.
#
# Two tests are controls and are the most important in the suite: one asserts the suite can
# detect a frozen step counter at all, and the other asserts it can detect unbarriered-state
# gradient corruption. A control that passes when the bug is present is worse than no test.
#
# Most of this suite is CPU-only by design: CPU autodiff works through Reactant's CPU plugin, which
# is why the whole suite runs in CI. Hiding the GPUs is not belt-and-braces: on a shared training
# host, letting Reactant pick up a visible device would collide with somebody else's run.
#
# ReTestItems drives the suite. Each file in test/ is a `*_tests.jl` of `@testitem`s, one
# testitem per file, so intra-file test order is preserved while ReTestItems spreads items
# across worker processes. The worker count defaults to half the cores (the measured sweet
# spot for this suite: XLA compiles are multi-threaded, and more processes than that just
# duplicate compile work and RAM), capped by the number of test files, and overridable with
# RETESTITEMS_NWORKERS. `retries` is a load-flake safety net (a prefetch cleanup race was that
# class of bug); set RETESTITEMS_RETRIES=0 to disable. `validate_paths` turns the silent
# skip of an unrenamed test file into an error.
ENV["CUDA_VISIBLE_DEVICES"] = ""

using ReTestItems
using ReactantNitro

runtests(
    ReactantNitro;
    nworkers = parse(
        Int, get(
            ENV, "RETESTITEMS_NWORKERS",
            string(clamp(max(1, Sys.CPU_THREADS ÷ 2), 1, 18))
        )
    ),
    worker_init_expr = :(ENV["CUDA_VISIBLE_DEVICES"] = ""),
    retries = parse(Int, get(ENV, "RETESTITEMS_RETRIES", "2")),
    validate_paths = true,
)
