"""
    NitroTestKit

The test suite's shared toy model and the precompile workload that warms the framework for it.

**Why a package rather than a `@testsetup`.** A test item pays the framework's Julia compile every
time it defines a new experiment type: inference of `train!`, the routing, the prefetch pipeline
and the checkpoint path all specialize on the type, and Enzyme and Reactant trace the programs for
it. Measured at 94 to 98 percent of each heavy item's wall time. Specializations a package's
workload triggers are cached in THAT package's pkgimage, including specializations of another
package's methods, so a test-only package that runs one training epoch of a shared type at
precompile time hands every worker those specializations for free. It lives under `test/` and is
reached through the root `Project.toml`'s `[sources]`, so a regular user's precompile is untouched.

**What to use from it.** `KitMLP` is the experiment, with a `GraphConst`, a `Device` and a `Host`
field, `metrics` with one scalar and one summed matrix, `train_metrics`, and decoupled decay
anchored at `:w0`, so the common paths are all warm. `KIT_TRAIN` and `KIT_VAL` are its data,
`kit_batches` makes more, and `kit_nitro` builds a handle in a fresh temporary run directory. A
test item that trains `KitMLP` starts warm; one that defines its own type pays the compile the
kit could not do for it, and that is the price of testing a type-specific behaviour.
"""
module NitroTestKit

using ReactantNitro
using Lux, Random, Statistics

export KitMLP, KIT_TRAIN, KIT_VAL, kit_batches, kit_nitro

@experiment struct KitMLP
    "Structural, so a change recompiles and the mismatch checks have something to refuse."
    width::GraphConst{Int} = 6
    "A traced input, so the Device path is warm; zero, so it changes no number by default."
    smoothing::Device{Float32} = 0.0f0
    max_epochs::Int = 1
end

kit_chain(w) = Lux.Chain(Lux.Dense(4 => w, tanh), Lux.Dense(w => 2))
ReactantNitro.build_model(e::KitMLP, rng) = (m = kit_chain(e.width); (m, Lux.setup(rng, m)...))
ReactantNitro.forward(::KitMLP, model, ps, st; x) = Lux.apply(model, x, ps, st)
ReactantNitro.loss(e::KitMLP, ŷ; y) = mean(abs2, ŷ .- y) + e.smoothing * mean(abs2, ŷ)
# One scalar metric and one summed matrix, so both halves of the `(sum, count)` contract are
# exercised: `mae` is divided at the end, `cm` never is.
ReactantNitro.metrics(::KitMLP, ŷ; y) =
    (; mae = (sum(abs, ŷ .- y), size(y, 2)), cm = (ones(Int, 2, 2), nothing))
ReactantNitro.train_metrics(::KitMLP, ŷ; y) = (; batch_mae = mean(abs, ŷ .- y))
ReactantNitro.learning_rate(::KitMLP) = 1.0f-2
# Decay toward `w0`, so the anchor machinery is warm and observable through the checksum.
ReactantNitro.lambda(::KitMLP) = 1.0f-3
ReactantNitro.decay_anchor(::KitMLP, ::Val{:default}) = :w0

"""
    kit_batches(n; seed = 3) -> Vector{NamedTuple}

`n` batches of `(; x, y)` with eight samples each, `x` of size `(4, 8)` and `y` of size `(2, 8)`,
from a seeded generator so two calls with the same seed agree.
"""
function kit_batches(n; seed = 3)
    rng = Random.MersenneTwister(seed)
    return [(; x = randn(rng, Float32, 4, 8), y = randn(rng, Float32, 2, 8)) for _ in 1:n]
end

const KIT_TRAIN = kit_batches(4)
const KIT_VAL = kit_batches(2; seed = 4)
ReactantNitro.build_data(::KitMLP, dist) = (; train = KIT_TRAIN, val = KIT_VAL)

"""
    kit_nitro(e = KitMLP(); kw...) -> Nitro

A handle in a fresh temporary run directory, checkpointing on `mae`. Any keyword overrides.
"""
kit_nitro(e = KitMLP(); kw...) = Nitro(
    e; run_dir = mktempdir(), checkpointer = TopKCheckpointer(; metric = :mae, mode = :min),
    kw...
)

include("precompile.jl")

end
