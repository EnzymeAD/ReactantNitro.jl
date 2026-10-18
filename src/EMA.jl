# EMA.jl
#
# An exponential moving average of the weights: a device copy of `ps` updated after every optimizer
# step by a small program of its own. Training never reads it; evaluation, checkpoint selection,
# `restore_best` and export read it under `eval_weights = :ema`.

"""
    EMA(beta; warmup = true, every = 1)

An exponential moving average of the weights, `θ_ema ← β θ_ema + (1 - β) θ`, updated after every
`every`-th optimizer step. `warmup` uses `min(beta, (1 + t) / (10 + t))` at optimizer step `t`, so
the average is not dragged by the initial weights for `1 / (1 - beta)` steps. Set it with the
[`ema`](@ref) accessor or the `Nitro` keyword:

```julia
ReactantNitro.ema(::MyExp) = EMA(0.999)
```

The average starts as a copy of the weights training starts from, is checkpointed in the record's
`ema` field and restored by a resume. Under `eval_weights = :ema` (the default with an EMA),
validation, checkpoint selection, `predict`, `restore_best` and export use it; training and
[`parameters`](@ref) use the raw weights. `Nitro(e; weights = ..., load_ema = true)` loads a
record's average in place of its weights. The average has the weights' element type.
"""
struct EMA
    beta::Float64
    warmup::Bool
    every::Int
end

function EMA(beta::Real; warmup::Bool = true, every::Integer = 1)
    0 <= beta <= 1 || error("ReactantNitro: `EMA($(beta))`: `beta` must be in [0, 1].")
    every >= 1 || error("ReactantNitro: `EMA(...; every = $(every))` must be at least 1.")
    return EMA(Float64(beta), warmup, Int(every))
end

Base.show(io::IO, x::EMA) = print(
    io, "EMA(", x.beta, x.warmup ? "" : "; warmup = false", x.every == 1 ? "" : "; every = $(x.every)", ")"
)

# β at optimizer step `t`, the step that just completed.
ema_beta(x::EMA, t::Integer) = x.warmup ? min(x.beta, (1 + t) / (10 + t)) : x.beta

"""
    ReactantNitro.ema_program(shadow, ps, beta) -> shadow

`shadow .= beta .* shadow .+ (1 - beta) .* ps`, leaf for leaf, written in place so the shadow is
allocated once and keeps its types across steps.
"""
function ema_program(shadow, ps, beta)
    Functors.fmap(shadow, ps) do s, p
        s isa AbstractArray && (s .= beta .* s .+ (one(beta) - beta) .* p)
        return s
    end
    return shadow
end

# One update after optimizer step `t`; returns the β used, or `nothing` when this step is skipped.
function ema_step!(nitro, t::Integer, memo)
    x = nitro.ema
    (nitro.ema_weights === nothing || t % x.every != 0) && return nothing
    T = eltype(first(Functors.fleaves(nitro.ps)))
    b = ema_beta(x, t)
    beta = memo_to_device(memo, (0, :ema_beta), T(b); mesh = nitro.mesh)
    thunk = compile_cached(
        ema_program, compile_view(nitro.e), nitro.ema_weights, nitro.ps, beta;
        phase = OptCompiling(), worlds = (), nitro
    )
    nitro.ema_weights = thunk(nitro.ema_weights, nitro.ps, beta)
    return b
end

"""
    ReactantNitro.eval_ps(nitro) -> parameters

The parameters evaluation uses: the EMA under `eval_weights = :ema`, else `nitro.ps`.
"""
eval_ps(nitro) =
    nitro.ema_weights !== nothing && nitro.eval_weights === :ema ? nitro.ema_weights : nitro.ps

# A device copy of a parameter tree, for a fresh shadow.
ema_copy(ps, mesh) = place_replicated(from_host(to_host(ps)), mesh)

# The `ema`, `eval_weights` and `load_ema` keywords, checked before anything is built; returns
# `eval_weights` resolved.
function check_ema(x, eval_weights, load_ema, weights)
    x === nothing || x isa EMA || error(
        "ReactantNitro: `ema` is an `EMA` or `nothing`; got a `$(typeof(x))`. Use `EMA(0.999)`."
    )
    w = something(eval_weights, x === nothing ? :raw : :ema)
    w in (:ema, :raw) || error(
        "ReactantNitro: `eval_weights` is `:ema` or `:raw`; got $(repr(w))."
    )
    w === :ema && x === nothing && error(
        "ReactantNitro: `eval_weights = :ema` evaluates the weights' moving average, and this \
         handle has none. Set `ema = EMA(0.999)`, or use `eval_weights = :raw`."
    )
    load_ema && weights === nothing && error(
        "ReactantNitro: `load_ema = true` loads a `weights` source's average, and no `weights` \
         was given."
    )
    return w
end

# A record whose `ps` is its EMA, for `load_ema = true`.
function ema_as_weights(record, source)
    record.ema === nothing && error(
        "ReactantNitro: `load_ema = true`, and the checkpoint `$(source)` keeps no EMA (the run \
         that wrote it had none)."
    )
    swap(f) = f === :ps ? record.ema : getfield(record, f)
    return CheckpointRecord(map(swap, fieldnames(CheckpointRecord))...)
end
