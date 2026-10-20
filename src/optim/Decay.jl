# Decay.jl
#
# Weight decay, decay toward an anchor, and the resume hazard that comes with the anchor.

"""
    Decay(lambda)
    Decay(lambda, anchor)
    Decay(lambda, anchor, no_decay_mask)

Decoupled weight decay toward an anchor: `apply!` adds `no_decay_mask * lambda * (x - anchor)` to
the update, or `no_decay_mask * lambda * x` when `anchor === nothing`. The anchor comes from the
group's [`decay_anchor`](@ref):

| `decay_anchor` | `anchor` | Decays toward |
| --- | --- | --- |
| `:zero` (default) | `nothing` | zero (ordinary weight decay) |
| `:w0` | that group's `w0` slice | the pretrained weights (L2-SP) |
| an array | that array | an explicit target |

A group whose [`lambda`](@ref) is zero gets no `Decay` in its chain. `anchor === nothing` is
resolved by type, so plain decay emits no subtraction. `Decay` must be the last rule in a chain,
after the base rule as in AdamW, so `lambda` arrives pre-multiplied by the group's learning rate;
any other position is a setup error. The anchor is a field rather than state captured at `init`,
which on a resume would capture the restored weights. There is no `WeightDecay` alias, since
`Optimisers.WeightDecay` already uses that name.
"""
struct Decay{L, W, M} <: Optimisers.AbstractRule
    lambda::L
    anchor::W          # nothing => decay toward zero
    no_decay_mask::M   # per-parameter exclusion
end

# `no_decay_mask` defaults to `true`, the broadcast identity; the framework always passes a real 0/1
# buffer.
Decay(lambda) = Decay(lambda, nothing, true)
Decay(lambda, anchor) = Decay(lambda, anchor, true)

# Stateless: the anchor is a field, never captured at `init`.
Optimisers.init(::Decay, x) = nothing

# `anchor === nothing` resolves at trace time, so plain decay emits no subtraction.
Optimisers.apply!(o::Decay, st, x, dx) = st, o.anchor === nothing ?
    @.(dx + o.no_decay_mask * o.lambda * x) :
    @.(dx + o.no_decay_mask * o.lambda * (x - o.anchor))

# Both remaining fields are parameter-sized, and a parameter-sized rule field is non-schedulable.
nonschedulable(::Type{<:Decay}) = (:anchor, :no_decay_mask)
