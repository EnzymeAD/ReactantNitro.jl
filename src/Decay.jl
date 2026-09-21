# Decay.jl
#
# Weight decay, decay toward an anchor, and the resume hazard that comes with the anchor.

"""
    Decay(lambda)
    Decay(lambda, anchor)
    Decay(lambda, anchor, no_decay_mask)

Weight decay and decay toward the pretrained weights as one rule, differing only in the anchor:

| Group configured with | `anchor` | `lambda` is the strength of |
| --- | --- | --- |
| `decay_anchor = :zero` (default) | `nothing` | decay toward zero, i.e. ordinary weight decay |
| `decay_anchor = :w0` | that group's `w0` slice | decay toward the pretrained weights |
| `decay_anchor = <array>` | that array | decay toward an explicit target |
| neither | rule omitted | nothing; no `Decay` in the chain |

`anchor === nothing` is a type-level distinction, so plain decay emits no subtraction ops. There is
deliberately no `WeightDecay` alias, which would collide with `Optimisers.WeightDecay` on the same
concept. Decoupled only: `Decay` goes after the base rule, matching AdamW, so `lambda` arrives
pre-multiplied by `η_g(t)`; anywhere but last in a chain is a setup error. The anchor is an explicit
field, never captured at `init`, which on resume would capture the restored weights.
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
