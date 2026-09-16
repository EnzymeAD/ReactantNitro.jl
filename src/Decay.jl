# Decay.jl
#
# Weight decay, decay toward an anchor, and the resume hazard that comes with the anchor.

"""
    Decay(lambda)
    Decay(lambda, anchor)
    Decay(lambda, anchor, no_decay_mask)

Weight decay and decay-toward-the-pretrained-weights as **one rule**, differing only in the anchor:

| Group configured with | `anchor` | `lambda` is the strength of |
| --- | --- | --- |
| `decay_anchor = :zero` (default) | `nothing` | decay toward zero, i.e. ordinary weight decay |
| `decay_anchor = :w0` | that group's `w0` slice | decay toward the pretrained weights |
| `decay_anchor = <array>` | that array | decay toward an explicit target |
| neither | rule omitted | nothing; no `Decay` in the chain |

`anchor === nothing` is a **type-level** distinction, so the branch resolves at trace time: plain
decay emits no subtraction ops and carries no parameter-sized zero buffer.

**`Decay` is the only name exposed, and there is deliberately no `WeightDecay` alias**, which would
collide with `Optimisers.WeightDecay` on the same concept. Name collisions are usually tolerable
because users qualify, but that argument holds for `train!` and `loss`, where the two meanings are
unrelated; here the two names would mean the same concept with different implementations, which is
the case where a collision genuinely misleads.

The field is `no_decay_mask` rather than `mask` because "mask" once meant two unrelated things: this
per-parameter exclusion (norm affines, biases, 1-D parameters), which stays automatic, and a deleted
per-group partition. Only this one survives.

**Decoupled only**: `Decay` goes **after** the base rule, matching AdamW, so it is decoupled
from the adaptive per-parameter scaling but still scaled by the learning rate. `lambda` therefore
arrives **pre-multiplied by `η_g(t)`**, and an LR schedule modulates regularization strength.
`Decay` before the base rule would feed the decay term through moment estimation, producing coupled
L2, which the framework does not support. **`Decay` anywhere but last in a chain is an error**,
checked at setup on the chain the factory returns.

**The anchor is an explicit field, never captured at `init`.** An `init`-captured anchor would grab
whatever `x` is at optimizer setup, which on resume is the *restored* weights.
"""
struct Decay{L, W, M} <: Optimisers.AbstractRule
    lambda::L
    anchor::W          # nothing => decay toward zero
    no_decay_mask::M   # per-parameter exclusion
end

# `no_decay_mask` defaults to `true`, the broadcast identity, so `Decay(λ)` decays every parameter.
# The framework always passes a real 0/1 buffer, the automatic exclusion of norm affines, biases,
# and 1-D parameters; the default is for a hand-written Level 2 chain that wants none.
Decay(lambda) = Decay(lambda, nothing, true)
Decay(lambda, anchor) = Decay(lambda, anchor, true)

# Stateless: the anchor is an explicit FIELD, never captured at `init`. An `init`-captured anchor
# would grab whatever `x` is at optimizer setup, which on resume is the RESTORED weights.
Optimisers.init(::Decay, x) = nothing

# `anchor === nothing` is a TYPE-level distinction, so this branch resolves at trace time: plain
# decay emits no subtraction ops and carries no parameter-sized zero buffer.
Optimisers.apply!(o::Decay, st, x, dx) = st, o.anchor === nothing ?
    @.(dx + o.no_decay_mask * o.lambda * x) :
    @.(dx + o.no_decay_mask * o.lambda * (x - o.anchor))

# Both remaining fields are parameter-sized, and the general rule is that any parameter-sized rule
# field is non-schedulable. Scheduling one would push a parameter-sized buffer to device every
# optimizer step, which is exactly what the schedule layer forbids.
nonschedulable(::Type{<:Decay}) = (:anchor, :no_decay_mask)
