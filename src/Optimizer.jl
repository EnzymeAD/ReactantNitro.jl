# Optimizer.jl
#
# The optimizer layer: device-resident optimizer state, the traced-rule allowlist, the flat
# per-group parameter layout, the three configuration levels, and framework-owned clipping.
#
# ── The governing fact of the whole design ──────────────────────────────────────────
#
# Stock `Optimisers.jl` rules trace correctly and match the host path BITWISE, but only if the
# framework normalizes the optimizer state to device residency first, INCLUDING its integer step
# counters. Left alone, `Optimisers.setup` produces state with host `Int` fields; those bake as
# trace-time constants, the counter freezes, and the run silently trains with the wrong bias
# correction.
#
# Re-measured here (Optimisers 0.4.8, Reactant 0.2.270, Julia 1.12.6, CPU), 64-element
# flat buffer, 3 steps, RAdam: with the counter device-resident the result is bitwise identical to
# the host path and the state is a type fixed point. With `track_numbers = AbstractFloat`, leaving
# `t` host, `t` freezes at 2 where the host reaches 4 and the parameters drift 1.3e-3. NO ERROR IS
# RAISED AT ANY POINT. The optimizer tests carry the control that asserts the second half.

# ── Framework-owned state initialization ────────────────────────────────────────────

"""
    ReactantNitro.is_host_number(x) -> Bool

A `Number` that is not device-resident. `Reactant.RNumber` is the common supertype of
`ConcretePJRTNumber` and `TracedRNumber`, so this one predicate covers both sides of a traced
boundary, which is what [`assert_device_state`](@ref) needs.
"""
is_host_number(x) = x isa Number && !(x isa Reactant.RNumber)

"""
    ReactantNitro.host_numbers(x) -> Vector{String}

Every host `Number` reachable from `x`, as `path::Type` strings. The paths are what makes the
assertion's failure legible: "which leaf is still on the host" is the only useful thing to say.

`Optimisers.Leaf.frozen` is skipped: it is declared `::Bool` and is not a type parameter, so it
**cannot** be promoted, which is the same fact that forces the rule and the state to convert
separately.
"""
function host_numbers(x, path = "", out = String[])
    if x isa Reactant.RNumber || x isa AbstractArray || x isa AbstractString || x isa Symbol
        return out
    elseif is_host_number(x)
        push!(out, string(path, "::", typeof(x)))
    elseif x isa Optimisers.Leaf
        host_numbers(x.rule, path * ".rule", out)
        host_numbers(x.state, path * ".state", out)
    elseif x isa Union{Tuple, NamedTuple}
        for (k, v) in pairs(x)
            host_numbers(v, string(path, ".", k), out)
        end
    elseif isstructtype(typeof(x))
        for f in fieldnames(typeof(x))
            host_numbers(getfield(x, f), string(path, ".", f), out)
        end
    end
    return out
end

"""
    ReactantNitro.assert_device_state(state; context) -> nothing

The no-host-`Number` assertion over an optimizer **state** tree, run both at setup and after a
restore. **This is the only thing standing between a user and the silent failure this file opens
with**, which is why it is a runtime check and not only a test.

It applies to the state, not to the rule. A rule legitimately keeps host fields, namely every field
in [`nonschedulable`](@ref); that side is governed by [`to_device_rule`](@ref) and is asserted
separately.
"""
function assert_device_state(state; context::AbstractString = "optimizer state")
    leaked = host_numbers(state)
    isempty(leaked) && return nothing
    error(
        """
        ReactantNitro: $context still holds host `Number` values after normalization, at:
            $(join(leaked, "\n    "))
        A host number in optimizer state BAKES as a trace-time constant: an integer step
        counter left on the host freezes, the bias correction goes wrong, and the run
        trains to a plausible-looking loss curve with no error ever being raised. Convert through
        `to_device_leaf`, which uses `track_numbers = Number` rather than `AbstractFloat`."""
    )
end

"""
    ReactantNitro.assert_opt_state_device(opt_state; context) -> nothing

The no-host-`Number` assertion over a USER-SUPPLIED optimizer state tree, the manual training
mode, for [`setup_optimizers`](@ref)'s result and its restored form: walk every `Optimisers.Leaf`
reachable from the tree and assert the leaf's STATE.

The automatic path asserts each group's `state` alone, because `build_opt_state` built the leaves.
A user tree carries the leaves' RULES too, and a rule legitimately keeps host fields, namely every
field in [`nonschedulable`](@ref), so the walk asserts states only, never rules: that is the same
boundary `to_device_leaf` preserves. `host_numbers` already skips `Leaf.frozen` explicitly.
"""
function assert_opt_state_device(
        x, context::AbstractString = "optimizer state from `setup_optimizers`"
    )
    if x isa Optimisers.Leaf
        assert_device_state(x.state; context)
    elseif x isa Union{Tuple, NamedTuple}
        for v in x
            assert_opt_state_device(v, context)
        end
    elseif x isa AbstractArray
        nothing
    elseif isstructtype(typeof(x)) && !(x isa AbstractString) && !(x isa Symbol)
        for f in fieldnames(typeof(x))
            assert_opt_state_device(getfield(x, f), context)
        end
    end
    return nothing
end

"""
    ReactantNitro.to_device_leaf(l::Optimisers.Leaf) -> Optimisers.Leaf

    Optimisers.Leaf(to_device_rule(l.rule),
                    Reactant.to_rarray(l.state; track_numbers = Number),
                    l.frozen)

**The rule and the state convert separately.** A single `to_rarray(leaf; track_numbers = Number)`
fails with `Reactant.NoFieldMatchError` (measured), because `Optimisers.Leaf.frozen` is declared
`::Bool` and is not a type parameter, so promoting it to `TracedRNumber{Bool}` cannot be
reconstructed.

The state uses a blanket `track_numbers = Number`, because it **must** promote integers, which is
the whole point of the fact this file opens with, and carries no `Bool`.

The same normalization runs on the **resume** path, over the restored `opt_state`. The record
stores **host** values, so the flow is uniform and has exactly one normalization point per path:
write host, read host, normalize on the way in. Skipping it on resume reacquires the frozen-counter
bug in full, and `resume = :auto` is the default.
"""
function to_device_leaf(l::Optimisers.Leaf; mesh = nothing)
    return Optimisers.Leaf(
        to_device_rule(l.rule; mesh),
        place_replicated(l.state, mesh; track_numbers = Number),
        l.frozen
    )
end
to_device_leaf(t::Union{Tuple, NamedTuple}; mesh = nothing) =
    map(x -> to_device_leaf(x; mesh), t)

"""
    ReactantNitro.to_device_rule(r) -> rule

Promote exactly the schedulable fields; leave structural ones host.

**The rule promotes per field, not by type**, and that is not a stylistic choice: **a type-based
policy cannot tell a schedulable hyperparameter from a structural one**, since they are often the
same type. `ClipNorm` is the worked case. Measured here: its fields are *declared* `Any, Any, Bool`
and a default `ClipNorm()` is `ClipNorm{Float64,Float64}(10.0, 2.0, true)`, so the schedulable
`omega` and the structural `p` are **the same type** and no type-based policy can separate them even
in principle. Any policy admitting `omega` also promotes `p`, which then fails `_norm`'s `::Real`
dispatch constraint. That failure is what once made this project record "ClipNorm cannot trace at
all", which was wrong in both halves.

**The set to promote is exactly `fieldnames(T)` minus [`nonschedulable`](@ref)`(T)`**, which is the
trait the schedule layer already declares. One declaration drives both, and it must: a field that
cannot be promoted cannot be scheduled, and a field that is scheduled must be promoted.

A parameter-sized field such as [`Decay`](@ref)'s `anchor` and `no_decay_mask` is in that opt-out
and is therefore passed through untouched. That is correct rather than a gap: the framework
constructs `Decay` with those already on device, from the per-group `hp` table, and
"non-schedulable" means "not pushed afresh every step", which is exactly what a parameter-sized
buffer must not be.

**The name is `to_device_rule`, not `promote_rule`.** `promote_rule` is exported from `Base`, and
defining it unqualified at module level does not error but makes `Base.promote_rule` unreachable by
that name inside the module.
"""
function to_device_rule(r; mesh = nothing)
    T = typeof(r)
    ns = nonschedulable(T)
    return T.name.wrapper(
        map(fieldnames(T)) do f
            v = getfield(r, f)
            f in ns ? v : place_replicated(v, mesh; track_numbers = Number)
        end...
    )
end
to_device_rule(c::Optimisers.OptimiserChain; mesh = nothing) =
    Optimisers.OptimiserChain(map(o -> to_device_rule(o; mesh), c.opts)...)

# ── The allowlist ───────────────────────────────────────────────────────────────────

"""
    ReactantNitro.ADMITTED_RULES

The traced-optimizer allowlist. Membership criterion, three parts, all mechanically checkable: the
rule **traces** with device-resident state and hyperparameters; its state is a **type fixed point**
across the traced boundary; and three steps are **bitwise identical** to the stock host path.

Re-measured under the per-field normalization above, 64-element flat buffer, 3 steps,
every rule constructed at the buffer's element type:

| Rule | Traces | Fixed point | maxdiff vs host |
| --- | --- | --- | --- |
| `Descent` | yes | yes | 1 ulp, see below |
| `Momentum`, `Nesterov` | yes | yes | 0.0 |
| `Adam`, `AdamW`, `RAdam` | yes | yes | 0.0 |
| `WeightDecay`, `ClipGrad` | yes | yes | 0.0 |
| `ClipNorm(ω, p; throw = false)` | yes | yes | 0.0 |
| `OptimiserChain(RAdam, WeightDecay)` | yes | yes | 0.0 |
| `OptimiserChain(ClipNorm, Adam)` | yes | yes | 0.0 |

**`Descent` is 1 ulp where every other admitted rule is exact.** Measured and diagnosed rather than
tolerated: `Descent`'s update is `x - eta * g`, a multiply immediately consumed by a subtract, which
XLA is free to contract into a **fused multiply-add**. FMA rounds once where the host rounds twice.
Three predictions were checked and all hold: with `eta` an exact power of two the multiply is exact
and the result is bitwise; the difference is one ulp and does not accumulate over steps; and the
compiled result equals host `fma.(-eta, g, x)` **exactly**. Every other admitted rule has a division
or a sqrt between the multiply and the subtract, which blocks contraction, which is why they are
bitwise.

So the optimizer tests assert bitwise equality against the **FMA-contracted** host reference for
`Descent` and against the plain host path for everything else. That keeps an exact assertion rather
than a tolerance, so the test still fails loudly if XLA's behavior changes, which is the whole
reason for asserting bitwise rather than to a tolerance.

**`AccumGrad` is deliberately absent**, and it is the one entry that traces and is *wrong*: see
`Train.jl` for the accumulation the framework does instead.

**Elementwise rules only.** Layer-adaptive rules (LARS, LAMB) would compute one trust ratio per group
rather than per layer, a behavioral difference that surfaces as a bad curve rather than an error, so
they are rejected even though nothing about them fails to trace.
"""
const ADMITTED_RULES = (
    Optimisers.Descent, Optimisers.Momentum, Optimisers.Nesterov,
    Optimisers.Adam, Optimisers.AdamW, Optimisers.RAdam,
    Optimisers.WeightDecay, Optimisers.ClipGrad, Optimisers.ClipNorm,
    Decay,
)

"""
    ReactantNitro.check_allowlist(rule) -> nothing

The admission check, enforced at setup. Its error names the admitted rules and points at the
upstream contribution path, because that error is the only runtime signal a user gets.

`ClipNorm` with the default `throw = true` is rejected **naming the flag**. It reaches
`if o.throw && !isfinite(nrm)`, a host branch on a traced value, and fails with
`TypeError: non-boolean (Reactant.TracedRNumber{Bool}) used in boolean context`, which names nothing
a user could act on. Measured, so the rejection is not precautionary.
"""
function check_allowlist(c::Optimisers.OptimiserChain)
    foreach(check_allowlist, c.opts)
    return nothing
end

function check_allowlist(rule)
    R = typeof(rule)
    if rule isa Optimisers.ClipNorm && rule.throw
        error(
            """
            ReactantNitro: `ClipNorm` is admitted only with `throw = false`, and this one has
            `throw = true`, which is the constructor default. With `throw = true` the rule reaches
            `if o.throw && !isfinite(nrm)`, a host branch on a traced value, and tracing fails with
            `TypeError: non-boolean (TracedRNumber{Bool}) used in boolean context`, which names
            neither the rule nor the flag. Write `ClipNorm(omega, p; throw = false)`.
            Note the framework's own gradient clip is global rather than per group and needs no
            chain member at all; reach for `ClipNorm` only if you specifically
            want per-group clipping in a Level 2 chain."""
        )
    end
    any(A -> rule isa A, ADMITTED_RULES) && return nothing
    error(
        """
        ReactantNitro: `$(nameof(R))` is not on the traced-optimizer allowlist.
        Admitted: $(join(nameof.(ADMITTED_RULES), ", ")), and `OptimiserChain`s of them.
        A rule is admitted once it is measured to trace with device-resident state, to be a type
        fixed point across the traced boundary, and to match the stock host path over three steps.
        If `$(nameof(R))` meets those, adding it is a one-line change plus a row in the optimizer
        tests' bitwise-equality table.
        Two known exclusions are deliberate rather than untested: `Optimisers.AccumGrad` traces and
        is numerically WRONG under tracing, and layer-adaptive rules such as LARS and LAMB would
        compute one trust ratio per parameter GROUP rather than per layer."""
    )
end

# ── Flat parameters, per-group slices ───────────────────────────────────────────────

"""
    ReactantNitro.LeafRow

One row of the flat permutation, which is a per-leaf table and **not** a `Vector{Int}` over
elements. `offset` is 0-based **within that leaf's group buffer**, since the framework materializes
G disjoint per-group arrays rather than one buffer with a mask.

Storing keypaths rather than raw offsets is what lets the resume check produce a **useful** diff:
it can say which leaf moved, rather than that two integer vectors differ. It is also small, one row
per parameter array rather than per element.
"""
const LeafRow = @NamedTuple{keypath::Tuple, group::Symbol, offset::Int, len::Int, size::Dims}

"""
    ReactantNitro.FlatLayout{G}

The flat-to-tree correspondence, built once at setup and closed over as a host-side constant for
the run. `G` is a type parameter because it is a trace-time constant: per-group
arithmetic then unrolls with no traced control flow, and the per-group collections are `NTuple{G}`
rather than `Vector`, which a thunk guard would reject on the second call for being abstractly typed.

`permutation` is the documented, checkpointed table; everything else is a derived lookup.
"""
struct FlatLayout{G}
    permutation::Vector{LeafRow}
    groups::NTuple{G, Symbol}
    lengths::NTuple{G, Int}
    index::Dict{Tuple, Int}                     # keypath -> row number in `permutation`
    # Row numbers per group, in flat order, as TUPLES rather than Vectors. That is not a
    # micro-optimization: `map` over a host `Vector` INSIDE a traced function is intercepted by
    # Reactant's overlay and vectorized into `elem_apply_via_while_loop`, which promotes the row
    # indices to `TracedRNumber{Int64}` and then fails with `getindex for Vector{Any} with types
    # Tuple{TracedRNumber{Int64}} is not supported`. `map` over a Tuple is Base's own and stays
    # host, which is what keeps the operand order host-known at trace time.
    group_rows::NTuple{G, Tuple{Vararg{Int}}}
    row_group::Vector{Int}                     # row number -> group number
end

n_groups(::FlatLayout{G}) where {G} = G

"""
    ReactantNitro.LayoutRef{ID}

A zero-field handle carrying a [`FlatLayout`](@ref) at the **type** level, the same technique
[`Router`](@ref) uses for its key set, and for the same reason plus one more.

**The layout cannot ride into a traced program as a value.** Measured: Reactant traverses
a struct argument and promotes the `Int`s inside it, so `layout.index[keypath]` comes back a
`TracedRNumber{Int64}` and indexing the leaf vector fails with `getindex for Vector{Any} with types
Tuple{Reactant.TracedRNumber{Int64}} is not supported`. The layout is **host bookkeeping**: both
directions of the map are closed over as host-side constants for the run, and a promoted offset is
neither host nor a constant.

Carrying it at the type level also puts it in the compile cache's key for free, through the
argument types,
which is what a `FlatLayout{G}` value argument would **not** have done: two different layouts with the
same `G` share a type, so they would have collided on a key that was otherwise identical.
"""
struct LayoutRef{ID} end

const LAYOUT_REGISTRY = Dict{UInt, Any}()
const LAYOUT_REGISTRY_LOCK = ReentrantLock()

"""
    ReactantNitro.layout_ref(layout) -> LayoutRef

Register `layout` under a content hash and return its type-level handle. The hash is over the
documented `permutation` table, so two structurally identical layouts share a handle and therefore a
compiled program, which is what lets a second `Nitro` over the same experiment reuse everything.
"""
function layout_ref(layout::FlatLayout)
    id = hash(layout.permutation)
    lock(LAYOUT_REGISTRY_LOCK) do
        LAYOUT_REGISTRY[id] = layout
    end
    return LayoutRef{id}()
end

@inline resolve_layout(::LayoutRef{ID}) where {ID} = LAYOUT_REGISTRY[ID]::FlatLayout

"""
    ReactantNitro.leaf_keypaths(ps) -> Vector{Tuple}

The traversal order: `Functors.fmap_with_path` over `ps`, depth-first and deterministic for a
fixed model, which is what makes the layout reproducible across sessions and processes rather than a
property of the framework. Verified on Functors 0.5.2 against a nested Lux `Chain`.
"""
function leaf_keypaths(ps)
    kps = Tuple[]
    Functors.fmap_with_path(ps) do kp, x
        x isa AbstractArray && push!(kps, Tuple(kp))
        return x
    end
    return kps
end

"""
    ReactantNitro.build_layout(e, ps) -> FlatLayout

Setup's layout step, **unconditional**, including for an evaluation `Nitro`: it is host-side
bookkeeping rather than device memory, and the checkpoint path validates the restored permutation
against the one computed here, which a `Nitro` that skipped this step could not do.

**The layout is a stable sort of the traversal-order leaves by group index.** Stability is what makes
it deterministic: within a group, leaves keep traversal order. `:default` is group 1 and the
remaining groups follow first-appearance order, which is stable for a fixed model and bakes into the
compiled program.
"""
function build_layout(e, ps)
    kps = leaf_keypaths(ps)
    sizes = Dims[]
    Functors.fmap_with_path(ps) do kp, x
        x isa AbstractArray && push!(sizes, size(x))
        return x
    end
    isempty(kps) && error("ReactantNitro: `build_model` returned a parameter tree with no arrays in \
                           it, so there is nothing to optimize.")

    gs = Symbol[param_group(e, kp) for kp in kps]
    names = unique(gs)
    :default in names && (names = vcat(:default, filter(!=(:default), names)))
    groups = Tuple(names)
    G = length(groups)
    gnum = Dict(n => i for (i, n) in enumerate(names))

    order = sortperm(gs; by = g -> gnum[g], alg = Base.Sort.MergeSort)  # stable

    rows = Vector{LeafRow}(undef, length(order))
    offsets = zeros(Int, G)
    row_group = Vector{Int}(undef, length(order))
    for (r, i) in enumerate(order)
        gi = gnum[gs[i]]
        len = prod(sizes[i])
        rows[r] = (; keypath = kps[i], group = gs[i], offset = offsets[gi], len, size = sizes[i])
        row_group[r] = gi
        offsets[gi] += len
    end

    index = Dict{Tuple, Int}(rows[r].keypath => r for r in eachindex(rows))
    group_rows = ntuple(gi -> Tuple(r for r in eachindex(rows) if row_group[r] == gi), G)
    return FlatLayout{G}(rows, groups, Tuple(offsets), index, group_rows, row_group)
end

"""
    ReactantNitro.flatten(tree, layout) -> NTuple{G}

Tree to G group buffers. **A concatenate**, which is a real copy of the parameter set rather than a
fold into a consumer, so calling it free would be wrong.

What it is *not* is a runtime permuted gather: because the permutation is one row per leaf rather
than one entry per element, the operand order is host-known at trace time, so grouping by leaf costs
nothing beyond the copy. That is also why this runs inside the traced program without emitting
control flow.
"""
function flatten(tree, layout::FlatLayout{G}, mesh = nothing) where {G}
    leaves = Vector{Any}(undef, length(layout.permutation))
    Functors.fmap_with_path(tree) do kp, x
        if x isa AbstractArray
            r = get(layout.index, Tuple(kp), 0)
            r == 0 && error("ReactantNitro: leaf `$(Tuple(kp))` is not in the flat layout. The tree \
                             being flattened has a different structure from the one `build_layout` \
                             saw.")
            leaves[r] = x
        end
        return x
    end
    return ntuple(gi -> _concat_group(leaves, layout.group_rows[gi], mesh), Val(G))
end

# CONCRETE DEVICE ARRAYS GO THROUGH THE HOST, and this is not an optimization, it is the only way
# this function works on a GPU at all.
#
# `vec` of a `ConcretePJRTArray` is a `Base.ReshapedArray` wrapping the device buffer, which is not a
# Reactant array type, so `vcat` of those misses Reactant's methods and lands in Base's generic
# `typed_vcat`. That allocates the destination and fills it with `setindex!`, elementwise. On host
# arrays it is legal and merely slow; on a GPU Reactant refuses:
#
#     ERROR: Scalar indexing is disallowed.
#     Invocation of getindex(::ConcretePJRTArray, ::Vararg{Int,N})
#
# Found by the first real GPU run, at `flat = flatten(ps, layout)` during `Nitro`
# construction, before step 1. THE FLAT PARAMETER LAYOUT COULD NOT BE BUILT ON A GPU, so nothing in
# this framework had ever run on one. 1243 green CPU tests could not have caught it: the same code
# passes on CPU while doing something pathological, so the suite was green and wrong.
#
# The host round trip is affordable because THIS PATH IS SETUP-ONLY. `flatten` sees concrete device
# arrays exactly twice per run, at `flat = flatten(ps, layout)` and in `decay_anchors`, both once
# during construction. Inside the traced programs the leaves are `TracedRArray`s, which are not
# `AbstractConcreteArray`, so they take the branch below unchanged and the hot path is untouched.
function _concat_group(leaves, rows::Tuple, mesh = nothing)
    xs = ntuple(i -> leaves[rows[i]], length(rows))
    if any(x -> x isa Reactant.AbstractConcreteArray, xs)
        h = map(x -> vec(x isa Reactant.AbstractConcreteArray ? Array(x) : x), xs)
        return place_replicated(length(h) == 1 ? copy(h[1]) : vcat(h...), mesh)
    end
    return length(rows) == 1 ? vec(xs[1]) : vcat(map(vec, xs)...)
end

"""
    ReactantNitro.unflatten(flat::NTuple{G}, template, layout) -> tree

G group buffers back to a tree, shaped like `template`. **A `reshape` over a contiguous slice**, per
leaf: `reshape(flat[offset .+ (1:len)], size)`.

Under trace both the slice and the reshape are ordinary XLA operations that fold into the consuming
op, so the reconstruction is free rather than a copy. **These are not host views**, which is why
the export caveat about `copy()`ing view-shaped outputs does not apply to parameters: nothing here
produces a Julia `SubArray`.
"""
function unflatten(flat::NTuple{G, Any}, template, layout::FlatLayout{G}) where {G}
    return Functors.fmap_with_path(template) do kp, x
        x isa AbstractArray || return x
        r = layout.index[Tuple(kp)]
        row = layout.permutation[r]
        # `copy` is load-bearing, not defensive. Without it the reconstruction comes back as a
        # `Base.ReshapedArray` wrapping the flat buffer, so `typeof(ps)` CHANGES after the first
        # optimizer step and every subsequent step misses the compile cache and recompiles. The
        # export caveat about view-shaped outputs is often waved off for parameters on the grounds
        # that nothing here produces a Julia `SubArray`, which is true of `SubArray` and false of
        # the substance: a
        # `ReshapedArray` is just as much a view, and it is what a materialized program output
        # actually carries. Under trace the copy is an XLA op that folds away.
        return copy(reshape(flat[layout.row_group[r]][row.offset .+ (1:row.len)], row.size))
    end
end
unflatten(flat::Tuple, template, layout::FlatLayout) =
    unflatten(NTuple{length(flat), Any}(flat), template, layout)

# ── The automatic no-decay exclusion ────────────────────────────────────────────────

"""
    ReactantNitro.no_decay_masks(e, ps, layout; elt = Float32) -> NTuple{G}

The per-parameter decay exclusion, as one 0/1 flat buffer per group: **1 means decayed**, 0 means
excluded. **This is the one piece of per-parameter behavior not expressed through groups**, and the
reason it is not is that biases and norm affines occur inside every group, so routing their
exclusion through [`param_group`](@ref) would force a group split that also splits the
learning-rate ratio.

The rule is [`no_decay`](@ref)`(e, keypath, leaf)`, which defaults to
[`default_no_decay`](@ref), `ndims(leaf) == 1`. That default is exactly what this function hardcoded
before the hook existed, so no run's numerics move by adding it.

Built **once, at construction**, and stored on the [`Nitro`](@ref). It used to be rebuilt at every
`train!` entry as well, which was harmless only because it was a pure function of frozen state; with
a user hook feeding it, recomputing would let a revised rule take effect on an existing handle and
contradict the rule that a handle's programs and numerics are fixed at construction.
"""
function no_decay_masks(e, ps, layout::FlatLayout{G}; elt::Type = Float32) where {G}
    keep = Dict{Tuple, Bool}()
    Functors.fmap_with_path(ps) do kp, x
        x isa AbstractArray && (keep[Tuple(kp)] = !no_decay(e, Tuple(kp), x))
        return x
    end
    return ntuple(Val(G)) do gi
        buf = Vector{elt}(undef, layout.lengths[gi])
        for r in layout.group_rows[gi]
            row = layout.permutation[r]
            buf[row.offset .+ (1:row.len)] .= keep[row.keypath] ? one(elt) : zero(elt)
        end
        buf
    end
end

# ── Three levels ────────────────────────────────────────────────────────────────────

"""
    ReactantNitro.resolve_hp(e, layout, gi; eta_t = nothing, anchors = nothing, masks = nothing)

The per-group hyperparameter carrier, resolved per group per step. Keys `eta`, `lambda`, `anchor`,
and `no_decay_mask` are always present, and the schedule layer adds one key per scheduled rule
field.

`eta` and `lambda` follow the effective-learning-rate definition, with [`effective_lr`](@ref) doing
the arithmetic:

    η_g(t) = eta_sched(t) * (learning_rate(e, Val(g)) / learning_rate(e))
    λ_g(t) = η_g(t) * lambda(e, Val(g))

`eta_t` is the `eta` schedule's value at this step, or `nothing` at setup and whenever no `eta`
schedule is configured, in which case `learning_rate(e)` is used.

**Both scalars are converted to the flat buffer's element type before device conversion**, and that
is load-bearing rather than tidy. Measured: `Optimisers.RAdam()` defaults `eta` to `Float64`, so a
`Float32` parameter buffer comes back out of `apply!` as `Float64`, the thunk guard rejects the
second call, and the run dies on step 2 with a message about argument types rather than about
precision. The element type is carried through the flat buffer, the optimizer state, the masks,
and the scheduled-scalar carrier; this is that carrier.

**`memo` is what keeps the constant case from re-uploading**, and the donation question it was
blocked on is now measured rather than open. Without a memo, `to_device` runs unconditionally, so
`eta` and a non-zero `lambda` are uploaded fresh on every optimizer step, per group, whether or not
anything is scheduled: with no `eta` schedule, `effective_lr` returns the constant base rate and it
is still converted from a host `Float32` each time. The parameter-sized entries were always right,
since `anchor` and `no_decay_mask` are passed through from values built at setup and never
re-uploaded, which is what the non-schedulable opt-out exists to protect. See
[`memo_to_device`](@ref) for the reuse rule.

**Measured on CUDA.** The blocker was whether the optimizer program DONATES the buffers
these scalars live in, because a cached scalar reused after donation is the freed-buffer hazard
wearing a different hat and presents as a WRONG LEARNING RATE rather than as a crash. Compiling an
optimizer-shaped program over an `Adam` leaf gave `donated_args_mask == Bool[0, 1, 1, 0, 0, 0, 0]`:
the two donated arguments are the moments, and `eta` is not donated. That follows from the program's
own shape rather than from luck. `apply_group` returns `Leaf(leaf.rule, ...)`, so the rule, and the
`eta` inside it, is returned as a pass-through block argument, which makes it a PRESERVED argument
under Reactant's `:auto` policy and preserved arguments are exactly the ones donation skips.
Reuse then held: 500 steps against one cached scalar, readback correct at every checkpoint, and the
parameters bit-identical to the fresh-upload path.

**The CPU backend cannot see any of this**, which is the CPU/device boundary showing up in the one
place it would have been easiest to trust. The same probe on CPU returns an all-false mask for
every argument
including pure consumption, so donation is simply not exercised there, and a green CPU suite is
evidence about the memo's bookkeeping only, never about whether reuse is safe.

**What the memo actually buys, with its scope attached**, because the unqualified number is
misleading. An isolated scalar upload measured 40.8 us on an empty queue and 61.3 us at queue depth
16. `resolve_hp` issues one upload per group per optimizer step at the default zero `lambda`, so a
275-step epoch over two groups is roughly 550 uploads, on the order of 20 ms per epoch of transfer.
An optimizer-step microbenchmark with no forward or backward pass in it showed 20.3% (100k
parameters) to 36.6% (10M) of step wall time recovered, but that denominator is the optimizer step
alone; against a real model measured at roughly 40 s of stepping per epoch the saving is **well under
one percent of training wall time**. Take this for being exact and nearly free, not for being fast.
"""
function resolve_hp(
        e, layout::FlatLayout, gi::Int; eta_t = nothing, anchors = nothing,
        masks = nothing, elt::Type = Float32, mesh = nothing, memo = nothing
    )
    g = layout.groups[gi]
    eta = elt(effective_lr(e, g, eta_t))
    lam = elt(eta * lambda(e, Val(g)))
    return (;
        eta = memo_to_device(memo, (gi, :eta), eta; mesh),
        lambda = iszero(lam) ? nothing : memo_to_device(memo, (gi, :lambda), lam; mesh),
        anchor = anchors === nothing ? nothing : anchors[gi],
        no_decay_mask = masks === nothing ? true : masks[gi],
    )
end

# ── The scheduled-scalar memo ───────────────────────────────────────────────────────

"""
    ReactantNitro.ScalarMemo() -> ScalarMemo

One slot per `(group, key)`, holding the host value last uploaded and the device scalar it produced,
so [`resolve_hp`](@ref) can skip the transfer while the host value is unchanged.

**One slot, not a growing table**, and that is the whole of the eviction policy. A scheduled `eta`
produces a different value on most steps, so a keyed cache would grow without bound and never hit;
replacing the slot on a miss makes the scheduled case cost exactly what it costs today and the
constant case cost one upload per run.

**Scoped to a `train!` call**, which is what keeps `mesh` out of the key. `nitro.mesh` is fixed for
the run, so every scalar a given memo holds was placed on the same mesh by construction. A
module-level memo could not say that, and a scalar placed with a device count of 1 against moments
replicated across a mesh is the exact failure `rebuild_rules` records from the first 4-device run.
"""
struct ScalarMemo
    slots::Dict{Tuple{Int, Symbol}, Tuple{Any, Any}}
end
ScalarMemo() = ScalarMemo(Dict{Tuple{Int, Symbol}, Tuple{Any, Any}}())

"""
    ReactantNitro.memo_to_device(memo, key, x; mesh = nothing) -> device scalar

[`to_device`](@ref) with a one-slot memo in front of it. `memo === nothing` is the unmemoized path
and is exactly the old behaviour, which is what `setup` and the unit tests take.

Two conditions gate reuse, and both are refusals rather than permissions:

  * **the host value must be `===` the one that produced the cached scalar.** `===` on a `Float32`
    is bitwise, so it never conflates `0.0` with `-0.0`, never claims a `Float32` and a `Float64`
    spelling of one number are the same, and needs no separate element-type key. `resolve_hp`
    converts to `elt` BEFORE calling here, so the value compared is the one that would be uploaded.
  * **the cached scalar must not be marked donated.** Reactant sets `donated` when it hands a
    buffer's storage to a program, and reusing one after that is the freed-buffer hazard, which
    presents as a wrong learning rate rather than a crash. A value with no `donated` field to read
    is refused too, so an IFRT or future concrete type this was never measured against falls back to
    uploading rather than guessing.

**The guard is not independent of its subject and should not be read as if it were.** `donated` is
Reactant's own record of what it told XLA, not an outside check on it, so it cannot catch a buffer
XLA consumed without Reactant marking it. The per-step upload hid that class of bug by never reusing
anything; this does not. What stands behind it is the measurement in [`resolve_hp`](@ref)'s
docstring, 500 steps of reuse on CUDA with the parameters bit-identical to the fresh path, and that
measurement has to be re-run on a Reactant bump rather than assumed to carry forward.
"""
memo_to_device(::Nothing, key, x; mesh = nothing) = to_device(x; mesh)

function memo_to_device(memo::ScalarMemo, key, x; mesh = nothing)
    hit = get(memo.slots, key, nothing)
    if hit !== nothing
        host, dev = hit
        host === x && !reuse_refused(dev) && return dev
    end
    dev = to_device(x; mesh)
    memo.slots[key] = (x, dev)
    return dev
end

# Refuse on a set `donated` flag, and refuse just as hard on a type that has no flag to read: the
# safe direction is one wasted upload, and the unsafe one is silent.
reuse_refused(x) = !hasfield(typeof(x), :donated) || getfield(x, :donated)

"""
    ReactantNitro.build_chain(e, group::Symbol, hp) -> Optimisers.AbstractRule

The three configuration levels resolved into one chain per group, plus the setup checks below.

**Levels 0 and 1: the framework constructs.** [`optimizer`](@ref) returns a rule *type*, so the
framework builds it by splatting the resolved values into its constructor, keyed by field name.
Every configured hyperparameter is therefore applied by construction, and a rule the
framework has never heard of works with no framework change.

**Level 2: the user's factory constructs** and the framework only supplies `hp`. It performs one
**name** check, which catches the typo and the wrong-rule case, and deliberately does **not** check
that the factory read what it was given; see [`optimizer`](@ref).

Three setup errors, all naming the offenders:

  * **A base rule declaring a `lambda` field at Level 1** (`AdamW`), since that rule plus the
    framework-composed [`Decay`](@ref) tail would decay twice, silently, at whatever the two
    coefficients sum to, and `opt.lambda` would then resolve against two rules in one chain.
  * **Two rules in one chain declaring the same schedulable field name**, without which a single
    schedule key would silently write two rules. This is the intra-chain sibling of the schedule
    layer's cross-namespace ambiguity error.
  * **`Decay` anywhere but last in a chain**, since anything after it would feed the decay term
    through moment estimation, producing coupled L2, which the framework does not support.
"""
function build_chain(e, group::Symbol, hp)
    chain = optimizer(e, group, hp)
    check_allowlist(chain)
    check_decay_last(chain)
    check_no_duplicate_fields(chain)
    return chain
end

"""
    ReactantNitro.level1_chain(e, group, hp) -> Optimisers.AbstractRule

The default method behind `optimizer(e, group, hp)`: the Level 1 composition of
[`optimizer`](@ref)`(e, Val(group))` with the framework's own [`Decay`](@ref) tail. Level 1 is the
default method of Level 2, so there is one concept with a shorthand.

The `Decay` tail is omitted entirely when the group's `lambda` is zero, which is the default, so a
bare experiment's chain is the base rule alone and emits no decay ops.
"""
function level1_chain(e, group::Symbol, hp)
    R = optimizer(e, Val(group))
    R isa Type || error(
        """
        ReactantNitro: `optimizer(e, ::Val{$(repr(group))})` must return a rule TYPE at Level 1, and
        returned a `$(typeof(R))`. The framework constructs the rule itself, because rules carry
        tracked device scalars rebuilt every step and a pre-built chain would freeze them as
        constants. To supply a constructed chain, define the Level 2 method
        `optimizer(e, group::Symbol, hp)` instead."""
    )
    :lambda in fieldnames(R) && error(
        """
        ReactantNitro: `$(nameof(R))` declares a `lambda` field of its own and applies decoupled
        decay itself, so composing it with the framework's `Decay` tail would decay TWICE, silently,
        at whatever the two coefficients sum to, and `opt.lambda` would resolve against two rules in
        the same chain.
        Use `optimizer(e) = Optimisers.$(nameof(R) === :AdamW ? "Adam" : "<a rule with no lambda>")`
        and set `lambda(e, ::Val{$(repr(group))})`, which gives identical semantics through the
        framework's own tail. `$(nameof(R))` remains available at Level 2, where you own the chain
        and the framework composes no tail."""
    )
    base = construct_rule(R, hp)
    hp.lambda === nothing && return base
    return Optimisers.OptimiserChain(base, Decay(hp.lambda, hp.anchor, hp.no_decay_mask))
end

"""
    ReactantNitro.construct_rule(R::Type, hp) -> rule

Build rule type `R` by splatting `hp`'s values in **by field name**, falling back to
`R()`'s own defaults for fields `hp` does not carry. Matching names make this one line with no
per-rule knowledge, which is what lets a rule the framework has never heard of work unchanged.
"""
function construct_rule(R::Type, hp)
    proto = R()
    vals = map(fieldnames(typeof(proto))) do f
        hasproperty(hp, f) ? getproperty(hp, f) : getfield(proto, f)
    end
    return Base.typename(typeof(proto)).wrapper(vals...)
end

_rules_of(c::Optimisers.OptimiserChain) = c.opts
_rules_of(r) = (r,)

"""
    ReactantNitro.check_decay_last(chain) -> nothing

`Decay` after the base rule is decoupled, matching AdamW. `Decay` **before** the base rule would
feed the decay term through moment estimation, producing coupled L2, which the framework does not
support, so `Decay` anywhere but last is an error.
"""
function check_decay_last(chain)
    rules = _rules_of(chain)
    for (i, r) in enumerate(rules)
        if r isa Decay && i != length(rules)
            error(
                """
                ReactantNitro: `Decay` is at position $i of $(length(rules)) in this chain and must
                be LAST. Decay after the base rule is DECOUPLED, matching AdamW; before it, the
                decay term is fed through moment estimation, which is coupled L2 and is out of
                scope. Chain order was:
                    $(join(string.(nameof.(typeof.(rules))), " -> "))"""
            )
        end
    end
    return nothing
end

"""
    ReactantNitro.check_no_duplicate_fields(chain) -> nothing

Two rules in one chain declaring the same schedulable field name is a setup error naming
both. Without it a single schedule key would silently write two rules. Only **schedulable** fields
collide, since a non-schedulable field is never written by a key.
"""
function check_no_duplicate_fields(chain)
    seen = Dict{Symbol, Symbol}()
    for r in _rules_of(chain)
        T = typeof(r)
        for f in fieldnames(T)
            f in nonschedulable(T) && continue
            if haskey(seen, f)
                error(
                    """
                    ReactantNitro: rules `$(seen[f])` and `$(nameof(T))` in the same chain both
                    declare a schedulable field `$f`, so a single `opt.$f` schedule key would
                    silently write both. Split them into different parameter
                    groups, or drop one of the two rules."""
                )
            end
            seen[f] = nameof(T)
        end
    end
    return nothing
end

# ── Building the optimizer state ────────────────────────────────────────────────────

"""
    ReactantNitro.build_opt_state(e, flat::NTuple{G}, layout; kwargs...) -> NTuple{G}

Setup's optimizer-state step, **training only**. Build the per-group chain, `Optimisers.setup` it
against that group's flat buffer, normalize through [`to_device_leaf`](@ref), and assert the
no-host-`Number` property on the result. Skipped by a `Nitro` built for evaluation, which would
otherwise allocate the moments
for nothing, which for Adam-family rules is twice the parameter memory on device.

**Call `Optimisers.apply!` per group.** `apply!` is array-level rather than tree-level, so handing it
a flat group slice skips the tree walk entirely, and per-group calls also solve the scalar-`eta`
limitation that blocks a single flat leaf.

**The collection is an `NTuple{G}`, never a `Vector`.** With two groups carrying different rules, a
`Vector` infers as `Vector{Optimisers.Leaf}`, which is abstractly typed, and the thunk guard then
rejects the second call.

`opt_state` remains **opaque to the user and to the checkpointer**: store and return whatever the
optimizer produced. It is not opaque to the framework, which owns its construction and normalization.
**Bias correction lives here**, since `βt` and `t` are state fields of the stock rules, so resume
restores it by restoring the state; the separately stored `step` exists for schedules.
"""
function build_opt_state(e, flat::NTuple{G, Any}, layout::FlatLayout{G}; kwargs...) where {G}
    state = ntuple(Val(G)) do gi
        hp = resolve_hp(e, layout, gi; kwargs...)
        chain = build_chain(e, layout.groups[gi], hp)
        to_device_leaf(Optimisers.setup(chain, flat[gi]); mesh = get(kwargs, :mesh, nothing))
    end
    for gi in 1:G
        assert_device_state(
            state[gi].state;
            context = "optimizer state for group $(repr(layout.groups[gi]))"
        )
    end
    return state
end

"""
    ReactantNitro.apply_group(leaf, x, g) -> (leaf_new, x_new)

One group's optimizer update, the array-level call the flat layout makes possible. The subtraction
mirrors `Optimisers.subtract!` in preserving `eltype(x)`; a bare `x .- dx` promotes the parameter
buffer whenever a rule hyperparameter is wider than it, which the thunk guard then rejects on the
second call.
"""
function apply_group(leaf::Optimisers.Leaf, x, g)
    st_new, dx = Optimisers.apply!(leaf.rule, leaf.state, x, g)
    return Optimisers.Leaf(leaf.rule, st_new, leaf.frozen), Base.eltype(x).(x .- dx)
end

# ── Manual mode: the optimizer-step helper ─────────────────────────────────────────

"""
    step_optimizer(opt_state, ps, grads) -> (states, ps_new)

One tree-level optimizer step for use inside a [`train_step`](@ref) closure: `Optimisers.update`
under trace, returning the new parameter tree and the new optimizer **states**.

**The rules are stripped from the return, and the driver re-attaches the rules it handed in.**
A measured rule of sharding: a replicated scalar, which is what a rule's device hyperparameters
are, cannot leave a compiled program as an output on a mesh, though arrays can, and that is the
same rule that makes the automatic `opt_program` return states. `opt_state` must be
device-normalized, which
[`setup_optimizers`](@ref) does; call this once per network or group, passing that subtree:

```julia
st_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
```

The subtraction mirrors `apply_group` in preserving `eltype(x)`, so the parameter tree is a type
fixed point across the step, which is what keeps the closure's program on one cache entry.
"""
function step_optimizer(opt_state, ps, grads)
    opt_state_new, ps_new = Optimisers.update(opt_state, ps, grads)
    return strip_rules(opt_state_new), ps_new
end

"""
    ReactantNitro.strip_rules(opt_state) -> states

Replace every `Optimisers.Leaf` reachable from a user's optimizer state tree with its
STATE, preserving structure. The mirror of [`merge_rules`](@ref), and the half of the
"rules in, states out" contract that runs inside the traced closure: a `Leaf` cannot leave a
compiled program on a mesh because its rule carries replicated scalars, so the closure returns
states and the driver re-attaches the rules.
"""
function strip_rules(x)
    if x isa Optimisers.Leaf
        return x.state
    elseif x isa Union{Tuple, NamedTuple}
        return map(strip_rules, x)
    elseif x isa AbstractArray || x isa Number || x isa AbstractString || x isa Symbol
        return x
    elseif isstructtype(typeof(x))
        T = typeof(x)
        fs = fieldnames(T)
        isempty(fs) && return x
        vals = ntuple(i -> strip_rules(getfield(x, fs[i])), length(fs))
        all(i -> vals[i] === getfield(x, fs[i]), 1:length(fs)) && return x
        return T.name.wrapper(vals...)
    end
    return x
end

"""
    ReactantNitro.merge_rules(combined, states) -> opt_state

The mirror of [`strip_rules`](@ref), run by the driver between closure calls: re-attach the rules
of the `opt_state` it handed in to the states the closure returned, rebuilding each
`Optimisers.Leaf` as `Leaf(input.rule, returned_state, input.frozen)`. The two trees must
have the same structure; a mismatch is a framework bug rather than a user error, and surfaces as a
method error here, at the merge, rather than inside a trace.
"""
function merge_rules(combined, states)
    if combined isa Optimisers.Leaf
        return Optimisers.Leaf(combined.rule, states, combined.frozen)
    elseif combined isa Union{Tuple, NamedTuple}
        return map((c, s) -> merge_rules(c, s), combined, states)
    elseif combined isa AbstractArray || combined isa Number ||
            combined isa AbstractString || combined isa Symbol
        return states
    elseif isstructtype(typeof(combined))
        T = typeof(combined)
        fs = fieldnames(T)
        isempty(fs) && return combined
        vals = ntuple(
            i -> merge_rules(getfield(combined, fs[i]), getfield(states, fs[i])),
            length(fs)
        )
        all(i -> vals[i] === getfield(combined, fs[i]), 1:length(fs)) && return combined
        return T.name.wrapper(vals...)
    end
    return states
end

# ── Clipping is framework-owned ─────────────────────────────────────────────────────

"""
    ReactantNitro.global_grad_norm(g_accum::NTuple{G}) -> scalar

The global L2 norm over every group at once. Global norm is the mode the flat representation
expresses exactly, and is impossible per-leaf over a parameter tree.
"""
global_grad_norm(g_accum::Tuple) = sqrt(sum(gi -> sum(abs2, gi), g_accum))

"""
    ReactantNitro.clip_by_global_norm(g_accum::NTuple{G}, threshold) -> NTuple{G}

The framework's own clip, applied at the top of the **optimizer** program on the fully accumulated
gradient, before any `apply!`. Never a chain member: a chain member clips **per group**, which is a
different algorithm producing different updates.

**The threshold is a trace-time host constant**, so `threshold <= 0` returns the accumulator
untouched and the compiled program is identical to one from a run that never configured clipping.
A device-resident threshold could not keep that: `0` would scale the gradient to zero norm, and on an
all-zero gradient `0/0` yields `NaN` silently. "Off" would have to be spelled `Inf`.

The scale is `threshold / max(norm, threshold)`, which is exactly `1` when the norm is under the
threshold and `threshold / norm` when over, and which cannot divide by zero because `threshold > 0`
on this branch. A gradient over the threshold therefore comes out at norm exactly the threshold,
which the accumulation tests assert.
"""
function clip_by_global_norm(g_accum::Tuple, threshold::Real)
    threshold > 0 || return g_accum          # HOST branch: emits no ops at all
    nrm = global_grad_norm(g_accum)
    scale = threshold / max(nrm, threshold)
    return map(gi -> gi .* Base.eltype(gi)(scale), g_accum)
end
