# Optimizer.jl
#
# The optimizer layer: device-resident optimizer state, the traced-rule allowlist, the flat
# per-group parameter layout, the three configuration levels, and framework-owned clipping.
#
# The governing fact: stock `Optimisers.jl` rules trace correctly and match the host path bitwise,
# but only if the optimizer state is normalized to device residency first, INCLUDING its integer
# step counters. Left host, a counter bakes as a trace-time constant and freezes, and the run
# silently trains with the wrong bias correction. The optimizer tests carry the control.

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

Every host `Number` reachable from `x`, as `path::Type` strings, so the assertion can say which leaf
is still on the host. `Optimisers.Leaf.frozen` is skipped: declared `::Bool`, it cannot be promoted.
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

The no-host-`Number` assertion over an optimizer STATE tree, at setup and after a restore, and the
only thing standing between a user and the silent frozen-counter failure. A rule legitimately keeps
host fields ([`nonschedulable`](@ref)) and is governed by [`to_device_rule`](@ref) instead.
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

The same assertion over a user-supplied optimizer state tree (manual mode): every `Optimisers.Leaf`
reachable from it has its STATE asserted, never its rule.
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

Normalize a leaf to device residency: the rule through [`to_device_rule`](@ref) and the state
through `to_rarray(...; track_numbers = Number)`, separately, because `Leaf.frozen` is declared
`::Bool` and a single `to_rarray` over the leaf cannot reconstruct it. `track_numbers = Number`
because the state must promote integers. The same normalization runs on the resume path over the
restored `opt_state`: write host, read host, normalize on the way in.
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

Promote exactly the schedulable fields of a rule and leave the structural ones host. Per field, not
by type: `ClipNorm`'s schedulable `omega` and structural `p` are both `Float64`, and promoting `p`
breaks `_norm`'s `::Real` dispatch. The set is `fieldnames(T)` minus [`nonschedulable`](@ref)`(T)`,
so one declaration drives scheduling and promotion. Parameter-sized fields such as [`Decay`](@ref)'s
`anchor` are in that opt-out and pass through, already on device. Named `to_device_rule` because
defining `promote_rule` unqualified would shadow `Base.promote_rule`.
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

The traced-optimizer allowlist. A rule is admitted once it traces with device-resident state and
hyperparameters, its state is a type fixed point across the traced boundary, and three steps match
the stock host path bitwise (64-element buffer, every rule at the buffer's element type):

| Rule | Traces | Fixed point | maxdiff vs host |
| --- | --- | --- | --- |
| `Descent` | yes | yes | 1 ulp, see below |
| `Momentum`, `Nesterov` | yes | yes | 0.0 |
| `Adam`, `AdamW`, `RAdam` | yes | yes | 0.0 |
| `WeightDecay`, `ClipGrad` | yes | yes | 0.0 |
| `ClipNorm(ω, p; throw = false)` | yes | yes | 0.0 |
| `OptimiserChain(RAdam, WeightDecay)` | yes | yes | 0.0 |
| `OptimiserChain(ClipNorm, Adam)` | yes | yes | 0.0 |

`Descent`'s update `x - eta * g` is a multiply feeding a subtract, which XLA contracts into a fused
multiply-add that rounds once; the compiled result equals host `fma.(-eta, g, x)` exactly, and the
tests assert against that. Every other rule has a division or sqrt in between. `AccumGrad` is
absent because it traces and is wrong (Train.jl). Layer-adaptive rules (LARS, LAMB) are rejected
because they would compute one trust ratio per group rather than per layer.
"""
const ADMITTED_RULES = (
    Optimisers.Descent, Optimisers.Momentum, Optimisers.Nesterov,
    Optimisers.Adam, Optimisers.AdamW, Optimisers.RAdam,
    Optimisers.WeightDecay, Optimisers.ClipGrad, Optimisers.ClipNorm,
    Decay,
)

"""
    ReactantNitro.check_allowlist(rule) -> nothing

The admission check, at setup. `ClipNorm` with the default `throw = true` is rejected naming the
flag: it reaches `if o.throw && !isfinite(nrm)`, a host branch on a traced value, and fails with a
`TypeError` that names nothing a user could act on.
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

One row of the flat permutation, a per-leaf table rather than a `Vector{Int}` over elements.
`offset` is 0-based within the leaf's group buffer. Storing keypaths is what lets the resume check
say which leaf moved.
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
    # Tuples rather than Vectors: `map` over a host `Vector` inside a traced function is intercepted
    # by Reactant and vectorized, promoting the row indices to traced numbers; `map` over a Tuple
    # stays host, so the operand order is known at trace time.
    group_rows::NTuple{G, Tuple{Vararg{Int}}}
    row_group::Vector{Int}                     # row number -> group number
end

n_groups(::FlatLayout{G}) where {G} = G

"""
    ReactantNitro.LayoutRef{ID}

A zero-field handle carrying a [`FlatLayout`](@ref) at the type level, as [`Router`](@ref) does its
key set. The layout cannot ride into a traced program as a value: Reactant promotes the `Int`s
inside a struct argument, and a promoted offset can index nothing. Carrying it at the type level
also puts it in the compile cache key, which a `FlatLayout{G}` value would not, since two layouts
with the same `G` share a type.
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

Setup's layout step, unconditional, including for an evaluation `Nitro`: host-side bookkeeping the
checkpoint path validates against. A stable sort of the traversal-order leaves by group index, so
within a group leaves keep traversal order; `:default` is group 1 and the rest follow
first-appearance order.
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

Tree to G group buffers, a concatenate (a real copy). Because the permutation is one row per leaf,
the operand order is host-known at trace time, so grouping emits no control flow.
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

# Concrete device arrays go through the host, which is the only way this works on a GPU: `vec` of
# a `ConcretePJRTArray` is a `ReshapedArray`, so `vcat` lands in Base's elementwise `typed_vcat`,
# which is legal on CPU and "Scalar indexing is disallowed" on a GPU (found by the first real GPU
# run, at `Nitro` construction). Affordable because this path is setup-only: inside the traced
# programs the leaves are `TracedRArray`s and take the other branch.
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

G group buffers back to a tree shaped like `template`, a `reshape` over a contiguous slice per leaf.
Under trace the slice and reshape fold into the consuming op, so this is free.
"""
function unflatten(flat::NTuple{G, Any}, template, layout::FlatLayout{G}) where {G}
    return Functors.fmap_with_path(template) do kp, x
        x isa AbstractArray || return x
        r = layout.index[Tuple(kp)]
        row = layout.permutation[r]
        # `copy` is load-bearing: without it the leaf is a `ReshapedArray` over the flat buffer, so
        # `typeof(ps)` changes after the first step and every step recompiles. Under trace the copy
        # folds away.
        return copy(reshape(flat[layout.row_group[r]][row.offset .+ (1:row.len)], row.size))
    end
end
unflatten(flat::Tuple, template, layout::FlatLayout) =
    unflatten(NTuple{length(flat), Any}(flat), template, layout)

# ── The automatic no-decay exclusion ────────────────────────────────────────────────

"""
    ReactantNitro.no_decay_masks(e, ps, layout; elt = Float32) -> NTuple{G}

The per-parameter decay exclusion, as one 0/1 flat buffer per group (1 decayed, 0 excluded), from
[`no_decay`](@ref)`(e, keypath, leaf)`. The one piece of per-parameter behavior not expressed
through groups, since biases occur inside every group. Built once at construction and stored on the
[`Nitro`](@ref), so a revised rule takes effect on the next handle.
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

The per-group hyperparameter carrier, resolved per group per step: `eta`, `lambda`, `anchor` and
`no_decay_mask`, plus one key per scheduled rule field. `eta` and `lambda` follow
[`effective_lr`](@ref); `eta_t` is the `eta` schedule's value or `nothing` when none is configured.

Both scalars are converted to the flat buffer's element type before device conversion:
`Optimisers.RAdam()` defaults `eta` to `Float64`, and a `Float32` buffer would come back `Float64`
from `apply!` and fail the thunk guard on step 2.

`memo` is a [`ScalarMemo`](@ref) that skips the upload while the host value is unchanged, so a
constant `eta` crosses once per run. Reuse is safe because the optimizer program does not donate
the rule's scalars (measured on CUDA: `apply_group` returns the rule as a pass-through argument, so
it is preserved; 500 steps of reuse were bit-identical to fresh uploads). The CPU backend never
exercises donation, so a green CPU suite says nothing about this. The saving is well under one
percent of training wall time; take it for being exact rather than fast.
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

One slot per `(group, key)`, holding the host value last uploaded and the device scalar it
produced. One slot, not a table, since a scheduled `eta` differs on most steps and a keyed cache
would grow without hitting. Scoped to a `train!` call, so `mesh` is an invariant rather than a key.
"""
struct ScalarMemo
    slots::Dict{Tuple{Int, Symbol}, Tuple{Any, Any}}
end
ScalarMemo() = ScalarMemo(Dict{Tuple{Int, Symbol}, Tuple{Any, Any}}())

"""
    ReactantNitro.memo_to_device(memo, key, x; mesh = nothing) -> device scalar

[`to_device`](@ref) with a one-slot memo in front; `memo === nothing` is the unmemoized path. Reuse
requires the host value to be `===` the one that produced the cached scalar (bitwise on a
`Float32`, so `-0.0` and a `Float64` spelling never conflate) and the cached scalar not to be marked
donated; a type with no `donated` field is refused too. `donated` is Reactant's own record, not an
outside check, so the CUDA measurement in [`resolve_hp`](@ref) is what stands behind this and has to
be re-run on a Reactant bump.
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

The three configuration levels resolved into one chain per group. At Levels 0 and 1 the framework
constructs the rule from its type by field name; at Level 2 the user's factory constructs and the
framework checks names only. Three setup errors: a base rule declaring its own `lambda` at Level 1
(it would decay twice), two rules in one chain declaring the same schedulable field (one key would
write both), and `Decay` anywhere but last (anything after it would produce coupled L2).
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

The default method behind `optimizer(e, group, hp)`: [`optimizer`](@ref)`(e, Val(group))` composed
with the [`Decay`](@ref) tail, which is omitted when the group's `lambda` is zero.
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

Setup's optimizer-state step, training only: build each group's chain, `Optimisers.setup` it
against the group's flat buffer, normalize through [`to_device_leaf`](@ref), and assert no host
`Number` survives. `apply!` is array-level, so a flat group slice skips the tree walk. The
collection is an `NTuple{G}`, never a `Vector`, which would infer abstractly and fail the thunk
guard. Bias correction lives in the state, so resume restores it with the state.
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
under trace, returning the new parameter tree and the new optimizer STATES. The rules are stripped
because a replicated scalar cannot leave a compiled program on a mesh; the driver re-attaches them.
Call it once per network or group, passing that subtree:

```julia
st_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
```
"""
function step_optimizer(opt_state, ps, grads)
    opt_state_new, ps_new = Optimisers.update(opt_state, ps, grads)
    return strip_rules(opt_state_new), ps_new
end

"""
    ReactantNitro.strip_rules(opt_state) -> states

Replace every `Optimisers.Leaf` in a user's optimizer state tree with its STATE, preserving
structure; the traced half of the "rules in, states out" contract. The mirror of
[`merge_rules`](@ref).
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

Re-attach the rules of the `opt_state` handed in to the states the closure returned, rebuilding
each `Leaf(input.rule, returned_state, input.frozen)`. A structure mismatch is a framework bug and
surfaces here rather than inside a trace.
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

The framework's clip, at the top of the optimizer program on the fully accumulated gradient. The
threshold is a trace-time host constant, so `threshold <= 0` returns the accumulator untouched and
emits nothing; a device-resident `0` would scale the gradient to `NaN`. The scale is
`threshold / max(norm, threshold)`, so a gradient over the threshold comes out at exactly the
threshold.
"""
function clip_by_global_norm(g_accum::Tuple, threshold::Real)
    threshold > 0 || return g_accum          # HOST branch: emits no ops at all
    nrm = global_grad_norm(g_accum)
    scale = threshold / max(nrm, threshold)
    return map(gi -> gi .* Base.eltype(gi)(scale), g_accum)
end
