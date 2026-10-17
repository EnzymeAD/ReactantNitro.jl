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

Whether `x` is a `Number` that is not on the device. `Reactant.RNumber` is the common supertype of
`ConcretePJRTNumber` and `TracedRNumber`, so the predicate holds on both sides of a traced boundary,
as [`assert_device_state`](@ref) needs.
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

Error if an optimizer state tree holds a host `Number`, checked at setup and after a restore. A host
step counter bakes into the trace as a constant and freezes, so bias correction goes wrong with no
error; this check is what catches it. Rules are not checked, since a rule may keep
[`nonschedulable`](@ref) fields on the host ([`to_device_rule`](@ref)).
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
    ReactantNitro.assert_opt_state_device(opt_state, context = "optimizer state from `setup_optimizers`") -> nothing

[`assert_device_state`](@ref) over a user-supplied optimizer state tree (manual mode): the state of
every `Optimisers.Leaf` reachable from it is checked, never its rule.
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
    ReactantNitro.to_device_leaf(l::Optimisers.Leaf; mesh = nothing) -> Optimisers.Leaf

Move a leaf to the device: the rule through [`to_device_rule`](@ref) and the state with
`track_numbers = Number`, so integer step counters move too. The two are converted separately
because `Leaf.frozen` is declared `::Bool`, which a single conversion over the leaf cannot
reconstruct. A `Tuple` or `NamedTuple` of leaves is converted elementwise. Resume runs the same
conversion over the restored `opt_state`, which a checkpoint stores on the host.
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
    ReactantNitro.to_device_rule(r; mesh = nothing) -> rule

Move a rule's schedulable fields to the device and leave the rest on the host. The choice is per
field, since types do not separate them: `ClipNorm`'s schedulable `omega` and structural `p` are
both `Float64`, and moving `p` breaks `_norm`'s `::Real` dispatch. The moved set is `fieldnames(T)`
minus [`nonschedulable`](@ref)`(T)`, so one declaration controls both scheduling and placement.
Parameter-sized fields such as [`Decay`](@ref)'s `anchor` are in the opt-out and pass through
unchanged, already on the device. Chains are converted rule by rule. (The name avoids
`promote_rule`, which would shadow `Base.promote_rule`.)
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
tests assert against that. Every other rule has a division or sqrt in between. The tuple also
admits this package's [`Decay`](@ref). `AccumGrad` is absent because it traces and gives wrong
results (Train.jl). Layer-adaptive rules (LARS, LAMB) are absent because they would compute one
trust ratio per group rather than per layer.
"""
const ADMITTED_RULES = (
    Optimisers.Descent, Optimisers.Momentum, Optimisers.Nesterov,
    Optimisers.Adam, Optimisers.AdamW, Optimisers.RAdam,
    Optimisers.WeightDecay, Optimisers.ClipGrad, Optimisers.ClipNorm,
    Decay,
)

"""
    ReactantNitro.check_allowlist(rule) -> nothing

Error at setup if `rule`, or any member of an `OptimiserChain`, is not in [`ADMITTED_RULES`](@ref).
`ClipNorm` with the default `throw = true` is rejected with an error naming the flag: it reaches
`if o.throw && !isfinite(nrm)`, a host branch on a traced value, and would otherwise fail with a
`TypeError` that names neither the rule nor the flag.
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

One row of the flat permutation: one row per parameter leaf, not one entry per element. `offset` is
0-based within the leaf's group buffer. The stored keypath lets the resume check say which leaf
moved.
"""
const LeafRow = @NamedTuple{keypath::Tuple, group::Symbol, offset::Int, len::Int, size::Dims}

"""
    ReactantNitro.FlatLayout{G}

The mapping between the parameter tree and the per-group flat buffers, built once at setup and held
as a host-side constant for the run. `G`, the group count, is a type parameter because it is a
trace-time constant: per-group arithmetic unrolls with no traced control flow, and the per-group
collections are `NTuple{G}`, since a `Vector` is abstractly typed and the thunk guard would reject
it on the second call.

`permutation` is the table a checkpoint stores; the other fields are lookups derived from it.
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

A zero-field handle carrying a [`FlatLayout`](@ref) at the type level, as [`Router`](@ref) carries
its key set. The layout cannot enter a traced program as a value: Reactant promotes the `Int`s
inside a struct argument, and a promoted offset cannot index. At the type level it is also part of
the compile cache key, which a `FlatLayout{G}` value would not be, since two layouts with the same
`G` share a type.
"""
struct LayoutRef{ID} end

const LAYOUT_REGISTRY = Dict{UInt, Any}()
const LAYOUT_REGISTRY_LOCK = ReentrantLock()

"""
    ReactantNitro.layout_ref(layout) -> LayoutRef

Register `layout` under a hash of its `permutation` table and return its type-level handle. Two
identical layouts share a handle and therefore compiled programs, so a second `Nitro` over the same
experiment reuses the first one's.
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

The keypaths of `ps`'s array leaves in traversal order: `Functors.fmap_with_path`, depth-first and
deterministic for a fixed model, so the layout is the same across sessions and processes (checked
on Functors 0.5.2 with a nested Lux `Chain`).
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

Build the [`FlatLayout`](@ref) for every handle, including an evaluation `Nitro`, since checkpoints
are validated against it. The leaves, in traversal order, are stably sorted by group index, so
within a group they keep traversal order. `:default` is group 1 and the rest follow
first-appearance order. Errors if `ps` has no arrays.
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
    ReactantNitro.flatten(tree, layout, mesh = nothing) -> NTuple{G}

Concatenate a tree's leaves into `G` group buffers, a real copy. The permutation has one row per
leaf, so the operand order is known on the host at trace time and grouping emits no control flow.
Errors on a leaf the layout does not contain.
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

Rebuild a tree shaped like `template` from `G` group buffers, one `reshape` over a contiguous slice
per leaf. Under trace the slice and reshape fold into the consuming op, so this costs nothing.
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

The per-leaf decay exclusion from [`no_decay`](@ref)`(e, keypath, leaf)`, as one 0/1 flat buffer
per group (1 decayed, 0 excluded). It is per leaf because biases occur inside every group. Built
once at construction and stored on the [`Nitro`](@ref), so a redefined `no_decay` takes effect on
the next handle.
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
    ReactantNitro.hp_host(e, group, sv = (;); elt = Float32) -> NamedTuple

The host values one group's rule is built from at one step, before device conversion. `sv` is the
group's scheduled rule fields at this step ([`group_schedule_values`](@ref)); `eta` and `lambda`
fall back to the group's accessors when `sv` lacks them. `eta` is the group's effective rate and
`lambda` is multiplied by it ([`effective_lr`](@ref)); every other field in `sv` is passed through
as is. All are converted to `elt`.
"""
function hp_host(e, group::Symbol, sv::NamedTuple = (;); elt::Type = Float32)
    eta = elt(effective_lr(e, group, get(sv, :eta, nothing)))
    lam = elt(eta * (haskey(sv, :lambda) ? sv.lambda : lambda(e, Val(group))))
    rest = map(elt, Base.structdiff(sv, NamedTuple{(:eta, :lambda)}))
    return (; eta, lambda = lam, rest...)
end

# The `hp` setup builds a group's chain from: `sv` is the group's [`setup_schedule_values`](@ref).
setup_hp(e, layout, gi, sv::NamedTuple; kwargs...) = resolve_hp(
    e, layout, gi; host = hp_host(e, layout.groups[gi], sv), keep_decay = haskey(sv, :lambda),
    kwargs...
)

"""
    ReactantNitro.resolve_hp(e, layout, gi; eta_t = nothing, host = nothing, keep_decay = false,
                             anchors = nothing, masks = nothing, elt = Float32, mesh = nothing,
                             memo = nothing, device = true)

Build group `gi`'s hyperparameter carrier `hp` for one step: `eta`, `lambda`, `anchor` and
`no_decay_mask`, plus one key per scheduled rule field. `host` is the group's [`hp_host`](@ref)
values; without it they are computed from `eta_t`, the `eta` schedule's value or `nothing`. A zero
`lambda` becomes `nothing`, which drops the `Decay` tail, unless `keep_decay` is set; a scheduled
`lambda` sets it, so the chain does not change when the schedule reaches zero. `device = false`
keeps host values, for the setup-time probe in [`schedule_bindings`](@ref).

Every scalar is converted to `elt`, the flat buffer's element type, before it moves to the device:
`Optimisers.RAdam()` defaults `eta` to `Float64`, and a `Float32` buffer would come back `Float64`
from `apply!` and fail the thunk guard on step 2.

`memo` is a [`ScalarMemo`](@ref) that skips the upload while the host value is unchanged, so a
constant `eta` is uploaded once per run. Reuse is safe because the optimizer program does not
donate the rule's scalars: `apply_group` returns the rule as a pass-through argument. This was
checked on CUDA (500 reused steps bit-identical to fresh uploads); the CPU backend never donates,
so the CPU suite cannot check it.
"""
function resolve_hp(
        e, layout::FlatLayout, gi::Int; eta_t = nothing, host = nothing, keep_decay = false,
        anchors = nothing, masks = nothing, elt::Type = Float32, mesh = nothing, memo = nothing,
        device::Bool = true
    )
    g = layout.groups[gi]
    h = host === nothing ? hp_host(e, g, eta_t === nothing ? (;) : (; eta = eta_t); elt) : host
    rest = Base.structdiff(h, NamedTuple{(:eta, :lambda)})
    # `device = false` keeps host values, for the setup-time binding probe.
    place(k, x) = device ? memo_to_device(memo, (gi, k), x; mesh) : x
    eta = place(:eta, h.eta)
    lambda = iszero(h.lambda) && !keep_decay ? nothing : place(:lambda, h.lambda)
    scheduled = NamedTuple{keys(rest)}(map(k -> place(k, getproperty(rest, k)), keys(rest)))
    return (;
        eta, lambda, scheduled...,
        anchor = anchors === nothing ? nothing : anchors[gi],
        no_decay_mask = masks === nothing ? true : masks[gi],
    )
end

# ── The scheduled-scalar memo ───────────────────────────────────────────────────────

"""
    ReactantNitro.ScalarMemo() -> ScalarMemo

A cache of uploaded scalars with one slot per `(group, key)`, holding the host value last uploaded
and the device scalar it produced. A slot keeps only the latest value, since a scheduled `eta`
differs on most steps and a cache keyed by value would grow without hits. It lives for one
`train!` call, so `mesh` is fixed and not part of the key.
"""
struct ScalarMemo
    slots::Dict{Tuple{Int, Symbol}, Tuple{Any, Any}}
end
ScalarMemo() = ScalarMemo(Dict{Tuple{Int, Symbol}, Tuple{Any, Any}}())

"""
    ReactantNitro.memo_to_device(memo, key, x; mesh = nothing) -> device scalar

[`to_device`](@ref) through the memo slot at `key`; `memo === nothing` always uploads. The cached
scalar is reused only if the host value is `===` the one that produced it (bitwise, so `-0.0` and a
`Float64` value never match a `Float32` `0.0`) and the cached scalar is not marked `donated`; a type
with no `donated` field is never reused. The `donated` flag is Reactant's own record, so the CUDA
check described in [`resolve_hp`](@ref) should be repeated after a Reactant upgrade.
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

Build group `group`'s chain through [`optimizer`](@ref)`(e, group, hp)` and check it. At Levels 0
and 1 the framework constructs the rule from its type by field name ([`level1_chain`](@ref)); at
Level 2 the user's factory constructs it, and setup records where each `hp` value lands
([`schedule_bindings`](@ref)), so two rules may share a field name. Errors on a rule outside
[`ADMITTED_RULES`](@ref), on a Level 1 rule declaring its own `lambda` (it would decay twice), and
on `Decay` anywhere but last (it would be coupled L2).
"""
function build_chain(e, group::Symbol, hp)
    chain = optimizer(e, group, hp)
    check_allowlist(chain)
    check_decay_last(chain)
    return chain
end

"""
    ReactantNitro.level1_chain(e, group, hp) -> Optimisers.AbstractRule

The default method behind `optimizer(e, group, hp)`: the rule type [`optimizer`](@ref)`(e, Val(group))`,
constructed from `hp` by [`construct_rule`](@ref), followed by a
[`Decay`](@ref)`(hp.lambda, hp.anchor, hp.no_decay_mask)` tail. The tail is omitted when
`hp.lambda` is `nothing`, that is, when the group's `lambda` is zero and not scheduled.
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

Build rule type `R` from `hp` by field name, using `R()`'s defaults for fields `hp` does not carry.
Matching by name needs no per-rule code.
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

Error if [`Decay`](@ref) is anywhere but last in `chain`. After the base rule, decay is decoupled,
matching AdamW. Before it, the decay term would pass through moment estimation, which is coupled L2
and is not supported.
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

# ── Building the optimizer state ────────────────────────────────────────────────────

"""
    ReactantNitro.build_opt_state(e, flat::NTuple{G}, layout; svs, kwargs...) -> NTuple{G}

Build the optimizer state at setup, for training only: for each group, build its chain from
`setup_hp` with `svs[gi]` ([`setup_schedule_values`](@ref)), call `Optimisers.setup` on the group's
flat buffer, move it to the device with [`to_device_leaf`](@ref), and assert no host `Number`
remains. `apply!` works on arrays, so a flat buffer needs no tree walk. The result is an
`NTuple{G}`, since a `Vector` would infer abstractly and fail the thunk guard. Bias correction lives
in the state, so resume restores it with the state.
"""
function build_opt_state(
        e, flat::NTuple{G, Any}, layout::FlatLayout{G}; svs = ntuple(_ -> (;), Val(G)), kwargs...
    ) where {G}
    state = ntuple(Val(G)) do gi
        hp = setup_hp(e, layout, gi, svs[gi]; kwargs...)
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

Apply one group's optimizer update to its flat buffer `x` with gradient `g`. Like
`Optimisers.subtract!`, the subtraction keeps `eltype(x)`; a bare `x .- dx` would widen the buffer
whenever a rule hyperparameter is wider than it, and the thunk guard would reject it on the second
call.
"""
function apply_group(leaf::Optimisers.Leaf, x, g)
    st_new, dx = Optimisers.apply!(leaf.rule, leaf.state, x, g)
    return Optimisers.Leaf(leaf.rule, st_new, leaf.frozen), Base.eltype(x).(x .- dx)
end

# ── Manual mode: the optimizer-step helper ─────────────────────────────────────────

"""
    step_optimizer(opt_state, ps, grads) -> (states, ps_new)

Run one optimizer step on a parameter tree inside a [`train_step`](@ref) closure. It calls
`Optimisers.update` and returns the new optimizer states, without their rules, and the new
parameters. The rules are removed because a replicated scalar cannot leave a compiled program on a
mesh; the driver re-attaches them after the step. Call it once per network, passing that subtree:

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

Replace every `Optimisers.Leaf` in an optimizer state tree with its state, keeping the structure.
This is the traced half of the manual-mode contract, rules in and states out; [`merge_rules`](@ref)
reverses it.
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

Re-attach the rules of the `opt_state` passed to the closure to the states it returned, rebuilding
each `Leaf(input.rule, returned_state, input.frozen)`. A structure mismatch is a framework bug and
fails here, outside the trace.
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

The L2 norm of the gradient over all groups' flat buffers together.
"""
global_grad_norm(g_accum::Tuple) = sqrt(sum(gi -> sum(abs2, gi), g_accum))

"""
    ReactantNitro.clip_by_global_norm(g_accum::NTuple{G}, threshold) -> NTuple{G}

Scale the fully accumulated gradient to a global norm of at most `threshold`, at the top of the
optimizer program. The scale is `threshold / max(norm, threshold)`, so a gradient over the
threshold comes out with norm exactly `threshold`. The threshold is a trace-time host constant, so
`threshold <= 0` returns `g_accum` unchanged and emits no ops; a device-resident `0` would give
`0/0 = NaN` on an all-zero gradient.
"""
function clip_by_global_norm(g_accum::Tuple, threshold::Real)
    threshold > 0 || return g_accum          # HOST branch: emits no ops at all
    nrm = global_grad_norm(g_accum)
    scale = threshold / max(nrm, threshold)
    return map(gi -> gi .* Base.eltype(gi)(scale), g_accum)
end
