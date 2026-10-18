# GradRules.jl
#
# Gradient rules decide which gradient a micro-batch adds to the accumulator. A rule is composed
# by the driver from the existing gradient program and small programs of its own: it never adds a
# traced variant of the gradient program, and the optimizer program never sees it.

"""
    GradRule

The supertype of what [`grad_rule`](@ref) returns.
"""
abstract type GradRule end

"""
    SingleGrad()

The default [`grad_rule`](@ref): one forward and backward pass per micro-batch.
"""
struct SingleGrad <: GradRule end

"""
    PerturbRule <: GradRule

Rules that take the gradient at perturbed parameters, `ps + ε`, where `ε` comes from the
unperturbed gradient. A subtype defines [`perturbation`](@ref); the driver runs the gradient
program twice per micro-batch, at `ps` and then at `ps + ε`, and accumulates the second gradient.
The reported loss, the layer state and the `train_metrics` outputs are the first pass's.
"""
abstract type PerturbRule <: GradRule end

"""
    SAM(rho; m = nothing)

Sharpness-aware minimization (Foret et al. 2021): the gradient taken at `ps + rho * g / ‖g‖`.

`rho` is a host number; the framework places it on the device, so changing it never recompiles.
Schedule it as `rho = ...`, as `grad = (; rho = ...)` when another namespace has a `rho`, or by
hand through `bind` and [`grad_rule`](@ref)`(e, hp)`. `ε` is computed per accumulation
micro-batch and excludes the leaves [`no_perturb`](@ref) excludes, none by default. The cost is two forward and
backward passes per micro-batch. `m`, a per-chunk sample count for independent m-sharpness, is
reserved and not supported yet.
"""
struct SAM{R, M} <: PerturbRule
    rho::R
    m::M
end

function SAM(rho; m = nothing)
    m === nothing || error(
        "ReactantNitro: `SAM(rho; m = $m)`: per-chunk m-sharpness is not supported yet. Omit \
         `m`; `ε` is then computed per accumulation micro-batch."
    )
    return SAM(rho, m)
end

nonschedulable(::Type{<:SAM}) = (:m,)

"""
    perturbation(rule::PerturbRule, g, p, mask) -> ε

The parameter perturbation for one chunk. `g` (the unperturbed gradient), `p` (the parameters),
`mask` (1 for a leaf that may be perturbed, 0 for one [`no_perturb`](@ref) excludes) and `ε` are
`NTuple{G}`s of flat per-group buffers. Norms must reduce over the parameter axes only, so a
per-chunk variant can map the method over chunks unchanged. Traced.
"""
function perturbation end

function perturbation(r::SAM, g, p, mask)
    gm = map(.*, g, mask)
    T = eltype(first(gm))
    s = r.rho / (global_grad_norm(gm) + eps(T))
    return map(x -> x .* s, gm)
end

"""
    ASAM(rho; eta = 0.01f0, m = nothing)

Adaptive SAM (Kwon et al. 2021), which perturbs each parameter in proportion to its own size.

The gradient is taken at `ps + ε` with `ε = rho * T² g / ‖T g‖` and `T = |w| + eta`, so the
sharpness it measures is invariant to rescaling the weights. Leaves [`adaptive_perturb`](@ref) declines (by
default the rank-1 leaves: biases, norm affines, scalar parameters) get `T = 1`, as under
[`SAM`](@ref). `rho` is relative, so typical values are about ten times SAM's, around 0.5 to 2
with SGD and smaller with Adam-family rules. `eta` is not schedulable; `m` is reserved, as for
`SAM`.
"""
struct ASAM{R, E, M} <: PerturbRule
    rho::R
    eta::E
    m::M
end

function ASAM(rho; eta = 0.01f0, m = nothing)
    m === nothing || error(
        "ReactantNitro: `ASAM(rho; m = $m)`: per-chunk m-sharpness is not supported yet. Omit \
         `m`; `ε` is then computed per accumulation micro-batch."
    )
    return ASAM(rho, eta, m)
end

# `eta` is structural, and an unqualified `eta` schedule key must keep meaning the learning rate.
nonschedulable(::Type{<:ASAM}) = (:eta, :m)

"""
    perturbation(rule::ASAM, g, p, mask, adaptive) -> ε

ASAM's perturbation. `adaptive` holds 1 where [`adaptive_perturb`](@ref) scales the leaf by
`|w| + eta` and 0 where it is left unscaled.
"""
function perturbation(r::ASAM, g, p, mask, adaptive)
    t = map((pi, ai) -> ai .* (abs.(pi) .+ r.eta) .+ (1 .- ai), p, adaptive)
    tg = map((ti, gi, mi) -> ti .* (gi .* mi), t, g, mask)
    T = eltype(first(tg))
    s = r.rho / (global_grad_norm(tg) + eps(T))
    return map((ti, x) -> ti .* x .* s, t, tg)
end

# The perturbation with what each rule reads: `adaptive` only for ASAM, so a `PerturbRule`
# written against the four-argument `perturbation` keeps working.
_perturbation(rule, g, p, mask, adaptive) = perturbation(rule, g, p, mask)
_perturbation(rule::ASAM, g, p, mask, adaptive) = perturbation(rule, g, p, mask, adaptive)

function check_grad_rule(r::GradRule)
    T = typeof(r)
    for f in fieldnames(T)
        v = getfield(r, f)
        (v isa Reactant.RNumber || v isa Reactant.AbstractConcreteArray) && error(
            "ReactantNitro: `$(nameof(T)).$f` holds a device value, such as a `Device` field. \
             Pass a host number and schedule it under `grad`: \
             `schedules = (; grad = (; $f = <factory>))`."
        )
    end
    return r
end
check_grad_rule(r) = error(
    "ReactantNitro: `grad_rule` returned a `$(typeof(r))`; it must return a `GradRule` such as \
     `SingleGrad()` or `SAM(e.rho)`."
)

# The `grad` schedule namespace: the rule's fields minus its `nonschedulable` ones.
schedulable_grad_fields(r) =
    Tuple(f for f in fieldnames(typeof(r)) if !(f in nonschedulable(typeof(r))))

# The `grad` schedules evaluated at `step`, uncoerced; `(;)` when none are configured.
grad_schedule_values(sched, step::Integer) =
    sched === nothing ? (;) : map(f -> f(step), sched.grad)

# Whether the experiment places values itself, through `grad_rule(e, hp)`.
grad_level2(e) = !is_framework_default(grad_rule, Tuple{typeof(e), Any})

# The grad rule's `hp`: its scheduled fields by name, plus `bind`.
function grad_hp(grad_raw::NamedTuple, bind_raw::NamedTuple; elt::Type = Float32)
    hp = map(elt, grad_raw)
    return isempty(bind_raw) ? hp : merge(hp, (; bind = map(elt, bind_raw)))
end

# Placeholders for setup, read from the unresolved table: every unqualified or `grad` key, and
# every `bind` key. Extra keys are harmless; a missing one would fail a `grad_rule(e, hp)`.
function grad_setup_hp(given::NamedTuple; elt::Type = Float32)
    ks, bs = Symbol[], Symbol[]
    for (ns, k, _) in _flatten_schedules(given, "`schedules`")
        ns === :bind && push!(bs, k)
        ns in (:auto, :grad) && !occursin('.', string(k)) && !(k in ks) && push!(ks, k)
    end
    z(n) = ntuple(_ -> zero(elt), n)
    return grad_hp(NamedTuple{Tuple(ks)}(z(length(ks))), NamedTuple{Tuple(bs)}(z(length(bs))))
end

"""
    ReactantNitro.build_grad_rule(e, hp) -> GradRule

The host rule for one step. A user `grad_rule(e, hp)` places `hp`'s values itself; otherwise
`grad_rule(e)`'s fields are filled from `hp` by name, as Level 1 fills an optimizer rule.
"""
function build_grad_rule(e, hp)
    r = check_grad_rule(grad_rule(e, hp))
    grad_level2(e) && return r
    T = typeof(r)
    filled(f) = haskey(hp, f) && !(f in nonschedulable(T)) ?
        convert(typeof(getfield(r, f)), getproperty(hp, f)) : getfield(r, f)
    return T.name.wrapper(map(filled, fieldnames(T))...)
end

# The host rule's schedulable fields moved to the device, through the memo.
function place_grad_rule(r, memo; mesh = nothing)
    T = typeof(r)
    ns = nonschedulable(T)
    place(f) = f in ns ? getfield(r, f) : memo_to_device(memo, (0, f), getfield(r, f); mesh)
    return T.name.wrapper(map(place, fieldnames(T))...)
end

"""
    ReactantNitro.probe_grad_rule(e, hp0) -> Dict{Symbol, Vector{Symbol}}

Which rule fields each key of `hp0` reaches (`bind` keys as `Symbol("bind.k")`): the rule built
from two values of that key, every other key held at its `hp0` value.
"""
function probe_grad_rule(e, hp0; elt::Type = Float32)
    changed(a, b) = Symbol[
        f for f in fieldnames(typeof(a))
            if getfield(a, f) isa Number && !isequal(getfield(a, f), getfield(b, f))
    ]
    at(hp, k, v) = merge(hp, NamedTuple{(k,)}((elt(v),)))
    out = Dict{Symbol, Vector{Symbol}}()
    for k in keys(hp0)
        k === :bind && continue
        out[k] = changed(build_grad_rule(e, at(hp0, k, 0.25)), build_grad_rule(e, at(hp0, k, 0.75)))
    end
    if haskey(hp0, :bind)
        for k in keys(hp0.bind)
            b(v) = merge(hp0, (; bind = at(hp0.bind, k, v)))
            out[Symbol("bind.", k)] =
                changed(build_grad_rule(e, b(0.25)), build_grad_rule(e, b(0.75)))
        end
    end
    return out
end

"""
    ReactantNitro.check_grad_bindings(e, resolved, landed) -> landed

The setup check behind `grad_rule(e, hp)` and `bind`: every scheduled grad field must change the
rule, and every `bind` key must land somewhere, in a Level 2 optimizer chain (`landed`, from
[`schedule_bindings`](@ref)) or in the grad rule. Returns `landed` with the grad rule's landings.
"""
function check_grad_bindings(e, resolved, landed)
    half(nt) = map(_ -> 0.5f0, nt)
    found = probe_grad_rule(e, grad_hp(half(resolved.grad), half(resolved.bind)))
    for k in keys(resolved.grad)
        isempty(found[k]) && error(
            "ReactantNitro: `grad.$k` is scheduled, but `grad_rule(e, hp)` returns the same rule \
             for two values of `hp.$k`. Read it (`SAM(hp.$k)`), or remove the schedule."
        )
    end
    out = Dict{Symbol, Vector{String}}(k => copy(v) for (k, v) in landed)
    for k in keys(resolved.bind)
        key = Symbol("bind.", k)
        R = nameof(typeof(build_grad_rule(e, grad_hp(half(resolved.grad), half(resolved.bind)))))
        append!(get!(out, key, String[]), ["$R.$f" for f in found[key]])
        isempty(out[key]) && error(
            """
            ReactantNitro: `bind.$k` is scheduled, but nothing reads it: neither a Level 2
            `optimizer(e, group, hp)` nor `grad_rule(e, hp)` changes when `hp.bind.$k` does, so the
            schedule would compute a value every step and discard it. Read `hp.bind.$k` in one of
            them, or remove the schedule."""
        )
    end
    return out
end

"""
    ReactantNitro.eps_program(rule, ps, g, masks, adaptive, lref) -> ps_adv

`ps + ε` as a fresh tree; `ps` itself is not written, since pass 1's values must survive.
"""
function eps_program(rule, ps, g, masks, adaptive, lref)
    layout = resolve_layout(lref)
    ε = unflatten(_perturbation(rule, g, flatten(ps, layout), masks, adaptive), ps, layout)
    return Functors.fmap(ps, ε) do p, e
        p isa AbstractArray ? program_out(p .+ e) : p
    end
end
