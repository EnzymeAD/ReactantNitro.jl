---
name: reactantnitro-manual
description: >
  Train a ReactantNitro experiment in manual mode, where the experiment owns the whole
  optimizer step: `train_step` and how defining it selects the mode, `setup_optimizers`,
  the checked return contract, `backward` with the objective discipline (a closure
  capture of a traced value silently zeroes the gradient), `step_optimizer`, path-bound
  `opt` schedules, and what the automatic loop still does for you. Invoke when the
  automatic loop cannot express an algorithm, the canonical case being a GAN with two
  optimizers, or when reading or writing a `train_step`.
---

# Manual training mode: you own the step

The automatic loop is one way to train, and it is the default: you declare `forward`, `loss`,
and the optimizer accessors, and the framework sequences the forwards, the backwards, the
optimizer steps, and the metrics. **Manual mode is the other way.** You define `train_step` for
your experiment type and the framework hands you the whole optimizer step: you sequence the
forwards, the backwards, the optimizer steps, and the device metrics, in whatever order your
algorithm demands. The framework keeps everything outside the step.

Reach for it when the automatic loop cannot express your algorithm. The canonical case is a GAN:
one generator and one discriminator in a single parameter tree, each with its own optimizer, two
losses computed against the same batch, and a generator gradient that must flow through the
discriminator's forward as a function of the fake data, never into the discriminator's parameters.

## What selects the mode

**Defining `train_step` for an experiment type is what switches `train!` to manual mode.** The
automatic loop has no `train_step`: its step is framework-owned, `grad_program` plus
`opt_program` sequenced by the driver. So the act of writing the method is the switch, and
`manual_training(e) = false` declines it while keeping the method in source, which is the
Revise-friendly toggle: redefining the accessor is an ordinary method edit, and the next `Nitro`
you build picks up the new driver.

**The mode is read once, at construction, and frozen**: an existing handle keeps
the driver it was built with, and `fixed_config_report` names the redefinition rather than
applying it silently. An experiment that defines neither method trains automatically, and an
experiment that defines `train_step` but is only evaluated never consults it.

## The step contract

```julia
train_step(e, model, ps, opt_state, st; <declared batch fields>)
    -> (; loss, ps, st, opt_state, stats)
```

Called **once per optimizer step**, in train mode (the framework owns the `Lux.trainmode`
switch), on one transferred **device** batch. The batch's fields are routed by keyword from what
this method declares, exactly like `forward`: a field only the closure reads (a GAN's noise) is
transferred only when declared. **The method is traced and compiled by the framework**, so it must
be a stable named method, never an anonymous closure built per step, and nothing recompiles after
step 1.

The return is a checked `NamedTuple`:

| Key | Meaning |
| --- | --- |
| `loss` | **Required.** The scalar the driver validates (non-finite stops the run, naming step and epoch) and logs |
| `ps` | The updated parameter tree |
| `st` | The updated layer state; a stateless model returns `st` unchanged |
| `opt_state` | The new optimizer **states only**, not the leaves; the driver re-attaches the rules |
| `stats` | Device scalars logged once per step (train metrics), `(;)` allowed |

## `setup_optimizers`

```julia
setup_optimizers(e, model, ps, st, mesh) -> opt_state
```

**Required for a manual experiment.** Build the optimizer states you own, one per network or
group:

```julia
ReactantNitro.setup_optimizers(e::ToyGAN, model, ps, st, mesh) = (;
    gen  = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.gen),
    disc = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.disc))
```

Called once at construction, after device conversion and `build_model`. The framework normalizes
the result to device residency and asserts the no-host-`Number` property on it, so a rule's
scalars are traced and an integer step counter that would freeze under trace is promoted before
it can. The result is what `train_step` receives as `opt_state`.

## `backward`: the objective discipline

```julia
backward(f, ps_sub, consts...) -> (loss, grads)
```

One reverse-mode pass over an objective, w.r.t. one parameter subtree. `f(ps_sub, consts...)`
returns the scalar loss; the returned `grads` matches `ps_sub`, and everything in `consts...` is
`Const`, so no gradient ever flows into it.

**Every traced value the objective reads must be an argument, never a closure capture.**
Measured: a captured parameter subtree silently zeroed the gradient, with the primal loss correct,
which is the worst failure mode because nothing errors at any point. A plain struct like the model,
which holds no arrays, is safe to capture; parameters, batches, and state go in `consts...`.

The non-saturating GAN formulation falls out of the activities. In the generator's backward, the
discriminator's parameters are a `Const` argument: the loss's adjoint flows through the
discriminator's forward as a function of the fake data and never into the discriminator's
parameters. No explicit stop-gradient op is needed.

## `step_optimizer`

```julia
step_optimizer(opt_state, ps, grads) -> (states, ps_new)
```

One tree-level optimizer step for use inside the closure: `Optimisers.update` under trace,
returning the new parameter tree and the new optimizer **states**. Call it once per network or
group, passing that subtree:

```julia
s_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
```

The rules are stripped from the return, and the driver re-attaches the rules it handed in. That
asymmetry is forced, not tidy: a rule's device hyperparameters are replicated scalars, and a
replicated scalar cannot leave a compiled program as an output on a mesh, though arrays can. The
same rule is why the automatic `opt_program` returns states.

## Schedules

`opt`-keyed schedules work in manual mode, with a **path-bound** binding. The automatic loop
binds `opt.eta` to every parameter group's chain, ratio-scaled through the `learning_rate`
accessors; manual mode has no groups, so the keys name the rules themselves. **A bare key is a
rule field name and binds every rule in `opt_state` with that field, at the absolute value**;
**a nested key is a path into the `opt_state` tree**:

```julia
ReactantNitro.schedules(e::ToyGAN) = (;
    opt = (;
        eta  = total -> t -> 1.0f-3 * (1 - t / total),  # every rule's eta, absolute
        gen  = (; eta = total -> t -> 1.0f-4 * ...),     # opt_state.gen's rules' eta
        disc = (; eta = _ -> _ -> 2.0f-3),               # opt_state.disc's rules' eta
    ))
```

**The structure IS the binding.** With two optimizers, `opt.gen.eta` says which one, and the two
can carry different curves. A path that does not exist in `opt_state` and a field that
`nonschedulable` excludes are setup errors naming the offender; the binding report shows each key
and the path it binds. The driver rebuilds the named rules between closure calls, host-side and
type-preserving (fresh scalar values, same rule types), so the same compiled program serves every
step, exactly like the automatic loop's `rebuild_rules`.

`device` keys work exactly as in the automatic loop: the driver rewrites the scheduled `Device`
fields before each call, and the closure reads them as traced inputs. For a per-step value the
framework does not schedule, the closure can read a `device`-scheduled field or a batch field and
rebuild its own rules with the traced value; that route is exact and recompiles nothing.

Two things the automatic loop owns that manual mode does not take: **`accum > 1` is a setup
error**, because accumulation is the automatic loop's mechanism, and the closure owns its own
multi-batch work by looping inside `train_step`; and **the `train_metrics` hook is unused**,
because `stats` come from the closure itself.

## What the automatic loop still does for you

Validation, checkpointing, early stopping, `request_stop!`, phases, and the prefetch pipeline all
work exactly as in the automatic loop. The eval side (`validate`, `evaluate`, `predict`) never
consults the mode: it uses `forward` and `metrics`, so a manual experiment defines those for its
eval surface, and `loss` becomes optional, needed only for the framework's `val_loss`
substitution when `metrics` is also absent.

## The very simple GAN

A complete, minimal least-squares GAN. Four patterns in it are the whole of the contract: noise
rides in the batch (routed from the keyword the closure declares, drawn fresh per batch so the
generator cannot memorize a fixed draw); each sub-network applies with its own state subtree while
`st` stays the whole tree for the backwards and the return; the backwards take everything traced
as arguments, which is what makes the non-saturating gradient work; and the optimizer steps
return states, not leaves.

```julia
using Statistics: mean   # or `using Statistics` at the top of the file

@experiment struct ToyGAN
    max_epochs::Host{Int} = 50
end

ReactantNitro.build_data(e::ToyGAN, dist) = (;
    train = [(; x = randn(Float32, 4, 32), z = randn(Float32, 2, 32)) for _ in 1:10])

ReactantNitro.build_model(e::ToyGAN, rng) = begin
    model = (; gen  = Lux.Chain(Lux.Dense(2 => 16, tanh), Lux.Dense(16 => 4)),
               disc = Lux.Chain(Lux.Dense(4 => 16, tanh), Lux.Dense(16 => 1)))
    (model, Lux.setup(rng, model)...)
end

ReactantNitro.setup_optimizers(e::ToyGAN, model, ps, st, mesh) = (;
    gen  = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.gen),
    disc = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.disc))

function ReactantNitro.train_step(e::ToyGAN, model, ps, opt_state, st; x, z)
    fake, st_gen = Lux.apply(model.gen, z, ps.gen, st.gen)
    d_real, _ = Lux.apply(model.disc, x, ps.disc, st.disc)
    d_fake, _ = Lux.apply(model.disc, fake, ps.disc, st.disc)

    l_d, g_d = ReactantNitro.backward(ps.disc, x, fake, st) do ps_d, xc, fc, stc
        d1, _ = Lux.apply(model.disc, xc, ps_d, stc.disc)
        d2, _ = Lux.apply(model.disc, fc, ps_d, stc.disc)
        mean(abs2, d1 .- 1.0f0) + mean(abs2, d2)    # real -> 1, fake -> 0
    end

    l_g, g_g = ReactantNitro.backward(ps.gen, ps.disc, z, st) do ps_g, ps_d, zc, stc
        f2, _ = Lux.apply(model.gen, zc, ps_g, stc.gen)
        d3, _ = Lux.apply(model.disc, f2, ps_d, stc.disc)
        mean(abs2, d3 .- 1.0f0)                     # non-saturating: fake -> 1
    end

    s_g, ps_g = ReactantNitro.step_optimizer(opt_state.gen, ps.gen, g_g)
    s_d, ps_d = ReactantNitro.step_optimizer(opt_state.disc, ps.disc, g_d)

    return (; loss = l_d + l_g,
              ps = (; gen = ps_g, disc = ps_d),
              st = (; gen = st_gen, disc = st.disc),
              opt_state = (; gen = s_g, disc = s_d),
              stats = (; l_d, l_g))
end

train!(Nitro(ToyGAN()))
```
