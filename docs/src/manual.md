# Manual training mode: you own the step

In the automatic loop you declare `forward`, `loss`, and the optimizer accessors, and the
framework sequences the forwards, the backwards, the optimizer steps, and the metrics. In **manual
mode** you define `train_step` and own the whole optimizer step, in whatever order your algorithm
demands; the framework keeps everything outside it.

Use it when the automatic loop cannot express your algorithm. The typical case is a GAN:
one generator and one discriminator in a single parameter tree, each with its own optimizer, two
losses computed against the same batch, and a generator gradient that must flow through the
discriminator's forward as a function of the fake data, never into the discriminator's parameters.

## The shape of the contract

Three methods select and describe a manual experiment:

```julia
# 1. Defining `train_step` selects manual mode.
ReactantNitro.train_step(e::MyGAN, model, ps, opt_state, st; x, z) -> (; loss, ps, st, opt_state, stats)

# 2. `setup_optimizers` supplies the optimizer states, one per network or group. The framework
#    moves the result to the device and asserts that no host `Number` remains in it.
ReactantNitro.setup_optimizers(e::MyGAN, model, ps, st, mesh) -> opt_state

# 3. `manual_training(e) -> Bool` is the mode flag. The default is whether `train_step` has a
#    method for this experiment type; return `false` to keep the method but use the automatic loop.
ReactantNitro.manual_training(::MyGAN) = false
```

`train_step` (the closure, below) is called **once per optimizer step**, in train mode (the
framework owns the `Lux.trainmode` switch), on one **device** batch whose fields are routed from the
keywords the method declares, as for `forward`. It is traced and compiled into one XLA program per
run; nothing recompiles after step 1.

## A minimal GAN

A complete least-squares GAN, the worked example the [`train_step`](@ref) docstring refers to:

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
    gen = Optimisers.setup(Optimisers.Adam(2.0f-4), ps.gen),
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
        mean(abs2, d3 .- 1.0f0)                    # non-saturating: fake -> 1
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

Four parts of the example are the contract:

- **Noise comes in the batch.** `z` is a batch field like any other, routed from the keyword the
  closure declares and transferred each step. Draw it fresh per batch: with a fixed noise sequence
  the generator can memorize the draw instead of learning the mapping from noise to data. Fresh
  draws happen in the loader's own iteration (reshuffling or regenerating), as for any other data
  source.
- **Each sub-network applies with its own state subtree.** `st.gen` and `st.disc` are separate, and
  `st` stays the whole tree for the backwards and the return. The returned state is assembled as
  `(; gen = st_gen, disc = st.disc)`.
- **The backwards take everything as arguments.** `backward`'s objective reads the model (a plain
  struct, safe to capture) and takes every *traced* value it needs as an explicit argument. A
  traced value captured by the closure is treated as a constant, which zeroes its gradient without
  any error. Passing `ps.disc` as an argument to the generator's `backward`, where it is a `Const`,
  makes the generator's loss flow through the discriminator's forward as a function of the fake
  data and never into the discriminator's parameters. No explicit stop-gradient op is needed.
- **The optimizer steps return states, not leaves.** `step_optimizer` returns the new optimizer
  *states*; the driver re-attaches the rules it handed in. This is required because a rule's device
  scalars cannot leave a compiled program as an output on a multi-device mesh, though arrays can;
  the automatic loop's optimizer program returns states for the same reason.

## The return contract

The driver checks the closure's return every step, validates `loss`, and logs `(; loss, stats...)`
on the train line, together with any schedule values:

| Key | Meaning |
| --- | --- |
| `loss` | **Required.** The scalar the driver validates (non-finite stops the run, naming step and epoch, unless `check_divergence = false`) and logs |
| `ps` | The updated parameter tree |
| `st` | The updated layer state |
| `opt_state` | The new optimizer **states only**; the driver re-attaches the rules |
| `stats` | Device scalars logged on each train line (every [`log_every_n_steps`](@ref) steps), `(;)` allowed |

## Learning rate schedules

`opt`-keyed schedules work in manual mode. Manual mode has no parameter groups and no
`learning_rate` ratios, so every value is **absolute** and the keys address the rules in
`opt_state` directly. A bare key is a rule field name and sets that field on **every** rule in
`opt_state` that has it; a nested key is a **path into the `opt_state` tree** and sets the field on
the rules under that path only:

```julia
ReactantNitro.schedules(e::ToyGAN) = (;
    opt = (;
        eta  = total -> t -> 2.0f-4 * (1 - t / total),  # every rule's eta
        disc = (; eta = _ -> _ -> 1.0f-4),              # opt_state.disc's rules' eta
    ))
```

Here the generator follows the linear decay and the discriminator stays at `1f-4`. A path key
overrides the bare key for the rules under its path, whatever order the two are written in.

The driver rebuilds the named rules on the host before each call, with new scalar values and the
same rule types, so one compiled program serves every step. A path that does not exist in
`opt_state`, a field no rule under the path has, a non-scalar field, and a field that
`nonschedulable` excludes are setup errors naming the offender. The binding report shows each key
and the path it binds.

The automatic loop uses the same syntax with different values: its paths name parameter groups,
and its values are scaled by each group's `learning_rate` ratio ([Schedules](schedules.md)).

`device`-keyed schedules work as in the automatic loop: the driver rewrites the scheduled
`Device` fields before each call, and the closure reads them as traced inputs.

For a per-step value an `opt` key cannot reach, the closure can read a `device`-scheduled field or
a batch field and rebuild its own rules with the traced value; this recompiles nothing.

## What manual mode does not give you in v1

- **`accum > 1` is a setup error.** Accumulation is the automatic loop's mechanism; the closure
  owns its own multi-batch work instead, by looping inside `train_step`.
- **The `train_metrics` hook is unused.** `stats` come from the closure itself.

## Everything else is unchanged

Validation, checkpointing, early stopping, `request_stop!`, phases, `data_wait_frac`, schedule
logging, and the prefetch pipeline work as in the automatic loop. The eval side (`validate`,
`evaluate`, `predict`) never consults the mode: it uses `forward` and `metrics`, so a manual
experiment defines those for its eval surface, and `loss` is only needed if you want the framework
to substitute `loss` when `metrics` is absent.

## Deferred

Out of v1:

- **`accum > 1`**, via a closure per micro-batch with a type-level last-in-group flag, or by
  handing the closure a tuple of micro-batches.
- **Host-resident metrics from the closure**: `stats` are in-graph scalars in v1; a host branch
  would transfer the closure's outputs per step.
- **Multi-device mesh verification**: CPU tests cannot see a mesh, and the rules-in, states-out
  contract needs a real multi-device run to verify.
- **A `Nitro(..., manual = ...)` keyword** as sugar over the `manual_training` accessor.

## Toggling the mode

Defining `train_step` selects manual mode; `manual_training(e) = false` declines it while keeping
the method in source. This works with Revise: redefining the accessor is an ordinary method edit,
and the next `Nitro` you build picks up the new driver. The mode is frozen at
construction, so an existing handle keeps its driver and `fixed_config_report`
names the redefinition.
