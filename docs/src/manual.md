# Manual training mode: you own the step

The automatic loop is one way to train, and it is the default: you declare `forward`, `loss`, and
the optimizer accessors, and the framework sequences the forwards, the backwards, the optimizer
steps, and the metrics. **Manual mode is the other way.** You define `train_step` for your
experiment type and the framework hands you the whole optimizer step: you sequence the forwards,
the backwards, the optimizer steps, and the device metrics, in whatever order your algorithm
demands. The framework keeps everything outside the step.

Reach for it when the automatic loop cannot express your algorithm. The canonical case is a GAN:
one generator and one discriminator in a single parameter tree, each with its own optimizer, two
losses computed against the same batch, and a generator gradient that must flow through the
discriminator's forward as a function of the fake data, never into the discriminator's parameters.

## The shape of the contract

Three things select and describe a manual experiment:

```julia
# 1. Defining `train_step` is what selects manual mode. The automatic loop has no `train_step`;
#    its step is framework-owned. `manual_training(e) = false` declines manual mode while keeping
#    the method in source.
ReactantNitro.train_step(e::MyGAN, model, ps, opt_state, st; x, z) -> (; loss, ps, st, opt_state, stats)

# 2. `setup_optimizers` supplies the optimizer states, one per network or group. The framework
#    normalizes the result to device residency and asserts the no-host-Number property.
ReactantNitro.setup_optimizers(e::MyGAN, model, ps, st, mesh) -> opt_state

# 3. `manual_training(e) -> Bool` is the mode flag. The default checks whether `train_step` has a
#    method for this experiment type; override it to keep the method while using the automatic
#    loop.
ReactantNitro.manual_training(::MyGAN) = false
```

The closure is called **once per optimizer step**, in train mode (the framework owns the
`Lux.trainmode` switch), on one transferred **device** batch whose fields are routed from the
keywords the method declares, exactly like `forward`. The method is traced and compiled into one
XLA program per run; nothing recompiles after step 1.

## The very simple GAN

A complete, minimal least-squares GAN, the same recipe the `train_step` docstring carries:

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

Four things in the example are the whole of the contract, and each has a reason:

- **Noise rides in the batch.** `z` is a batch field like any other, routed from the keyword the
  closure declares, so it is transferred per step. Draw it fresh per batch: a GAN's generator must
  learn the mapping from noise to data, and a fixed noise sequence lets it memorize the draw. The
  loader's own iteration (reshuffling or regenerating) is where fresh draws happen, exactly as for
  any other data source.
- **Each sub-network applies with its own state subtree.** `st.gen` and `st.disc` are separate, and
  `st` stays the whole tree for the backwards and the return. The returned state is assembled as
  `(; gen = st_gen, disc = st.disc)`.
- **The backwards take everything as arguments.** `backward`'s objective reads the model (a plain
  struct, safe to capture) and takes every *traced* value it needs as an explicit argument. A
  closure capture of a traced value is silently treated as a constant, which zeroes the gradient
  without any error. The all-arguments form is what makes the non-saturating generator gradient
  work: `ps.disc` as a `Const` argument means the generator's loss flows through the
  discriminator's forward as a function of the fake data and never into the discriminator's
  parameters. No explicit stop-gradient op is needed.
- **The optimizer steps return states, not leaves.** `step_optimizer` returns the new optimizer
  *states*; the driver re-attaches the rules it handed in. The asymmetry is forced: a rule's device
  scalars cannot leave a compiled program as an output on a multi-device mesh, though arrays can
  (the same rule that makes the automatic `opt_program` return states).

## The return contract

The driver checks the closure's return every step and validates `loss` against the framework's
fail-fast rule, then logs `(; loss, stats...)`:

| Key | Meaning |
| --- | --- |
| `loss` | **Required.** The scalar the driver validates (non-finite stops the run, naming step and epoch) and logs |
| `ps` | The updated parameter tree |
| `st` | The updated layer state |
| `opt_state` | The new optimizer **states only**; the driver re-attaches the rules |
| `stats` | Device scalars logged once per step, `(;)` allowed |

## Learning rate schedules

`opt`-keyed schedules work in manual mode, with a **path-bound** binding. The automatic loop
binds `opt.eta` to every parameter group's chain, ratio-scaled through the `learning_rate`
accessors; manual mode has no groups, so the keys name the rules themselves. A bare key is a rule
field name and binds **every** rule in `opt_state` with that field, at the absolute value; a
nested key is a **path into the `opt_state` tree**:

```julia
ReactantNitro.schedules(e::ToyGAN) = (;
    opt = (;
        eta  = total -> t -> 1.0f-3 * (1 - t / total),  # every rule's eta, absolute
        gen  = (; eta = total -> t -> 1.0f-4 * ...),     # opt_state.gen's rules' eta
        disc = (; eta = _ -> _ -> 2.0f-3),               # opt_state.disc's rules' eta
    ))
```

The structure IS the binding: with two optimizers, `opt.gen.eta` says which one, and the two can
carry different curves. The driver rebuilds the named rules between closure calls, host-side and
type-preserving (fresh scalar values, same rule types), so the same compiled program serves every
step, exactly like the automatic loop's `rebuild_rules`. A path that does not exist in `opt_state`
and a field that `nonschedulable` excludes are setup errors naming the offender; the binding
report shows each key and the path it binds.

The automatic loop's `opt` keys name **parameter groups** instead of `opt_state` paths, and its
path values are ratio-scaled through the `learning_rate` accessors rather than absolute. The two
modes share the dotted-key syntax and differ in value semantics by design
([Schedules](schedules.md)).

`device`-keyed schedules work exactly as in the automatic loop: the driver rewrites the scheduled
`Device` fields before each call, and the closure reads them as traced inputs.

For a per-step value the framework does not schedule, the closure can read a `device`-scheduled
field or a batch field and rebuild its own rules with the traced value; that route is exact and
recompiles nothing.

## What manual mode does not give you in v1

- **`accum > 1` is a setup error.** Accumulation is the automatic loop's mechanism; the closure
  owns its own multi-batch work instead, by looping inside `train_step`.
- **The `train_metrics` hook is unused.** `stats` come from the closure itself.

## Everything else is unchanged

Validation, checkpointing, early stopping, `request_stop!`, phases, `data_wait_frac`, and the
prefetch pipeline all work exactly as in the automatic loop. The eval side (`validate`,
`evaluate`, `predict`) never consults the mode: it uses `forward` and `metrics`, so a manual
experiment defines those for its eval surface, and `loss` is only needed if you want the framework
to substitute `val_loss` when `metrics` is absent.

## Deferred

Explicitly out of v1, and the natural next rounds:

- **`accum > 1`**, via a closure per micro-batch with a type-level last-in-group flag, or by
  handing the closure a tuple of micro-batches.
- **Host-resident metrics from the closure**: `stats` are in-graph scalars in v1; a host branch
  would transfer the closure's outputs per step.
- **Multi-device mesh verification**: CPU tests cannot see a mesh, and the rules-in/states-out
  contract is exactly the kind of thing that wants a real-device run.
- **A `Nitro(..., manual = ...)` keyword** as sugar over the `manual_training` accessor.

`opt`-keyed schedules were on this list; they are built (path-bound keys, above).

## Toggling the mode

Defining `train_step` selects manual mode; `manual_training(e) = false` declines it while keeping
the method in source, which is the Revise-friendly toggle: redefining the accessor is an ordinary
method edit, and the next `Nitro` you build picks up the new driver. The mode is frozen at
construction, so an existing handle keeps its driver and `fixed_config_report`
names the redefinition.
