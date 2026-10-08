# Devices and meshes

## One process, one XLA client

A Julia process initializes its XLA client once, at the first device access, and the client is
fixed for the life of the process. Three things follow:

- **The backend is chosen once.** Reactant picks the highest-priority working backend at first
  access: a GPU where one is visible, else the CPU. Call [`setup_devices!`](@ref) before the first
  `Nitro` to choose explicitly.
- **Device visibility is frozen.** `CUDA_VISIBLE_DEVICES` must be set before the process starts.
- **`n_devs` creates no processes.** It selects visible devices to build a 1-D
  `Reactant.Sharding.Mesh`, and XLA partitions one compiled program over it. There is no MPI and no
  NCCL setup to write.

## Choosing the backend

```julia
setup_devices!(backend = "cpu")               # run everything on CPU
setup_devices!(backend = "cuda", n_devs = 2)  # two of the visible CUDA devices
setup_devices!()                              # report what is in effect, change nothing
```

`backend` is a Reactant backend name: `"cpu"`, `"gpu"` (CUDA or ROCm, whichever is present),
`"cuda"`, `"rocm"`, `"tpu"`. The call returns `(; backend, visible, n_devs, pinned)`. An unknown
backend, or more devices than are visible, is an error rather than a fallback. The Kaimon tool
`nitro_setup` is a wrapper over the same function; see [Kaimon](kaimon.md).

The call is optional. Without it a run uses Reactant's default backend and every visible device.

## What `n_devs` means

`n_devs` is the number of local, visible devices the batch is sharded over, on the mesh's `:data`
axis. `1` skips the mesh.

- **The batch size is global.** `batch_size = 32` on four devices puts 8 samples on each, and the
  numerics are those of a 32-sample batch. More devices buy throughput, not a larger batch.
- **The default is every visible device**, so a host with four GPUs trains data-parallel without
  being asked. Pin a count, or narrow `CUDA_VISIBLE_DEVICES`, if that is not what you want.
- **A batch that does not divide evenly across the mesh** is a setup error naming both numbers.

The count resolves in this order, highest first:

1. an `n_devs` keyword on `Nitro` or `train!`, for one run
2. the session pin from `setup_devices!(n_devs = k)`
3. the experiment's [`n_devs`](@ref), which reads an `n_devs` field by default
4. `length(Reactant.devices())`

## What the mesh computes

The mesh runs one global program, so a layer sees the whole batch, not one device's share:

- **BatchNorm statistics** are over the global batch.
- **A `Dropout` mask** is one global mask, split across the devices. Every `ReactantRNG` is placed
  with the `"PHILOX"` algorithm at any device count, so one seed draws the same masks on one device
  or many.
- **Results match `n_devs = 1` to a tolerance, not bitwise.** Reductions such as the gradient sum
  run in a different order across devices.

Parameters, optimizer state, and RNG state are replicated; each batch is sharded along its last
axis.

## Restricting GPUs

Set `CUDA_VISIBLE_DEVICES` in the shell or job script before Julia starts. `n_devs` can then only
select within what is visible, and asking for more raises:

```
ReactantNitro: `n_devs = 4` but Reactant sees 2 device(s).
`n_devs` counts LOCAL, VISIBLE devices and the default is all of them, so this is either a
hand-set value that is too large or a `CUDA_VISIBLE_DEVICES` narrower than you meant.
Note the backend matters: with no GPU visible, Reactant reports a single CPU device.
```

## Testing a mesh on CPU

The CPU backend reports one device. To exercise a mesh without GPUs, force more host devices
before Reactant loads:

```julia
ENV["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"
using ReactantNitro
setup_devices!(backend = "cpu", n_devs = 2)
```

The framework's own multi-device tests run this way, in a subprocess, since the flag must be set
before the XLA client exists.
