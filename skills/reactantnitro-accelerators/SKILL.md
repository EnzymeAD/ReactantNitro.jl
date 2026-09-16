---
name: reactantnitro-accelerators
description: >
  Choosing and configuring the accelerator a ReactantNitro run executes on: CPU, CUDA, ROCm,
  TPU, and how many devices it shards over. The one-process-one-XLA model, `setup_devices!`
  (REPL) and its tool form `nitro_setup` (Kaimon) as one shared code path, the `n_devs` pin
  and its precedence, and `CUDA_VISIBLE_DEVICES` for restricting GPUs. Invoke when starting a
  run on a specific accelerator, when a run unexpectedly uses the wrong backend or the wrong
  number of devices, or when deciding what `n_devs` means for a session.
---

# Accelerators and devices

## The model: one process, one XLA

A Julia process initializes its XLA/PJRT client **once**, at the first device access, and the
client is then fixed for the process's lifetime. Everything in this page is a consequence of
that fact:

- **`n_devs` never creates clients or processes.** It slices the already-visible device set to
  build the `Sharding.Mesh`; XLA partitions ONE compiled program over that mesh. There is
  no MPI and no NCCL here, and `n_devs = 4` is not "four training processes".
- **Backend choice is a process event.** Reactant's default backend is the highest-priority
  working one, decided at first device access (GPU where one is visible, else CPU). Changing
  the backend after the fact is allowed (`set_default_backend` swaps the pointer) but device
  VISIBILITY is frozen: `CUDA_VISIBLE_DEVICES` must be set before the process starts.
- **The CPU backend reports a single device.** With no GPU visible, `Reactant.devices()`
  returns one device, so a CPU session's `n_devs` is 1 and the mesh is skipped.

## Choosing the backend

One code path, two entry points: the framework function `ReactantNitro.setup_devices!`, and the
Kaimon tool `nitro_setup`, which is a thin wrapper over exactly that function. A REPL session
and a Kaimon session therefore configure identically.

**REPL / a script (the non-Kaimon workflow):**

```julia
using ReactantNitro
setup_devices!(backend = "cpu")            # run everything on CPU
setup_devices!(backend = "cuda", n_devs = 2)  # two of the visible CUDA devices
setup_devices!()                           # report what is in effect
```

`backend` passes through to `Reactant.set_default_backend` and names a Reactant backend:
`"cpu"`, `"gpu"` (whichever of CUDA/ROCm is available), `"cuda"`, `"rocm"`, `"tpu"`.

**Kaimon (the tool form):**

```julia
nitro_setup(backend = "cpu")
nitro_setup(backend = "cuda", n_devs = 2)
nitro_setup()   # report
```

The tool returns a one-line report; the function returns `(; backend, visible, n_devs, pinned)`.

**Calling it is optional.** A session that never calls it runs on Reactant's default backend
with `n_devs` = every visible device, `length(Reactant.devices())`, which is what a CPU
machine gets automatically. The setup call exists to be explicit and to fail fast: an unknown
backend or a device count larger than what is visible is a loud error, not a silent fallback.

**If you are porting:** your model may have run under a different framework where the backend
was chosen per-run or per-process via environment or a config file. Here the backend is set at
the process level before any run, and a run never re-selects it.

## `n_devs`: what it is and what it is not

- It is the number of LOCAL, VISIBLE devices the batch is sharded over on the `:data` axis of a
  1-D mesh. `1` skips the mesh entirely.
- It is NOT a process count, a batch multiplier, or a memory setting. The batch size is GLOBAL
  and is split across the mesh: `batch_size = 32` on four devices puts 8 samples on each, and
  its numerics are those of a 32-sample batch, not a 128-sample one. Adding devices buys
  throughput and does not change the effective batch.
- The default is every visible device, so a host with four GPUs data-parallelizes **without
  being asked**. If that is surprising, pin a count or set `CUDA_VISIBLE_DEVICES`.

### The pin and its precedence

`setup_devices!(n_devs = k)` (or `nitro_setup(n_devs = k)`) pins the count for the whole
process. Resolved order, highest wins:

1. an explicit `n_devs` keyword on `Nitro` / `train!` / `nitro_train`, for one run
2. the session pin from `setup_devices!` / `nitro_setup`
3. the experiment's declared `n_devs` field
4. `length(Reactant.devices())`, every visible device

So a pinned session overrides what an experiment declares (the session operator's explicit
choice beats the model author's default), and a per-run keyword overrides the pin. A batch that
does not divide evenly across the mesh is a setup error naming both numbers.

## Restricting GPUs: `CUDA_VISIBLE_DEVICES`

To run on a subset of the GPUs on a host, set `CUDA_VISIBLE_DEVICES` **before starting the
process** (in the shell, the job script, or the session launcher); visibility is fixed at the
first XLA client access and cannot be changed mid-process. `n_devs` can then only select within
what is visible, and asking for more than is visible is an error that names the variable:

```
ReactantNitro: `n_devs = 4` but Reactant sees 2 device(s).
`n_devs` counts LOCAL, VISIBLE devices and the default is all of them, so this is either a
hand-set value that is too large or a `CUDA_VISIBLE_DEVICES` narrower than you meant.
```

The same rule applies at the Reactant level: `Reactant.devices()` is what the process can see,
and it is fixed once the client exists.

## Reading the current state

`setup_devices!()` / `nitro_setup()` with no arguments change nothing and report: the backend
in use (`Reactant.XLA.platform_name(Reactant.XLA.default_backend())`), how many devices are
visible, and the `n_devs` the next run will use, plus whether it is pinned or the default.

For the underlying library's full platform matrix (driver versions, detection, disabling GPU
support), the Reactant documentation's configuration page is the reference; the framework does
not restate it.
