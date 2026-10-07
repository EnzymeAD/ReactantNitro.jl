# Compile options

Every program a [`Nitro`](@ref) handle compiles goes through `Reactant.Compiler.compile`, the call
behind `@compile`. The `compile_options` keyword passes Reactant's compile keywords to all of them,
so anything you would write as `@compile raise = true f(x)` is set once, on the handle.

```julia
nitro = Nitro(e; compile_options = (; raise = true))
train!(nitro)
```

## Two forms

A `NamedTuple` of the keywords `Reactant.Compiler.compile` takes:

```julia
Nitro(e; compile_options = (; raise = true, cudnn_hlo_optimize = true, transpose_propagate = :down))
Nitro(e; compile_options = (; xla_debug_options = (; xla_gpu_exhaustive_tiling_search = true)))
```

A `Reactant.CompileOptions`, for the pass switches that are fields of `CompileOptions` but not
compile keywords:

```julia
Nitro(e; compile_options = Reactant.CompileOptions(; raise = true, disable_slice_to_batch_passes = false))
```

The two combine: a `NamedTuple` may carry a `CompileOptions` under its own `compile_options` key.

## Which keywords exist

The accepted set is read from your installed Reactant, not hard-coded, so a keyword added upstream
works with no change here. Commonly useful ones:

| Keyword | Effect |
| --- | --- |
| `raise` | `true` runs Reactant's raising pipeline, which lowers custom kernels (CUDA, KernelAbstractions) to StableHLO; a `String` is a custom raising pass pipeline |
| `raise_first` | raise before the main optimization passes rather than after |
| `optimize` | `false` skips the optimization passes, or a `Symbol`/`String` picks a pipeline |
| `cudnn_hlo_optimize` | enables the cuDNN HLO rewrites |
| `transpose_propagate`, `reshape_propagate` | `:up`, `:down` or `:none` |
| `no_nan`, `all_finite` | let the optimizer assume finite values |
| `xla_debug_options` | XLA debug flags, as a `NamedTuple` |

Reactant forces `raise = true` on TPU and for sharded programs, so setting it there changes nothing.

## XLA options

Three keywords pass fields straight to XLA's own protos. The fields are XLA's, so XLA documents
them; this page only shows how to reach them.

| Keyword | XLA proto | Reference |
| --- | --- | --- |
| `xla_debug_options` | `DebugOptions` | [`xla.proto`](https://github.com/openxla/xla/blob/main/xla/xla.proto), [flag guidance](https://openxla.org/xla/flags_guidance) |
| `xla_executable_build_options` | `ExecutableBuildOptionsProto` | [`compile_options.proto`](https://github.com/openxla/xla/blob/main/xla/pjrt/proto/compile_options.proto), [effort levels](https://openxla.org/xla/effort_levels) |
| `xla_compile_options` | `CompileOptionsProto` | [`compile_options.proto`](https://github.com/openxla/xla/blob/main/xla/pjrt/proto/compile_options.proto) |

### Autotuning

GPU autotuning is set through `DebugOptions`:

```julia
Nitro(e; compile_options = (; xla_debug_options = (;
    xla_gpu_autotune_level = 0,                 # no GEMM/convolution autotuning: faster compiles
    xla_gpu_exhaustive_tiling_search = true,    # autotune every block-level fusion: slower compiles
)))
```

XLA's overall compile effort is a separate setting, on `ExecutableBuildOptionsProto`:

```julia
const Effort = Reactant.Proto.xla.var"ExecutionOptions.EffortLevel"
Nitro(e; compile_options = (; xla_executable_build_options = (; optimization_level = Effort.EFFORT_O3)))
```

Reactant keeps a [persisted autotune cache](https://openxla.org/xla/persisted_autotuning) when
its persistent compile cache is on, and a fusion found there is not autotuned again. To compare
autotuning settings, clear it first with `Reactant.PersistentCompileCache.clear_compilation_cache!()`.

## Validation

`Nitro` checks the options before the setup sequence runs, so a mistake fails in seconds rather
than at the first compile:

- an unknown keyword is refused, and the error lists the accepted ones
- a `CompileOptions` field passed as a keyword is refused, with a pointer to the `CompileOptions` form
- an unknown field of `xla_debug_options`, `xla_executable_build_options` or `xla_compile_options`
  is refused
- `donated_args` other than `:auto` is refused: the parameters and the gradient accumulator are
  updated in place, and without donation every superseded buffer waits for the GC

See [`ReactantNitro.check_compile_options`](@ref).

## Recompilation

The options are part of each program's compile-cache key (see [Recompilation](recompilation.md)).
A handle with `raise = true` and one without get separate programs, and the default programs stay
cached alongside them. An empty `compile_options` leaves the key exactly as it was.

## Scope and limits

The options apply to every program the handle compiles: the gradient, optimizer, evaluation and
metric programs. There is no per-phase setting.

They do not reach:

- [Export](export.md): the bundle is compiled by the export backend, which does not receive the
  handle's options
- the [Kaimon](kaimon.md) tools: `nitro_train` and the others have no `compile_options` argument,
  so a Kaimon-launched run uses Reactant's defaults
