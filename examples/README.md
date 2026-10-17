# Examples

One directory per example. Each has its own `Project.toml`, which takes `ReactantNitro` from the
repository root through `[sources]`, so the example runs against the checkout it sits in. Run one
from the repository root:

```
julia --project=examples/<name> -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/<name> examples/<name>/<entry>.jl
```

| Example | What it shows | Entry |
|---|---|---|
| `mnist/` | The [tutorial](../docs/src/tutorial.md)'s experiment as one script: an MLP on MNIST with two parameter groups, gradient accumulation, a one-cycle schedule, clipping, top-K checkpointing, early stopping and the default JSON logger. | `mnist_tutorial.jl` |

A notebook example is also a directory, with the notebook as its entry. A Pluto notebook carries
its own environment inside its `.jl` file, so its directory has no `Project.toml`; a Jupyter
notebook keeps one beside its `.ipynb`.

Every example reads two environment variables: `NITRO_EXAMPLE_EPOCHS` sets the epoch count, so
`NITRO_EXAMPLE_EPOCHS=1` is a smoke test, and `NITRO_EXAMPLE_BACKEND=cuda` runs it on a GPU (the
default is `cpu`).
