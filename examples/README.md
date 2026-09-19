# Examples

One directory per example. Each is self contained: its own `Project.toml`, with `ReactantNitro`
taken from the repository root through `[sources]` so the example runs against the tree it sits
in, and its own entry point named in its header. Run one from the repository root:

```
julia --project=examples/<name> -e 'using Pkg; Pkg.instantiate()'
julia --project=examples/<name> examples/<name>/<entry>.jl
```

| Example | What it shows | Entry |
|---|---|---|
| `mnist/` | The [tutorial](../docs/src/tutorial.md) as one runnable script: an MLP on MNIST with gradient accumulation, a schedule, a checkpointer and a logger. | `mnist_tutorial.jl` |

A notebook example is a directory too, with the notebook as its entry: a Pluto notebook is a
single `.jl` file that carries its own environment, so its directory needs no `Project.toml`; a
Jupyter notebook keeps one beside its `.ipynb`.

Every example honours `NITRO_EXAMPLE_EPOCHS`, so `NITRO_EXAMPLE_EPOCHS=1` is a smoke test, and
`NITRO_EXAMPLE_BACKEND=cuda` puts it on a GPU.
