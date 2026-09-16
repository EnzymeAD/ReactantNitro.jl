---
name: reactantnitro-visualization
description: >
  Build visualizations for a model on ReactantNitro: what to draw that a metric cannot
  tell you, why the failures rather than a random sample, the two hooks (`visualize`,
  `save_figure`) and the `render` driver that owns the iteration, the slicing, and the
  naming, the four moments worth rendering at, and how the hooks keep the plotting
  package out of the training environment. Invoke when adding figures to a project,
  when a metric looks right and you do not believe it, when setting up a new model's
  data check, or when deciding what to render during a long run.
---

# Visualizing a model

**The framework ships the driver and no plotting dependency.** `render` owns the split
iteration, the per-sample slicing, the padding removal and the file naming; `visualize` and
`save_figure` are the two hooks you write. The package depends on no plotting library, exactly
as it depends on no logger, and it will not gain one without a design pass. This page is the
contract between the three, plus the judgement about what is worth drawing at all.

## What a figure is for, which is the part no framework can do for you

Three questions a figure answers and a metric does not.

**1. Is the data what I think it is?** Render the post-augmentation sample the model actually
receives, with its label drawn on it, **before you spend GPU hours**. Augmentation ranges that
looked reasonable in a config routinely produce examples a human could not label either, and no
loss curve says so. Render through the real loader: a figure built from a hand-assembled array
checks your plotting code rather than your pipeline.

**2. Is the model doing the thing, or scoring well by accident?** A metric aggregates away the
mechanism, so a model that learned a shortcut and one that learned the task report the same
number.

**3. What does it get WRONG?** The one most often skipped and the one that pays.

**Render the failures, not a random sample.** A random draw from a 95%-accurate model is nineteen
figures confirming what you knew and one you needed. You almost always have the per-sample number
already, because your metric computed it before reducing: keep it, sort by it, render the tail.

**A figure that can only ever confirm is decoration.** Ask of any panel you are about to add: what
would it look like if this were broken? If you cannot answer, it is not yet a diagnostic.

**Then render one and LOOK at it, before you trust any of it.** A figure can be arithmetically
correct and communicate nothing, and no test can tell you which you have: a suite can check that
the driver produced a file, and nothing can check that the file is legible. Both defects found the
first time this page's own recipe was used were of that kind, and both were invisible to a green
suite.

## The two hooks

```julia
visualize(e, outputs; <declared batch fields>) -> figure
save_figure(e, fig, stem) -> path
```

`visualize` is deliberately shaped like `metrics`: positional experiment, positional
outputs, batch fields by keyword, routed to exactly what the method declares. **It is called once
per SAMPLE, with the batch dimension already dropped.** You write a function of one example and
never write `[:, :, :, i]` nor reason about batch layout. A rank-1 field yields its **element**, so
a `Vector{String}` of case identifiers arrives as a `String` rather than as a zero-dimensional view,
which would interpolate into a title as `fill("case_B")`.

**`outputs` is `nothing` in data mode, and `nothing` is dispatchable.** One generic method
therefore covers both jobs, and two methods split them when the figures have little in common:

```julia
visualize(::MyExp, ::Nothing; img, y) = data_panel(img, y)
visualize(::MyExp, outputs; img, y)   = pred_panel(img, y, outputs)
```

The two methods may declare **different batch fields**, which is usually wanted since the data
figure needs less than the prediction figure. Shared axes are a shared plain function both call.
There is **no default method**: visualization being optional means "you need not call `render`",
never "`render` may quietly do nothing", so a missing method is an error naming the experiment type
and which of the two modes was missing.

`save_figure` writes what `visualize` returned and **returns the path actually written**. `e` is
in the signature so that the method is a specialization on a type you own rather than type piracy
(two model packages loaded in one session would otherwise overwrite each other silently), and
`stem` carries **no extension**, because the format belongs to your backend, not to the framework:

```julia
function ReactantNitro.save_figure(::MyExp, fig::Makie.Figure, stem::AbstractString)
    path = stem * ".png"
    Makie.save(path, fig; px_per_unit = 2)
    return path
end
```

The framework never inspects what `visualize` returned. It hands the value to `save_figure` and
reports the path that came back, which is what lets a figure be a Makie figure, an image array,
an SVG string, a video, or a text dump, and what keeps a plotting package out of the training
environment entirely.

## The driver

```julia
render(nitro; split = :val, batches = 1, predictions = false, out_dir, tag) -> Vector{String}
render(nitro, batch; predictions = false, out_dir, tag = "") -> Vector{String}
```

Three entry points, one driver, none of which needs `train!`:

```julia
render(Nitro(e); split = :val)                                        # the data gate
render(Nitro(e; checkpoint = "runs/x/best.jld2"); predictions = true) # from a checkpoint
render(nitro, batch; predictions = true)                              # a batch in hand
```

The default output directory is `run_dir(nitro)/viz`, tagged by split. Filenames are the sample's
index within the render call, zero-padded, because that is the only thing true of every problem; a
case identifier belongs in the figure's title, where `visualize` can put it by declaring the field,
which costs nothing since a field no other hook declares is never transferred to device.

## The four moments

| When | How you get it | What it answers |
| --- | --- | --- |
| **Data gate**, before any training | `Nitro(e)`, with `predictions = false` | question 1 |
| **During a run**, every k epochs | a phase monitor | whether a failure mode is moving |
| **Post-hoc**, on a checkpoint | `Nitro(e; checkpoint = path)` | questions 2 and 3 |
| **Inference**, on unseen data | `Nitro(e; checkpoint = path, data = (; test = loader))` | what shipping looks like |

**Three of the four are free**, and they are free because of a property worth naming: `Nitro(e)`
runs the setup sequence and nothing else, so parameters, data, keyword routing and the compiled
`forward` all exist with no `train!` anywhere in the process. A rendering script is not a special
mode. It is an ordinary handle you never trained.

## Five things the driver does so you do not have to

- **`predictions = false` is the default, which is the data gate.** `predict` compiles the
  eval-mode `forward` on its first call, a phase the framework itself describes as taking hundreds
  of seconds. Question 1 never looks at the model's output, so leaving predictions on there buys a
  long compile and an overlay drawn from untrained weights.
- **You say how many BATCHES, not how many samples.** There is no sample cap and so no interaction
  between a cap and a short final batch. The cost is worth knowing: at `batch_size = 64`,
  `batches = 1` writes 64 figures, and the batch form takes whatever batch you hand it, so a
  narrower one is the way to render fewer.
- **`predict` returns HOST arrays**, already sliced to the real sample count, with the framework's
  residency assertion passed on the way out. Do not add a defensive conversion, and do not reach
  into the device path yourself. A device array reaching a plotting call fails deep inside the
  plotting library with a message naming neither your figure nor the array; see
  `reactantnitro-device-boundary` for why that class of mistake is invisible on CPU.
- **Padding is already gone.** `predict` pads a short final batch, runs the one compiled program,
  and slices every output leaf back to the real sample count. Iterating the raw batch width
  yourself instead would render duplicated filler rows as though they were data.
- **It slices to one sample before calling you.** Everything downstream is then a function of one
  example, and the same `visualize` method serves the data gate, the training monitor and
  inference unchanged. The slice is a copy rather than a view, so a figure can never hold a view
  onto a buffer the framework may free.

## One router fact worth knowing

**A batch field declared only by `visualize` is never transferred to device.** The framework
resolves the visualization hooks' routing separately from the training routing, so a bookkeeping
field the data figure needs, a `Vector{String}` of case identifiers, costs nothing: it rides in
the host batch, is routed to `visualize`, and no training batch ever carries it across the device
boundary. A field is transferred only when a hook that runs in the training loop declares it. The
batch-last rule is asserted on the way in, so a model whose batch dimension is not last gets the
wrong axis sliced per sample, with no error.

## Keep the plotting package out of the training environment

**A plotting library is a heavy dependency that training never uses**, and paying its load and
compile cost on every run of a long job buys nothing. Two conventional answers, and they compose:
put your figure methods behind a **package extension** on a weak dependency, so they exist when a
user has loaded the plotting package and cost nothing otherwise; and give rendering **its own
project environment**, which adds the plotting package and shares the model package. Rendering is
data-path only, so that environment needs no accelerator.

The framework takes no position on which plotting package you use, exactly as it takes none on
which logger or which schedule library. Nothing here dispatches on a figure.

## During a run

Use the phase registry. **The transition OUT of `EvalStepping` is the moment to render**: the
weights are current, the evaluation has just finished, and `info` carries `metrics` only on that
transition.

```julia
register_phase_monitor!(nitro) do phase, step, epoch, info
    haskey(info, :metrics) && epoch !== nothing && epoch % 5 == 0 || return nothing
    render(info.nitro; split = :val, batches = 1, out_dir = "viz/epoch-$(lpad(epoch, 4, '0'))")
    return nothing
end
```

**A monitor watches and never steers.** Keep it cheap: one batch of figures every fifth epoch is
noise against an epoch, rendering a split every epoch is not. The batch form of `render` exists
for exactly this: if one batch's worth of figures is too many, hand it a narrower batch.

**A monitor that throws does not take the run with it, and that is the hazard rather than the
comfort.** The framework catches it and warns that the run is unaffected, then goes quiet: the
warning fires **once per monitor per run**, so a figure call that fails on the first sample gives
you one line early in a multi-hour log and nothing afterwards. The run finishes clean and the
directory is empty. Wrap the body in a `try` and log the failure **yourself, every time**, or
render one epoch's worth outside a run first and only then register the monitor.

## Determinism is your loader's, not the framework's

The property you want is that **validation renders are diffable across data-prep changes while
training renders vary**, so a changed figure means a changed pipeline rather than a changed draw.

The framework cannot give you this: it never looks inside your data and ships no batching or
shuffling, so ordering and augmentation seeding are entirely your loader's. Do not shuffle the
validation split, and seed its augmentation from something stable about the sample rather than
from entropy. Taking the first `batches` from the head is then reproducible, and that is the whole
mechanism. The `PrefetchIterator` that wraps the train split is a passthrough on `iterate`, so it
perturbs nothing here (see `reactantnitro-experiment` for what it does change).

## Four traps

- **Two series drawn on top of each other.** Most plotting libraries draw the second call OVER the
  first at the same positions, so wherever the two agree, one of them is invisible and the panel
  shows a single series under a legend claiming two. **The case where they agree is usually the
  case you most need to see**: a model that has not learned anything yet returns its input, and
  that is exactly when the comparison silently collapses to one bar. Dodge them, or offset one, so
  that equal reads as equal. This is not hypothetical; it is the second defect this page's own
  recipe produced.
- **Rendering the aggregate instead of the sample.** A grid of thumbnails answers nothing about a
  model whose errors are a few pixels wide. Zoom to where the error lives and put the scale on the
  figure, or you are looking at a picture of your dataset rather than at your model.
- **A title that promises more than the panel contains.** The data-gate mode has no predictions, so
  a panel labelled "input versus prediction" there is claiming something it is not showing. That is
  the same failure the figure exists to catch in the model, and it is the first defect the recipe
  produced.
- **Reading a figure as evidence about device behaviour.** Rendering runs on host arrays after a
  transfer, so it is indifferent to the accelerator and says nothing about it.
