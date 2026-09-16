# Visualize.jl
#
# Two hook declarations with NO default method, and the driver that owns every step of a figure
# except the drawing. Almost nothing here is new: `predict` already runs the model, pads a short
# final batch, reads back to host and slices to the real sample count, and the batch-routing
# machinery already answers "which batch fields does this hook want". This file is the glue.
#
# THE FRAMEWORK NEVER INSPECTS WHAT `visualize` RETURNED. It hands the value to `save_figure` and
# reports the path that came back, which is what lets a figure be a Makie figure, an image array, an
# SVG string, a video, or a text dump, and what keeps a plotting package out of the training
# environment entirely. The logger contract is the precedent and was copied closely.
#
# THE VIZ ROUTERS ARE RESOLVED HERE AND ARE NEVER MERGED INTO `nitro.routing`. This is the
# one rule in this file that prevents a defect rather than explaining a choice. `routed_fields`
# drives `to_device_batch`, `validate_batch`, and `batch_size_of`, so a bookkeeping field declared
# only by `visualize` would be transferred to device on EVERY TRAINING BATCH for the rest of the run,
# breaking routing rule 1's promise on the hot path for a hook that runs a handful of times. The
# obvious implementation, appending two entries to `resolve_routing`'s spec tuple, is exactly the
# wrong one, and `validate_batch` would not catch it first: a `Vector{String}` is an array with a
# concrete element type, so the failure lands later, at `place_batch`, naming nothing useful.

# ── The contract ────────────────────────────────────────────────────────────────────

"""
    visualize(e, outputs; <declared batch fields>) -> figure

Draw **one sample**. Deliberately shaped like [`metrics`](@ref): positional experiment, positional
outputs, batch fields by keyword, routed to exactly what the method declares. A reader who
knows `metrics` knows this.

**It is called once per SAMPLE, with the batch dimension already dropped.** That is the whole
boilerplate reduction: you write a function of one example and never write `[:, :, :, i]` nor reason
about batch layout. A rank-1 field yields its **element**, so a `Vector{String}` of case identifiers
arrives as a `String` rather than as a zero-dimensional view, which would interpolate into a title as
`fill("case_B")`.

**`outputs` is `nothing` in data mode, and `nothing` is dispatchable.** One generic method therefore
covers both jobs, and two methods split them when the figures have little in common:

```julia
visualize(::MyExp, ::Nothing; img, y) = data_panel(img, y)
visualize(::MyExp, outputs; img, y)   = pred_panel(img, y, outputs)
```

The two methods may declare **different batch fields**, which is usually wanted since the data figure
needs less than the prediction figure. Shared axes are a shared plain function both call.

**There is no default method.** Visualization being optional means "you need not call [`render`](@ref)",
never "`render` may quietly do nothing", so a missing method is an error naming the experiment type
and which of the two modes was missing.

The return value is whatever [`save_figure`](@ref) knows how to write. The framework does not look
at it.
"""
function visualize end

"""
    save_figure(e, fig, stem) -> path

Write what [`visualize`](@ref) returned, and **return the path actually written**.

**`e` is in the signature so that writing this method is not type piracy.** Without it the one method
every model must write is `save_figure(::Makie.Figure, stem)`, which owns neither the function nor
the type; two model packages defining it and loaded in one session overwrite each other silently.
With `e` a model package owns `MyExp`, the specialization is ordinary, and two models can save
differently. It is also the only place per-experiment format and resolution can live, and it keeps
this hook consistent with every other one in the framework, all of which take `e` first.

**`stem` carries no extension**, because the framework has no opinion about file formats and a
framework handing over a path ending in `.png` has one: that suffix is what every plotting backend
dispatches format on.

**The return value is what [`render`](@ref) reports.** A backend that writes a video, three files, or
a directory of frames says so, rather than having the framework report a single path it guessed at.

```julia
function ReactantNitro.save_figure(::MyExp, fig::Makie.Figure, stem::AbstractString)
    path = stem * ".png"
    Makie.save(path, fig; px_per_unit = 2)
    return path
end
```
"""
function save_figure end

# ── The driver ──────────────────────────────────────────────────────────────────────

"""
    render(nitro; split = :val, batches = 1, predictions = false, out_dir, tag) -> Vector{String}
    render(nitro, batch; predictions = false, out_dir, tag = "") -> Vector{String}

Render a few examples from the real pipeline, and return the paths written.

**`predictions` defaults to `false`**, which is the data gate: no forward pass, so no compile. That
matters because `Nitro(e)` compiles nothing and the first `forward` costs an
[`EvalCompiling`](@ref) phase, which is slow by design. Paying it to draw figures from untrained
weights is the one render nobody wants.

**You say how many BATCHES, not how many samples.** There is no sample cap and so no interaction
between a cap and a short final batch. The cost is worth knowing: at `batch_size = 64`, `batches = 1`
writes 64 figures. Pass [`render`](@ref)'s batch form a narrower batch if that is too many.

**It pulls batches, not epochs**, from the head of the split.

Three entry points, one driver, none of which needs `train!`:

```julia
render(Nitro(e); split = :val)                                        # the data gate
render(Nitro(e; checkpoint = "runs/x/best.jld2"); predictions = true) # from a checkpoint
render(nitro, batch; predictions = true)                              # a batch in hand
```

Ordering is the loader's, not the framework's: this takes the head of the split in the source's own
order, and [`PrefetchIterator`](@ref)'s `iterate` is a passthrough so nothing here perturbs it. If
you want validation renders that diff cleanly across data-prep changes, your validation loader must
be deterministic; the framework cannot make it so.
"""
function render(
        nitro::Nitro;
        split::Symbol = :val,
        batches::Integer = 1,
        predictions::Bool = false,
        out_dir::AbstractString = joinpath(run_dir(nitro), "viz"),
        tag::AbstractString = String(split),
    )
    batches >= 1 || error("ReactantNitro: `render` was asked for $batches batches; it must be at \
                           least 1.")
    data = nitro.data
    data === nothing && error(
        """
        ReactantNitro: this `Nitro` has no data collection, so `render` has no `$split` split to
        pull from. Build one with a split (`Nitro(e)`), supply it directly
        (`Nitro(e; data = (; val = loader))`), or render a batch you already have with
        `render(nitro, batch)`."""
    )
    haskey(data, split) || error(
        """
        ReactantNitro: there is no `$split` split. The data collection has $(keys(data)).
        `render` names the split it wants and errors on one that does not exist, exactly as
        `evaluate` does, which is why the data collection is a NamedTuple rather than a
        positional tuple."""
    )
    # `with_repl` wraps only the loop, not the argument checks above: an error raised before
    # any work started is not a phase transition, and publishing `Repl` for it would be
    # announcing that a run went idle when it never began. `spawn = false`: rendering hooks are
    # user code that may own a display backend (GLMakie) needing the main thread, and this is
    # not the long XLA loop the worker spawn exists for.
    return with_repl(nitro; spawn = false) do
        paths = String[]
        for batch in Iterators.take(getproperty(data, split), batches)
            append!(paths, _render_batch(nitro, batch, length(paths); predictions, out_dir, tag))
        end
        return paths
    end
end

function render(
        nitro::Nitro, batch::NamedTuple;
        predictions::Bool = false,
        out_dir::AbstractString = joinpath(run_dir(nitro), "viz"),
        tag::AbstractString = "",
    )
    return with_repl(nitro; spawn = false) do
        _render_batch(nitro, batch, 0; predictions, out_dir, tag)
    end
end

# One batch, `offset` figures already written, so filenames run continuously across the batches the
# split form pulls rather than restarting at 1 in each.
function _render_batch(
        nitro::Nitro, batch::NamedTuple, offset::Integer;
        predictions::Bool, out_dir::AbstractString, tag::AbstractString
    )
    e = nitro.e
    # HOST ONLY, UNCONDITIONALLY, and there is no residency knob. `predict` returns host
    # arrays already sliced to the real sample count and asserts that on the way out, so both halves
    # of what reaches the hook are host by construction: the outputs through `predict`, and the batch
    # fields because they are the loader's own host batch and are never transferred here at all.
    outputs = predictions ? predict(nitro, batch) : nothing
    router = viz_router(e, batch, outputs)
    # The REAL sample count for this batch, not the compiled width. They differ on an eval split's
    # short final batch, and `nitro.batch_size` would over-count there and render padding.
    n = batch_size_of(batch, (; visualize = router))
    # A model whose batch dimension is not last would otherwise have the wrong axis sliced per sample
    # and would get the wrong data in every figure with no error. `predict` only checks this when it
    # actually padded, so the check belongs here too.
    outputs === nothing || check_output_batch_dim(outputs, n)
    mkpath(out_dir)
    fields = router(batch)
    paths = String[]
    for i in 1:n
        o = outputs === nothing ? nothing : _sample_tree(outputs, i)
        fig = _call_visualize(e, o, _sample_tree(fields, i), batch, router)
        isempty(paths) && check_save_figure(e, fig)
        stem = joinpath(out_dir, _stem(tag, offset + i))
        push!(paths, _checked_path(save_figure(e, fig, stem), e, fig))
    end
    return paths
end

"""
    ReactantNitro.viz_router(e, batch, outputs) -> Router

Per-mode routing for `visualize`, resolved **here and not merged into `nitro.routing`**; see this
file's header for what merging it would cost.

Resolved against `typeof(e)` and **not** `typeof(compile_view(e))`. Setup resolves against the
stripped type because the trace site calls with it; `visualize` is host-only and is called with
`e`, so inheriting that rule here would be copying a rule whose reason does not apply.

The mode is carried in the second argument type: `Nothing` selects a `::Nothing` method when one
exists and falls through to a generic method when it does not, and `Any` selects the generic method
only. That is what makes an experiment defining only the data-mode method a **named error** under
`predictions = true` rather than a confusing `MethodError`, and it costs one `hasmethod` call.
"""
function viz_router(e, batch::NamedTuple, outputs)
    O = outputs === nothing ? Nothing : Any
    decl = declared(visualize, Tuple{typeof(e), O})
    decl === nothing && _no_visualize_error(e, outputs)
    return Router{route_keys(decl, keys(batch), :visualize)}()
end

# `call_hook`'s error with the FULL batch rather than the routed subset. `call_hook` itself cannot be
# reused verbatim: it applies the router to what it is given, and what is passed here is one sample's
# fields, so its message would report the routed keys as though they were the batch's own.
function _call_visualize(e, outputs, sample_fields::NamedTuple, batch::NamedTuple, router::Router)
    try
        return visualize(e, outputs; sample_fields...)
    catch err
        err isa UndefKeywordError || rethrow()
        _missing_kw_error(:visualize, err, batch, router)
    end
end

"""
    ReactantNitro.check_save_figure(e, fig) -> nothing

The missing-method rule for the backend verb, checked once per `render` call against the figure
the experiment actually produced.

A bare `MethodError` would be survivable here, but it would name a `stem::String` the user never
wrote and would not say that this is a hook they are expected to implement. The check costs one
`hasmethod` per call, not per sample.
"""
function check_save_figure(e, fig)
    hasmethod(save_figure, Tuple{typeof(e), typeof(fig), String}) && return nothing
    return error(
        """
        ReactantNitro: `visualize` returned a `$(typeof(fig))` for `$(typeof(e))`, and there is no
        `save_figure` method to write it. Define

            ReactantNitro.save_figure(::$(nameof(typeof(e))), fig::$(nameof(typeof(fig))), stem::AbstractString)

        It receives a stem with NO extension, so it chooses the format, and it must RETURN the path
        it wrote, which is what `render` reports. The experiment is in the signature so that this
        method is a specialization on a type you own rather than type piracy."""
    )
end

# `save_figure` returning nothing is the mistake a `-> nothing` contract would invite, and the paths
# `render` returns are its whole product, so a backend that forgets the return is caught here rather
# than handing back a vector of `nothing`.
_checked_path(path::AbstractString, e, fig) = String(path)
@noinline _checked_path(path, e, fig) = error(
    """
    ReactantNitro: `save_figure(::$(nameof(typeof(e))), ::$(nameof(typeof(fig))), stem)` returned a
    `$(typeof(path))`, and it must return the path it wrote, as a string.
    `render` returns those paths and has no other way to know what a backend produced: a stem carries
    no extension, and a backend may legitimately write a video or a directory of frames."""
)

@noinline function _no_visualize_error(e, outputs)
    mode = outputs === nothing ? "DATA mode (`outputs === nothing`)" : "PREDICTION mode"
    sig = outputs === nothing ? "::Nothing" : "outputs"
    hint = outputs === nothing ?
        "A generic method covering `outputs::Any` serves BOTH modes, so one method is enough." :
        "Defining only the `::Nothing` method is what produces this under `predictions = true`; \
         add a generic method, or call `render` without predictions."
    return error(
        """
        ReactantNitro: no `visualize` method for `$(typeof(e))` in $mode. Define

            ReactantNitro.visualize(e::$(nameof(typeof(e))), $sig; <the batch fields you want>)

        It is called once per SAMPLE with the batch dimension already dropped, and it is routed the
        batch fields it declares and no others.
        $hint"""
    )
end

# ── Per-sample slicing ──────────────────────────────────────────────────────────────

"""
    ReactantNitro._sample_tree(tree, i) -> tree

Take sample `i` out of every array leaf, **dropping the batch dimension**, which is what lets
`visualize` be a function of one example.

Built on `selectdim` with a scalar index rather than on [`slice_last`](@ref) plus a `dropdims`: a
scalar index drops the dimension directly, and the rank-1 case wants the ELEMENT rather than a
zero-dimensional array, which `dropdims` cannot give. For a `Vector` the last dimension IS the batch
dimension, so plain indexing is both simpler and correct there.

A `copy` rather than a view, for the reason [`slice_last`](@ref) gives: a view is a different type
from the array it came from, and handing user code a view onto a buffer the framework may free is a
sharp edge for no gain at these sizes.
"""
_sample_tree(tree, i::Integer) = _map_array_leaves(x -> _sample_leaf(x, i), tree)

_sample_leaf(x::AbstractVector, i::Integer) = x[i]
_sample_leaf(x::AbstractArray, i::Integer) =
    ndims(x) == 0 ? x : copy(selectdim(x, ndims(x), i))
_sample_leaf(x, i::Integer) = x

# Filenames are the sample's index, zero-padded, because that is the only thing true of every
# problem. A case identifier belongs in the figure's title, where `visualize` can put it by declaring
# the field, which costs nothing since a field no other hook declares is never transferred to device.
_stem(tag::AbstractString, idx::Integer) =
    isempty(tag) ? lpad(idx, 3, '0') : string(tag, "_", lpad(idx, 3, '0'))
