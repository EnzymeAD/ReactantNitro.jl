# Visualize.jl
#
# Two hook declarations with no default method, and the driver that owns every step of a figure
# except the drawing: `predict` runs the model and slices to the real sample count, and the routing
# machinery answers which batch fields the hook wants. The framework never inspects what
# `visualize` returned; it hands the value to `save_figure`, which keeps plotting packages out of
# the training environment.
#
# The viz routers are resolved here and never merged into `nitro.routing`: `routed_fields` drives
# `to_device_batch`, so a bookkeeping field declared only by `visualize` would be transferred on
# every training batch, and `validate_batch` would not catch a `Vector{String}` first.

# ── The contract ────────────────────────────────────────────────────────────────────

"""
    visualize(e, outputs; <declared batch fields>) -> figure

Draw one sample. Shaped like [`metrics`](@ref): positional experiment, positional outputs, batch
fields by keyword, routed to what the method declares. Called once per SAMPLE with the batch
dimension already dropped; a rank-1 field yields its element, so a `Vector{String}` of case ids
arrives as a `String`. `outputs` is `nothing` in data mode, and `nothing` is dispatchable, so one
generic method covers both modes or two methods split them:

```julia
visualize(::MyExp, ::Nothing; img, y) = data_panel(img, y)
visualize(::MyExp, outputs; img, y)   = pred_panel(img, y, outputs)
```

There is no default method: a missing one is an error naming the experiment and the mode. The
return value is whatever [`save_figure`](@ref) knows how to write.
"""
function visualize end

"""
    save_figure(e, fig, stem) -> path

Write what [`visualize`](@ref) returned and return the path actually written. `e` is in the
signature so the method a model package writes is a specialization on a type it owns rather than
type piracy on `Makie.Figure`. `stem` carries no extension, since the framework has no opinion
about formats. The return value is what [`render`](@ref) reports, so a backend writing a video or a
directory of frames says so.

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

Render a few examples from the real pipeline, and return the paths written. `predictions` defaults
to `false`, the data gate: no forward pass, so no compile. `batches` counts batches, not samples,
so at `batch_size = 64` one batch writes 64 figures; pass the batch form a narrower batch if that
is too many. Batches are taken from the head of the split in the loader's own order, so validation
renders diff cleanly only if the validation loader is deterministic.

```julia
render(Nitro(e); split = :val)                                        # the data gate
render(Nitro(e; checkpoint = "runs/x/best.jld2"); predictions = true) # from a checkpoint
render(nitro, batch; predictions = true)                              # a batch in hand
```
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
    # `with_repl` wraps only the loop; an error before any work started is not a phase transition.
    # `spawn = false`: a rendering hook may own a display backend that needs the main thread.
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
    # Host only: `predict` returns host arrays sliced to the real count, and the batch fields are
    # the loader's own host batch.
    outputs = predictions ? predict(nitro, batch) : nothing
    router = viz_router(e, batch, outputs)
    # The real sample count for this batch, not the compiled width.
    n = batch_size_of(batch, (; visualize = router))
    # `predict` checks the batch axis only when it padded, so the check belongs here too.
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

Per-mode routing for `visualize`, resolved here and never merged into `nitro.routing`, and against
`typeof(e)` rather than the compile view, since the hook is host-only. The mode is carried in the
second argument type: `Nothing` selects a `::Nothing` method or falls through to a generic one,
and `Any` selects the generic method only, so an experiment defining only the data-mode method
gets a named error under `predictions = true`.
"""
function viz_router(e, batch::NamedTuple, outputs)
    O = outputs === nothing ? Nothing : Any
    decl = declared(visualize, Tuple{typeof(e), O})
    decl === nothing && _no_visualize_error(e, outputs)
    return Router{route_keys(decl, keys(batch), :visualize)}()
end

# `call_hook`'s error with the full batch rather than the routed subset.
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

The missing-method check for the backend verb, once per `render` call against the figure the
experiment actually produced, so the error says this is a hook to implement.
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

# The paths `render` returns are its whole product, so a backend that forgets the return is caught
# here rather than handing back a vector of `nothing`.
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

Take sample `i` out of every array leaf, dropping the batch dimension. A scalar `selectdim` drops
the dimension directly, and a `Vector` yields its element rather than a zero-dimensional array. A
`copy` rather than a view, so user code never holds a view onto a buffer the framework may free.
"""
_sample_tree(tree, i::Integer) = _map_array_leaves(x -> _sample_leaf(x, i), tree)

_sample_leaf(x::AbstractVector, i::Integer) = x[i]
_sample_leaf(x::AbstractArray, i::Integer) =
    ndims(x) == 0 ? x : copy(selectdim(x, ndims(x), i))
_sample_leaf(x, i::Integer) = x

# Filenames are the sample's zero-padded index; a case identifier belongs in the figure's title.
_stem(tag::AbstractString, idx::Integer) =
    isempty(tag) ? lpad(idx, 3, '0') : string(tag, "_", lpad(idx, 3, '0'))
