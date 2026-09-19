# ReactantNitroMakieExt.jl
#
# `plot(history(nitro))` and `plot(nitro)`, for whichever Makie backend the session has loaded.
#
# ── Why the weak dependency is Makie and not a backend ───────────────────────────────
#
# Every backend loads Makie: CairoMakie for a file or a notebook image, GLMakie for a window,
# WGLMakie for Pluto and Jupyter. An extension on Makie therefore activates with any of them and
# takes no position on which, and a training environment that loads none of them carries no
# plotting code at all. The same figure is what a REPL saves to PNG and what a notebook shows
# inline with hover and zoom, which is the bridge to those environments: there is one object,
# and the environment decides how it is displayed.
#
# ── What is decided here, and what is not ────────────────────────────────────────────
#
# Nothing about the data. `history_series` in the core chose the curves, the axis, and the marked
# point, and is tested without this package. This file turns its panels into a `SpecApi` layout:
# one `Axis` per curve, arranged in a grid, under a title. The `SpecApi` route rather than a
# recipe because a recipe draws into ONE axis and the default here is one axis per metric, and
# because a spec is what lets `plot(fig[2, 1], h)` place the whole grid inside a caller's figure.
#
# ── The one mark ─────────────────────────────────────────────────────────────────────
#
# The best epoch is drawn as a star and named in that panel's title. A shape and a word, not a
# colour alone, so it reads in print, on a projector, and under any theme: colours here are
# Makie's own, from whatever theme the session set, and this extension sets none.
module ReactantNitroMakieExt

import ReactantNitro
using ReactantNitro: MetricHistory, Nitro, history, history_series
import Makie

const S = Makie.SpecApi

# The keywords `convert_arguments` receives from `plot(h; x, metrics)` rather than the backend.
Makie.used_attributes(::MetricHistory) = (:x, :metrics)

function Makie.convert_arguments(
        ::Type{<:Makie.AbstractPlot}, h::MetricHistory; x = :epoch, metrics = nothing
    )
    s = history_series(h; x, metrics)
    n = length(s.panels)
    # One curve is one axis. Up to four sit two abreast; more go three abreast, which keeps a
    # panel wide enough that its title still fits on one line at Makie's default figure size.
    ncols = n == 1 ? 1 : n <= 4 ? 2 : 3
    content = Pair{Tuple{Int, UnitRange{Int}}, Union{Makie.BlockSpec, Makie.GridLayoutSpec}}[]
    push!(
        content,
        (1, 1:ncols) => S.Label(;
            text = s.title, font = :bold, fontsize = 18, halign = :left, tellwidth = false
        ),
    )
    # The theme's first series colour, taken explicitly: the line would get it by cycling, but a
    # stroke does not cycle, and the star's stroke must be the line's colour so it reads as a
    # point ON the curve rather than a second series.
    colour = Makie.to_color(first(Makie.to_value(Makie.theme(:palette)[:color])))
    for (i, p) in enumerate(s.panels)
        r, c = fldmod1(i, ncols)
        plots = Makie.PlotSpec[S.Lines(p.x, p.y; linewidth = 2, color = colour)]
        if p.best !== nothing
            push!(
                plots,
                S.Scatter(
                    [p.best[1]], [p.best[2]];
                    marker = :star5, markersize = 20, color = :white,
                    strokewidth = 2, strokecolor = colour,
                ),
            )
        end
        # The best epoch is the extreme of its curve, so on a marked panel the automatic limits
        # get more air than Makie's default 5%, or a small panel clips half the star.
        margin = p.best === nothing ? (0.05f0, 0.05f0) : (0.12f0, 0.12f0)
        push!(
            content,
            (r + 1, c:c) => S.Axis(;
                plots, xlabel = s.xlabel, title = p.subtitle, titlealign = :left,
                titlesize = 14, yautolimitmargin = margin,
            ),
        )
    end
    return S.GridLayout(content)
end

# A handle plots as its history, so `plot(nitro)` after `train!` is the curve of the run.
Makie.plot(n::Nitro; kw...) = Makie.plot(history(n); kw...)
Makie.plot(fig::Union{Makie.Figure, Makie.GridPosition}, n::Nitro; kw...) =
    Makie.plot(fig, history(n); kw...)

end
