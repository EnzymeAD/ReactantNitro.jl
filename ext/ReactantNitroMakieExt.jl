# ReactantNitroMakieExt.jl
#
# `plot(history(nitro))` and `plot(nitro)`. The weak dependency is Makie, not a backend, so
# CairoMakie, GLMakie and WGLMakie all activate it. `history_series` in the core decides what is
# drawn; this file lays it out with SpecApi rather than a recipe, so the figure can be one axis
# per curve and embed with `plot(fig[2, 1], h)`. Colours are the session's theme; the best epoch
# is a star named in the title, not a colour.
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
    # Up to four panels two abreast, more three abreast, so a title still fits its panel.
    ncols = n == 1 ? 1 : n <= 4 ? 2 : 3
    content = Pair{Tuple{Int, UnitRange{Int}}, Union{Makie.BlockSpec, Makie.GridLayoutSpec}}[]
    push!(
        content,
        (1, 1:ncols) => S.Label(;
            text = s.title, font = :bold, fontsize = 18, halign = :left, tellwidth = false
        ),
    )
    # Explicit because a stroke does not cycle, and the star's stroke must match the line.
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
        # The star sits at the curve's extreme; Makie's default 5% margin clips half of it.
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

Makie.plot(n::Nitro; kw...) = Makie.plot(history(n); kw...)
Makie.plot(fig::Union{Makie.Figure, Makie.GridPosition}, n::Nitro; kw...) =
    Makie.plot(fig, history(n); kw...)

end
