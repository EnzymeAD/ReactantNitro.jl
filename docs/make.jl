# The Documenter build, on the default Documenter HTML theme with two plugins:
# DocumenterLandingPage renders the VitePress-style landing page (hero + emoji
# capability tiles) from the YAML frontmatter in docs/src/index.md, and
# DocumenterCodeBlocks enhances the code blocks (line numbers, reference
# links, hover tooltips, JuliaSyntax highlighting). `doctest = false` on
# purpose: the guide pages carry illustrative Julia rather than executable
# examples, because executing any of them would compile XLA programs and turn
# the docs build into a GPU-hours exercise. The API page reads docstrings only,
# which is the point of @autodocs.
#
# The landing page frontmatter (docs/src/index.md) is exactly the VitePress
# home layout the docs previously carried under DocumenterVitepress; the
# plugin replaces the block with rendered HTML at build time.
using Documenter
using DocumenterCodeBlocks
using DocumenterLandingPage
using Dates
using ReactantNitro

makedocs(;
    sitename = "ReactantNitro.jl",
    modules = [ReactantNitro],
    authors = "Carroll Vance <cvance@medicalmetrics.com>",
    doctest = false,
    # The API page deliberately documents the exported surface plus the internal
    # helpers the docstrings reference; the rest of the module's internal
    # docstrings are not in the manual, which would otherwise fail the build.
    # Unresolved @refs are errors by default in this Documenter, and there are
    # none left by the time this is committed; the category below is the
    # safety valve while a docstring is being edited.
    warnonly = [:missing_docs],
    repo = Documenter.Remotes.GitHub("EnzymeAD", "ReactantNitro.jl"),
    format = Documenter.HTML(
        edit_link = "main",
        canonical = "https://enzymead.github.io/ReactantNitro.jl/",
        # The brand theme: overrides the plugin's default gradient defaults
        # (blue/purple) with the logo's fire tones (burnt brown, dark orange,
        # gold), per theme; the override contract is documented in the
        # plugin's styling page.
        assets = ["assets/brand.css", "assets/brand.js"],
        # docs/Project.toml is an environment, not a package, so Documenter
        # cannot infer a version for the search inventory; set it explicitly
        # to match ReactantNitro.
        inventory_version = "0.1.0",
        # The single API page collects the whole exported surface (plus the
        # internals its docstrings reference), which renders past the default
        # 200 KiB size_threshold; exempt that one page and keep the guard for
        # everything else.
        size_threshold_ignore = ["api.md"],
        footer = "Made with [Documenter.jl](https://documenter.juliadocs.org/stable/), [DocumenterLandingPage.jl](https://github.com/csvance/DocumenterLandingPage.jl), and [DocumenterCodeBlocks.jl](https://github.com/fredrikekre/DocumenterCodeBlocks.jl)<br>© Copyright $(Dates.year(Dates.today())).",
    ),
    plugins = [
        LandingPage(),
        CodeBlocks(),
    ],
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "Binding cheat sheet" => "binding.md",
        "Pitfalls" => "pitfalls.md",
        "Experiments" => "experiments.md",
        "Recompilation" => "recompilation.md",
        "Metrics" => "metrics.md",
        "Optimization" => "optimization.md",
        "Schedules" => "schedules.md",
        "Manual training" => "manual.md",
        "Logging" => "logging.md",
        "Export" => "export.md",
        "Kaimon" => "kaimon.md",
        "API" => "api.md",
    ],
)

# PRs get a preview URL via `push_preview = true`; pushes to main (or a tag)
# deploy the live site. This is Documenter's own deployer, unlike the
# VitePress-specific DocumenterVitepress.deploydocs the docs previously used.
Documenter.deploydocs(;
    repo = "github.com/EnzymeAD/ReactantNitro.jl.git",
    push_preview = true,
    devbranch = "main",
)
