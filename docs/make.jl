using Documenter
using MultiResponseVarianceComponentModels

makedocs(
    modules = [MultiResponseVarianceComponentModels],
    sitename = "MRVCModels",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
        warn_outdated = true,
        collapselevel = 3,
        ),
    pages = [
        "Home" => "index.md",
        "Examples" => "examples.md",
        "Advanced details" => "advanced.md",
        "API"  => "api.md",
        ]
    )

deploydocs(;
    repo = "github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl.git",
    target = "build",
    forcepush = true
    )