using GeometryLearning
using Documenter

DocMeta.setdocmeta!(GeometryLearning, :DocTestSetup, :(using GeometryLearning); recursive=true)

makedocs(;
    modules=[GeometryLearning],
    authors="Vedant Puri <vedantpuri@gmail.com> and contributors",
    repo="https://github.com/vpuri3/GeometryLearning.jl/blob/{commit}{path}#{line}",
    sitename="GeometryLearning.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://vpuri3.github.io/GeometryLearning.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/vpuri3/GeometryLearning.jl",
    devbranch="master",
)
