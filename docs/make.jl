using NeuralROMs
using Documenter

DocMeta.setdocmeta!(NeuralROMs, :DocTestSetup, :(using NeuralROMs); recursive=true)

makedocs(;
    modules=[NeuralROMs],
    authors="Vedant Puri <vedantpuri@gmail.com> and contributors",
    # repo="https://github.com/vpuri3/NeuralROMs.jl/blob/{commit}{path}#{line}",
    repo=Remotes.GitHub( "vpuri3", "NeuralROMs.jl"),
    sitename="NeuralROMs.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://vpuri3.github.io/NeuralROMs.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "API" => "API.md",
    ],
)

deploydocs(;
    repo="github.com/vpuri3/NeuralROMs.jl",
    devbranch="master",
)
