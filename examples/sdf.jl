#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "examples", "SDF.jl"))

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

casename = "Gear.npz"
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

E = 700
h, w = 5, 32

isdir(modeldir) && rm(modeldir, recursive = true)
model, ST, md = train_SDF(casename, modeldir, h, w, E; rng, device)
#======================================================#
nothing
