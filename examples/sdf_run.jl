#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "examples", "SDF.jl"))

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

try
    MeshCat.close_server!(vis.core)
catch
end

#======================================================#

casename = "Gear.npz"
modeldir  = joinpath(@__DIR__, "dump1")

# casename = "Mech.npz"
# modeldir  = joinpath(@__DIR__, "dump2")

modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

E = 700
h, w = 5, 64

# isdir(modeldir) && rm(modeldir, recursive = true)
# model, ST, md = train_SDF(casename, modeldir, h, w, E; rng, device)
vis = postprocess_SDF(casename, modelfile)
open(vis; start_browser = false)
#======================================================#
nothing
