#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "experiments_SDF", "SDF.jl"))

rng = Random.default_rng()
Random.seed!(rng, 199)
#======================================================#

# casename = "Gear.npz"
# modeldir  = joinpath(@__DIR__, "dump1")

# casename = "Temple.npz"
# modeldir  = joinpath(@__DIR__, "dump2")

# casename = "Burger.npz"
# modeldir  = joinpath(@__DIR__, "dump3")

# casename = "HumanSkull.npz"
# modeldir  = joinpath(@__DIR__, "dump4")

casename = "Cybertruck.npz"
modeldir  = joinpath(@__DIR__, "dump5")

modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

E = 490
h, w = 5, 512

# # Train
# isdir(modeldir) && rm(modeldir, recursive = true)
# model, ST, md = train_SDF(casename, modeldir, h, w, E; rng, device)

isdefined(Main, :server) && close(server)
server = postprocess_SDF(modelfile; device)
#======================================================#
