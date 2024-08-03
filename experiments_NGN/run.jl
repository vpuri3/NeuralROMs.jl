#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "neuralGalerkin.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect1D/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "projectT0", "model_08.jld2")
device = Lux.gpu_device()

case = 1
data_kws = (; Ix = :, It = :)
train_params = (;)
evolve_params = (;)

(NN, p, st), _, _ = ngProject(datafile, modeldir, case; train_params, data_kws, device)
ngEvolve(prob, datafile, modelfile, case; rng, device)
#======================================================#
nothing
