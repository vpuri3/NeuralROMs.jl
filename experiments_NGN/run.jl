#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect1D/", "data.jld2")
device = Lux.gpu_device()

case = 1
data_kws = (; Ix = :, It = :)

#------------------------------------------------------#
# DNN
#------------------------------------------------------#
# train_params = (;)
# evolve_params = (;)
# makemodel = makemodelDNN
# modeldir  = joinpath(@__DIR__, "dump_dnn")
# modelfile = joinpath(modeldir, "projectT0", "model_08.jld2")

#------------------------------------------------------#
# Gaussian
#------------------------------------------------------#
train_params = (;)
evolve_params = (;)
makemodel = makemodelGaussian
modeldir  = joinpath(@__DIR__, "dump_gaussian")
modelfile = joinpath(modeldir, "projectT0", "model.jld2")

# TODO
# - is error due to large time-step size or due to improper integration
# - see how error varies with number of collocation points.

#------------------------------------------------------#
# Evolve
#------------------------------------------------------#
(NN, p, st), _, _ = ngProject(prob, datafile, modeldir, makemodel, case; rng, train_params, data_kws, device)
(Xd, Td, Ud, Up, ps), _ = ngEvolve(prob, datafile, modelfile, case; rng, device)

#======================================================#
nothing
