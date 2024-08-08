#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

prob = AdvectionDiffusion1D(1.00f0, 0f-2)
# prob = AdvectionDiffusion1D(0.25f0, 1f-2)
datafile  = joinpath(@__DIR__, "data_advectionDiffusion1D/", "data.jld2")
device = Lux.gpu_device()

case = 1
data_kws = (; Ix = :, It = :)

#------------------------------------------------------#
# DNN
#------------------------------------------------------#
# train_params = (;)
# evolve_params = (; T = Float32,)
#
# makemodel = makemodelDNN
# modeldir  = joinpath(@__DIR__, "dump_dnn")
# modelfile = joinpath(modeldir, "project$(case)", "model_08.jld2")

#------------------------------------------------------#
# Gaussian
#------------------------------------------------------#
train_params = (;)
# data_kws = (; Ix = 1:64:256, It = :)
evolve_params = (; data_kws, T = Float64, timealg = RK4(),)

makemodel = makemodelGaussian
modeldir  = joinpath(@__DIR__, "dump_gaussian")
modelfile = joinpath(modeldir, "project$(case)", "model.jld2")

#------------------------------------------------------#
# Evolve
#------------------------------------------------------#

# parameters
cs = evolve_params.T[1, 0, 1]
νs = evolve_params.T[0, 0.01, 0.01]

for case in 1:3
    prob = AdvectionDiffusion1D(cs[case], νs[case])
    (NN, p, st), _, _ = ngProject(prob, datafile, modeldir, makemodel, case; rng, train_params, data_kws, device)
    (Xd, Td, Ud, Up, ps), _ = ngEvolve(prob, datafile, modelfile, case; rng, evolve_params, device)
    sleep(1)
end

#======================================================#
nothing
