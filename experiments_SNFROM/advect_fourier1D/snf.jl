#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

# latent = 2
# train_params = (; E = 1400, wd = 64, α = 0f-0, γ = 1f-2, makedata_kws,)
# makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

evolve_kw = (; hyper_reduction = true,)
postprocess_SNF(prob, datafile, modelfile; rng, evolve_kw, device)
#======================================================#
nothing
