#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = BurgersViscous2D(1f-3)
datafile = joinpath(@__DIR__, "data_burgers2D/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_07.jld2")
device = Lux.gpu_device()

# train
#
# latent = 2
# grid = (512, 512,)
# _batchsize = prod(grid) * 1
# batchsize_ = prod(grid) * 8
# _Ib, Ib_ = [1, 2, 3, 5, 6, 7], [4,]
# makedata_kws = (; Ix = :, _Ib, Ib_, _It = :, It_ = :)
# train_params = (; E = 210, wd = 128, γ = 1f-2, makedata_kws, _batchsize, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # modeldir/results
# postprocess_SNF(prob, datafile, modelfile; rng, device)

# fomfile = joinpath(@__DIR__, "FOM_timings.jl")
# sROM, sFOM, sfile = hyper_timings(prob, datafile, modelfile, "exp4", 4, fomfile)
outdir = joinpath(pkgdir(NeuralROMs), "figs", "method")
hyper_plots(datafile, modeldir, outdir, "exp4", 4; makefigs = true)
#======================================================#
