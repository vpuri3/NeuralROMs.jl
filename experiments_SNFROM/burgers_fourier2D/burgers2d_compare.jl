#
using NeuralROMs
using Plots, LaTeXStrings

joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = BurgersViscous2D(1f-3)
datafile = joinpath(@__DIR__, "data_burgers2D/", "data.jld2")
device = Lux.gpu_device()

_Ib, Ib_ = [1, 2, 3, 5, 6, 7], [4,]
makedata_kws = (; Ix = :, _Ib, Ib_, _It = :, It_ = :)

# latent 
latent = 2
l_pca  = 8

#==================#
# train
#==================#

l0 = lpad(latent, 2, "0")
lp = lpad(l_pca , 2, "0")

modeldir_PCA = joinpath(@__DIR__, "model_PCA$(lp)") # traditional
modeldir_CAE = joinpath(@__DIR__, "model_CAE$(l0)") # Lee, Carlberg
modeldir_SNL = joinpath(@__DIR__, "model_SNL$(l0)") # us (Lipschitz)
modeldir_SNW = joinpath(@__DIR__, "model_SNW$(l0)") # us (Weight decay)

# # train PCA
# train_PCA(datafile, modeldir_PCA, l_pca; rng, makedata_kws, device)

# # train_CAE
# train_params_CAE = (; E = 350, w = 64, act = elu, makedata_kws, _batchsize = 4, batchsize_ = 300)
# train_CAE_compare(prob, latent, datafile, modeldir_CAE, train_params_CAE; rng, device)

grid = (512, 512,)
_batchsize = prod(grid) * 1 # 1 (gpu usage 40% constant)
batchsize_ = prod(grid) * 8 # up to 32

# # train_SNL
# train_params_SNL = (; E = 210, wd = 128, α = 1f-4, makedata_kws, _batchsize, batchsize_)
# train_SNF_compare(latent, datafile, modeldir_SNL, train_params_SNL; rng, device)

# # train_SNW
# train_params_SNW = (; E = 210, wd = 128, γ = 1f-2, makedata_kws, _batchsize, batchsize_)
# train_SNF_compare(latent, datafile, modeldir_SNW, train_params_SNW; rng, device)

#==================#
# postprocess
#==================#

modelfile_PCA = joinpath(modeldir_PCA, "model.jld2")
modelfile_CAE = joinpath(modeldir_CAE, "model_07.jld2")
modelfile_SNL = joinpath(modeldir_SNL, "model_08.jld2")
modelfile_SNW = joinpath(modeldir_SNW, "model_07.jld2")

# postprocess_PCA(prob, datafile, modelfile_PCA; rng, device)
# postprocess_CAE(prob, datafile, modelfile_CAE; rng)
# postprocess_SNF(prob, datafile, modelfile_SNL; rng, device)
# postprocess_SNF(prob, datafile, modelfile_SNW; rng, device)

cases = 1:7
cases = [4,]

# for case in cases
#     evolve_CAE(prob, datafile, modelfile_CAE, case; rng)
#     evolve_SNF(prob, datafile, modelfile_SNL, case; rng, device)
#     evolve_SNF(prob, datafile, modelfile_SNW, case; rng, device)
# end

#==================#
# small DT
#==================#

# T  = 0.5f0
# Nt = 500
# It = LinRange(1, Nt, 50) .|> Base.Fix1(round, Int)
# data_kws = (; Ix = :, It)
# evolve_kw = (; Δt = T, data_kws, adaptive = false)
#
# outdir_SNW = joinpath(modeldir_SNW, "dt")
# outdir_SNL = joinpath(modeldir_SNL, "dt")
# outdir_CAE = joinpath(modeldir_CAE, "dt")
#
# for case in cases
#     evolve_CAE(prob, datafile, modelfile_CAE, case; rng, outdir = outdir_CAE, evolve_kw...,)
#     evolve_SNF(prob, datafile, modelfile_SNL, case; rng, outdir = outdir_SNL, evolve_kw..., device)
#     evolve_SNF(prob, datafile, modelfile_SNW, case; rng, outdir = outdir_SNW, evolve_kw..., device)
# end

#==================#
# make figures
#==================#

# casename = "burgers2d"
# modeldirs = (; modeldir_PCA, modeldir_CAE, modeldir_SNL, modeldir_SNW,)
# labels = ("POD ($(2*l_pca) modes)", "CAE", "SNFL (ours)", "SNFW (ours)")
#
# for case in cases
#     _casename = casename * "case$case"
#     @show _casename
#     compare_plots(modeldirs, labels, @__DIR__, _casename, case, grid; ifdt = true)
# end
#======================================================#
