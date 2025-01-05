#
using NeuralROMs
using Plots, LaTeXStrings

joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)

# latent 
latent = 2
l_pca  = 16

#==================#
# train
#==================#

l0 = lpad(latent, 2, "0")
lp = lpad(l_pca , 2, "0")

modeldir_PCA = joinpath(@__DIR__, "model_PCA$(lp)") # traditional
modeldir_CAE = joinpath(@__DIR__, "model_CAE$(l0)") # Lee, Carlberg
modeldir_SNW = joinpath(@__DIR__, "model_SNW$(l0)") # us (Weight decay)
modeldir_SNL = joinpath(@__DIR__, "model_SNL$(l0)") # us (Lipschitz)
modeldir_INR = joinpath(@__DIR__, "model_INR$(l0)") # C-ROM

periodic_padding = periodic_layer = true
# periodic_padding = periodic_layer = false

# # train PCA
# train_PCA(datafile, modeldir_PCA, l_pca; makedata_kws, rng, device)
#
# # train_CAE
# train_params_CAE = (; E = 1400, w = 32, makedata_kws, act = elu)
# train_CAE_compare(prob, latent, datafile, modeldir_CAE, train_params_CAE; periodic_padding, rng, device)
#
# train_SNW
train_params_SNW = (; E = 1400, wd = 64, α = 0f-0, γ = 1f-2, makedata_kws,)
train_SNF_compare(latent, datafile, modeldir_SNW, train_params_SNW; periodic_layer, rng, device)

# # train_SNL
# train_params_SNL = (; E = 1400, wd = 64, α = 1f-4, γ = 0f-0, makedata_kws,)
# train_SNF_compare(latent, datafile, modeldir_SNL, train_params_SNL; periodic_layer rng, device)

#==================#
# postprocess
#==================#

modelfile_PCA = joinpath(modeldir_PCA, "model.jld2")
modelfile_CAE = joinpath(modeldir_CAE, "model_07.jld2")
modelfile_SNW = joinpath(modeldir_SNW, "model_08.jld2")
modelfile_SNL = joinpath(modeldir_SNL, "model_08.jld2")
modelfile_INR = joinpath(modeldir_INR, "model_07.jld2")

# postprocess_PCA(prob, datafile, modelfile_PCA; rng, device)
# postprocess_CAE(prob, datafile, modelfile_CAE; rng)
# postprocess_SNF(prob, datafile, modelfile_SNW; rng, device)
# postprocess_SNF(prob, datafile, modelfile_SNL; rng, device)

# #==================#
# # make figures
# #==================#
# grid = (128,)
# casename = "advect1d"
# modeldirs = (; modeldir_PCA, modeldir_CAE, modeldir_SNL, modeldir_SNW)
# label = ("POD ($(l_pca) modes)", "CAE", "SNFL (ours)", "SNFW (ours)")
#
# p1, p2, p3 = compare_plots(modeldirs, label, @__DIR__, casename, 1, grid; ifdt = false)
#======================================================#
