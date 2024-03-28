#
using GeometryLearning
using Plots, LaTeXStrings

joinpath(pkgdir(GeometryLearning), "examples", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = KuramotoSivashinsky1D(0.01f0)
datafile = joinpath(@__DIR__, "data_ks/", "data.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)

# latent 
latent = 1
l_pca  = 2

#==================#
# train
#==================#

l0 = lpad(latent, 2, "0")
lp = lpad(l_pca , 2, "0")

modeldir_PCA = joinpath(@__DIR__, "model_PCA$(lp)") # traditional
modeldir_CAE = joinpath(@__DIR__, "model_CAE$(l0)") # Lee, Carlberg
modeldir_SNW = joinpath(@__DIR__, "model_SNW$(l0)") # us (Weight decay)
modeldir_SNL = joinpath(@__DIR__, "model_SNL$(l0)") # us (Lipschitz)

# # train PCA
# train_PCA(datafile, modeldir_PCA, l_pca; rng, makedata_kws, device)
#
# # train_CAE
# train_params_CAE = (; E = 1400, w = 64, makedata_kws,)
# train_CAE_compare(prob, latent, datafile, modeldir_CAE, train_params_CAE; rng, device)
#
# # train_SNW
# train_params_SNW = (; E = 1400, wd = 128, α = 0f-0, γ = 1f-2, makedata_kws,)
# train_SNF_compare(latent, datafile, modeldir_SNW, train_params_SNW; rng, device)
#
# train_SNL
# train_params_SNL = (; E = 1400, wd = 128, wh = 8, hh = 0, α = 1f-7, γ = 0f-0, makedata_kws,)
# train_SNF_compare(latent, datafile, modeldir_SNL, train_params_SNL; rng, device)

#==================#
# postprocess
#==================#

modelfile_PCA = joinpath(modeldir_PCA, "model.jld2")
modelfile_CAE = joinpath(modeldir_CAE, "model_07.jld2")
modelfile_SNW = joinpath(modeldir_SNW, "model_08.jld2")
modelfile_SNL = joinpath(modeldir_SNL, "model_08.jld2")

# postprocess_PCA(prob, datafile, modelfile_PCA; rng, device)
# postprocess_CAE(prob, datafile, modelfile_CAE; rng, device)
# postprocess_SNF(prob, datafile, modelfile_SNW; rng, device)
# postprocess_SNF(prob, datafile, modelfile_SNL; rng, device)

#==================#
# make figures
#==================#
grid = (256,)
casename = "ks1d"
modeldirs = (; modeldir_PCA, modeldir_CAE, modeldir_SNW, modeldir_SNL,)
labels = ("PCA R = $(l_pca)", "Lee & Carlberg", "SNFW (ours)", "SNFL (ours)")

p1, p2, p3 = compare_plots(modeldirs, labels, @__DIR__, casename, 1, grid)

#======================================================#
nothing
