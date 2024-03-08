#
using GeometryLearning
using Plots, LaTeXStrings
using NPZ

begin
    pcapath = joinpath(pkgdir(GeometryLearning), "examples", "PCA.jl")
    snfpath = joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl")
    inrpath = joinpath(pkgdir(GeometryLearning), "examples", "convINR.jl")

    include(snfpath)
    include(inrpath)
end
#======================================================#
_Ib, Ib_ = [1,], [1,]
Ix  = Colon()
_It = Colon()
makedata_kws = (; Ix, _Ib, Ib_, _It, It_ = :)
case = 1
#======================================================#

function advect1d_train_CINR(
    l::Integer, 
    modeldir::String;
    device = Lux.cpu_device(),
)
    E   = 50000  # epochs
    h   = 5      # num decoder hidden
    we  = 32     # encoder width
    wd  = 64     # decoder width
    act = tanh   # relu, tanh

    NN = convINR_network(prob, l, h, we, wd, act)

    isdir(modeldir) && rm(modeldir, recursive = true)
    train_CINR(datafile, modeldir, NN, E; rng, warmup = true, device)
end

function advect1d_train_SNFW(
    l::Integer, 
    modeldir::String;
    device = Lux.cpu_device(),
)
    E = 7000  # epochs
    h = 5     # num hidden
    w = 64    # width

    λ1, λ2 = 0f0, 0f0     # L1 / L2 reg
    σ2inv, α = 1f-1, 0f-0 # code / Lipschitz regularization
    weight_decays = 1f-2  # AdamW weight decay

    train_SNF(datafile, modeldir, l, h, w, E;
        rng, warmup = true, makedata_kws,
        λ1, λ2, σ2inv, α, weight_decays, device,
    )
end

function advect1d_train_SNFL(
    l::Integer, 
    modeldir::String;
    device = Lux.cpu_device(),
)
    E = 7000  # epochs
    h = 5     # num hidden
    w = 64    # width

    λ1, λ2 = 0f0, 0f0     # L1 / L2 reg
    σ2inv, α = 1f-3, 1f-4 # code / Lipschitz regularization
    weight_decays = 0f-0  # AdamW weight decay

    train_SNF(datafile, modeldir, l, h, w, E;
        rng, warmup = true, makedata_kws,
        λ1, λ2, σ2inv, α, weight_decays, device,
    )
end
#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")

latent = 2

l0 = latent * 2^0
l1 = latent * 2^1
l2 = latent * 2^2

ll0 = lpad(latent * 2^0, 2, "0")
ll1 = lpad(latent * 2^1, 2, "0")
ll2 = lpad(latent * 2^2, 2, "0")

device = Lux.gpu_device()

#==================#
# train
#==================#

modeldir_CINR = joinpath(@__DIR__, "model_CINR_l_$(ll0)")
modeldir_SNFW = joinpath(@__DIR__, "model_SNFW_l_$(ll0)")
modeldir_SNFL = joinpath(@__DIR__, "model_SNFL_l_$(ll0)")
modeldir_PCA0 = joinpath(@__DIR__, "model_PCA_l_$(ll0)")
modeldir_PCA1 = joinpath(@__DIR__, "model_PCA_l_$(ll1)")
modeldir_PCA2 = joinpath(@__DIR__, "model_PCA_l_$(ll2)")

advect1d_train_CINR(latent, modeldir_CINR; device)
# advect1d_train_SNFW(latent, modeldir_SNFW; device)
# advect1d_train_SNFL(latent, modeldir_SNFL; device)
# train_PCA(datafile, modeldir_PCA0, l0; makedata_kws, device)
# train_PCA(datafile, modeldir_PCA1, l1; makedata_kws, device)
# train_PCA(datafile, modeldir_PCA2, l2; makedata_kws, device)

#==================#
# evolve
#==================#

modelfile_CINR = joinpath(modeldir_CINR, "model_08.jld2")
modelfile_SNFW = joinpath(modeldir_SNFW, "model_08.jld2")
modelfile_SNFL = joinpath(modeldir_SNFL, "model_08.jld2")

modelfile_PCA0 = joinpath(modeldir_PCA0, "model.jld2")
modelfile_PCA1 = joinpath(modeldir_PCA1, "model.jld2")
modelfile_PCA2 = joinpath(modeldir_PCA2, "model.jld2")

x1, t1, ud1, up1, _ = evolve_CINR(prob, datafile, modelfile_CINR, case; rng, device)
x2, t2, ud2, up2, _ = evolve_SNF( prob, datafile, modelfile_SNFW, case; rng, device)
x3, t3, ud3, up3, _ = evolve_SNF( prob, datafile, modelfile_SNFL, case; rng, device)
#
x4, t4, ud4, up4, _ = evolve_PCA( prob, datafile, modelfile_PCA0, case; rng, device)
x5, t5, ud5, up5, _ = evolve_PCA( prob, datafile, modelfile_PCA1, case; rng, device)
x6, t6, ud6, up6, _ = evolve_PCA( prob, datafile, modelfile_PCA2, case; rng, device)

#==================#
# clean data
#==================#

x1 = dropdims(x1; dims = 1)
x2 = dropdims(x2; dims = 1)
x3 = dropdims(x3; dims = 1)
x4 = dropdims(x4; dims = 1)
x5 = dropdims(x5; dims = 1)
x6 = dropdims(x6; dims = 1)

ud1, up1 = dropdims.((ud1, up1); dims = 1)
ud2, up2 = dropdims.((ud2, up2); dims = 1)
ud3, up3 = dropdims.((ud3, up3); dims = 1)

ud4, up4 = dropdims.((ud4, up4); dims = 1)
ud5, up5 = dropdims.((ud5, up5); dims = 1)
ud6, up6 = dropdims.((ud6, up6); dims = 1)

#==================#
# save data
#==================#

filename = joinpath(@__DIR__, "advect1d.npz")

dict = Dict(
    "xdata" => x1, "tdata" => t1, "udata" => ud1,
    "xCROM" => x1, "tCROM" => t1, "uCROM" => up1,
    "xSNFW" => x2, "tSNFW" => t2, "uSNFW" => up2,
    "xSNFL" => x3, "tSNFL" => t3, "uSNFL" => up3,
    #
    "xPCA0" => x4, "tPCA0" => t4, "uPCA0" => up4, # 1l modes
    "xPCA1" => x5, "tPCA1" => t5, "uPCA1" => up5, # 2l modes
    "xPCA2" => x6, "tPCA2" => t6, "uPCA2" => up6, # 4l modes
)

npzwrite(filename, dict)

#==================#
# figure
#==================#
plt = plot(;
    xlabel = L"x",
    ylabel = L"u(x, t)",

    title = "1D Advection",

    legendfontsize = 10,
    legend = :bottomleft,
)

i1, i2 = 1, length(t1)

# # data
# plot!(plt, x1, ud1[:, i1], w = 4, s = :solid, c = :black, label = "Ground truth")
# plot!(plt, x1, ud1[:, i2], w = 4, s = :solid, c = :black, label = nothing)
#
# # models
# plot!(plt, x1, up1[:, i2], w = 4, s = :solid, c = :green, label = "C-ROM (ref. [2])")
# # plot!(plt, x2, up2[:, i2], w = 4, s = :solid, c = :red  , label = "Smooth-NFW (ours)")
# # plot!(plt, x3, up3[:, i2], w = 4, s = :solid, c = :blue , label = "Smooth-NFL (ours)")
#
# # PCA
# # plot!(plt, x4, up4[:, i2], w = 4, s = :solid, c = :orange, label = "PCA R=2")
# # plot!(plt, x5, up5[:, i2], w = 4, s = :solid, c = :brown , label = "PCA R=4")
# plot!(plt, x6, up6[:, i2], w = 4, s = :solid, c = :magenta, label = "PCA R=8")
#
# pltname = joinpath(@__DIR__, "compare_l_$(latent)")
# png(plt, pltname)

plt = plot(x1, ud1[:, [1, 100, 200, 300, 400, 500]]; w = 4, c = :black, label = "Data")
plot!(plt, x1, up1[:, [1, 100, 200, 300, 400, 500]]; w = 4, label = "CROM")

display(plt)
#======================================================#
nothing
