#
using GeometryLearning
using Plots, LaTeXStrings
using NPZ

begin
    snfpath = joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl")
    inrpath = joinpath(pkgdir(GeometryLearning), "examples", "convINR.jl")

    include(snfpath)
    include(inrpath)
end
#======================================================#

function advect1d_train_CINR(
    l::Integer, 
    modeldir::String;
    device = Lux.cpu_device(),
)
    E   = 7000  # epochs
    # l   = 4     # latent
    h   = 5     # num decoder hidden
    we  = 32    # encoder width
    wd  = 64    # decoder width
    act = tanh  # relu, tanh

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
    # l = 4     # latent
    h = 5     # num hidden
    w = 64    # width

    λ1, λ2 = 0f0, 0f0     # L1 / L2 reg
    σ2inv, α = 1f-3, 0f-0 # code / Lipschitz regularization
    weight_decays = 1f-3  # AdamW weight decay

    train_SNF(datafile, modeldir, l, h, w, E;
        rng, warmup = true,
        λ1, λ2, σ2inv, α, weight_decays, device,
    )
end

function advect1d_train_SNFL(
    l::Integer, 
    modeldir::String;
    device = Lux.cpu_device(),
)
    E = 7000  # epochs
    # l = 4     # latent
    h = 5     # num hidden
    w = 64    # width

    λ1, λ2 = 0f0, 0f0     # L1 / L2 reg
    σ2inv, α = 1f-3, 1f-4 # code / Lipschitz regularization
    weight_decays = 0f-0  # AdamW weight decay

    train_SNF(datafile, modeldir, l, h, w, E;
        rng, warmup = true,
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

latent = 4
ll = lpad(latent, 2, "0")

device = Lux.gpu_device()

#==================#
# train
#==================#

modeldir_CINR = joinpath(@__DIR__, "model_CINR_l_$(ll)")
modeldir_SNFW = joinpath(@__DIR__, "model_SNFW_l_$(ll)")
modeldir_SNFL = joinpath(@__DIR__, "model_SNFL_l_$(ll)")

# advect1d_train_CINR(latent, modeldir_CINR; device)
# advect1d_train_SNFW(latent, modeldir_SNFW; device)
# advect1d_train_SNFL(latent, modeldir_SNFL; device)

#==================#
# evolve
#==================#

case = 1

modelfile_CINR = joinpath(modeldir_CINR, "model_08.jld2")
modelfile_SNFW = joinpath(modeldir_SNFW, "model_08.jld2")
modelfile_SNFL = joinpath(modeldir_SNFL, "model_08.jld2")

x1, t1, ud1, up1, _ = evolve_CINR(prob, datafile, modelfile_CINR, case; rng, device)
x2, t2, ud2, up2, _ = evolve_SNF( prob, datafile, modelfile_SNFW, case; rng, device)
x3, t3, ud3, up3, _ = evolve_SNF( prob, datafile, modelfile_SNFL, case; rng, device)

#==================#
# clean data
#==================#

x1 = dropdims(x1; dims = 1)
x2 = dropdims(x2; dims = 1)
x3 = dropdims(x3; dims = 1)

ud1, up1 = dropdims.((ud1, up1); dims = 1)
ud2, up2 = dropdims.((ud2, up2); dims = 1)
ud3, up3 = dropdims.((ud3, up3); dims = 1)

#==================#
# save data
#==================#
filename = joinpath(@__DIR__, "advect1d.npz")

dict = Dict(
    "xdata" => x1,
    "tdata" => t1,
    "udata" => ud1,

    "xCROM" => x1,
    "tCROM" => t1,
    "uCROM" => up1,

    "xSNFW" => x2,
    "tSNFW" => t2,
    "uSNFW" => up2,

    "xSNFL" => x3,
    "tSNFL" => t3,
    "uSNFL" => up3,
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

Nt = length(t1)
i1 = 1
i2 = Nt

# data
plot!(plt, x1, ud1[:, i1], w = 4, s = :solid, c = :black, label = "Ground truth")
plot!(plt, x1, ud1[:, i2], w = 4, s = :solid, c = :black, label = nothing)

# plots
plot!(plt, x1, up1[:, i2], w = 4, s = :dashdot   , c = :green, label = "C-ROM (ref. [2])")
plot!(plt, x2, up2[:, i2], w = 4, s = :dashdotdot, c = :red  , label = "Smooth-NFW (ours)")
plot!(plt, x3, up3[:, i2], w = 4, s = :dashdotdot, c = :blue , label = "Smooth-NFL (ours)")

pltname = joinpath(@__DIR__, "compare_l_$(latent)")
png(plt, pltname)

display(plt)
#======================================================#
nothing
