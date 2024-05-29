#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "convAE.jl") |> include

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

# parameters
E = 700    # epochs
l = 2      # latent
w = 32     # width
act = tanh # relu, tanh

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "CAE")
modelfile = joinpath(modeldir, "model_07.jld2")
outdir    = joinpath(modeldir, "results")
device = Lux.gpu_device()

## train
# NN = cae_network(prob, l, w, act)
# isdir(modeldir) && rm(modeldir, recursive = true)
# model, ST, metadata = train_CAE(datafile, modeldir, NN, E; rng, warmup = false, device)

## evolve (Jacobian fails on GPU)
case = 1
postprocess_CAE(prob, datafile, modelfile)
x, t, ud, up, ps = evolve_CAE(prob, datafile, modelfile, case; rng,)
# plt = plot(vec(x), ud[1, :, [1, 100, 200, 300, 400, 500]]; w = 4, c = :black, label = "Data")
# plot!(plt, vec(x), up[1, :, [1, 100, 200, 300, 400, 500]]; w = 4, label = "Lee, Carlberg")
# display(plt)
#======================================================#
nothing
#
