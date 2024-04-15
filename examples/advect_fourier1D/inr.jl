#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "examples", "convINR.jl"))

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 210)

# parameters
E   = 7000  # epochs
l   = 4     # latent
h   = 5     # hidden
we  = 32    # width
wd  = 64    # width
act = tanh  # relu, tanh

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "INR-dump")
modelfile = joinpath(modeldir, "model_08.jld2")
outdir    = joinpath(modeldir, "results")
device = Lux.gpu_device()

NN = inr_network(prob, l, h, we, wd, act)

## check sizes
# p, st = Lux.setup(rng, NN)
# p = ComponentArray(p)
# _data, _, _ = makedata_INR(datafile)
# @show _data[1] |> size
# @show _data[2] |> size
# @show NN(_data[1], p, st)[1] |> size
# @show length(p.encode.encoder)
# @show length(p.decoder)

## train
isdir(modeldir) && rm(modeldir, recursive = true)
model, ST, metadata = train_CINR(datafile, modeldir, NN, E; rng, warmup = true, device)

## evolve
x, t, ud, up, p = evolve_CINR(prob, datafile, modelfile, outdir; rng, device)
#======================================================#
nothing
#
