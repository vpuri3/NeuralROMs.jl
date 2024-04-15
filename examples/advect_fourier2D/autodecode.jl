#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "examples", "cases.jl"))
include(joinpath(pkgdir(NeuralROMs), "examples", "autodecode.jl"))

#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

prob = Advection2D(0.25f0, 0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

E = 1400  # epochs
l = 4     # latent
h = 5     # num hidden
w = 128   # width

λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-2, 0f-0
weight_decays = 1f-2
WeightDecayOpt = IdxWeightDecay
weight_decay_ifunc = decoder_W_indices

Ix  = Colon()
_It = Colon()
_Ib, Ib_ = [1,], [1,]
makedata_kws = (; Ix, _Ib, Ib_, _It = _It, It_ = :)

batchsize_ = (96 * 96) * 500 ÷ 4

## train
isdir(modeldir) && rm(modeldir, recursive = true)
train_SNF(datafile, modeldir, l, h, w, E;
    rng, warmup = true, makedata_kws,
    λ1, λ2, σ2inv, α, weight_decays, device,
    WeightDecayOpt, weight_decay_ifunc,
    batchsize_
)

## process
# outdir = joinpath(modeldir, "results")
# postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
# test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
#======================================================#
nothing
#
