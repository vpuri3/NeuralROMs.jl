#
"""
Train an autoencoder on 1D advection data
"""

using GeometryLearning

include(joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl"))

#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 111)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_ks/", "data.jld2")

modeldir = joinpath(@__DIR__, "model3")
modelfile = joinpath(modeldir, "model_08.jld2")

prob = KuramotoSivashinsky1D(0.01f0)

## train
E = 7_000
_It = LinRange(1, 1000, 100) .|> Base.Fix1(round, Int) # 200
_batchsize = 256 * 5
l, h, w = 16, 5, 64
λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-1, 0f-5 # 1f-1, 1f-3
weight_decays = 1f-2  # 1f-2

isdir(modeldir) && rm(modeldir, recursive = true)
makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = _It, It_ = :)
model, STATS = train_autodecoder(datafile, modeldir, l, h, w, E;
    λ1, λ2, σ2inv, α, weight_decays, device, makedata_kws,
    _batchsize,
)

## process
outdir = joinpath(modeldir, "results")
postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
x, up, ud = test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
#======================================================#
nothing
#
