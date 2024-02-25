#
using GeometryLearning

begin
    path = joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl")
    include(path)
end

#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "model1")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

case = 1

## train
E = 1400
_It = Colon()
_batchsize = 1280
l, h, w = 4, 5, 32 # (2, 4), 5, 32
λ1, λ2 = 0f0, 0f0
σ2inv, α = 0f-0, 0f0
weight_decays = 1f-3

# isdir(modeldir) && rm(modeldir, recursive = true)
# makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = _It, It_ = :)
# model, STATS = train_autodecoder(datafile, modeldir, l, h, w, E;
#     rng, warmup = true, _batchsize,
#     λ1, λ2, σ2inv, α, weight_decays, makedata_kws, device,
# )

## process
# outdir = joinpath(modeldir, "results")
# postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
x, up, ud = evolve_autodecoder(prob, datafile, modelfile, case; rng, device)
#======================================================#
# ## train (5x less snapshots)
# E = 1400
# _It = LinRange(1, 500, 201) .|> Base.Fix1(round, Int)
# _batchsize = 128 * 1
# l, h, w = 4, 5, 32 # (2, 4), 5, 32
# λ1, λ2 = 0f0, 0f0
# σ2inv, α = 1f-3, 0f-6 # 1f-0, 1f-0
# weight_decays = 1f-3  # 1-f0,

# ## train (intrinsic latent space size)
# E = 3500
# _It = Colon()
# _batchsize = 128 * 10
# l, h, w = 1, 5, 64 # 1, 5, 64
# λ1, λ2 = 0f0, 0f0
# σ2inv, α = 1f-2, 0f-6 # 1f-3
# weight_decays = 1f-3  # 2f-3
#======================================================#

nothing
#
