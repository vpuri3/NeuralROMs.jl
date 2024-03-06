#
using GeometryLearning
include(joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 123)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_ks/", "data.jld2")
modeldir = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")

prob = KuramotoSivashinsky1D(0.01f0)

E = 2100  # epochs
l = 16    # latent
h = 5     # num hidden
w = 128   # width

### WORKS
λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-1, 0f-0 # 1f-1, 1f-3
weight_decays = 1f-2  # 1f-2
WeightDecayOpt = DecoderWeightDecay
weight_decay_ifunc = nothing

## train
isdir(modeldir) && rm(modeldir, recursive = true)
train_SNF(datafile, modeldir, l, h, w, E;
    rng, warmup = true,
    λ1, λ2, σ2inv, α, weight_decays, device,
    WeightDecayOpt, weight_decay_ifunc,
)

outdir = joinpath(modeldir, "results")
postprocess_SNF(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
x, t, ud, up, _ = evolve_SNF(prob, datafile, modelfile, 1; rng, device, verbose = true)
plt = plot(x[1,:], ud[1,:,begin], w = 4, c = :black, label = nothing)
plot!(plt, x[1,:], ud[1,:,end  ], w = 4, c = :black, label = "data")
plot!(plt, x[1,:], up[1,:,end  ], w = 4, c = :red  , label = "pred")
display(plt)

@show sqrt(mse(up, ud) / mse(ud, 0 * ud))
@show norm(up - ud, Inf) / sqrt(mse(ud, 0 * ud))
#======================================================#
nothing

## train
# E = 7_000
# _It = LinRange(1, 1000, 100) .|> Base.Fix1(round, Int) # 200
# _batchsize = 256 * 5
# l, h, w = 16, 5, 64
# λ1, λ2 = 0f0, 0f0
# σ2inv, α = 1f-1, 0f-0 # 1f-1, 1f-3
# weight_decays = 1f-2  # 1f-2
#
# isdir(modeldir) && rm(modeldir, recursive = true)
# makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = _It, It_ = :)
# model, STATS = train_SNF(datafile, modeldir, l, h, w, E;
#     λ1, λ2, σ2inv, α, weight_decays, device, makedata_kws,
#     _batchsize,
# )
#
