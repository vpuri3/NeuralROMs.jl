#
using GeometryLearning
include(joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

prob = BurgersViscous1D(1f-4)
device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.jld2")
modeldir = joinpath(@__DIR__, "dump1")
modelfile = joinpath(modeldir, "model_08.jld2")

E = 3500  # epochs
l = 8     # latent
h = 5     # num hidden
w = 128   # width

λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-1, 0f-0 # 1f-1, 1f-3
weight_decays = 1f-2  # 1f-2
WeightDecayOpt = IdxWeightDecay
weight_decay_ifunc = decoder_indices

_Ib, Ib_ = [5,], [5,] # 1, 2, 3 work
Ix  = Colon() # LinRange(1, 8192, 1024) .|> Base.Fix1(round, Int)
_It = Colon() # LinRange(1, 1000, 200 ) .|> Base.Fix1(round, Int)
makedata_kws = (; Ix, _Ib, Ib_, _It = _It, It_ = :)

## train
isdir(modeldir) && rm(modeldir, recursive = true)
train_SNF(datafile, modeldir, l, h, w, E;
    rng, warmup = true, makedata_kws,
    λ1, λ2, σ2inv, α, weight_decays, device,
    WeightDecayOpt, weight_decay_ifunc,
)

outdir = joinpath(modeldir, "results")
postprocess_SNF(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
x, t, ud, up, _ = evolve_SNF(prob, datafile, modelfile, Ib_[1]; rng, device, verbose = true)
plt = plot(x[1,:], ud[1,:,begin], w = 4, c = :black, label = nothing)
plot!(plt, x[1,:], ud[1,:,end  ], w = 4, c = :black, label = "data")
plot!(plt, x[1,:], up[1,:,end  ], w = 4, c = :red  , label = "pred")
display(plt)

@show sqrt(mse(up, ud) / mse(ud, 0 * ud))
@show norm(up - ud, Inf) / sqrt(mse(ud, 0 * ud))
#======================================================#
nothing
