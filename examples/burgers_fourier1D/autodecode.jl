#
using GeometryLearning
include(joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

prob = BurgersViscous1D(1f-4)
device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_burg1D", "data.jld2")
modeldir = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")

E = 3500  # epochs
l = 8     # latent
h = 5     # num hidden
w = 128   # width

λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-1, 0f-0 # 1f-1, 1f-3
weight_decays = 1f-2  # 1f-2
WeightDecayOpt = IdxWeightDecay
weight_decay_ifunc = decoder_W_indices

_Ib, Ib_ = [1,3,], [2,]
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

# E = 2100
#
# _Ib, Ib_ = [1,], [1,]
# Ix  = LinRange(1, 8192, 1024) .|> Base.Fix1(round, Int)
# _It = Colon() # LinRange(1, 1000, 200 ) .|> Base.Fix1(round, Int)
# makedata_kws = (; Ix, _Ib, Ib_, _It = _It, It_ = :)
#
# for (l, h, w) in (
#     # (8, 5, 64),
#     (8, 5, 96),
#     # (8, 10, 64),
#     # (8, 10, 96),
# )
#     ll = lpad(l, 2, "0")
#     hh = lpad(h, 2, "0")
#     ww = lpad(w, 3, "0")
#
#     modeldir  = joinpath(@__DIR__, "model_dec_sin_$(ll)_$(hh)_$(ww)_reg")
#     modelfile = joinpath(modeldir, "model_08.jld2")
#
#     # train
#     λ1, λ2   = 0f0, 0f0
#
#     ## Weight decay
#
#     ## works for trajectory 7
#     σ2inv, α = 1f-3, 0f-6 # 1f-3, 0f-0
#     weight_decays = 3.0f-2  # 2.5f-2
#
#     ## works for trajectory 3
#     # σ2inv, α = 1f-3, 0f-6 # 1f-3, 0f-0
#     # weight_decays = 3.5f-2  # 2.5f-2
#
#     ## works for trajectory 1
#     # σ2inv, α = 5f-3, 0f-4
#     # weight_decays = 5f-2
#
#     ## Lipschitz regularization
#     # σ2inv, α = 1f-3, 1f-5 # 1f-3 (bump up), 0f-0
#     # weight_decays = 0f-0  # 1f-2 (bump up)
#
#     ## train
#     isdir(modeldir) && rm(modeldir, recursive = true)
#     model, STATS = train_autodecoder(datafile, modeldir, l, h, w, E;
#         λ1, λ2, σ2inv, α, weight_decays, makedata_kws, device,
#     )
#
#     ## process
#     outdir = joinpath(modeldir, "results")
#     postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#         makeplot = true, verbose = true)
#     test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#         makeplot = true, verbose = true)
# end
# #======================================================#
# nothing
#
