#
using GeometryLearning

joinpath(pkgdir(GeometryLearning), "examples", "smoothNF.jl") |> include
joinpath(pkgdir(GeometryLearning), "examples", "problems.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

prob = BurgersViscous1D(1f-4)
device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_burg1D", "data.jld2")
modeldir = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")

E = 3500  # epochs
l = 4     # latent
h = 5     # num hidden
w = 128   # width

λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-1, 0f-0
weight_decays = 5f-3
WeightDecayOpt = IdxWeightDecay
weight_decay_ifunc = decoder_W_indices

_Ib, Ib_ = [1,3,], [2,]
Ix  = Colon()
_It = Colon()
makedata_kws = (; Ix, _Ib, Ib_, _It = _It, It_ = :)

## train
# isdir(modeldir) && rm(modeldir, recursive = true)
# train_SNF(datafile, modeldir, l, h, w, E;
#     rng, warmup = true, makedata_kws,
#     λ1, λ2, σ2inv, α, weight_decays, device,
#     WeightDecayOpt, weight_decay_ifunc,
# )

outdir = joinpath(modeldir, "results")
postprocess_SNF(prob, datafile, modelfile, outdir; rng, device, makeplot = true, verbose = true)
x, t, ud, up, _ = evolve_SNF(prob, datafile, modelfile, Ib_[1]; rng, device, verbose = true)
p1 = plot(x[1,:], ud[1,:,begin], w = 4, c = :black, label = nothing)
plot!(p1, x[1,:], ud[1,:,end  ], w = 4, c = :black, label = "data")
plot!(p1, x[1,:], up[1,:,end  ], w = 4, c = :red  , label = "pred")

n  = sum(abs2, ud) / length(ud)
e  = (up - ud) / n # (X, T)
et = sum(abs2, e; dims = (1,2)) / size(e, 2) |> vec
p2 = plot(t, et; w = 4, yaxis = :log, ylims = (10^-9, 1.0))
display(p2)
@show sqrt(mse(up, ud) / mse(ud, 0 * ud))
@show norm(up - ud, Inf) / sqrt(mse(ud, 0 * ud))
#======================================================#
nothing
#
