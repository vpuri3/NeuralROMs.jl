#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "examples", "cases.jl"))
include(joinpath(pkgdir(NeuralROMs), "examples", "autodecode.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 213)

prob = KuramotoSivashinsky1D(0.01f0)
device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_ks/", "data.jld2")
modeldir = joinpath(@__DIR__, "dump/")
modelfile = joinpath(modeldir, "model_08.jld2")

# E = 3500  # epochs
# l = 1     # latent
# h = 5     # num hidden
# w = 192   # width
#
# λ1, λ2 = 0f0, 0f0
# σ2inv, α = 1f-2, 0f-0
# weight_decays = 5f-3
# WeightDecayOpt = IdxWeightDecay
# weight_decay_ifunc = decoder_W_indices
#
# ## train
# isdir(modeldir) && rm(modeldir, recursive = true)
# train_SNF(datafile, modeldir, l, h, w, E;
#     rng, warmup = true,
#     λ1, λ2, σ2inv, α, weight_decays, device,
#     WeightDecayOpt, weight_decay_ifunc,
# )

postprocess_SNF(prob, datafile, modelfile; rng, device, makeplot = true, verbose = true)
x, t, ud, up, _ = evolve_SNF(prob, datafile, modelfile, 1; rng, device, verbose = true, learn_ic = false)
#
# plt = plot(x[1,:], ud[1,:,begin], w = 4, c = :black, label = nothing)
# plot!(plt, x[1,:], ud[1,:,end  ], w = 4, c = :black, label = "data")
# plot!(plt, x[1,:], up[1,:,end  ], w = 4, c = :red  , label = "pred")
# display(plt)

@show sqrt(mse(up, ud) / mse(ud, 0 * ud))
@show norm(up - ud, Inf) / sqrt(mse(ud, 0 * ud))
#======================================================#
nothing
