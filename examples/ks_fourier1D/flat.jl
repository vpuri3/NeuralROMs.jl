#
using GeometryLearning
include(joinpath(pkgdir(GeometryLearning), "examples", "smoothNF.jl"))
include(joinpath(pkgdir(GeometryLearning), "examples", "problems.jl"))
include(joinpath(pkgdir(GeometryLearning), "examples", "compare.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 213)

prob = KuramotoSivashinsky1D(0.01f0)
datafile = joinpath(@__DIR__, "data_ks/", "data.jld2")
modeldir = joinpath(@__DIR__, "dump/")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

# _It = LinRange(1, 1000, 1000) .|> Base.Fix1(round, Int)
_It = LinRange(1, 1000, 100) .|> Base.Fix1(round, Int)
makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It, It_ = :)

train_params = (; E = 1400, wd = 128, α = 0f-0, γ = 1f-2, makedata_kws,)
train_SNF_compare(1, datafile, modeldir, train_params; rng, device)
postprocess_SNF(prob, datafile, modelfile; rng, device)

# ## train
# E = 1400
# l = 2
# hh, wh = 0, 8
# hd, wd = 5, 64
# λ2, α, weight_decays = 1f-3, 0f0, 1f-2
#
# isdir(modeldir) && rm(modeldir, recursive = true)
# model, STATS, metadata = train_SNF(
#     datafile, modeldir, l, hh, hd, wh, wd, E;
#     rng, warmup = true, λ2, α, weight_decays, device,
# )

#======================================================#
nothing
