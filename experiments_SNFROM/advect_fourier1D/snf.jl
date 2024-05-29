#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "smoothNF.jl"))
include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "cases.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

_It = LinRange(1, 500, 50) .|> Base.Fix1(round, Int)
makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It, It_ = :,)
train_params = (; E = 1400, wd = 64, α = 0f-0, γ = 1f-2, makedata_kws,)

train_SNF_compare(2, datafile, modeldir, train_params; rng, device)
postprocess_SNF(prob, datafile, modelfile; rng, device)

#======================================================#
nothing
