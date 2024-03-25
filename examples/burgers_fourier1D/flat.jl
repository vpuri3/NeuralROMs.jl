#
using GeometryLearning
include(joinpath(pkgdir(GeometryLearning), "examples", "smoothNF.jl"))
include(joinpath(pkgdir(GeometryLearning), "examples", "problems.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 213)

prob = BurgersViscous1D(1f-4)
datafile = joinpath(@__DIR__, "data_burg1D", "data.jld2")
modeldir = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

# makedata
_Ib, Ib_ = [1,3,], [2,]
Ix  = Colon()
_It = Colon()
makedata_kws = (; Ix, _Ib, Ib_, _It = _It, It_ = :)

# _data, data_, md = makedata_SNF(datafile; makedata_kws...)

## train
E = 1400
l = 2
hh, wh = 0, 8
hd, wd = 5, 128
λ2, α, weight_decays = 1f-3, 0f0, 1f-2

isdir(modeldir) && rm(modeldir, recursive = true)
model, STATS, metadata = train_SNF(
    datafile, modeldir, l, hh, hd, wh, wd, E;
    rng, warmup = true, λ2, α, weight_decays, device,
)

## process
postprocess_SNF(prob, datafile, modelfile; rng, device)
#======================================================#
nothing
