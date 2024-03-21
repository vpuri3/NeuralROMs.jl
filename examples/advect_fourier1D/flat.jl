#
using GeometryLearning

include(joinpath(pkgdir(GeometryLearning), "examples", "flatNF.jl"))
include(joinpath(pkgdir(GeometryLearning), "examples", "problems.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "flat")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

case = 1

_data, data_, md_data = makedata_FNF(datafile)

## train
E = 1400
l = 2
hh, wh = 3, 8
hd, wd = 5, 64
λ2, α, weight_decays = 1f-3, 0f0, 1f-3

# isdir(modeldir) && rm(modeldir, recursive = true)
# model, STATS, metadata = train_FNF(datafile, modeldir,
#     l, hh, hd, wh, wd, E;
#     rng, warmup = true, λ2, α, weight_decays, device,
# )

## process
# postprocess_FNF(prob, datafile, modelfile; rng, device)
x, t, up, ud, _ = evolve_FNF(prob, datafile, modelfile, case; rng, device)

@show sqrt(mse(up, ud) / mse(ud, 0 * ud))
@show norm(up - ud, Inf) / sqrt(mse(ud, 0 * ud))
#======================================================#
nothing
