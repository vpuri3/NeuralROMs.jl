#
using GeometryLearning

joinpath(pkgdir(GeometryLearning), "examples", "autodecode.jl") |> include
joinpath(pkgdir(GeometryLearning), "examples", "problems.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 220)

prob = BurgersViscous2D(1f-3)
datafile = joinpath(@__DIR__, "data_burgers2D/", "data.jld2")
device = Lux.gpu_device()

modeldir = joinpath(@__DIR__, "model2")
modelfile = joinpath(modeldir, "model_08.jld2")

cb_epoch = nothing

Nx = 512
Ix  = LinRange(1, 512, 128) .|> Base.Fix1(round, Int)
Ix  = LinearIndices((Nx, Nx))[Ix, Ix] |> vec
_It = LinRange(1, 500, 201) .|> Base.Fix1(round, Int) # 101

Ix = Colon()

E = 1400
l, h, w = 8, 5, 128
λ1, λ2   = 0f0, 0f0
σ2inv, α = 1f-2, 0f-6
weight_decays = 2f-2

# isdir(modeldir) && rm(modeldir, recursive = true)
# makedata_kws = (; Ix, _Ib = :, Ib_ = :, _It = _It, It_ = :)
# model, STATS, metadata = train_autodecoder(datafile, modeldir, l, h, w, E;
#     λ1, λ2, σ2inv, α, weight_decays, cb_epoch, device, makedata_kws,
#     _batchsize = 16384,
#     batchsize_ = (Nx * Nx ÷ 10),
# )

## process
outdir = joinpath(modeldir, "results")
# postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
x, up, ud = test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
#======================================================#
nothing
#
