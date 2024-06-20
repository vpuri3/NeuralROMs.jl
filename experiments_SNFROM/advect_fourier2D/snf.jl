#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection2D(0.25f0, 0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

# # train
# latent = 2
# batchsize_ = (128 * 128) * 500 ÷ 4
# makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)
# train_params = (; E = 1400, wd = 128, α = 0f-0, γ = 1f-2, makedata_kws, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # modeldir/results
# postprocess_SNF(prob, datafile, modelfile; rng, device)

# # modeldir/hyper
# outdir = joinpath(modeldir, "hyper")
# hyper_reduction_path = joinpath(modeldir, "hyper.jld2")
# evolve_kw = (; hyper_reduction_path, verbose = false,)
# postprocess_SNF(prob, datafile, modelfile; rng, evolve_kw, outdir, device)

# modeldir/hyperDT
outdir = joinpath(modeldir, "hyperDT")
hyper_reduction_path = joinpath(modeldir, "hyperDT.jld2")
T, Nt = 4.0f0, 500
It = LinRange(1, Nt, 100) .|> Base.Fix1(round, Int)
data_kws = (; Ix = :, It = :)
evolve_kw = (; Δt = T, data_kws, hyper_reduction_path, adaptive = true,)
postprocess_SNF(prob, datafile, modelfile; rng, evolve_kw, outdir, device)

#======================================================#
nothing
