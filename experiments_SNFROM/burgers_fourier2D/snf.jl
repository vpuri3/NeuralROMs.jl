#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = BurgersViscous2D(1f-3)
datafile = joinpath(@__DIR__, "data_burgers2D/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_07.jld2")
device = Lux.gpu_device()

# # train
grid = (512, 512,)
# _batchsize = prod(grid) * length(_It) ÷ 500
# batchsize_ = prod(grid) * length(_It) ÷ 500
#
# makedata_kws = (; Ix = :, _Ib = [1], Ib_ = [1], _It = :, It_ = :)
#
# latent = 2
# train_params = (; E = 700, wd = 128, γ = 1f-2, makedata_kws, _batchsize, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # train
# latent = 2
# batchsize_ = (128 * 128) * 500 ÷ 4
# makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)
# train_params = (; E = 1400, wd = 128, α = 0f-0, γ = 1f-2, makedata_kws, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # modeldir/results
# postprocess_SNF(prob, datafile, modelfile; rng, device)

ids = zeros(Bool, grid...)
@views ids[1:32:end, 1:32:end] .= true
hyper_indices = findall(isone, vec(ids))

# # modeldir/hyper
# outdir = joinpath(modeldir, "hyper")
# hyper_reduction_path = joinpath(modeldir, "hyper.jld2")
# evolve_kw = (; hyper_reduction_path, hyper_indices, verbose = false,)
# postprocess_SNF(prob, datafile, modelfile; rng, evolve_kw, outdir, device)

# # modeldir/hyperDT
outdir = joinpath(modeldir, "hyperDT")
hyper_reduction_path = joinpath(modeldir, "hyperDT.jld2")

# It = LinRange(1, 500, 50) .|> Base.Fix1(round, Int)
It = LinRange(1, 500, 100) .|> Base.Fix1(round, Int)
data_kws = (; Ix = :, It)
evolve_kw = (; Δt = 10f0, data_kws, hyper_reduction_path, hyper_indices, verbose = false,)
postprocess_SNF(prob, datafile, modelfile; rng, evolve_kw, outdir, device)

#======================================================#
nothing
