using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "PCA.jl"))

#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

R = 8
prob = Advection1D(0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir = joinpath(@__DIR__, "PCA$(R)")
modelfile = joinpath(modeldir, "model.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1], Ib_ = [1], _It = :, It_ = :)

train_PCA(datafile, modeldir, R; makedata_kws, makeplot = false, device,)
x, t, ud, up, _ = evolve_PCA(prob, datafile, modelfile, 1; rng, device,)
plt = plot(vec(x), ud[1,:, [1, 250, 500]]; w = 3, c = :black)
plot!(plt, vec(x), up[1,:, [1, 250, 500]]; w = 3) |> display
#======================================================#
nothing
