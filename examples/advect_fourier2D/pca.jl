#
using GeometryLearning

include(joinpath(pkgdir(GeometryLearning), "examples", "PCA.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

R = 8
prob = Advection2D(0.25f0, 0.00f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir = joinpath(@__DIR__, "PCA$(R)")
modelfile = joinpath(modeldir, "model.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1], Ib_ = [1], _It = :, It_ = :)

train_PCA(datafile, modeldir, R; makedata_kws, makeplot = false, device,)
x, t, ud, up, _ = evolve_PCA(prob, datafile, modelfile, 1; rng, device,)

Nx, Ny = 128, 128
Nt = 1000

p1 = heatmap(reshape(up[1,:,1 ], Nx, Ny))
p2 = heatmap(reshape(up[1,:,Nt], Nx, Ny))

n  = sum(abs2, ud) / length(ud)
e  = (up - ud) / n # (X, T)
et = sum(abs2, e; dims = (1,2)) / size(e, 2) |> vec
@show sqrt(mse(up, ud) / mse(ud, 0 * ud))
@show norm(up - ud, Inf) / sqrt(mse(ud, 0 * ud))

p3 = plot(t, et; w = 4, yaxis = :log, ylims = (10^-9, 1.0))
display(p3)
plot(p1, p2, p3) |> display
#======================================================#
nothing
