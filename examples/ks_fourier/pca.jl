#
using GeometryLearning

include(joinpath(pkgdir(GeometryLearning), "examples", "PCA.jl"))
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

R = 4
prob = KuramotoSivashinsky1D(0.01f0)
datafile = joinpath(@__DIR__, "data_ks/", "data.jld2")
modeldir = joinpath(@__DIR__, "PCA$(R)")
modelfile = joinpath(modeldir, "model.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1], Ib_ = [1], _It = :, It_ = :)

train_PCA(datafile, modeldir, R; makedata_kws, makeplot = false, device,)
x, t, ud, up, _ = evolve_PCA(prob, datafile, modelfile, 1; rng, device,)
p1 = plot(vec(x), ud[1,:, [1, 250, 500, 750, 1000]]; w = 3, c = :black)
plot!(p1, vec(x), up[1,:, [1, 250, 500, 750, 1000]]; w = 3) |> display

n  = sum(abs2, ud) / length(ud)
e  = (up - ud) / n # (X, T)
et = sum(abs2, e; dims = (1,2)) / size(e, 2) |> vec
p2 = plot(t, et; w = 4, yaxis = :log, ylims = (10^-9, 1.0))
display(p2)
@show sqrt(mse(up, ud) / mse(ud, 0 * ud))
@show norm(up - ud, Inf) / sqrt(mse(ud, 0 * ud))
plot(p1, p2) |> display
#======================================================#
nothing
