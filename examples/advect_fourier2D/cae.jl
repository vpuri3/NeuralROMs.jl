#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "examples", "convAE.jl"))

#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

# parameters
E = 700    # epochs
l = 2      # latent
w = 32     # width
act = tanh # relu, tanh

prob = Advection2D(0.25f0, 0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "CAE")
modelfile = joinpath(modeldir, "model_07.jld2")
outdir    = joinpath(modeldir, "results")
device = Lux.gpu_device()

## train
# NN = cae_network(prob, l, w, act)
# isdir(modeldir) && rm(modeldir, recursive = true)
# model, ST, metadata = train_CAE(datafile, modeldir, NN, E; rng, warmup = false, device)

# evolve
x, t, ud, up, ps = evolve_CAE(prob, datafile, modelfile, 1; rng,)

Nx, Ny = 128, 128
Nt = 1000

p1 = heatmap(reshape(ud[1, :, 1 ], Nx, Ny))
p2 = heatmap(reshape(up[1, :, Nt], Nx, Ny))

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
#
