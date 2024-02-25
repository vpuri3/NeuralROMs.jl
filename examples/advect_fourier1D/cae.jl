#
#======================================================#
using GeometryLearning
begin
    path = joinpath(pkgdir(GeometryLearning), "examples", "convAE.jl")
    include(path)
end

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

# parameters
E = 700    # epochs
l = 1      # latent
w = 32     # width
act = tanh # relu, tanh

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "CAE")
modelfile = joinpath(modeldir, "model_07.jld2")
outdir    = joinpath(modeldir, "results")
device = Lux.gpu_device()

NN = cae_network(prob, l, w, act)

p, st = Lux.setup(rng, NN)
p = ComponentArray(p)
_data, _, _ = makedata_CAE(datafile)
@show _data[1] |> size
@show NN(_data[1], p, st)[1] |> size
@show length(p.encoder)
@show length(p.decoder)

# ## train
# isdir(modeldir) && rm(modeldir, recursive = true)
# model, ST, metadata = train_CAE(datafile, modeldir, NN, E; rng, warmup = false, device)

# ## evolve
x, u, p = evolve_CAE(prob, datafile, modelfile, outdir; rng, device)

#======================================================#
nothing
#
