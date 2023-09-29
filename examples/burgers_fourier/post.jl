#
using LinearAlgebra, Lux, BSON

# load data
datafile = joinpath(@__DIR__, "burg_visc_re10k_traveling", "data.bson")
data = BSON.load(datafile)
x = data[:x]
Udata = data[:u]

Nx, Nb, Nt = size(Udata)

# load model
modelfile = joinpath(@__DIR__, "CAE_traveling", "model.bson")
model = BSON.load(modelfile)
NN, p, st = model[:model]
md = model[:metadata] # (; mean, var, _Ib, Ib_, It)

Unorm = (Udata .- md.mean) / sqrt(md.var)
Unorm = reshape(Unorm, Nx, 1, :)

Upred = NN(Unorm, p, st)[1]
Upred = Upred * sqrt(md.var) .+ md.mean

Upred = reshape(Upred, Nx, Nb, Nt)

_Ib = md._Ib
Ib_ = md.Ib_

_Udata = @view Udata[:, _Ib, :]
Udata_ = @view Udata[:, Ib_, :]

# compare w. unseen at time

nothing
