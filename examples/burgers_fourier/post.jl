#
using LinearAlgebra, Lux, BSON

# load data
file = joinpath(@__DIR__, "visc_burg_re01k/data.bson")
data = BSON.load(file)
x = data[:x]
u = data[:u]

Nx, Nt, K = size(u)

Utrue = begin # normalize each trajectory
    u = reshape(u, Nx, 1, Nt * K)
    l = InstanceNorm(1; affine = false) # arg: channel size = 1
    p, st = Lux.setup(rng, l)
    u = l(u, p, st)[1]
end

# load model
file = joinpath(@__DIR__, "CAE_deep", "model.bson")
model = BSON.load(file)
_data = model[:_data]
data_ = model[:data_]
NN, p, st = model[:model]

Upred = NN(Utrue, p, st)[1]

Utrue = reshape(Utrue, Nx, Nt, K)
Upred = reshape(Utrue, Nx, Nt, K)

nothing
