#
"""
Learn solution to diffusion equation

    -∇⋅ν₀∇ u = f

for constant ν₀, and variable f

test bed for Fourier Neural Operator experiments where
forcing is learned separately.
"""

using GeometryLearning

# PDE stack
using FourierSpaces, LinearAlgebra

# ML stack
using Lux, Random, Optimisers

# vis/analysis, serialization
using Plots, BSON

""" data """
function datagen(rng, V, K, f0 = nothing, discr = Collocation())

    N = size(V, 1)

    # constant forcing
    f0 = isnothing(f0) ? ones(Float32, N) : f0
    f0 = kron(f0, ones(K)')

    x, = points(V)
    x = kron(x, ones(K)')

    ν = 1 .+ 1 * rand(rng, Float32, N, K)

    @assert size(f0) == size(ν)

    V = make_transform(V, ν)
    F = transformOp(V)

    # rm high freq modes
    Tr = truncationOp(V, (0.5,))
    ν  = Tr * ν
    f0  = Tr * f0

    # true sol
    A = diffusionOp(ν, V, discr)
    u = A \ f0

    d0 = zeros(Float32, (2, N, K))
    d0[1, :, :] = ν
    d0[2, :, :] = x
    d1 = reshape(u, (1, N, K))

    data = (d0, d1)

    V, data
end

""" main program """

# parameters
N = 128    # problem size
K = 100    # X-samples
E = 500  # epochs

rng = Random.default_rng()
Random.seed!(rng, 917)

# datagen
V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
discr = Collocation()

f0 = 20 * rand(Float32, N)
_V, _data = datagen(rng, V, K, f0) # train
V_, data_ = datagen(rng, V, K, f0) # test

###
# FNO model
###

# w = 64
# c = size(_data[1], 1)
# o = size(_data[2], 1)
#
# NN = Lux.Chain(
#     Lux.Dense(c , w, tanh),
#     Lux.Dense(w , w, tanh),
#     Lux.Dense(w , w, tanh),
#     Lux.Dense(w , o),
# )


w = 32    # width
m = (16,) # modes
c = size(_data[1], 1) # in  channels
o = size(_data[2], 1) # out channels
# NN = Lux.Chain(Lux.Dense(c , w, tanh), OpKernel(w, w, m, Lux.tanh_fast), Lux.Dense(w , o)) # FNO
# NN = OpKernel(c, o, m)
NN = OpConv(c, o, m)

opt = Optimisers.Adam()
learning_rates = (1f-5, 1f-1, 1f-2, 1f-3, 1f-4)
maxiters  = E .* (0.05, 0.05, 0.10, 0.50, 0.30) .|> Int
dir = @__DIR__

p, st, _STATS = train_model(rng, NN, _data, data_, _V, opt;
                            maxiters, learning_rates, dir, cbstep = 1)

nothing
#
