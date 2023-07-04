#
"""
Learn solution to diffusion equation

    -∇⋅ν∇u = f

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

include("../datagen.jl")

""" data """
function combine_data(data)
    x, ν, f, u = data

    N, K = size(x)

    x1 = zeros(3, N, K) # x, ν, f
    y  = zeros(1, N, K) # u

    x1[1, :, :] = x
    x1[2, :, :] = ν
    x1[3, :, :] = f

    y[1, :, :] = u

    (x1, y)
end

function split_data(data)
    x, ν, f, u = data

    N, K = size(x)

    x1 = zeros(2, N, K) # ν, x
    x2 = zeros(1, N, K) # f
    y  = zeros(1, N, K) # u

    x1[1, :, :] = x
    x1[2, :, :] = ν

    x2[1, :, :] = f

    y[1, :, :] = u

    ((x1, x2), y)
end

""" main program """

# parameters
N  = 128  # problem size
K1 = 50   # X-samples
K2 = 50   # X-samples
E  = 200  # epochs

rng = Random.default_rng()
Random.seed!(rng, 917)

# datagen
_V, _data, _, _ = datagen(rng, N, K1, K2) # train
V_, data_, _, _ = datagen(rng, N, K1, K2) # test

###
# FNO model - works very well
###

_data = combine_data(_data)
data_ = combine_data(data_)

w = 16    # width
m = (32,) # modes
c = size(_data[1], 1) # in  channels
o = size(_data[2], 1) # out channels

NN = Lux.Chain(
    # lifting
    PermutedBatchNorm(c, 3),
    Dense(c , w, Lux.tanh_fast),

    # FNO
    OpKernel(w, w, m, Lux.tanh_fast),
    OpKernel(w, w, m, Lux.tanh_fast),

    # projection
    Dense(w, w, Lux.tanh_fast),
    Dense(w , o)
)

###
# Bilinear model
###

# _data = split_data(_data)
# data_ = split_data(data_)

# w = 16    # width
# m = (32,) # modes
# c1 = size(_data[1][1], 1) # in  channel 1
# c2 = size(_data[1][2], 1) # in  channel 2
# o  = size(_data[2]   , 1) # out channel

# # nonlin = OpKernel(c1, w, m, Lux.tanh_fast)
# nonlin = Chain(Dense(c1, w, Lux.tanh_fast), OpKernel(w, w, m, Lux.tanh_fast))
# # nonlin = Chain(Dense(c1, w, tanh), Dense(w, w, tanh))
# linear = Dense(c2, w)
# bilin  = OpConvBilinear(w, w, o, m)

# NN = linear_nonlinear(nonlin, linear, bilin)

opt = Optimisers.Adam()
learning_rates = (1f-1, 1f-2, 1f-3, 1f-4)
maxiters  = E .* (0.10, 0.20, 0.50, 0.20) .|> Int

dir = @__DIR__

p, st, STATS = train_model(rng, NN, _data, data_, _V, opt;
                           learning_rates, maxiters, dir, cbstep = 1)

nothing
#
