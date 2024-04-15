#
"""
Learn solution to diffusion equation

    -∇⋅ν∇u = f

for constant ν₀, and variable f

test bed for Fourier Neural Operator experiments where
forcing is learned separately.
"""

using NeuralROMs

# PDE stack
using LinearAlgebra

# ML stack
using Lux, Random, Optimisers

# vis/analysis, serialization
using Plots, BSON

# accelerator
using CUDA, KernelAbstractions
CUDA.allowscalar(false)
import Lux: cpu, gpu

# misc
using Tullio, Zygote

using FFTW, LinearAlgebra
BLAS.set_num_threads(2)
FFTW.set_num_threads(8)

include("../datagen.jl")

# parameters
N  = 128  # problem size
K1 = 32   # ν-samples
K2 = 32   # f-samples
E  = 200  # epochs

rng = Random.default_rng()
Random.seed!(rng, 117)

# datagen
_V, _data, _, _ = datagen(rng, N, K1, K2) # train
V_, data_, _, _ = datagen(rng, N, K1, K2) # test

###
# nonlienar FNO model
###
if true

__data = combine_data(_data)
data__ = combine_data(data_)

w = 16    # width
m = (32,) # modes
c = size(__data[1], 1) # in  channels
o = size(__data[2], 1) # out channels

NN = Lux.Chain(
    PermutedBatchNorm(c, 3),
    Dense(c , w, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    Dense(w , o)
)

opt = Optimisers.Adam()
learning_rates = (1f-3,)
maxiters  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "dump")

model, _ = train_model(rng, NN, __data, data__, _V, opt;
               learning_rates, maxiters, dir, cbstep = 1, device = gpu, make_plots = false)
end

###
# Bilinear (linear / nonlin) model
###

if true

__data = split_data(_data)
data__ = split_data(data_)

w1 = 16    # width nonlin
w2 = 16    # width linear
m = (32,) # modes
c1 = size(__data[1][1], 1) # in  channel nonlin
c2 = size(__data[1][2], 1) # in  channel linear
o  = size(__data[2]   , 1) # out channel

nonlin = Chain(PermutedBatchNorm(c1, 3), Dense(c1, w1, tanh), OpKernel(w1, w1, m, tanh))
linear = Dense(c2, w2, use_bias = false)
bilin  = OpConvBilinear(w1, w2, o, m)

NN = linear_nonlinear(nonlin, linear, bilin)

opt = Optimisers.Adam()
learning_rates = (1f-3,)
maxiters  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "dump")

model, _ = train_model(rng, NN, __data, data__, _V, opt;
        learning_rates, maxiters, dir, cbstep = 1, device = gpu, make_plots = false)
end

nothing
#
