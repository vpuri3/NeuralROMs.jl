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
using LinearAlgebra, FourierSpaces

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

# parameters
N = 32   # problem size
K = 1024 # batch size
E = 300  # epochs

V = FourierSpace(N, N)

# get data
include("../datagen.jl")
BSON.@load joinpath(@__DIR__, "..", "data2D_N$(N).bson") _data data_

rng = Random.default_rng()
Random.seed!(rng, 201)

###
# FNO model
###
if true

__data = combine_data2D(_data, K)
data__ = combine_data2D(data_, K)

w = 12        # width
m = (16, 16,) # modes
c = size(__data[1], 1) # in  channels
o = size(__data[2], 1) # out channels

NN = Lux.Chain(
    PermutedBatchNorm(c, 4),
    Dense(c , w, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    Dense(w , o)
)

opt = Optimisers.Adam()
learning_rates = (1f-2, 1f-3,)
maxiters  = E .* (0.10, 0.90,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_nonlin")
device = Lux.gpu

FNO_nonlin = train_model(rng, NN, __data, data__, V, opt;
        learning_rates, maxiters, dir, cbstep = 1, device)

end

###
# Bilinear (linear / nonlin) model
###

if true

__data = split_data2D(_data)
data__ = split_data2D(data_)

w1 = 16    # width nonlin
w2 = 16    # width linear
m = (16, 16,) # modes
c1 = size(__data[1][1], 1) # in  channel nonlin
c2 = size(__data[1][2], 1) # in  channel linear
o  = size(__data[2]   , 1) # out channel

nonlin = Chain(
        PermutedBatchNorm(c1, 4),
        Dense(c1, w1, tanh),
        OpKernel(w1, w1, m, tanh),
        OpKernel(w1, w1, m, tanh),
    )
linear = Chain(
        Dense(c2, w2, use_bias = false),
        OpKernel(w2, w2, m),
        # OpKernel(w2, w2, m),
    )
# linear = Dense(c2, w2, use_bias = false)
bilin  = OpConvBilinear(w1, w2, o, m)
# bilin  = OpKernelBilinear(w1, w2, o, m) # errors
# bilin  = Bilinear((w1, w2) => o)

NN = linear_nonlinear(nonlin, linear, bilin)

opt = Optimisers.Adam()
learning_rates = (1f-3,)
maxiters  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_linear_nonlinear")
device = Lux.gpu

model, _ = train_model(rng, NN, __data, data__, V, opt;
        learning_rates, maxiters, dir, cbstep = 1, device)

end

nothing
#
