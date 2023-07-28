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
# N = 32  # problem size
# E = 300 # epochs

N = 64  # problem size
E = 200 # epochs

V = FourierSpace(N, N)

# get data
include("../datagen.jl")
BSON.@load joinpath(@__DIR__, "..", "data2D_N$(N).bson") _data data_

rng = Random.default_rng()
Random.seed!(rng, 345)

__data = combine_data2D(_data)
data__ = combine_data2D(data_)

###
# FNO model
###
if true

w = 12        # width
m = (12, 12,) # modes
c = size(__data[1], 1) # in  channels
o = size(__data[2], 1) # out channels

NN = Lux.Chain(
    Dense(c , w, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    Dense(w , o)
)

opt = Optimisers.Adam()
batchsize = 128
learning_rates = (1f-2, 1f-3,)
nepochs  = E .* (0.10, 0.90,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_nonlin")
device = Lux.gpu

FNO_nl = train_model(rng, NN, __data, data__, V, opt;
        batchsize, learning_rates, nepochs, dir, device)

end

###
# Bilinear (linear / nonlin) model
###

if false

# fixed params
c1 = 3     # in  channel nonlin
c2 = 1     # in  channel linear
o  = size(__data[2]   , 1) # out channel

# hyper params
w1 = 8       # width nonlin
w2 = 8       # width linear
wo = 8        # width project
m = (12, 12,) # modes

split = SplitRows(1:3, 4)
nonlin = Chain(
        Dense(c1, w1, tanh),
        OpKernel(w1, w1, m, tanh),
        OpKernel(w1, w1, m, tanh),
    )
linear = Chain(
        Dense(c2, w2, use_bias = false),
        OpConv(w2, w2, m)
    )
bilin  = OpConvBilinear(w1, w2, wo, m)
# bilin  = OpKernelBilinear(w1, w2, o, m) # errors

project = Dense(wo, o, use_bias = false)

NN = linear_nonlinear(split, nonlin, linear, bilin, project)

opt = Optimisers.Adam()
batchsize = 128
learning_rates = (1f-3,)
nepochs  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_linear_nonlinear")
device = Lux.gpu

FNO_bl, _ = train_model(rng, NN, __data, data__, V, opt;
        batchsize, learning_rates, nepochs, dir, device)

end

nothing
#
