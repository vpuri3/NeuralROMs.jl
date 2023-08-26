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
BLAS.set_num_threads(12)
FFTW.set_num_threads(24)

include("../datagen.jl")

# parameters
N  = 128  # problem size
K1 = 32   # ν-samples
K2 = 32   # f-samples
E  = 100  # epochs

rng = Random.default_rng()
Random.seed!(rng, 199)

# datagen
_V, _data, _, _ = datagen1D(rng, N, K1, K2) # train
V_, data_, _, _ = datagen1D(rng, N, K1, K2) # train
# V_, data_, _, _ = datagen1D(rng, N, K1, K2; mode = :test) # test

__data = combine_data1D(_data)
data__ = combine_data1D(data_)

###
# FNO model
###
if false

w = 16    # width
m = (32,) # modes
c = size(__data[1], 1) # in  channels
o = size(__data[2], 1) # out channels

NN = Lux.Chain(
    PermutedBatchNorm(c, 3),
    Dense(c , w, Lux.relu),
    OpKernel(w, w, m, Lux.relu),
    OpKernel(w, w, m, Lux.relu),
    OpKernel(w, w, m, Lux.relu),
    Dense(w , o)
)

opt = Optimisers.Adam()
batchsize = 128 #size(__data[1])[end]
learning_rates = (1f-2, 1f-3,)
nepochs  = E .* (0.10, 0.90,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_nonlin")
device = Lux.gpu

FNO_nonlin = train_model(rng, NN, __data, data__, _V, opt;
        batchsize, learning_rates, nepochs, dir, cbstep = 1, device)

end

###
# Bilinear (linear / nonlin) model
###

if true

# fixed params
c1 = 2     # in  channel nonlin
c2 = 1     # in  channel linear
o  = size(__data[2], 1) # out channel

# hyper params
w1 = 16   # width nonlin
w2 = 16   # width linear
wo = 8    # width project
m = (32,) # modes

split  = SplitRows(1:2, 3)
nonlin = Chain(PermutedBatchNorm(c1, 3), Dense(c1, w1, tanh), OpKernel(w1, w1, m, tanh))
linear = Dense(c2, w2, use_bias = false)
bilin  = OpConvBilinear(w1, w2, o, m)
# bilin  = OpKernelBilinear(w1, w2, o, m) # errors

NN = linear_nonlinear(split, nonlin, linear, bilin)

opt = Optimisers.Adam()
batchsize = 128 # size(__data[1])[end] # 1024
learning_rates = (1f-3,)
nepochs = E .* (1.00,) .|> Int
# learning_rates = (1f-3, 5f-4, 2.5f-4, 1.25f-4,)
# nepochs        = E .* (0.25, 0.25, 0.25, 0.25,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_linear_nonlinear")
device = Lux.gpu

model, ST = train_model(rng, NN, __data, data__, _V, opt;
        batchsize, learning_rates, nepochs, dir, cbstep = 1, device)

end

plot_training(ST...) |> display

nothing
#
