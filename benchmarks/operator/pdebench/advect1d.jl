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
BLAS.set_num_threads(4)
FFTW.set_num_threads(8)

rng = Random.default_rng()
Random.seed!(rng, 345)

N = 1024
E = 10 # epochs

# trajectories
_K = 4196
K_ = 512

# get data
dir = @__DIR__
filename = joinpath(dir, "1D_Advection_Sols_beta1.0.hdf5")
include(joinpath(dir, "pdebench.jl"))
_data, data_ = advect1D(filename, _K, K_, rng)

V = FourierSpace(N)

###
# FNO model
###

w = 32    # width
m = (16,) # modes
c = size(_data[1], 1) # in  channels
o = size(_data[2], 1) # out channels

NN = Lux.Chain( # todo - double check parameter size
    Dense(c, w, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    Dense(w, o),
)

opt = Optimisers.Adam()
batchsize = 32
learning_rates = (1f-2, 1f-3,)
nepochs  = E .* (0.10, 0.90,) .|> Int
dir = joinpath(@__DIR__, "model_advect1D")
device = Lux.cpu

model, ST = train_model(rng, NN, _data, data_, V, opt;
    batchsize, learning_rates, nepochs, dir, device)

nothing
#
