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
import Lux: cpu, gpu, relu

# misc
using Tullio, Zygote

using FFTW, LinearAlgebra
BLAS.set_num_threads(4)
FFTW.set_num_threads(8)

rng = Random.default_rng()
Random.seed!(rng, 983254)

N = 1024
E = 100 # epochs

# trajectories
_K = 1024
K_ = 128

# get data
dir = @__DIR__
filename = joinpath(dir, "1D_Burgers_Sols_Nu0.001.hdf5")
include(joinpath(dir, "pdebench.jl"))
_data, data_ = burgers1D(filename, _K, K_, rng)

_data = (reshape(_data[1][2, :, :], 1, N, :), _data[2]) # omit x-coordinate
data_ = (reshape(data_[1][2, :, :], 1, N, :), data_[2])

V = FourierSpace(N)

###
# FNO model
###

w = 32    # width
m = (128,) # modes
c = size(_data[1], 1) # in  channels
o = size(_data[2], 1) # out channels

NN = Lux.Chain(
    Dense(c, w,       relu),
    OpKernel(w, w, m, relu),
    OpKernel(w, w, m, relu),
    OpKernel(w, w, m, relu),
    OpKernel(w, w, m, relu),
    OpKernel(w, w, m, relu),
    OpKernel(w, w, m, relu),
    Dense(w, o),
)

opt = Optimisers.Adam()
batchsize = 32

learning_rates = (1f-2, 1f-3, 1f-4, 1f-5)
nepochs  = E .* (0.25, 0.25, 0.25, 0.25) .|> Int

# learning_rates = (1f-3, 5f-4, 2.5f-4, 1.25f-4)
# nepochs  = E .* (0.25, 0.25, 0.25, 0.25) .|> Int

dir = joinpath(@__DIR__, "dump")
device = Lux.gpu

model, ST = train_model(rng, NN, _data, data_, V, opt;
    batchsize, learning_rates, nepochs, dir, device)

plot_training(ST...)
#
