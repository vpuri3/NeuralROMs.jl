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
BLAS.set_num_threads(12)
FFTW.set_num_threads(24)

rng = Random.default_rng()
Random.seed!(rng, 345)

N = 128
E = 100

# trajectories
_K = 512
K_ = 128

# get data
dir = @__DIR__
filename = joinpath(dir, "2D_DarcyFlow_beta0.01_Train.hdf5")
include(joinpath(dir, "pdebench.jl"))
_data, data_ = darcy2D(filename, _K, K_, rng)

V = FourierSpace(N, N)

###
# FNO model
###

w = 64        # width
m = (32, 32,) # modes
c = size(_data[1], 1) # in  channels
o = size(_data[2], 1) # out channels
act = Lux.relu

NN = Lux.Chain(
    Dense(c, w, act),
    OpKernel(w, w, m, act),
    OpKernel(w, w, m, act),
    OpKernel(w, w, m, act),
    OpKernel(w, w, m, act),
    Dense(w, o),
)

opt = Optimisers.Adam()
batchsize = 32
learning_rates = (1f-2, 1f-3, 5f-4, 2.5f-4,)
nepochs  = E .* (0.25, 0.25, 0.25, 0.25,) .|> Int
dir = joinpath(@__DIR__, "model_darcy2D")
device = Lux.gpu

model, ST = train_model(rng, NN, _data, data_, V, opt;
    batchsize, learning_rates, nepochs, dir, device)

plot_training(ST...) |> display

nothing
#
