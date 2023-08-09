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
E = 20

# trajectories
_K = 512
K_ = 64

# get data
dir = @__DIR__
filename = joinpath(dir, "1D_Burgers_Sols_Nu0.001.hdf5")
include(joinpath(dir, "pdebench.jl"))
_data, data_ = burgers1D(filename, _K, K_, rng)

_data = (reshape(_data[1][2, :, :], 1, N, :), _data[2]) # omit x-coordinate
data_ = (reshape(data_[1][2, :, :], 1, N, :), data_[2])

V = FourierSpace(N)

###
# Bilin FNO model
###

#============================#
w = 16     # width
l = 4
m = (128,) # modes
c = size(_data[1], 1) # in  channels
o = size(_data[2], 1) # out channels

root = Chain(
    Dense(c, w, relu),
    OpKernel(w, w, m, relu),
    OpKernel(w, w, m, relu),
)

branch = Chain(
    OpKernel(w, w, m, relu), # use_bias = true
    OpKernel(w, w, m, relu),
)

fuse = OpConvBilinear(w, w, l, m)

project = Chain(
    Dense(l, l, relu),
    Dense(l, o),
)

NN = Chain(
    root,
    BranchLayer(deepcopy(branch), deepcopy(branch)),
    fuse,
    project,
)

#============================#

opt = Optimisers.Adam()
batchsize = 64

learning_rates = (1f-2, 1f-3, 1f-4, 1f-5)
nepochs  = E .* (0.25, 0.25, 0.25, 0.25) .|> Int

# learning_rates = (1f-3, 5f-4, 2.5f-4, 1.25f-4)
# nepochs  = E .* (0.25, 0.25, 0.25, 0.25) .|> Int

dir = joinpath(@__DIR__, "model_burgers1D_nu0.001_bilinear")
device = Lux.cpu

model, ST = train_model(rng, NN, _data, data_, V, opt;
    batchsize, learning_rates, nepochs, dir, device)

plot_training(ST...)
#
