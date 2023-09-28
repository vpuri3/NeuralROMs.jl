#
"""
Train an autoencoder on 1D Burgers data
"""

using GeometryLearning

# PDE stack
using LinearAlgebra, FourierSpaces

# ML stack
using Lux, Random, Optimisers, MLUtils

# vis/analysis, serialization
using Plots, BSON

# accelerator
using CUDA, LuxCUDA #, KernelAbstractions
CUDA.allowscalar(false)
import Lux: cpu, gpu

# misc
using Tullio, Zygote

using FFTW, LinearAlgebra
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    FFTW.set_num_threads(nt)
end

rng = Random.default_rng()
Random.seed!(rng, 199)

# datagen
x, V, _data, data_ = begin
    file = joinpath(@__DIR__, "visc_burg_re01k/data.bson")
    data = BSON.load(file)
    x = data[:x]
    u = data[:u] # [Nx, Nt, K]

    # skip every other timestep
    u = @view u[:, :, begin:4:end]

    N = size(u, 1)
    u = reshape(u, N, 1, :)

    # normalize each trajectory
    # https://lux.csail.mit.edu/dev/api/Building_Blocks/LuxLib#Normalization
    l = InstanceNorm(1; affine = false) # arg: channel size = 1
    p, st = Lux.setup(rng, l)
    u = l(u, p, st)[1]

    _u, u_ = splitobs(u; at = 0.8, shuffle = true)

    V = nothing #FourierSpace(N)

    x, V, (_u, _u), (u_, u_)
end

# parameters
if false
    E = 300 # epochs
    w = 128 # width
    l = 64  # latent
    act = tanh # relu

    opt = Optimisers.Adam()
    batchsize  = 100
    batchsize_ = 200
    learning_rates = 1f-3 ./ (2 .^ (0:9))
    nepochs = E/10 * ones(10) .|> Int
    device = Lux.gpu_device()
    dir = joinpath(@__DIR__, "CAE_deep")

    NN = begin
        encoder = Chain(
            Conv((2,), 1  =>  8, act; stride = 2),
            Conv((2,), 8  => 16, act; stride = 2),
            Conv((2,), 16 => 32, act; stride = 2),
            Conv((2,), 32 =>  w, act; stride = 2),
            # BatchNorm(w),
            Conv((2,), w  =>  w, act; stride = 2),
            Conv((2,), w  =>  w, act; stride = 2),
            Conv((2,), w  =>  w, act; stride = 2),
            Conv((2,), w  =>  w, act; stride = 2),
            Conv((2,), w  =>  w, act; stride = 2),
            Conv((2,), w  =>  w, act; stride = 2),
            flatten,
            Dense(w, w, act),
            Dense(w, l),
        )

        decoder = Chain(
            Dense(l, w, act),
            Dense(w, w, act),
            ReshapeLayer((1, w)),
            ConvTranspose((2,), w  =>  w, act; stride = 2),
            ConvTranspose((2,), w  =>  w, act; stride = 2),
            ConvTranspose((2,), w  =>  w, act; stride = 2),
            ConvTranspose((2,), w  =>  w, act; stride = 2),
            ConvTranspose((2,), w  =>  w, act; stride = 2),
            ConvTranspose((2,), w  =>  w, act; stride = 2),
            # BatchNorm(w),
            ConvTranspose((2,), w  => 32, act; stride = 2),
            ConvTranspose((2,), 32 => 16, act; stride = 2),
            ConvTranspose((2,), 16 =>  8, act; stride = 2),
            ConvTranspose((2,), 8  =>  1     ; stride = 2),
        )

        Chain(encoder, decoder)
    end

    model, ST = train_model(rng, NN, _data, data_, V, opt;
        batchsize, batchsize_, learning_rates, nepochs, dir,
        cbstep = 1, device)

    plot_training(ST...) |> display
end

if true
    E = 200 # epochs
    w = 64 # width
    l = 64 # latent
    act = tanh # relu

    opt = Optimisers.Adam()
    batchsize  = 100
    batchsize_ = 200
    learning_rates = 1f-3 ./ (2 .^ (0:9))
    nepochs = E/10 * ones(10) .|> Int
    device = Lux.gpu_device()
    dir = joinpath(@__DIR__, "CAE_wide")

    NN = begin
        encoder = Chain(
            Conv((9,), 1 => w, act; stride = 5, pad = 0),
            Conv((9,), w => w, act; stride = 5, pad = 0),
            # BatchNorm(w, act),
            Conv((9,), w => w, act; stride = 5, pad = 0),
            Conv((7,), w => w, act; stride = 1, pad = 0),
            flatten,
            # Dense(w, w, act),
            Dense(w, l, act),
        )

        decoder = Chain(
            Dense(l, w, act),
            # Dense(w, w, act),
            ReshapeLayer((1, w)),
            ConvTranspose((7,), w => w, act; stride = 1, pad = 0),
            ConvTranspose((10,),w => w, act; stride = 5, pad = 0),
            # BatchNorm(w, act),
            ConvTranspose((9,), w => w, act; stride = 5, pad = 0),
            ConvTranspose((9,), w => 1,    ; stride = 5, pad = 0),
        )

        Chain(encoder, decoder)
    end

    model, ST = train_model(rng, NN, _data, data_, V, opt;
        batchsize, batchsize_, learning_rates, nepochs, dir,
        cbstep = 1, device)
    
    plot_training(ST...) |> display
end

nothing

    #=
# DAE paper
# 4 conv layers, 1 deep layer
# filter = (25,), stride = 2 (first), 4 (after)

# NN = Chain(...)
# u = rand(Float32, 1024, 1, 10)
# # u = rand(Float32, 1, 1, 10)
# p, st = Lux.setup(rng, NN)
# @show u |> size
# @show NN(u, p, st)[1] |> size

    =#
#
