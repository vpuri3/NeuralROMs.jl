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

dir = joinpath(@__DIR__, "model")
datadir = joinpath(@__DIR__, "burg_visc_re10k/data.bson")

# get data
V, _data, data_, metadata = begin
    data = BSON.load(datadir)

    u = data[:u] # [Nx, Nb, Nt]
    mu = data[:mu] # [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    # get sizes
    Nx, Nb, Nt = size(u)

    Nx = Int(Nx / 8)
    Ix = 1:8:8192

    # normalize arrays
    mean = sum(u) / length(u)
    var  = sum(abs2, u .- mean) / length(u)
    u    = (u .- mean) / sqrt(var)

    # train/test split
    _Ib, Ib_ = splitobs(1:Nb; at = 0.5, shuffle = true)
    # _Ib, Ib_ = [4, 6], [1, 2, 3, 5, 7]

    # train on times 0.0 - 0.5s
    _It = Colon() # 1:1:Int(Nt/2) |> Array
    It_ = Colon() # 1:2:Nt        |> Array

    _u = @view u[Ix, _Ib, _It]
    u_ = @view u[Ix, Ib_, It_]

    _u = reshape(_u, Nx, 1, :)
    u_ = reshape(u_, Nx, 1, :)

    V = nothing # FourierSpace(N)

    readme = "Train/test on 0.0-0.5."

    metadata = (; mean, var, _Ib, Ib_, _It, readme)

    V, (_u, _u), (u_, u_), metadata
end

# parameters
E = 200 # epochs
w = 64  # width
l = 16  # latent
act = tanh # relu

opt = Optimisers.Adam()
batchsize  = 50
batchsize_ = 50
learning_rates = 1f-3 ./ (2 .^ (0:9))
nepochs = E/10 * ones(10) .|> Int
device = Lux.gpu_device()

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
    batchsize, batchsize_, learning_rates, nepochs, dir, device, metadata)

plot_training(ST...) |> display

nothing
#
