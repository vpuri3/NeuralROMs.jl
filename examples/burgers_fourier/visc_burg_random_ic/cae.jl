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

dir = joinpath(@__DIR__, "CAE_stationary")
datadir = joinpath(@__DIR__, "burg_visc_re10k_stationary/data.bson")

# dir = joinpath(@__DIR__, "CAE_traveling")
# datadir = joinpath(@__DIR__, "burg_visc_re10k_traveling/data.bson")

# get data
V, _data, data_, metadata = begin
    data = BSON.load(datadir)

    u = data[:u] # [Nx, Nb, Nt]

    # get sizes
    Nx, Nt, Nb = size(u)

    # normalize arrays
    mean = sum(u) / length(u)
    var  = sum(abs2, u .- mean) / length(u)
    u    = (u .- mean) / sqrt(var)

    # train/test split
    _Ib, Ib_ = splitobs(1:Nb; at = 0.8, shuffle = true)

    # train on times 0 - 5s
    _It = 1:2:Int(Nt/2) |> Array
    It_ = 1:4:Nt        |> Array

    _u = @view u[:, _Ib, _It]
    u_ = @view u[:, Ib_, It_]

    _u = reshape(_u, Nx, 1, :)
    u_ = reshape(u_, Nx, 1, :)

    V = nothing # FourierSpace(N)

    readme = "Train from time 0-5, test on 0-10."

    metadata = (; mean, var, _Ib, Ib_, _It, readme)

    V, (_u, _u), (u_, u_), metadata
end

# parameters
E = 200 # epochs
w = 128 # width
l = 64  # latent
act = tanh # relu

opt = Optimisers.Adam()
batchsize  = 100
batchsize_ = 200
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
