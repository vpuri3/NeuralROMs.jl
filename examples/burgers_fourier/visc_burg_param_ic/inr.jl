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
using CUDA, LuxCUDA, KernelAbstractions
CUDA.allowscalar(false)

# tensor prod
using Tullio

# AD
using Zygote, ChainRulesCore

using FFTW, LinearAlgebra
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    FFTW.set_num_threads(nt)
end

rng = Random.default_rng()
Random.seed!(rng, 199)

dir = joinpath(@__DIR__, "model_inr")
datadir = joinpath(@__DIR__, "burg_visc_re10k/data.bson")

# get data
V, _data, data_, metadata = begin
    data = BSON.load(datadir)

    x = data[:x]
    u = data[:u] # [Nx, Nb, Nt]
    mu = data[:mu] # [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    # get sizes
    Nx, Nb, Nt = size(u)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192

    # normalize solution
    ū  = sum(u) / length(u)
    σu = sum(abs2, u .- ū) / length(u)
    u  = (u .- ū) / sqrt(σu)

    # normalize space
    x̄  = sum(x) / length(x)
    σx = sum(abs2, x .- x̄) / length(x)
    x  = (x .- x̄) / sqrt(σx)

    # train/test trajectory split
    _Ib, Ib_ = splitobs(1:Nb; at = 0.5, shuffle = true)
    # _Ib, Ib_ = [4, 6], [1, 2, 3, 5, 7]
    _Ib, Ib_ = [1, 4, 7], [2, 3, 5, 6]

    # train on times 0.0 - 0.5s
    _It = Colon() # 1:1:Int(Nt/2) |> Array
    It_ = Colon() # 1:2:Nt        |> Array

    x = @view x[Ix]

    _u = @view u[Ix, _Ib, _It]
    u_ = @view u[Ix, Ib_, It_]

    _u = reshape(_u, Nx, 1, :)
    u_ = reshape(u_, Nx, 1, :)

    _Ns = size(_u, 3) # number of snapshots
    Ns_ = size(u_, 3)

    _x = zeros(Float32, Nx, 2, _Ns)
    x_ = zeros(Float32, Nx, 2, Ns_)

    _x[:, 1, :] = _u
    x_[:, 1, :] = u_

    _x[:, 2, :] .= x
    x_[:, 2, :] .= x

    V = nothing # FourierSpace(N)

    readme = "Train/test on 0.0-0.5."

    metadata = (; ū, σu, x̄, σx, _Ib, Ib_, _It, readme)

    V, (_x, _u), (x_, u_), metadata
end

# parameters
N = size(_data[1], 1)
E = 400 # epochs
we = 64 # width
wd = 32 # width
l  = 4  # latent
act = relu # relu

opt = Optimisers.Adam()
batchsize  = 20
batchsize_ = 50
learning_rates = 1f-3 ./ (2 .^ (0:9))
nepochs = E/10 * ones(10) .|> Int
device = Lux.gpu_device()

NN = begin
    encoder = Chain(
        Conv((2,), 1  =>  8, act; stride = 2),
        Conv((2,), 8  => 16, act; stride = 2),
        Conv((2,), 16 => 32, act; stride = 2),
        Conv((2,), 32 => we, act; stride = 2),
        Conv((2,), we => we, act; stride = 2),
        Conv((2,), we => we, act; stride = 2),
        Conv((2,), we => we, act; stride = 2),
        Conv((2,), we => we, act; stride = 2),
        Conv((2,), we => we, act; stride = 2),
        Conv((2,), we => we, act; stride = 2),
        FlattenLayer(),
        Dense(we, we, act),
        Dense(we, l)
        ;name = :encoder
    )

    decoder = Chain(
        Dense(l+1, wd,sin; init_weight = scaled_siren_init(10), init_bias = rand),
        Dense(wd, wd, sin; init_weight = init_siren, init_bias = rand),
        Dense(wd, wd, sin; init_weight = init_siren, init_bias = rand),
        Dense(wd, wd, sin; init_weight = init_siren, init_bias = rand),
        Dense(wd, 1)
        ;name = :decoder
    )

    __ntimes = Base.Fix2(GeometryLearning._ntimes, N)

    Chain(
        SplitRows(1, 2; channel_dim = 2),        # u[N, 1, B], x[N, 1, B]
        Parallel(nothing, encoder, NoOpLayer()), # ũ[L, B]   , x[N, 1, B]
        Parallel(vcat, WrappedFunction(__ntimes), ReshapeLayer((1, N))), # [L,N,B], [1,N,B] -> [L+1,N,B]
        decoder, 
        ReshapeLayer((N, 1)),
    )
end

# p, st = Lux.setup(rng, NN)

model, ST = train_model(rng, NN, _data, data_, V, opt;
    batchsize, batchsize_, learning_rates, nepochs, dir, device, metadata)

plot_training(ST...) |> display

nothing
#
