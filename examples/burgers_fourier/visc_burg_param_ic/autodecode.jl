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

#======================================================#
function makedata_INR(datafile)
    
    data = BSON.load(datafile)

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
    # _Ib, Ib_ = splitobs(1:Nb; at = 0.5, shuffle = true)
    _Ib, Ib_ = [1, 4, 7], [2, 3, 5, 6]

    # train on times 0.0 - 0.5s
    _It = Colon() # 1:1:Int(Nt/2) |> Array
    It_ = Colon() # 1:2:Nt        |> Array

    x = @view x[Ix]

    _u = @view u[Ix, _Ib, _It]
    u_ = @view u[Ix, Ib_, It_]

    _u = reshape(_u, Nx, :)
    u_ = reshape(u_, Nx, :)

    _x = zeros(Float32, Nx, 2, length(_u)) # [x, idx]
    x_ = zeros(Float32, Nx, 2, length(u_))

    _y = reshape(_u, 1, :)
    y_ = reshape(u_, 1, :)

    _x[:, 1, :] .= x
    x_[:, 1, :] .= x

    _x[:, 2, :] .= 1:size(_u, 2)
    x_[:, 2, :] .= 1:size(u_, 2)

    _x = reshape(_x, 2, :)
    x_ = reshape(x_, 2, :)

    V = nothing # FourierSpace(N)

    readme = "Train/test on 0.0-0.5."

    metadata = (; ū, σu, x̄, σx, _Ib, Ib_, _It, readme)

    V, (_x, _y), (x_, y_), metadata
end

#======================================================#
datafile = joinpath(@__DIR__, "burg_visc_re10k/data.bson")
dir = joinpath(@__DIR__, "model_inr")
V, _data, data_, metadata = makedata_INR(datafile)

# parameters
N = size(_data[1], 1)
E = 1000 # epochs
w = 64   # width
l = 4   # latent
act = relu # relu

opt = Optimisers.Adam()
batchsize  = 20
batchsize_ = 100
learning_rates = 1f-3 ./ (2 .^ (0:9))
nepochs = E/10 * ones(10) .|> Int
device = Lux.gpu_device()

NN = begin

    decoder = Chain(
        Dense(l+1, w, sin), #; init_weight = scaled_siren_init(30), init_bias = rand),
        Dense(w  , w, sin), #; init_weight = scaled_siren_init(1), init_bias = rand),
        Dense(w  , w, sin), #; init_weight = scaled_siren_init(1), init_bias = rand),
        Dense(w  , w, sin), #; init_weight = scaled_siren_init(1), init_bias = rand),
        Dense(w  , 1; use_bias = false),
    )

    dim = 1
    nbatches = size(_data[1], 2)

    AutoDecoder(decoder, dim, nbatches, l) #; init_weight = randn32)
end

p, st = Lux.setup(rng, NN)

# model, ST = train_model(rng, NN, _data, data_, V, opt;
#     batchsize, batchsize_, learning_rates, nepochs, dir, device, metadata)
#
# plot_training(ST...) |> display

nothing
#
