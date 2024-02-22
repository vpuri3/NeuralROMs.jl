#
using GeometryLearning
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2                                 # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU

CUDA.allowscalar(false)

# using FFTW
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
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

#======================================================#
datafile = joinpath(@__DIR__, "burg_visc_re10k/data.bson")
dir = joinpath(@__DIR__, "model_inr")
V, _data, data_, metadata = makedata_INR(datafile)

# parameters
N = size(_data[1], 1)
E = 1000 # epochs
we = 32  # width
wd = 64  # width
l  = 4   # latent
act = relu # relu

opt = Optimisers.Adam()
batchsize  = 20
batchsize_ = 100
learning_rates = 1f-3 ./ (2 .^ (0:9))
nepochs = E/10 * ones(10) .|> Int
device = Lux.gpu_device()

NN = begin
    # encoder = Chain(
    #     Conv((9,), 1  => we, act; stride = 5),
    #     Conv((9,), we => we, act; stride = 5),
    #     Conv((9,), we => we, act; stride = 5),
    #     Conv((7,), we => we, act; stride = 1),
    #     flatten,
    #     Dense(we, we, act),
    #     Dense(we, l, act),
    # )

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
        Dense(we, l),
    )

    decoder = Chain(
        Dense(l+1, wd,sin; init_weight = scaled_siren_init(30), init_bias = rand),
        Dense(wd, wd, sin; init_weight = scaled_siren_init(1), init_bias = rand),
        Dense(wd, wd, sin; init_weight = scaled_siren_init(1), init_bias = rand),
        Dense(wd, wd, sin; init_weight = scaled_siren_init(1), init_bias = rand),
        Dense(wd, 1; use_bias = false),
    )

    ImplicitEncoderDecoder(encoder, decoder, (N,), 1)
end

p, st = Lux.setup(rng, NN)

# model, ST = train_model(rng, NN, _data, data_, opt;
#     batchsize, batchsize_, learning_rates, nepochs, dir, device, metadata)
#
# plot_training(ST...) |> display

nothing
#
