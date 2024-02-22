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

#======================================================#
function makedata_INR(
    datafile::String;
    Ix = Colon(), # subsample in space
    _Ib = Colon(), # train/test split in batches
    Ib_ = Colon(),
    _It = Colon(), # train/test split in time
    It_ = Colon(),
)
    #==============#
    # load data
    #==============#
    data = jldopen(datafile)
    t  = data["t"]
    x  = data["x"]
    u  = data["u"]
    mu = data["mu"]
    md_data = data["metadata"]
    close(data)

    @assert ndims(u) ∈ (3,4,)
    @assert x isa AbstractVecOrMat
    x = x isa AbstractVector ? reshape(x, 1, :) : x # (Dim, Npoints)

    if ndims(u) == 3 # [Nx, Nb, Nt]
        u = reshape(u, 1, size(u)...) # [1, Nx, Nb, Nt]
    end

    in_dim  = size(x, 1)
    out_dim = size(u, 1)

    println("input size $in_dim with $(size(x, 2)) points per trajectory.")
    println("output size $out_dim.")

    @assert eltype(x) === Float32
    @assert eltype(u) === Float32

    mu = isnothing(mu) ? fill(nothing, Nb) |> Tuple : mu
    mu = isa(mu, AbstractArray) ? vec(mu) : mu

    #==============#
    # normalize
    #==============#

    ū  = sum(u, dims = (2,3,4)) / (length(u) ÷ out_dim) |> vec
    σu = sum(abs2, u .- ū, dims = (2,3,4)) / (length(u) ÷ out_dim) .|> sqrt |> vec
    u  = normalizedata(u, ū, σu)

    x̄  = sum(x, dims = 2) / size(x, 2) |> vec
    σx = sum(abs2, x .- x̄, dims = 2) / size(x, 2) .|> sqrt |> vec
    x  = normalizedata(x, x̄, σx)

    #==============#
    # subsample in space, time
    #==============#
    _x = @view x[:, Ix]
    x_ = @view x[:, Ix]

    _t = @view t[_It]
    t_ = @view t[It_]

    #==============#
    # train/test split
    #==============#

    _u = @view u[:, Ix, _Ib, _It]
    u_ = @view u[:, Ix, Ib_, It_]

    Nx = size(_x, 2)
    @assert size(_u, 2) == size(_x, 2) "size(_u): $(size(_u)), size(_x): $(size(_x))"

    println("Using $Nx sample points per trajectory.")

    _Ns = size(_u, 3) * size(_u, 4) # number of codes i.e. # trajectories
    Ns_ = size(u_, 3) * size(u_, 4)

    println("$_Ns / $Ns_ trajectories in train/test sets.")

    readme = "Train/test on the same trajectory."

    # u [out_dim, Nx, Nb, Nt]
    # x [in_dim, Nx]

    #################
    _u = permutedims(_u, (2, 1, 3, 4)) # [Nx, out_dim, Nb, Nt]
    u_ = permutedims(u_, (2, 1, 3, 4))

    _Ns = size(_u, 3) * size(_u, 4) # number of codes i.e. # trajectories
    Ns_ = size(u_, 3) * size(u_, 4)

    _x = zeros(Float32, Nx, out_dim + in_dim, _Ns)
    x_ = zeros(Float32, Nx, out_dim + in_dim, Ns_)

    _data[:, begin:out_dim, :] = _u
    data_[:, begin:out_dim, :] = u_

    _data[:, out_dim+1:end, :] .= x
    data_[:, out_dim+1:end, :] .= x

    readme = "Train/test on 0.0-0.5."

    metadata = (; ū, σu, x̄, σx, _Ib, Ib_, _It, readme)

    (_x, _u), (x_, u_), metadata
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
    encoder = Chain(
        Conv((8,), 1  => we, acte; stride = 4, pad = SamePad()),
        Conv((8,), we => we, acte; stride = 4, pad = SamePad()),
        Conv((8,), we => we, acte; stride = 4, pad = SamePad()),
        Conv((8,), we => we, acte; stride = 2, pad = SamePad()),
        flatten,
        Dense(w, l),
    )

    decoder = Chain(
        Dense(l+1, wd,sin; init_weight = scaled_siren_init(30), init_bias = rand),
        Dense(wd, wd, sin; init_weight = scaled_siren_init(1),  init_bias = rand),
        Dense(wd, wd, sin; init_weight = scaled_siren_init(1),  init_bias = rand),
        Dense(wd, wd, sin; init_weight = scaled_siren_init(1),  init_bias = rand),
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
