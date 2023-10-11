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
using Plots, BSON, JLD2

# accelerator
using CUDA, LuxCUDA, KernelAbstractions
CUDA.allowscalar(false)

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
function makedata_autodecode(datafile)
    
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

    _Ns = size(_u, 2) # num_codes
    Ns_ = size(u_, 2)

    _xyz = zeros(Float32, Nx, _Ns)
    xyz_ = zeros(Float32, Nx, Ns_)

    _idx = zeros(Int32, Nx, _Ns)
    idx_ = zeros(Int32, Nx, Ns_)

    _y = reshape(_u, 1, :)
    y_ = reshape(u_, 1, :)

    _xyz[:, :] .= x
    xyz_[:, :] .= x

    _idx[:, :] .= 1:_Ns |> adjoint
    idx_[:, :] .= 1:Ns_ |> adjoint

    _x = (reshape(_xyz, 1, :), reshape(_idx, 1, :),) |> reverse
    x_ = (reshape(xyz_, 1, :), reshape(idx_, 1, :),) |> reverse

    readme = "Train/test on 0.0-0.5."

    metadata = (; ū, σu, x̄, σx, _Ib, Ib_, _It, readme, _Ns, Ns_)

    (_x, _y), (x_, y_), metadata
end

#======================================================#

datafile = joinpath(@__DIR__, "burg_visc_re10k/data.bson")
dir = joinpath(@__DIR__, "model_hyp")
_data, data_, metadata = makedata_autodecode(datafile)

# parameters
E = 400 # epochs
l = 3   # latent

opt = Optimisers.Adam()
batchsize  = 1024 * 10
batchsize_ = 1024 * 100
learning_rates = 1f-4 ./ (2 .^ (0:4))
nepochs = E/5 * ones(5) .|> Int
device = Lux.gpu_device()

w = 128  # width of weight generator
e = 16 # width of evaluator

#; init_weight = scaled_siren_init(30.0), init_bias = rand),

evaluator = Chain(
    Dense(1, e, sin),
    Dense(e, e, sin),
    Dense(e, e, sin),
    Dense(e, e, sin),
    Dense(e, e, sin),
    Dense(e, 1; use_bias = false),
)

weight_gen = Chain(
    Dense(l, w, elu),
    Dense(w, w, elu),
    Dense(w, w, elu),
    Dense(w, w, elu),
    Dense(w, Lux.parameterlength(evaluator)),
)

code_gen = Chain(;
    vec  = WrappedFunction(vec),
    code = Embedding(metadata._Ns => l),
    gen  = weight_gen,
)

NN = HyperNet(code_gen, evaluator)

model, ST = train_model(rng, NN, _data, _data, opt;
    batchsize, batchsize_, learning_rates, nepochs, dir, device, metadata)

plot_training(ST...) |> display

#======================================================#
nothing
#
