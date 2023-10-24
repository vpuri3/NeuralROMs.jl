#
"""
Train an autoencoder on 1D Burgers data
"""

using GeometryLearning

# PDE stack
using LinearAlgebra, FFTW, ComponentArrays

# ML stack
using Random, Lux, Optimisers, OptimizationOptimJL, MLUtils, LineSearches

# vis/analysis, serialization
using Plots, BSON, JLD2

# accelerator
using CUDA, LuxCUDA, KernelAbstractions
CUDA.allowscalar(false)

# misc
using Setfield

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

    _x = (reshape(_xyz, 1, :), reshape(_idx, 1, :))
    x_ = (reshape(xyz_, 1, :), reshape(idx_, 1, :))

    readme = "Train/test on 0.0-0.5."

    metadata = (; ū, σu, x̄, σx, _Ib, Ib_, _It, readme, _Ns, Ns_)

    (_x, _y), (x_, y_), metadata
end

#======================================================#
function post_process_autodecoder(datafile, modelfile, outdir)

    # load data
    data = BSON.load(datafile)
    Tdata = data[:t]
    Xdata = data[:x]
    Udata = data[:u]
    mu = data[:mu]

    # get sizes
    Nx, Nb, Nt = size(Udata)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192
    Udata = @view Udata[Ix, :, :]
    Xdata = @view Xdata[Ix]

    # load model
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"] # (; ū, σu, _Ib, Ib_, _It, It_, readme)
    close(model)

    _Udata = @view Udata[:, md._Ib, :]
    Udata_ = @view Udata[:, md.Ib_, :]

    # normalize
    Xnorm = (Xdata .- md.x̄) / sqrt(md.σx)

    _Ns = Nt * length(md._Ib) # num_codes
    Ns_ = Nt * length(md.Ib_)

    _xyz = zeros(Float32, Nx, _Ns)
    xyz_ = zeros(Float32, Nx, Ns_)

    _xyz[:, :] .= Xnorm
    xyz_[:, :] .= Xnorm

    _idx = zeros(Int32, Nx, _Ns)
    idx_ = zeros(Int32, Nx, Ns_)

    _idx[:, :] .= 1:_Ns |> adjoint
    idx_[:, :] .= 1:Ns_ |> adjoint

    _x = (reshape(_xyz, 1, :), reshape(_idx, 1, :))
    x_ = (reshape(xyz_, 1, :), reshape(idx_, 1, :))

    _Upred = NN(_x, p, st)[1]
    _Upred = _Upred * sqrt(md.σu) .+ md.ū

    _Upred = reshape(_Upred, Nx, length(md._Ib), Nt)

    mkpath(outdir)

    for k in 1:length(md._Ib)
        udata = @view _Udata[:, k, :]
        upred = @view _Upred[:, k, :]
        _mu = round(mu[md._Ib[k]], digits = 2)
        anim = animate1D(udata, upred, Xdata, Tdata; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)
    end

    if haskey(md, :readme)
        RM = joinpath(outdir, "README.md")
        RM = open(RM, "w")
        write(RM, md.readme)
        close(RM)
    end

    nothing
end

#======================================================#
# parameters
#======================================================#

E = 3000
opts = 1f-4 .* (10, 5, 2, 1, 0.5, 0.2,) .|> Optimisers.Adam
nepochs = E/6 * ones(6) .|> Int |> Tuple
datafile = joinpath(@__DIR__, "burg_visc_re10k/data.bson")
dir = joinpath(@__DIR__, "model_dec")
_data, data_, metadata = makedata_autodecode(datafile)

opts = (opts..., BFGS(),)
nepochs = (nepochs..., E)

w = 32 # width
l = 3  # latent

opt = Optimisers.Adam()
_batchsize = 1024 * 10
batchsize_ = 1024 * 300
device = Lux.gpu_device()

decoder = Chain(
    Dense(l+1, w, sin; init_weight = scaled_siren_init(3f1), init_bias = rand),
    Dense(w  , w, sin; init_weight = scaled_siren_init(1f0), init_bias = rand),
    Dense(w  , w, elu; init_weight = scaled_siren_init(1f0), init_bias = rand),
    Dense(w  , w, elu; init_weight = scaled_siren_init(1f0), init_bias = rand),
    Dense(w  , 1; use_bias = false),
)

NN = AutoDecoder(decoder, metadata._Ns, l)

# p, st = Lux.setup(rng, NN)
# p = ComponentArray(p)
# u = NN(_data[1] |> device, p |> device, st |> device)

# model, ST = train_model(NN, _data;
#     rng, _batchsize, batchsize_, opts, nepochs, device, metadata, dir)
# plot_training(ST...) |> display

#======================================================#
# reload model
#======================================================#

modelfile = joinpath(@__DIR__, "model_dec", "model.jld2")
model = jldopen(modelfile)["model"]
decoder, _code = GeometryLearning.get_autodecoder(model...)

#======================================================#
# learn test code
#======================================================#
N_ = 1
data_ = begin
    data = _data
    # data = data_
    x_ = data[1][1][1:N_ * 1024]
    i_ = data[1][2][1:N_ * 1024]
    u_ = data[2][1:N_ * 1024]

    ((reshape(x_, 1, :), reshape(i_, 1, :)), reshape(u_, 1, :),)
end

decoder_frozen = Lux.Experimental.freeze(decoder...)
NN_ = AutoDecoder(decoder_frozen[1], N_, l; init_weight = zeros32)
p_, st_ = Lux.setup(rng, NN_)

@set! st_.decoder.frozen_params = ComponentArray(getdata(decoder[2]) |> copy, getaxes(decoder[2]))
@assert st_.decoder.frozen_params == decoder[2]

# gauss newton
include(joinpath(@__DIR__, "gaussnewton.jl"))

for linesearch in (
    # Static,
    # BackTracking,
    HagerZhang,
    # MoreThuente,
    # StrongWolfe,
)
    println(linesearch)
    linesearch = linesearch === Static ? Static() : linesearch{Float32}()
    nlsq(NN_, p_, st_, data_; device, linesearch, α0 = 1f-1)
    println()
end

## BFGS, Newton, etc (i.e. calls to Optim.jl)

# E = 50
# opts, nepochs = (BFGS(), NewtonTrustRegion(),), (E, E,)
#
# _batchsize = N_ * 1024
# batchsize_ = N_ * 1024
#
# device = Lux.cpu_device()
#
# model_, ST = train_model(NN_, data_;
#     rng, _batchsize, batchsize_, opts, nepochs, device, metadata, dir,
#     name = "code_", p = p_, st = st_,
# )
# plot_training(ST...) |> display

#======================================================#

# datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.bson")
# modelfile = joinpath(@__DIR__, "model_dec", "model_bfgs.jld2")
# outdir = joinpath(@__DIR__, "result_dec")
#
# post_process_autodecoder(datafile, modelfile, outdir)
#======================================================#
nothing
#
