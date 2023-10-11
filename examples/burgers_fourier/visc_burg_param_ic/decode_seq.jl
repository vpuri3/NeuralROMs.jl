#
#======================================================#
# First learn a nonlinaer manifold via autodecode method.
# Then, one of two ways to prceed here for learning correction:
#
# 1. Learn another manifold to correct the error. Then,
#
#   u(x) = NN1(code1, x; theta1) + NN2(code2, x; theta2)
#
#   code = (code1, code2), theta = (theta1, theta2)
#
# 2. Learn correction manifold from the same code, i.e.
#   learn decoder from the same code.
#
#   u(x) = NN2(code, x; theta1) + NN2(code, x; theta2)
#
#   code = (code), theta = (theta1, theta2)
#
# 3. Apply Gauss-Newton for learning the manifold correction
#======================================================#
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

    _x = (reshape(_xyz, 1, :), reshape(_idx, 1, :))
    x_ = (reshape(xyz_, 1, :), reshape(idx_, 1, :))

    readme = "Train/test on 0.0-0.5."

    metadata = (; ū, σu, x̄, σx, _Ib, Ib_, _It, readme, _Ns, Ns_)

    (_x, _y), (x_, y_), metadata
end

#======================================================#
function visualize_error(datafile, modelfile, outdir)

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
    model = BSON.load(modelfile)
    NN, p, st = model[:model]
    md = model[:metadata] # (; ū, σu, _Ib, Ib_, _It, It_, readme)

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
        u = udata - upred
        _mu = round(mu[md._Ib[k]], digits = 2)
        anim = animate1D(u, Xdata, Tdata; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)
    end

    nothing
end
#======================================================#
function post_process_Autodecoders(datafile, outdir, modelfiles...)

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
    models = (BSON.load(modelfile) for modelfile in modelfile)
    NN, p, st = model[:model]
    md = model[:metadata] # (; ū, σu, _Ib, Ib_, _It, It_, readme)

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

datafile = joinpath(@__DIR__, "burg_visc_re10k/data.bson")
modelfile = joinpath(@__DIR__, "model_dec", "model.bson")
_data, data_, metadata = makedata_autodecode(datafile)

dir = joinpath(@__DIR__, "model_dec1")

# visualize_error(datafile, modelfile, dir)

# parameters
E = 400 # epochs
w = 16  # width
l = 4   # latent

opt = Optimisers.Adam()
batchsize  = 1024 * 50
batchsize_ = 1024 * 100
learning_rates = 1f-4 ./ (2 .^ (0:4))
nepochs = E/5 * ones(5) .|> Int
device = Lux.gpu_device()

# l = 4
decoder = Chain(
    Dense(l+1, w, sin; init_weight = scaled_siren_init(30.0), init_bias = rand),
    Dense(w  , w, sin; init_weight = scaled_siren_init(10.0), init_bias = rand),
    Dense(w  , w, sin; init_weight = scaled_siren_init(1.0), init_bias = rand),
    Dense(w  , w, sin; init_weight = scaled_siren_init(1.0), init_bias = rand),
    Dense(w  , 1; use_bias = false),
)

#======================================================#
# use the same code - i.e. learn correction manifold
#======================================================#
NN = decoder

_model = BSON.load(modelfile)[:model]
_, _code = GeometryLearning.get_autodecoder(_model...)
x1 = _data[1][1]
i1 = _data[1][2] |> vec
c1, _ =  _code[1](i1, _code[2], _code[3])
x1 = vcat(x1, c1)

#======================================================#
# learn new code + manifold. this will increase latent space size.
#======================================================#
# NN = AutoDecoder(decoder, maximum(i1) |> Int, l)
# x1 = (_data[1][1], i1)

#======================================================#
u1 = _data[2] - _model[1](_data[1], _model[2], _model[3])[1]
ū1 = sum(u1) / length(u1)
σu1= sum(abs2, u1 .- ū1) / length(u1)
u1 = (u1 .- ū1) / sqrt(σu1)

_data1 = (x1, u1)
#======================================================#

model, ST = train_model(rng, NN, _data1, _data1, opt;
    batchsize, batchsize_, learning_rates, nepochs, dir, device, metadata,
    lossfun = pnorm(6),
)

plot_training(ST...) |> display

#======================================================#
nothing
#
