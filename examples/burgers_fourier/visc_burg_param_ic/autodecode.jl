#
"""
Train an autoencoder on 1D Burgers data
"""

using GeometryLearning

using LinearAlgebra, ComponentArrays

using Random, Lux, MLUtils                        # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, BSON, JLD2                           # vis / analysis
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using Setfield                                    # misc

CUDA.allowscalar(false)

begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
end

rng = Random.default_rng()
Random.seed!(rng, 199)

#======================================================#
function makedata_autodecode(datafile)
    
    #==============#
    # load data
    #==============#
    data = BSON.load(datafile)
    x = data[:x]
    u = data[:u] # [Nx, Nb, Nt]
    mu = data[:mu] # [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    # data sizes
    Nx, Nb, Nt = size(u)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192

    #==============#
    # normalize data
    #==============#
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
function infer_autodecoder(
    decoder::NTuple{3, Any},
    data::Tuple,
    p0::AbstractVector;
    device = Lux.cpu_device(),
    learn_init::Bool = false,
)
    # data
    fill!(data[1][2], true)
    loader = DataLoader(data; batchsize = 1)

    if device isa Lux.LuxCUDADevice
        loader = loader |> CuIterator
    end

    # model
    decoder_frozen = Lux.Experimental.freeze(decoder...)
    NN = AutoDecoder(decoder_frozen[1], 1, l; init_weight = zeros32)
    p, st = Lux.setup(rng, NN)
    p = ComponentArray(p)
    @set! st.decoder.frozen_params = decoder[2]

    # optimizer
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()

    # opt = LevenbergMarquardt(; autodiff, linsolve)
    opt = GaussNewton(;autodiff, linsolve)

    copy!(p, p0)

    codes  = ()
    upreds = ()
    MSEs   = []

    p, st = (p, st) |> device
    shape = (1, Colon())

    iter = 1

    for batch in loader
        xdata = reshape.(batch[1], shape...)
        ydata = reshape(batch[2], shape...)

        bdata = (xdata, ydata)

        if learn_init & (iter == 1)
            p = nlsq(NN, p, st, bdata, Optimisers.Adam(1f-1))
            p = nlsq(NN, p, st, bdata, Optimisers.Adam(1f-2))
            p = nlsq(NN, p, st, bdata, Optimisers.Adam(1f-3))
        end

        p = nlsq(NN, p, st, bdata, opt; maxiters = 20, verbose = false)

        # eval
        u = NN(xdata, p, st)[1]
        l = round(mse(u, ydata); sigdigits = 8)

        # large error in between time-steps => need smaller step size.

        codes  = push(codes, p)
        upreds = push(upreds, u)
        push!(MSEs, l)

        println("Iter $iter, MSE: $l")
        iter += 1
    end

    code  = mapreduce(getdata, hcat, codes ) |> Lux.cpu_device()
    upred = mapreduce(adjoint, hcat, upreds) |> Lux.cpu_device()

    return code, upred, MSEs
end

#======================================================#

function evolve_autodecoder(
    decoder::NTuple{3, Any},
    data::Tuple,
    p0::AbstractVector;
    device = Lux.cpu_device(),
)
    # data
    fill!(data[1][2], true)

    # model
    decoder_frozen = Lux.Experimental.freeze(decoder...)
    NN = AutoDecoder(decoder_frozen[1], 1, l; init_weight = zeros32)
    p, st = Lux.setup(rng, NN)
    p = ComponentArray(p)
    @set! st.decoder.frozen_params = decoder[2]

    # use FiniteDiff.jl ?
    function uderv(NN, p, st, data)
        xdata, ydata = data
        x, i = xdata

        T = eltype(ydata)
        ϵ = cbrt(T)

        x_ = x .+ ϵ
        _x = x .- ϵ

        ϵinv = 1 ./ ϵ

        _u = NN((_x, i), p, st)[1]
        u  = NN((x , i), p, st)[1]
        u_ = NN((x_, i), p, st)[1]

        udx  = T(0.5) * ϵinv * (u_ - _u)
        udxx = T(1.0) * ϵinv^2 * (u_ + _u - 2u)

        u, udx, udxx
    end

    function dudt(NN, p, st, data; ν = 1f4) # burgers RHS
        u, udx, udxx = uderv(NN, p, st, data)

        -u .* udx + (1/ν) * udxx
    end

    function residual(NN, p, st, data; Δt = 1f-3)
        rhs = dudt(NN, p, st, data)
    end
end

#======================================================#
function post_process_autodecoder(
    datafile::String,
    modelfile::String,
    outdir::String;
    device = Lux.cpu_device(),
)

    #==============#
    # load data
    #==============#
    data = BSON.load(datafile)
    Tdata = data[:t]
    Xdata = data[:x]
    Udata = data[:u]
    mu = data[:mu]

    # data sizes
    Nx, _, Nt = size(Udata)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192
    Udata = @view Udata[Ix, :, :]
    Xdata = @view Xdata[Ix]

    #==============#
    # load model
    #==============#
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"] # (; ū, σu, _Ib, Ib_, _It, It_, readme)
    close(model)

    #==============#
    # normalize data
    #==============#
    xdata = (Xdata .- md.x̄) / sqrt(md.σx)
    udata = (Udata .- md.ū) / sqrt(md.σu)

    #==============#
    # train/test split
    #==============#
    _Udata = @view Udata[:, md._Ib, :]
    Udata_ = @view Udata[:, md.Ib_, :]

    _udata = udata[:, md._Ib, :]
    udata_ = udata[:, md.Ib_, :]

    _Ns = Nt * length(md._Ib) # num_codes
    Ns_ = Nt * length(md.Ib_)

    #==============#
    # make data
    #==============#
    _xyz = zeros(Float32, Nx, _Ns)
    xyz_ = zeros(Float32, Nx, Ns_)

    _xyz[:, :] .= xdata
    xyz_[:, :] .= xdata

    _idx = zeros(Int32, Nx, _Ns)
    idx_ = zeros(Int32, Nx, Ns_)

    _idx[:, :] .= 1:_Ns |> adjoint
    idx_[:, :] .= 1:Ns_ |> adjoint

    _x = (reshape(_xyz, 1, :), reshape(_idx, 1, :))
    # x_ = (reshape(xyz_, 1, :), reshape(idx_, 1, :))

    mkpath(outdir)

    #==============#
    # Training
    #==============#
    _Upred = NN(_x, p, st)[1]
    _Upred = _Upred * sqrt(md.σu) .+ md.ū
    _Upred = reshape(_Upred, Nx, length(md._Ib), Nt)

    for k in 1:length(md._Ib)
        udata = @view _Udata[:, k, :]
        upred = @view _Upred[:, k, :]
        _mu = round(mu[md._Ib[k]], digits = 2)
        anim = animate1D(udata, upred, Xdata, Tdata; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)
    end

    #==============#
    # Inference
    #==============#
    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)

    id = ones(Int32, Nx, Nt)
    xd = (vec(xdata) * ones(1, Nt), id)
    qq = _code[2].weight[:, 1]

    # on train data
    for k in 1:length(md._Ib)
        ud = @view _udata[:, k, :]
        data  = (xd, ud)

        _, up, er = infer_autodecoder(decoder, data, qq; device)

        Ud = ud * sqrt(md.σu) .+ md.ū
        Up = up * sqrt(md.σu) .+ md.ū

        _mu = round(mu[md._Ib[k]], digits = 2)
        anim = animate1D(Ud, Up, Xdata, Tdata; linewidth=2,
            xlabel="x", ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "infer_train$(k).gif"), fps=30)
    end

    # on test data
    for k in 1:length(md.Ib_)
        ud = @view udata_[:, k, :]
        data  = (xd, ud)

        _, up, er = infer_autodecoder(decoder, data, qq; device)

        Ud = ud * sqrt(md.σu) .+ md.ū
        Up = up * sqrt(md.σu) .+ md.ū

        _mu = round(mu[md.Ib_[k]], digits = 2)
        anim = animate1D(Ud, Up, Xdata, Tdata; linewidth=2,
            xlabel="x", ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "infer_test$(k).gif"), fps=30)
    end

    #==============#
    # Done
    #==============#
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

function _train()
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

    model, ST = train_model(NN, _data;
        rng, _batchsize, batchsize_, opts, nepochs, device, metadata, dir)

    plot_training(ST...) |> display

    model
end

#======================================================#
# main
#======================================================#

# train
# _train()

datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.bson")
modelfile = joinpath(@__DIR__, "model_dec", "model.jld2")
outdir = joinpath(@__DIR__, "result_dec")

device = Lux.gpu_device()
post_process_autodecoder(datafile, modelfile, outdir; device)

#======================================================#
# learn test code
#======================================================#
# # reload for analysis
# modelfile = joinpath(@__DIR__, "model_dec", "model.jld2")
# model = jldopen(modelfile)["model"]
# decoder, _code = GeometryLearning.get_autodecoder(model...)
#
# data_ = begin
#     data, b = _data, 3
#     data, b = data_, 4
#
#     shape = (1024, b, 100) # Nx, Nb, Nt
#     x_ = reshape(data[1][1], shape)
#     u_ = reshape(data[2], shape)
#
#     i = 1
#     x_ = x_[:, i, :]
#     u_ = u_[:, i, :]
#
#     i_ = similar(x_, Int32)
#     fill!(i_, true)
#
#     (x_, i_), u_
# end
#
# q = _code[2].weight[:, 1]
# code_, upred_, MSE_ = infer_autodecoder(decoder, data_, q; device)
# p1 = plot(MSE_, ylims = (1f-5, Inf), yaxis = :log)
# p2 = plot(upred_[:, 60:5:100])
# plot(p1, p2)

#======================================================#
#======================================================#
#======================================================#
# nothing
#
