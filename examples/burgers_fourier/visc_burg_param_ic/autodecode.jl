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
    data::Tuple, # (x, u, t)
    p0::AbstractVector;
    device = Lux.cpu_device(),
    learn_init::Bool = false,
    verbose::Bool = true,
)
    # make data
    xdata, udata, tdata = data
    Nx, Nt = size(udata)
    id = ones(Int32, Nx)
    x, u = (xdata, id), udata

    # model
    decoder_frozen = Lux.Experimental.freeze(decoder...)
    code_len = length(p0)
    NN = AutoDecoder(decoder_frozen[1], 1, code_len)
    p, st = Lux.setup(rng, NN)
    p = ComponentArray(p)

    copy!(p, p0)
    @set! st.decoder.frozen_params = decoder[2]

    # optimizer
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()
    linesearch = LineSearch()

    # linesearchalg = Static()
    # linesearchalg = BackTracking()
    # linesearchalg = HagerZhang()
    # linesearch = LineSearch(; method = linesearchalg, autodiff = AutoZygote())
    # linesearch = LineSearch(; method = linesearchalg, autodiff = AutoFiniteDiff())

    # nls = BFGS()
    # nls = LevenbergMarquardt(; autodiff, linsolve)
    nls = GaussNewton(;autodiff, linsolve, linesearch)

    codes  = ()
    upreds = ()
    MSEs   = []

    x, u  = (x, u ) |> device
    p, st = (p, st) |> device

    for iter in 1:Nt

        xbatch = reshape.(x, 1, Nx)
        ubatch = reshape(u[:, iter], 1, Nx)
        batch  = xbatch, ubatch

        if learn_init & (iter == 1)
            p = nlsq(NN, p, st, batch, Optimisers.Adam(1f-1); verbose)
            p = nlsq(NN, p, st, batch, Optimisers.Adam(1f-2); verbose)
            p = nlsq(NN, p, st, batch, Optimisers.Adam(1f-3); verbose)
        end

        p = nlsq(NN, p, st, batch, nls; maxiters = 20, verbose)

        # eval
        upred = NN(xbatch, p, st)[1]
        l = round(mse(upred, ubatch); sigdigits = 8)

        codes  = push(codes, p)
        upreds = push(upreds, upred)
        push!(MSEs, l)

        if verbose
            println("Iter $iter, MSE: $l")
            iter += 1
        end
    end

    code  = mapreduce(getdata, hcat, codes ) |> Lux.cpu_device()
    upred = mapreduce(adjoint, hcat, upreds) |> Lux.cpu_device()

    return code, upred, MSEs
end
#======================================================#
using ForwardDiff
using ForwardDiff: Dual, Partials, value, partials

# struct FDDeriv1Tag end
# struct FDDeriv2Tag end
# struct FDDeriv2TagInternal end

"""
Based on SparseDiffTools.auto_jacvec

MWE:

```julia
# f = x -> exp.(x)
f = x -> x .^ 2
x = [1.0, 2.0, 3.0, 4.0]
v = ones(4)

forwarddiff_deriv1(f, x)
forwarddiff_deriv2(f, x)
```
"""
function forwarddiff_deriv1(f, x)
    T = eltype(x)
    tag = ForwardDiff.Tag(f, T)
    y = Dual{typeof(tag)}.(x, one(T))

    fy = f(y)
    fx = value.(fy)
    df = partials.(fy, 1)
    
    fx, df
end

# SparseDiffTools.SparseDiffToolsTag()
# SparseDiffTools.DeivVecTag()
# ForwardDiff.Tag(FDDeriv1Tag(), eltype(x))

# function ForwardDiff.checktag(
#     ::Type{<:ForwardDiff.Tag{<:SparseDiffToolsTag, <:T}},
#     f::F, x::AbstractArray{T}) where {T, F}
#     return true
# end

function forwarddiff_deriv2(f, x)
    T = eltype(x)
    tag1 = ForwardDiff.Tag(f, T)
    tag2 = ForwardDiff.Tag(f, T)
    z = Dual{typeof(tag1)}.(Dual{typeof(tag2)}.(x, one(T)), one(T))

    fz = f(z)
    fx = value.(value.(fz))
    df = value.(partials.(fz, 1))
    d2f = partials.(partials.(fz, 1), 1)

    fx, df, d2f
end

function finitediff_deriv2(f, x)
    
    T = real(eltype(x))
    ϵ = cbrt(eps(T)) # * x̄
    ϵinv = inv(ϵ)

    _fx = f(x .- ϵ)
    fx  = f(x)
    fx_ = f(x .+ ϵ)

    df  = T(0.5) * ϵinv   * (fx_ - _fx)
    d2f = T(1.0) * ϵinv^2 * (fx_ + _fx - 2fx)

    fx, df, d2f
end
#======================================================#

function evolve_autodecoder(
    decoder::NTuple{3, Any},
    data::Tuple,
    p0::AbstractVector;
    device = Lux.cpu_device(),
    verbose::Bool = true,
    md = nothing,
)
    # make data
    xdata, udata, tdata = data
    Nx, Nt = size(udata)
    id = ones(Int32, Nx)
    x, u = (xdata, id), udata

    # model
    decoder_frozen = Lux.Experimental.freeze(decoder...)
    code_len = length(p0)
    NN = AutoDecoder(decoder_frozen[1], 1, code_len)
    p, st = Lux.setup(rng, NN)
    p = ComponentArray(p)

    copy!(p, p0)
    @set! st.decoder.frozen_params = decoder[2]

    x, u  = (x, u ) |> device
    p, st = (p, st) |> device

    # TODO form residual vector in un-normalized space
    _normalize(u::AbstractArray, μ::Number, σ::Number) = (u .- μ) / sqrt(σ)
    _unnormalize(u::AbstractArray, μ::Number, σ::Number) = u * sqrt(σ) .+ μ

    function uderv(NN, p, st, xdata, md) # xdata - normalized x points
        x, i = xdata

        function makeUfromX(X; NN = NN, p = p, st = st, i = i, md = md)
            x = _normalize(X, md.x̄, md.σx)
            u = NN((x, i), p, st)[1]
            _unnormalize(u, md.ū, md.σu)
        end

        X = _unnormalize(x, md.x̄, md.σx)
        # finitediff_deriv2(makeUfromX, X)
        forwarddiff_deriv2(makeUfromX, X)
    end

    function dudt(NN, p, st, xdata, md, ν) # burgers RHS
        u, udx, udxx = uderv(NN, p, st, xdata, md)

        # -u .* udx + (1/ν) * udxx
        -u .* udx
    end

    function residual_eulerbwd(NN, p, st, data, nlsp)
        xdata, u0 = data
        t, Δt, ν, p0 = nlsp

        Rhs = dudt(NN, p, st, xdata, md, ν) # RHS formed with current `p`
        u1 = NN(xdata, p, st)[1]

        U0 = _unnormalize(u0, md.ū, md.σu)
        U1 = _unnormalize(u1, md.ū, md.σu)

        Resid = U1 - U0 - Δt * Rhs
        vec(Resid)
    end

    function residual_eulerfwd(NN, p, st, data, nlsp)
        xdata, u0 = data
        t, Δt, ν, p0 = nlsp

        Rhs = dudt(NN, p0, st, xdata, md, ν) # RHS formed with `p0`
        u1 = NN(xdata, p, st)[1]

        U0 = _unnormalize(u0, md.ū, md.σu)
        U1 = _unnormalize(u1, md.ū, md.σu)

        Resid = U1 - U0 - Δt * Rhs
        vec(Resid)
    end

    residual = residual_eulerbwd
    # residual = residual_eulerfwd

    # TODO: make an ODEProblem about p
    # - how would ODEAlg compute abstol/reltol?
    #

    # large error in between time-steps => need smaller step size.

    # optimizer
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()
    linesearch = LineSearch()
    nls = GaussNewton(;autodiff, linsolve, linesearch)

    # learn IC
    if verbose
        println("Iter: 0, time: 0.0 - learn IC")
    end

    xbatch = reshape.(x, 1, Nx)
    ubatch = reshape(u[:, 1], 1, Nx)
    batch  = (xbatch, ubatch)

    # return uderv(NN, p, st, xbatch, md) |> Lux.cpu_device()

    p0 = nlsq(NN, p, st, batch, nls; maxiters = 20, verbose)
    u0 = NN(xbatch, p0, st)[1]

    t = 0f0
    Δt = 1f-3
    ν = 1f-4

    ps = (p0,)
    us = (u0,)
    ts = (t,)

    xplt = vec(xdata) |> Array
    plt = plot(xplt, vec(u0 |> Array), w = 2.0, label = "Step 0")
    display(plt)

    for iter in 1:100
        batch = (xbatch, u0)
        nlsp  = (t, Δt, ν, p0)

        t += Δt
    
        if verbose
            t_round = round(t; sigdigits=6)
            println("Iter: $iter, time: $t_round")
        end

        p1 = nlsq(NN, p0, st, batch, nls; residual, nlsp, maxiters = 20, verbose)
        u1 = NN(xbatch, p1, st)[1]

        ps = push(ps, p1)
        us = push(us, u1)
        ts = push(ts, t)

        if iter % 10 == 0
            uplt = vec(u1) |> Array
            plot!(plt, xplt, uplt, w = 2.0, label = "Step: $iter")

            # iplt = Int(iter / 10)
            # Iplt = 1:32:Nx
            # scatter!(plt, xplt[Iplt], udata[Iplt, iplt], label = "data: $iter")
            display(plt)
        end
    
        p0 = p1
        u0 = u1
    end

    code = mapreduce(getdata, hcat, ps) |> Lux.cpu_device()
    pred = mapreduce(adjoint, hcat, us) |> Lux.cpu_device()
    tyms = [ts...]

    return code, pred, tyms
end

#======================================================#
function post_process_autodecoder(
    datafile::String,
    modelfile::String,
    outdir::String;
    device = Lux.cpu_device(),
    makeplot::Bool = true,
    verbose::Bool = true,
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
    # make outdir path
    #==============#
    mkpath(outdir)

    #==============#
    # normalize data
    #==============#
    xdata = (Xdata .- md.x̄) / sqrt(md.σx)
    udata = (Udata .- md.ū) / sqrt(md.σu)

    #==============#
    # train/test split
    #==============#
    _Udata = @view Udata[:, md._Ib, :] # un-normalized
    Udata_ = @view Udata[:, md.Ib_, :]

    _udata = udata[:, md._Ib, :] # normalized
    udata_ = udata[:, md.Ib_, :]

    #=

    #==============#
    # Training
    #==============#

    _data, data_, _ = makedata_autodecode(datafile)

    _upred = NN(_data[1] |> device, p |> device, st |> device)[1] |> Lux.cpu_device()
    _Upred = _upred * sqrt(md.σu) .+ md.ū
    _Upred = reshape(_Upred, Nx, length(md._Ib), Nt)

    for k in 1:length(md._Ib)
        Ud = @view _Udata[:, k, :]
        Up = @view _Upred[:, k, :]
        _mu = round(mu[md._Ib[k]], digits = 2)

        if makeplot
            anim = animate1D(Ud, Up, Xdata, Tdata; linewidth=2, xlabel="x",
                ylabel="u(x,t)", title = "μ = $_mu, ")
            gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)
        end
    end

    #==============#
    # Inference (via data regression)
    #==============#

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    # on train data
    for k in 1:length(md._Ib)
        ud = _udata[:, k, :]
        data = (xdata, ud, Tdata)

        _, up, er = infer_autodecoder(decoder, data, p0; device, verbose)

        Ud = ud * sqrt(md.σu) .+ md.ū
        Up = up * sqrt(md.σu) .+ md.ū

        if makeplot
            _mu = round(mu[md._Ib[k]], digits = 2)
            anim = animate1D(Ud, Up, Xdata, Tdata; linewidth=2,
                xlabel="x", ylabel="u(x,t)", title = "μ = $_mu, ")
            gif(anim, joinpath(outdir, "infer_train$(k).gif"), fps=30)
        end
    end

    # on test data
    for k in 1:length(md.Ib_)
        ud = udata_[:, k, :]
        data = (xdata, ud, Tdata)

        _, up, er = infer_autodecoder(decoder, data, p0; device)

        Ud = ud * sqrt(md.σu) .+ md.ū
        Up = up * sqrt(md.σu) .+ md.ū

        if makeplot
            _mu = round(mu[md.Ib_[k]], digits = 2)
            anim = animate1D(Ud, Up, Xdata, Tdata; linewidth=2,
                xlabel="x", ylabel="u(x,t)", title = "μ = $_mu, ")
            gif(anim, joinpath(outdir, "infer_test$(k).gif"), fps=30)
        end
    end

    =#

    #==============#
    # evolve
    #==============#

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    begin
        k = 1
        ud = udata[:, k, :]
        data = (xdata, ud, Tdata)

        code, pred, times = evolve_autodecoder(decoder, data, p0; device, verbose, md)

        # m = md.μ[k] # plot derivatives
        # plt = plot()
        # m = 0.6
        # plot!(plt, Xdata, U    |> vec, label = "u"   , w = 2.0)
        # plot!(plt, Xdata, Udx  |> vec, label = "udx" , w = 2.0)
        # # plot!(plt, xdata, udxx |> vec, label = "udxx", w = 2.0)
        #
        # Utrue = Udata[:, 1, 1]
        # plot!(plt, Xdata, Utrue, label = "u true", w = 2.0)
        #
        # Num = (Utrue[2:end] - Utrue[1:end-1]) / (Xdata[2] - Xdata[1])
        # plot!(plt, Xdata[1:end-1], Num, label = "udx true", w = 2.0)
        #
        # # # num = (utrue[3:end] + utrue[1:end-2] - 2utrue[2:end-1]) / (xdata[2] - xdata[1])^2
        # # # plot!(plt, xdata[2:end-1], num, label = "udx true", w = 2.0)
        # # # ylims!(plt, (-15, 15))
        # # display(plt)
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

function train_autodecoder(
    datafile::String,
    modelfile::String;
    device = Lux.cpu_device(),
)
    E = 3000
    opts = 1f-4 .* (10, 5, 2, 1, 0.5, 0.2,) .|> Optimisers.Adam
    nepochs = E/6 * ones(6) .|> Int |> Tuple
    _data, data_, metadata = makedata_autodecode(datafile)

    dir = modelfile
    opts = (opts..., BFGS(),)
    nepochs = (nepochs..., E)

    w = 32 # width
    l = 3  # latent

    _batchsize = 1024 * 10
    batchsize_ = 1024 * 300

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

device = Lux.gpu_device()

datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.bson")
modelfile = joinpath(@__DIR__, "model_dec", "model.jld2")
outdir = joinpath(@__DIR__, "result_dec")

# train_autodecoder(datafile, modelfile; device)
post_process_autodecoder(datafile, modelfile, outdir;
    device, makeplot = true, verbose = true)
#======================================================#
# f = x -> x .^ 2
# # f = x -> exp.(x)
# x = [1.0, 2.0, 3.0, 4.0]
# v = ones(4)
# forwarddiff_deriv1(f, x)
# forwarddiff_deriv2(f, x)

# nothing
#
