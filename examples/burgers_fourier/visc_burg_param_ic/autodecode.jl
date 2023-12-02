#
"""
Train an autoencoder on 1D Burgers data
"""

using GeometryLearning

using LinearAlgebra, ComponentArrays

using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2                                 # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using Setfield                                    # misc

CUDA.allowscalar(false)

begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
end

#======================================================#
function makedata_autodecode(datafile::String)
    
    #==============#
    # load data
    #==============#
    data = jldopen(datafile)
    x = data["x"]
    u = data["u"] # [Nx, Nb, Nt]
    mu = data["mu"] # [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    close(data)

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
    rng::Random.AbstractRNG = Random.default_rng(),
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
            p, _ = nlsq(NN, p, st, batch, Optimisers.Adam(1f-1); verbose)
            p, _ = nlsq(NN, p, st, batch, Optimisers.Adam(1f-2); verbose)
            p, _ = nlsq(NN, p, st, batch, Optimisers.Adam(1f-3); verbose)
        end

        p, _ = nlsq(NN, p, st, batch, nls; maxiters = 20, verbose)

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

_normalize(u::AbstractArray, μ::Number, σ::Number) = (u .- μ) / sqrt(σ)
_unnormalize(u::AbstractArray, μ::Number, σ::Number) = u * sqrt(σ) .+ μ

function makeUfromX(X, NN, p, st, Icode, md)
    x = _normalize(X, md.x̄, md.σx)
    u = NN((x, Icode), p, st)[1]
    _unnormalize(u, md.ū, md.σu)
end

function dUdX(X, NN, p, st, Icode, md)

    function _makeUfromX(X; NN = NN, p = p, st = st, Icode = Icode, md = md)
        makeUfromX(X, NN, p, st, Icode, md)
    end

    finitediff_deriv2(_makeUfromX, X; ϵ = 0.05f0)
    # finitediff_deriv2(_makeUfromX, X)#; ϵ = 0.001f0)
    # forwarddiff_deriv2(_makeUfromX, X)
end

function dUdp(X, NN, p, st, Icode, md)

    function _makeUfromX(p; X = X, NN = NN, st = st, Icode = Icode, md = md)
        makeUfromX(X, NN, p, st, Icode, md)
    end

    forwarddiff_jacobian(_makeUfromX, p)
end

function dUdt(X, NN, p, st, Icode, md) # burgers RHS
    U, Udx, Udxx = dUdX(X, NN, p, st, Icode, md)

    # ν = md.ν
    # -U .* Udx + (1/ν) * Udxx # visc burgers
    -U .* Udx                # inviscid burgers
end

function residual_eulerbwd_burgers(NN, p, st, batch, nlsp)
    XI, U0 = batch
    t, Δt, ν, p0, md = nlsp
    X, Icode = XI

    Rhs = dUdt(X, NN, p, st, Icode, md) # RHS formed with current `p`
    U1  = makeUfromX(X, NN, p, st, Icode, md)

    Resid = U1 - U0 - Δt * Rhs
    vec(Resid)
end

function residual_eulerfwd_burgers(NN, p, st, batch, nlsp)
    XI, U0 = batch
    t, Δt, ν, p0, md = nlsp
    X, Icode = XI

    Rhs = dUdt(X, NN, p, st, Icode, md) # RHS formed with `p0`
    U1  = makeUfromX(X, NN, p, st, Icode, md)

    Resid = U1 - U0 - Δt * Rhs
    vec(Resid)
end

function residual_learn(NN, p, st, batch, nlsp)
    XI, U0 = batch
    X, Icode = XI
    md = nlsp

    U1 = makeUfromX(X, NN, p, st, Icode, md)

    vec(U1 - U0)
end

#========================================================#

function makemodel_autodecoder(
    decoder::NTuple{3, Any},
    p0::AbstractVector;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    decoder_frozen = Lux.Experimental.freeze(decoder...)
    code_len = length(p0)
    NN = AutoDecoder(decoder_frozen[1], 1, code_len)
    p, st = Lux.setup(rng, NN)
    p = ComponentArray(p)

    copy!(p, p0)
    @set! st.decoder.frozen_params = decoder[2]
    
    NN, p, st
end

function plot_derv(
    decoder::NTuple{3, Any},
    data::Tuple,
    p0::AbstractVector;
    md = nothing,
)
    Xdata, Udata = data
    NN, p, st = makemodel_autodecoder(decoder, p0; rng)

    Nx = length(Xdata)
    Xbatch = reshape(Xdata, 1, Nx)
    Icode = ones(Int32, 1, Nx)

    U, Udx, Udxx = dUdX(Xbatch, NN, p, st, Icode, md) .|> vec

    plt = plot()
    plot!(plt, Xdata, Udata[:, 1], label = "u data", w = 2.0)

    plot!(plt, Xdata, U  , label = "u"      , w = 2.0)
    plot!(plt, Xdata, Udx, label = "udx"    , w = 2.0)
    # plot!(_plt, Xplt, Udxx, label = "udxx", w = 2.0)
    png(plt, "deriv_plt")
    display(plt)

    return plt
end

#========================================================#

function evolve_autodecoder(
    decoder::NTuple{3, Any},
    data::Tuple,
    p0::AbstractVector;
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
    verbose::Bool = true,
    md = nothing,
)
    # make data
    Xdata, Udata, Tdata = data
    Nx, Nt = size(Udata)
    Icode = ones(Int32, Nx)

    # make model
    NN, p, st = makemodel_autodecoder(decoder, p0; rng)
    T = eltype(p0)

    # move to device
    Xdata = Xdata   |> device
    Icode = Icode   |> device
    Udata = Udata   |> device
    p, st = (p, st) |> device

    # TODO: make an ODEProblem about p
    # - how would ODEAlg compute abstol/reltol?

    # optimizer
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()
    linesearch = LineSearch() # TODO
    nls = GaussNewton(;autodiff, linsolve, linesearch)

    # misc
    projection_type = Val(:LSPG)
    nl_iters = 100

    #============================#
    # learn IC
    #============================#
    iter = 0
    if verbose
        println("#============================#")
        println("Iter: $iter, time: 0.0 - learn IC")
    end

    Xbatch = reshape.((Xdata, Icode), 1, Nx)
    Ubatch = reshape(Udata[:, 1], 1, Nx)
    batch  = (Xbatch, Ubatch)

    p0, _ = nlsq(NN, p, st, batch, nls;
        residual = residual_learn,
        nlsp = md,
        maxiters = nl_iters,
        verbose,
    )

    U0 = makeUfromX(Xbatch[1], NN, p, st, Xbatch[2], md)

    if verbose
        println("#============================#")
    end

    if projection_type isa Val{:PODGalerkin}
        J = dUdp(Xbatch[1], NN, p, st, Icode, md)
        # compute v = dp/dt
        # evolve p with time-stepper
    end

    #============================#
    # Set up solver
    #============================#
    Tinit, Tfinal = extrema(Tdata)
    t0 = t1 = T(Tinit)
    Δt = T(1f-2)
    ν  = T(1f-4)

    residual = residual_eulerbwd_burgers
    # residual = residual_eulerfwd_burgers

    ps = (p0,)
    Us = (U0,)
    ts = (t0,)

    #============================#
    # Time loop
    #============================#
    while t1 <= Tfinal

        #============================#
        # set up
        #============================#
        iter += 1
        batch = (Xbatch, U0)
        t1 = t0 + Δt
        nlsp = t1, Δt, ν, p0, md

        if verbose
            t_round  = round(t1; sigdigits=6)
            Δt_round = round(Δt; sigdigits=6)
            println("Iter: $iter, Time: $t_round, Δt: $Δt_round")
        end

        #============================#
        # solve
        #============================#
        @time p1, l = nlsq(NN, p0, st, batch, nls;
            residual, nlsp, maxiters = nl_iters, verbose)

        # adaptive time-stepper
        while l > 1f-5
            if Δt < 5f-5
                println("Δt = $Δt")
                break
            end

            Δt /= 2

            l_round = round(l; sigdigits = 6)
            Δt_round = round(Δt; sigdigits = 6)
            println("REPEATING Iter: $iter, MSE: $l_round, Time: $t_round, Δt: $Δt_round")

            nlsp  = t1, Δt, ν, p0, md
            @time p1, l = nlsq(NN, p0, st, batch, nls; residual, nlsp, maxiters = nl_iters, verbose)
        end

        #============================#
        # Evaluate, save, etc
        #============================#
        U1 = makeUfromX(Xbatch[1], NN, p1, st, Xbatch[2], md)

        ps = push(ps, p1)
        Us = push(Us, U1)
        ts = push(ts, t1)

        if verbose
            println("#============================#")
        end
    
        #============================#
        # update states
        #============================#
        p0 = p1
        U0 = U1
        t0 = t1
    end

    code = mapreduce(getdata, hcat, ps) |> Lux.cpu_device()
    pred = mapreduce(adjoint, hcat, Us) |> Lux.cpu_device()
    tyms = [ts...]

    return code, pred, tyms
end

#======================================================#
function postprocess_autodecoder(
    datafile::String,
    modelfile::String,
    outdir::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
    makeplot::Bool = true,
    verbose::Bool = true,
)

    #==============#
    # load data
    #==============#
    data = jldopen(datafile)
    Tdata = data["t"]
    Xdata = data["x"]
    Udata = data["u"]
    mu = data["mu"] |> vec
    close(data)

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
    # from training data
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
    # inference (via data regression)
    #==============#

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    for k in axes(mu, 1)
        ud = _udata[:, k, :]
        data = (xdata, ud, Tdata)

        _, up, er = infer_autodecoder(decoder, data, p0; rng, device, verbose)

        Ud = ud * sqrt(md.σu) .+ md.ū
        Up = up * sqrt(md.σu) .+ md.ū

        if makeplot
            _mu = round(mu[k], digits = 2)
            anim = animate1D(Ud, Up, Xdata, Tdata; linewidth=2,
                xlabel="x", ylabel="u(x,t)", title = "μ = $_mu, ")
            _name = k in md._Ib ? "infer_train$(k)" : "infer_test$(k)"
            gif(anim, joinpath(outdir, "$(_name).gif"), fps=30)
        end
    end

    #==============#
    # evolve
    #==============#

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    for k in 1:1 #[1, 2, 3, 4, 5] # axes(mu, 1)
        Ud = Udata[:, k, :]
        data = (Xdata, Ud, Tdata)
        _, Up, Tpred = evolve_autodecoder(decoder, data, p0; rng, device, verbose, md)
    
        if makeplot
            _mu = round(mu[k], digits = 2)
            # anim = animate1D(Up, Xdata, Tpred; linewidth=2,
            #     xlabel="x", ylabel="u(x,t)", title = "μ = $_mu, ")
            # _name = k in md._Ib ? "evolve_train$(k)" : "evolve_test$(k)"
            # gif(anim, joinpath(outdir, "$(_name).gif"), fps=30)
           
            _plt = plot(title = "μ=$_mu", xlabel = "x", ylabel = "x", legend = false)
            plot!(_plt, Xdata, Up[:, 1:10:min(size(Up, 2), 300)])
            png(_plt, joinpath(outdir, "evolve_$k"))
        end
    end

    =#

    #==============#
    # check derivative
    #==============#

    begin
        k = 1
        i = 80
        _data = (Xdata, _Udata[:, k, i])
    
        decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
        p0 = _code[2].weight[:, 100]
    
        plt = plot_derv(decoder, _data, p0; md)
        png(plt, joinpath(outdir, "derv"))
        display(plt)
    end

    # begin
    #     k = 1
    #     Ud = Udata[:, k, :]
    #     data = (Xdata, Ud, Tdata)
    #
    #     decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    #     p0 = _code[2].weight[:, 1]
    #
    #     _, Up, Tpred = evolve_autodecoder(decoder, data, p0; rng, device, verbose, md)
    #
    #     _plt = plot(xlabel = "x", ylabel = "x", legend = false)
    #     plot!(_plt, Xdata, Up[:, 1:10:min(size(Up, 2), 300)])
    #     png(_plt, joinpath(outdir, "evolve_$k"))
    #     display(_plt)
    # end

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
    modeldir::String;
    device = Lux.cpu_device(),
)
    _data, data_, metadata = makedata_autodecode(datafile)
    dir = modeldir

    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#
    ################
    E = 1_750
    lrs = (1f-2, 1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    opts = Optimisers.Adam.(lrs)
    nepochs = (50, Int.(E/7*ones(7))...)
    schedules = Step.(lrs, 1f0, Inf32)

    ################
    # E = 2000
    # nepochs = (50, E,)
    # schedules = (
    #     Step(1f-2, 1f0, Inf32),                        # constant warmup
    #     # Triangle(λ0 = 1f-5, λ1 = 1f-3, period = 20), # gradual warmup
    #     # Exp(1f-3, 0.996f0),
    #     SinExp(λ0 = 1f-5, λ1 = 1f-3, period = 50, γ = 0.995f0),
    #     # CosAnneal(λ0 = 1f-5, λ1 = 1f-3, period = 50),
    # )
    # opts = fill(Optimisers.Adam(), length(nepochs)) |> Tuple
    ################

    _batchsize, batchsize_  = 1024 .* (10, 300)

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#
    l = 08  # latent
    h = 10  # hidden layers
    w = 096 # width
    act = sin

    init_wt_in = scaled_siren_init(3f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    # lossfun = mse
    lossfun = l2reg(mse, 1f0; property = :decoder)
    # reg: 1f-1 -> some change, MSE 4f-5
    # reg: 1f-0 -> wiggles go down. MSE 1f-4

    #----------------------#----------------------#

    in_layer = Dense(l+1, w, act; init_weight = init_wt_in, init_bias)
    hd_layer = Dense(w  , w, act; init_weight = init_wt_hd, init_bias)
    fn_layer = Dense(w  , 1; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

    decoder = Chain(
        in_layer,
        fill(hd_layer, h)...,
        fn_layer,
    )

    NN = AutoDecoder(decoder, metadata._Ns, l)

    # second order optimizer
    # if it can fit on 11 gb vRAM on 2080Ti
    if Lux.parameterlength(decoder) < 35_000
        opts = (opts..., LBFGS(),)
        nepochs = (nepochs..., round(Int, E / 10))
        schedules = (schedules..., Step(1f0, 1f0, Inf32))
    end

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_,
        opts, nepochs, schedules,
        device, dir, metadata, lossfun
    )

    plot_training(ST...) |> display

    model
end

#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.jld2")

modeldir = joinpath(@__DIR__, "model_dec_sin_08_10_96_reg")
modelfile = joinpath(modeldir, "model_08.jld2")

# isdir(modeldir) && rm(modeldir, recursive = true)
# model, STATS = train_autodecoder(datafile, modeldir; device)

# modeldir = joinpath(@__DIR__, "model_dec_sin_03_10_96_reg/")
# modelfile = joinpath(modeldir, "model_08.jld2")

outdir = joinpath(dirname(modelfile), "results")
postprocess_autodecoder(datafile, modelfile, outdir; rng,
    device, makeplot = true, verbose = true)

#======================================================#
# IDEAS: controlling noisy gradients
# - L2 regularization on weights
# - Gradient supervision
# - secondary (corrector) network for gradient supervision
# - use ParameterSchedulers.jl
#======================================================#
# nothing
#
