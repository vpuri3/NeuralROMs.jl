#
"""
Train an autoencoder on 1D advection data
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
    md_data = data["metadata"]
    close(data)

    # data sizes
    Nx, Nb, Nt = size(u)

    # subsample in space
    Ix = Colon()

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
    _Ib, Ib_ = [1], []

    # train on times 0.0 - 0.5s
    _It = Colon() # 1:1:Int(Nt/2) |> Array
    It_ = Colon() # 1:2:Nt        |> Array

    x = @view x[Ix]

    _u = @view u[Ix, _Ib, _It]
    u_ = @view u[Ix, Ib_, It_]

    _u = reshape(_u, Nx, :)
    u_ = reshape(u_, Nx, :)

    _Ns = size(_u, 2) # number of codes i.e. # trajectories
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

    readme = "Train/test on the same trajectory."

    metadata = (; ū, σu, x̄, σx, _Ib, Ib_, _It, readme, _Ns, Ns_, md_data)

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
    NN, p, st = freeze_autodecoder(decoder, p0; rng)

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

    # nlsolver = BFGS()
    # nlsolver = LevenbergMarquardt(; autodiff, linsolve)
    nlsolver = GaussNewton(;autodiff, linsolve, linesearch)

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
            p, _ = nonlinleastsq(NN, p, st, batch, Optimisers.Adam(1f-1); verbose)
            p, _ = nonlinleastsq(NN, p, st, batch, Optimisers.Adam(1f-2); verbose)
            p, _ = nonlinleastsq(NN, p, st, batch, Optimisers.Adam(1f-3); verbose)
        end

        p, _ = nonlinleastsq(NN, p, st, batch, nlsolver; maxiters = 20, verbose)

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
function dUdtRHS_advection(X, NN, p, st, Icode, md, t)
    U, UdX, UdXX = dUdX(X, NN, p, st, Icode, md)
    c = md.md_data.c
    -c .* UdX
end

function dUdtRHS_advection_diffusion(X, NN, p, st, Icode, md, t)
    U, UdX, UdXX = dUdX(X, NN, p, st, Icode, md)
    c = md.md_data.c
    ν = md.md_data.ν
    @. -c * UdX + ν * UdXX
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

    # finitediff_deriv2(_makeUfromX, X; ϵ = 0.05f0)
    # finitediff_deriv2(_makeUfromX, X)
    forwarddiff_deriv2(_makeUfromX, X)
end

function dUdp(X, NN, p, st, Icode, md)
    function _makeUfromX(p; X = X, NN = NN, st = st, Icode = Icode, md = md)
        makeUfromX(X, NN, p, st, Icode, md)
    end
    forwarddiff_jacobian(_makeUfromX, p)
end

function plot_derivatives1D_autodecoder(
    decoder::NTuple{3, Any},
    Xdata,
    p0::AbstractVector,
    md = nothing;
    second_derv::Bool = true,
)
    NN, p, st = freeze_autodecoder(decoder, p0; rng)

    Nx = length(Xdata)
    Xbatch = reshape(Xdata, 1, Nx)
    Icode = ones(Int32, 1, Nx)

    U, Udx, Udxx = dUdX(Xbatch, NN, p, st, Icode, md) .|> vec

    plt = plot(xabel = "x", ylabel = "u(x,t)")
    plot!(plt, Xdata, U  , label = "u"  , w = 2.0)
    plot!(plt, Xdata, Udx, label = "udx", w = 2.0)

    if second_derv
        plot!(plt, Xdata, Udxx, label = "udxx", w = 2.0)
    end

    return plt
end

#========================================================#
function make_residual(
    dUdtRHS,
    timestepper_residual;
    implicit::Bool = false
)
    function make_residual_internal(NN, p, st, batch, nlsp)
        XI, U0 = batch
        t0, t1, Δt, p0, md = nlsp
        X, Icode = XI

        _p = implicit ? p  : p0
        _t = implicit ? t1 : t0

        Rhs = dUdtRHS(X, NN, _p, st, Icode, md, _t)
        U1  = makeUfromX(X, NN, p, st, Icode, md)

        timestepper_residual(Δt, U0, U1, Rhs) |> vec
    end
end

function timestepper_residual_euler(Δt, U0, U1, Rhs)
    U1 - U0 - Δt * Rhs |> vec
end

function residual_learn(NN, p, st, batch, nlsp)
    XI, U0 = batch
    X, Icode = XI
    md = nlsp

    U1 = makeUfromX(X, NN, p, st, Icode, md)

    vec(U1 - U0)
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
    NN, p, st = freeze_autodecoder(decoder, p0; rng)
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
    nlsolver = GaussNewton(;autodiff, linsolve, linesearch)

    # critical to performance
    nlsmaxiters = 10
    nlsabstol = T(1f-6)
    Δt = T(5f-2)

    #============================#
    # learn IC
    #============================#
    timestep = 0
    if verbose
        println("#============================#")
        println("Time Step: $timestep, time: 0.0 - learn IC")
    end

    Xbatch = reshape.((Xdata, Icode), 1, Nx)
    Ubatch = reshape(Udata[:, 1], 1, Nx)
    batch  = (Xbatch, Ubatch)

    p0, nlssol, = nonlinleastsq(NN, p, st, batch, nlsolver;
        residual = residual_learn,
        nlsp = md,
        maxiters = nlsmaxiters,
        abstol = nlsabstol,
        termination_condition = AbsTerminationMode(),
        verbose,
    )

    U0 = makeUfromX(Xbatch[1], NN, p, st, Xbatch[2], md)

    if verbose
        println("#============================#")
    end

    #============================#
    # Set up solver
    #============================#
    dUdtRHS = dUdtRHS_advection
    timestepper_residual = timestepper_residual_euler
    residual = make_residual(dUdtRHS, timestepper_residual; implicit = false)

    # projection_type = Val(:PODGalerkin)
    projection_type = Val(:LSPG)

    Tinit, Tfinal = extrema(Tdata)
    t0 = t1 = T(Tinit)

    ps = (p0,)
    Us = (U0,)
    ts = (t0,)

    ##########
    # LOOK INSIDE THE NONLINEAR SOLVER. FOLLOW CROM METHODOLOGY
    # - why is the nonlinear solver converging when it shouldn't???
    ##########
    # RANDOM IDEAS
    # - NN partition of Unity model combined with SIRENs
    # - can we use variational inference in the NN models?
    ##########

    #============================#
    # Time loop
    #============================#
    while t1 <= Tfinal
    
        #============================#
        # set up
        #============================#
        timestep += 1
        batch = (Xbatch, U0)
        t1 = t0 + Δt
        nlsp = t0, t1, Δt, p0, md
    
        if verbose
            t_print  = round(t1; sigdigits=6)
            Δt_print = round(Δt; sigdigits=6)
            println("Time Step: $timestep, Time: $t_print, Δt: $Δt_print")
        end
    
        #============================#
        # solve
        #============================#
        p1, l = if projection_type isa Val{:LSPG}
            @time p1, nlssol = nonlinleastsq(
                NN, p0, st, batch, nlsolver;
                residual, nlsp, maxiters = nlsmaxiters, abstol = nlsabstol,
            )

            nlsmse = sum(abs2, nlssol.resid) / length(nlssol.resid)
            println("Nonlinear Steps: $(nlssol.stats.nsteps), \
                MSE: $(round(nlsmse, sigdigits = 8)), \
                Ret: $(nlssol.retcode)"
            )

            #===== ADAPTIVE TIME-STEPPER ====#

            if (nlsmse < nlsabstol) & (nlssol.stats.nsteps < 4)
                Δt *= T(2f0)
            end

            while (nlsmse > nlsabstol) #| SciMLBase.successful_retcode(nlretcode)
                if Δt < T(1f-4)
                    println("Δt = $Δt")
                    break
                end

                Δt /= T(2f0)
                t1 = t0 + Δt
            
                l_print = round(nlsmse; sigdigits = 6)
                t_print = round(t1; sigdigits = 6)
                Δt_print = round(Δt; sigdigits = 6)

                println("REPEATING Time Step: $timestep, \
                    Time: $t_print, MSE: $l_print, Δt: $Δt_print")
            
                nlsp = t0, t1, Δt, p0, md
                @time p1, nlssol = nonlinleastsq(
                    NN, p0, st, batch, nlsolver;
                    residual, nlsp,
                    maxiters = nlsmaxiters, abstol = nlsabstol,
                )

                nlsmse = sum(abs2, nlssol.resid) / length(nlssol.resid)
                println("Nonlinear Steps: $(nlssol.stats.nsteps), \
                    MSE: $(round(nlsmse, sigdigits = 8)), \
                    Ret: $(nlssol.retcode)"
                )
            end

            p1, nlsmse
        elseif projection_type isa Val{:PODGalerkin}
            # dU/dp, dU/dt
            J = dUdp(Xbatch[1], NN, p, st, Icode, md) # (N, n)
            r = dUdtRHS_advection(Xbatch[1], NN, p0, st, Icode, md, t0) # (N,)
            dpdt = J \ vec(r)
            p1 = p0 + Δt * dpdt # evolve with euler forward
            p1, T(0f0)
        else
            error("Projection type must be `Val(:LSPG)`, or `Val(:PODGalerkin)`.")
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
    mu = data["mu"]

    close(data)

    # data sizes
    Nx, Nb, Nt = size(Udata)

    mu = isnothing(mu) ? fill(nothing, Nb) : mu

    # subsample in space
    Ix = Colon()
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

    # #=

    #==============#
    # from training data
    #==============#

    # _data, data_, _ = makedata_autodecode(datafile)
    #
    # _upred = NN(_data[1] |> device, p |> device, st |> device)[1] |> Lux.cpu_device()
    # _Upred = _upred * sqrt(md.σu) .+ md.ū
    # _Upred = reshape(_Upred, Nx, length(md._Ib), Nt)
    #
    # for k in 1:length(md._Ib)
    #     Ud = @view _Udata[:, k, :]
    #     Up = @view _Upred[:, k, :]
    #
    #     if makeplot
    #         xlabel = "x"
    #         ylabel = "u(x, t)"
    #
    #         _mu = mu[md._Ib[k]]
    #         title  = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"
    #
    #         anim = animate1D(Ud, Up, Xdata, Tdata;
    #             w = 2, xlabel, ylabel, title)
    #         gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)
    #     end
    # end

    #==============#
    # inference (via data regression)
    #==============#

    # decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    # p0 = _code[2].weight[:, 1]
    #
    # for k in 1:1 # axes(mu, 1)
    #     ud = _udata[:, k, :]
    #     data = (xdata, ud, Tdata)
    #
    #     _, up, er = infer_autodecoder(decoder, data, p0; rng, device, verbose)
    #
    #     Ud = ud * sqrt(md.σu) .+ md.ū
    #     Up = up * sqrt(md.σu) .+ md.ū
    #
    #     if makeplot
    #         xlabel = "x"
    #         ylabel = "u(x, t)"
    #
    #         _mu   = mu[k]
    #         title = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"
    #         _name = k in md._Ib ? "infer_train$(k)" : "infer_test$(k)"
    #
    #         anim = animate1D(Ud, Up, Xdata, Tdata;
    #             w = 2, xlabel, ylabel, title)
    #         gif(anim, joinpath(outdir, "$(_name).gif"), fps=30)
    #     end
    # end

    #==============#
    # evolve
    #==============#

    # decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    # p0 = _code[2].weight[:, 1]
    #
    # for k in 1:1 # axes(mu, 1)
    #     Ud = Udata[:, k, :]
    #     data = (Xdata, Ud, Tdata)
    #     _, Up, Tpred = evolve_autodecoder(decoder, data, p0; rng, device, verbose, md)
    #
    #     if makeplot
    #         xlabel = "x"
    #         ylabel = "u(x, t)"
    #
    #         _mu   = mu[k]
    #         title = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"
    #         _name = k in md._Ib ? "evolve_train$(k)" : "evolve_test$(k)"
    #
    #         # anim = animate1D(Up, Xdata, Tpred; linewidth=2,
    #         #     xlabel="x", ylabel="u(x,t)", title = "μ = $_mu, ")
    #         # gif(anim, joinpath(outdir, "$(_name).gif"), fps=30)
    #     
    #         _plt = plot(title = "μ=$_mu", xlabel = "x", ylabel = "x", legend = false)
    #         plot!(_plt, Xdata, Up[:, 1:10:min(size(Up, 2), 300)])
    #         png(_plt, joinpath(outdir, "evolve_$k"))
    #     end
    # end

    # =#

    #==============#
    # check derivative
    #==============#

    begin
        k = 1
        i = 100
    
        decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
        p0 = _code[2].weight[:, i]
    
        plt = plot_derivatives1D_autodecoder(decoder, Xdata, p0, md,
            second_derv = false)
        png(plt, joinpath(outdir, "derv"))
        display(plt)
    end

    begin
        k = 1
        Ud = Udata[:, k, :]
        data = (Xdata, Ud, Tdata)

        decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
        p0 = _code[2].weight[:, 1]

        @time _, Up, Tpred = evolve_autodecoder(decoder, data, p0; rng, device, verbose, md)

        idx = LinRange(1, size(Up, 2), 11) .|> Base.Fix1(round, Int)

        function uIC(x; μ = -0.5f0, σ = 0.1f0)
            u = @. exp(-1f0/2f0 * ((x-μ)/σ)^2)
            reshape(u, :, 1)
        end

        function uExact(x, c, t; μ = -0.5f0, σ=0.1f0)
            uIC(x; μ = μ + c * t, σ)
        end

        tidx = Tpred[idx]
        upred = Up[:, idx]

        utrue = Tuple(uExact(Xdata, md.md_data.c, t) for t in tidx)
        utrue = hcat(utrue...)

        plt = plot(xlabel = "x", ylabel = "u(x, t)", legend = false)
        plot!(plt, Xdata, upred, w = 2, s = :solid)
        plot!(plt, Xdata, utrue, w = 2, s = :dash)

        error = sum(abs, (upred - utrue).^2) / length(utrue)
        print("MSE: $(error)")

        png(plt, joinpath(outdir, "evolve_$k"))
        display(plt)
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
    modeldir::String;
    device = Lux.cpu_device(),
)
    _data, data_, metadata = makedata_autodecode(datafile)
    dir = modeldir

    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#
    E = 700
    lrs = (1f-2, 1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    opts = Optimisers.Adam.(lrs)
    nepochs = (50, Int.(E/7*ones(7))...)
    schedules = Step.(lrs, 1f0, Inf32)

    Ndata = numobs(_data)
    _batchsize, batchsize_  = Int(Ndata / 50), Ndata

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#
    l = 04  # latent
    h = 05  # hidden layers
    w = 32  # width
    act = sin

    init_wt_in = scaled_siren_init(3f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    lossfun = mse
    # lossfun = l2reg(mse, 1f0; property = :decoder)

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

    @time model, STATS = train_model(NN, _data; rng,
        _batchsize, batchsize_,
        opts, nepochs, schedules,
        device, dir, metadata, lossfun
    )

    plot_training(ST...) |> display

    model, STATS
end

#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")

modeldir = joinpath(@__DIR__, "model2")
modelfile = joinpath(modeldir, "model_09.jld2")

# isdir(modeldir) && rm(modeldir, recursive = true)
# model, STATS = train_autodecoder(datafile, modeldir; device)

outdir = joinpath(dirname(modelfile), "results")
postprocess_autodecoder(datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)

nothing
#======================================================#
# IDEAS: controlling noisy gradients
# - L2 regularization on weights
# - Gradient supervision
# - secondary (corrector) network for gradient supervision
# - use ParameterSchedulers.jl
#======================================================#
#
