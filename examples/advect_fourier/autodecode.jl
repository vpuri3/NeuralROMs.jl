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
    σu = sum(abs2, u .- ū) / length(u) |> sqrt
    u  = normalizedata(u, ū, σu)

    # normalize space
    x̄  = sum(x) / length(x)
    σx = sum(abs2, x .- x̄) / length(x) |> sqrt
    x  = normalizedata(x, x̄, σx)

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
    model::NeuralSpaceModel,
    data::Tuple, # (X, U, T)
    p0::AbstractVector;
    device = Lux.cpu_device(),
    learn_init::Bool = false,
    verbose::Bool = false,
)
    # make data
    xdata, udata, _ = data
    Nx, Nt = size(udata)

    model = model |> device # problem here.
    xdata = xdata |> device
    udata = udata |> device
    p = p0 |> device

    # optimizer
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()
    linesearch = LineSearch()

    # linesearchalg = Static()
    # linesearchalg = BackTracking()
    # linesearchalg = HagerZhang()
    # linesearch = LineSearch(; method = linesearchalg, autodiff = AutoZygote())
    # linesearch = LineSearch(; method = linesearchalg, autodiff = AutoFiniteDiff())

    # nlssolve = BFGS()
    # nlssolve = LevenbergMarquardt(; autodiff, linsolve)
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)

    codes  = ()
    upreds = ()
    MSEs   = []

    for iter in 1:Nt

        xbatch = reshape(xdata, 1, Nx)
        ubatch = reshape(udata[:, iter], 1, Nx)
        batch  = xbatch, ubatch

        # if learn_init & (iter == 1)
        #     p, _ = nonlinleastsq(model, p, batch, Optimisers.Adam(1f-1); verbose)
        #     p, _ = nonlinleastsq(model, p, batch, Optimisers.Adam(1f-2); verbose)
        #     p, _ = nonlinleastsq(model, p, batch, Optimisers.Adam(1f-3); verbose)
        # end

        # @show typeof(batch[1]) # ReshapedArray :/
        p, _ = nonlinleastsq(model, p, batch, nlssolve; maxiters = 20, verbose)

        # eval
        upred = model(xbatch, p)
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
function dudtRHS_advection(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real,
    md;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    u, udx, udxx = dudx2(model, x, p; autodiff, ϵ)

    c = md.md_data.c
    -c .* udx
end

#========================================================#
function evolve_autodecoder(
    decoder::NTuple{3, Any},
    data::Tuple,
    p0::AbstractVector;
    rng::Random.AbstractRNG = Random.default_rng(),
    nlssolve = nothing,
    linsolve = nothing,
    device = Lux.cpu_device(),
    verbose::Bool = true,
    md = nothing,
)
    #============================#
    # set up
    #============================#

    # Inputs should be
    # - model::NeuralSpaceModel
    # - prob::PDEProblem
    # - data (x, u0)
    # - tspan
    # - p0,
    # - kws...

    # data
    xdata, udata, tdata = data
    Nx = size(udata, 1)

    # make model
    NN, p0, st = freeze_autodecoder(decoder, p0; rng)
    Icode = ones(Int32, 1, Nx)
    model = NeuralSpaceModel(NN, st, Icode, md.x̄, md.σx, md.ū, md.σu)

    # move to device
    xdata = xdata |> device
    Icode = Icode |> device
    udata = udata |> device
    model = model |> device
    p0 = p0 |> device

    T = eltype(p0)

    # solvers
    linsolve = isnothing(linsolve) ? QRFactorization() : linsolve

    nlssolve = if isnothing(nlssolve)
        autodiff = AutoForwardDiff()
        linesearch = LineSearch() # TODO
        nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    else
        nlssolve
    end

    #============================#
    # args/kwargs
    #============================#
    nlsmaxiters = 10
    nlsabstol = T(1f-6)
    Δt = T(5f-2)

    implicit_timestepper = false
    autodiff_space = AutoForwardDiff()
    ϵ_space = nothing

    dudtRHS = dudtRHS_advection
    timestepper_residual = timestepper_residual_euler

    # scheme = LeastSqPetrovGalerkin()
    # scheme = PODGalerkin(nothing)

    projection_type = Val(:PODGalerkin)
    # projection_type = Val(:LSPG)

    adaptive_timestep = true

    #============================#
    # learn IC
    #============================#
    tstep = 0

    if verbose
        println("#============================#")
        println("Time Step: $tstep, time: 0.0 - learn IC")
    end

    xbatch = reshape(xdata, 1, Nx)
    ubatch = reshape(udata[:, 1], 1, Nx)
    batch  = (xbatch, ubatch)

    p0, nlssol, = nonlinleastsq(model, p0, batch, nlssolve;
        residual = residual_learn,
        maxiters = nlsmaxiters * 5,
        abstol = nlsabstol,
        termination_condition = AbsTerminationMode(),
        verbose,
    )
    u0 = model(xbatch, p0)

    if verbose
        println("#============================#")
    end

    #============================#
    # Set up solver
    #============================#
    residual = make_residual(dudtRHS, timestepper_residual;
        implicit = implicit_timestepper,
        autodiff = autodiff_space,
        ϵ = ϵ_space,
        md,
    )

    tinit, tfinal = extrema(tdata) .|> T
    t0 = t1 = tinit

    ps = (p0,)
    us = (u0,)
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
    while t1 <= tfinal
    
        #============================#
        # set up
        #============================#
        tstep += 1
        batch = (xbatch, u0)
        t1 = t0 + Δt
    
        if verbose
            t_print  = round(t1; sigdigits=6)
            Δt_print = round(Δt; sigdigits=6)
            println("Time Step: $tstep, Time: $t_print, Δt: $Δt_print")
        end
    
        #============================#
        # solve
        #============================#

        # p1 = do_timestep(projection_type, adaptive_timestep, NN, p0, st, batch, )

        p1 = if projection_type isa Val{:LSPG}
            nlsp = t1, Δt, t0, p0, u0

            p1, nlssol = nonlinleastsq(
                model, p0, batch, nlssolve;
                residual, nlsp, maxiters = nlsmaxiters, abstol = nlsabstol,
            )

            nlsmse = sum(abs2, nlssol.resid) / length(nlssol.resid)
            nlsinf = norm(nlssol.resid, Inf)
            println("\tNonlinear Steps: $(nlssol.stats.nsteps), \
                MSE: $(round(nlsmse, sigdigits = 8)), \
                ||∞: $(round(nlsinf, sigdigits = 8)), \
                Ret: $(nlssol.retcode)"
            )

            #===== ADAPTIVE TIME-STEPPER =====#

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

                println("REPEATING Time Step: $tstep, \
                    Time: $t_print, MSE: $l_print, Δt: $Δt_print")
            
                nlsp = t1, Δt, t0, p0, u0

                p1, nlssol = nonlinleastsq(
                    model, p0, batch, nlssolve;
                    residual, nlsp,
                    maxiters = nlsmaxiters, abstol = nlsabstol,
                )

                nlsmse = sum(abs2, nlssol.resid) / length(nlssol.resid)
                nlsinf = norm(nlssol.resid, Inf)
                println("Nonlinear Steps: $(nlssol.stats.nsteps), \
                    MSE: $(round(nlsmse, sigdigits = 8)), \
                    ||∞: $(round(nlsinf, sigdigits = 8)), \
                    Ret: $(nlssol.retcode)"
                )
            end

            p1
        elseif projection_type isa Val{:PODGalerkin}
            # dU/dp (N, n), dU/dt (N,)
            J0 = dudp(model, xbatch, p0; autodiff = autodiff_space, ϵ = ϵ_space)
            rhs0 = dudtRHS(model, xbatch, p0, t0, md)

            dpdt0 = J0 \ vec(rhs0)
            p1 = p0 + Δt * dpdt0

            u1 = model(xbatch, p1)
            rhs1 = dudtRHS(model, xbatch, p1, t1, md)
            resid = timestepper_residual(Δt, u0, u1, rhs1)

            linmse = sum(abs2, resid) / length(resid)
            lininf = norm(resid, Inf)
            println("Linear Steps: $(0), \
                MSE: $(round(linmse, sigdigits = 8)), \
                ||∞: $(round(lininf, sigdigits = 8)), \
                Ret: $(nlssol.retcode)"
            )

            p1
        else
            error("Projection type must be `Val(:LSPG)`, or `Val(:PODGalerkin)`.")
        end
    
        #============================#
        # Evaluate, save, etc
        #============================#
        u1 = model(xbatch, p1)
    
        ps = push(ps, p1)
        us = push(us, u1)
        ts = push(ts, t1)
    
        if verbose
            println("#============================#")
        end
    
        #============================#
        # update states
        #============================#
        p0 = p1
        u0 = u1
        t0 = t1
    end

    code = mapreduce(getdata, hcat, ps) |> Lux.cpu_device()
    pred = mapreduce(adjoint, hcat, us) |> Lux.cpu_device()
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
    fps::Int = 300,
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

    mu = isnothing(mu) ? fill(nothing, Nb) |> Tuple : mu

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
    # train/test split
    #==============#
    _Udata = @view Udata[:, md._Ib, :] # un-normalized
    Udata_ = @view Udata[:, md.Ib_, :]

    #==============#
    # from training data
    #==============#

    # _data, _, _ = makedata_autodecode(datafile)
    # _Icode, _xdata = _data[1]
    # _xdata = unnormalizedata(_xdata, md.x̄, md.σx)
    #
    # model = NeuralSpaceModel(NN, st, _Icode, md.x̄, md.σx, md.ū, md.σu) |> device
    # _Upred = model(_xdata |> device, p |> device) |> Lux.cpu_device()
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
    #         idx = LinRange(1, size(Up, 2), 11) .|> Base.Fix1(round, Int)
    #         plt = plot(;title, xlabel, ylable)
    #         plot!(Xdata, Ud[:, idx], w = 2.0, label = "True")
    #         plot!(Xdata, Up[:, idx], w = 2.0, label = "Pred")
    #         png(plt, "train$(k)")
    #
    #         anim = animate1D(Ud, Up, Xdata, Tdata;
    #             w = 2, xlabel, ylabel, title)
    #         gif(anim, joinpath(outdir, "train$(k).gif"), fps)
    #     end
    # end

    #==============#
    # inference (via data regression)
    #==============#

    # decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    #
    # p0 = _code[2].weight[:, 1]
    # Icode = ones(Int32, 1, Nx)
    #
    # NN, p0, st = freeze_autodecoder(decoder, p0; rng)
    # model = NeuralSpaceModel(NN, st, Icode, md.x̄, md.σx, md.ū, md.σu)
    #
    # for k in 1:1 # axes(mu, 1)
    #     Ud = Udata[:, k, :]
    #     data = (Xdata, Ud, Tdata)
    #
    #     _, Up, _ = infer_autodecoder(model, data, p0; device, verbose)
    #
    #     if makeplot
    #         xlabel = "x"
    #         ylabel = "u(x, t)"
    #
    #         _mu   = mu[k]
    #         title = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"
    #         _name = k in md._Ib ? "infer_train$(k)" : "infer_test$(k)"
    #
    #         idx = LinRange(1, size(Up, 2), 11) .|> Base.Fix1(round, Int)
    #         plt = plot(;title, xlabel, ylabel)
    #         plot!(Xdata, Up[:, idx], w = 2.0, label = "Pred", s = :solid)
    #         plot!(Xdata, Ud[:, idx], w = 2.0, label = "True", s = :dash)
    #         # png(plt, _name)
    #         display(plt)
    #
    #         # anim = animate1D(Ud, Up, Xdata, Tdata;
    #         #     w = 2, xlabel, ylabel, title)
    #         # gif(anim, joinpath(outdir, "$(_name).gif"), fps)
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
    #         # gif(anim, joinpath(outdir, "$(_name).gif"), fps)
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

    # begin
    #     k = 1
    #     i = 100
    #
    #     decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    #     p0 = _code[2].weight[:, i]
    #
    #     plt = GeometryLearning.plot_derivatives1D_autodecoder(decoder, Xdata, p0, md,
    #         second_derv = false)
    #     png(plt, joinpath(outdir, "derv"))
    #     display(plt)
    # end

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

    plot_training(STATS...) |> display

    model, STATS
end

#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")

modeldir = joinpath(@__DIR__, "model3")
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
