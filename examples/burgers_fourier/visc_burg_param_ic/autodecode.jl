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
    σu = sum(abs2, u .- ū) / length(u) |> sqrt
    u  = normalizedata(u, ū, σu)

    # normalize space
    x̄  = sum(x) / length(x)
    σx = sum(abs2, x .- x̄) / length(x) |> sqrt
    x  = normalizedata(x, x̄, σx)

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

#========================================================#
function evolve_autodecoder(
    prob::AbstractPDEProblem,
    decoder::NTuple{3, Any},
    metadata::NamedTuple,
    data::Tuple,
    p0::AbstractVector;
    rng::Random.AbstractRNG = Random.default_rng(),
    nlssolve = nothing,
    linsolve = nothing,
    device = Lux.cpu_device(),
    verbose::Bool = true,
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
    model = NeuralSpaceModel(NN, st, Icode,
        metadata.x̄, metadata.σx, metadata.ū, metadata.σu)

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
    abstol = T(1f-6)
    Δt = T(1f-3)

    autodiff_space = AutoForwardDiff()
    ϵ_space = nothing

    # autodiff_space = AutoFiniteDiff()
    # ϵ_space = 0.005f0

    # timestepper = EulerBackward()
    timestepper = EulerForward()

    # scheme = LeastSqPetrovGalerkin(nlsolve)
    # scheme = PODGalerkin(linsolve)

    projection_type = Val(:PODGalerkin)

    # projection_type = Val(:LSPG)
    # Δt = T(1f-4)

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
        termination_condition = AbsTerminationMode(),
        abstol,
        verbose,
    )
    u0 = model(xbatch, p0)

    if verbose
        println("#============================#")
    end

    #============================#
    # Set up solver
    #============================#
    residual = make_residual(prob, timestepper;
        autodiff = autodiff_space,
        ϵ = ϵ_space,
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

        p1 = if projection_type isa Val{:LSPG}
            nlsp = t1, Δt, t0, p0, u0

            # TODO - try the CROM way?

            p1, nlssol = nonlinleastsq(
                model, p0, batch, nlssolve;
                residual, nlsp, maxiters = nlsmaxiters, abstol,
            )

            _mse = sum(abs2, nlssol.resid) / length(nlssol.resid)
            _inf = norm(nlssol.resid, Inf)
            println("\tNonlinear Steps: $(nlssol.stats.nsteps), \
                MSE: $(round(_mse, sigdigits = 8)), \
                ||∞: $(round(_inf, sigdigits = 8)), \
                Ret: $(nlssol.retcode)"
            )

            if adaptive_timestep
                if (_mse < abstol) & (nlssol.stats.nsteps < 4)
                    Δt *= T(2f0)
                end

                while (_mse > abstol)
                    if Δt < T(1f-4)
                        println("Δt = $Δt")
                        break
                    end

                    Δt /= T(2f0)
                    t1 = t0 + Δt

                    l_print = round(_mse; sigdigits = 6)
                    t_print = round(t1; sigdigits = 6)
                    Δt_print = round(Δt; sigdigits = 6)

                    println("REPEATING Time Step: $tstep, \
                        Time: $t_print, MSE: $l_print, Δt: $Δt_print")

                    nlsp = t1, Δt, t0, p0, u0

                    p1, nlssol = nonlinleastsq(
                        model, p0, batch, nlssolve;
                        residual, nlsp,
                        maxiters = nlsmaxiters, abstol,
                    )

                    _mse = sum(abs2, nlssol.resid) / length(nlssol.resid)
                    _inf = norm(nlssol.resid, Inf)
                    println("Nonlinear Steps: $(nlssol.stats.nsteps), \
                        MSE: $(round(_mse, sigdigits = 8)), \
                        ||∞: $(round(_inf, sigdigits = 8)), \
                        Ret: $(nlssol.retcode)"
                    )
                end
            end

            p1
        elseif projection_type isa Val{:PODGalerkin}
            # dU/dp (N, n), dU/dt (N,)
            # J0 = dudp(model, xbatch, p0; autodiff = autodiff_space, ϵ = ϵ_space)
            J0 = dudp(model, xbatch, p0; autodiff)
            rhs0 = dudtRHS(prob, model, xbatch, p0, t0; autodiff = autodiff_space, ϵ = ϵ_space)

            dpdt0 = J0 \ vec(rhs0)
            p1 = apply_timestep(timestepper, Δt, p0, dpdt0)

            u1 = model(xbatch, p1)
            rhs1 = dudtRHS(prob, model, xbatch, p1, t1)
            resid = compute_residual(timestepper, Δt, u0, u1, rhs1)

            _mse = sum(abs2, resid) / length(resid)
            _inf = norm(resid, Inf)
            println("Linear Steps: $(0), \
                MSE: $(round(_mse, sigdigits = 8)), \
                ||∞: $(round(_inf, sigdigits = 8))"
            )

            p1

            if adaptive_timestep
                if (_mse < abstol) & (nlssol.stats.nsteps < 4)
                    Δt *= T(2f0)
                end

                while (_inf > 1f-3)
                    if Δt < T(1f-4)
                        println("Δt = $Δt")
                        break
                    end

                    Δt /= T(2f0)
                    t1 = t0 + Δt

                    l_print = round(_mse; sigdigits = 6)
                    t_print = round(t1; sigdigits = 6)
                    Δt_print = round(Δt; sigdigits = 6)

                    println("REPEATING Time Step: $tstep, \
                        Time: $t_print, MSE: $l_print, Δt: $Δt_print")

                    J0 = dudp(model, xbatch, p0; autodiff)
                    rhs0 = dudtRHS(prob, model, xbatch, p0, t0; autodiff = autodiff_space, ϵ = ϵ_space)

                    dpdt0 = J0 \ vec(rhs0)
                    p1 = apply_timestep(timestepper, Δt, p0, dpdt0)

                    u1 = model(xbatch, p1)
                    rhs1 = dudtRHS(prob, model, xbatch, p1, t1)
                    resid = compute_residual(timestepper, Δt, u0, u1, rhs1)

                    _mse = sum(abs2, resid) / length(resid)
                    _inf = norm(resid, Inf)
                    println("Linear Steps: $(0), \
                        MSE: $(round(_mse, sigdigits = 8)), \
                        ||∞: $(round(_inf, sigdigits = 8))"
                    )
                end
            end

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
    mu = isa(mu, AbstractArray) ? vec(mu) : mu

    # subsample in space
    Ix = 1:8:Nx
    Udata = @view Udata[Ix, :, :]
    Xdata = @view Xdata[Ix]
    Nx = length(Xdata)

    #==============#
    # load model
    #==============#
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"] # (; ū, σu, _Ib, Ib_, _It, It_, readme)
    close(model)

    # TODO - rm after retraining this model
    @set! md.σx = sqrt(md.σx)
    @set! md.σu = sqrt(md.σu)

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
    # _xdata, _Icode = _data[1]
    # _xdata = unnormalizedata(_xdata, md.x̄, md.σx)
    #
    # model = NeuralSpaceModel(NN, st, _Icode, md.x̄, md.σx, md.ū, md.σu) |> device
    # _Upred = model(_xdata |> device, p |> device) |> Lux.cpu_device()
    # _Upred = reshape(_Upred, Nx, length(md._Ib), Nt)
    #
    # for k in 1:1 # length(md._Ib)
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
    #         idx_pred = LinRange(1, size(Ud, 2), 10) .|> Base.Fix1(round, Int)
    #         idx_data = idx_pred
    #
    #         upred = Up[:, idx_pred]
    #         udata = Ud[:, idx_data]
    #
    #         Iplot = 1:32:Nx
    #
    #         plt = plot(xlabel = "x", ylabel = "u(x, t)", legend = false)
    #         plot!(plt, Xdata, upred, w = 2, palette = :tab10)
    #         scatter!(plt, Xdata[Iplot], udata[Iplot, :], w = 1, palette = :tab10)
    #         png(plt, "train$(k)")
    #
    #         anim = animate1D(Ud, Up, Xdata, Tdata;
    #             w = 2, xlabel, ylabel, title)
    #         gif(anim, joinpath(outdir, "train$(k).gif"); fps)
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
    #         idx = LinRange(1, size(Ud, 2), 101) .|> Base.Fix1(round, Int)
    #         plt = plot(;title, xlabel, ylabel, legend = false)
    #         plot!(Xdata, Up[:, idx], w = 2.0,  s = :solid)
    #         # plot!(Xdata, Ud[:, idx], w = 4.0,  s = :dash)
    #         # png(plt, _name)
    #         display(plt)
    #
    #         # anim = animate1D(Ud, Up, Xdata, Tdata;
    #         #     w = 2, xlabel, ylabel, title)
    #         # gif(anim, joinpath(outdir, "$(_name).gif"); fps)
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
    #     @time _, Up, Tpred = evolve_autodecoder(prob, decoder, md, data, p0;
    #         rng, device, verbose)
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
    #         # gif(anim, joinpath(outdir, "$(_name).gif"); fps)
    #    
    #         _plt = plot(title = "μ=$_mu", xlabel = "x", ylabel = "x", legend = false)
    #         plot!(_plt, Xdata, Up[:, 1:10:min(size(Up, 2), 300)])
    #         png(_plt, joinpath(outdir, "evolve_$k"))
    #     end
    # end

    #==============#
    # check derivative
    #==============#
    # begin
    #     i = 1700
    #
    #     decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    #     p0 = _code[2].weight[:, i] # (l, 3k)
    #
    #     plt = GeometryLearning.plot_derivatives1D_autodecoder(
    #         decoder, Xdata, p0, md, second_derv = false,
    #         # autodiff = AutoFiniteDiff(),
    #         # ϵ=2f-2
    #     )
    #     png(plt, joinpath(outdir, "derv"))
    #     display(plt)
    # end

    begin
        k = 7
        Ud = Udata[:, k, :]
        data = (Xdata, Ud, Tdata)
    
        decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
        p0 = _code[2].weight[:, 1]

        prob = BurgersInviscid1D()
        # prob = BurgersViscous1D(1/(4f3))
        @time _, Up, Tpred = evolve_autodecoder(prob, decoder, md, data, p0;
            rng, device, verbose)

        idx_pred = LinRange(1, size(Up, 2), 10) .|> Base.Fix1(round, Int)
        t_pred   = Tpred[idx_pred]
    
        idx_data = Tuple(findmin(abs.(Tdata .- t))[2] for t in t_pred)
        idx_data = [idx_data...]
    
        upred = Up[:, idx_pred]
        udata = Ud[:, idx_data]
    
        Iplot = 1:32:Nx
    
        plt = plot(xlabel = "x", ylabel = "u(x, t)", legend = false)
        plot!(plt, Xdata, upred, w = 2, palette = :tab10)
        scatter!(plt, Xdata[Iplot], udata[Iplot, :], w = 1, palette = :tab10)
    
        error = sum(abs, (upred - udata).^2) / length(udata)
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
    l::Int, # latent space size
    h::Int, # num hidden layers
    w::Int, # hidden layer width
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
    E = 1400
    lrs = (1f-2, 1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    opts = Optimisers.Adam.(lrs)
    nepochs = (20, Int.(E/7*ones(7))...)
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

    _batchsize, batchsize_  = 1024 .* (10, 3000)

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#
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
    # if Lux.parameterlength(decoder) < 35_000
    #     opts = (opts..., LBFGS(),)
    #     nepochs = (nepochs..., round(Int, E / 10))
    #     schedules = (schedules..., Step(1f0, 1f0, Inf32))
    # end

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

# for l in (3, 8)
#     for h in (5,)
#         for w in (96, 128)
#             ll = lpad(l, 2, "0")
#             hh = lpad(h, 2, "0")
#             ww = lpad(w, 3, "0")
#
#             (l == 3) & (h == 5) & (w == 96)
#
#             modeldir = joinpath(@__DIR__, "model_dec_sin_$(ll)_$(hh)_$(ww)_reg")
#
#             # isdir(modeldir) && rm(modeldir, recursive = true)
#             # model, STATS = train_autodecoder(l, h, w, datafile, modeldir; device)
#
#             modelfile = joinpath(modeldir, "model_08.jld2")
#             outdir = joinpath(dirname(modelfile), "results")
#             postprocess_autodecoder(datafile, modelfile, outdir; rng, device,
#                 makeplot = true, verbose = true)
#         end
#     end
# end

for modeldir in (
    # joinpath(@__DIR__, "model_dec_sin_03_05_128_reg/"),
    # joinpath(@__DIR__, "model_dec_sin_08_05_096_reg/"),
    joinpath(@__DIR__, "model_dec_sin_08_05_128_reg/"),
)
    modelfile = joinpath(modeldir, "model_08.jld2")
    outdir = joinpath(dirname(modelfile), "results")
    postprocess_autodecoder(datafile, modelfile, outdir; rng, device,
        makeplot = true, verbose = true)
end

# modeldir = joinpath(@__DIR__, "model_dec_sin_08_05_128_reg/")
# modelfile = joinpath(modeldir, "model_08.jld2")
#
# outdir = joinpath(dirname(modelfile), "results")
# postprocess_autodecoder(datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)

#======================================================#
# IDEAS: controlling noisy gradients
# - L2 regularization on weights
# - Gradient supervision
# - secondary (corrector) network for gradient supervision
# - use ParameterSchedulers.jl
#======================================================#
# nothing
#
