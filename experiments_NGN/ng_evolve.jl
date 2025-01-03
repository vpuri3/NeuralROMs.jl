#
using NeuralROMs
using OrdinaryDiffEq
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2, UnPack, Setfield, LaTeXStrings # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using BenchmarkTools
import Logging

CUDA.allowscalar(false)

begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))
    BLAS.set_num_threads(nc)
end

#===========================================================#
function ngProject(
    prob::AbstractPDEProblem,
    datafile::String,
    modeldir::String,
    makemodelfunc,
    case::Int = 1;
    rng::Random.AbstractRNG = Random.default_rng(),
    train_params = (;),
    data_kws = (; Ix = :, It = :,),
    verbose::Bool = true,
    makeplot::Bool = true,
    device = gpu_device(),
)

    projectdir = joinpath(modeldir, "project$(case)")
    isdir(projectdir) && rm(projectdir; recursive = true)
    mkpath(projectdir)

    #--------------------------------------------#
    # make data
    #--------------------------------------------#

    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile; verbose)

    T = eltype(Xdata)

    # get sizes
    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    if verbose
        println("PROJECT_NGN: input dimension: $in_dim")
        println("PROJECT_NGN: output dimension: $out_dim")
    end

    # subsample in space
    Udata = @view Udata[:, data_kws.Ix, :, data_kws.It]
    Xdata = @view Xdata[:, data_kws.Ix]
    Tdata = @view Tdata[data_kws.It]

    Xnorm, x̄, σx = normalize_x(Xdata, [-1, 1])
    U0, ū, σu = normalize_u(Udata[:, :, case, begin], [0, 1])

    readme = ""
    data_kws = (; case, data_kws...,)
    metadata = (; ū, σu, x̄, σx,
        data_kws, md_data, readme,
    )

    data = (Xnorm, U0) .|> Array
    periods = repeat(T[2], in_dim)

    model, ST, metadata = makemodelfunc(data, train_params, periods, metadata, projectdir; rng, verbose, device)

    #-------------------------------------------#
    # visualize
    #-------------------------------------------#

    if makeplot
        neuralmodel = NeuralModel(model[1], model[3], metadata)
        Upred = eval_model(neuralmodel, Xdata, model[2]; batchsize = 10, device)

        for od in 1:out_dim
            Nx = length(Xdata)

            ud = Udata[od, :, case, 1]
            up = Upred[od, :]
            nr = sum(abs2, ud) / Nx |> sqrt
            er = (up - ud) / nr
            er = sum(abs2, er) / Nx .|> sqrt

            println("[ngProject] out_dim: $out_dim \t Error $(er).")

            if in_dim == 1
				imagefile = joinpath(projectdir, "plt$od.png")
                plt = plot(; xlabel = "x", ylabel = "y", legend = false)
                plot!(plt, vec(Xdata), ud; c = :black, w = 4)
                plot!(plt, vec(Xdata), up; c = :red  , w = 2)
                png(plt, imagefile)
				@info "saving plot at $(imagefile)"
            elseif in_dim == 2
            end
        end
    end

    model, ST, metadata
end

#======================================================#
function ngEvolve(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String,
    case::Integer;
    rng::Random.AbstractRNG = Random.default_rng(),
    data_kws = (; Ix = :, It = :),
    evolve_params = (;),
    learn_ic::Bool = false,

    verbose::Bool = true,
    benchmark::Bool = false,
    device = gpu_device(),
)
    mkpath(modeldir)

    #==============#
    # load data/model
    #==============#

    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile; verbose)

    # load model
    (NN, p0, st), md = loadmodel(modelfile)

    # subsample in space
    Udata = @view Udata[:, data_kws.Ix, :, data_kws.It]
    Xdata = @view Xdata[:, data_kws.Ix]
    Tdata = @view Tdata[data_kws.It]

    #==============#
    # make model
    #==============#
    model = NeuralModel(NN, st, md)

    #==============#
    # solver setup
    #==============#
    T        = haskey(evolve_params, :T       ) ? evolve_params.T        : Float32
    Δt       = haskey(evolve_params, :Δt      ) ? evolve_params.Δt       : -(-(extrema(Tdata)...)) |> T
    IX       = haskey(evolve_params, :IX      ) ? evolve_params.IX       : Colon()
    timealg  = haskey(evolve_params, :timealg ) ? evolve_params.timealg  : EulerForward()
    scheme   = haskey(evolve_params, :scheme  ) ? evolve_params.scheme   : :GalerkinProjection
    adaptive = haskey(evolve_params, :adaptive) ? evolve_params.adaptive : true

    # autodiff
    ϵ_xyz = nothing
    autodiff_xyz = AutoForwardDiff()

    #==============#
    # Hyper-reduction
    #==============#

    U0 = @view Udata[:, IX, case, 1]
    Xd = @view Xdata[:, IX]
    Td = @view Tdata[:]

    # create data arrays
    data = (Xd, U0, Td)
    data = map(x -> T.(x), data) # ensure no subarrays, correct eltype

    #==============#
    # convert eltypes
    #==============#
    if T ∉ (Float32, Float64)
        @error "Unsupported eltype $T detected. Choose T = Float32, or Float64"
    end

    if T === Float64
        @info "[NG_EVOLVE] Running calculation on CPU with Float64 precision."
        device = cpu_device()
    end

    # convert model eltype
    Tconv = T === Float32 ? f32 : f64
    model = Tconv(model)
    p0 = Tconv(p0)

    #==============#
    # scheme
    #==============#

    scheme = if scheme ∈ (:GalerkinProjection, :LSPG)
        autodiff = AutoForwardDiff()
        linsolve = QRFactorization(ColumnNorm())
        linesearch = LineSearchesJL()
        nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
        nlsmaxiters = 20

        abstol_inf, abstol_mse = if T === Float32
            T(1e-3), T(1e-6)
        elseif T === Float64
            T(1e-5), T(1e-10)
        end

        if scheme === :GalerkinProjection
            GalerkinProjection(linsolve, abstol_inf, abstol_mse)
        else
            LeastSqPetrovGalerkin(nlssolve, nlsmaxiters, T(1f-6), abstol_inf, abstol_mse)
        end
    elseif scheme === :GalerkinCollocation
        α = 0f0
        debug = false
        normal = false
        linalg = QRFactorization(ColumnNorm())

        # debug = true
        # debug = false
        #
        # normal = false
        # linalg = KrylovJL_LSMR()
        # linalg = QRFactorization(ColumnNorm())

        # α = 1f-4
        # normal = true
        # linalg = SimpleGMRES()
        # linalg = KrylovJL_GMRES()
        # linalg = QRFactorization(ColumnNorm())

        GalerkinCollocation(prob, model, p0, data[1]; linalg, normal, debug, α)
    end

    #==============#
    # evolve
    #==============#
    
    if isa(scheme, GalerkinCollocation)
        if !isa(timealg, SciMLBase.AbstractODEAlgorithm)
            # explicit: Euler(), RK4(), SSPRK43(), Tsit5()
            # implicit: ImplicitEuler(autodiff = false), Rosenbrock32(autodiff = false), Rodas5(autodiff = false)
            # for (autodiff = true), use PreallocationTools.dual_cache ?

            timealg = Tsit5()
            # timealg = Rodas5(autodiff = false) # linsolve = ??
        end

        iip = false
        tspan = extrema(data[3])
        saveat = data[3]

        dt = 1f-2
        callback = begin
            function affect!(int)
                if int.iter % 1 == 0
                    println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
                end
            end
            DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
        end

		@show timealg

        odefunc = ODEFunction{iip}(scheme)# ; jac)
        odeprob = ODEProblem(odefunc, getdata(p0), tspan; saveat)
        integrator = SciMLBase.init(odeprob, timealg; dt, callback)

        if benchmark
            if device isa MLDataDevices.AbstractGPUDevice
                timeROM  = @belapsed CUDA.@sync $solve!($integrator)
                statsROM = CUDA.@timed solve!(integrator)
            else
                timeROM  = @belapsed $solve!($integrator)
                statsROM = @timed solve!(integrator)
            end

            sol = statsROM.value
            @set! statsROM.value = nothing
            @set! statsROM.time  = timeROM
            @show statsROM.time
        else
            @time sol = solve!(integrator)
            statsROM = nothing
        end

        #==============#
        # # DAE problem
        # mass_matrix = true
        # scheme = GalerkinCollocation(prob, model, p0, data[1]; mass_matrix)
        #
        # daealg = DImplicitEuler()
        #
        # daefunc = DAEFunction{false}(scheme)#; jac)
        # daeprob = DAEProblem(daefunc, getdata(p0), getdata(p0), tspan; saveat)
        # integrator = SciMLBase.init(daeprob, daealg; dt)
        # solve!(integrator)
        #==============#

        @show sol.stats
        @show sol.retcode
        @show sol.stats.nf, sol.stats.nw, sol.stats.nsolve, sol.stats.njacs
        @assert SciMLBase.successful_retcode(sol)

        ps = Array(sol)
    else
        # LSPG, GalerkinProjection

        args = (prob, device(model), timealg, scheme, (device(data[1:2])..., data[3]), device(p0 .|> T), Δt)
        kwargs = (; adaptive, autodiff_xyz, ϵ_xyz, learn_ic, verbose, device,)
    
        if benchmark # assume CUDA
            Logging.disable_logging(Logging.Warn)
            timeROM = @belapsed CUDA.@sync $evolve_model($args...; $kwargs...)
        end
    
        statsROM = if device isa MLDataDevices.AbstractGPUDevice
            CUDA.@timed _, ps, _ = evolve_model(args...; kwargs...)
        else
            @timed _, ps, _ = evolve_model(args...; kwargs...)
        end

        @set! statsROM.value = nothing
        if benchmark
            @set! statsROM.time  = timeROM
        end

        @show statsROM.time
    end

    #==============#
    # analysis
    #==============#

    Ud = @view Udata[:, :, case, :]                        # get data
    Up = eval_model(model, Xdata, ps, getaxes(p0); device) # query decoder

    # print error metrics
    if verbose
        N = length(Up)
        Nr = sum(abs2, Up) / N |> sqrt # normalizer
        Ep = (Up - Ud) ./ Nr
    
        Er = sum(abs2, Ep) / N |> sqrt # rel-error
        Em = norm(Ep, Inf)             # ||∞-error
    
        println("Rel-Error: $(100 * Er) %")
        println("Max-Error: $(100 * Em) %")
    end
    
    # field visualizations
    fieldplot(Xdata, Tdata, Ud, Up, ps, get_prob_grid(prob), modeldir, "evolve", case)
    
    # save files
    filename = joinpath(modeldir, "evolve$(case).jld2")
    jldsave(filename; Xdata, Tdata, Udata = Ud, Upred = Up, Ppred = ps)
	@info "saving file $(filename)."
    
    # return
    (Xdata, Tdata, Ud, Up, ps), statsROM
end
#======================================================#
#
