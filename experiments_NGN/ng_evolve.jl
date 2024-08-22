#
using NeuralROMs
using OrdinaryDiffEq
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2, Setfield, LaTeXStrings         # vis / save
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
    device = Lux.gpu_device(),
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
                plt = plot(; xlabel = "x", ylabel = "y", legend = false)
                plot!(plt, vec(Xdata), ud; c = :black, w = 4)
                plot!(plt, vec(Xdata), up; c = :red  , w = 2)
                png(plt, joinpath(projectdir, "plt$od.png"))
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
    hyper_indices = nothing,

    verbose::Bool = true,
    benchmark::Bool = false,
    device = Lux.gpu_device(),
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
    timealg  = haskey(evolve_params, :timealg ) ? evolve_params.timealg  : EulerForward()
    scheme   = haskey(evolve_params, :scheme  ) ? evolve_params.scheme   : :GalerkinProjection
    adaptive = haskey(evolve_params, :adaptive) ? evolve_params.adaptive : true

    # autodiff
    ϵ_xyz = nothing
    autodiff_xyz = AutoForwardDiff()

    #==============#
    # Hyper-reduction
    #==============#

    IX = isnothing(hyper_indices) ? Colon() : hyper_indices

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
        linsolve = QRFactorization()
        linesearch = LineSearch()
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
        linalg = SimpleGMRES() # good for block matrices (?)
        linalg = KrylovJL_GMRES()

        GalerkinCollocation(prob, model, p0, data[1]; linalg)
    end

    #==============#
    # evolve
    #==============#

    if !isa(scheme, GalerkinCollocation)
        args = (prob, device(model), timealg, scheme, (device(data[1:2])..., data[3]), device(p0 .|> T), Δt)
        kwargs = (; adaptive, autodiff_xyz, ϵ_xyz, learn_ic, verbose, device,)
    
        if benchmark # assume CUDA
            Logging.disable_logging(Logging.Warn)
            timeROM = @belapsed CUDA.@sync $evolve_model($args...; $kwargs...)
        end
    
        statsROM = if device isa LuxDeviceUtils.AbstractLuxGPUDevice
            CUDA.@timed _, ps, _ = evolve_model(args...; kwargs...)
        else
            @timed _, ps, _ = evolve_model(args...; kwargs...)
        end
    
        @set! statsROM.value = nothing
        if benchmark
            @set! statsROM.time  = timeROM
        end
        @show statsROM.time
    else

        if !isa(timealg, SciMLBase.AbstractODEAlgorithm)
            timealg = Tsit5()
            # timealg = Rodas5(autodiff = false)
        end

        # # AD 1D
        # timealg = Rodas5(autodiff = false)

        # explicit
        # timealg = Euler()
        # timealg = RK4()
        # timealg = SSPRK43()
        # timealg = Tsit5()

        # # implicit
        # timealg = ImplicitEuler(autodiff = false)
        # timealg = Rosenbrock32(autodiff = false)
        # timealg = Rodas5(autodiff = false)

        # for autodiff = true, use PreallocationTools.dual_cache ?

        # ODE Problem
        dt = 1f-4
        iip = false
        tspan = extrema(data[3])
        saveat = data[3]

        odecb = begin
            function affect!(int)
                if int.iter % 1 == 0
                    println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
                end
            end
            DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
        end

        odefunc = ODEFunction{iip}(scheme)# ; jac)
        odeprob = ODEProblem(odefunc, getdata(p0), tspan; saveat)
        integrator = SciMLBase.init(odeprob, timealg; dt)#, callback = odecb)

        if benchmark
            if device isa LuxDeviceUtils.AbstractLuxGPUDevice
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
    
    # return
    (Xdata, Tdata, Ud, Up, ps), statsROM
end
#======================================================#
#
