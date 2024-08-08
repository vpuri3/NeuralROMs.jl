#
using NeuralROMs
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

    # isdir(modeldir) && rm(modeldir; recursive = true)
    projectdir = joinpath(modeldir, "project$(case)")
    mkpath(projectdir)

    #--------------------------------------------#
    # make data
    #--------------------------------------------#

    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile; verbose)

    # get sizes
    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    if verbose
        println("PROJECT_NGN: input size: $in_dim")
        println("PROJECT_NGN: output size: $out_dim")
    end

    # subsample in space
    Udata = @view Udata[:, data_kws.Ix, :, data_kws.It]
    Xdata = @view Xdata[:, data_kws.Ix]
    Tdata = @view Tdata[data_kws.It]

    # normalize
    Xnorm, x̄, σx = normalize_x(Xdata)
    Unorm, ū, σu = normalize_u(Udata)

    readme = ""
    data_kws = (; case, data_kws...,)
    metadata = (; ū, σu, x̄, σx,
        data_kws, md_data, readme,
    )

    _data = (Xnorm, Unorm[:, :, case, begin]) .|> Array
    periods = -(-(get_prob_domain(prob)...)) ./ σx

    model, ST, metadata = makemodelfunc(_data, train_params, periods, metadata, projectdir; rng, verbose, device)

    #-------------------------------------------#
    # visualize
    #-------------------------------------------#

    if makeplot
        Upred = eval_model(model, Xdata; batchsize = 10, device)

        for od in 1:out_dim
            Nx = length(Xdata)

            ud = Udata[od, :, case, 1]
            up = Upred[od, :]
            nr = sum(abs2, ud) / Nx |> sqrt
            er = (up - ud) / nr
            er = sum(abs2, er) / Nx .|> sqrt

            println("[ngProject] out_dim: $out_dim \t Error $(er * 100)%.")

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
    #==============#
    # load data/model
    #==============#

    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile; verbose)

    # load model
    (NN, p0, st), md = loadmodel(modelfile)

    # get sizes
    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    # subsample in space
    Udata = @view Udata[:, data_kws.Ix, :, data_kws.It]
    Xdata = @view Xdata[:, data_kws.Ix]
    Tdata = @view Tdata[data_kws.It]

    #==============#
    mkpath(modeldir)
    #==============#

    #==============#
    # freeze layers (??)
    #==============#
    # IDEA: partial freezing? So not exactly ROMs.
    # How do you guarantee expressivity?
    # NN, p0, st = freeze_decoder(decoder, length(p0); rng, p0)

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
    autodiff = AutoForwardDiff()
    autodiff_xyz = AutoForwardDiff()

    # solver
    linsolve = QRFactorization()
    linesearch = LineSearch()
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    nlsmaxiters = 20

    # scheme
    scheme = if scheme === :GalerkinProjection
        # abstol_inf, abstol_mse
        GalerkinProjection(linsolve, T(1f-3), T(1f-6))
    elseif scheme === :LSPG
        # abstol_nls, abstol_inf, abstol_mse
        LeastSqPetrovGalerkin(nlssolve, nlsmaxiters, T(1f-6), T(1f-3), T(1f-6))
    end

    #==============#
    # Hyper-reduction
    #==============#

    IX = isnothing(hyper_indices) ? Colon() : hyper_indices

    U0 = @view Udata[:, IX, case, 1]
    Xd = @view Xdata[:, IX]
    Td = @view Tdata[:]

    # create data arrays
    data = (Xd, U0, Td)
    data = map(x -> T.(x), data) # ensure no subarrays

    #==============#
    # evolve
    #==============#

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
    fieldplot(Xdata, Tdata, Ud, Up, ps, get_prob_grid(grid), modeldir, "evolve", case)

    # save files
    filename = joinpath(modeldir, "evolve$(case).jld2")
    jldsave(filename; Xdata, Tdata, Udata = Ud, Upred = Up, Ppred = ps)

    # return
    (Xdata, Tdata, Ud, Up, ps), statsROM
end
#======================================================#
#
