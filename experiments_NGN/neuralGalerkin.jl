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

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "cases.jl") |> include

#===========================================================#

function ngProject(
    datafile::String,
    modeldir::String,
    case::Int = 1;
    train_params = (;),
    data_kws = (; Ix = :, It = :,),
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    makeplot::Bool = true,
    device = Lux.gpu_device(),
)

    isdir(modeldir) && rm(modeldir; recursive = true)
    projectdir = joinpath(modeldir, "projectT0")
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

    #--------------------------------------------#
    # get train params
    #--------------------------------------------#

    h = haskey(train_params, :h) ? train_params.h : 1
    w = haskey(train_params, :w) ? train_params.w : 10
    E = haskey(train_params, :E) ? train_params.E : 2100
    act = haskey(train_params, :act) ? train_params.act : sin

    γ = haskey(train_params, :γ) ? train_params.γ : 1f-2
    λ = haskey(train_params, :λ) ? train_params.λ : 0f-0

    _batchsize = haskey(train_params, :_batchsize) ? train_params._batchsize : nothing
    batchsize_ = haskey(train_params, :batchsize_) ? train_params.batchsize_ : nothing

    warmup = haskey(train_params, :warmup) ? train_params.warmup : true

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    periods = [2.0f0] ./ σx

    # periodic = NoOpLayer()
    periodic = PeriodicEmbedding(1:in_dim, periods)
    # periodic = PeriodicLayer(w, periods)

    decoder = begin
        if act ∈ (sin, cos)
            init_wt_in = scaled_siren_init(1f1)
            init_wt_hd = scaled_siren_init(1f0)
            init_wt_fn = glorot_uniform
            init_bias = rand32
        else
            init_wt_in = glorot_uniform
            init_wt_hd = glorot_uniform
            init_wt_fn = glorot_uniform
            init_bias = zeros32
        end

        use_bias_fn = false

        i = if periodic isa PeriodicEmbedding
            2 * in_dim
        elseif periodic isa PeriodicLayer
            w
        elseif periodic isa NoOpLayer
            in_dim
        end

        o = out_dim

        in_layer = Dense(i, w, act; init_weight = init_wt_in, init_bias)
        hd_layer = Dense(w, w, act; init_weight = init_wt_hd, init_bias)
        fn_layer = Dense(w, o     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

        Chain(in_layer, fill(hd_layer, h)..., fn_layer)
    end

    NN = Chain(; periodic, decoder)

    #-------------------------------------------#
    # training hyper-params
    #-------------------------------------------#

    _batchsize = isnothing(_batchsize) ? numobs(_data) ÷ 10 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data) ÷ 1  : batchsize_

    lossfun = mse

    idx = ps_W_indices(NN, :decoder; rng)
    weightdecay = IdxWeightDecay(0f0, idx)
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, warmup, weightdecay)

    #-------------------------------------------#

    train_args = (; h, w, E, λ, γ, _batchsize, batchsize_)
    metadata   = (; metadata..., train_args)

    display(NN)

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_, weight_decays = γ,
        opts, nepochs, schedules, early_stoppings,
        device, dir = projectdir, metadata, lossfun,
    )

    plot_training!(ST...) |> display

    # visualize model
    if makeplot
        if in_dim == 1
        elseif in_dim == 2
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

    # time-stepper
    Δt = haskey(evolve_params, :Δt) ? evolve_params.Δt : -(-(extrema(Tdata)...))
    timealg = haskey(evolve_params, :timealg) ? evolve_params.timealg : EulerForward()
    adaptive = haskey(evolve_params, :adaptive) ? evolve_params.adaptive : true

    # autodiff
    ϵ_xyz = nothing
    autodiff = autodiff_xyz = AutoForwardDiff()

    # solver
    linsolve = QRFactorization()
    linesearch = LineSearch()
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    nlsmaxiters = 20

    scheme = GalerkinProjection(linsolve, 1f-3, 1f-6) # abstol_inf, abstol_mse

    #==============#
    # Hyper-reduction
    #==============#

    IX = isnothing(hyper_indices) ? Colon() : hyper_indices

    U0 = @view Udata[:, IX, case, 1]
    Xd = @view Xdata[:, IX]
    Td = @view Tdata[:]

    # create data
    data = (Xd, U0, Td)
    data = Array.(data) # ensure no subarrays

    #==============#
    # evolve
    #==============#

    args = (prob, device(model), timealg, scheme, (device(data[1:2])..., data[3]), device(p0), Δt)
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
    grid = get_prob_grid(prob)
    fieldplot(Xdata, Tdata, Ud, Up, grid, modeldir, "evolve", case)
    
    # save files
    filename = joinpath(modeldir, "evolve$(case).jld2")
    jldsave(filename; Xdata, Tdata, Udata = Ud, Upred = Up, Ppred = ps)

    (Xdata, Tdata, Ud, Up, ps), statsROM
end

#======================================================#

function make_optimizer(
    E::Integer,
    warmup::Bool,
    weightdecay = nothing,
)
    lrs = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    Nlrs = length(lrs)

    # Grokking (https://arxiv.org/abs/2201.02177)
    # Optimisers.Adam(lr, (0.9f0, 0.95f0)), # 0.999 (default), 0.98, 0.95
    # https://www.youtube.com/watch?v=IHikLL8ULa4&ab_channel=NeelNanda
    opts = if isnothing(weightdecay)
        Tuple(
            Optimisers.Adam(lr) for lr in lrs
        )
    else
        Tuple(
            OptimiserChain(
                Optimisers.Adam(lr),
                weightdecay,
            )
            for lr in lrs
        )
    end

    nepochs = (round.(Int, E / (Nlrs) * ones(Nlrs))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, Nlrs)...,)

    if warmup
        opt_warmup = if isnothing(weightdecay)
            Optimisers.Adam(1f-2)
        else
            OptimiserChain(Optimisers.Adam(1f-2), weightdecay,)
        end
        nepochs_warmup = 10
        schedule_warmup = Step(1f-2, 1f0, Inf32)
        early_stopping_warmup = true

        ######################
        opts = (opt_warmup, opts...,)
        nepochs = (nepochs_warmup, nepochs...,)
        schedules = (schedule_warmup, schedules...,)
        early_stoppings = (early_stopping_warmup, early_stoppings...,)
    end

    opts, nepochs, schedules, early_stoppings
end

#===========================================================#

function ps_W_indices(
    NN,
    property::Union{Symbol, Nothing} = nothing;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    p = Lux.setup(copy(rng), NN)[1]
    p = ComponentArray(p)

    idx = Int32[]
    pprop = isnothing(property) ? p : getproperty(p, property)

    pNames = propertynames(pprop)
    pNum   = length(pNames)

    for i in 1:(pNum-1)
        lName = pNames[i]

        w = getproperty(pprop, lName).weight # reshaped array

        @assert ndims(w) == 2

        i = if w isa Base.ReshapedArray
            only(w.parent.indices)
        elseif w isa SubArray
            w.indices
        end

        println("[ps_W_indices]: Grabbing weight indices from [$i / $pNum] $(property) layer $(lName), size $(size(w)).")
        idx = vcat(idx, Int32.(i))
    end

    println("[ps_W_indices]: Passing $(length(idx)) / $(length(p)) $(property) parameters to IdxWeightDecay")

    idx
end
#===========================================================#
#
