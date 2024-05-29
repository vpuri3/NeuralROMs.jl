#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2                                 # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU

CUDA.allowscalar(false)

# using FFTW
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
end

include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "cases.jl"))

#======================================================#
function makedata_INR(
    datafile::String;
    Ix = Colon(), # subsample in space
    _Ib = Colon(), # train/test split in batches
    Ib_ = Colon(),
    _It = Colon(), # train/test split in time
    It_ = Colon(),
)
    # load data
    x, t, mu, u, md_data = loaddata(datafile)

    _Ib = isa(_Ib, Colon) ? (1:size(u, 3)) : _Ib
    Ib_ = setdiff(1:size(u, 3), _Ib)
    Ib_ = isempty(Ib_) ? _Ib : Ib_

    # normalize
    x, x̄, σx = normalize_x(x)
    u, ū, σu = normalize_u(u)
    t, t̄, σt = normalize_t(t)

    # subsample, test/train split
    _x = @view x[:, Ix]
    x_ = @view x[:, Ix]

    _u = @view u[:, Ix, _Ib, _It]
    u_ = @view u[:, Ix, Ib_, It_]

    Nx = size(_x, 2)
    @assert size(_u, 2) == size(_x, 2)

    # get dimensinos
    in_dim  = size(x, 1)
    out_dim = size(u, 1)

    _Ns = size(_u, 3) * size(_u, 4)
    Ns_ = size(u_, 3) * size(u_, 4)

    println("Using $Nx sample points per trajectory.")
    println("$_Ns / $Ns_ trajectories in train/test sets.")

    grid = if in_dim == 1
        (Nx,)
    elseif in_dim == 2
        md_data.grid
    end

    # make arrays
    _uperm = permutedims(_u, (2, 1, 3, 4)) # [Nx, out_dim, Nbatch, Ntime]
    uperm_ = permutedims(u_, (2, 1, 3, 4))

    _uperm = reshape(_uperm, Nx, in_dim, _Ns)
    uperm_ = reshape(uperm_, Nx, in_dim, Ns_)

    _xperm = permutedims(_x, (2, 1))
    xperm_ = permutedims(x_, (2, 1))

    _X = zeros(Float32, Nx, in_dim + out_dim, _Ns) # [x; u]
    X_ = zeros(Float32, Nx, in_dim + out_dim, Ns_)

    _X[:, begin:in_dim, :] .= _xperm
    X_[:, begin:in_dim, :] .= xperm_

    _X[:, in_dim+1:end, :] = _uperm
    X_[:, in_dim+1:end, :] = uperm_

    _U = reshape(_u, out_dim, grid..., _Ns)
    U_ = reshape(u_, out_dim, grid..., Ns_)

    readme = ""
    makedata_kws = (; Ix, _Ib, Ib_, _It, It_,)
    metadata = (; ū, σu, x̄, σx,
        Nx, _Ns, Ns_,
        makedata_kws, md_data, readme,
    )

    (_X, _U), (X_, U_), metadata
end

#======================================================#
function train_CINR(
    datafile::String,
    modeldir::String,
    NN::Lux.AbstractExplicitLayer,
    E::Int; # num epochs
    rng::Random.AbstractRNG = Random.default_rng(),
    warmup::Bool = true,
    _batchsize = nothing,
    batchsize_ = nothing,
    makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :,),
    cb_epoch = nothing,
    device = Lux.cpu_device(),
)

    _data, _, metadata = makedata_INR(datafile; makedata_kws...)
    dir = modeldir

    lossfun = function(NN, p, st, batch)
        x, ŷ = batch
        y, st = NN(x, p, st)
        loss = sum(abs2, ŷ - y) / length(ŷ)

        loss, st, (;)
    end

    _batchsize = isnothing(_batchsize) ? numobs(_data) ÷ 50 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data)      : batchsize_

    #--------------------------------------------#
    # optimizer
    #--------------------------------------------#
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, warmup)

    train_args = (; E, _batchsize,)
    metadata   = (; metadata..., train_args)

    display(NN)
    displaymetadata(metadata)

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_, 
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
        cb_epoch,
    )

    displaymetadata(metadata)

    plot_training(ST...) |> display

    model, ST, metadata
end
#======================================================#
function postprocess_CINR(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    makeplot::Bool = true,
    verbose::Bool = true,
    fps::Int = 300,
    device = Lux.cpu_device(),
    evolve_kw::NamedTuple = (;),
)
    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile)

    # load model
    (NN, p, st), md = loadmodel(modelfile)

    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    #==============#
    # setup train/test split
    #==============#
    _Ib = isa(md.makedata_kws._Ib, Colon) ? (1:size(Udata, 3)) : md.makedata_kws._Ib
    _It = isa(md.makedata_kws._It, Colon) ? (1:size(Udata, 4)) : md.makedata_kws._It

    Ib_ = setdiff(1:size(Udata, 3), _Ib)
    Ib_ = isempty(Ib_) ? _Ib : Ib_
    It_ = 1:size(Udata, 4)

    displaymetadata(md)

    #==============#
    # train/test split
    #==============#
    _Udata = @view Udata[:, :, _Ib, _It] # un-normalized
    Udata_ = @view Udata[:, :, Ib_, It_]

    #==============#
    # get encoder / decoer
    #==============#
    encoder, decoder = NeuralROMs.get_encoder_decoder(NN, p, st)
    grid = get_prob_grid(prob)

    #==============#
    # evaluate model
    #==============#

    modeldir = dirname(modelfile)
    outdir = joinpath(modeldir, "results")
    isdir(outdir) && rm(outdir; recursive = true)
    mkpath(outdir)

    #==============#
    # Evolve
    #==============#
    for case in union(_Ib, Ib_)
        evolve_CINR(prob, datafile, modelfile, case; rng, device, evolve_kw...,)
    end

    #==============#
    # Compare evolution with training plots
    #==============#

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
#
#======================================================#

function evolve_CINR(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String,
    case::Integer;
    rng::Random.AbstractRNG = Random.default_rng(),
    data_kws = (; Ix = :, It = :),
    Δt::Union{Real, Nothing} = nothing,
    timealg::NeuralROMs.AbstractTimeAlg = EulerForward(),
    adaptive::Bool = false,
    scheme::Union{Nothing, NeuralROMs.AbstractSolveScheme} = nothing,
    autodiff_xyz::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ_xyz::Union{Real, Nothing} = 1f-2,
    learn_ic::Bool = true,
    verbose::Bool = true,
    device = Lux.cpu_device(),
)
    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile)

    # load model
    (NN, p, st), md = loadmodel(modelfile)

    Nt = length(Tdata)
    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    #==============#
    # subsample in space
    #==============#
    Udata = @view Udata[:, data_kws.Ix, :, data_kws.It]
    Xdata = @view Xdata[:, data_kws.Ix]
    Tdata = @view Tdata[data_kws.It]
    Nx = size(Xdata, 2)

    Ud = Udata[:, :, case, :]
    U0 = Ud[:, :, 1]

    data = (Xdata, U0, Tdata)
    data = copy.(data) # ensure no SubArrays

    #==============#
    # get encoder / decoer
    #==============#
    encoder, decoder = NeuralROMs.get_encoder_decoder(NN, p, st)

    #==============#
    # get p0
    #==============#
    grid = get_prob_grid(prob)

    Ud_norm = normalizedata(Ud, md.ū, md.σu)
    Ud_perm = permutedims(Ud_norm, (2, 1, 3))
    Ud_resh = reshape(Ud_perm, grid..., out_dim, Nt) # [Nx, Ny, out_dim, Nt]

    _ps = encoder[1](Ud_resh, encoder[2], encoder[3])[1]
    p0 = _ps[:, 1]

    #==============#
    # make model
    #==============#
    decoder = freeze_decoder(decoder, length(p0); rng, p0)
    p0 = ComponentArray(p0, getaxes(decoder[2]))

    # model = NeuralModel(decoder[1], decoder[3], md)
    model = INRModel(decoder[1], decoder[3], Xdata, grid, md)

    #==============#
    # evolve
    #==============#
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()
    linesearch = LineSearch()
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    nlsmaxiters = 20

    Δt = isnothing(Δt) ? -(-(extrema(Tdata)...)) / 100.0f0 : Δt

    if isnothing(scheme)
        scheme  = GalerkinProjection(linsolve, 1f-3, 1f-6) # abstol_inf, abstol_mse
        # scheme = LeastSqPetrovGalerkin(nlssolve, nlsmaxiters, 1f-6, 1f-3, 1f-6)
    end

    @time _, ps, Up = evolve_model(
        prob, model, timealg, scheme, data, p0, Δt;
        adaptive, autodiff_xyz, ϵ_xyz, learn_ic, verbose, device,
    )

    #==============#
    # visualization
    #==============#

    modeldir = dirname(modelfile)
    outdir = joinpath(modeldir, "results")
    mkpath(outdir)

    # field visualizations
    grid = get_prob_grid(prob)
    fieldplot(Xdata, Tdata, Ud, Up, grid, outdir, "evolve", case)

    # parameter plots
    plt = plot(; title = L"$\tilde{u}$ distribution, case " * "$(case)")
    plt = make_param_scatterplot(ps, Tdata; plt, label = "Dynamics solve", color = :blues, cbar = false)
    png(plt, joinpath(outdir, "evolve_p_scatter_case$(case)"))

    plt = plot(; title = L"$\tilde{u}$ evolution, case " * "$(case)")
    plot!(plt, Tdata, ps', w = 3.0, label = "Dynamics solve")
    png(plt, joinpath(outdir, "evolve_p_case$(case)"))

    # save files
    filename = joinpath(outdir, "evolve$case.jld2")
    jldsave(filename; Xdata, Tdata, Udata = Ud, Upred = Up, Ppred = ps, Plrnd = _ps)

    Xdata, Tdata, Ud, Up, ps
end
#======================================================#
