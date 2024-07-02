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

#======================================================#

function pca_basis(
    R::Integer,
    u::AbstractMatrix;
    device = Lux.cpu_device(),
)
    if length(u) > 10^8
        device = Lux.cpu_device()
    end

    F = svd(u |> device)
    F.U[:, 1:R] |> Lux.cpu_device()
end

#======================================================#
function makedata_PCA(
    datafile::String;
    Ix = Colon(),  # subsample in space
    _Ib = Colon(), # train/test split in batches
    Ib_ = Colon(), # disregard Ib_. set to everything but _Ib
    _It = Colon(), # train/test split in time
    It_ = Colon(),
)
    # load data
    x, t, mu, u, md_data = loaddata(datafile)

    in_dim  = size(x, 1)
    out_dim = size(u, 1)

    Ib_ = setdiff(1:size(u, 3), _Ib)

    # normalize
    x, x̄, σx = normalize_x(x)
    u, ū, σu = normalize_u(u)
    # t, t̄, σt = normalize_t(t)

    # subsample, train/test split
    _x = @view x[:, Ix]
    x_ = @view x[:, Ix]

    _u = @view u[:, Ix, _Ib, _It]
    u_ = @view u[:, Ix, Ib_, It_]

    _t = @view t[_It]
    t_ = @view t[It_]

    Nx = size(_x, 2)
    @assert size(_u, 2) == size(_x, 2) "size(_u): $(size(_u)), size(_x): $(size(_x))"

    println("Using $Nx sample points per trajectory.")

    _Ns = size(_u, 3) * size(_u, 4) # number of codes i.e. # trajectories
    Ns_ = size(u_, 3) * size(u_, 4)

    println("$_Ns / $Ns_ trajectories in train/test sets.")

    readme = "Train/test on the same trajectory."

    makedata_kws = (; Ix, _Ib, Ib_, _It, It_)

    metadata = (; ū, σu, x̄, σx,
        Nx, _Ns, Ns_, mu,
        makedata_kws, md_data, readme,
    )

    (_x, _u, _t), (x_, u_, t_), metadata
end

#======================================================#
function train_PCA(
    datafile::String,
    modeldir::String,
    R::Int;
    rng::Random.AbstractRNG = Random.default_rng(),
    makeplot::Bool = true,
    makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :,),
    device = Lux.cpu_device(),
)
    _data, data_, md = makedata_PCA(datafile; makedata_kws...)

    #==============================#
    # load data
    #==============================#

    _x, _u, _t = _data
    x_, u_, t_ = _data

    out_dim, Nx, _Nb, _Nt = size(_u)
    out_dim, Nx, Nb_, Nt_ = size(u_)

    # parameters
    _Ib = isa(makedata_kws._Ib, Colon) ? (1:size(_u, 3)) : makedata_kws._Ib
    Ib_ = isa(makedata_kws.Ib_, Colon) ? (1:size(u_, 3)) : makedata_kws.Ib_

    _It = isa(makedata_kws._It, Colon) ? (1:size(_u, 4)) : makedata_kws._It
    It_ = isa(makedata_kws.It_, Colon) ? (1:size(u_, 4)) : makedata_kws.It_

    # misc
    if out_dim == 1
        _u = reshape(_u, Nx, _Nb, _Nt)
        u_ = reshape(u_, Nx, Nb_, Nt_)
    else
        @warn "PCA for out_dim > 1 creates basis matrix only for the first output field"
        _u = _u[1, :, :, :]
        u_ = u_[1, :, :, :]
    end

    #==============================#
    # SVD
    #==============================#

    _udata = reshape(_u, Nx, :) |> copy
    udata_ = reshape(u_, Nx, :)

    P = pca_basis(R, copy(_udata); device)

    #==============================#

    mkpath(modeldir)

    # save model
    modelfile = joinpath(modeldir, "model.jld2")
    jldsave(modelfile; Pmatrix = P, metadata = md)

    #==============================#
    # visualize
    #==============================#

    modeldir = dirname(modelfile)
    outdir = joinpath(modeldir, "results")
    isdir(outdir) && rm(outdir; recursive = true)
    mkpath(outdir)

    if makeplot

        _a = P' * _udata
        _upred = P  * _a

        a_ = P' * udata_
        upred_ = P  * a_

        _v = reshape(_upred, Nx, _Nb, _Nt)
        v_ = reshape(upred_, Nx, Nb_, Nt_)

        xlabel = "x"
        ylabel = "u(x, t)"

        # grid = get_prob_grid(prob)
        #
        # for case in axes(_Ib, 1)
        #     u = _u[:, :, case, :]
        #     v = _v[:, :, case, :]
        #     fieldplot(_x, _t, u, v, grid, outdir, "train", case)
        # end

        for k in 1:length(_Ib)
            u = @view _u[:, k, :]
            v = @view _v[:, k, :]

            title = "Case $(_Ib[k])"

            It = LinRange(1, size(u, 2), 5) .|> Base.Fix1(round, Int)

            plt = plot(; title, xlabel, ylabel)
            plot!(plt, _x[1,:], u[:, It], linewidth=3, c = :black)
            plot!(plt, _x[1,:], v[:, It], linewidth=3, c = :red)
            png(plt, joinpath(modeldir, "train_$k"))
            display(plt)

            # anim = animate1D(u, v, _x[1,:], _t; linewidth=2, xlabel, ylabel, title)
            # gif(anim, joinpath(modeldir, "train$(k).gif"), fps=30)
        end

        for k in 1:length(Ib_)
            u = @view u_[:, k, :]
            v = @view v_[:, k, :]

            title = "Case $(Ib_[k])"

            It = LinRange(1, size(u, 2), 5) .|> Base.Fix1(round, Int)
            
            plt = plot(; title, xlabel, ylabel)
            plot!(plt, _x[1,:], u[:, It], linewidth=3, c = :black)
            plot!(plt, _x[1,:], v[:, It], linewidth=3, c = :red)
            png(plt, joinpath(modeldir, "test_$k"))
            display(plt)

            # anim = animate1D(u, v, x_[1,:], t_; linewidth=2, xlabel, ylabel, title)
            # gif(anim, joinpath(modeldir, "test$(k).gif"), fps=30)
        end
    end

    nothing
end
#======================================================#

function evolve_PCA(
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

    verbose::Bool = true,
    device = Lux.cpu_device(),
)
    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile)

    # load model
    model = jldopen(modelfile)
    P  = model["Pmatrix"]
    md = model["metadata"]
    close(model)

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
    # get initial state
    #==============#

    # only out_dim == 1 supported
    pl, Ul = begin
        Ud_norm = normalizedata(Ud[1, :, :], md.ū, md.σu)
        pl = P' * Ud_norm

        Ul = unnormalizedata(P * pl, md.ū, md.σu)
        Ul = reshape(Ul, 1, size(Ul)...)
        Ul = repeat(Ul, out_dim, 1)

        pl, Ul
    end

    p0 = pl[:, 1]

    #==============#
    # get model
    #==============#
    grid = get_prob_grid(prob)
    model = PCAModel(P, Xdata, grid, out_dim, md)

    #==============#
    # evolve
    #==============#
    linsolve = QRFactorization()
    autodiff = AutoForwardDiff()
    linesearch = LineSearch()
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    nlsmaxiters = 10

    Δt = isnothing(Δt) ? -(-(extrema(Tdata)...)) / 100.0f0 : Δt

    if isnothing(scheme)
        scheme  = GalerkinProjection(linsolve, 1f-3, 1f-6) # abstol_inf, abstol_mse
    end

    @time _, ps, Up = evolve_model(
        prob, model, timealg, scheme, data, p0, Δt;
        adaptive, autodiff_xyz, ϵ_xyz, verbose, device,
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
    filename = joinpath(outdir, "evolve$(case).jld2")
    jldsave(filename; Xdata, Tdata, Udata = Ud, Upred = Up, Ppred = ps, Plrnd = pl, Ulrnd = Ul)

    # print error metrics
    @show sqrt(mse(Up, Ud) / mse(Ud, 0 * Ud))
    @show norm(Up - Ud, Inf) / sqrt(mse(Ud, 0 * Ud))

    Xdata, Tdata, Ud, Up, ps
end

#======================================================#
function postprocess_PCA(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    makeplot::Bool = true,
    verbose::Bool = true,
    fps::Int = 300,
    device = Lux.cpu_device(),
)
    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile)

    #==============#
    # Evolve
    #==============#
    Nb = size(Udata, 3)

    for case in 1:Nb
        evolve_PCA(prob, datafile, modelfile, case; rng, device)
    end
    
    # model = jldopen(modelfile)
    # P  = model["Pmatrix"]
    # _ps = P' * Udata[1, :, :, :]

end

#======================================================#
nothing
