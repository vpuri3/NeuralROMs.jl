#
using GeometryLearning
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
    F = svd(u |> device)
    F.U[:, 1:R] |> Lux.cpu_device()
end

#======================================================#
function makedata_PCA(
    datafile::String;
    Ix = Colon(), # subsample in space
    _Ib = Colon(), # train/test split in batches
    Ib_ = Colon(),
    _It = Colon(), # train/test split in time
    It_ = Colon(),
)
    #==============#
    # load data
    #==============#
    data = jldopen(datafile)
    t  = data["t"]
    x  = data["x"]
    u  = data["u"]
    mu = data["mu"]
    md_data = data["metadata"]
    close(data)

    @assert ndims(u) ∈ (3,4,)
    @assert x isa AbstractVecOrMat
    x = x isa AbstractVector ? reshape(x, 1, :) : x # (Dim, Npoints)

    if ndims(u) == 3 # [Nx, Nb, Nt]
        u = reshape(u, 1, size(u)...) # [1, Nx, Nb, Nt]
    end

    in_dim  = size(x, 1)
    out_dim = size(u, 1)

    println("input size $in_dim with $(size(x, 2)) points per trajectory.")
    println("output size $out_dim.")

    @assert eltype(x) === Float32
    @assert eltype(u) === Float32

    mu = isnothing(mu) ? fill(nothing, Nb) |> Tuple : mu
    mu = isa(mu, AbstractArray) ? vec(mu) : mu

    #==============#
    # normalize
    #==============#

    ū  = sum(u, dims = (2,3,4)) / (length(u) ÷ out_dim) |> vec
    σu = sum(abs2, u .- ū, dims = (2,3,4)) / (length(u) ÷ out_dim) .|> sqrt |> vec
    u  = normalizedata(u, ū, σu)

    x̄  = sum(x, dims = 2) / size(x, 2) |> vec
    σx = sum(abs2, x .- x̄, dims = 2) / size(x, 2) .|> sqrt |> vec
    x  = normalizedata(x, x̄, σx)

    #==============#
    # subsample in space, time
    #==============#
    _x = @view x[:, Ix]
    x_ = @view x[:, Ix]

    _t = @view t[_It]
    t_ = @view t[It_]

    #==============#
    # train/test split
    #==============#

    _u = @view u[:, Ix, _Ib, _It]
    u_ = @view u[:, Ix, Ib_, It_]

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
    @assert out_dim == 1 "work on Burgers 2D later"
    _u = reshape(_u, Nx, _Nb, _Nt)
    u_ = reshape(u_, Nx, Nb_, Nt_)

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

    if makeplot

        _a = P' * _udata
        _upred = P  * _a

        a_ = P' * udata_
        upred_ = P  * a_

        _v = reshape(_upred, Nx, _Nb, _Nt)
        v_ = reshape(upred_, Nx, Nb_, Nt_)

        xlabel = "x"
        ylabel = "u(x, t)"


        for k in 1:length(_Ib)
            u = @view _u[:, k, :]
            v = @view _v[:, k, :]

            _mu = md.mu[_Ib[k]]
            title  = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"

            it = LinRange(1, size(u, 2), 4) .|> Base.Fix1(round, Int)

            plt = plot(; title, xlabel, ylabel)
            plot!(plt, _x[1,:], u[:, it], linewidth=3, c = :black)
            plot!(plt, _x[1,:], v[:, it], linewidth=3, c = :red)
            png(plt, joinpath(modeldir, "train_$k"))
            display(plt)

            # anim = animate1D(u, v, _x[1,:], _t; linewidth=2, xlabel, ylabel, title)
            # gif(anim, joinpath(modeldir, "train$(k).gif"), fps=30)
        end

        for k in 1:length(Ib_)
            u = @view u_[:, k, :]
            v = @view v_[:, k, :]

            _mu = md.mu[_Ib[k]]
            title  = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"

            # it = LinRange(1, size(u, 2), 4) .|> Base.Fix1(round, Int)
            #
            # plt = plot(; title, xlabel, ylabel)
            # plot!(png, _x[1,:], u[:, it], linewidth=3, c = :black)
            # plot!(png, _x[1,:], v[:, it], linewidth=3, c = :red)
            # # png(plt, joinpath(modeldir, "test_$k"))

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
    timealg::GeometryLearning.AbstractTimeAlg = EulerForward(),
    adaptive::Bool = false,
    scheme::Union{Nothing, GeometryLearning.AbstractSolveScheme} = nothing,

    autodiff_xyz::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ_xyz::Union{Real, Nothing} = 1f-2,

    verbose::Bool = true,
    device = Lux.cpu_device(),
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

    @assert ndims(Udata) ∈ (3,4,)
    @assert Xdata isa AbstractVecOrMat
    Xdata = Xdata isa AbstractVector ? reshape(Xdata, 1, :) : Xdata # (Dim, Npoints)

    if ndims(Udata) == 3 # [Nx, Nb, Nt]
        Udata = reshape(Udata, 1, size(Udata)...) # [out_dim, Nx, Nb, Nt]
    end

    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    mu = isnothing(mu) ? fill(nothing, Nb) |> Tuple : mu
    mu = isa(mu, AbstractArray) ? vec(mu) : mu

    #==============#
    # load model
    #==============#
    model = jldopen(modelfile)
    P  = model["Pmatrix"]
    md = model["metadata"]
    close(model)

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
    U0_norm = normalizedata(U0, md.ū, md.σu)
    p0 = P' * vec(U0_norm)

    #==============#
    # get model
    #==============#
    model = PCAModel(P, Xdata, md)

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

    learn_ic = false

    @time _, ps, Up = evolve_model(prob, model, timealg, scheme, data, p0, Δt;
        learn_ic, nlssolve, nlsmaxiters, adaptive, autodiff_xyz, ϵ_xyz,
        verbose, device,
    )

    Xdata, Tdata, Ud, Up, ps
end

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

R = 32
prob = Advection1D(0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir = joinpath(@__DIR__, "PCA$(R)")
modelfile = joinpath(modeldir, "model.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1], Ib_ = [1], _It = :, It_ = :)

train_PCA(datafile, modeldir, R; makedata_kws, device,)
# x, t, ud, up, _ = evolve_PCA(prob, datafile, modelfile, 1; rng)
# plt = plot(vec(x), up[1,:, [1,end]]) |> display
#======================================================#
nothing
