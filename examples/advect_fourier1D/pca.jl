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
function pca_project(
    datafile::String,
    modeldir::String,
    R::Int;
    rng::Random.AbstractRNG = Random.default_rng(),
    makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :,),
    device = Lux.cpu_device(),
)
    _data, data_, md = makedata_PCA(datafile; makedata_kws...)

    # load data
    _x, _u, _t = _data
    x_, u_, t_ = _data

    out_dim, Nx, _Nb, _Nt = size(_u)
    out_dim, Nx, Nb_, Nt_ = size(u_)

    # parameters
    _Ib = isa(makedata_kws._Ib, Colon) ? (1:size(_u, 3)) : makedata_kws._Ib
    Ib_ = isa(makedata_kws.Ib_, Colon) ? (1:size(u_, 3)) : makedata_kws.Ib_

    _It = isa(makedata_kws._It, Colon) ? (1:size(_u, 4)) : makedata_kws._It
    It_ = isa(makedata_kws.It_, Colon) ? (1:size(u_, 4)) : makedata_kws.It_

    #################
    # misc
    #################
    @assert out_dim == 1 "work on Burgers 2D later"
    _u = reshape(_u, Nx, _Nb, _Nt)
    u_ = reshape(u_, Nx, Nb_, Nt_)

    _x = vec(_x)
    x_ = vec(x_)

    #################
    # SVD
    #################

    _udata = reshape(_u, Nx, :)
    udata_ = reshape(u_, Nx, :)

    F = svd(_udata)
    U = F.U[:, 1:R]

    # project data
    _a = U' * _udata
    _upred = U  * _a

    a_ = U' * udata_
    upred_ = U  * a_

    _v = reshape(_upred, Nx, _Nb, _Nt)
    v_ = reshape(upred_, Nx, Nb_, Nt_)
    #################

    mkpath(modeldir)

    for k in 1:length(_Ib)
        u = @view _u[:, k, :]
        v = @view _v[:, k, :]
        @show md

        _mu = md.mu[_Ib[k]]
        title  = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"

        anim = animate1D(u, v, _x, _t; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title)
        gif(anim, joinpath(modeldir, "train$(k).gif"), fps=30)
    end

    for k in 1:length(Ib_)
        u = @view u_[:, k, :]
        v = @view v_[:, k, :]

        _mu = md.mu[_Ib[k]]
        title  = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"

        anim = animate1D(u, v, x_, t_; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title)
        gif(anim, joinpath(modeldir, "test$(k).gif"), fps=30)
    end

    nothing
end

#======================================================#
device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")

for R in (4, 8, 16, 32)
    modeldir = joinpath(@__DIR__, "PCA$(R)")
    pca_project(datafile, modeldir, R)
end
#======================================================#
