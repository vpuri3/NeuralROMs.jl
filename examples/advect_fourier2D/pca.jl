#
using LinearAlgebra, JLD2, Plots, CUDA

CUDA.allowscalar(false)

# using FFTW
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
end

#======================================================#
function makedata_PCA(datafile)

    data = jldopen(datafile)
    x = data["x"]
    u = data["u"] # [Nx, Nb, Nt]
    md_data = data["metadata"]
    close(data)

    # get sizes
    Nx, Nb, Nt = size(u)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192

    # normalize solution
    ū = sum(u) / length(u)
    ū = sum(u; dims = (2,3)) / length(u) |> flatten
    u = u .- ū

    # train/test trajectory split
    _Ib, Ib_ = [1, 4, 7], [2, 3, 5, 6]

    # train on all time
    _It = Colon()
    It_ = Colon()

    x = @view x[Ix]

    _u = @view u[Ix, _Ib, _It]
    u_ = @view u[Ix, Ib_, It_]

    metadata = (; μ, x, t, ū, _Ib, Ib_,)

    _u, u_, metadata
end

#======================================================#
function pca_project(R::Int, datafile, outdir)

    # load data
    _u, u_, md = makedata_PCA(datafile)

    Nx, _Nb, Nt = size(_u)
    Nx, Nb_, Nt = size(u_)

    # SVD
    _udata = reshape(_u, Nx, :)
    udata_ = reshape(u_, Nx, :)

    F = svd(_udata)
    U = F.U[:, 1:R]

    # project data
    _a = U' * _udata
    _upred = U  * _a

    a_ = U' * udata_
    upred_ = U  * a_

    _v = reshape(_upred, Nx, _Nb, Nt)
    v_ = reshape(upred_, Nx, Nb_, Nt)

    mkpath(outdir)
    
    for k in 1:length(md._Ib)
        u = @view _u[:, k, :]
        v = @view _v[:, k, :]
        _mu = round(md.μ[md._Ib[k]], digits = 2)

        anim = animate1D(u, v, md.x, md.t; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)
    end

    for k in 1:length(md.Ib_)
        u = @view u_[:, k, :]
        v = @view v_[:, k, :]
        _mu = round(md.μ[md.Ib_[k]], digits = 2)

        anim = animate1D(u, v, md.x, md.t; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "test$(k).gif"), fps=30)
    end

    nothing
end

#======================================================#
_makedir = Base.Fix1(joinpath, @__DIR__)
datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.bson")

for R in (4, 8, 16, 32)
    R = 4
    outdir = joinpath(@__DIR__, "results_RCA$(R)")
    pca_project(R, datafile, outdir)
end
#======================================================#
