#
using LinearAlgebra, Lux, BSON, Plots
using MLUtils, GeometryLearning

#======================================================#
function post_process_INR(datafile, modelfile, outdir)

    # load data
    data = BSON.load(datafile)
    Tdata = data[:t]
    Xdata = data[:x]
    Udata = data[:u]
    mu = data[:mu]

    Nx, Nb, Nt = size(Udata)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192
    Udata = @view Udata[Ix, :, :]
    Xdata = @view Xdata[Ix]

    # load model
    model = BSON.load(modelfile)
    NN, p, st = model[:model]
    md = model[:metadata] # (; ū, σu, _Ib, Ib_, _It, It_, readme)

    Unorm = (Udata .- md.ū) / sqrt(md.σu)
    Unorm = reshape(Unorm, Nx, 1, :)

    Xnorm = (Xdata .- md.x̄) / sqrt(md.σx)

    X = zeros(Float32, Nx, 2, size(Unorm, 3))
    X[:, 1, :] = Unorm
    X[:, 2, :] .= Xnorm

    Upred = NN(X, p, st)[1]
    Upred = Upred * sqrt(md.σu) .+ md.ū

    Upred = reshape(Upred, Nx, Nb, Nt)

    _Ib = md._Ib
    Ib_ = md.Ib_

    _Udata = @view Udata[:, _Ib, :]
    Udata_ = @view Udata[:, Ib_, :]

    _Upred = @view Upred[:, _Ib, :]
    Upred_ = @view Upred[:, Ib_, :]

    mkpath(outdir)

    for k in 1:length(_Ib)
        udata = @view _Udata[:, k, :]
        upred = @view _Upred[:, k, :]
        _mu = round(mu[_Ib[k]], digits = 2)
        anim = animate1D(udata, upred, Xdata, Tdata; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)

        # add energy plot here
    end

    for k in 1:length(Ib_)
        udata = @view Udata_[:, k, :]
        upred = @view Upred_[:, k, :]
        _mu = round(mu[Ib_[k]], digits = 2)
        anim = animate1D(udata, upred, Xdata, Tdata; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "test$(k).gif"), fps=30)

        # add energy plot here
    end

    if haskey(md, :readme)
        RM = joinpath(outdir, "README.md")
        RM = open(RM, "w")
        write(RM, md.readme)
        close(RM)
    end

    nothing
end

datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.bson")
modelfile = joinpath(@__DIR__, "model_inr", "model.bson")
outdir = joinpath(@__DIR__, "result_inr")

post_process_INR(datafile, modelfile, outdir)

nothing
