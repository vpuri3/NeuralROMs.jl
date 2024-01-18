#
using LinearAlgebra, Lux, BSON, Plots
using MLUtils, GeometryLearning

#======================================================#
function post_process_CAE(datafile, modelfile, outdir)

    # load data
    data = BSON.load(datafile)
    x = data[:x]
    t = data[:t]
    Udata = data[:u]
    mu = data[:mu]

    Nx, Nb, Nt = size(Udata)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192
    Udata = @view Udata[Ix, :, :]
    x = @view x[Ix]

    # load model
    model = BSON.load(modelfile)
    NN, p, st = model[:model]
    md = model[:metadata] # (; mean, var, _Ib, Ib_, _It, It_, readme)

    Unorm = (Udata .- md.mean) / sqrt(md.var)
    Unorm = reshape(Unorm, Nx, 1, :)

    Upred = NN(Unorm, p, st)[1]
    Upred = Upred * sqrt(md.var) .+ md.mean

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
        _mu = round(mu[k], digits = 2)
        anim = animate1D(udata, upred, x, t; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)

        # add energy plot here
    end

    for k in 1:length(Ib_)
        udata = @view Udata_[:, k, :]
        upred = @view Upred_[:, k, :]
        _mu = round(mu[k], digits = 2)
        anim = animate1D(udata, upred, x, t; linewidth=2, xlabel="x",
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
modelfile = joinpath(@__DIR__, "model_cae", "model.bson")
outdir = joinpath(@__DIR__, "result_cae")

post_process_CAE(datafile, modelfile, outdir)

nothing
