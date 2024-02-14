#
using LinearAlgebra, Lux, BSON, Plots
using MLUtils

#======================================================#
function post_process_CAE(datafile, modelfile, outdir)

    # load data
    data = BSON.load(datafile)
    x = data[:x]
    t = LinRange(0, 10, 100) |> Array
    Udata = data[:u]

    Nx, Nb, Nt = size(Udata)

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

    for k in 1:5
        udata = @view _Udata[:, k, :]
        upred = @view _Upred[:, k, :]
        anim = _animate(udata, upred, x, t; linewidth=2, xlabel="x", ylabel="u(x,t)")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=15)

        # add energy plot here
    end

    for k in 1:5
        udata = @view Udata_[:, k, :]
        upred = @view Upred_[:, k, :]
        anim = _animate(udata, upred, x, t; linewidth=2, xlabel="x", ylabel="u(x,t)")
        gif(anim, joinpath(outdir, "test$(k).gif"), fps=15)

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

#======================================================#
function _animate(u::AbstractMatrix, v::AbstractMatrix, x::AbstractVector,
    t::AbstractVector; kwargs...)

    ylims = begin
        mi = minimum(u)
        ma = maximum(u)
        buf = (ma - mi) / 5
    (mi - buf, ma + buf)
    end
    kw = (; ylims, kwargs...)
    anim = @animate for i in 1:size(u, 2)
        titlestr = "time = $(round(t[i], digits=8))"
        plt = plot(x, u[:, i]; kw..., label = "Ground Truth", title = titlestr, c = :black)
        plot!(plt, x, v[:, i]; kw..., label = "Prediction"  , title = titlestr, c = :red)
    end
end
#======================================================#

datafile = joinpath(@__DIR__, "burg_visc_re10k_stationary", "data.bson")
modelfile = joinpath(@__DIR__, "CAE_stationary", "model.bson")
outdir = joinpath(@__DIR__, "Stationary")

post_process_CAE(datafile, modelfile, outdir)

datafile = joinpath(@__DIR__, "burg_visc_re10k_traveling", "data.bson")
modelfile = joinpath(@__DIR__, "CAE_traveling", "model.bson")
outdir = joinpath(@__DIR__, "Traveling")

post_process_CAE(datafile, modelfile, outdir)

nothing
