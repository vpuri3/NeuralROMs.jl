#
using LinearAlgebra, Lux, ComponentArrays, BSON, JLD2, Plots
using MLUtils, GeometryLearning

#======================================================#
function post_process_Autodecoder(datafile, modelfile, outdir)

    # load data
    data = BSON.load(datafile)
    Tdata = data[:t]
    Xdata = data[:x]
    Udata = data[:u]
    mu = data[:mu]

    # get sizes
    Nx, Nb, Nt = size(Udata)

    # subsample in space
    Nx = Int(Nx / 8)
    Ix = 1:8:8192
    Udata = @view Udata[Ix, :, :]
    Xdata = @view Xdata[Ix]

    # load model
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"] # (; ū, σu, _Ib, Ib_, _It, It_, readme)
    close(model)

    _Udata = @view Udata[:, md._Ib, :]
    Udata_ = @view Udata[:, md.Ib_, :]

    # normalize
    Xnorm = (Xdata .- md.x̄) / sqrt(md.σx)

    _Ns = Nt * length(md._Ib) # num_codes
    Ns_ = Nt * length(md.Ib_)

    _xyz = zeros(Float32, Nx, _Ns)
    xyz_ = zeros(Float32, Nx, Ns_)

    _xyz[:, :] .= Xnorm
    xyz_[:, :] .= Xnorm

    _idx = zeros(Int32, Nx, _Ns)
    idx_ = zeros(Int32, Nx, Ns_)

    _idx[:, :] .= 1:_Ns |> adjoint
    idx_[:, :] .= 1:Ns_ |> adjoint

    _x = (reshape(_xyz, 1, :), reshape(_idx, 1, :))
    x_ = (reshape(xyz_, 1, :), reshape(idx_, 1, :))

    _Upred = NN(_x, p, st)[1]
    _Upred = _Upred * sqrt(md.σu) .+ md.ū

    _Upred = reshape(_Upred, Nx, length(md._Ib), Nt)

    mkpath(outdir)

    for k in 1:length(md._Ib)
        udata = @view _Udata[:, k, :]
        upred = @view _Upred[:, k, :]
        _mu = round(mu[md._Ib[k]], digits = 2)
        anim = animate1D(udata, upred, Xdata, Tdata; linewidth=2, xlabel="x",
            ylabel="u(x,t)", title = "μ = $_mu, ")
        gif(anim, joinpath(outdir, "train$(k).gif"), fps=30)
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
modelfile = joinpath(@__DIR__, "model_dec", "model.jld2")
outdir = joinpath(@__DIR__, "result_dec")

post_process_Autodecoder(datafile, modelfile, outdir)

# datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.bson")
# modelfile = joinpath(@__DIR__, "model_dec", "model.bson")
# outdir = joinpath(@__DIR__, "result_dec")
#
# post_process_Autodecoder(datafile, modelfile, outdir)

nothing
