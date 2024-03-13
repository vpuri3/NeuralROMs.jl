#
using GeometryLearning
include(joinpath(pkgdir(GeometryLearning), "examples", "smoothNF.jl"))
include(joinpath(pkgdir(GeometryLearning), "examples", "problems.jl"))

#======================================================#
function test_autodecoder(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String,
    outdir::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
    makeplot::Bool = true,
    verbose::Bool = true,
    fps::Int = 300,
)

    #==============#
    # load data
    #==============#
    data = jldopen(datafile)
    Tdata = data["t"]
    Xdata = data["x"]
    Udata = data["u"]
    mu = data["mu"]
    md_data = data["metadata"]

    close(data)

    # data sizes
    Nx, Nb, Nt = size(Udata)

    mu = isnothing(mu) ? fill(nothing, Nb) |> Tuple : mu
    mu = isa(mu, AbstractArray) ? vec(mu) : mu

    #==============#
    # load model
    #==============#
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"] # (; ū, σu, _Ib, Ib_, _It, It_, readme)
    close(model)

    #==============#
    # subsample in space
    #==============#
    Udata = @view Udata[md.makedata_kws.Ix, :, :]
    Xdata = @view Xdata[:, md.makedata_kws.Ix]

    in_dim, Nx = size(Xdata)

    #==============#
    mkpath(outdir)
    #==============#

    k = 1
    It = LinRange(1,length(Tdata), 10) .|> Base.Fix1(round, Int)

    Udata = Udata[:, k, It]
    data = (Xdata, reshape(Udata[:, 1], 1, :), Tdata[It])

    data = copy.(data) # ensure no SubArrays

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    # time evolution prams
    timealg = EulerForward() # EulerForward(), RK2(), RK4()
    Δt = 1f-2
    adaptive = false

    @time _, _, Upred = evolve_autodecoder(
        prob, decoder, md, data, p0, timealg, Δt, adaptive;
        rng, device, verbose)

    # visualiaztion
    kw = (; xlabel = "x", ylabel = "y", zlabel = "u(x,t)")

    x_re = reshape(Xdata[1, :], md_data.Nx, md_data.Ny)
    y_re = reshape(Xdata[2, :], md_data.Nx, md_data.Ny)

    upred_re = reshape(Upred, md_data.Nx, md_data.Ny, length(It))
    udata_re = reshape(Udata, md_data.Nx, md_data.Ny, length(It))

    for i in eachindex(It)
        up_re = upred_re[:, :, i]
        ud_re = udata_re[:, :, i]

        p1 = plot()
        p1 = meshplt(x_re, y_re, ud_re; plt = p1, c=:black, w = 1.0, kw...,)
        p1 = meshplt(x_re, y_re, up_re; plt = p1, c=:red  , w = 0.2, kw...,)

        png(p1, joinpath(outdir, "evolve_$(k)_time_$(i)"))

        er_re = up_re - ud_re
        p2 = meshplt(x_re, y_re, er_re; title = "Error", kw...,)

        png(p2, joinpath(outdir, "evolve_$(k)_time_$(i)_error"))
    end

    anim = animate2D(udata_re, upred_re, x_re, y_re, Tdata[It])
    gif(anim, joinpath(outdir, "evolve_$(k).gif"), fps = 10)

    _inf  = norm(Upred - Udata, Inf)
    _mse  = sum(abs2, Upred - Udata) / length(Udata)
    _rmse = sum(abs2, Upred - Udata) / sum(abs2, Udata) |> sqrt
    println("||∞ : $(_inf)")
    println("MSE : $(_mse)")
    println("RMSE: $(_rmse)")

    nothing
end
#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

prob = Advection2D(0.25f0, 0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

E = 1400  # epochs
l = 4     # latent
h = 5     # num hidden
w = 128   # width

λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-2, 0f-0
weight_decays = 1f-2
WeightDecayOpt = IdxWeightDecay
weight_decay_ifunc = decoder_W_indices

Ix  = Colon()
_It = Colon()
_Ib, Ib_ = [1,], [1,]
makedata_kws = (; Ix, _Ib, Ib_, _It = _It, It_ = :)

batchsize_ = (96 * 96) * 500 ÷ 4

## train
isdir(modeldir) && rm(modeldir, recursive = true)
train_SNF(datafile, modeldir, l, h, w, E;
    rng, warmup = true, makedata_kws,
    λ1, λ2, σ2inv, α, weight_decays, device,
    WeightDecayOpt, weight_decay_ifunc,
    batchsize_
)

## process
# outdir = joinpath(modeldir, "results")
# postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
# test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
#======================================================#
nothing
#
