#
"""
Train an autoencoder on 1D advection data
"""

using GeometryLearning

include(joinpath(pkgdir(GeometryLearning), "examples", "autodecoder.jl"))

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
    Xdata = @view Xdata[md.makedata_kws.Ix]
    Nx = length(Xdata)

    #==============#
    mkpath(outdir)
    #==============#

    k = 1
    It = LinRange(1,length(Tdata), 10) .|> Base.Fix1(round, Int)

    Ud = Udata[:, k, It]
    U0 = Ud[:, 1]
    data = (reshape(Xdata, 1, :), reshape(U0, 1, :), Tdata[It])

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    # time evolution prams
    timealg = EulerForward() # EulerForward(), RK2(), RK4()
    Δt = 1f-2
    adaptive = false

    @time _, _, Up = evolve_autodecoder(
        prob, decoder, md, data, p0, timealg, Δt, adaptive;
        rng, device, verbose)

    Up = dropdims(Up; dims = 1)

    Ix_plt = 1:4:Nx
    plt = plot(xlabel = "x", ylabel = "u(x, t)", legend = false)
    plot!(plt, Xdata, Up, w = 2, palette = :tab10)
    scatter!(plt, Xdata[Ix_plt], Ud[Ix_plt, :], w = 1, palette = :tab10)

    denom  = sum(abs2, Ud) / length(Ud) |> sqrt
    _max  = norm(Up - Ud, Inf) / sqrt(denom)
    _mean = sqrt(sum(abs2, Up - Ud) / length(Ud)) / denom
    println("Max error  (normalized): $(_max * 100 )%")
    println("Mean error (normalized): $(_mean * 100)%")

    png(plt, joinpath(outdir, "evolve_$k"))
    display(plt)

    Xdata, Up, Ud[Ix_plt, :]
end
#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 220)

prob = Advection1D(0.25f0)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")

modeldir = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")

cb_epoch = nothing

## train (original)
# E = 1400
# _It = 1:1:500
# _batchsize = 1280
# l, h, w = 4, 5, 32 # (2, 4), 5, 32
# λ1, λ2, σ2inv, α = 0f-0, 0f-0, 0f-0, 0f0
# weight_decays = 1f-3

## train (5x less snapshots)
E = 1400
_It = LinRange(1, 500, 201) .|> Base.Fix1(round, Int)
_batchsize = 128 * 1
l, h, w = 4, 5, 32 # (2, 4), 5, 32
λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-3, 0f-6 # 1f-0, 1f-0
weight_decays = 1f-3  # 1-f0,

## train (unit dim latent space) very hard to train
# E = 2100
# _It = LinRange(1, 500, 201) .|> Base.Fix1(round, Int)
# _batchsize = 128 * 1
# l, h, w = 1, 5, 32 # (2, 4), 5, 32
# λ1, λ2, σ2inv, α = 0f-0, 0f-0, 1f-3, 1f-6 # 0, 0, 1f-3, 1f-6
# weight_decays = 0f-3                      # 0, 1f-3

isdir(modeldir) && rm(modeldir, recursive = true)
makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = _It, It_ = :)
model, STATS = train_autodecoder(datafile, modeldir, l, h, w, E;
    λ1, λ2, σ2inv, α, weight_decays, cb_epoch, device, makedata_kws,
)

## process
outdir = joinpath(modeldir, "results")
# postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
x, up, ud = test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
#======================================================#
nothing
#
