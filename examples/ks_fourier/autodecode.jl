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
    Δt = 1f-4
    adaptive = false

    @time _, _, Up = evolve_autodecoder(
        prob, decoder, md, data, p0, timealg, Δt, adaptive;
        rng, device, verbose)

    Ix_plt = 1:8:Nx
    plt = plot(xlabel = "x", ylabel = "u(x, t)", legend = false)
    plot!(plt, Xdata, Up, w = 2, palette = :tab10)
    scatter!(plt, Xdata[Ix_plt], Ud[Ix_plt, :], w = 1, palette = :tab10)

    denom  = sum(abs2, Ud) / length(Ud) |> sqrt
    _max  = norm(Up - Ud, Inf) / sqrt(denom)
    _mean = sqrt(sum(abs2, Up - Ud) / length(Ud)) / denom
    println("Max error  (normalized): $(_max * 100 ) %")
    println("Mean error (normalized): $(_mean * 100) %")

    png(plt, joinpath(outdir, "evolve_$k"))
    display(plt)

    Xdata, Up, Ud[Ix_plt, :]
end
#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 111)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "data_ks/", "data.jld2")

modeldir = joinpath(@__DIR__, "model3")
modelfile = joinpath(modeldir, "model_08.jld2")

prob = KuramotoSivashinsky1D(0.01f0)

cb_epoch = nothing

## train
E = 7_000
_It = LinRange(1, 1000, 100) .|> Base.Fix1(round, Int) # 200
_batchsize = 256 * 5
l, h, w = 16, 5, 64
λ1, λ2 = 0f0, 0f0
σ2inv, α = 1f-1, 0f-5 # 1f-1, 1f-3
weight_decays = 1f-2  # 1f-2

isdir(modeldir) && rm(modeldir, recursive = true)
makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = _It, It_ = :)
model, STATS = train_autodecoder(datafile, modeldir, l, h, w, E;
    λ1, λ2, σ2inv, α, weight_decays, cb_epoch, device, makedata_kws,
    _batchsize,
)

## process
outdir = joinpath(modeldir, "results")
postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
x, up, ud = test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
#======================================================#
nothing
#
