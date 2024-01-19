#
"""
Train an autoencoder on 1D Burgers data
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

    k = 7 # 1, 7
    It = LinRange(1,length(Tdata), 10) .|> Base.Fix1(round, Int)

    Ud = Udata[:, k, It]
    U0 = Ud[:, 1]
    data = (reshape(Xdata, 1, :), reshape(U0, 1, :), Tdata[It])

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    ## time evolution prams
    timealg = EulerForward() # EulerForward(), RK2(), RK4()
    Δt = 1f-3
    adaptive = false

    # timealg = EulerForward() # EulerForward(), RK2(), RK4()
    # Δt = 1f-4
    # adaptive = false

    @time _, _, Up = evolve_autodecoder(
        prob, decoder, md, data, p0, timealg, Δt, adaptive;
        rng, device, verbose)

    Ix = 1:32:Nx
    plt = plot(xlabel = "x", ylabel = "u(x, t)", legend = false)
    plot!(plt, Xdata, Up, w = 2, palette = :tab10)
    scatter!(plt, Xdata[Ix], Ud[Ix, :], w = 1, palette = :tab10)

    _inf  = norm(Up - Ud, Inf)
    _mse  = sum(abs2, Up - Ud) / length(Ud)
    _rmse = sum(abs2, Up - Ud) / sum(abs2, Ud) |> sqrt
    println("||∞ : $(_inf)")
    println("MSE : $(_mse)")
    println("RMSE: $(_rmse)")

    png(plt, joinpath(outdir, "evolve_$k"))
    display(plt)

    nothing
end

#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 460)

device = Lux.gpu_device()
datafile = joinpath(@__DIR__, "burg_visc_re10k", "data.jld2")

Ix = 1:8:8192
_Ib, Ib_ = [1, 4, 7], [2, 3, 5, 6]
_batchsize, batchsize_  = 1024 .* (10, 3000)

makedata_kws = (; Ix, _Ib, Ib_, _It = :, It_ = :)

prob = BurgersInviscid1D()
prob = BurgersViscous1D(1f-4)

E = 1400
λ = 1f-0
for (l, h, w) in (
    (8, 10, 64),
    # (8, 5, 128),
)
    ll = lpad(l, 2, "0")
    hh = lpad(h, 2, "0")
    ww = lpad(w, 3, "0")
    modeldir = joinpath(@__DIR__, "model_dec_sin_$(ll)_$(hh)_$(ww)_reg")
    modelfile = joinpath(modeldir, "model_08.jld2")

    ## train
    # isdir(modeldir) && rm(modeldir, recursive = true)
    # model, STATS = train_autodecoder(datafile, modeldir, l, h, w, E; λ = 1f-1,
    #     _batchsize, batchsize_, makedata_kws, device)

    ## process
    outdir = joinpath(modeldir, "results")
    # postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    #     makeplot = true, verbose = true)
    test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
        makeplot = true, verbose = true)
end
#======================================================#
nothing
#
