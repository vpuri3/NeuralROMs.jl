#
using GeometryLearning

joinpath(pkgdir(GeometryLearning), "examples", "smoothNF.jl") |> include
joinpath(pkgdir(GeometryLearning), "examples", "problems.jl") |> include
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

    @assert ndims(Udata) ∈ (3,4,)
    @assert Xdata isa AbstractVecOrMat
    Xdata = Xdata isa AbstractVector ? reshape(Xdata, 1, :) : Xdata # (Dim, Npoints)

    if ndims(Udata) == 3 # [Nx, Nb, Nt]
        Udata = reshape(Udata, 1, size(Udata)...) # [out_dim, Nx, Nb, Nt]
    end

    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

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
    # Udata = @view Udata[:, md.makedata_kws.Ix, :, :]
    # Xdata = @view Xdata[:, md.makedata_kws.Ix]
    Nx = size(Xdata, 2)

    #==============#
    mkpath(outdir)
    #==============#

    k = 1
    It = LinRange(1,length(Tdata), 4) .|> Base.Fix1(round, Int)

    Ud = Udata[:, :, k, It]
    U0 = Ud[:, :, 1]

    data = (reshape(Xdata, in_dim, :), reshape(U0, out_dim, :), Tdata[It])
    data = copy.(data) # ensure no SubArrays

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    # time evolution prams
    timealg = EulerForward() # EulerForward(), RK2(), RK4()
    Δt = 5f-3
    adaptive = false

    @time _, _, Up = evolve_autodecoder(
        prob, decoder, md, data, p0, timealg, Δt, adaptive;
        rng, device, verbose)

    # analysis
    denom  = sum(abs2, Ud) / length(Ud) |> sqrt
    _max  = norm(Up - Ud, Inf) / sqrt(denom)
    _mean = sqrt(sum(abs2, Up - Ud) / length(Ud)) / denom
    println("Max error  (normalized): $(_max * 100 ) %")
    println("Mean error (normalized): $(_mean * 100) %")

    # visualiaztion
    kw = (; xlabel = "x", ylabel = "y", zlabel = "u(x,t)")

    Nx = Ny = Int(sqrt(Nx))

    upred_re = reshape(Up, out_dim, Nx, Ny, length(It))
    udata_re = reshape(Ud, out_dim, Nx, Ny, length(It))

    for od in 1:out_dim
        for i in eachindex(It)
            up_re = upred_re[od, :, :, i]
            ud_re = udata_re[od, :, :, i]

            p1 = heatmap(up_re; kw...,)
            p2 = heatmap(ud_re - up_re; kw...,)

            png(p1, joinpath(outdir, "evolve_u$(od)_$(k)_time_$(i)_pred"))
            png(p2, joinpath(outdir, "evolve_u$(od)_$(k)_time_$(i)_errr"))
        end
    end

    # anim = animate2D(udata_re, upred_re, x_re, y_re, Tdata[It])
    # gif(anim, joinpath(outdir, "evolve_$(k).gif"), fps = 10)

    Xdata, Up, Ud
end
#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 220)

prob = BurgersViscous2D(1f-3)
datafile = joinpath(@__DIR__, "data_burgers2D/", "data.jld2")
device = Lux.gpu_device()

modeldir = joinpath(@__DIR__, "model2")
modelfile = joinpath(modeldir, "model_08.jld2")

cb_epoch = nothing

Nx = 512
Ix  = LinRange(1, 512, 128) .|> Base.Fix1(round, Int)
Ix  = LinearIndices((Nx, Nx))[Ix, Ix] |> vec
_It = LinRange(1, 500, 201) .|> Base.Fix1(round, Int) # 101

Ix = Colon()

E = 1400
l, h, w = 8, 5, 128
λ1, λ2   = 0f0, 0f0
σ2inv, α = 1f-2, 0f-6
weight_decays = 2f-2

# isdir(modeldir) && rm(modeldir, recursive = true)
# makedata_kws = (; Ix, _Ib = :, Ib_ = :, _It = _It, It_ = :)
# model, STATS, metadata = train_autodecoder(datafile, modeldir, l, h, w, E;
#     λ1, λ2, σ2inv, α, weight_decays, cb_epoch, device, makedata_kws,
#     _batchsize = 16384,
#     batchsize_ = (Nx * Nx ÷ 10),
# )

## process
outdir = joinpath(modeldir, "results")
# postprocess_autodecoder(prob, datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)
x, up, ud = test_autodecoder(prob, datafile, modelfile, outdir; rng, device,
    makeplot = true, verbose = true)
#======================================================#
nothing
#
