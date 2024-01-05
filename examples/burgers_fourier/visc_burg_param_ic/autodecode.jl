#
"""
Train an autoencoder on 1D Burgers data
"""

using GeometryLearning

using LinearAlgebra, ComponentArrays

using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2                                 # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using Setfield                                    # misc

CUDA.allowscalar(false)

begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
end

#======================================================#
function test_autodecoder(
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

    # subsample in space
    Ix = 1:8:Nx
    Udata = @view Udata[Ix, :, :]
    Xdata = @view Xdata[Ix]
    Nx = length(Xdata)

    #==============#
    # load model
    #==============#
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"] # (; ū, σu, _Ib, Ib_, _It, It_, readme)
    close(model)

    # TODO - rm after retraining this model
    @set! md.σx = sqrt(md.σx)
    @set! md.σu = sqrt(md.σu)

    #==============#
    # make outdir path
    #==============#
    mkpath(outdir)

    k = 1# 1, 7
    It = LinRange(1,length(Tdata), 10) .|> Base.Fix1(round, Int)

    Ud = Udata[:, k, It]
    U0 = Ud[:, 1]
    data = (reshape(Xdata, 1, :), reshape(U0, 1, :), Tdata[It])

    decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    p0 = _code[2].weight[:, 1]

    prob = BurgersInviscid1D()
    prob = BurgersViscous1D(1/(4f3))

    CUDA.@time _, _, Up = evolve_autodecoder(prob, decoder, md, data, p0;
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

prob = BurgersInviscid1D()
prob = BurgersViscous1D(1/(4f3))

# for l in (3, 8,)
#     for h in (5, 8)
#         for w in (128)
#             ll = lpad(l, 2, "0")
#             hh = lpad(h, 2, "0")
#             ww = lpad(w, 3, "0")
#
#             (l == 3) & (h == 5) & (w == 96)
#
#             modeldir = joinpath(@__DIR__, "model_dec_sin_$(ll)_$(hh)_$(ww)_reg")
#
#             # isdir(modeldir) && rm(modeldir, recursive = true)
#             # model, STATS = train_autodecoder(l, h, w, datafile, modeldir; device)
#
#             modelfile = joinpath(modeldir, "model_08.jld2")
#             outdir = joinpath(dirname(modelfile), "results")
#             postprocess_autodecoder(datafile, modelfile, outdir; rng, device,
#                 makeplot = true, verbose = true)
#         end
#     end
# end

for modeldir in (
    # joinpath(@__DIR__, "model_dec_sin_03_05_128_reg/"),
    # joinpath(@__DIR__, "model_dec_sin_08_05_096_reg/"),
    joinpath(@__DIR__, "model_dec_sin_08_05_128_reg/"),
)
    modelfile = joinpath(modeldir, "model_08.jld2")
    outdir = joinpath(dirname(modelfile), "results")
    postprocess_autodecoder(datafile, modelfile, outdir; rng, device,
        makeplot = true, verbose = true)
end

# modeldir = joinpath(@__DIR__, "model_dec_sin_08_05_128_reg/")
# modelfile = joinpath(modeldir, "model_08.jld2")
#
# outdir = joinpath(dirname(modelfile), "results")
# postprocess_autodecoder(datafile, modelfile, outdir; rng, device,
#     makeplot = true, verbose = true)

# nothing
#
