#
"""
Train an autoencoder on 1D advection data
"""

using GeometryLearning
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2                                 # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU

CUDA.allowscalar(false)

# using FFTW
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
end

#======================================================#
# data gen
#======================================================#
function uData(x; σ = 0.3f0)
    @. sin(1f1 * Float32(pi) * x) * exp(-(x/σ)^2)
end

function datagen_reg(Nx, datafile)
    x = LinRange(-1f0, 1f0, Nx) |> Array
    u = uData(x)
    metadata = (; readme = "")

    jldsave(datafile; x, u, metadata)

    filename = joinpath(dirname(datafile), "plt_data")

    plt = plot(x, u, w = 3)
    png(plt, filename)
    display(plt)

    x, u
end

#======================================================#
# model setup
#======================================================#
function train_reg(
    datafile::String,
    dir::String,
    E, l, h, w, λ;
    rng::Random.AbstractRNG = Random.default_rng(),
    _batchsize = nothing,
    cb_epoch = nothing,
    device = Lux.cpu_device(),
)
    metadata = (;)

    #--------------------------------------------#
    # get data and normalize
    #--------------------------------------------#
    data = jldopen(datafile)
    x = data["x"]
    u = data["u"]
    md_data = data["metadata"]
    close(data)

    x, u = reshape.((x, u), 1, :)

    # normalize u
    ū  = sum(u) / length(u)
    σu = sum(abs2, u .- ū) / length(u) |> sqrt
    u  = normalizedata(u, ū, σu)

    # noramlize x
    x̄  = sum(x, dims = 2) / size(x, 2) |> vec
    σx = sum(abs2, x .- x̄, dims = 2) / size(x, 2) .|> sqrt |> vec
    x  = normalizedata(x, x̄, σx)

    data = x, u
    metadata = (; metadata..., md_data, x̄, ū, σx, σu)
    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#
    lrs = (1f-2, 1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    opts = Optimisers.Adam.(lrs)
    nepochs = (20, round.(Int, E / 7 * ones(7))...)
    schedules = Step.(lrs, 1f0, Inf32)

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#
    act = sin

    init_wt_in = scaled_siren_init(3f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    lossfun = l2reg(mse, λ) # ; property = :decoder)

    #----------------------#----------------------#
    in_layer = Dense(l+1, w, act; init_weight = init_wt_in, init_bias)
    hd_layer = Dense(w  , w, act; init_weight = init_wt_hd, init_bias)
    fn_layer = Dense(w  , 1; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

    NN = Chain(
        in_layer,
        fill(hd_layer, h)...,
        fn_layer,
    )
    #----------------------#----------------------#

    _batchsize = isnothing(_batchsize) ? numobs(data) : _batchsize

    train_args = (; l, h, w, E, _batchsize, λ)
    metadata = (; metadata..., train_args)

    @show metadata

    @time model, ST = train_model(NN, data; rng,
        _batchsize,
        opts, nepochs, schedules,
        device, dir, metadata, lossfun,
        cb_epoch,
    )

    plot_training(ST...) |> display

    model, ST
end

function post_reg(
    datafile::String,
    modelfile::String,
    outdir::String,
    params::Vector = [],
    fps::Int = 30,
)
    mkpath(outdir)

    data = jldopen(datafile)
    x = data["x"]
    close(data)

    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    close(model)

    @show md

    model = NeuralModel(NN, st, md)
    xbatch = reshape(x, 1, :)

    autodiff = AutoForwardDiff()
    ϵ = nothing

    u, ud1x, ud2x = dudx2(model, xbatch, p; autodiff, ϵ) .|> vec
    ũ, ũd1x, ũd2x = forwarddiff_deriv2(uData, x)

    if !isempty(params)
        _u = ()
        _ud1x = ()
        _ud2x = ()

        for p in params
            __u, __ud1x, __ud2x = dudx2(model, xbatch, p; autodiff, ϵ) .|> vec
            _u = (_u..., __u)
            _ud1x = (_ud1x..., __ud1x)
            _ud2x = (_ud2x..., __ud2x)
        end
        _u = hcat(_u...)
        _ud1x = hcat(_ud1x...)
        _ud2x = hcat(_ud2x...)

        o = ones(Float32, 1, length(params))

        anim = animate1D(ũ * o, _u, x; w = 2, xlabel = "x", title = "u(x, t)")
        gif(anim, joinpath(outdir, "derv0.gif"); fps)

        anim = animate1D( ũd1x * o, _ud1x, x; w = 2, xlabel = "x", title = "u(x, t)")
        gif(anim, joinpath(outdir, "derv1.gif"); fps)

        anim = animate1D(ũd2x * o, _ud2x, x; w = 2, xlabel = "x", title = "u(x, t)")
        gif(anim, joinpath(outdir, "derv2.gif"); fps)
    end

    p0 = plot(xabel = "x", title = "u(x,t)")
    p1 = plot(xabel = "x", title = "u'(x,t)")
    p2 = plot(xabel = "x", title = "u''(x,t)")

    plot!(p0, x, ũ, label = "Ground Truth"  , w = 2, c = :black)
    plot!(p0, x, u, label = "Prediction"  , w = 2, c = :red)

    plot!(p1, x, ũd1x, label = "Ground Truth", w = 2, c = :black)
    plot!(p1, x, ud1x, label = "Prediction", w = 2, c = :red)

    plot!(p2, x, ũd2x, label = "Ground Truth", w = 2, c = :black)
    plot!(p2, x, ud2x, label = "Prediction", w = 2, c = :red)

    png(p0, joinpath(outdir, "derv0"))
    png(p1, joinpath(outdir, "derv1"))
    png(p2, joinpath(outdir, "derv2"))

    display(p1)
end

#======================================================#
# main
#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 460)

datafile = joinpath(@__DIR__, "data_reg.jld2")
device = Lux.cpu_device()

# E = 50
# Nx = 32768
# _batchsize = 128
# fps = 20

E = 200
Nx = 8192
_batchsize = 4096
fps = Int(E/5)

l, h, w = 0, 5, 16

λs = LinRange(0, 1, 3) .|> Float32
ps = ()

## data gen
# data = datagen_reg(Nx, datafile)

## train
for (i, λ) in enumerate(λs)
    _ps = []
    cb_epoch = function(NN, p, st)
        push!(_ps, p)
        nothing
    end

    modeldir  = joinpath(@__DIR__, "model$i")
    modelfile = joinpath(modeldir, "model_08.jld2")
    outdir    = joinpath(modeldir, "results")

    isdir(modeldir) && rm(modeldir, recursive = true)
    model, STATS = train_reg(datafile, modeldir,
        E, l, h, w, λ; _batchsize,
        cb_epoch, device,
    )

    global ps = (ps..., _ps)

    ## process
    post_reg(datafile, modelfile, outdir, _ps, fps)

end

jldsave(joinpath(@__DIR__, "ps.jld2"); ps)
#======================================================#
nothing
#
