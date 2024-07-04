#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using JLD2, Plots                                 # vis/ save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using LaTeXStrings

CUDA.allowscalar(false)

begin
    nc = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
    BLAS.set_num_threads(nc)
end

include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "cases.jl"))

#======================================================#
function uData(x; σ = 1.0f0)
    pi32 = Float32(pi)

    # @. tanh(2f0 * x)

    # @. sin(5f0 * x^1) * exp(-(x/σ)^2)
    # @. sin(3f0 * x^2) * exp(-(x/σ)^2)

    # @. exp(sin(x))
    # @. sin(abs(x))

    @. (x - pi32/2f0) * sin(x) * exp(-(x/σ)^2)
end

function datagen_reg(datafile; _N = 1024, N_ = 16384)
    pi32 = Float32(pi)
    L = 2pi32

    _x = LinRange(-L, L, _N) |> Array
    x_ = LinRange(-L, L, N_) |> Array

    _u = uData(_x)
    u_ = uData(x_)
    metadata = (;)

    _data = (_x, _u)
    data_ = (x_, u_)

    jldsave(datafile; _data, data_, metadata)

    filename = joinpath(dirname(datafile), "plt_data")

    plt = plot(_x, _u, w = 3)
    png(plt, filename)

    plt
end

#======================================================#
function train_reg(
    datafile::String,
    dir::String,
    E, l, h, w;
    λ1::Real = 0f0,
    λ2::Real = 0f0,
    α::Real = 0f0,
    weight_decays::Union{Real,NTuple{M,<:Real}} = 0f0,
    rng::Random.AbstractRNG = Random.default_rng(),
    _batchsize = nothing,
    device = Lux.cpu_device(),
) where{M}

    #--------------------------------------------#
    # get data and normalize
    #--------------------------------------------#
    data = jldopen(datafile)
    _data = data["_data"]
    data_ = data["data_"]
    md_data = data["metadata"]
    close(data)

    _x, _u = reshape.(_data, 1, :)
    x_, u_ = reshape.(data_, 1, :)

    # normalize
    _x, x̄, σx = normalize_x(_x)
    _u, ū, σu = normalize_u(_u)

    x_, x̄, σx = normalize_x(x_)
    u_, ū, σu = normalize_u(u_)

    # metadata
    metadata = (; md_data, x̄, ū, σx, σu)

    _data = (_x, _u)
    data_ = (x_, u_)

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#

    NN = begin
        init_wt_in = scaled_siren_init(1f1)
        init_wt_hd = scaled_siren_init(1f0)
        init_wt_fn = glorot_uniform

        init_bias = rand32 # zeros32
        use_bias_fn = false

        act = sin

        wi = 1
        wd = w
        wo = 1

        in_layer = Dense(wi, wd, act; init_weight = init_wt_in, init_bias)
        hd_layer = Dense(wd, wd, act; init_weight = init_wt_hd, init_bias)
        fn_layer = Dense(wd, wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

        Chain(in_layer, fill(hd_layer, h)..., fn_layer)
    end

    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#

    lossfun = NeuralROMs.regularize_decoder(mse; α, λ1, λ2)
    _batchsize = isnothing(_batchsize) ? numobs(data) : _batchsize

    idx = ps_W_indices(NN; rng)
    weightdecay = IdxWeightDecay(0f0, idx)
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, true, weightdecay)

    #--------------------------------------------#
    display(NN)

    train_args = (; l, h, w, E, _batchsize, λ1, λ2, weight_decays, α)
    metadata = (; metadata..., train_args)

    @show metadata

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, weight_decays,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    @show metadata

    model, ST
end

#======================================================#
function post_reg(
    datafile::String,
    modelfile::String,
    outdir::String,
)
    mkpath(outdir)

    data = jldopen(datafile)
    x, _ = data["data_"]
    close(data)

    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    close(model)

    @show Lux.parameterlength(NN)
    @show md

    xbatch = reshape(x, 1, :)
    model = NeuralModel(NN, st, md)

    autodiff = AutoForwardDiff()
    ϵ = nothing

    u, ud1x, ud2x, ud3x, ud4x = dudx4_1D(model, xbatch, p; autodiff, ϵ) .|> vec
    ũ, ũd1x, ũd2x, ũd3x, ũd4x = forwarddiff_deriv4(uData, x)

    # print errors
    begin
        ud0_den = mse(u   , 0*u) |> sqrt
        ud1_den = mse(ũd1x, 0*u) |> sqrt
        ud2_den = mse(ũd2x, 0*u) |> sqrt
        ud3_den = mse(ũd3x, 0*u) |> sqrt
        ud4_den = mse(ũd4x, 0*u) |> sqrt

        ud0x_relrmse_er = sqrt(mse(u   , ũ   )) / ud0_den
        ud1x_relrmse_er = sqrt(mse(ud1x, ũd1x)) / ud1_den
        ud2x_relrmse_er = sqrt(mse(ud2x, ũd2x)) / ud2_den
        ud3x_relrmse_er = sqrt(mse(ud3x, ũd3x)) / ud3_den
        ud4x_relrmse_er = sqrt(mse(ud4x, ũd4x)) / ud4_den

        ud0x_relinf_er = norm(u    - ũ   , Inf) / ud0_den
        ud1x_relinf_er = norm(ud1x - ũd1x, Inf) / ud1_den
        ud2x_relinf_er = norm(ud2x - ũd2x, Inf) / ud2_den
        ud3x_relinf_er = norm(ud3x - ũd3x, Inf) / ud3_den
        ud4x_relinf_er = norm(ud4x - ũd4x, Inf) / ud4_den

        @show round.((ud0x_relrmse_er, ud0x_relinf_er), sigdigits = 8)
        @show round.((ud1x_relrmse_er, ud1x_relinf_er), sigdigits = 8)
        @show round.((ud2x_relrmse_er, ud2x_relinf_er), sigdigits = 8)
        @show round.((ud3x_relrmse_er, ud3x_relinf_er), sigdigits = 8)
        @show round.((ud4x_relrmse_er, ud4x_relinf_er), sigdigits = 8)
    end

    p0 = plot(xabel = "x", title = "u(x,t)")
    p1 = plot(xabel = "x", title = "u'(x,t)")
    p2 = plot(xabel = "x", title = "u''(x,t)")
    p3 = plot(xabel = "x", title = "u'''(x,t)")
    p4 = plot(xabel = "x", title = "u''''(x,t)")

    plot!(p0, x, ũ, label = "Ground Truth"  , w = 4, c = :black)
    plot!(p0, x, u, label = "Prediction"  , w = 2, c = :red)

    plot!(p1, x, ũd1x, label = "Ground Truth", w = 4, c = :black)
    plot!(p1, x, ud1x, label = "Prediction", w = 2, c = :red)

    plot!(p2, x, ũd2x, label = "Ground Truth", w = 4, c = :black)
    plot!(p2, x, ud2x, label = "Prediction", w = 2, c = :red)

    plot!(p3, x, ũd3x, label = "Ground Truth", w = 4, c = :black)
    plot!(p3, x, ud3x, label = "Prediction", w = 2, c = :red)

    plot!(p4, x, ũd4x, label = "Ground Truth", w = 4, c = :black)
    plot!(p4, x, ud4x, label = "Prediction", w = 2, c = :red)

    png(p0, joinpath(outdir, "derv0"))
    png(p1, joinpath(outdir, "derv1"))
    png(p2, joinpath(outdir, "derv2"))
    png(p3, joinpath(outdir, "derv3"))
    png(p4, joinpath(outdir, "derv4"))

    p0, p1, p2, p3, p4
end

#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 474)

datafile = joinpath(@__DIR__, "data_reg.jld2")
device = Lux.gpu_device()

E = 1400
_N, N_ = 1024, 8192 # 512, 32768
_batchsize = 32

## weight norm experiment
l, h, w = 1, 5, 64

datagen_reg(datafile; _N, N_) |> display

#############
modeldir1 = joinpath(@__DIR__, "model1") # vanilla
modeldir2 = joinpath(@__DIR__, "model2") # L2
modeldir3 = joinpath(@__DIR__, "model3") # lipschitz
modeldir4 = joinpath(@__DIR__, "model4") # weight

# α, weight_decays, λ2 = 0f-5, 0f-2, 0f-2 # vanilla
# isdir(modeldir1) && rm(modeldir1, recursive = true)
# _, ST1 = train_reg(datafile, modeldir1, E, l, h, w; λ2, α, weight_decays, _batchsize, device,)
#
# α, weight_decays, λ2 = 0f-5, 0f-2, 1f-1 # L2
# isdir(modeldir2) && rm(modeldir2, recursive = true)
# train_reg(datafile, modeldir2, E, l, h, w; λ2, α, weight_decays, _batchsize, device,)
#
# α, weight_decays, λ2 = 5f-5, 0f-2, 0f-2 # Lipschitz
# isdir(modeldir3) && rm(modeldir3, recursive = true)
# train_reg(datafile, modeldir3, E, l, h, w; λ2, α, weight_decays, _batchsize, device,)
#
# α, weight_decays, λ2 = 0f-5, 5f-2, 0f-0 # Weight # 2f-2
# isdir(modeldir4) && rm(modeldir4, recursive = true)
# train_reg(datafile, modeldir4, E, l, h, w; λ2, α, weight_decays, _batchsize, device,)

# #############

# α, weight_decays, λ2 = 1f-5, 0f-2, 0f-2
#
# modeldir  = joinpath(@__DIR__, "dump")
# modelfile = joinpath(modeldir, "model_08.jld2")
# outdir    = joinpath(modeldir, "results")
#
# isdir(modeldir) && rm(modeldir, recursive = true)
# model, STATS = train_reg(datafile, modeldir,
#     E, l, h, w; λ2, α, weight_decays,
#     _batchsize, device,
# )
#
# p0, p1, p2, p3, p4 = post_reg(datafile, modelfile, outdir)
#
# ptrain = plot_training(STATS...)
# plt = plot(ptrain, p0, p1, p2, p3, p4; size = (1300, 800))
# png(plt, joinpath(modeldir, "result"))
# display(plt)
#
# #############

#======================================================#
nothing
#
