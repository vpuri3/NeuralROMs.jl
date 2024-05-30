#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using JLD2                                        # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using LaTeXStrings

using KolmogorovArnold

CUDA.allowscalar(false)

begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
end

# include(joinpath(pkgdir(NeuralROMs), "examples", "cases.jl"))

#======================================================#
function uData(x; σ = 1.0f0)
    pi32 = Float32(pi)

    # @. tanh(2f0 * x)
    # @. sin(1f0 * x)

    # @. sin(5f0 * x^1) * exp(-(x/σ)^2)
    # @. sin(3f0 * x^2) * exp(-(x/σ)^2)

    @. (x - pi32/2f0) * sin(x) * exp(-(x/σ)^2)
end

function datagen_reg(_N, datafile; N_ = 32768)
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

function post_kan(
    datafile::String,
    modelfile::String,
)
    data = jldopen(datafile)
    x, ũ = data["data_"]
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

    u, ud1x = dudx4_1D(model, xbatch, p; autodiff, ϵ) .|> vec
    ũ, ũd1x = forwarddiff_deriv4(uData, x)

    begin
        ud0_den = mse(u   , 0*u) |> sqrt
        ud1_den = mse(ũd1x, 0*u) |> sqrt

        ud0x_relrmse_er = sqrt(mse(u   , ũ   )) / ud0_den
        ud1x_relrmse_er = sqrt(mse(ud1x, ũd1x)) / ud1_den

        ud0x_relinf_er = norm(u    - ũ   , Inf) / ud0_den
        ud1x_relinf_er = norm(ud1x - ũd1x, Inf) / ud1_den

        @show round.((ud0x_relrmse_er, ud0x_relinf_er), sigdigits = 8)
        @show round.((ud1x_relrmse_er, ud1x_relinf_er), sigdigits = 8)
    end

    p0 = plot(xabel = "x", title = "u(x,t)" , legend = false)
    p1 = plot(xabel = "x", title = "u'(x,t)", legend = false)

    plot!(p0, x, ũ, label = "Ground Truth"  , w = 4, c = :black)
    plot!(p0, x, u, label = "Prediction"  , w = 2, c = :red)

    # plot!(p1, x, ũd1x, label = "Ground Truth", w = 4, c = :black)
    # plot!(p1, x, ud1x, label = "Prediction", w = 2, c = :red)

    plot(p0)
end

#======================================================#
function train_kan(
    datafile::String,
    dir::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
)
    #--------------------------------------------#
    # get data
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

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#

    wi, wo = 1, 1

    G  = 10
    h  = 1
    wh = 10

    use_base_act = false
    basis_func = rswaf # rbf, rswaf
    normalizer = identity # softsign # tanh, sigmoid, softsign

    init_C = glorot_normal
    # init_C = glorot_uniform

    # TODO
    # - take a look at layer outputs
    # - take a look at C. Plot the basis functions that come out

    in_layer = KDense(wi, wh, G; use_base_act, normalizer, basis_func, init_C)
    hd_layer = KDense(wh, wh, G; use_base_act, normalizer, basis_func, init_C)
    fn_layer = KDense(wh, wo, G; use_base_act, normalizer, basis_func, init_C)

    NN = Chain(in_layer, fill(hd_layer, h)..., fn_layer)

    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#
    _batchsize = 64
    E = 1000

    lossfun = mse
    _batchsize = 128

    # lrs  = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    lrs  = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    opts = Tuple(Optimisers.Adam(lr) for lr in lrs)
    Nlrs = length(lrs)

    nepochs = (round.(Int, E / (Nlrs) * ones(Nlrs))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, Nlrs)...,)

    # BFGS
    nepochs = (nepochs..., E,)
    opts = (opts..., LBFGS(),)
    schedules = (schedules..., Step(0f0, 1f0, Inf32),)
    early_stoppings = (early_stoppings..., true)

    #--------------------------------------------#
    # train
    #--------------------------------------------#
    display(NN)

    train_args = (; G, h, wh, E, _batchsize)
    metadata = (; metadata..., train_args)

    @show metadata

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    # @show metadata

    model, ST
end

#======================================================#
# main
#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 123)

datafile = joinpath(@__DIR__, "data_reg.jld2")
modeldir = joinpath(@__DIR__, "kan")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

E = 100
_N, N_ = 1024, 8192 # 512, 32768
_batchsize = 32

datagen_reg(_N, datafile; N_) |> display

isdir(modeldir) && rm(modeldir, recursive = true)

model, ST = train_kan(datafile, modeldir; rng, device)
plt = post_kan(datafile, modelfile)
display(plt)

#======================================================#
nothing
