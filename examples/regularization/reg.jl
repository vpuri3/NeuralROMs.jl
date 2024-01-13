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
function uData(x; σ = 0.3f0)
    @. sin(1f1 * Float32(pi) * x) * exp(-(x/σ)^2)
end

function datagen_reg(_N, datafile; N_ = 32768)

    _x = LinRange(-1f0, 1f0, _N) |> Array
    x_ = LinRange(-1f0, 1f0, N_) |> Array

    _u = uData(_x)
    u_ = uData(x_)
    metadata = (; readme = "")

    _data = (_x, _u)
    data_ = (x_, u_)

    jldsave(datafile; _data, data_, metadata)

    filename = joinpath(dirname(datafile), "plt_data")

    plt = plot(_x, _u, w = 3)
    png(plt, filename)

    nothing
end

#======================================================#
function train_reg(
    datafile::String,
    dir::String,
    E, l, h, w, λ1, λ2, weight_decays;
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
    _data = data["_data"]
    data_ = data["data_"]
    md_data = data["metadata"]
    close(data)

    _x, _u = reshape.(_data, 1, :)
    x_, u_ = reshape.(data_, 1, :)

    # normalize u
    ū  = sum(u_) / length(u_)
    σu = sum(abs2, u_ .- ū) / length(u_) |> sqrt

    _u = normalizedata(_u, ū, σu)
    u_ = normalizedata(u_, ū, σu)

    # noramlize x
    x̄  = sum(x_, dims = 2) / size(x_, 2) |> vec
    σx = sum(abs2, x_ .- x̄, dims = 2) / size(x_, 2) .|> sqrt |> vec

    _x  = normalizedata(_x, x̄, σx)
    x_  = normalizedata(x_, x̄, σx)

    # metadata
    metadata = (; metadata..., md_data, x̄, ū, σx, σu)

    # learn embedding
    _i = ones(Int32, 1, size(_x, 2))
    i_ = ones(Int32, 1, size(x_, 2))

    _data, data_ = ((_x, _i), _u), ((x_, i_), u_)

    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#
    lrs = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    Nlrs = length(lrs)

    opts = Optimisers.AdamW.(lrs) # Grokking (https://arxiv.org/abs/2201.02177)
    nepochs = (round.(Int, E / (Nlrs) * ones(Nlrs))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, Nlrs)...,)

    # warm up
    opts = (Optimisers.AdamW(1f-1), opts...)
    nepochs = (10, nepochs...,)
    schedules = (Step(1f-2, 1f0, Inf32), schedules...,)
    early_stoppings = (false, early_stoppings...,)

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#
    act = sin

    init_wt_in = scaled_siren_init(3f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    # lossfun = iszero(λ) ? mse : l2reg(mse, λ) # ; property = :decoder)
    lossfun = elasticreg(mse, λ1, λ2) # ; property = :decoder)

    #--------------------------------------------#
    # AutoDecoder
    #--------------------------------------------#
    decoder = begin
        in_layer = Dense(l+1, w, act; init_weight = init_wt_in, init_bias)
        hd_layer = Dense(w  , w, act; init_weight = init_wt_hd, init_bias)
        fn_layer = Dense(w  , 1; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

        Chain(
            in_layer,
            fill(hd_layer, h)...,
            fn_layer,
        )
    end

    NN = AutoDecoder(decoder, 1, l)

    #--------------------------------------------#
    # Split Decoder
    #--------------------------------------------#

    # pou = begin
    #     in_layer = Dense(1, w, act; init_weight = init_wt_in, init_bias)
    #     hd_layer = Dense(w, w, act; init_weight = init_wt_hd, init_bias)
    #     fn_layer = Dense(w, 1; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)
    #
    #     # in_layer = Dense(l+1, w, sin)
    #     # hd_layer = Dense(w  , w, sin)
    #     # fn_layer = Dense(w  , 1)
    #
    #     Chain(
    #         in_layer,
    #         fill(hd_layer, h)...,
    #         fn_layer,
    #     )
    # end
    #
    # coef = begin
    #     act2 = tanh
    #     act2 = elu
    #
    #     vec_layer = WrappedFunction(vec)
    #     embedding = Embedding(1 => l)
    #
    #     in_layer = Dense(l, w, act2)
    #     hd_layer = Dense(w, w, act2)
    #     fn_layer = Dense(w, 1)
    #
    #     Chain(
    #         vec_layer,
    #         embedding,
    #         in_layer,
    #         fill(hd_layer, h)...,
    #         fn_layer,
    #         softmax,
    #     )
    # end
    #
    # connection = (x -> sum(x; dims = 1)) ∘ .*
    # NN = Parallel(connection, pou, coef) # (x -> NN1) .* (ũ -> NN2)
    
    #--------------------------------------------#
    display(NN)

    _batchsize = isnothing(_batchsize) ? numobs(data) : _batchsize

    train_args = (; l, h, w, E, _batchsize, λ1, λ2, weight_decays)
    metadata = (; metadata..., train_args)

    @show metadata

    @time model, ST = train_model(NN, _data, data_; rng,
        _batchsize, weight_decays,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
        cb_epoch,
    )

    plot_training(ST...) |> display

    model, ST
end

#======================================================#
function post_reg(
    datafile::String,
    modelfile::String,
    outdir::String,
    params::Vector = [],
    fps::Int = 30,
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
    model = NeuralEmbeddingModel(NN, st, xbatch, md)

    autodiff = AutoForwardDiff()
    ϵ = nothing

    u, ud1x, ud2x = dudx2(model, xbatch, p; autodiff, ϵ) .|> vec
    ũ, ũd1x, ũd2x = forwarddiff_deriv2(uData, x)

    # print errors
    begin
        ud0_den = mse(u   , 0*u) |> sqrt
        ud1_den = mse(ũd1x, 0*u) |> sqrt
        ud2_den = mse(ũd2x, 0*u) |> sqrt

        ud0x_relrmse_er = sqrt(mse(u   , ũ   )) / ud0_den
        ud1x_relrmse_er = sqrt(mse(ud1x, ũd1x)) / ud1_den
        ud2x_relrmse_er = sqrt(mse(ud2x, ũd2x)) / ud2_den

        ud0x_relinf_er = norm(u    - ũ   , Inf) / ud0_den
        ud1x_relinf_er = norm(ud1x - ũd1x, Inf) / ud1_den
        ud2x_relinf_er = norm(ud2x - ũd2x, Inf) / ud2_den

        @show round.((ud0x_relrmse_er, ud0x_relinf_er), digits = 8)
        @show round.((ud1x_relrmse_er, ud1x_relinf_er), digits = 8)
        @show round.((ud2x_relrmse_er, ud2x_relinf_er), digits = 8)
    end

    # if !isempty(params)
    #     _u = ()
    #     _ud1x = ()
    #     _ud2x = ()
    #
    #     for p in params
    #         __u, __ud1x, __ud2x = dudx2(model, xbatch, p; autodiff, ϵ) .|> vec
    #         _u = (_u..., __u)
    #         _ud1x = (_ud1x..., __ud1x)
    #         _ud2x = (_ud2x..., __ud2x)
    #     end
    #     _u = hcat(_u...)
    #     _ud1x = hcat(_ud1x...)
    #     _ud2x = hcat(_ud2x...)
    #
    #     o = ones(Float32, 1, length(params))
    #    
    #     anim = animate1D(ũ * o, _u, x; w = 2, xlabel = "x", title = "u(x, t)")
    #     gif(anim, joinpath(outdir, "derv0.gif"); fps)
    #    
    #     anim = animate1D( ũd1x * o, _ud1x, x; w = 2, xlabel = "x", title = "u(x, t)")
    #     gif(anim, joinpath(outdir, "derv1.gif"); fps)
    #    
    #     anim = animate1D(ũd2x * o, _ud2x, x; w = 2, xlabel = "x", title = "u(x, t)")
    #     gif(anim, joinpath(outdir, "derv2.gif"); fps)
    # end

    p0 = plot(xabel = "x", title = "u(x,t)")
    p1 = plot(xabel = "x", title = "u'(x,t)")
    p2 = plot(xabel = "x", title = "u''(x,t)")

    plot!(p0, x, ũ, label = "Ground Truth"  , w = 4, c = :black)
    plot!(p0, x, u, label = "Prediction"  , w = 2, c = :red)

    plot!(p1, x, ũd1x, label = "Ground Truth", w = 4, c = :black)
    plot!(p1, x, ud1x, label = "Prediction", w = 2, c = :red)

    plot!(p2, x, ũd2x, label = "Ground Truth", w = 4, c = :black)
    plot!(p2, x, ud2x, label = "Prediction", w = 2, c = :red)

    png(p0, joinpath(outdir, "derv0"))
    png(p1, joinpath(outdir, "derv1"))
    png(p2, joinpath(outdir, "derv2"))

    display(plot(p0, p1, p2))
end

#======================================================#
# main
#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 460)

datafile = joinpath(@__DIR__, "data_reg.jld2")
device = Lux.cpu_device()

E = 2000
_N, N_ = 512, 8192 # 512, 32768
_batchsize = 32
fps = Int(E / 5)

l, h, w = 1, 5, 32
λ1s = (0.0f0,)
λ2s = (0.0f0,) # 0.01f0
weight_decays = 0.005f0 # 5f-3

### TODO
# - POU networks. Inspiration from Finite Element methods. Some relation to DeepONets.
#   Take a second look at my Git repo, slides. And reread N. Trask's paper.
#   We can also involve a bilinear layer of some sort. Not sure.
#   Likely that such a network where two NNs are multiplied would need a smaller LR
#   and many more iters than a deep NN. Read the papers.
# - add features `x` -> `[x, x^2, x^3, tan(x)]`, or devise an architecutre that lets
#   you multiply trunks. Kind of like multihead transformer.
#   With sinosudal transformer, how would MHA work? We want terms like x^2, but
#   `sin(x) * sin(x) ~ sin(2x)`.
###

# ps = ()

datagen_reg(_N, datafile; N_)

for (i, (λ1, λ2)) in enumerate(zip(λ1s, λ2s))
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
        E, l, h, w, λ1, λ2, weight_decays; _batchsize,
        cb_epoch, device,
    )

    # global ps = (ps..., _ps)

    ## process
    post_reg(datafile, modelfile, outdir, _ps, fps)
end

# jldsave(joinpath(@__DIR__, "ps.jld2"); ps)
#======================================================#
nothing
#
