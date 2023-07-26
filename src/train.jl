#
"""
$SIGNATURES

# Arguments
- `V`: function space
- `NN`: Lux neural network
- `_data`: training data as `(x, y)`. `x` may be an AbstractArray or a tuple of arrays
- `data_`: testing data (same requirement as `_data)

Data arrays, `x`, `y` must be `AbstractMatrix`, or `AbstractArray{T,3}`.
In the former case, the dimensions are assumed to be `(points, batch)`,
and `(chs, points, batch)` in the latter, where the points dimension
is equal to `length(V)`.

# Keyword Arguments
- `opts`: `NTuple` of optimizers
- `maxiters`: Number of iterations for each optimization cycle
- `cbstep`: prompt `callback` function every `cbstep` epochs
- `dir`: directory to save model, plots
- `io`: io for printing stats
"""
function train_model(
    rng::Random.AbstractRNG,
    NN::Lux.AbstractExplicitLayer,
    _data::NTuple{2, Any},
    data_::NTuple{2, Any},
    V::Spaces.AbstractSpace,
    opt::Optimisers.AbstractRule = Optimisers.Adam();
    learning_rates::NTuple{N, Float32} = (1f-3,),
    maxiters::NTuple{N} = (100,),
    cbstep::Int = 10,
    dir = "",
    name = "model",
    nsamples = 5,
    io::IO = stdout,
    p = nothing,  # initial parameters
    st = nothing, # initial state
    lossfun = GeometryLearning.mse,
    device = Lux.cpu,
    make_plots = true,
) where{N}

    # make directory for saving model
    mkpath(dir)

    for data in (_data, data_)
        @assert data[1] isa AbstractArray
        @assert data[2] isa AbstractArray
    end

    _devicedata, devicedata_ = (_data, data_) |> device

    @assert length(learning_rates) == length(maxiters)

    # utility functions
    _model, _loss, _stats = model_setup(NN, _devicedata; lossfun)
    model_, loss_, stats_ = model_setup(NN, devicedata_; lossfun)

    # analysis callback
    CB = (p, st; io = io) -> callback(p, st; io, _loss, _stats, loss_, stats_)

    # if initial parameters not provided,
    # get model parameters with rng

    _p, _st = Lux.setup(rng, NN)
    # _p = _p |> ComponentArray # not nice for real + complex

    p  = isnothing(p ) ? _p  : p
    st = isnothing(st) ? _st : st

    p, st = (p, st) |> device

    # print stats
    CB(p, st)

    # training callback
    ITER  = Int[]
    _LOSS = Float32[]
    LOSS_ = Float32[]

    cb = (p, st, iter, maxiter; io = io) -> callback(p, st; io,
                                                _loss, _LOSS, loss_, LOSS_,
                                                ITER, iter, maxiter, step = cbstep)

        println(io, "#======================#")
        println(io, "Starting Trainig Loop")
        println(io, "Optimizer: $opt")
        println(io, "#======================#")

    # set up optimizer
    opt_st = nothing

    for i in eachindex(maxiters)
        learning_rate = learning_rates[i]
        maxiter = maxiters[i]

        @set! opt.eta = learning_rate

        println(io, "#======================#")
        println(io, "Learning Rate: $learning_rate, ITERS: $maxiter")
        println(io, "#======================#")

        @time p, st, opt_st = optimize(_loss, p, st, maxiter; opt, opt_st, cb)

        CB(p, st)
    end

    # save statistics
    statsfile = open(joinpath(dir, "statistics.txt"), "w")
    CB(p, st; io = statsfile)
    close(statsfile)

    # transfer to host device and free stuff
    if device == Lux.gpu
        CUDA.unsafe_free!.(_devicedata)
        CUDA.unsafe_free!.(devicedata_)
    end

    p, st = (p, st) |> Lux.cpu

    # visualization
    if make_plots
        plot_training(ITER, _LOSS, LOSS_; dir)
        visualize(V, _data, data_, NN, p, st; nsamples, dir)
    end

    model = NN, p, st
    STATS = ITER, _LOSS, LOSS_
 
    BSON.@save joinpath(dir, "$name.bson") _data data_ model

    model, STATS
end

"""
$SIGNATURES

"""
function model_setup(NN::Lux.AbstractExplicitLayer, data; lossfun = mse)

    x, ŷ = data

    function model(p, st)
        NN(x, p, st)[1]
    end

    function loss(p, st)
        y = NN(x, p, st)[1]

        lossfun(y, ŷ), st
    end

    function stats(p, st; io::Union{Nothing, IO} = stdout)
        y = model(p, st)
        Δy = y - ŷ

        R2 = rsquare(y, ŷ)
        MSE = mse(y, ŷ)

        meanAE = norm(Δy, 1) / length(ŷ)
        maxAE  = norm(Δy, Inf)

        rel   = Δy ./ ŷ
        meanRE = norm(rel, 1) / length(ŷ)
        maxRE  = norm(rel, Inf)

        if !isnothing(io)
            str = ""
            str *= string("R² score:       ", round(R2    , digits=8), "\n")
            str *= string("mean SQR error: ", round(MSE   , digits=8), "\n")
            str *= string("mean ABS error: ", round(meanAE, digits=8), "\n")
            str *= string("max  ABS error: ", round(maxAE , digits=8), "\n")
            str *= string("mean REL error: ", round(meanRE, digits=8), "\n")
            str *= string("max  REL error: ", round(maxRE , digits=8))

            println(io, str)
        end

        R2, MSE, meanAE, maxAE, meanRE, maxRE
    end

    model, loss, stats
end

"""
$SIGNATURES

"""
function callback(p, st; io::Union{Nothing, IO} = stdout,
                  _loss = nothing, _LOSS = nothing, _stats = nothing,
                  loss_ = nothing, LOSS_ = nothing, stats_ = nothing,
                  iter = nothing, step = 0, maxiter = 0, ITER = nothing,
                 )

    str = if !isnothing(iter)
        step = iszero(step) ? 10 : step

        if iter % step == 0 || iter == 1 || iter == maxiter
            "Iter $iter: "
        else
            return
        end
    else
        ""
    end

    # log iter
    if !isnothing(ITER) & !isnothing(iter)
        push!(ITER, iter)
    end

    # log training loss
    _l = if !isnothing(_loss)
        _l = _loss(p, st)[1]
        !isnothing(_LOSS) && push!(_LOSS, _l)
        _l
    else
        nothing
    end

    # log test loss
    l_ = if !isnothing(loss_)
        l_ = loss_(p, st)[1]
        !isnothing(LOSS_) && push!(LOSS_, l_)
        l_
    else
        nothing
    end

    isnothing(io) && return

    if !isnothing(_l)
        str *= string("TRAIN LOSS: ", round(_l, digits=8), " ")
    end

    if !isnothing(l_)
        str *= string("TEST LOSS: ", round(l_, digits=8), " ")
    end

    println(io, str)

    if !isnothing(io)
        if !isnothing(_stats)
            println(io, "#======================#")
            println(io, "TRAIN STATS")
            _stats(p, st; io)
            println(io, "#======================#")
        end
        if !isnothing(stats_) 
            println(io, "#======================#")
            println(io, "TEST  STATS")
            stats_(p, st; io)
            println(io, "#======================#")
        end
    end

    return
end

"""
$SIGNATURES

Train parameters `p` to minimize `loss` using optimization strategy `opt`.

# Arguments
- Loss signature: `loss(p, st) -> y, st`
- Callback signature: `cb(p, st iter, maxiter) -> nothing` 
"""
function optimize(loss, p, st, maxiter;
    opt = Optimisers.Adam(),
    opt_st = nothing,
    cb = nothing
)

    function grad(p, st)
        loss2 = Base.Fix2(loss, st)
        (l, st), pb = Zygote.pullback(loss2, p)
        g = pb((one.(l), nothing))[1]

        l, g, st
    end

    # print stats
    !isnothing(cb) && cb(p, st, 0, maxiter)

    # dry run
    grad(p, st)

    # init optimizer
    opt_st = isnothing(opt_st) ? Optimisers.setup(opt, p) : opt_st

    for iter in 1:maxiter
        _, g, st = grad(p, st)
        opt_st, p = Optimisers.update(opt_st, p, g)

        !isnothing(cb) && cb(p, st, iter, maxiter)
    end

    p, st, opt_st
end

function plot_training(ITER, _LOSS, LOSS_; dir = nothing)
    z = findall(iszero, ITER)

    # fix ITER to account for multiple training loops
    if length(z) > 1
            for i in 2:length(z)-1
            idx =  z[i]:z[i+1] - 1
            ITER[idx] .+= ITER[z[i] - 1]
        end
        ITER[z[end]:end] .+= ITER[z[end] - 1]
    end

    plt = plot(title = "Training Plot", yaxis = :log,
               xlabel = "Epochs", ylabel = "Loss (MSE)",
               ylims = (minimum(_LOSS) / 10, maximum(LOSS_) * 10))

    plot!(plt, ITER, _LOSS, w = 2.0, c = :green, label = "Train Dataset")
    plot!(plt, ITER, LOSS_, w = 2.0, c = :red, label = "Test Dataset")

    vline!(plt, ITER[z[2:end]], c = :black, w = 2.0, label = nothing)

    if !isnothing(dir)
        png(plt, joinpath(dir, "plt_training"))
    end

    plt
end

"""
$SIGNATURES

"""
function visualize(V::Spaces.AbstractSpace{<:Any, 1},
    _data::NTuple{2, Any},
    data_::NTuple{2, Any},
    NN::Lux.AbstractExplicitLayer,
    p,
    st;
    nsamples = 5,
    dir = nothing,
)
    x, = points(V)

    _x, _ŷ = _data
    x_, ŷ_ = data_

    _y = NN(_x, p, st)[1]
    y_ = NN(x_, p, st)[1]

    N, _K = size(_y)[end-1:end]
    N, K_ = size(y_)[end-1:end]

    _I = rand(1:_K, nsamples)
    I_ = rand(1:K_, nsamples)
    n = 4
    ms = 4

    cmap = range(HSV(0,1,1), stop=HSV(-360,1,1), length = nsamples + 1)

    # Trajectory plots
    kw = (; legend = false, xlabel = "x", ylabel = "u(x)")

    _p0 = plot(;title = "Training Comparison", kw...)
    p0_ = plot(;title = "Testing Comparison" , kw...)

    for i in 1:nsamples
        c = cmap[i]
        _i = _I[i]
        i_ = I_[i]

        kw_data = (; markersize = ms, c = c,)
        kw_pred = (; s = :solid, w = 2.0, c = c)

        _idx, idx_ = if _y isa AbstractMatrix
            (Colon(), _i), (Colon(), i_)
        elseif _y isa AbstractArray{<:Any, 3}
            # plot only the first output channel
            # make separate dispatches for visualize later
            (1, Colon(), _i), (1, Colon(), i_)
        end

        # training
        __y = _y[_idx...]
        __ŷ = _ŷ[_idx...]
        scatter!(_p0, x[begin:n:end], __ŷ[begin:n:end]; kw_data...)
        plot!(_p0, x, __y; kw_pred...)

        # testing
        y__ = y_[idx_...]
        ŷ__ = ŷ_[idx_...]
        scatter!(p0_, x[begin:n:end], ŷ__[begin:n:end]; kw_data...)
        plot!(p0_, x, y__; kw_pred...)
    end

    # R2 plots

    _R2 = round(rsquare(_y, _ŷ), digits = 8)
    R2_ = round(rsquare(y_, ŷ_), digits = 8)

    kw = (; legend = false, xlabel = "Data", ylabel = "Prediction", aspect_ratio = :equal)

    _p1 = plot(; title = "Training R² = $_R2", kw...)
    p1_ = plot(; title = "Testing R² = $R2_", kw...)

    scatter!(_p1, vec(_y), vec(_ŷ), ms = 1)
    scatter!(p1_, vec(y_), vec(ŷ_), ms = 1)

    _l = [extrema(_y)...]
    l_ = [extrema(y_)...]
    plot!(_p1, _l, _l, w = 4.0, c = :red)
    plot!(p1_, l_, l_, w = 4.0, c = :red)

    plts = _p0, p0_, _p1, p1_

    if !isnothing(dir)
        png(plts[1],   joinpath(dir, "plt_traj_train"))
        png(plts[2],   joinpath(dir, "plt_traj_test"))
        png(plts[3],   joinpath(dir, "plt_r2_train"))
        png(plts[4],   joinpath(dir, "plt_r2_test"))
    end

    plts
end

function visualize(V::Spaces.AbstractSpace{<:Any, 2}, args...; kwargs...)
end
#
