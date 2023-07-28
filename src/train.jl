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
- `nepochs`: Number of epochs for each optimization cycle
- `cbstep`: prompt `callback` function every `cbstep` epochs
- `dir`: directory to save model, plots
- `io`: io for printing stats
"""
function train_model(
    rng::Random.AbstractRNG,
    NN::Lux.AbstractExplicitLayer,
    _data::NTuple{2, AbstractArray},
    data_::NTuple{2, AbstractArray},
    V::Spaces.AbstractSpace,
    opt::Optimisers.AbstractRule = Optimisers.Adam();
    batchsize::Int = 32,
    learning_rates::NTuple{N, Float32} = (1f-3,),
    nepochs::NTuple{N} = (100,),
    cbstep::Int = 1,
    dir = "",
    name = "model",
    nsamples = 5,
    io::IO = stdout,
    p = nothing,  # initial parameters
    st = nothing, # initial state
    lossfun = mse,
    device = Lux.cpu,
    make_plots = true,
) where{N}

    @assert length(learning_rates) == length(nepochs)

    # make directory for saving model
    mkpath(dir)

    # create data loaders
    _loader = DataLoader(_data; batchsize, rng, shuffle = true)
    loader_ = DataLoader(data_; batchsize, rng, shuffle = true)

    if device == Lux.gpu
        _loader, loader_ = (_loader, loader_) .|> CuIterator
    end

    # full batch statistics functions
    _stats = (p, st; io = io) -> statistics(NN, p, st, _loader; io)
    stats_ = (p, st; io = io) -> statistics(NN, p, st, loader_; io)

    # full batch losses for CB
    _loss = (p, st) -> batch_metric(NN, p, st, _loader, lossfun)
    loss_ = (p, st) -> batch_metric(NN, p, st, loader_, lossfun)

    # callback functions
    EPOCH = Int[]
    _LOSS = Float32[]
    LOSS_ = Float32[]

    # callback for printing statistics
    CB = (p, st; io = io) -> callback(p, st; io, _loss, _stats, loss_, stats_)

    # callback for training
    cb = (p, st, epoch, nepoch; io = io) -> callback(p, st; io,
                                                _loss, _LOSS, loss_, LOSS_,
                                                EPOCH, epoch, nepoch, step = cbstep)

    # parameters
    _p, _st = Lux.setup(rng, NN)
    # _p = _p |> ComponentArray # not nice for real + complex

    p  = isnothing(p ) ? _p  : p
    st = isnothing(st) ? _st : st

    p, st = (p, st) |> device

    # print stats
    CB(p, st)

    println(io, "#======================#")
    println(io, "Starting Trainig Loop")
    println(io, "Optimizer: $opt")
    println(io, "#======================#")

    # set up optimizer
    opt_st = nothing

    for i in eachindex(nepochs)
        learning_rate = learning_rates[i]
        nepoch = nepochs[i]

        @set! opt.eta = learning_rate

        println(io, "#======================#")
        println(io, "Learning Rate: $learning_rate, EPOCHS: $nepoch")
        println(io, "#======================#")

        CUDA.@time p, st, opt_st = optimize(NN, p, st, _loader, nepoch; lossfun, opt, opt_st, cb, io)

        CB(p, st)
    end

    # save statistics
    statsfile = open(joinpath(dir, "statistics.txt"), "w")
    CB(p, st; io = statsfile)
    close(statsfile)

    # transfer model to host device
    p, st = (p, st) |> Lux.cpu

    # visualization
    if make_plots
        plot_training(EPOCH, _LOSS, LOSS_; dir)
        visualize(V, _data, data_, NN, p, st; nsamples, dir)
    end

    model = NN, p, st
    STATS = EPOCH, _LOSS, LOSS_
 
    BSON.@save joinpath(dir, "$name.bson") _data data_ model

    model, STATS
end

#===============================================================#
function batch_metric(NN, p, st, loader, metric)
    x, ŷ = first(loader)
    y = NN(x, p, st)[1]
    metric(y, ŷ)
end

function fullbatch_metric(NN, p, st, loader, metric, ismean = false)
    L = 0f0
    N = 0

    for (x, ŷ) in loader
        y = NN(x, p, st)[1]
        l = metric(y, ŷ)

        if ismean
            n = length(ŷ)
            N += n
            L += l * n
        else
            L += l
        end
    end

    ismean ? L / N : L
end

"""
$SIGNATURES

"""
function statistics(NN::Lux.AbstractExplicitLayer, p, st, loader;
    io::Union{Nothing, IO} = stdout
)
    N = 0
    SUM   = 0f0
    VAR   = 0f0
    ABSER = 0f0
    SQRER = 0f0

    MAXER = 0f0

    for (x, ŷ) in loader
        y, _ = NN(x, p, st)
        Δy = y - ŷ

        N += length(ŷ)
        SUM += sum(y)

        ABSER += sum(abs , Δy)
        SQRER += sum(abs2, Δy)
        MAXER  = max(MAXER, norm(Δy, Inf))
    end

    ȳ   = SUM / N
    MSE = SQRER / N

    meanAE = ABSER / N
    maxAE  = MAXER

    # variance
    for (x, _) in loader
        y, _ = NN(x, p, st)

        VAR += sum(abs2, y .- ȳ) / N
    end

    R2 = 1f0 - MSE / (VAR + eps(Float32))

    # rel   = Δy ./ ŷ
    # meanRE = norm(rel, 1) / length(ŷ)
    # maxRE  = norm(rel, Inf)

    if !isnothing(io)
        str = ""
        str *= string("R² score:       ", round(R2    , digits=8), "\n")
        str *= string("mean SQR error: ", round(MSE   , digits=8), "\n")
        str *= string("mean ABS error: ", round(meanAE, digits=8), "\n")
        str *= string("max  ABS error: ", round(maxAE , digits=8), "\n")
        # str *= string("mean REL error: ", round(meanRE, digits=8), "\n")
        # str *= string("max  REL error: ", round(maxRE , digits=8))

        println(io, str)
    end

    R2, MSE, meanAE, maxAE #, meanRE, maxRE
end

"""
$SIGNATURES

"""
function callback(p, st; io::Union{Nothing, IO} = stdout,
                  _loss = nothing, _LOSS = nothing, _stats = nothing,
                  loss_ = nothing, LOSS_ = nothing, stats_ = nothing,
                  epoch = nothing, step = 0, nepoch = 0, EPOCH = nothing,
                 )

    str = if !isnothing(epoch)
        step = iszero(step) ? 10 : step

        if epoch % step == 0 || epoch == 1 || epoch == nepoch
            "Epoch [$epoch / $nepoch]"
        else
            return
        end
    else
        ""
    end

    # log epochs
    if !isnothing(EPOCH) & !isnothing(epoch)
        push!(EPOCH, epoch)
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
        str *= string("\t TRAIN LOSS: ", round(_l, digits=8))
    end

    if !isnothing(l_)
        str *= string("\t TEST LOSS: ", round(l_, digits=8))
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

#===============================================================#
struct Loss{TNN, Tst, Tdata, Tl}
    NN::TNN
    st::Tst
    data::Tdata
    lossfun::Tl
end

function (L::Loss)(p)
    x, ŷ = L.data

    y, st = L.NN(x, p, L.st)
    L.lossfun(y, ŷ), st # Lux interface
end

function grad(loss::Loss, p)
    (l, st), pb = Zygote.pullback(loss, p)
    gr = pb((one.(l), nothing))[1]

    l, st, gr
end

"""
$SIGNATURES

Train parameters `p` to minimize `loss` using optimization strategy `opt`.

# Arguments
- Loss signature: `loss(p, st) -> y, st`
- Callback signature: `cb(p, st epoch, nepochs) -> nothing` 
"""
function optimize(NN, p, st, loader, nepochs;
    lossfun = mse,
    opt = Optimisers.Adam(),
    opt_st = nothing,
    cb = nothing,
    io::Union{Nothing, IO} = stdout,
)

    # print stats
    !isnothing(cb) && cb(p, st, 0, nepochs; io)

    function loss(x, ŷ, p, st)
        y, st = NN(x, p, st)
        lossfun(y, ŷ), st
    end

    # warm up
    begin
        loss = Loss(NN, st, first(loader), lossfun)
        grad(loss, p)
    end

    # init optimizer
    opt_st = isnothing(opt_st) ? Optimisers.setup(opt, p) : opt_st

    for epoch in 1:nepochs
        for batch in loader
            loss = Loss(NN, st, batch, lossfun)
            l, st, g = grad(loss, p)
            opt_st, p = Optimisers.update!(opt_st, p, g)

            println(io, "Epoch [$epoch / $nepochs]" * "\t Batch loss: $l")

            GC.gc(false)
        end
        # GC.gc(true)

        # todo: make this async
        println(io, "#=======================#")
        !isnothing(cb) && cb(p, st, epoch, nepochs; io)
        println(io, "#=======================#")
    end

    p, st, opt_st
end

#===============================================================#
function plot_training(EPOCH, _LOSS, LOSS_; dir = nothing)
    z = findall(iszero, EPOCH)

    # fix EPOCH to account for multiple training loops
    if length(z) > 1
            for i in 2:length(z)-1
            idx =  z[i]:z[i+1] - 1
            EPOCH[idx] .+= EPOCH[z[i] - 1]
        end
        EPOCH[z[end]:end] .+= EPOCH[z[end] - 1]
    end

    plt = plot(title = "Training Plot", yaxis = :log,
               xlabel = "Epochs", ylabel = "Loss (MSE)",
               ylims = (minimum(_LOSS) / 10, maximum(LOSS_) * 10))

    plot!(plt, EPOCH, _LOSS, w = 2.0, c = :green, label = "Train Dataset")
    plot!(plt, EPOCH, LOSS_, w = 2.0, c = :red, label = "Test Dataset")

    vline!(plt, EPOCH[z[2:end]], c = :black, w = 2.0, label = nothing)

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
#===============================================================#
#
