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
    _data::NTuple{2, AbstractArray{Float32}},
    data_::NTuple{2, AbstractArray{Float32}},
    V::Union{Nothing, Spaces.AbstractSpace},
    opt::Optimisers.AbstractRule = Optimisers.Adam();
    batchsize::Int = 32,
    batchsize_::Union{Int, Nothing} = nothing,
    learning_rates = (1f-3,),
    nepochs = (100,),
    cbstep::Int = 1,
    dir = "",
    name = "model",
    nsamples::Int = 5,
    io::IO = stdout,
    p = nothing,        # initial parameters
    st = nothing,       # initial state
    lossfun = mse,
    device = Lux.cpu_device,
    make_plots = true,
    patience = 20,
    # format = :CNB, # :NCB = [Nx, Ny, C, K], :CNB = [C, Nx, Ny, K] # TODO
    metadata = nothing,
)

    @assert length(learning_rates) == length(nepochs)

    # make directory for saving model
    mkpath(dir)

    # create data loaders
    _batchsize = batchsize
    batchsize_ = isnothing(batchsize_) ? batchsize : batchsize_

    _loader = DataLoader(_data; batchsize = _batchsize, rng, shuffle = true)
    loader_ = DataLoader(data_; batchsize = batchsize_, rng, shuffle = true)

    if (device === Lux.gpu) | (device isa Lux.LuxCUDADevice)
        _loader, loader_ = (_loader, loader_) .|> CuIterator
    end

    # full batch statistics functions
    _stats = (p, st; io = io) -> statistics(NN, p, st, _loader; io, mode = :train)
    stats_ = (p, st; io = io) -> statistics(NN, p, st, loader_; io, mode = :test)

    # full batch losses for cb_stats
    _loss = (p, st) -> minibatch_metric(NN, p, st, _loader, lossfun)
    loss_ = (p, st) -> minibatch_metric(NN, p, st, loader_, lossfun)

    # callback functions
    EPOCH = Int[]
    _LOSS = Float32[]
    _LOSS_MINIBATCH = Float32[]
    LOSS_ = Float32[]

    # callback for printing statistics
    cb_stats = (p, st; io = io) -> callback(p, st; io, _loss, _stats, loss_, stats_)

    # cb_batch = 

    # callback for training
    cb_epoch = (p, st, epoch, nepoch; io = io) -> callback(p, st; io,
                                                _loss, _LOSS, loss_, LOSS_,
                                                EPOCH, epoch, nepoch, step = cbstep)

    # parameters
    _p, _st = Lux.setup(rng, NN)
    # _p = _p |> ComponentArray # not nice for real + complex

    p  = isnothing(p ) ? _p  : p
    st = isnothing(st) ? _st : st

    p, st = (p, st) |> device

    # print stats
    cb_stats(p, st)

    println(io, "#======================#")
    println(io, "Starting Trainig Loop")
    println(io, "Optimizer: $opt")
    println(io, "#======================#")

    # set up optimizer
    st = Lux.trainmode(st)
    opt_st = nothing

    for i in eachindex(nepochs)
        learning_rate = learning_rates[i]
        nepoch = nepochs[i]

        @set! opt.eta = learning_rate

        println(io, "#======================#")
        println(io, "Learning Rate: $learning_rate, EPOCHS: $nepoch")
        println(io, "#======================#")

        if (device === Lux.gpu) | (device isa Lux.LuxCUDADevice)
            CUDA.@time p, st, opt_st = optimize(NN, p, st, _loader, loader_, nepoch;
                lossfun, opt, opt_st, cb_epoch, io, patience)
        else
            @time p, st, opt_st = optimize(NN, p, st, _loader, loader_, nepoch;
                lossfun, opt, opt_st, cb_epoch, io, patience)
        end

        cb_stats(p, st)
    end

    # TODO - output a train.log file with timings
    # add ProgressMeters.jl, or TensorBoardLogger.jl

    # save statistics
    statsfile = open(joinpath(dir, "statistics.txt"), "w")
    cb_stats(p, st; io = statsfile)
    close(statsfile)

    # transfer model to host device
    p, st = (p, st) |> Lux.cpu

    # training plot
    if make_plots
        plot_training(EPOCH, _LOSS, LOSS_; dir)
    end

    model = NN, p, st
    STATS = EPOCH, _LOSS, LOSS_
 
    BSON.@save joinpath(dir, "$name.bson") model metadata

    model, STATS
end
#===============================================================#

function autodecode(arguments)

end

#===============================================================#
function minibatch_metric(NN, p, st, loader, metric)
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
    mode::Symbol = :train,
    io::Union{Nothing, IO} = stdout,
)
    N = 0
    SUM   = 0f0
    VAR   = 0f0
    ABSER = 0f0
    SQRER = 0f0

    MAXER = 0f0

    st = if mode === :train
        Lux.trainmode(st)
    elseif mode === :test
        Lux.testmode(st)
    else
        error("mode must be :train or :test")
    end

    for (x, ŷ) in loader
        y, _ = NN(x, p, st)
        Δy = y - ŷ

        N += length(ŷ)
        SUM += sum(y)

        ABSER += sum(abs , Δy)
        SQRER += sum(abs2, Δy)
        MAXER  = max(MAXER, norm(Δy, Inf32))
    end

    ȳ   = SUM / N
    MSE = SQRER / N
    RMSE = sqrt(MSE)

    meanAE = ABSER / N
    maxAE  = MAXER # TODO - seems off

    # variance
    for (x, _) in loader
        y, _ = NN(x, p, st)

        VAR += sum(abs2, y .- ȳ) / N
    end

    R2 = 1f0 - MSE / (VAR + eps(Float32))

    # rel   = Δy ./ ŷ
    # meanRE = norm(rel, 1) / length(ŷ)
    # maxRE  = norm(rel, Inf32)

    if !isnothing(io)
        str = ""
        str *= string("R² score:                   ", round(R2    , digits=8), "\n")
        str *= string("MSE (mean SQR error):       ", round(MSE   , digits=8), "\n")
        str *= string("RMSE (root mean SQR error): ", round(RMSE  , digits=8), "\n")
        str *= string("MAE (mean ABS error):       ", round(meanAE, digits=8), "\n")
        str *= string("maxAE (max ABS error)       ", round(maxAE , digits=8), "\n")
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
function optimize(NN, p, st, _loader, loader_, nepochs;
    lossfun = mse,
    opt = Optimisers.Adam(),
    opt_st = nothing,
    cb_batch = nothing,
    cb_epoch = nothing,
    io::Union{Nothing, IO} = stdout,
    patience = 20,
)

    # print stats
    !isnothing(cb_epoch) && cb_epoch(p, st, 0, nepochs; io)

    # warm up
    begin
        loss = Loss(NN, st, first(_loader), lossfun)
        grad(loss, p)
    end

    # get config for early stopping
    minconfig = (; count = 0, patience, l = Inf32, p, st, opt_st)

    # init optimizer
    opt_st = isnothing(opt_st) ? Optimisers.setup(opt, p) : opt_st

    for epoch in 1:nepochs
        for batch in _loader
            loss = Loss(NN, st, batch, lossfun)
            l, st, g = grad(loss, p)
            opt_st, p = Optimisers.update!(opt_st, p, g)

            println(io, "Epoch [$epoch / $nepochs]" * "\t Batch loss: $l")
            !isnothing(cb_batch) && cb_batch(p, st, step)
        end

        println(io, "#=======================#")
        !isnothing(cb_epoch) && cb_epoch(p, st, epoch, nepochs; io)

        # early stopping based on mini-batch loss from test set
        # https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_4_early_stop.ipynb

        l, _ = Loss(NN, st, first(loader_), lossfun)(p)
        if l < minconfig.l
            println(io, "Improvement in loss found: $(l) < $(minconfig.l)")
            minconfig = (; minconfig..., count = 0, l, p, st, opt_st)
        else
            println(io, "No improvement in loss found in the last $(minconfig.count) epochs. Here, $(l) > $(minconfig.l)")
            @set! minconfig.count = minconfig.count + 1
        end
        if minconfig.count >= minconfig.patience
            println(io, "Early Stopping triggered after $(minconfig.count) epochs of no improvement.")
            break
        end

        println(io, "#=======================#")

    end

    minconfig.p, minconfig.st, minconfig.opt_st
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

    plot!(plt, EPOCH, _LOSS, w = 2.0, c = :green, label = "Train Dataset") # (; ribbon = (lower, upper))
    plot!(plt, EPOCH, LOSS_, w = 2.0, c = :red, label = "Test Dataset")

    vline!(plt, EPOCH[z[2:end]], c = :black, w = 2.0, label = nothing)

    if !isnothing(dir)
        png(plt, joinpath(dir, "plt_training"))
            end

    plt
end

#===============================================================#
#
