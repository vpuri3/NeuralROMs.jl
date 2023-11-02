 #
"""
$SIGNATURES

# Arguments
- `NN`: Lux neural network
- `_data`: training data as `(x, y)`. `x` may be an AbstractArray or a tuple of arrays
- `data_`: testing data (same requirement as `_data)

# Keyword Arguments
- `rng`: random nunmber generator
- `_batchsize/batchsize_`: train/test batch size
- `opts/nepochs`: `NTuple` of optimizers, # epochs per optimizer
- `cbstep`: prompt `callback` function every `cbstep` epochs
- `dir/name`: directory to save model, plots, model name
- `io`: io for printing stats
- `p/st`: initial model parameter, state. if nothing, initialized with `Lux.setup(rng, NN)`
"""
function train_model(
    NN::Lux.AbstractExplicitLayer,
    _data::NTuple{2, Any},
    data_::NTuple{2, Any} = _data;
#
    rng::Random.AbstractRNG = Random.default_rng(),
#
    _batchsize::Int = 32,
    batchsize_::Int = _batchsize,
    __batchsize::Int = size(_data[2][end]), # > batchsize for BFGS, callback
#
    opts::NTuple{M, Any} = (Optimisers.Adam(1f-3),),
    nepochs::NTuple{M, Int} = (100,),
#
    dir::String = "dump",
    name = "model",
    metadata = nothing,
    io::IO = stdout,
#
    p = nothing,
    st = nothing,
    lossfun = mse,
    device = Lux.cpu_device,
#
    early_stopping::Union{Bool, Nothing} = nothing,
    patience::Union{Int, Nothing} = nothing,
) where{M}

    # make directory for saving model
    mkpath(dir)

    # create data loaders
    _loader  = DataLoader(_data; batchsize = _batchsize , rng, shuffle = true)
    loader_  = DataLoader(data_; batchsize = batchsize_ , rng, shuffle = true)
    __loader = DataLoader(_data; batchsize = __batchsize, rng)

    if device isa Lux.LuxCUDADevice
        _loader, loader_, __loader = (_loader, loader_, __loader) .|> CuIterator
    end

    # callback functions
    EPOCH = Int[]
    _LOSS = Float32[]
    LOSS_ = Float32[]
    STATS = EPOCH, _LOSS, LOSS_

    cb = make_callback(NN, __loader, loader_, lossfun; io, STATS, stats = false)
    cb_stats = make_callback(NN, __loader, loader_, lossfun; io, stats = true)

    # parameters
    _p, _st = Lux.setup(rng, NN)

    p  = isnothing(p ) ? _p  : p
    st = isnothing(st) ? _st : st

    _p = p |> ComponentArray
    p = isreal(_p) ? _p : p

    p, st = (p, st) |> device

    # print stats
    cb_stats(p, st)

    println(io, "#======================#")
    println(io, "Starting Trainig Loop")
    println(io, "#======================#")

    # set up optimizer
    # # TODO
    st = Lux.trainmode(st) # https://github.com/LuxDL/Lux.jl/issues/432
    opt_st = nothing

    time0 = time()
    for i in eachindex(nepochs)
        time1 = time()

        opt = opts[i]
        nepoch = nepochs[i]

        println(io, "#======================#")
        println(io, "Optimization Round $i, EPOCHS: $nepoch")
        println(io, "Optimizer $opt")
        println(io, "#======================#")

        args = (opt, NN, p, st, nepoch, _loader, loader_, __loader)
        kwargs = (;lossfun, opt_st, cb, io, early_stopping, patience)

        @time p, st, opt_st = optimize(args...; kwargs...)

        time2 = time()
        t21 = round(time2 - time1; sigdigits = 8)
        t20 = round(time2 - time0; sigdigits = 8)
        println(io, "#======================#")
        println(io, "Optimization Round $i done")
        println(io, "Time: $(t21) || Total time: $(t20)")
        println(io, "#======================#")

        cb_stats(p, st)
    end

    println(io, "#======================#")
    println(io, "Optimization done")
    println(io, "#======================#")

    # TODO - output a train.log file with timings
    # add ProgressMeters.jl, or TensorBoardLogger.jl

    # save statistics
    statsfile = open(joinpath(dir, "statistics.txt"), "w")
    cb_stats(p, st; io = statsfile)
    close(statsfile)

    # transfer model to host device
    p, st = (p, st) |> Lux.cpu

    # training plot
    plot_training(STATS...; dir)

    model = NN, p, st
 
    # save
    filename = joinpath(dir, "$name.jld2")
    isfile(filename) && rm(filename)
    jldsave(filename; model, metadata, STATS)

    model, STATS
end

#===============================================================#
"""
    minibatch_metric(NN, p, st, loader, lossfun, ismean) -> l

Only for callbacks. Enforce this by setting Lux.testmode

- `NN, p, st`: neural network
- `loader`: data loader
- `lossfun`: loss function: (x::Array, y::Array) -> l::Real
"""
function minibatch_metric(NN, p, st, loader, lossfun)
    st = Lux.testmode(st) # https://github.com/LuxDL/Lux.jl/issues/432
    x, ŷ = first(loader)
    y, _ = NN(x, p, st)
    lossfun(y, ŷ)
end

"""
    fullbatch_metric(NN, p, st, loader, lossfun, ismean) -> l

Only for callbacks. Enforce this by setting Lux.testmode

- `NN, p, st`: neural network
- `loader`: data loader
- `lossfun`: loss function: (x::Array, y::Array) -> l::Real
- `ismean`: lossfun takes a mean
"""
function fullbatch_metric(NN, p, st, loader, lossfun, ismean = true)
    L = 0f0
    N = 0

    st = Lux.testmode(st) # https://github.com/LuxDL/Lux.jl/issues/432

    for (x, ŷ) in loader
        y = NN(x, p, st)[1]
        l = lossfun(y, ŷ)

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
function statistics(
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    loader::Union{CuIterator, MLUtils.DataLoader};
    io::Union{Nothing, IO} = stdout,
)
    st = Lux.testmode(st) # https://github.com/LuxDL/Lux.jl/issues/432

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
        str *= string("R² score:                   ", round(R2    ; sigdigits=8), "\n")
        str *= string("MSE (mean SQR error):       ", round(MSE   ; sigdigits=8), "\n")
        str *= string("RMSE (root mean SQR error): ", round(RMSE  ; sigdigits=8), "\n")
        str *= string("MAE (mean ABS error):       ", round(meanAE; sigdigits=8), "\n")
        str *= string("maxAE (max ABS error)       ", round(maxAE ; sigdigits=8), "\n")
        # str *= string("mean REL error: ", round(meanRE, digits=8), "\n")
        # str *= string("max  REL error: ", round(maxRE , digits=8))

        println(io, str)
    end

    R2, MSE, meanAE, maxAE #, meanRE, maxRE
end

#===============================================================#
"""
$SIGNATURES
"""
function make_callback(
    NN::Lux.AbstractExplicitLayer,
    _loader::Union{CuIterator, MLUtils.DataLoader},
    loader_::Union{CuIterator, MLUtils.DataLoader},
    lossfun;
    STATS::Union{Nothing, NTuple{3, Any}} = nothing,
    stats::Bool = false,
    io = stdout,
)
    _loss = (p, st) -> fullbatch_metric(NN, p, st, _loader, lossfun)
    loss_ = (p, st) -> fullbatch_metric(NN, p, st, loader_, lossfun)

    kwargs = (; _loss, loss_,)

    if stats
        _stats = (p, st; io = io) -> statistics(NN, p, st, _loader; io)
        stats_ = (p, st; io = io) -> statistics(NN, p, st, loader_; io)

        kwargs = (;kwargs..., _stats, stats_)
    end

    if !isnothing(STATS)
        kwargs = (;kwargs..., STATS)
    end

    (p, st; epoch = 0, nepoch = 0, io = io) -> callback(p, st; io, epoch, nepoch, kwargs...)
end

"""
$SIGNATURES

"""
function callback(p, st;
    io::Union{Nothing, IO} = stdout,
    #
    _loss  = nothing,
    loss_  = nothing,
    #
    _stats = nothing,
    stats_ = nothing,
    #
    STATS  = nothing,
    #
    epoch  = nothing,
    nepoch = 0,
)
    EPOCH, _LOSS, LOSS_ = isnothing(STATS) ? ntuple(Returns(nothing), 3) : STATS

    # println(io, "Epoch [$epoch[] / $nepoch]"
    #     * "\t Loss: $l || MSE: $(ms) || MAE: $(ma) || MAXAE: $(mx)")

    str = if !isnothing(epoch)
        cbstep = 1
        if epoch % cbstep == 0 || epoch == 1 || epoch == nepoch
            "Epoch [$epoch / $nepoch]\t"
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
        _l = _loss(p, st)
        !isnothing(_LOSS) && push!(_LOSS, _l)
        _l
    else
        nothing
    end

    # log test loss
    l_ = if !isnothing(loss_)
        l_ = loss_(p, st)
        !isnothing(LOSS_) && push!(LOSS_, l_)
        l_
    else
        nothing
    end

    isnothing(io) && return

    if !isnothing(_l)
        str *= string("TRAIN LOSS: ", round(_l; sigdigits=8))
    end

    if !isnothing(l_)
        str *= string(" || TEST LOSS: ", round(l_; sigdigits=8))
    end

    if !isnothing(io)
        println(io, str)
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

    _l, l_
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
#===============================================================#

"""
$SIGNATURES

Train parameters `p` to minimize `loss` using optimization strategy `opt`.

# Arguments
- Loss signature: `loss(p, st) -> y, st`
- Callback signature: `cb(p, st epoch, nepoch) -> nothing` 
"""
function optimize(
    opt::Optimisers.AbstractRule,
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    nepoch::Integer,
    _loader::Union{CuIterator, MLUtils.DataLoader},
    loader_::Union{CuIterator, MLUtils.DataLoader},
    __loader::Union{CuIterator, MLUtils.DataLoader} = _loader;
    lossfun = mse,
    opt_st = nothing,
    cb = nothing,
    io::Union{Nothing, IO} = stdout,
    early_stopping::Union{Bool, Nothing} = nothing,
    patience::Union{Int, Nothing} = nothing,
    kwargs...,
)
    cb = isnothing(cb) ? make_callback(NN, __loader, loader_, lossfun; io) : cb

    # print stats
    _, l_ = cb(p, st; epoch = 0, nepoch, io)

    # warm up
    loss = Loss(NN, st, first(_loader), lossfun)
    grad(loss, p)

    # set up early_stopping
    early_stopping = isnothing(early_stopping) ? true : early_stopping
    patience = isnothing(patience) ? round(Int, nepoch // 4) : patience
    minconfig = make_minconfig(early_stopping, patience, l_, p, st, opt_st)

    # init optimizer
    opt_st = isnothing(opt_st) ? Optimisers.setup(opt, p) : opt_st

    for epoch in 1:nepoch
        for batch in _loader
            loss = Loss(NN, st, batch, lossfun)
            l, st, g = grad(loss, p)
            opt_st, p = Optimisers.update!(opt_st, p, g)

            println(io, "Epoch [$epoch / $nepoch]" * "\t Batch loss: $l")
        end

        println(io, "#=======================#")

        # callback, early stopping
        _, l_ = cb(p, st; epoch, nepoch, io)
        minconfig, ifbreak = update_minconfig(minconfig, l_, p, st, opt_st; io)
        ifbreak && break

        println(io, "#=======================#")

    end

    minconfig.p, minconfig.st, minconfig.opt_st
end

"""
# references

https://docs.sciml.ai/Optimization/stable/tutorials/minibatch/
https://lux.csail.mit.edu/dev/tutorials/advanced/1_GravitationalWaveForm#training-the-neural-network

"""
function optimize(
    opt::Optim.AbstractOptimizer,
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    nepoch::Integer,
    _loader::Union{CuIterator, MLUtils.DataLoader},
    loader_::Union{CuIterator, MLUtils.DataLoader},
    __loader::Union{CuIterator, MLUtils.DataLoader} = _loader;
    lossfun = mse,
    opt_st = nothing,
    cb = nothing,
    io::Union{Nothing, IO} = stdout,
    early_stopping::Union{Bool, Nothing} = nothing,
    patience::Union{Int, Nothing} = nothing,
    kwargs...
)
    if opt isa Union{
        Optim.Newton,
        Optim.BFGS,
        Optim.LBFGS,
        }

        # NOTE: _loader, __loader should have the same data
        @assert size(_loader.data) == size(__loader.data)

        dsize = size(_loader.data)[end]
        bsize = __loader.batchsize

        _loader = __loader
        @info "Using optimizer " * string(opt) * " with batchsize $bsize" *
             " with data set of $dsize samples."

        @warn "Hessian-based optimizers such as Newton / BFGS / L-BFGS do
         not work with mini-batching. Set batchsize to equal data-size,
         or else the method may be unstable. If you want a stochastic
         optimizer, try `Optimisers.jl`."
    end

    # callback
    cb = isnothing(cb) ? make_callback(NN, _loader, loader_, lossfun) : cb

    # early stopping
    early_stopping = isnothing(early_stopping) ? true : early_stopping
    patience = isnothing(patience) ? round(Int, nepoch // 4) : patience
    mincfg = Ref(make_minconfig(early_stopping, patience, Inf32, p, st, opt_st))

    # current state
    state = Ref(st)
    epoch = Ref(0)

    #======================#
    # optimizer functions
    #======================#

    function optloss(optx, optp, batch...)
        xdata, ydata = batch

        p, st = optx, state[]
        ypred, st = NN(xdata, p, st)
        lossfun(ydata, ypred), batch, ypred, st
    end

    function optcb(p, l, batch, ypred, st)

        # TODO - finish minibatching here
        # # TODO - optcb is called at every minibatch. only call at epoch
        # if minibatch
        #     ll = round(l; sigdigits = 8)
        #     print(io, "Epoch [$(epoch[]) / $(nepoch)]\tBatch Loss: $(ll)")
        #     return false
        # end

        state[] = st
        println(io, "#=======================#")

        _, l_ = cb(p, st; epoch = epoch[], nepoch, io)
        minconfig, ifbreak = update_minconfig(mincfg[], l_, p, st, opt_st; io)
        mincfg[] = minconfig
        epoch[] += 1

        return ifbreak
    end

    #======================#
    # set up optimization solve
    #======================#
    adtype  = AutoZygote()
    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, p, st)

    @time optres = solve(optprob, opt, ncycle(_loader, nepoch); callback = optcb)

    obj = round(optres.objective; sigdigits = 8)
    tim = round(optres.solve_time; sigdigits = 8)
    println(io, "#=======================#")
    @show optres.retcode
    println(io, "Achieved objective value $(obj) in time $(tim)s.")
    println(io, "#=======================#")

    mincfg[].p, mincfg[].st, mincfg[].opt_st
end

"""
early stopping based on mini-batch loss from test set
https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_4_early_stop.ipynb
"""
function make_minconfig(early_stopping, patience, l, p, st, opt_st)
    (; count = 0, early_stopping, patience, l, p, st, opt_st)
end

function update_minconfig(
    minconfig::NamedTuple,
    l::Real,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    opt_st;
    io::Union{IO, Nothing} = stdout,
)
    ifbreak = false

    if l < minconfig.l
        println(io, "Improvement in loss found: $(l) < $(minconfig.l)")
        minconfig = (; minconfig..., count = 0, l, p, st, opt_st)
    else
        println(io, "No improvement in loss found in the last "
            * "$(minconfig.count) epochs. Here, $(l) > $(minconfig.l)")
        @set! minconfig.count = minconfig.count + 1
    end

    if (minconfig.count >= minconfig.patience) & minconfig.early_stopping
        println(io, "Early Stopping triggered after $(minconfig.count)"
            * "epochs of no improvement.")
        ifbreak = true
    end

    minconfig, ifbreak
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
