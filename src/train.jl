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
    data_::Union{Nothing,NTuple{2, Any}} = nothing;
#
    rng::Random.AbstractRNG = Random.default_rng(),
#
    _batchsize::Int = 32,
    batchsize_::Int = numobs(_data),
    __batchsize::Int = batchsize_, # > batchsize for BFGS, callback
#
    opts::NTuple{M, Any} = (Optimisers.Adam(1f-3),),
    nepochs::NTuple{M, Int} = (100,),
    schedules::Union{Nothing, NTuple{M, ParameterSchedulers.AbstractSchedule}} = nothing,
#
    early_stoppings::Union{Bool, NTuple{M, Bool}, Nothing} = nothing,
    patience_fracs::Union{Real, NTuple{M, Real}, Nothing} = nothing,
    weight_decays::Union{Real, NTuple{M, Real}} = 0f0,
#
    dir::String = "dump",
    name::String = "model",
    metadata::NamedTuple = (;),
    io::IO = stdout,
#
    p = nothing,
    st = nothing,
    lossfun = mse,
    device = Lux.cpu_device(),
#
    cb_epoch = nothing, # (NN, p, st) -> nothing
) where{M}

    notestdata = isnothing(data_)

    if notestdata
        data_ = _data
        println("[train_model] No test dataset provided.")
    end

    # create data loaders
    _loader  = DataLoader(_data; batchsize = _batchsize , rng, shuffle = true)
    loader_  = DataLoader(data_; batchsize = batchsize_ , rng, shuffle = true)
    __loader = DataLoader(_data; batchsize = __batchsize, rng)

    if device isa Lux.LuxCUDADevice
        _loader  = _loader  |> CuIterator
        loader_  = loader_  |> CuIterator
        __loader = __loader |> CuIterator
    end

    # callback functions

    # EPOCH, _LOSS, LOSS_, _MSE, MSE_, _MAE, MAE_
    STATS = Int[], Float32[], Float32[], Float32[], Float32[], Float32[], Float32[]

    cb = makecallback(NN, __loader, loader_, lossfun; io, STATS, cb_epoch, notestdata)
    cb_stats = makecallback(NN, __loader, loader_, lossfun; io, stats = true, notestdata)

    # parameters
    _p, _st = Lux.setup(rng, NN)

    p  = isnothing(p)  ? _p  : p
    st = isnothing(st) ? _st : st

    _p = p |> ComponentArray
    p = isreal(_p) ? _p : p

    p, st = (p, st) |> device

    @assert eltype(p) ∈ (Float32, ComplexF32) "eltype(p) = $(eltype(p))"
    # @assert eltype(first(_loader))
    # @assert eltype(data) ∈ (Bool, Int8, Int16, Int32, Float32, ComplexF32)

    println(io, "#======================#")
    println(io, "Starting Trainig Loop")
    println(io, "Model size: $(length(p)) parameters")
    println(io, "#======================#")

    # print stats
    cb_stats(p, st)

    st = Lux.trainmode(st)
    opt_st = nothing

    time0 = time()

    if early_stoppings isa Union{Bool, Nothing}
        early_stoppings = fill(early_stoppings, M)
    end

    if patience_fracs isa Union{Bool, Nothing}
        patience_fracs = fill(patience_fracs, M)
    end

    if weight_decays isa Real
        weight_decays = fill(weight_decays, M)
    end

    for iopt in 1:M
        time1 = time()

        opt = opts[iopt] |> device
        nepoch = nepochs[iopt]
        schedule = isnothing(schedules) ? nothing : schedules[iopt]

        early_stopping = early_stoppings[iopt]
        patience_frac = patience_fracs[iopt]
        patience = isnothing(patience_frac) ? nothing : round(Int, patience_frac * nepoch)

        if !isnothing(opt_st) & isa(opt, Optimisers.AbstractRule)
            @set! opt_st.rule = opt
        end

        weight_decay = weight_decays[iopt]

        if !iszero(weight_decay) & (opt isa OptimiserChain)
            isWD = Base.Fix2(isa, Union{WeightDecay, DecoderWeightDecay, IdxWeightDecay})
            iWD = findall(isWD, opt.opts)

            if isempty(iWD)
                @error "weight_decay = $weight_decay, but no WeightDecay optimizer in $opt"
            else
                @assert length(iWD) == 1 """More than one WeightDecay() found
                    in optimiser chain $opt."""

                iWD = only(iWD)
                @set! opt.opts[iWD].lambda = weight_decay
            end
        end

        println(io, "#======================#")
        println(io, "Optimization Round $iopt, EPOCHS: $nepoch")
        println(io, "Optimizer $opt")
        println(io, "#======================#")

        args = (opt, NN, p, st, nepoch, _loader, loader_, __loader)
        kwargs = (;lossfun, opt_st, cb, io, early_stopping, patience, schedule)

        @time p, st, opt_st = optimize(args...; kwargs...)

        time2 = time()
        t21 = round(time2 - time1; sigdigits = 8)
        t20 = round(time2 - time0; sigdigits = 8)
        println(io, "#======================#")
        println(io, "Optimization Round $iopt done")
        println(io, "Time: $(t21)s || Total time: $(t20)s")
        println(io, "#======================#")

        cb_stats(p, st)

        if iopt == M
            savemodel!(NN, p, st, metadata, STATS, cb_stats, dir, name, iopt)
        else
            savemodel!(NN, p, st, metadata, deepcopy(STATS), cb_stats, dir, name, iopt)
        end

    end

    # TODO - output a train.log file with timings.

    println(io, "#======================#")
    println(io, "Optimization done")
    println(io, "#======================#")

    p, st = Lux.cpu_device()((p, st))

    (NN, p, st), STATS
end

#===============================================================#
function savemodel!( # modifies STATS
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    metadata,
    STATS::NTuple{7, Vector},
    cb::Function,
    dir::String,
    name::String,
    count::Integer,
)
    mkpath(dir)
    count = lpad(count, 2, "0")

    # save statistics
    statsfile = open(joinpath(dir, "statistics_$(count).txt"), "w")
    cb(p, st; io = statsfile)
    close(statsfile)

    # transfer model to host device
    p, st = (p, st) |> Lux.cpu
    model = NN, p, st

    # training plot
    plt = plot_training!(STATS...)
    png(plt, joinpath(dir, "plt_training"))
    display(plt)

    # save model
    if length(name) > 5
        if name[end-4:end] !== ".jld2"
            name = name[1:end-5]
        end
    end

    filename = joinpath(dir, "$(name)_$(count).jld2")
    isfile(filename) && rm(filename)
    jldsave(filename; model, metadata, STATS)

    @info "Saved model at $filename"

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
    lossfun(NN, p, Lux.testmode(st), first(loader))
end

"""
    fullbatch_metric(NN, p, st, loader, lossfun, ismean) -> l

Only for callbacks. Enforce this by setting Lux.testmode

- `NN, p, st`: neural network
- `loader`: data loader
- `lossfun`: loss function: (x::Array, y::Array) -> l::Real
"""
function fullbatch_metric(NN, p, st, loader, lossfun)
    N = 0
    L = 0f0

    SK = nothing # stats keys
    SV = nothing # stats values

    st = Lux.testmode(st)

    for batch in loader
        l, _, stats = lossfun(NN, p, st, batch)

        if isnothing(SK)
            SK = keys(stats)
        end

        n = numobs(batch)
        N += n

        # compute mean stats
        if isnothing(SV)
            SV = values(stats) .* n
        else
            SV = SV .+ values(stats) .* n
        end

        L += l * n
    end

    SV   = SV ./ N
    loss = L   / N
    stats = NamedTuple{SK}(SV)

    loss, stats
end

"""
$SIGNATURES

"""
function printstatistics(
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

    cbound = compute_cbound(NN, p, st)

    if !isnothing(io)
        str = ""
        str *= string("R² score:             ", round(R2     ; sigdigits=8), "\n")
        str *= string("MSE (mean SQR error): ", round(MSE    ; sigdigits=8), "\n")
        str *= string("RMSE (Root MSE):      ", round(RMSE   ; sigdigits=8), "\n")
        str *= string("MAE (mean ABS error): ", round(meanAE ; sigdigits=8), "\n")
        str *= string("maxAE (max ABS error) ", round(maxAE  ; sigdigits=8), "\n")
        # str *= string("mean REL error: ", round(meanRE, digits=8), "\n")
        # str *= string("max  REL error: ", round(maxRE , digits=8))

        str *= string("Lipschitz bound:      ", round(cbound ; sigdigits=8), "\n")

        println(io, str)
    end

    R2, MSE, meanAE, maxAE #, meanRE, maxRE
end

#===============================================================#
"""
$SIGNATURES
"""
function makecallback(
    NN::Lux.AbstractExplicitLayer,
    _loader::Union{CuIterator, MLUtils.DataLoader},
    loader_::Union{CuIterator, MLUtils.DataLoader},
    lossfun;
    STATS::Union{Nothing, NTuple{7, Vector}} = nothing,
    stats::Bool = false,
    io::IO = stdout,
    cb_epoch = nothing, # (NN, p, st) -> nothing
    notestdata::Bool = false,
)
    _loss = (p, st) -> fullbatch_metric(NN, p, st, _loader, lossfun)
    loss_ = (p, st) -> fullbatch_metric(NN, p, st, loader_, lossfun)

    kwargs = (; _loss, loss_, notestdata)

    if stats
        _printstatistics = (p, st; io = io) -> printstatistics(NN, p, st, _loader; io)
        printstatistics_ = (p, st; io = io) -> printstatistics(NN, p, st, loader_; io)

        kwargs = (;kwargs..., _printstatistics, printstatistics_)
    end

    if !isnothing(STATS)

        if lossfun === mse
            STATS = (STATS[1:3]..., STATS[2:3]..., STATS[6], STATS[7])
        end

        if lossfun === mae
            STATS = (STATS[1:5]...,  STATS[2:3]...)
        end

        kwargs = (;kwargs..., STATS)
    end

    function makecallback_internal(p, st; epoch = 0, nepoch = 0, io = io)
        isnothing(cb_epoch) || cb_epoch(NN, p, st)
        callback(p, st; io, epoch, nepoch, kwargs...)
    end
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
    _printstatistics = nothing,
    printstatistics_ = nothing,
    #
    STATS  = nothing,
    #
    epoch  = nothing,
    nepoch = 0,
    #
    notestdata::Bool = false,
)
    EPOCH, _LOSS, LOSS_, _MSE, MSE_, _MAE, MAE_ = isnothing(STATS) ? ntuple(Returns(nothing), 7) : STATS

    if !isnothing(epoch)
        cbstep = 1
        if epoch % cbstep == 0 || epoch == 1 || epoch == nepoch
            !isnothing(io) && print(io, "Epoch [$epoch / $nepoch]\t")
        else
            return
        end
    end

    # log epochs
    if !isnothing(EPOCH) & !isnothing(epoch)
        push!(EPOCH, epoch)
    end

    # log training loss
    _l, _stats = if !isnothing(_loss)
        _loss(p, st)
    else
        nothing, nothing
    end

    # log test loss
    l_, stats_ = if notestdata
        _l, _stats
    else
        if !isnothing(loss_)
            l_, stats_ = loss_(p, st)
            l_, stats_
        else
            nothing, nothing
        end
    end

    !isnothing(_LOSS) && !isnothing(_l) && push!(_LOSS, _l)
    !isnothing(LOSS_) && !isnothing(l_) && push!(LOSS_, l_)

    if !isnothing(_MSE) & haskey(_stats, :mse)
        push!(_MSE, _stats.mse)
    end

    if !isnothing(MSE_) & haskey(stats_, :mse)
        push!(MSE_, stats_.mse)
    end

    if !isnothing(_MAE) & haskey(_stats, :mae)
        push!(_MAE, _stats.mae)
    end

    if !isnothing(MAE_) & haskey(stats_, :mae)
        push!(MAE_, stats_.mae)
    end

    isnothing(io) && return

    if !isnothing(_l)
        print(io, "TRAIN LOSS: ")
        _lprint = round(_l; sigdigits=8)
        printstyled(io, _lprint; color = :magenta)
    end

    if !isnothing(l_)
        print(io, " || TEST LOSS: ")
        lprint_ = round(l_; sigdigits = 8)
        printstyled(io, lprint_; color = :magenta)
    end

    println(io)

    # if !isnothing(_stats)
    #     println(io, "TRAIN STATS:", _stats)
    # end

    # if isnothing(stats_)
    #     println(io, "TRAIN STATS:", stats_)
    # end

    if !isnothing(_printstatistics)
        println(io, "#======================#")
        println(io, "TRAIN STATS")
        _printstatistics(p, st; io)
        println(io, "#======================#")
    end

    if !isnothing(printstatistics_) 
        println(io, "#======================#")
        println(io, "TEST  STATS")
        printstatistics_(p, st; io)
        println(io, "#======================#")
    end

    # terminate optimization
    MSE_MIN = 5f-7
    ifbreak = false

    # # avoid over-fitting on training set
    # if !isnothing(_stats)
    #     if haskey(_stats, :mse)
    #         lmse = _stats[:mse]
    #         if lmse < MSE_MIN
    #             println("Ending optimization")
    #             println("MSE = $lmse < 5f-7 reached on training set.")
    #             ifbreak = true
    #         end
    #     end
    # end
    #
    # # avoid over-fitting on test set
    # if !isnothing(stats_)
    #     if haskey(stats_, :mse)
    #         lmse = stats_[:mse]
    #         if lmse < MSE_MIN
    #             println("Ending optimization")
    #             println("MSE = $lmse < 5f-7 reached on test set.")
    #             ifbreak = true
    #         end
    #     end
    # end

    _l, l_, ifbreak
end

#===============================================================#
struct Loss{TNN, Tst, Tbatch, Tl}
    NN::TNN
    st::Tst
    batch::Tbatch
    lossfun::Tl
end

function (L::Loss)(p)
    L.lossfun(L.NN, p, L.st, L.batch)
end

function grad(loss::Loss, p)
    (l, st, stats), pb = Zygote.pullback(loss, p)
    gr = pb((one.(l), nothing, nothing))[1]

    l, st, stats, gr
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
    schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = nothing,
    kwargs...,
)
    # ensure testing mode
    st = Lux.trainmode(st)

    # make callback
    cb = isnothing(cb) ? makecallback(NN, __loader, loader_, lossfun; io) : cb

    # print stats
    _, l_ = cb(p, st; epoch = 0, nepoch, io)

    # warm up run
    begin
        loss = Loss(NN, st, first(_loader), lossfun)
        grad(loss, p)[3] |> display
    end

    # set up early_stopping
    early_stopping = isnothing(early_stopping) ? true : early_stopping
    patience = isnothing(patience) ? round(Int, nepoch // 5) : patience
    minconfig = make_minconfig(early_stopping, patience, l_, p, st, opt_st)

    # init optimizer
    opt_st = isnothing(opt_st) ? Optimisers.setup(opt, p) : opt_st

    num_batches = length(_loader)

    for epoch in 1:nepoch
        # update LR
        LR = schedule(epoch)
        LR_round = round(LR; sigdigits = 3)
        isnothing(schedule) || Optimisers.adjust!(opt_st, LR)

        # progress bar
        prog = ProgressMeter.Progress(
            num_batches;
            barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
            desc = "Epoch [$epoch / $nepoch] LR: $LR_round",
            dt = 1e-4, barlen = 25, color = :normal, showspeed = true, output = io,
        )

        for batch in _loader
            loss = Loss(NN, st, batch, lossfun)
            l, st, stats, g = grad(loss, p)
            opt_st, p = Optimisers.update!(opt_st, p, g)

            if isnan(l)
                throw(ErrorException("Loss in NaN"))
            end

            # progress bar
            showvalues = Any[(:LOSS, round(l; sigdigits = 8)),]
            !isempty(stats) && push!(showvalues, (:INFO, stats))
            ProgressMeter.next!(prog; showvalues, valuecolor = :magenta)
        end

        ProgressMeter.finish!(prog)

        # callback, early stopping
        _, l_, ifbreak_cb = cb(p, st; epoch, nepoch, io)
        minconfig, ifbreak_mc = update_minconfig(minconfig, l_, p, st, opt_st; io)

        if ifbreak_cb | ifbreak_mc
            break
        end

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

        @warn "Hessian-based optimizers such as Newton / BFGS / L-BFGS do \
        not work with mini-batching. Set batchsize to equal data-size, \
        or else the method may be unstable. If you want a stochastic \
        optimizer, try `Optimisers.jl`."

        dsize = numobs(__loader)
        bsize = numobs(first(__loader))

        @info "Using batchsize $bsize with data set of $dsize samples."

        @assert length(__loader) == 1 "__loader must have exactly one minibatch."

        _loader = __loader
    end

    # ensure testing mode
    st = Lux.trainmode(st)

    # callback
    cb = isnothing(cb) ? makecallback(NN, _loader, loader_, lossfun) : cb

    # early stopping
    early_stopping = isnothing(early_stopping) ? true : early_stopping
    patience = isnothing(patience) ? round(Int, nepoch // 10) : patience
    mincfg = Ref(make_minconfig(early_stopping, patience, Inf32, p, st, opt_st))

    # current state
    state = Ref(st)
    epoch = Ref(0)
    count = Ref(0)
    num_batches = length(_loader)

    #======================#
    # optimizer functions
    #======================#

    function optloss(optx, optp, batch...)
        lossfun(NN, optx, state[], batch)..., batch
    end

    function optcb(optx, l, st, stats, batch)
        count[] += 1
        nextepoch = iszero(count[] % num_batches)

        ll = round(l; sigdigits = 8)
        state[] = st

        if nextepoch
            println(io, "Epoch [$(epoch[]) / $(nepoch)]\tBatch Loss: $(ll)")
            # println(io, "Iter: $(optx.iter), Objective: $(optx.objective)")

            println(io, "#=======================#")

            _, l_ = cb(optx.u, st; epoch = epoch[], nepoch, io)
            minconfig, ifbreak = update_minconfig(mincfg[], l_, optx.u, st, opt_st; io)
            mincfg[] = minconfig
            epoch[] += 1

            return ifbreak
        else
            println(io, "Epoch [$(epoch[]) / $(nepoch)]\tBatch Loss: $(ll)")

            return false
        end
    end

    #======================#
    # set up optimization solve
    #======================#
    adtype  = AutoZygote()
    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, p, st)

    @time optsol = solve(optprob, opt, ncycle(_loader, nepoch); callback = optcb)

    println(io, "#=======================#")
    @show optsol.retcode
    println(io, "#=======================#")

    mincfg[].p, mincfg[].st, mincfg[].opt_st
end

"""
early stopping based on mini-batch loss from test set
https://github.com/jeffheaton/app_deep_learning/blob/main/t81_558_class_03_4_early_stop.ipynb
"""
function make_minconfig(early_stopping, patience, l, p, st, opt_st)
    (; count = 0, early_stopping, patience, l, p, st, opt_st,)
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
        printstyled(io,
            "Improvement in loss found: $(l) < $(minconfig.l)\n",
            color = :green,
        )
        p = deepcopy(p)
        st = deepcopy(st)
        opt_st = deepcopy(opt_st)
        minconfig = (; minconfig..., count = 0, l, p, st, opt_st,)
    else
        printstyled(io,
            "No improvement in loss found in the last \
            $(minconfig.count) epochs. $(l) > $(minconfig.l)\n",
            color = :red,
        )
        @set! minconfig.count = minconfig.count + 1
    end

    if (minconfig.count >= minconfig.patience) & minconfig.early_stopping
        printstyled(io, "Early Stopping triggered after $(minconfig.count) \
            epochs of no improvement.\n",
            color = :red,
        )
        ifbreak = true
    end

    minconfig, ifbreak
end

#===============================================================#
function plot_training!(EPOCH, _LOSS, LOSS_, _MSE, MSE_, _MAE, MAE_)
    z = findall(iszero, EPOCH)

    # fix EPOCH to account for multiple training loops
    if length(z) > 1
            for i in 2:length(z)-1
            idx =  z[i]:z[i+1] - 1
            EPOCH[idx] .+= EPOCH[z[i] - 1]
        end
        EPOCH[z[end]:end] .+= EPOCH[z[end] - 1]
    end

    plt = plot(
        title = "Training Plot", yaxis = :log,
        xlabel = "Epochs", ylabel = "Loss",
        yticks = (@. 10.0^(-7:1)),
    )

    # (; ribbon = (lower, upper))
    plot!(plt, EPOCH, _LOSS, w = 2.0, s = :solid, c = :red , label = "LOSS (Train)")
    plot!(plt, EPOCH, LOSS_, w = 2.0, s = :solid, c = :blue, label = "LOSS (Test)")

    if !isempty(_MSE)
        plot!(plt, EPOCH, _MSE, w = 2.0, s = :dash, c = :magenta, label = "MSE (Train)")
        plot!(plt, EPOCH, MSE_, w = 2.0, s = :dash, c = :cyan   , label = "MSE (Test)")
    end

    if !isempty(_MAE)
        plot!(plt, EPOCH, _MAE, w = 2.0, s = :dot, c = :magenta, label = "MAE (Train)")
        plot!(plt, EPOCH, MAE_, w = 2.0, s = :dot, c = :cyan   , label = "MAE (Test)")
    end

    vline!(plt, EPOCH[z[2:end]], c = :black, w = 2.0, label = nothing)

    plt
end

#===============================================================#
#
