#
# function train_model(
#     NN::AbstractLuxLayer,
#     _data::NTuple{2, Any},
#     data_::Union{Nothing,NTuple{2, Any}} = nothing;
# #
#     rng::Random.AbstractRNG = Random.default_rng(),
# #
#     _batchsize::Union{Int, NTuple{M, Int}} = 32,
#     batchsize_::Int = numobs(_data),
#     __batchsize::Int = batchsize_, # > batchsize for BFGS, callback
# #
#     opts::NTuple{M, Any} = (Optimisers.Adam(1f-3),),
#     nepochs::NTuple{M, Int} = (100,),
#     schedules::Union{Nothing, NTuple{M, ParameterSchedulers.AbstractSchedule}} = nothing,
# #
#     early_stoppings::Union{Bool, NTuple{M, Bool}, Nothing} = nothing,
#     patience_fracs::Union{Real, NTuple{M, Any}, Nothing} = nothing,
#     weight_decays::Union{Real, NTuple{M, Real}} = 0f0,
# #
#     dir::String = "dump",
#     name::String = "model",
#     metadata::NamedTuple = (;),
#     io::IO = stdout,
# #
#     p = nothing,
#     st = nothing,
#     lossfun = mse,
#     device = gpu_device(),
# #
#     cb_epoch = nothing, # (NN, p, st) -> nothing
# ) where{M}
#
#     if early_stoppings isa Union{Bool, Nothing}
#         early_stoppings = fill(early_stoppings, M)
#     end
#
#     if patience_fracs isa Union{Bool, Nothing}
#         patience_fracs = fill(patience_fracs, M)
#     end
#
#     if weight_decays isa Real
#         weight_decays = fill(weight_decays, M)
#     end
#
#     if _batchsize isa Integer
#         _batchsize = fill(_batchsize, M)
#     end
#
#     time0 = time()
# 	opt_st = nothing
# 	STATS = nothing
#
# 	for iopt in 1:M
# 		# TODO: weight decay
#
# 		early_stopping = early_stoppings[iopt]
# 		if isnothing(early_stopping)
# 			early_stopping = true
# 		end
#
# 		patience_frac = patience_fracs[iopt]
# 		if isnothing(patience_frac)
# 			patience_frac = 0.2
# 		end
#
# 		verbose = true
# 		return_minimum = true
#
# 		opt = opts[iopt]
# 		if opt isa Optim.AbstractOptimizer
# 			opt_st = nothing
# 		end
#
#         weight_decay = weight_decays[iopt]
#         if !iszero(weight_decay) & (opt isa OptimiserChain)
#             isWD = Base.Fix2(isa, Union{WeightDecay, DecoderWeightDecay, IdxWeightDecay})
#             iWD = findall(isWD, opt.opts)
#    
#             if isempty(iWD)
#                 @error "weight_decay = $weight_decay, but no WeightDecay optimizer in $opt"
#             else
#                 @assert length(iWD) == 1 """More than one WeightDecay() found
#                     in optimiser chain $opt."""
#    
#                 iWD = only(iWD)
#                 @set! opt.opts[iWD].lambda = weight_decay
#             end
#         end
#
# 		trainer = Trainer(
# 			NN, _data, data_; p, st,
# 			_batchsize = _batchsize[iopt], batchsize_, __batchsize,
# 			opt, opt_st, lossfun,
# 			nepochs = nepochs[iopt], schedule,
# 			early_stopping, return_minimum, patience_frac,
# 			io, rng, name, verbose, device
# 		)
# 		train!(trainer)
#
# 		# update essentials
# 		p, st, opt_st = trainer.p, trainer.st, trainer.opt_st
#
# 		# update stats
# 		if isnothing(STATS)
# 			STATS = deepcopy(trainer.STATS)
# 		else
# 			STATS = NamedTuple{keys(STATS)}(
# 				vcat(S1, S2) for (S1, S2) in zip(STATS, trainer.STATS)
# 			)
# 		end
# 		trainer.STATS = STATS # hack
#
# 		# save model
# 		count = lpad(iopt, 2, "0")
# 		name_ = "$(count)"
# 		save_trainer(trainer, dir, name_; metadata)
# 	end
#
#     p, st = cpu_device()((p, st))
#     (NN, p, st), STATS
# end
#===============================================================#

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
    NN::AbstractLuxLayer,
    _data::NTuple{2, Any},
    data_::Union{Nothing,NTuple{2, Any}} = nothing;
#
    rng::Random.AbstractRNG = Random.default_rng(),
#
    _batchsize::Union{Int, NTuple{M, Int}} = 32,
    batchsize_::Int = numobs(_data),
    __batchsize::Int = batchsize_, # > batchsize for BFGS, callback
#
    opts::NTuple{M, Any} = (Optimisers.Adam(1f-3),),
    nepochs::NTuple{M, Int} = (100,),
    schedules::Union{Nothing, NTuple{M, ParameterSchedulers.AbstractSchedule}} = nothing,
#
	fullbatch_freq::Int = 1, # evaluate full-batch loss every K epochs. (0 => never!)
    early_stoppings::Union{Bool, NTuple{M, Bool}, Nothing} = nothing,
    patience_fracs::Union{Real, NTuple{M, Any}, Nothing} = nothing,
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
    device = gpu_device(),
#
    cb_epoch = nothing, # (NN, p, st) -> nothing
) where{M}

    notestdata = isnothing(data_)

    if notestdata
        data_ = _data
        println("[train_model] No test dataset provided.")
    end

    # create loader, batchsize schedule

    if isa(_batchsize, Integer)
        _batchsize0 = _batchsize
    else
        _batchsize0 = first(_batchsize)
    end

    _loader  = DataLoader(_data; batchsize = _batchsize0, rng, shuffle = true)
    loader_  = DataLoader(data_; batchsize = batchsize_ , rng, shuffle = true)
    __loader = DataLoader(_data; batchsize = __batchsize, rng, shuffle = true)

    if device isa AbstractGPUDevice
		_loader  = DeviceIterator(device, _loader )
		loader_  = DeviceIterator(device, loader_ )
		__loader = DeviceIterator(device, __loader)
    end

    # callback functions

    # EPOCH, TIME, _LOSS, LOSS_, _MSE, MSE_, _MAE, MAE_
	STATS = Int[], Float32[], Float32[], Float32[], Float32[], Float32[], Float32[], Float32[]

    cb = makecallback(NN, __loader, loader_, lossfun; io, STATS, cb_epoch, notestdata)
    cb_stats = makecallback(NN, __loader, loader_, lossfun; io, stats = true, notestdata)

    # parameters
    _p, _st = Lux.setup(rng, NN)

    p  = isnothing(p)  ? _p  : p
    st = isnothing(st) ? _st : st

    _p = p |> ComponentArray
    p = isreal(_p) ? _p : p

    p, st = (p, st) |> device

    println(io, "#======================#")
    println(io, "Starting Trainig Loop")
    println(io, "Model size: $(length(p)) parameters")
    println(io, "#======================#")

    # print stats
    cb_stats(p, st)

    st = Lux.trainmode(st)
    opt_st = nothing

    if early_stoppings isa Union{Bool, Nothing}
        early_stoppings = fill(early_stoppings, M)
    end

    if patience_fracs isa Union{Bool, Nothing}
        patience_fracs = fill(patience_fracs, M)
    end

    if weight_decays isa Real
        weight_decays = fill(weight_decays, M)
    end

    if _batchsize isa Integer
        _batchsize = fill(_batchsize, M)
    end
    @assert all(x -> x ≤ batchsize_,_batchsize)

    time0 = time()

    for iopt in 1:M
        time1 = time()

        opt = opts[iopt] |> device
        nepoch = nepochs[iopt]
        schedule = isnothing(schedules) ? nothing : schedules[iopt]

        early_stopping = early_stoppings[iopt]
        patience_frac = patience_fracs[iopt]
        patience = isnothing(patience_frac) ? nothing : round(Int, patience_frac * nepoch)

        if _loader isa DeviceIterator
			@set! _loader.iterator.batchsize = _batchsize[iopt]
        else
            @set! _loader.batchsize = _batchsize[iopt]
        end

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
        println(io, "Optimizer: $opt")
        println(io, "Nepochs: $nepoch")
        println(io, "Schedule: $schedule")

        println(io, "Early-stopping: $early_stopping")
        println(io, "Patience frac: $patience_frac")
        println(io, "Patience: $patience")
        println(io, "Batch-size: $(_batchsize[iopt])")

        println(io, "#======================#")

        args = (opt, NN, p, st, nepoch, _loader, loader_, __loader)
        kwargs = (;lossfun, opt_st, cb, io, fullbatch_freq, early_stopping, patience, schedule)

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

    # TODO - schedule batchsize
    # TODO - output a train.log file with timings.

    println(io, "#======================#")
    println(io, "Optimization done")
    println(io, "#======================#")

    p, st = cpu_device()((p, st))

    (NN, p, st), STATS
end

#===============================================================#
"""
$SIGNATURES
"""
function makecallback(
    NN::AbstractLuxLayer,
    _loader::Union{DeviceIterator, MLUtils.DataLoader},
    loader_::Union{DeviceIterator, MLUtils.DataLoader},
    lossfun;
    STATS::Union{Nothing, NTuple{8, Vector}} = nothing,
    stats::Bool = false,
    io::IO = stdout,
    cb_epoch = nothing, # (NN, p, st) -> nothing
    notestdata::Bool = false,
)
    _loss = (p, st) -> fullbatch_metric(NN, p, st, _loader, lossfun)
    loss_ = (p, st) -> fullbatch_metric(NN, p, st, loader_, lossfun)

    kwargs = (; _loss, loss_, notestdata)

    if stats
        _printstatistics = (p, st) -> statistics(NN, p, st, _loader)
        printstatistics_ = (p, st) -> statistics(NN, p, st, loader_)

        kwargs = (;kwargs..., _printstatistics, printstatistics_)
    end

    if !isnothing(STATS)

        if lossfun === mse
            STATS = (STATS[1:4]..., STATS[3:4]..., STATS[7], STATS[8])
        end

        if lossfun === mae
            STATS = (STATS[1:6]...,  STATS[3:4]...)
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
    EPOCH, TIME, _LOSS, LOSS_, _MSE, MSE_, _MAE, MAE_ = isnothing(STATS) ? ntuple(Returns(nothing), 8) : STATS

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
		_stats, _str = _printstatistics(p, st)
        println(io, "#======================#")
        println(io, "TRAIN STATS")
		println(io, _str)
        println(io, "#======================#")
    end

    if !isnothing(printstatistics_) 
		stats_, str_ = printstatistics_(p, st)
        println(io, "#======================#")
        println(io, "TEST  STATS")
		println(io, str_)
        println(io, "#======================#")
    end

    # terminate optimization if
    ifbreak = false

    _l, l_, ifbreak
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
    NN::AbstractLuxLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    nepoch::Integer,
    _loader::Union{DeviceIterator, MLUtils.DataLoader},
    loader_::Union{DeviceIterator, MLUtils.DataLoader},
    __loader::Union{DeviceIterator, MLUtils.DataLoader} = _loader;
    lossfun = mse,
    opt_st = nothing,
    cb = nothing,
    io::Union{Nothing, IO} = stdout,
	fullbatch_freq::Int = 1,
    early_stopping::Union{Bool, Nothing} = nothing,
    patience::Union{Int, Nothing} = nothing,
    schedule::Union{Nothing, ParameterSchedulers.AbstractSchedule} = nothing,
    kwargs...,
)
    # ensure testing mode
    st = Lux.trainmode(st)

	if isnothing(schedule)
		schedule = ParameterSchedulers.Step(Float32(opt.eta), 1f0, Inf32)
	end

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
		if !iszero(fullbatch_freq)
			if ((epoch % fullbatch_freq) == 0) | (epoch == nepoch)
				_, l_, ifbreak_cb = cb(p, st; epoch, nepoch, io)
				minconfig, ifbreak_mc = update_minconfig(minconfig, l_, p, st, opt_st; io, fullbatch_freq)
				(ifbreak_cb | ifbreak_mc) && break
			end
		else
			@set! minconfig.p = p
			@set! minconfig.st = st
			@set! minconfig.out_st = opt_st
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
    NN::AbstractLuxLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    nepoch::Integer,
    _loader::Union{DeviceIterator, MLUtils.DataLoader},
    loader_::Union{DeviceIterator, MLUtils.DataLoader},
    __loader::Union{DeviceIterator, MLUtils.DataLoader} = _loader;
    lossfun = mse,
    opt_st = nothing,
    cb = nothing,
    io::Union{Nothing, IO} = stdout,
	fullbatch_freq::Int = 1,
    early_stopping::Union{Bool, Nothing} = nothing,
    patience::Union{Int, Nothing} = nothing,
    kwargs...
)
    if opt isa Union{
        Optim.Newton,
        Optim.BFGS,
        Optim.LBFGS,
    }

        if length(__loader) != 1
            @warn " Data loader has $(length(__loader)) minibatches. \
            Hessian-based optimizers such as Newton / BFGS / L-BFGS may be \
            unstable with mini-batching. Set batchsize to equal data-size, \
            or else the method may be unstable. If you want a stochastic \
            optimizer, try `Optimisers.jl`."
        end

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

	batch = if __loader isa DeviceIterator
		# Adapt.adapt(__loader, __loader.batches.data)
		__loader.iterator.data |> __loader.dev
	else
		__loader.data
	end
	# batch = first(__loader)

    function optloss(optx, optp)
        lossfun(NN, optx, state[], batch)
    end

    function optcb(optx, l, st, stats)
        count[] += 1
        nextepoch = iszero(count[] % num_batches)

        ll = round(l; sigdigits = 8)
        state[] = st

        if nextepoch
            println(io, "Epoch [$(epoch[]) / $(nepoch)]\tBatch Loss: $(ll)")
            # println(io, "Iter: $(optx.iter), Objective: $(optx.objective)")

            println(io, "#=======================#")

            _, l_ = cb(optx.u, st; epoch = epoch[], nepoch, io)
            minconfig, ifbreak = update_minconfig(mincfg[], l_, optx.u, st, opt_st; io, fullbatch_freq)
            mincfg[] = minconfig
            epoch[] += 1

            return ifbreak
        else
            println(io, "Epoch [$(epoch[]) / $(nepoch)]\tBatch Loss: $(ll)")

            return false
        end
    end

    #======================#
    # set up optimization and solve
    #======================#
    adtype  = AutoZygote()
    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, p)

    @time optsol = solve(optprob, opt; callback = optcb, maxiters = nepoch)

    println(io, "#=======================#")
    @show optsol.retcode
    println(io, "#=======================#")

    mincfg[].p, mincfg[].st, mincfg[].opt_st
end

#===============================================================#
function savemodel!( # modifies STATS
    NN::AbstractLuxLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    metadata,
    STATS::NTuple{8, Vector},
    cb::Function,
    dir::String,
    name::String,
    count::Integer,
)
    # save statistics
    statsfile = joinpath(dir, "statistics.txt")

    mkpath(dir)
    touch(statsfile)

    count = lpad(count, 2, "0")
    statsio = open(joinpath(dir, "statistics.txt"), "a")
    println(statsio, "CHECKPOINT $count")
    cb(p, st; io = statsio)
    close(statsio)

    # transfer model to host device
	p, st = (p, st) |> cpu_device()
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
# EARLY STOPPING
#===============================================================#
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
    io::IO = stdout,
	verbose::Bool = true,
	fullbatch_freq::Int = 1,
)
    ifbreak = false

    if l < minconfig.l
		if verbose
			msg = "Improvement in loss found: $(l) < $(minconfig.l)\n"
			printstyled(io, msg, color = :green)
		end

        p = deepcopy(p)
        st = deepcopy(st)
        opt_st = deepcopy(opt_st)
        minconfig = (; minconfig..., count = 0, l, p, st, opt_st,)
    else
		if verbose
		    msg = "No improvement in loss found in the last $(minconfig.count) epochs. $(l) > $(minconfig.l)\n"
	        printstyled(io, msg, color = :red)
		end
        @set! minconfig.count = minconfig.count + fullbatch_freq
    end

    if (minconfig.count >= minconfig.patience) & minconfig.early_stopping
		if verbose
			msg = "Early Stopping triggered after $(minconfig.count) epochs of no improvement.\n"
			printstyled(io, msg, color = :red)
		end
        ifbreak = true
    end

    minconfig, ifbreak
end
#===============================================================#
#
