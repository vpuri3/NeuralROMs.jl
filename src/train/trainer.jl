#
#===============================================================#
# 1. make io async with @async or Threads.@spawn
#===============================================================#
abstract type AbstractTrainState end

@concrete mutable struct TrainState <: AbstractTrainState
	NN
	p
	st
	opt_st
	count
	loss_
end

function Adapt.adapt_structure(to, state::TrainState)
    p  = Adapt.adapt_structure(to, state.p )
    st = Adapt.adapt_structure(to, state.st)
    opt_st = Adapt.adapt_structure(to, state.opt_st)

	TrainState(state.NN, p, st, opt_st, state.count, state.loss_)
end

#===============================================================#
abstract type AbstractTrainer end

@concrete mutable struct Trainer <: AbstractTrainer
	state # (NN, p, st, opt_st, count, patience, early_stopping)
	data  # (_data, data_)
	batchsizes
	notestdata

	# optimizer
	opt
	opt_iter
	opt_args
	lossfun

	# misc
	io
	rng
	name
	STATS
	device
	verbose
	callbacks
end

function Trainer(
    NN::Lux.AbstractExplicitLayer,
    _data::NTuple{2, Any},
    data_::Union{Nothing,NTuple{2, Any}} = nothing;
# MODEL PARAMETER/ STATES
    p = nothing,
    st = nothing,
# DATA BATCH-SIZE
	_batchsize::Union{Int, Nothing}  = nothing,
	batchsize_::Union{Int, Nothing}  = nothing,
	__batchsize::Union{Int, Nothing} = nothing, # larger training batch size for BFGS, statistics
# OPTIMIZER and LOSS
	opt = Optimisers.Adam(),
	opt_st = nothing,
    lossfun = mse,
# OPTIMIZATION ARGUMENTS
    nepochs::Int = 100,
	schedule = nothing, # handle lr, batchsize scheduling via callbacks later
	early_stopping::Bool = true,
	return_last::Bool = false, # returns final state as opposed to minconfig
	patience_frac::Real = 0.2,
# MISC
    io::IO = stdout,
    rng::Random.AbstractRNG = Random.default_rng(),
	name::String = "model",
    device = Lux.gpu_device(),
	verbose::Bool = true,
	callbacks = nothing,
)
	#========= MODEL =========#
    p_new, st_new = Lux.setup(rng, NN)
    p  = isnothing(p)  ? p_new  : p
    st = isnothing(st) ? st_new : st

    p_ca = p |> ComponentArray
    p = isreal(p_ca) ? p_ca : p

	#========= DATA =========#

    notestdata = isnothing(data_)
    if notestdata
        data_ = _data
		verbose && println(io, "[train_model] No test dataset provided.")
    end

	if isnothing(_batchsize)
		_batchsize = max(1, numobs(_data) ÷ 100)
	end
	if isnothing(batchsize_)
		batchsize_ = numobs(data_)
	end
	if isnothing(__batchsize)
		__batchsize = numobs(_data)
	end

	batchsizes = (; _batchsize, batchsize_, __batchsize)

	#========= OPT =========#

	if opt isa Optimisers.AbstractRule
		if isnothing(opt_st)
			opt_st = Optimisers.setup(opt, p)
		else
			@set! opt_st.rule = opt
		end
	elseif opt isa Optim.AbstractOptimizer
		if opt isa Union{
			Optim.Newton, Optim.BFGS, Optim.LBFGS,
		}
			if __batchsize != numobs(_data)
				@warn "Got batchsize $(__batchsize) < $(numobs(_data)) (numobs). \
				Hessian-based optimizers such as Newton / BFGS / L-BFGS may be \
				unstable with mini-batching. Set batchsize to equal data-size, \
				or else the method may be unstable. If you want a stochastic \
				optimizer, try `Optimisers.jl`." maxlog = 1
			end
		else
			msg = "Optimizer of type $(typeof(opt)) is not supported."
			throw(ArgumentError(msg))
		end
		if !isnothing(opt_st)
			@warn "Optimization state of type $(opt_st) provided to \
			Optim.AbstractOptimizer." maxlog = 1
		end
		opt_st = nothing
	else
		msg = "Optimizer of type $(typeof(opt)) is not supported."
		throw(ArgumentError(msg))
	end

	patience = round(Int, nepochs * patience_frac)
	opt_args = (; nepochs, schedule, early_stopping, patience, return_last)
	opt_iter = (; epoch = [0], start_time=[0f0], epoch_time=[0f0], epoch_dt=[0f0],)
	
	#========= MISC =========#

	STATS = (;
		EPOCH = Int[]    , TIME  = Float32[],
		_LOSS = Float32[], LOSS_ = Float32[],
		_MSE  = Float32[], MSE_  = Float32[],
		_MAE  = Float32[], MAE_  = Float32[],

	)

	#==========================#
	data  = (; _data, data_)
	state = TrainState(NN, p, st, opt_st, 0, Inf32)

	Trainer(
		state, data, batchsizes, notestdata,
		opt, opt_iter, opt_args, lossfun,
		io, rng, name, STATS, device, verbose, callbacks
	)
end

#===============================================================#
function train!(trainer::Trainer)
	train!(trainer, trainer.state)
end

function train!(
	trainer::Trainer,
	state::TrainState,
)
	@unpack opt_args, opt_iter = trainer
	@unpack io, name, device, verbose = trainer

	if verbose
		println(io, "#============================================#")
		println(io, "Trainig $(name) with $(length(state.p)) parameters.")
		println(io, "Using optimizer $(string(trainer.opt))")
		println(io, "with args: $(opt_args)")
		println(io, "#============================================#")
	end

	state = state |> device
	loaders = make_dataloaders(trainer)

	verbose && printstatistics(trainer, state, loaders)
	evaluate(trainer, state, loaders)
	trigger_callback!(trainer, :START_TRAINING)

	opt_iter.start_time[] = time()
	verbose && println(io, "\nStarting Trainig Loop\n")

	state = train_loop!(trainer, state, loaders) # loop over epochs

	if opt_args.return_last
		trainer.state = state |> Lux.cpu_device()
		verbose && println(io, "Returning state at final iteration.")
	else
		state = trainer.state |> device
		verbose && println(io, "Returning state with minimum loss.")
	end

	verbose && printstatistics(trainer, state, loaders)
	verbose && evaluate(trainer, state, loaders; update_stats = false)
	trigger_callback!(trainer, :END_TRAINING)

	return trainer.state, trainer.STATS
end

#============================================================#

function trigger_callback!(trainer::Trainer, event::Symbol)
	nothing
end

function trigger_callback!(
	trainer::Trainer,
	state::TrainState,
	event::Symbol
)
	nothing
end

function train_loop!(trainer::Trainer, state::TrainState, loaders::NamedTuple)
	train_loop!(trainer, trainer.opt, state, loaders)
end

function printstatistics(
	trainer::Trainer,
	state::TrainState,
	loaders::NamedTuple
)
	printstatistics(trainer, state, loaders, trainer.io, trainer.verbose)
end

#============================================================#
function printstatistics(
	trainer::Trainer,
	state::TrainState,
	loaders::NamedTuple,
	io::IO, verbose::Bool
)
	@unpack NN, p, st = state
	@unpack notestdata = trainer
	@unpack __loader, loader_ = loaders

	_, _str = statistics(NN, p, st, __loader)

	if verbose
        println(io, "#----------------------#")
        println(io, "TRAIN DATA STATISTICS")
        println(io, "#----------------------#")
		println(io, _str)
        println(io, "#----------------------#")
	end

	if !notestdata
		_, str_ = statistics(NN, p, st, loader_)
		if verbose
			println(io, "#----------------------#")
			println(io, "TEST DATA STATISTICS")
			println(io, "#----------------------#")
			println(io, str_)
			println(io, "#----------------------#")
		end
	end

	return
end

function evaluate(
	trainer::Trainer,
	state::TrainState,
	loaders::NamedTuple;
	update_stats::Bool = true
)
	@unpack NN, p, st = state
	@unpack notestdata, lossfun = trainer
	@unpack opt_args, opt_iter = trainer
	@unpack io, verbose, STATS = trainer
	@unpack __loader, loader_ = loaders

	_l, _stats = fullbatch_metric(NN, p, st, __loader, lossfun)

	if trainer.notestdata
		l_, stats_ = _l, _stats
	else
		l_, stats_ = fullbatch_metric(NN, p, st, loader_, lossfun)
	end

	if verbose
		print(io, "Epoch [$(opt_iter.epoch[]) / $(opt_args.nepochs)]\t")

        print(io, "TRAIN LOSS: ")
        _lprint = round(_l; sigdigits=8)
        printstyled(io, _lprint; color = :magenta)
		
        print(io, " || TEST LOSS: ")
        lprint_ = round(l_; sigdigits = 8)
        printstyled(io, lprint_; color = :magenta)
		println(io)
	end

	if update_stats
		push!(STATS.EPOCH, opt_iter.epoch[])
		push!(STATS.TIME , opt_iter.epoch_dt[])
		push!(STATS._LOSS, _l)
		push!(STATS.LOSS_, l_)

		haskey(_stats, :mse) && push!(STATS._MSE, _stats.mse)
		haskey(stats_, :mse) && push!(STATS.MSE_, stats_.mse)
		haskey(_stats, :mae) && push!(STATS._MAE, _stats.mae)
		haskey(stats_, :mae) && push!(STATS.MAE_, stats_.mae)
	end

	return
end

#============================================================#
function make_dataloaders(trainer::Trainer)
	@unpack device, rng = trainer
	@unpack _data, data_ = trainer.data
	@unpack _batchsize, batchsize_, __batchsize = trainer.batchsizes

	_loader  = DataLoader(_data; batchsize = _batchsize , rng, shuffle = true)
	loader_  = DataLoader(data_; batchsize = batchsize_ , rng, shuffle = false)
	__loader = DataLoader(_data; batchsize = __batchsize, rng, shuffle = false)
	
    if device isa Lux.LuxCUDADevice
        _loader  = _loader  |> CuIterator
        loader_  = loader_  |> CuIterator
        __loader = __loader |> CuIterator
    end

	(; _loader, loader_, __loader)
end

#===============================================================#
function update_trainer_state!(trainer::Trainer, state::TrainState)
	@unpack opt_args, device, io, verbose = trainer
    ifbreak = false

	transfer = if device isa LuxDeviceUtils.AbstractLuxGPUDevice
		Lux.cpu_deivce()
	else
		deepcopy
	end

    if state.loss_ < trainer.state.loss_
		if verbose
			msg = "Improvement in loss found: $(state.loss_) < $(trainer.state.loss_)\n"
			printstyled(io, msg, color = :green)
		end
		state.count = 0
		trainer.state = state |> transfer
    else
		@set! state.count = state.count + 1
		if verbose
		    msg = "No improvement in loss found in the last $(state.count) epochs. $(state.loss_) > $(trainer.state.loss_)\n"
	        printstyled(io, msg, color = :red)
		end
    end

    if (state.count >= opt_args.patience) & opt_args.early_stopping
		if verbose
			msg = "Early Stopping triggered after $(state.count) epochs of no improvement.\n"
			printstyled(io, msg, color = :red)
		end
        ifbreak = true
    end

    state, ifbreak
end

#============================================================#
function save_trainer(
	trainer::Trainer,
	dir::String,
	name::String = "";
	metadata::NamedTuple = (;),
	verbose::Union{Bool, Nothing} = nothing,
)
	@unpack state, STATS = trainer

	name = isempty(name) ? trainer.name : name
	if isnothing(verbose)
		verbose = trainer.verbose
	end

	statsfile = joinpath(dir, "stats_$(name).txt")
	imagefile = joinpath(dir, "image_$(name).png")
	modelfile = joinpath(dir, "model_$(name).jld2")
	chkptfile = joinpath(dir, "chkpt_$(name).jld2")

	mkpath(dir)

	# STATISTICS
	touch(statsfile)
	statsio = open(statsfile, "a")
	printstatistics(trainer, statsio, true)
	close(statsio)
	verbose && @info "Saving statistics at $(statsfile)"

	# IMAGE
	plt = plot_training!(deepcopy(trainer.STATS)...)
    png(plt, imagefile)
    trainer.verbose && display(plt)
	verbose && @info "Saving plot at $(imagefile)"

	# MODEL
	model = state.NN, state.p, state.st
    jldsave(modelfile; model, metadata)
	verbose && @info "Saving model at $(modelfile)"

	# CHECKPOINT
	opt_st = opt_st |> Lux.cpu_device()
    jldsave(chkptfile; opt_st, STATS)
	verbose && @info "Saving model at $(chkptfile)"

	return
end

function plot_trainer(trainer::Trainer)
	plot_training!(deepcopy(trainer.STATS))
end

function load_trainer(
	trainer::Trainer,
	dir::String,
	name::String
)
end
#============================================================#
#
