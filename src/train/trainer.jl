#
#===============================================================#
# 1. make io async with @async or Threads.@spawn
# 2. call evaluate, update_trainer_state! every so many steps
# 3. asssert that evaluate has been called before update_trainer_state!
# 4. make data transfer in update_trainer_state! async
#===============================================================#
abstract type AbstractTrainState end

@concrete mutable struct TrainState <: AbstractTrainState
	NN
	p
	st
	opt_st
end

# discussion: https://github.com/FluxML/Optimisers.jl/pull/180 is merged
function (dev::MLDataDevices.CPUDevice)(state::TrainState)
	TrainState(state.NN, dev(state.p), dev(state.st), dev(state.opt_st))
end

for DEV in MLDataDevices.GPU_DEVICES
	function (dev::DEV)(state::TrainState)
		TrainState(state.NN, dev(state.p), dev(state.st), dev(state.opt_st))
	end
end

#===============================================================#
abstract type AbstractTrainer end

@concrete mutable struct Trainer <: AbstractTrainer
	state     # (NN, p, st, opt_st)
	data      # (_data, data_)
	data_args # (notestdata, batchsizes....)

	# optimizer
	opt
	opt_iter
	opt_args
	lossfun

	# misc
	rng
	STATS
	device
	io_args
	callbacks
end

function Trainer(
    NN::AbstractLuxLayer,
    _data::NTuple{2, Any},
    data_::Union{Nothing,NTuple{2, Any}} = nothing;
# MODEL PARAMETER/ STATES
    p = nothing,
    st = nothing,
	make_ca = false, # make p a component array
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
	early_stopping::Bool = true,
	fullbatch_freq::Int = 1,   # evaluate full-batch loss every K epochs. (0 => never!)
	return_last::Bool = false, # returns final state as opposed to minconfig
	patience::Int = round(Int, nepochs/5),
# MISC
    rng::Random.AbstractRNG = Random.default_rng(),
    device = gpu_device(),
# CALLBACKS
	cb_epoch = DEFAULT_CB_EPOCH, # (trainer, state, epoch) -> state
	cb_batch = DEFAULT_CB_BATCH, # (trainer, state, batch, loss, grad, epoch, ibatch) -> state
# IO-ARGS
    io::IO = stdout,
	name::String = "model",
	verbose::Bool = true,
	print_config::Bool = true,
	print_stats::Bool = true,
	print_batch::Bool = true,
	print_epoch::Bool = true,
)

	#========= MODEL =========#
    p_new, st_new = Lux.setup(rng, NN)
    p  = isnothing(p)  ? p_new  : p
    st = isnothing(st) ? st_new : st

	if make_ca
		p = ComponentArray(p)
	end

	#========= DATA =========#

    notestdata = isnothing(data_)
    if notestdata
        data_ = _data
		if verbose & print_config
			println(io, "No test dataset provided.")
		end
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

	data_args = (; notestdata, _batchsize, batchsize_, __batchsize)

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

	opt_args = (; nepochs, early_stopping, patience, fullbatch_freq, return_last)
	opt_iter = (;
		epoch = [0], start_time=[0f0], epoch_time=[0f0], epoch_dt=[0f0],
		count = [0], loss_mincfg = [Inf32], # early stopping
	)

	#========= MISC =========#

	io_args = (;
		io, name, verbose, print_config, print_stats, print_batch, print_epoch,
	)

	STATS = (;
		EPOCH = Int[]    , TIME  = Float32[],
		_LOSS = Float32[], LOSS_ = Float32[],
		_MSE  = Float32[], MSE_  = Float32[],
		_MAE  = Float32[], MAE_  = Float32[],
	)

	callbacks = (; cb_epoch, cb_batch,)

	#==========================#
	data  = (; _data, data_)
	state = TrainState(NN, p, st, opt_st)

	Trainer(
		state, data, data_args,
		opt, opt_iter, opt_args, lossfun,
		rng, STATS, device, io_args, callbacks
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
	@unpack opt_args, opt_iter, device, io_args = trainer
	@unpack io, verbose, print_config, print_stats = io_args
	@unpack fullbatch_freq = opt_args

	if verbose & print_config
		println(io, "#============================================#")
		println(io, "Trainig $(io_args.name) with $(length(state.p)) parameters.")
		println(io, "Optimizer: $(string(trainer.opt)) with args $(opt_args)")
		println(io, "Device: $(device)")
		println(io, "#============================================#")
	end

	state = state |> device
	loaders = make_dataloaders(trainer) |> device

	if verbose & print_stats
		printstatistics(trainer, state, loaders)
	end
	if !iszero(fullbatch_freq)
		evaluate(trainer, state, loaders)
	end

	opt_iter.start_time[] = time()
	state = train_loop!(trainer, state, loaders) # loop over epochs

	if opt_args.return_last
		trainer.state = state |> cpu_device()
		if verbose & print_config
			println(io, "Returning state at final iteration.")
		end
	else
		state = trainer.state |> device
		if verbose & print_config
			println(io, "Returning state with minimum loss.")
		end
	end

	if verbose
		print_stats && printstatistics(trainer, state, loaders)
		!iszero(fullbatch_freq) && evaluate(trainer, state, loaders; update_stats = false)
	end

	return trainer.state, trainer.STATS
end

#============================================================#
function train_loop!(trainer::Trainer, state::TrainState, loaders::NamedTuple)
	train_loop!(trainer, trainer.opt, state, loaders)
end

#============================================================#
DEFAULT_CB_EPOCH = (trainer, state, epoch) -> (state, false)
DEFAULT_CB_BATCH = (trainer, state, batch, loss, grad, epoch, ibatch) -> (state, false)

#============================================================#
function printstatistics(trainer::Trainer, state::TrainState, loaders::NamedTuple)
	printstatistics(trainer, state, loaders, trainer.io_args.io)
end

function printstatistics(
	trainer::Trainer,
	state::TrainState,
	loaders::NamedTuple,
	io::IO,
)
	@unpack NN, p, st = state
	@unpack notestdata = trainer.data_args
	@unpack __loader, loader_ = loaders

	_, _str = statistics(NN, p, st, __loader)

	println(io, "#----------------------#")
	println(io, "TRAIN DATA STATISTICS")
	println(io, "#----------------------#")
	println(io, _str)
	println(io, "#----------------------#")

	if !notestdata
		_, str_ = statistics(NN, p, st, loader_)
		println(io, "#----------------------#")
		println(io, "TEST DATA STATISTICS")
		println(io, "#----------------------#")
		println(io, str_)
		println(io, "#----------------------#")
	end

	return
end

#============================================================#
function evaluate(
	trainer::Trainer,
	state::TrainState,
	loaders::NamedTuple;
	update_stats::Bool = true
)
	@unpack NN, p, st = state
	@unpack __loader, loader_ = loaders
	@unpack data_args, opt_args, opt_iter, lossfun, STATS, io_args = trainer
	@unpack io, verbose, print_epoch = io_args

	_l, _stats = fullbatch_metric(NN, p, st, __loader, lossfun)

	if data_args.notestdata
		l_, stats_ = _l, _stats
	else
		l_, stats_ = fullbatch_metric(NN, p, st, loader_, lossfun)
	end

	if verbose & print_epoch
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
	@unpack rng, data, data_args = trainer
	@unpack _data, data_ = data
	@unpack _batchsize, batchsize_, __batchsize = data_args

	_loader  = DataLoader(_data; batchsize = _batchsize , rng, shuffle = true)
	loader_  = DataLoader(data_; batchsize = batchsize_ , rng, shuffle = false)
	__loader = DataLoader(_data; batchsize = __batchsize, rng, shuffle = false)
	
	(; _loader, loader_, __loader)
end

#===============================================================#
function update_trainer_state!(trainer::Trainer, state::TrainState)
	@unpack device, io_args = trainer
	@unpack io, verbose, print_epoch = io_args
	@unpack opt_args, opt_iter, STATS = trainer
    ifbreak = false

	# make this transfer async
	transfer = if device isa AbstractGPUDevice
		cpu_device()
	else
		deepcopy
	end

	# LATEST EPOCH LOSS: STATS.LOSS_[end]
	# MIN-CONFIG   LOSS: opt_iter.loss_mincfg[]

	if STATS.LOSS_[end] < opt_iter.loss_mincfg[]
		if verbose & print_epoch
			msg = "Improvement in loss found: $(STATS.LOSS_[end]) < $(opt_iter.loss_mincfg[])\n"
			printstyled(io, msg, color = :green)
		end
		opt_iter.count[] = 0
		trainer.state = state |> transfer
		opt_iter.loss_mincfg[] = STATS.LOSS_[end] # new
    else
		opt_iter.count[] += 1
		if verbose & print_epoch
			msg = "No improvement in loss found in the last $(opt_iter.count[]) epochs. $(STATS.LOSS_[end]) > $(opt_iter.loss_mincfg[])\n"
	        printstyled(io, msg, color = :red)
		end
    end

	if (opt_iter.count[] >= opt_args.patience) & opt_args.early_stopping
		if verbose & print_epoch
			msg = "Early Stopping triggered after $(opt_iter.count[]) epochs of no improvement.\n"
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
	opt_st = opt_st |> cpu_device()
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
