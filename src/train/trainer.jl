#
#===============================================================#
abstract type AbstractTrainer end

@concrete mutable struct Trainer <: AbstractTrainer
	# model
	NN
	p
	st

	# data
	_loader
	__loader
	loader_
	notestdata

	# optimizer
	opt
	opt_st
	opt_iter
	opt_args
	lossfun

	# optim_loss
	# optim_cb

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
	return_minimum::Bool = true, # returns state with smallest loss value
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

    p, st = (p, st) |> device

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

    _loader  = DataLoader(_data; batchsize = _batchsize , rng, shuffle = true)
    loader_  = DataLoader(data_; batchsize = batchsize_ , rng, shuffle = false)
    __loader = DataLoader(_data; batchsize = __batchsize, rng, shuffle = false)

    if device isa Lux.LuxCUDADevice
        _loader  = _loader  |> CuIterator
        loader_  = loader_  |> CuIterator
        __loader = __loader |> CuIterator
    end

	#========= OPT =========#

	if opt isa Optimisers.AbstractRule
		if isnothing(opt_st)
			opt_st = Optimisers.setup(opt, p)
		else
			@set! opt_st.rule = opt
		end
	elseif opt isa Optim.AbstractOptimizer
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
		opt_st = nothing
	else
		msg = "Optimizer of type $(typeof(opt)) is not supported."
		throw(ArgumentError(msg))
	end

	opt = opt |> device
	opt_st = opt_st |> device

	patience = round(Int, nepochs * patience_frac)
	opt_args = (; nepochs, schedule, early_stopping, patience, return_minimum)
	opt_iter = (; epoch = [0], start_time=[0f0], epoch_time=[0f0], epoch_dt=[0f0],)
	
	#========= MISC =========#

    # EPOCH, TIME, _LOSS, LOSS_, _MSE, MSE_, _MAE, MAE_
	STATS = (;
		EPOCH = Int[]    , TIME  = Float32[],
		_LOSS = Float32[], LOSS_ = Float32[],
		_MSE  = Float32[], MSE_  = Float32[],
		_MAE  = Float32[], MAE_  = Float32[],

	)

	Trainer(
		NN, p, st,
		_loader, __loader, loader_, notestdata,
		opt, opt_st, opt_iter, opt_args, lossfun,
		io, rng, name, STATS, device, verbose, callbacks
	)
end
#===============================================================#

function train!(trainer::Trainer)
	# TODO - move to device here?

	@unpack opt_args, opt_iter = trainer
	@unpack io, name, device, verbose = trainer

	if verbose
		println(io, "#============================================#")
		println(io, "Trainig $(name) with $(length(trainer.p)) parameters.")
		println(io, "Using optimizer $(string(trainer.opt))")
		println(io, "with args: $(opt_args)")
		println(io, "#============================================#")
	end

	verbose && statistics(trainer)
	evaluate(trainer)
	minconfig = make_minconfig(trainer)

	if verbose
		println(io)
		println(io, "Starting Trainig Loop")
		println(io)
	end

	opt_iter.start_time[] = time()

	# train loop
	for _ in 1:opt_args.nepochs
		opt_iter.epoch[] += 1
		opt_iter.epoch_time[] = time() - opt_iter.start_time[]

		doepoch!(trainer)
		evaluate(trainer)

		opt_iter.epoch_dt[] = time() - opt_iter.epoch_time[] - opt_iter.start_time[]

		# update minconfig
		minconfig, ifbreak = update_minconfig(trainer, minconfig)
		ifbreak && break
	end

	if opt_args.return_minimum
		if verbose
			println(io, "Returning minimum value from training run.")
		end
		trainer.p  = minconfig.p          |> device
		trainer.st = minconfig.st         |> device
		trainer.opt_st = minconfig.opt_st |> device
	end

	if verbose
		statistics(trainer)
		evaluate(trainer; update_stats=false)
	end

	NN = trainer.NN
	p  = trainer.p  |> Lux.cpu_device()
	st = trainer.st |> Lux.cpu_device()

	return (NN, p, st), trainer.STATS
end

#============================================================#
function statistics(trainer::Trainer)
	@unpack NN, p, st = trainer
	@unpack notestdata, __loader, loader_ = trainer
	@unpack io, verbose = trainer

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

function evaluate(trainer::Trainer; update_stats::Bool = true)
	@unpack NN, p, st, lossfun = trainer
	@unpack notestdata, __loader, loader_ = trainer
	@unpack opt_args, opt_iter = trainer
	@unpack io, verbose, STATS = trainer

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

function doepoch!(trainer::Trainer)
	doepoch!(trainer, trainer.opt)
end

function doepoch!(trainer::Trainer, ::Optimisers.AbstractRule)

	@unpack _loader, opt_st, lossfun = trainer
	@unpack opt, opt_args, opt_iter, io = trainer

	if trainer.verbose
		prog_meter = ProgressMeter.Progress(
			length(_loader);
			barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
			desc = "Epoch [$(opt_iter.epoch[]) / $(opt_args.nepochs)] LR: $(round(opt.eta; sigdigits=3))",
			dt = 1e-4, barlen = 25, color = :normal, showspeed = true, output = io,
		)
	end

	trainer.st = Lux.trainmode(trainer.st)

	for batch in _loader
		loss = Loss(trainer.NN, trainer.st, batch, lossfun)
		l, st, stats, g = grad(loss, trainer.p)
		opt_st, p = Optimisers.update!(opt_st, trainer.p, g)

		if isnan(l)
			throw(ErrorException("Loss in NaN"))
		end

		trainer.p = p
		trainer.st = st
		trainer.opt_st = opt_st

		if trainer.verbose
			showvalues = Any[(:LOSS, round(l; sigdigits = 8)),]
			!isempty(stats) && push!(showvalues, (:INFO, stats))
			ProgressMeter.next!(prog_meter; showvalues, valuecolor = :magenta)
		end
	end

	if trainer.verbose
		ProgressMeter.finish!(prog_meter)
	end

	return
end

function doepoch!(trainer::Trainer, opt::Optim.AbstractOptimizer)
	return
end

#============================================================#
# 1. make copies here as Optimisers.update! is in place
# 2. move minconfig to cpu to save GPU memory
# 3. make io async (move to separate thread?)
#============================================================#
function save_checkpoint(trainer::Trainer, filename::String)
	# including optimzier state if trainer.verbose
		@info "Saving checkpoint $(filename)"
end

function save_model(trainer::Trainer, filename::String)
	if trainer.verbose
		@info "Saving model at $(filename)"
	end
end

function load_checkpoint(trainer::Trainer, filename::String)
	if trainer.verbose
		@info "Loading checkpoint $(filename)"
	end
end

#============================================================#

function make_minconfig(trainer::Trainer)
	@unpack p, st, opt_st = trainer
	@unpack early_stopping, patience = trainer.opt_args
	@unpack device = trainer

	transfer = if device isa LuxDeviceUtils.AbstractLuxGPUDevice
		LuxDeviceUtils.cpu_device()
	else
		deepcopy
	end

	l  = trainer.STATS.LOSS_[end]
	p  = trainer.p  |> transfer
	st = trainer.st |> transfer
	opt_st = trainer.opt_st |> transfer

    (; count = 0, early_stopping, patience, l, p, st, opt_st,)
end

function update_minconfig(trainer::Trainer, minconfig)
	@unpack p, st, opt_st = trainer
	@unpack io, verbose = trainer
	l = trainer.STATS.LOSS_[end]

	update_minconfig(minconfig, l, p, st, opt_st; io, verbose)
end
#============================================================#
#
