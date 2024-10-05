#
#============================================================#
# Optimisers.jl
#============================================================#
@concrete struct Loss
    NN
    st
    batch
    lossfun
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
function train_loop!(
	trainer::Trainer,
	opt::Optimisers.AbstractRule,
	state::TrainState,
	loaders::NamedTuple,
)
	@unpack opt_args, opt_iter = trainer
	@unpack fullbatch_freq = opt_args
	ifbreak = false

	# epoch loop

	while opt_iter.epoch[] < opt_args.nepochs
		opt_iter.epoch[] += 1
		opt_iter.epoch_time[] = time() - opt_iter.start_time[]

		state = doepoch(trainer, state, opt, loaders._loader)

		opt_iter.epoch_dt[] = time() - opt_iter.epoch_time[] - opt_iter.start_time[]

		trigger_callback!(trainer, :EPOCH_END)
		if !iszero(fullbatch_freq)
			if (opt_iter.epoch[] % fullbatch_freq) == 0
				evaluate(trainer, state, loaders)
				state, ifbreak = update_trainer_state!(trainer, state)
			end
		end
		ifbreak && break
	end

	return state
end

function doepoch(
	trainer::Trainer,
	state::TrainState,
	opt::Optimisers.AbstractRule,
	_loader,
)
	@unpack opt_args, opt_iter, io_args = trainer
	@unpack io, verbose, print_batch = io_args

	show_batch = (length(_loader) > 1) & verbose & print_batch

	if show_batch
		prog_meter = ProgressMeter.Progress(
			length(_loader);
			barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
			desc = "Epoch [$(opt_iter.epoch[]) / $(opt_args.nepochs)] LR: $(round(opt.eta; sigdigits=3))",
			dt = 1e-4, barlen = 25, color = :normal, showspeed = true, output = io,
		)
	end

	# state.st = Lux.trainmode(state.st)
	@set! state.st = Lux.trainmode(state.st)

	for (k, batch) in enumerate(_loader)
		state, (l, stats) = step(trainer, state, opt, batch)
		trigger_callback!(trainer, :BATCH_END)

		if show_batch
			showvalues = Any[(:LOSS, round(l; sigdigits = 8)),]
			!isempty(stats) && push!(showvalues, (:INFO, stats))
			ProgressMeter.next!(prog_meter; showvalues, valuecolor = :magenta)
		end
	end

	if show_batch
		ProgressMeter.finish!(prog_meter)
	end

	return state
end

function step(
	trainer::Trainer,
	state::TrainState,
	::Optimisers.AbstractRule,
	batch,
)
	@unpack lossfun = trainer
	@unpack NN, p, st, opt_st = state

	loss = Loss(NN, st, batch, lossfun)
	l, st, stats, g = grad(loss, p)
	opt_st, p = Optimisers.update!(opt_st, p, g)

	if isnan(l)
		throw(ErrorException("Loss in NaN"))
	end

	state = TrainState(NN, p, st, opt_st)
	return state, (l, stats)
end

#============================================================#
# OptimzationOptimJL.jl
#============================================================#
function train_loop!(
	trainer::Trainer,
	opt::Optim.AbstractOptimizer,
	state::TrainState,
	loaders::NamedTuple,
)
	@unpack __loader = loaders
	@unpack lossfun, opt_args, opt_iter, io_args, device = trainer
	@unpack io, verbose, print_epoch, print_config = io_args
	@unpack fullbatch_freq = opt_args

	ifbreak = false

	batch = if __loader isa MLDataDevices.DeviceIterator
		__loader.iterator.data |> device
	else
		__loader.data
	end

	# https://github.com/SciML/Optimization.jl/issues/839

    function optloss(optx, optp)
		l, st, stats = lossfun(state.NN, optx, state.st, batch)
		@set! state.st = st
		l
    end

	function optcb(optx, l)
		@set! state.p = optx.u

		# if !isempty(stats) & verbose & print_epoch
		# 	println(io, stats)
		# end

		if !iszero(fullbatch_freq)
			if (opt_iter.epoch[] % fullbatch_freq) == 0
				evaluate(trainer, state, loaders)
				state, ifbreak = update_trainer_state!(trainer, state)
			end
		end

		opt_iter.epoch[] += 1
		opt_iter.epoch_dt[] = time() - opt_iter.epoch_time[] - opt_iter.start_time[]
		opt_iter.epoch_time[] = time() - opt_iter.start_time[]
		trigger_callback!(trainer, :EPOCH_END)

		return ifbreak
	end

    #======================#
    # set up optimization and solve
    #======================#
    adtype  = AutoZygote()
    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, state.p)

    optsol = solve(optprob, trainer.opt;
		callback = optcb, maxiters = opt_args.nepochs,
	)

	if verbose & print_config
		@show optsol.retcode
	end

	return state
end
#============================================================#
# Gradient API?
#============================================================#
#
