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
	@unpack opt_args, opt_iter, callbacks = trainer
	@unpack fullbatch_freq, nepochs = opt_args
	@unpack verbose = trainer.io_args

	# epoch loop

	while opt_iter.epoch[] < nepochs
		# update epoch
		opt_iter.epoch[] += 1

		# do epoch
		state, ifbreak = doepoch(trainer, state, opt, loaders._loader)
		ifbreak && break

		# fullbatch metric
		if !iszero(fullbatch_freq)
			if (opt_iter.epoch[] % fullbatch_freq) == 0
				evaluate(trainer, state, loaders)
				state, ifbreak = update_trainer_state!(trainer, state)
				ifbreak && break
			end
		end

		# callback
		state, ifbreak = callbacks.cb_epoch(trainer, state, opt_iter.epoch[])
		if ifbreak
			if verbose
				@info "Got signal from epoch callback to stop training at \
				epoch $(opt_iter.epoch[])."
			end
			break
		end
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

	ifbreak = false
	show_batch = (length(_loader) > 1) & verbose & print_batch

	if show_batch
		prog_meter = ProgressMeter.Progress(
			length(_loader);
			barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
			desc = "Epoch [$(opt_iter.epoch[]) / $(opt_args.nepochs)] LR: $(round(opt.eta; sigdigits=3))",
			dt = 1e-4, barlen = 25, color = :normal, showspeed = true, output = io,
		)
	end

	@set! state.st = Lux.trainmode(state.st)

	for (ibatch, batch) in enumerate(_loader)
		state, l, stats, ifbreak = step(trainer, state, opt, batch, ibatch)

		if show_batch
			showvalues = Any[(:LOSS, round(l; sigdigits = 8)),]
			!isempty(stats) && push!(showvalues, (:INFO, stats))
			ProgressMeter.next!(prog_meter; showvalues, valuecolor = :magenta)
		end

		ifbreak && break
	end

	if show_batch
		ProgressMeter.finish!(prog_meter)
	end

	return state, ifbreak
end

function step(
	trainer::Trainer,
	state::TrainState,
	::Optimisers.AbstractRule,
	batch,
	ibatch::Integer,
)
	@unpack NN, p, st, opt_st = state
	@unpack lossfun, callbacks = trainer
	epoch = trainer.opt_iter.epoch[]

	# compute gradient
	loss = Loss(NN, st, batch, lossfun)
	l, st, stats, g = grad(loss, p)
	isnan(l) && throw(ErrorException("Loss in NaN"))

	# callback
	state, ifbreak = callbacks.cb_batch(trainer, state, batch, l, g, epoch, ibatch)
	if ifbreak
		@info "Got signal from batch callback to stop training at batch \
		$(ibatch) of epoch $(epoch)."
		return state, l, stats, true # return w/o applying grads
	end

	# apply gradient
	opt_st, p = Optimisers.update!(opt_st, p, g)
	state = TrainState(NN, p, st, opt_st)

	return state, l, stats, false
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
	@unpack lossfun, opt_iter, device, callbacks = trainer
	@unpack nepochs, fullbatch_freq = trainer.opt_args
	@unpack io, verbose, print_epoch, print_config = trainer.io_args

	stats_global = nothing

	batch = if __loader isa MLDataDevices.DeviceIterator
		__loader.iterator.data |> device
	else
		__loader.data
	end

    function optloss(optx, optp)
		l, st, stats = lossfun(state.NN, optx, state.st, batch)
		@set! state.st = st
		stats_global = stats
		l
    end

	function optcb(optx, l)
		@set! state.p = optx.u

		if !isempty(stats_global) & verbose & print_epoch
			println(io, stats)
		end

		if !iszero(fullbatch_freq)
			if (opt_iter.epoch[] % fullbatch_freq) == 0
				evaluate(trainer, state, loaders)
				state, ifbreak = update_trainer_state!(trainer, state)
				ifbreak && return true
			end
		end

		# callback
		state, ifbreak = callbacks.cb_epoch(trainer, state, opt_iter.epoch[])
		if ifbreak
			if verbose
				@info "Got signal from epoch callback to stop training at \
				epoch $(opt_iter.epoch[])."
			end
			return true
		end

		# update epoch
		opt_iter.epoch[] += 1

		return false
	end

    #======================#
    # set up optimization and solve
    #======================#
    adtype  = AutoZygote()
    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, state.p)

    optsol = solve(optprob, trainer.opt;
		callback = optcb, maxiters = nepochs,
	)

	if verbose
		println(io, "Optimiser retcode: $(optsol.retcode)")
	end

	return state
end
#============================================================#
# Gradient API?
#============================================================#
#
