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

	while opt_iter.epoch[] < opt_args.nepochs
		opt_iter.epoch[] += 1
		opt_iter.epoch_time[] = time() - opt_iter.start_time[]

		state = doepoch(trainer, state, opt, loaders._loader)
		evaluate(trainer, state, loaders)

		opt_iter.epoch_dt[] = time() - opt_iter.epoch_time[] - opt_iter.start_time[]
		trigger_callback!(trainer, :EPOCH_END)

		# update state loss
		@set! state.loss_ = trainer.STATS.LOSS_[end]

		# save state to CPU if loss improves
		state, ifbreak = update_trainer_state!(trainer, state)
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
	@unpack opt_args, opt_iter, io = trainer

	if trainer.verbose
		prog_meter = ProgressMeter.Progress(
			length(_loader);
			barglyphs = ProgressMeter.BarGlyphs("[=> ]"),
			desc = "Epoch [$(opt_iter.epoch[]) / $(opt_args.nepochs)] LR: $(round(opt.eta; sigdigits=3))",
			dt = 1e-4, barlen = 25, color = :normal, showspeed = true, output = io,
		)
	end

	state.st = Lux.trainmode(state.st)

	for batch in _loader
		state, (l, stats) = step(trainer, state, opt, batch)
		trigger_callback!(trainer, :BATCH_END)

		if trainer.verbose
			showvalues = Any[(:LOSS, round(l; sigdigits = 8)),]
			!isempty(stats) && push!(showvalues, (:INFO, stats))
			ProgressMeter.next!(prog_meter; showvalues, valuecolor = :magenta)
		end
	end

	if trainer.verbose
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

	# wrong l. want STATS.LOSS_[end]
	state = TrainState(NN, p, st, opt_st, state.count, state.loss_)

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
	@unpack lossfun, opt_args, opt_iter = trainer
	@unpack io, verbose = trainer

	# batch = first(__loader)
	batch = if __loader isa CuIterator
		# Adapt.adapt(__loader, __loader.batches.data)
		__loader.batches.data |> cu
	else
		__loader.data
	end

	### TODO: using old st in BFGS
    function optloss(optx, optp)
        lossfun(state.NN, optx, state.st, batch)
    end

	function optcb(optx, l, st, stats)
		evaluate(trainer, state, loaders)
		state = TrainState(state.NN, optx.u, Lux.trainmode(st), state.opt_st, state.count, trainer.STATS.LOSS_[end])

		if !isempty(stats) & verbose
			println(io, stats)
		end

		opt_iter.epoch[] += 1
		opt_iter.epoch_dt[] = time() - opt_iter.epoch_time[] - opt_iter.start_time[]
		opt_iter.epoch_time[] = time() - opt_iter.start_time[]
		trigger_callback!(trainer, :EPOCH_END)

		state, ifbreak = update_trainer_state!(trainer, state)
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

	if trainer.verbose
		@show optsol.retcode
	end

	return state
end
#============================================================#
# Gradient API?
#============================================================#
#
