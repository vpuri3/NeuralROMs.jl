#
#============================================================#
# Optimisers.jl
#============================================================#
function train_loop!(
	trainer::Trainer,
	::Optimisers.AbstractRule,
	minconfig::NamedTuple,
)
	@unpack opt_args, opt_iter = trainer

	for _ in 1:opt_args.nepochs
		opt_iter.epoch[] += 1
		opt_iter.epoch_time[] = time() - opt_iter.start_time[]

		epoch!(trainer)
		evaluate(trainer)

		opt_iter.epoch_dt[] = time() - opt_iter.epoch_time[] - opt_iter.start_time[]
		trigger_callback!(trainer, :EPOCH_END)

		# update minconfig
		minconfig, ifbreak = update_minconfig(trainer, minconfig)
		ifbreak && break
	end

	return minconfig
end

function epoch!(
	trainer::Trainer,
	::Optimisers.AbstractRule
)
	@unpack _loader = trainer
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
		l, stats = step!(trainer, batch)
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

	return
end

function step!(
	trainer::Trainer,
	::Optimisers.AbstractRule,
	batch,
)
	loss = Loss(trainer.NN, trainer.st, batch, trainer.lossfun)
	l, st, stats, g = grad(loss, trainer.p)
	opt_st, p = Optimisers.update!(trainer.opt_st, trainer.p, g)

	if isnan(l)
		throw(ErrorException("Loss in NaN"))
	end

	trainer.p = p
	trainer.st = st
	trainer.opt_st = opt_st

	return l, stats
end

#============================================================#
# OptimzationOptimJL.jl
#============================================================#
function train_loop!(
	trainer::Trainer,
	::Optim.AbstractOptimizer,
	minconfig::NamedTuple,
)
	@unpack __loader, lossfun = trainer
	@unpack opt_args, opt_iter = trainer
	@unpack io, verbose = trainer

	batch = if __loader isa CuIterator
		# Adapt.adapt(__loader, __loader.batches.data)
		__loader.batches.data |> cu
	else
		__loader.data
	end
	# batch = first(__loader)

    function optloss(optx, optp)
        lossfun(trainer.NN, optx, trainer.st, batch)
    end

	function optcb(optx, l, st, stats)
		trainer.p  = optx.u
		trainer.st = st

		evaluate(trainer)
		trainer.st = Lux.trainmode(trainer.st)

		if !isempty(stats) & verbose
			println(io, stats)
		end

		opt_iter.epoch[] += 1
		opt_iter.epoch_dt[] = time() - opt_iter.epoch_time[] - opt_iter.start_time[]
		opt_iter.epoch_time[] = time() - opt_iter.start_time[]
		trigger_callback!(trainer, :EPOCH_END)

		minconfig, ifbreak = update_minconfig(trainer, minconfig)

		return ifbreak
	end

    #======================#
    # set up optimization and solve
    #======================#
    adtype  = AutoZygote()
    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, trainer.p)

	kwargs = (; callback = optcb, maxiters = opt_args.nepochs)
    optsol = solve(optprob, trainer.opt; kwargs...)

	if trainer.verbose
		@show optsol.retcode
	end

	return minconfig
end

#============================================================#
#
