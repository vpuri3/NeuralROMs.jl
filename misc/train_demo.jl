#
using NeuralROMs, Lux
using Optimisers, OptimizationOptimJL
using CUDA, LuxCUDA, KernelAbstractions

N = 1000
W = 32

x = rand(Float32, 5, N)
y = rand(Float32, 1, N)

data = (x, y)
NN = Chain(Dense(5, W, tanh), Dense(W, W, tanh), Dense(W, 1))
device = Lux.cpu_device()
device = Lux.gpu_device()

# MIXED TEST
@time (NN, p, st), ST = train_model(
	NN, data; device,
	opts = (Optimisers.Adam(), Optimisers.Adam(), Optim.BFGS(),),
	nepochs = (10, 10, 10),
)

# @time (NN, p, st), ST = train_model(NN, data)
# trainer = Trainer(NN, data; device, verbose = false)
# @time model, ST = train!(trainer)

# @time train_model(NN, data; opts = (Optim.LBFGS(),), device)
# trainer = Trainer(NN, data, verbose = true, opt = Optim.BFGS())
# @time model, ST = train!(trainer)

#
nothing

# function train_model(
#     NN::Lux.AbstractExplicitLayer,
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
#     device = Lux.gpu_device(),
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
#     p, st = Lux.cpu_device()((p, st))
#     (NN, p, st), STATS
# end
