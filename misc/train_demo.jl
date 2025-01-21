#
using NeuralROMs, Lux
using Optimisers, OptimizationOptimJL
using CUDA, LuxCUDA, KernelAbstractions

N = 1000
W = 32

x = LinRange(0f0, 1f0, N) |> Array
x = reshape(x, 1, N)
y = @. sin(1x)

_data = (x, y)
data_ = (x, y)
NN = Chain(Dense(1, W, tanh), Dense(W, W, tanh), Dense(W, 1))
device = cpu_device()
device = gpu_device()

cb_epoch = function(trainer, state, epoch)
	println("Epoch: $(epoch)")
	return state, false
end

cb_batch = function(trainer, state, batch, loss, grad, epoch, ibatch)
	println("Epoch: $(epoch) batch $(ibatch)")
	return state, false
end

trainer = Trainer(NN, _data; device,
	verbose = true, print_config = false, print_stats = false,
	print_batch = false, print_epoch = false, fullbatch_freq = 10,
	# cb_batch,
)
@time model, ST = train!(trainer)

# trainer = Trainer(
# 	NN, _data; make_ca = true, opt = Optim.BFGS(),
# 	verbose = true, print_config = false, print_stats = false, print_epoch = false,
# 	fullbatch_freq = 10, cb_epoch
# )
# @time model, ST = train!(trainer)

# @time train_model(NN, data; opts = (Optim.LBFGS(),), device)
# @time (NN, p, st), ST = train_model(NN, _data)
# @time (NN, p, st), ST = train_model(
# 	NN, _data; device,
# 	opts = (Optimisers.Adam(), Optimisers.Adam(), Optim.BFGS(),),
# 	nepochs = (10, 10, 10),
# )

#
nothing
