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
