#
using NeuralROMs
using Lux
using OptimizationOptimJL

N = 1000

x = rand(5, N)
y = rand(1, N)

data = (x, y)
NN = Chain(Dense(5, 8, tanh), Dense(8, 8, tanh), Dense(8, 1))

# @time (NN, p, st), ST = train_model(
# 	NN, data; rng, p, st, _batchsize, batchsize_,
# 	opts, nepochs, schedules, early_stoppings,
# 	device, dir, metadata, lossfun,
# )

# @time (NN, p, st), ST = train_model(NN, data)
# trainer = Trainer(NN, data, verbose = true)
# @time model, ST = train!(trainer)

# @time (NN, p, st), ST = train_model(NN, data; opts = (LBFGS(),))
trainer = Trainer(NN, data, verbose = true, opt = LBFGS())
@time model, ST = train!(trainer)

#
nothing
