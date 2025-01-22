#
using Lux, MLDataDevices
using Zygote, Enzyme, Reactant
using CUDA, LuxCUDA, KernelAbstractions
using Random, Printf, Optimisers, MLUtils

function train(device, adtype)

	N = 10000
	W = 64
	B = div(N, 100)
	E = 1

	# model
	NN = Chain(Dense(1 => W, gelu), Dense(W => W, gelu), Dense(W => 1))
	ps, st = Lux.setup(Random.default_rng(), NN)

	# data
	x = LinRange(0f0, 1f0, N) |> Array
	x = reshape(x, 1, N)
	y = @. sinpi(2x)

	# data loader
	DL = DataLoader((x, y); batchsize = B)

	# device transfer
	ps = ps |> device
	st = st |> device
	DL = DeviceIterator(device, DL)

	# training
    train_state = Training.TrainState(NN, ps, st, Adam(0.001f0))

    for epoch in 1:E
		loss = 0
        for batch in DL
            _, loss, _, train_state = Training.single_train_step!(
                adtype, MSELoss(), batch, train_state)
        end
    end

    return train_state
end

####
# CPU
####

@time train(cpu_device(), AutoZygote())
@time train(cpu_device(), AutoEnzyme())
Reactant.set_default_backend("cpu")
@time train(reactant_device(), AutoEnzyme())

####
# GPU
####

@time train(gpu_device(), AutoZygote())
@time train(gpu_device(), AutoEnzyme()) # failing
Reactant.set_default_backend("gpu")
@time train(reactant_device(), AutoEnzyme())

#====================================================#
nothing
#
