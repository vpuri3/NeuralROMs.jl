#
using Lux, MLDataDevices
using CUDA, LuxCUDA, KernelAbstractions
using Random, Printf, Optimisers, MLUtils

using Zygote
using Enzyme
using Reactant

function train(device, adtype)

	N = 10000
	W = 64
	E = 500

	# model
	NN = Chain(Dense(1 => W, gelu), Dense(W => W, gelu), Dense(W => 1))
	ps, st = Lux.setup(Random.default_rng(), NN)

	# data
	x = LinRange(0f0, 1f0, N) |> Array
	x = reshape(x, 1, N)
	y = @. sinpi(2x)

	# data loader
	DL = DataLoader((x, y); batchsize = div(N, 100))

	# device transfer
	ps = ps |> device
	st = st |> device
	DL = DeviceIterator(device, DL)

	# training
    train_state = Training.TrainState(NN, ps, st, Adam(0.001f0))

    for epoch in 1:E
        for (i, (xᵢ, yᵢ)) in enumerate(DL)
            _, loss, _, train_state = Training.single_train_step!(
                adtype, MSELoss(), (xᵢ, yᵢ), train_state)
            if (epoch % E == 0 || epoch == 1) && i == 1
				println("Epoch $(epoch)/$(E)\tLoss: $(loss)")
            end
        end
    end

    return train_state
end

# @time train(cpu_device(), AutoZygote())
# @time train(gpu_device(), AutoZygote())

# @time train(cpu_device(), AutoEnzyme())
# @time train(gpu_device(), AutoEnzyme())

# Reactant.set_default_backend("cpu")
# @time train(reactant_device(), AutoEnzyme())
Reactant.set_default_backend("gpu")
@time train(reactant_device(), AutoEnzyme())

#====================================================#
nothing
#
