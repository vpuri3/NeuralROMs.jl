#
using Lux, MLDataDevices
using Zygote, Enzyme, Reactant
using CUDA, LuxCUDA
using Random, Optimisers, MLUtils

function train(device, adtype, verbose)

	N = 10000
	H = 1
	W = 128
	B = div(N, 100)
	E = 100

	# model
	NN = Chain(
		Dense(1 => W, gelu),
		[Dense(W => W, gelu) for _ in 1:H]...,
		Dense(W => 1)
	)
	ps, st = Lux.setup(Random.default_rng(), NN)

	# data
	x = LinRange(0f0, 1f0, N) |> Array
	x = reshape(x, 1, N)
	y = @. sinpi(2x)

	# data loader
	DL = DataLoader((x, y); batchsize = B, partial = false)

	# device transfer
	ps = ps |> device
	st = st |> device
	DL = DeviceIterator(device, DL)

	# training
    train_state = Training.TrainState(NN, ps, st, Adam(0.001f0))

	verbose && println("##################")
	verbose && println("$(device) + $(adtype)")
	verbose && println("##################")

	# warm up
	verbose && println("Warm up:")
	@time begin
		batch = first(DL)
		_, _, _, train_state = Training.single_train_step!(
			adtype, MSELoss(), batch, train_state;
			return_gradients = Training.False(),
		)
	end

	verbose && println("Training:")
    @time for epoch in 1:E
		loss = 0
        for batch in DL
            _, loss, _, train_state = Training.single_train_step!(
                adtype, MSELoss(), batch, train_state;
				return_gradients = Training.False(),
			)
        end
    end

	verbose && println("")

    return train_state
end

####
# CPU
####

# train(cpu_device(), AutoZygote(), false)
# train(cpu_device(), AutoEnzyme(), false)
# Reactant.set_default_backend("cpu")
# train(reactant_device(), AutoEnzyme(), false)
#
# train(cpu_device(), AutoZygote(), true)
# train(cpu_device(), AutoEnzyme(), true)
# train(reactant_device(), AutoEnzyme(), true)

####
# GPU
####

# train(gpu_device(), AutoZygote(), false); CUDA.reclaim()
# train(gpu_device(), AutoEnzyme(), false) # failing
Reactant.set_default_backend("gpu")
train(reactant_device(), AutoEnzyme(), false)

# train(gpu_device(), AutoZygote(), true); CUDA.reclaim()
# @time train(gpu_device(), AutoEnzyme(), true) # failing
train(reactant_device(), AutoEnzyme(), true)

#====================================================#
nothing
#
