#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(1.0f0)
datafile = joinpath(@__DIR__, "data_advect", "data.jld2")
device = Lux.gpu_device()

_It = 1:500
makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It, It_ = :)
data_kws = (; Ix = :, It = :)
evolve_kw = (; data_kws,)
periodic_layer = true
# periodic_layer = false

#==================#
# train
#==================#

for latent in [2] # [1, 2]
	modeldir  = joinpath(@__DIR__, "dump$(latent)")
	modelfile = joinpath(modeldir, "model_08.jld2")

	train_params = (;
		E = 700,
		wd = 64,
		α = 0f-0,
		γ = 0f-2,
		makedata_kws,
		# _batchsize = 128,

		# hyper
		λ2 = 0f-6,
		hh = 2, # hidden
		wh = 32, # width
	)
	train_SNF_compare(latent, datafile, modeldir, train_params; periodic_layer, rng, device)
	# postprocess_SNF(prob, datafile, modelfile; rng, device, evolve_kw)
end

#======================================================#
nothing
