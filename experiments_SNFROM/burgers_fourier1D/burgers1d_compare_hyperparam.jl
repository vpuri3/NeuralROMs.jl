#
using NeuralROMs
using LaTeXStrings
import CairoMakie
import CairoMakie.Makie

joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = BurgersViscous1D(1f-3)
datafile = joinpath(@__DIR__, "data_burg1D", "data.jld2")
device = Lux.gpu_device()

function compare_burg1d_param(
    prob::NeuralROMs.AbstractPDEProblem,
    datafile::String;
	train::Bool = false,
	evolve::Bool = false,
	makeplot::Bool = true,
	rng = Random.default_rng(),
	device = Lux.gpu_device(),
)
	
	latent = 2
	_Ib, Ib_ = [1, 3, 5,], [2, 4, 6]
	Ix  = Colon()
	_It = Colon()
	makedata_kws = (; Ix, _Ib, Ib_, _It, It_ = :)

	alphas = (1f-4, 5f-5, 1f-5, 5f-6, 1f-6, 5f-7, 1f-7)
	gammas = (5f-1, 1f-1, 5f-2, 1f-2, 5f-3, 1f-3, 5f-4)

	k = 3

	if makeplot
		cases = (;)
		_, t, _, ud, _ = loaddata(datafile)[4]
		ud = ud[:,:,k,:] # [out_dim, Nx, Nb, Nt]
        nr = sqrt(sum(abs2, ud; dims=2) / size(ud, 2))
	end

	for (i, (a, g)) in enumerate(zip(alphas, gammas))

		l0 = lpad(latent, 2, "0")

		modeldir_SNW = joinpath(@__DIR__, "model_SNW$(l0)_$(i)")
		modeldir_SNL = joinpath(@__DIR__, "model_SNL$(l0)_$(i)")

		#==================#
		# train
		#==================#

		if train
			train_params_SNW = (; E = 1400, wd = 128, α = 0f-0, γ = g, makedata_kws)
			train_params_SNL = (; E = 1400, wd = 128, α = a, γ = 0f-0, makedata_kws)

			train_SNF_compare(latent, datafile, modeldir_SNW, train_params_SNW; rng, device)
			train_SNF_compare(latent, datafile, modeldir_SNL, train_params_SNL; rng, device)
		end

		#==================#
		# evolve
		#==================#

		modelfile_SNW = joinpath(modeldir_SNW, "model_08.jld2")
		modelfile_SNL = joinpath(modeldir_SNL, "model_08.jld2")

		if evolve
			postprocess_SNF(prob, datafile, modelfile_SNW; rng, device)
			postprocess_SNF(prob, datafile, modelfile_SNL; rng, device)
		end

		if makeplot
			kL = Symbol("SNFL-$(i)")
			kW = Symbol("SNFW-$(i)")

			nameL = L"SNFL-ROM ($\alpha = %$(alphas[i])$)"
			nameW = L"SNFL-ROM ($\gamma = %$(gammas[i])$)"

			epL = 0
			erL = 0

			fileL = joinpath(modelfile_SNL, "evolve$(k).jld2")
			fileW = joinpath(modelfile_SNW, "evolve$(k).jld2")

			evL = jldopen(fileL)
			evW = jldopen(fileW)

			upL = evL["Upred"]
			upW = evW["Upred"]

			epL = (up - ud)

			er = (up - ud) ./ nr
			er = sum(abs2, er; dims=2) / size(ud, 2) |> vec
			er = sqrt.(er) + 1f-12
			er = sqrt(sum(abs2, er) / length(er)) # error vs time
		end

	end

	#==================#
	# make plot
	#==================#

	if makeplot
		fig = Makie.Figure(; size = (600, 400), backgroundcolor = :white, grid = :off)
		ax  = Makie.Axis(
			fig[1,1];
			xlabel = L"t",
			ylabel = L"ε(t)",
			xlabelsize = 16,
			ylabelsize = 16,
		)

		for (k, (v, n)) in pairs(cases)
			println(k, " ", v)
			Makie.scatterlines!(npoints, v; label = n, linewidth = 2)
		end

		Makie.axislegend(ax)
		# fig[1,2] = Makie.Legend(fig, ax)

		save(joinpath(@__DIR__, "param.png"), fig)
		# save(joinpath(pkgdir(NeuralROMs), "figs", "method", "exp_param.pdf"), fig)
	end
end


#======================================================#
# compare_burg1d_param(prob, datafile; rng, device, train = true, evolve = true)
compare_burg1d_param(prob, datafile; rng, device)
#======================================================#
nothing
