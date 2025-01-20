#
using NeuralROMs
using LaTeXStrings, Printf
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

	icase = 4

	if makeplot
		casesL = (;)
		casesW = (;)
		_, t, _, ud, _ = loaddata(datafile; verbose = false) # [Nt], [out_dim, Nx, Nb, Nt]
		ud = ud[:,:,icase,:] # [1, Nx, Nt]
		nr = sqrt.(sum(abs2, ud; dims=2) / size(ud, 2)) # [1, 1, Nt]
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

			fileL = joinpath(modeldir_SNL, "results", "evolve$(icase).jld2")
			fileW = joinpath(modeldir_SNW, "results", "evolve$(icase).jld2")

			evL = jldopen(fileL)
			evW = jldopen(fileW)

			# projection error
			ulL = evL["Ulrnd"]
			ulW = evW["Ulrnd"]

			epL = (ulL - ud) ./ nr
			epW = (ulW - ud) ./ nr

			# epL = sum(abs2, epL) / length(epL) |> sqrt
			# epW = sum(abs2, epW) / length(epL) |> sqrt

			epL = sum(abs2, epL; dims=2) / size(ud, 2) .|> sqrt |> vec
			epW = sum(abs2, epW; dims=2) / size(ud, 2) .|> sqrt |> vec

			# prediction error
			upL = evL["Upred"]
			upW = evW["Upred"]

			erL = (upL - ud) ./ nr
			erW = (upW - ud) ./ nr

			erL = sum(abs2, erL; dims=2) / size(ud, 2) .|> sqrt |> vec
			erW = sum(abs2, erW; dims=2) / size(ud, 2) .|> sqrt |> vec

			# merge
			kL = Symbol("SNFL-$(i)")
			kW = Symbol("SNFW-$(i)")

			aa = @sprintf("%.1e", a) # Produces "5.0e-04"
			gg = @sprintf("%.1e", g)

			aa = replace(aa, ".0" => "") # Removes trailing ".0"
			gg = replace(gg, ".0" => "") # "5e-04"

			aa = replace(aa, "-0" => "-") # "5e-4"
			gg = replace(gg, "-0" => "-")

			nL = L"$\alpha = %$(aa)$"
			nW = L"$\gamma = %$(gg)$"

			cL = (; name = nL, er = erL, ep = epL)
			cW = (; name = nW, er = erW, ep = epW)

			# _cases = NamedTuple{(kL, kW,)}((cL, cW,))
			casesL = merge(casesL, NamedTuple{(kL,)}((cL,)))
			casesW = merge(casesW, NamedTuple{(kW,)}((cW,)))
		end
	end

	#==================#
	# make plot
	#==================#

	if makeplot
		fig = Makie.Figure(; size = (900, 500), backgroundcolor = :white, grid = :off)
		# fig = Makie.Figure(; size = (900, 900), backgroundcolor = :white, grid = :off)
		kwa = (; xlabel = L"t", ylabel = L"ε(t)", xlabelsize = 16, ylabelsize = 16, yscale = log10,)
		axL = Makie.Axis(fig[1,1]; kwa...)
		axW = Makie.Axis(fig[1,2]; kwa...)

		# kwa2 = (; ylabel = L"Projection error $e_\text{proj}$", xlabelsize = 16, ylabelsize = 16, xscale = log10, yscale = log10,)
		# axL2 = Makie.Axis(fig[3,1]; xlabel = L"\alpha", kwa2...)
		# axW2 = Makie.Axis(fig[3,2]; xlabel = L"\gamma", kwa2...)

		# epL = []
		# epW = []

		for (k, (n, er, ep)) in pairs(casesL)
			Makie.lines!(axL, t, er; label = n, linewidth = 3)
			# Makie.lines!(axL, t, ep; linestyle = :dash)
			# push!(epL, ep)
		end

		for (k, (n, er, ep)) in pairs(casesW)
			Makie.lines!(axW, t, er; label = n, linewidth = 3)
			# Makie.lines!(axW, t, ep; linestyle = :dash)
			# push!(epW, ep)
		end

		# Makie.lines!(axL2, [alphas...], epL; linewidth = 3)
		# Makie.lines!(axW2, [gammas...], epW; linewidth = 3)

		# Makie.axislegend(axL; orientation = :horizontal, nbanks = 2)
		# Makie.axislegend(axW; orientation = :horizontal, nbanks = 2)

		kwl = (;orientation = :horizontal, nbanks = 3, fontsize = 16, framevisible = false)
		fig[0, 1] = Makie.Legend(fig, axL; kwl...)
		fig[0, 2] = Makie.Legend(fig, axW; kwl...)

        Makie.Label(fig[2,1], L"(a) SNFL‐ROM$$", fontsize = 16)
        Makie.Label(fig[2,2], L"(b) SNFW‐ROM$$", fontsize = 16)

        Makie.colsize!(fig.layout, 1, Makie.Relative(0.50))
        Makie.colsize!(fig.layout, 2, Makie.Relative(0.50))

		save(joinpath(@__DIR__, "param.png"), fig)
		save(joinpath(pkgdir(NeuralROMs), "figs", "method", "exp_param.pdf"), fig)
	end
end

#======================================================#
# compare_burg1d_param(prob, datafile; rng, device, train = true, evolve = true)
compare_burg1d_param(prob, datafile; rng, device)
#======================================================#
nothing
