#
using NeuralROMs, LaTeXStrings
import CairoMakie
import CairoMakie.Makie

joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)

function compare_advect1d_s(
    prob::NeuralROMs.AbstractPDEProblem,
    datafile::String;
	evolve::Bool = false,
	makeplot::Bool = true,
	rng = Random.default_rng(),
	device = Lux.gpu_device(),
)
	Nx = 128
	casenum = 1

	npoints = [16, 32, 64, 128]
	latents = [2,] # [1, 2, 4]
	models = ["SNL", "SNW"]

	idx_unif = Array(1:Nx)
	idx_rand = randperm(Nx)

	if makeplot
		cases = (;)
		ud = loaddata(datafile)[4]
		ud = dropdims(ud; dims=3)
        nr = sqrt(sum(abs2, ud) / length(ud))
	end

	for latent in latents
		ll = lpad(latent, 2, "0")

		for model in models
			modeldir = joinpath(@__DIR__, "model_" * model * ll)
			modelfile = joinpath(modeldir, "model_08.jld2")
			modelname = model == "SNL" ? "SNFL-ROM" : "SNFW-ROM"

			for sample_type in (:uniform, :random)
				key   = Symbol("$(model)-$(ll)-$(sample_type)")
				error = zeros(Float32, length(npoints))
				name  = LaTeXString("$(modelname) ($(sample_type) sampling)")

				case = (; error, name)
				cases = merge(cases, NamedTuple{(key,)}((case,)))

				for (i, N) in enumerate(npoints)

					hyper_indices = if sample_type == :random
						idx_rand[1:N]
					elseif sample_type == :uniform
						indices = map(x -> round(Int, x), LinRange(1, Nx, N))
					else
						@assert false
					end

					# hyper-indices
					evolve_kw = (; hyper_indices, learn_ic = false, verbose = false)

					# directory
					case = Symbol("N_$(N)_$(sample_type)")
					outdir = joinpath(modeldir, "hyper_$(case)")

					# evolve
					if evolve
						evolve_SNF(prob, datafile, modelfile, casenum; rng, outdir, evolve_kw..., device)
					end

					if makeplot
						evolvefile = joinpath(outdir, "evolve1.jld2")
						ev = jldopen(evolvefile)
						up = ev["Upred"]
						er = @. (up - ud) / nr
						# er = sqrt(sum(abs2, er) / length(er))
						er = sum(abs2, up[:,:,end] - ud[:, :, end]) / sum(abs2, ud[:, :, end]) |> sqrt
						error[i] = er

						# case  = Symbol("$(model)-$(ll)-$(sample_type)-$(N)")
						# cases = merge(cases, NamedTuple{(case,)}((up,)))
					end

				end # N points
			end # sampling type

		end # model (SNFL, SNFW)
	end # latent

	if makeplot

        fig = Makie.Figure(; size = (600, 400), backgroundcolor = :white, grid = :off)
        ax  = Makie.Axis(
			fig[1,1];
		    xlabel = L"Number of hyper-reduction points ($|X_\text{proj}|$)",
		    ylabel = L"$\varepsilon(T)$",
			xlabelsize = 16,
			ylabelsize = 16,
            xscale = log2,
            yscale = log10,
        )

		for (k, (v, n)) in pairs(cases)
			println(k, v)
			Makie.scatterlines!(npoints, v; label = n, linewidth = 2)
		end

	    Makie.axislegend(ax)
		# fig[1,2] = Makie.Legend(fig, ax)

	    # Makie.xlims!(ax, 2^4, 2^7)
	    # Makie.ylims!(ax, 1e-4, 1e-1)
		save(joinpath(@__DIR__, "sampling.png"), fig)
		save(joinpath(pkgdir(NeuralROMs), "figs", "method", "exp_samp.pdf"), fig)
	end

    nothing
end

#======================================================#
compare_advect1d_s(prob, datafile; rng, device) #, evolve = true)
#======================================================#
nothing
