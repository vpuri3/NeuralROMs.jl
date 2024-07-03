#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = BurgersViscous2D(1f-3)
datafile = joinpath(@__DIR__, "data_burgers2D/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_07.jld2")
device = Lux.gpu_device()

grid = (512, 512,)

# train
#
# latent = 2
# _batchsize = prod(grid) * 1
# batchsize_ = prod(grid) * 8
# _Ib, Ib_ = [1, 2, 3, 5, 6, 7], [4,]
# makedata_kws = (; Ix = :, _Ib, Ib_, _It = :, It_ = :)
# train_params = (; E = 210, wd = 128, γ = 1f-2, makedata_kws, _batchsize, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # modeldir/results
# postprocess_SNF(prob, datafile, modelfile; rng, device)

function timings_burg2D(casenum::Integer = 1)
    statsROM = (;)

    for dt_mult in reverse([1, 2, 5, 10]) # time-step
        for iskip in reverse([4, 8, 16, 32, 64]) # indices
            # hyper-indices
            ids = zeros(Bool, grid...)
            @views ids[1:iskip:end, 1:iskip:end] .= true
            hyper_indices = findall(isone, vec(ids))
            hyper_reduction_path = joinpath(modeldir, "hyper.jld2")

            # time-step
            It = LinRange(1, 500, 500 ÷ dt_mult) .|> Base.Fix1(round, Int)
            data_kws = (; Ix = :, It)

            learn_ic = false
            evolve_kw = (; Δt = 10f0, data_kws, hyper_reduction_path, hyper_indices, learn_ic, verbose = false,)

            # directory
            N = length(hyper_indices)
            casename = "N$(N)_dt$(dt_mult)"
            outdir = joinpath(modeldir, "hyper_$(casename)")

            # run
            _, stats = evolve_SNF(prob, datafile, modelfile, casenum; rng, outdir, evolve_kw..., device)
            statsROM = (; statsROM..., Symbol(casename) => stats)
        end
    end

    # FOM stats
    statsFOM = include(joinpath(@__DIR__, "FOM_timings.jl"))

    statsfile = joinpath(modeldir, "hyperstats.jld2")
    jldsave(statsfile; statsROM, statsFOM)
    hyper_plots(datafile, modeldir, casenum)

    statsROM, statsROM
end

sROM, sFOM = timings_burg2D(4)
# e2hyper = joinpath(modeldir, "hypercompiled.jld2")
# df_exp2 = makeplots_hyper(e2hyper, outdir, "exp2")
#======================================================#
nothing
