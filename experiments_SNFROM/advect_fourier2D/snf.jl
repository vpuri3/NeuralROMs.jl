#
using NeuralROMs
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection2D(0.25f0, 0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "dump")
modelfile = joinpath(modeldir, "model_08.jld2")
device = Lux.gpu_device()

grid = (128, 128,)

# # train
# latent = 2
# batchsize_ = (128 * 128) * 500 ÷ 4
# makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)
# train_params = (; E = 1400, wd = 128, α = 0f-0, γ = 1f-2, makedata_kws, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # modeldir/results
# postprocess_SNF(prob, datafile, modelfile; rng, device)

# # modeldir/hyper
# outdir = joinpath(modeldir, "hyper")
# hyper_reduction_path = joinpath(modeldir, "hyper.jld2")
# evolve_kw = (; hyper_reduction_path, hyper_indices, verbose = false,)
# postprocess_SNF(prob, datafile, modelfile; rng, evolve_kw, outdir, device)

solvestats = (;)

for dt_mult in [1, 2, 5] # time-step
    for iskip in [1, 2, 4, 8, 16] # indices
        # hyper-indices
        ids = zeros(Bool, grid...)
        @views ids[1:iskip:end, 1:iskip:end] .= true
        hyper_indices = findall(isone, vec(ids))
        hyper_reduction_path = joinpath(modeldir, "hyper.jld2")

        # time-step
        It = LinRange(1, 500, 500 ÷ dt_mult) .|> Base.Fix1(round, Int)
        data_kws = (; Ix = :, It)
        evolve_kw = (; Δt = 10f0, data_kws, hyper_reduction_path, hyper_indices, verbose = false,)

        # directory
        N = length(hyper_indices)
        casename = "N$(N)_dt$(dt_mult)"
        outdir = joinpath(modeldir, "hyper_$(casename)")

        # run
        _, stats = evolve_SNF(prob, datafile, modelfile, 1; rng, outdir, evolve_kw..., device)

        global solvestats = (; solvestats..., Symbol(casename) => stats)
    end
end

statsfile = joinpath(modeldir, "hyperstats.jld2")
jldsave(statsfile; solvestats)
hyper_plots(datafile, modeldir, 1)

#======================================================#

# # timing plots
# tme_FOM = 0.499433 # s
# mem_FOM = 1.602    # GiB
#
# Ngrid = prod(grid)
# skips = 4 .^ Array(0:4) # indices: 1, 2, 4, 8, 16
#
# Nps = Ngrid ./ skips
# DTs = [1, 2, 5, 10]
#
# # s
# tme_01 = [8.655742 4.336693 1.748981 0.896192]
# tme_02 = [2.492559 1.242272 0.511401 0.267714]
# tme_04 = [1.009825 0.500941 0.211161 0.087910]
# tme_08 = [0.759159 0.358999 0.147521 0.074116]
# tme_16 = [0.839552 0.351671 0.144932 0.072996]
#
# # GiB
# mem_01 = [355.435 177.884 71.353 35.842]
# mem_02 = [89.158 44.621 17.899 8.991]
# mem_04 = [22.588 11.305 4.535 2.278]
# mem_08 = [5.946 2.976 1.194 614.583/1024]
# mem_16 = [1.785 915.264/1024 367.540/1024 184.966/1024]
#
# # rel-MSE-error %
# err_01 = [0.33465478 0.63720053 1.568733 3.1377885]
# err_02 = [0.33475608 0.63742095 1.5687758 3.1377559]
# err_04 = [0.333026 0.63556844 1.5664246 3.1343515]
# err_08 = [0.32001957 0.6240544 1.5579201 3.1301193]
# err_16 = [0.27075416 0.5654563 1.487445 3.042574]
#
# tmes = vcat(tme_01, tme_02, tme_04, tme_08, tme_16)
# mems = vcat(mem_01, mem_02, mem_04, mem_08, mem_16)
# errs = vcat(err_01, err_02, err_04, err_08, err_16)
#
# # speedup
# speedups = tme_FOM ./ tmes
# memorys  = mem_FOM ./ mems

#======================================================#
nothing
