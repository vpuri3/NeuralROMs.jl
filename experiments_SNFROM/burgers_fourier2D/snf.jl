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

# # train
# _batchsize = prod(grid) * length(_It) ÷ 500
# batchsize_ = prod(grid) * length(_It) ÷ 500
#
# makedata_kws = (; Ix = :, _Ib = [1], Ib_ = [1], _It = :, It_ = :)
#
# latent = 2
# train_params = (; E = 700, wd = 128, γ = 1f-2, makedata_kws, _batchsize, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # train
# latent = 2
# batchsize_ = (128 * 128) * 500 ÷ 4
# makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)
# train_params = (; E = 1400, wd = 128, α = 0f-0, γ = 1f-2, makedata_kws, batchsize_)
# train_SNF_compare(latent, datafile, modeldir, train_params; rng, device)

# # modeldir/results
# postprocess_SNF(prob, datafile, modelfile; rng, device)

function timings()
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
            _, stats = evolve_SNF(prob, datafile, modelfile, 1; rng, outdir, evolve_kw..., device)
            statsROM = (; statsROM..., Symbol(casename) => stats)
        end
    end

    # FOM stats
    statsFOM = include(joinpath(@__DIR__, "FOM_timings.jl"))

    statsfile = joinpath(modeldir, "hyperstats.jld2")
    jldsave(statsfile; statsROM, statsFOM)
    hyper_plots(datafile, modeldir, 1)
    nothing
end

timings()

#======================================================#

# # timing plots
#
# tme_FOM = 15.779027 # s
# mem_FOM = 163.760   # GiB
#
# Ngrid = 512 * 512
# skips = 4 .^ Array(1:6) # indices: 2, 4, 8, 16, 32, 64
#
# Nps = Ngrid ./ skips
# DTs = [1, 2, 5, 10]
#
# # s
# tme_02 = [ 44.472603 22.082768 8.903516 4.608044 ]
# tme_04 = [ 11.086413 5.557169 2.298053 1.317982 ]
# tme_08 = [ 3.148577 1.633576 0.714816 0.565096 ]
# tme_16 = [ 1.438282 0.800424 0.375744 0.152445 ]
# tme_32 = [ 0.879129 0.505148 0.203869 0.134057 ]
# tme_64 = [ 0.886587 0.453801 0.155870 0.090066 ]
#
# # GiB
# mem_02 = [ 2.132 * 1024 1.067 * 1024 437.687 219.508 ]
# mem_04 = [ 546.079 273.206 109.482 54.907 ]
# mem_08 = [ 136.819 68.451 27.431 13.757 ]
# mem_16 = [ 34.503 17.263 6.918 3.470 ]
# mem_32 = [ 8.925 4.465 1.790 919/1024 ]
# mem_64 = [ 2.530 1.266 520.059/1024 261.227/1024 ]
#
# # rel-MSE-error %
# err_02 = [ 0.057878006 0.129677 0.35801634 0.74683654 ]
# err_04 = [ 0.05751661 0.13067256 0.35610333 0.734099 ]
# err_08 = [ 0.03925267 0.105229855 0.3400021 0.69685143 ]
# err_16 = [ 0.050767798 0.11261147 0.33057037 0.7003481 ]
# err_32 = [ 0.0455603 0.103269145 0.32034075 0.7164427 ]
# err_64 = [ 0.09922275 0.05392034 0.23775052 0.5678324 ]
#
# tmes = vcat(tme_02, tme_04, tme_08, tme_16, tme_32, tme_64)
# mems = vcat(mem_02, mem_04, mem_08, mem_16, mem_32, mem_64)
# errs = vcat(err_02, err_04, err_08, err_16, err_32, err_64)
#
# # speedup
# speedups = tme_FOM ./ tmes
# memorys  = mem_FOM ./ mems

#======================================================#
nothing
