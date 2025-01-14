#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

datafile  = joinpath(@__DIR__, "data_AD1D", "data.jld2")
device = Lux.gpu_device()

case = 1
data_kws = (; Ix = :, It = :)

#------------------------------------------------------#
# DNN
#------------------------------------------------------#
# train_params = (;)
# evolve_params = (;)
#
# makemodel = makemodelDNN
# modelfilename = "model_08.jld2"
# modeldir  = joinpath(@__DIR__, "dump_dnn")

#------------------------------------------------------#
# Tanh Kernels
#------------------------------------------------------#
# train_params  = (;)
# evolve_params  = (; scheme = :GalerkinCollocation)
#
# makemodel = makemodelTanh
# modelfilename = "model_05.jld2"
# modeldir  = joinpath(@__DIR__, "dump_tanh")

#------------------------------------------------------#
# Tanh Kernels (split)
#------------------------------------------------------#
train_params  = (; N = 1, Nsplits = 3)
evolve_params  = (; scheme = :GalerkinCollocation)

makemodel = makemodelTanh
modelfilename = joinpath("split$(train_params.Nsplits)","model_05.jld2")
modeldir  = joinpath(@__DIR__, "dump_tanh_split")

#------------------------------------------------------#
# Gaussian Kernels
#------------------------------------------------------#
# data_kws = (; Ix = LinRange(1, 512, 64), It = LinRange(1, 500, 500))
# data_kws = map(x -> round.(Int, x), data_kws)

# train_params  = (;)
# evolve_params  = (; scheme = :GalerkinCollocation)
# # evolve_params = (; timealg = RungeKutta4())#, Δt = 1e-3, adaptive = false)
#
# makemodel = makemodelGaussian
# modelfilename = "model_05.jld2"
# modeldir  = joinpath(@__DIR__, "dump_gaussian")

#------------------------------------------------------#
# Gaussian (exact IC)
#------------------------------------------------------#
# data_kws = (; Ix = LinRange(1, 256, 64), It = LinRange(1, 500, 500))
# data_kws = map(x -> round.(Int, x), data_kws)

# train_params  = (; N = 1, exactIC = (; c = [1.0], x̄ = [-0.5], σ = [0.1]))
# evolve_params = (; timealg = RungeKutta4(),)

# makemodel = makemodelGaussian
# modelfilename = "model.jld2"
# modeldir  = joinpath(@__DIR__, "dump_gaussian_exact")

#------------------------------------------------------#
# MFN (Fourier)
#------------------------------------------------------#
# train_params = (; MFNfilter = :Fourier)
# evolve_params = (; timealg = RungeKutta4())
#
# makemodel = makemodelMFN
# modelfilename = "model_08.jld2"
# modeldir  = joinpath(@__DIR__, "dump_mfn_fourier")

#------------------------------------------------------#
# Evolve
#------------------------------------------------------#

# parameters
cs = Float32[1,    0,    1, 2.5]
νs = Float32[0, 0.01, 0.01,   0]

XD = TD = UD = UP = PS = ()
NN, p, st = repeat([nothing], 3)

for case in (6, 7,)
# for case in (1, 2, 3, 5, 6, 7)
    cc = mod1(case, 4)

    prob = AdvectionDiffusion1D(cs[cc], νs[cc])
    modelfile = joinpath(modeldir, "project$(case)", modelfilename)

    # global (NN, p, st), _, _ = ngProject(prob, datafile, modeldir, makemodel, case; rng, train_params, device)
    (Xd, Td, Ud, Up, ps), _ = ngEvolve(prob, datafile, modelfile, case; rng, data_kws, evolve_params, device)

    # global XD = (XD..., Xd)
    # global TD = (TD..., Td)
    # global UD = (UD..., Ud)
    # global UP = (UP..., Up)
    # global PS = (PS..., ps)
    # sleep(2)
end

#======================================================#
nothing
