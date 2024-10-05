#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

datafile  = joinpath(@__DIR__, "data_burg1D", "data.jld2")
device = gpu_device()
device = cpu_device()

data_kws = (; Ix = :, It = :)

#------------------------------------------------------#
# DNN
#------------------------------------------------------#
# train_params = (; E = 140)
# evolve_params = (;)
#
# makemodel = makemodelDNN
# modelfilename = "model_08.jld2"
# modeldir  = joinpath(@__DIR__, "dump_dnn")

#------------------------------------------------------#
# Gaussian
#------------------------------------------------------#
# data_kws = (; Ix = LinRange(1, 8192, 64), It = :)
#
# train_params  = (; E = 100, Ng = 1, Nf = 1, train_freq = false)
# evolve_params  = (; scheme = :GalerkinCollocation)
# # evolve_params = (; timealg = RungeKutta4(), Δt = 1e-4, adaptive = false)
# # evolve_params = (; timealg = EulerForward(), Δt = 1e-4, adaptive = false)
#
# makemodel = makemodelGaussian
# modelfilename = "model_05.jld2"
# modeldir  = joinpath(@__DIR__, "dump_gaussian")
#
#------------------------------------------------------#
# Tanh with adaptive training
#------------------------------------------------------#
train_params = (;)
evolve_params = (;)

makemodel = makemodelKernel
modelfilename = "model.jld2"
modeldir  = joinpath(@__DIR__, "dump")

#------------------------------------------------------#
# Tanh kernels
#------------------------------------------------------#
# # data_kws = (; Ix = LinRange(1, 512, 64), It = LinRange(1, 500, 500))
# # data_kws = map(x -> round.(Int, x), data_kws)
#
# train_params  = (; N = 1, Nsplits = 0)
# evolve_params  = (; scheme = :GalerkinCollocation)
# # evolve_params  = (; scheme = :GalerkinCollocation, timealg = Tsit5())
# # evolve_params = (; timealg = EulerForward())#, Δt = 1e-3, adaptive = false)
# # evolve_params = (; timealg = RungeKutta4())#, Δt = 1e-3, adaptive = false)
#
# makemodel = makemodelTanh
# modelfilename = "model_05.jld2"
# modeldir  = joinpath(@__DIR__, "dump_tanh_orig")

#------------------------------------------------------#
# Tanh kernels (with boosting)
#------------------------------------------------------#
# train_params  = (; N = 1, Nsplits = 0, Nboosts = 5)
# evolve_params  = (; scheme = :GalerkinCollocation)
#
# makemodel = makemodelTanh
# modelfilename = joinpath("boost$(train_params.Nboosts)","model_05.jld2")
# modeldir  = joinpath(@__DIR__, "dump_tanh_boost")

#------------------------------------------------------#
# Tanh kernels (with splitting)
#------------------------------------------------------#
# IX = Colon()
# # IX = LinRange(1, 8192, 1024) .|> Base.Fix1(round, Int)
#
# train_params  = (; N = 1, Nsplits = 2)
# evolve_params  = (; scheme = :GalerkinCollocation, IX)
#
# makemodel = makemodelTanh
# modelfilename = joinpath("split$(train_params.Nsplits)","model_05.jld2")
# modeldir  = joinpath(@__DIR__, "dump_tanh_split")

#------------------------------------------------------#
# Evolve
#------------------------------------------------------#

# parameters
XD = TD = UD = UP = PS = ()
NN, p, st = repeat([nothing], 3)

# for case in (1, 2, 3,)
# for case in (1,)
for case in (2,)
    cc = mod1(case, 4)
    prob = BurgersViscous1D(1f-4)
    modelfile = joinpath(modeldir, "project$(case)", modelfilename)

    global (NN, p, st), _, _ = ngProject(prob, datafile, modeldir, makemodel, case; rng, train_params, device)
    # (Xd, Td, Ud, Up, ps), _ = ngEvolve(prob, datafile, modelfile, case; rng, data_kws, evolve_params, device)

    # global XD = (XD..., Xd)
    # global TD = (TD..., Td)
    # global UD = (UD..., Ud)
    # global UP = (UP..., Up)
    # global PS = (PS..., ps)
    # sleep(2)
end

#======================================================#
#
# ARCHITECTURE
# - Check out multiplicative feature networks.
#   Maybe they can speed-up SDF type problems.
#
# GAUSSIAN REFINEMENT/CULLING
# - 
#
# TANH KERNELS
# - intead of/ along with a global shift, have a localized shift by forming
#   a plateau with tanh.
#   The plateau can degrade to 0, or produce sharper features
#
#
# HYPER-REDUCTION
# - Each Gaussian needs ~5 points to be evolved properly. This should be
#   helpful in hyper-reduction. We should do local sampling around each
#   Gaussian. That is: uniformly pick 5 x ∈ [x̄ - 2σ, x̄ + 2σ]
#
# LITERATURE
# - Check out Gaussian process literature
#
# NEW CONTRIB
# - Make parameterization probabilistic. Then you get UQ for free.
#======================================================#
nothing
