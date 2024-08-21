#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

datafile  = joinpath(@__DIR__, "data_burg1D", "data.jld2")
device = Lux.gpu_device()

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
# data_kws = map(x -> round.(Int, x), data_kws)

# train_params  = (; E = 100,)
# # evolve_params  = (; scheme = :GalerkinCollocation)
# # evolve_params = (; timealg = RungeKutta4(), Δt = 1e-4, adaptive = false)
# # evolve_params = (; timealg = EulerForward(), Δt = 1e-4, adaptive = false)
#
# makemodel = makemodelGaussian
# modelfilename = "model_05.jld2"
# modeldir  = joinpath(@__DIR__, "dump_gaussian")

#------------------------------------------------------#
# RSWAF
#------------------------------------------------------#
# data_kws = (; Ix = LinRange(1, 512, 64), It = LinRange(1, 500, 500))
# data_kws = map(x -> round.(Int, x), data_kws)

train_params  = (; type = :RSWAF)
evolve_params  = (; scheme = :GalerkinCollocation)
# evolve_params = (; timealg = RungeKutta4())#, Δt = 1e-3, adaptive = false)

makemodel = makemodelGaussian
modelfilename = "model_05.jld2"
modeldir  = joinpath(@__DIR__, "dump_rswaf")

#------------------------------------------------------#
# Evolve
#------------------------------------------------------#

# parameters
XD = TD = UD = UP = PS = ()
NN, p, st = repeat([nothing], 3)

# for case in 1:1
for case in 1:6
    cc = mod1(case, 4)

    prob = BurgersViscous1D(1f-4)
    # prob = AdvectionDiffusion1D(1f0, 0f-1)
    modelfile = joinpath(modeldir, "project$(case)", modelfilename)

    global (NN, p, st), _, _ = ngProject(prob, datafile, modeldir, makemodel, case; rng, train_params, device)
    (Xd, Td, Ud, Up, ps), _ = ngEvolve(prob, datafile, modelfile, case; rng, data_kws, evolve_params, device)

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
# RSWAF
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
