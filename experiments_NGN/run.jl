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
train_params = (;)
evolve_params = (; T = Float32,)

makemodel = makemodelDNN
modeldir  = joinpath(@__DIR__, "dump_dnn")

#------------------------------------------------------#
# MFN (Fourier)
#------------------------------------------------------#
# train_params = (; MFNfilter = :Fourier)
# evolve_params = (; T = Float32, timealg = RK4())
#
# makemodel = makemodelMFN
# modeldir  = joinpath(@__DIR__, "dump_mfn_fourier")

#------------------------------------------------------#
# MFN (Gabor)
#------------------------------------------------------#
# train_params = (; MFNfilter = :Gabor, γ = 0f0)
# evolve_params = (; T = Float32,)
#
# makemodel = makemodelMFN
# modeldir  = joinpath(@__DIR__, "dump_mfn_gabor")

#------------------------------------------------------#
# Gaussian
#------------------------------------------------------#
# data_kws = (; Ix = LinRange(1, 256, 64), It = LinRange(1, 500, 500))
# data_kws = map(x -> round.(Int, x), data_kws)
#
# train_params  = (;)
# evolve_params = (; T = Float64, timealg = RK4(),)
#
# makemodel = makemodelGaussian
# modeldir  = joinpath(@__DIR__, "dump_gaussian")

#------------------------------------------------------#
# Evolve
#------------------------------------------------------#

# parameters
cs = evolve_params.T[1, 0, 1]
νs = evolve_params.T[0, 0.01, 0.01]

XD = TD = UD = UP = PS = ()
NN, p, st = repeat([nothing], 3)

# for case in 4:4# 1:6
for case in 1:6
    cc = mod1(case, 3)

    prob = AdvectionDiffusion1D(cs[cc], νs[cc])
    modelfile = joinpath(modeldir, "project$(case)", "model_08.jld2")

    global (NN, p, st), _, _ = ngProject(prob, datafile, modeldir, makemodel, case; rng, train_params, device)
    (Xd, Td, Ud, Up, ps), _ = ngEvolve(prob, datafile, modelfile, case; rng, data_kws, evolve_params, device)

    global XD = (XD..., Xd)
    global TD = (TD..., Td)
    global UD = (UD..., Ud)

    global UP = (UP..., Up)
    global PS = (PS..., ps)

    sleep(2)
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
