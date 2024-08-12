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
# evolve_params = (; T = Float32,)
#
# makemodel = makemodelDNN
# modelfilename = "model_08.jld2"
# modeldir  = joinpath(@__DIR__, "dump_dnn")

#------------------------------------------------------#
# Gaussian
#------------------------------------------------------#
# data_kws = (; Ix = LinRange(1, 256, 64), It = LinRange(1, 500, 500))
# data_kws = map(x -> round.(Int, x), data_kws)

train_params  = (;)
evolve_params = (; T = Float64, timealg = RK4())#, Δt = 1e-3, adaptive = false)

makemodel = makemodelGaussian
modelfilename = "model_05.jld2"
modeldir  = joinpath(@__DIR__, "dump_gaussian")

#------------------------------------------------------#
# Gaussian (exact)
#------------------------------------------------------#
# data_kws = (; Ix = LinRange(1, 256, 64), It = LinRange(1, 500, 500))
# data_kws = map(x -> round.(Int, x), data_kws)

# train_params  = (; N = 1, exactIC = (; c = [1.0], x̄ = [-0.5], σ = [0.1]))
# evolve_params = (; T = Float64, timealg = RK4(),)

# makemodel = makemodelGaussian
# modelfilename = "model.jld2"
# modeldir  = joinpath(@__DIR__, "dump_gaussian_exact")

#------------------------------------------------------#
# MFN (Fourier)
#------------------------------------------------------#
# train_params = (; MFNfilter = :Fourier)
# evolve_params = (; T = Float32, timealg = RK4())
#
# makemodel = makemodelMFN
# modelfilename = "model_08.jld2"
# modeldir  = joinpath(@__DIR__, "dump_mfn_fourier")

#------------------------------------------------------#
# MFN (Gabor)
#------------------------------------------------------#
# train_params = (; MFNfilter = :Gabor, γ = 0f0)
# evolve_params = (; T = Float32,)
#
# makemodel = makemodelMFN
# modelfilename = "model_08.jld2"
# modeldir  = joinpath(@__DIR__, "dump_mfn_gabor")

#------------------------------------------------------#
# Evolve
#------------------------------------------------------#

# parameters
cs = evolve_params.T[1, 0, 1]
νs = evolve_params.T[0, 0.01, 0.01]

XD = TD = UD = UP = PS = ()
NN, p, st = repeat([nothing], 3)

# for case in 1:1
# for case in 4:4
# for case in 1:3
# for case in 4:6
for case in 4:4
    cc = mod1(case, 3)

    prob = AdvectionDiffusion1D(cs[cc], νs[cc])
    modelfile = joinpath(modeldir, "project$(case)", modelfilename)

    global (NN, p, st), _, _ = ngProject(prob, datafile, modeldir, makemodel, case; rng, train_params, device)
    # (Xd, Td, Ud, Up, ps), _ = ngEvolve(prob, datafile, modelfile, case; rng, data_kws, evolve_params, device)
    #
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
