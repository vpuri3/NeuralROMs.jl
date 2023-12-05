module GeometryLearning

using DocStringExtensions

# PDE stack
using FourierSpaces

# ML Stack
using Lux
using MLUtils
using Optimisers
using Optimization
using OptimizationOptimJL
using ParameterSchedulers
import WeightInitializers: _nfan

# autodiff
using Zygote
using ChainRulesCore

using FiniteDiff
using ForwardDiff
using ForwardDiff: Dual, Partials, value, partials

using Random
using LinearAlgebra

# visualization
using Plots
using Colors
using ProgressMeter

# serialization
using JLD2

# GPU stack
using Adapt
using CUDA
using CUDA: AbstractGPUArray
using KernelAbstractions

# numerical
using FFTW
using NNlib
using Tullio

# data management
using ComponentArrays
using Setfield: @set!
using UnPack
using ConcreteStructs
using IterTools

# linear/nonlinear solvers
using NonlinearSolve
using LineSearches

abstract type AbstractNeuralModel end
abstract type AbstractPDEProblem end
abstract type AbstractTimeStepper end

export AbstractNeuralModel, AbstractPDEProblem, AbstractTimeStepper

include("utils.jl")
export fix_kw, init_siren, scaled_siren_init, remake_ca #, _ntimes

include("metrics.jl")
export mae, mse, pnorm, l2reg, rsquare

include("autodiff.jl")
export forwarddiff_deriv1, forwarddiff_deriv2, forwarddiff_jacobian,
    finitediff_deriv1, finitediff_deriv2, finitediff_jacobian

include("layers.jl")
export Atten, Diag, PermutedBatchNorm, SplitRows, ImplicitEncoderDecoder,
    AutoDecoder, get_autodecoder, freeze_autodecoder,
    HyperNet, get_hyperdecoder

include("transform.jl")
export FourierTransform, CosineTransform

include("operator.jl")
export OpKernel, OpConv, OpKernelBilinear, OpConvBilinear, linear_nonlinear

include("neuralmodel.jl")
export normalizedata, unnormalizedata, NeuralSpaceModel, dudx1, dudx2, dudp

include("problems.jl")
export dudtRHS
export
    Advection1D, AdvectionDiffusion1D,
    BurgersInviscid1D, BurgersViscous1D

include("nonlinleastsq.jl")
export nonlinleastsq

include("evolve.jl")
export
    TimeIntegrator, update_integrator!, solve_timestep, perform_timestep!,
    make_residual, residual_learn,
    compute_residual, apply_timestep, timeinteg_isimplicit, timeinteg_order,
    EulerForward, EulerBackward,
    Galerkin, LeastSqPetrovGalerkin

include("train.jl")
export train_model, callback, optimize, plot_training

include("vis.jl")
export animate1D, plot_1D_surrogate_steady

end # module
