module NeuralROMs

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
using SparseArrays: sparse

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
using LinearSolve
using NonlinearSolve
using LineSearches

abstract type AbstractNeuralModel end
abstract type AbstractPDEProblem end
abstract type AbstractTimeAlg end
abstract type AbstractTimeIntegrator end
abstract type AbstractSolveScheme end

export AbstractNeuralModel, AbstractPDEProblem, AbstractTimeAlg,
    AbstractTimeIntegrator, AbstractSolveScheme

include("utils.jl")
export init_siren, scale_init, scaled_siren_init

include("metrics.jl")
export mae, mae_clamped, mse, rsquare, pnorm, elasticreg, codereg,
    regularize_autodecoder

include("autodiff.jl")
export
    forwarddiff_deriv1, forwarddiff_deriv2, forwarddiff_deriv4, forwarddiff_jacobian,
    finitediff_deriv1, finitediff_deriv2, finitediff_deriv4, finitediff_jacobian

include("layers.jl")
export
    Atten, Diag, PermutedBatchNorm, SplitRows, ImplicitEncoderDecoder,
    HyperNet,
    AutoDecoder, get_autodecoder,
    FlatDecoder, get_flatdecoder, freeze_decoder,
    HyperDecoder, get_hyperdecoder

include("optimisers.jl")
export DecoderWeightDecay, IdxWeightDecay

include("transform.jl")
export FourierTransform, CosineTransform

include("operator.jl")
export OpKernel, OpConv, OpKernelBilinear, OpConvBilinear, linear_nonlinear

include("neuralmodel.jl")
export
    normalizedata, unnormalizedata,
    NeuralModel,
    dudx1_1D, dudx2_1D, dudx4_1D,
    dudx1_2D, dudx2_2D # , dudx4_2D,
    dudp

include("neuralgridmodel.jl")

include("problems.jl")
export dudtRHS
export
    Advection1D, Advection2D,
    AdvectionDiffusion1D,
    BurgersInviscid1D,
    BurgersViscous1D, BurgersViscous2D,
    KuramotoSivashinsky1D

include("nonlinleastsq.jl")
export nonlinleastsq

include("timeintegrator.jl")
export TimeIntegrator, perform_timestep!, evolve_integrator!, evolve_model

include("evolve.jl")
export
    # timestepper types
    EulerForward, EulerBackward, RK2, RK4,
    # timestepper interface
    compute_residual, apply_timestep,
    # solve scheme types
    GalerkinProjection, LeastSqPetrovGalerkin,
    # residual functions
    make_residual, residual_learn

include("train.jl")
export train_model, callback, optimize, plot_training

include("vis.jl")
export
    animate1D, animate2D, meshplt,
    plot_derivatives1D, plot_derivatives1D_autodecoder,
    plot_1D_surrogate_steady

end # module
