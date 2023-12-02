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
using CUDA
using CUDA: AbstractGPUArray
using KernelAbstractions

using FFTW
using NNlib
using Tullio
using ComponentArrays
using Setfield: @set!
using IterTools

using NonlinearSolve
using LineSearches

include("utils.jl")
include("vis.jl")
include("metrics.jl")

include("autodiff.jl")

include("train.jl")
include("nlsq.jl")

include("layers.jl")

include("transform.jl")
include("operator.jl")

export
    # vis
    animate1D,
    plot_1D_surrogate_steady,
    
    # utils
    # _ntimes,
    fix_kw,
    init_siren,
    scaled_siren_init,
    remake_ca,
    
    # autodiff
    forwarddiff_deriv1,
    forwarddiff_deriv2,
    forwarddiff_jacobian,

    finitediff_deriv1,
    finitediff_deriv2,
    finitediff_jacobian,

    # layers
    Atten,
    Diag,
    PermutedBatchNorm,
    SplitRows,
    ImplicitEncoderDecoder,
    AutoDecoder,
    HyperNet,

    # transforms
    FourierTransform,
    CosineTransform,

    # operator layers
    OpKernel,
    OpConv,

    OpKernelBilinear,
    OpConvBilinear,
    linear_nonlinear,

    # training
    train_model,
    callback,
    optimize,
    plot_training,
    
    # nlsq
    nlsq,
    
    # metrics
    mae,
    mse,
    pnorm,
    l2reg,
    rsquare

end # module
