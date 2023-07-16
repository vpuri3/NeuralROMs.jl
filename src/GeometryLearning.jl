module GeometryLearning

using DocStringExtensions

# PDE stack
using FourierSpaces
using FourierSpaces: linspace

# ML Stack
using Lux
using Optimisers

# autodiff
using Zygote
using ChainRulesCore

using Random
using LinearAlgebra

# vis stack
using Plots
using Colors

# serialization
using BSON

# GPU stack
using CUDA
using CUDA: AbstractGPUArray
using KernelAbstractions

using FFTW
using NNlib
using Tullio
using ComponentArrays
using Setfield: @set!

include("utils.jl")
include("metrics.jl")
include("train.jl")

include("layers.jl")

include("transform.jl")
include("operator.jl")

export
       # utils
       linspace,

       # layers
       Atten,
       Diag,
       PermutedBatchNorm,

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
       model_setup,
       callback,
       optimize,

       plot_training,
       visualize,
       mse,
       rsquare

end # module
