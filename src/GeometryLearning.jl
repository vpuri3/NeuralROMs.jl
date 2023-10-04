module GeometryLearning

using DocStringExtensions

# PDE stack
using FourierSpaces

# ML Stack
using Lux
using MLUtils
using Optimisers
import WeightInitializers: _nfan

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
include("vis.jl")
include("metrics.jl")
include("train.jl")

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

       # layers
       Atten,
       Diag,
       PermutedBatchNorm,
       SplitRows,
       ImplicitEncoderDecoder,

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
       mse,
       rsquare

end # module
