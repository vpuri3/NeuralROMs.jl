module GeometryLearning

using DocStringExtensions

# PDE stack
using FourierSpaces
using FourierSpaces: linspace

# ML Stack
using Lux
using Zygote
using Optimisers
using ComponentArrays

using Random
using LinearAlgebra

# vis stack
using Plots
using Colors

# serialization
using BSON

include("layers.jl")
include("train.jl")

export
       # utils
       linspace,

       # layers
       Atten,
       Diag,

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
