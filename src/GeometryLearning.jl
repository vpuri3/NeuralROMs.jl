module GeometryLearning

# PDE stack
using FourierSpaces
using FourierSpaces: linspace

# ML Stack
using Lux
using Zygote
using Optimisers

using Random
using LinearAlgebra

# vis stack
using Plots
using Colors

include("layers.jl")
include("train.jl")

export
       # utils
       linspace,

       # layers
       Atten,
       Diag,

       # training
       model_setup,
       callback,
       train,
       plot_training,
       visualize,
       mse,
       rsquare


end # module
