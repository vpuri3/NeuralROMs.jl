module GeometryLearning

# Write your package code here.

using FourierSpaces
using FourierSpaces: linspace
using Lux

using Random
using LinearAlgebra

include("layers.jl")

export
       # utils
       linspace,

       # layers
       Atten,
       Diag

end # module
