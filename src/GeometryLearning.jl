module GeometryLearning

# Write your package code here.

using Reexport
using FourierSpaces: FourierSpaces, linspace
using Random
using LinearAlgebra
using Lux

export linspace
export Atten, Diag

include("layers.jl")

end # module
