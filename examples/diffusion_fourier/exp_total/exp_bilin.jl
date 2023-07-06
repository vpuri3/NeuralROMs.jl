#
"""
Learn solution to diffusion equation

    -∇⋅ν∇u = f

for constant ν₀, and variable f

test bed for Fourier Neural Operator experiments where
forcing is learned separately.
"""

using GeometryLearning

# PDE stack
using FourierSpaces, LinearAlgebra

# ML stack
using Lux, Random, Optimisers

# vis/analysis, serialization
using Plots, BSON

include("../datagen.jl")

# parameters
N  = 128  # problem size
K1 = 25   # ν-samples
K2 = 25   # f-samples
E  = 200  # epochs

rng = Random.default_rng()
Random.seed!(rng, 123)

# datagen
_V, _data, _, _ = datagen(rng, N, K1, K2) # train
V_, data_, _, _ = datagen(rng, N, K1, K2) # test

###
# nonlienar FNO model
###
if true
            
__data = combine_data(_data)
data__ = combine_data(data_)

w = 16    # width
m = (32,) # modes
c = size(__data[1], 1) # in  channels
o = size(__data[2], 1) # out channels

NN = Lux.Chain(
    PermutedBatchNorm(c, 3),
    Dense(c , w, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    Dense(w , o)
)

opt = Optimisers.Adam()
learning_rates = (1f-2, 1f-3, 1f-3)
maxiters  = E .* (0.10, 0.20, 0.70) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_nonlin")
mkpath(dir)

FNO_nonlin = train_model(rng, NN, __data, data__, _V, opt;
                        learning_rates, maxiters, dir, cbstep = 1)

end

###
# linear FNO model
###

if true

__data = combine_data(_data)
data__ = combine_data(data_)

w = 16    # width
m = (32,) # modes
c = size(__data[1], 1) # in  channels
o = size(__data[2], 1) # out channels

NN = Lux.Chain(
    Dense(c , w),
    OpKernel(w, w, m),
    Dense(w , o)
)

opt = Optimisers.Adam()
learning_rates = (1f-2, 1f-3, 1f-3)
maxiters  = E .* (0.10, 0.20, 0.70) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_linear")
mkpath(dir)

FNO_linear = train_model(rng, NN, __data, data__, _V, opt;
                        learning_rates, maxiters, dir, cbstep = 1)
end

###
# Bilinear (linear / nonlin) model
###

if true
    
__data = split_data(_data)
data__ = split_data(data_)

w1 = 16    # width nonlin
w2 = 16    # width linear
m = (32,) # modes
c1 = size(__data[1][1], 1) # in  channel nonlin
c2 = size(__data[1][2], 1) # in  channel linear
o  = size(__data[2]   , 1) # out channel

nonlin = Chain(PermutedBatchNorm(c1, 3), Dense(c1, w1, tanh), OpKernel(w1, w1, m, tanh))
linear = Dense(c2, w2)
bilin  = OpConvBilinear(w1, w2, o, m)
# bilin  = OpKernelBilinear(w1, w2, o, m) # errors
# bilin  = Bilinear((w1, w2) => o)

NN = linear_nonlinear(nonlin, linear, bilin)

opt = Optimisers.Adam()
learning_rates = (1f-3,)
maxiters  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_linear_nonlinear")
mkpath(dir)

FNO_linear_nonlinear = train_model(rng, NN, __data, data__, _V, opt;
                           learning_rates, maxiters, dir, cbstep = 1)

end

###
# Bilinear (linear / linear) model
###

if true
    
__data = split_data(_data)
data__ = split_data(data_)

w1 = 16    # width nonlin
w2 = 16    # width linear
m = (32,) # modes
c1 = size(__data[1][1], 1) # in  channel nonlin
c2 = size(__data[1][2], 1) # in  channel linear
o  = size(__data[2]   , 1) # out channel

nonlin = Chain(PermutedBatchNorm(c1, 3), Dense(c1, w1), OpKernel(w1, w1, m))
linear = Dense(c2, w2)
bilin  = OpConvBilinear(w1, w2, o, m)

NN = linear_nonlinear(nonlin, linear, bilin)

opt = Optimisers.Adam()
learning_rates = (1f-3,)
maxiters  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "exp_FNO_linear_linear")
mkpath(dir)

FNO_linear_linear = train_model(rng, NN, __data, data__, _V, opt;
                           learning_rates, maxiters, dir, cbstep = 1)

end
################
# Visualization
################

if true
    ST_linear = FNO_linear[2]
    ST_nonlin = FNO_nonlin[2]
    ST_linear_linear = FNO_linear_linear[2]
    ST_linear_nonlinear = FNO_linear_nonlinear[2]

    plt = plot(title = "Training Plot", yaxis = :log,
               xlabel = "Epochs", ylabel = "Loss (MSE)",
               ylims = (10^-2, 10^5),
    )

    plot!(plt, ST_linear[1], ST_linear[2], w = 2.0, c = :green, s = :solid, label = "FNO linear")
    plot!(plt, ST_linear[1], ST_linear[3], w = 2.0, c = :green, s = :dash, label = nothing)

    plot!(plt, ST_nonlin[1], ST_nonlin[2], w = 2.0, c = :black, s = :solid, label = "FNO nonlin")
    plot!(plt, ST_nonlin[1], ST_nonlin[3], w = 2.0, c = :black, s = :dash, label = nothing)

    plot!(plt, ST_linear_linear[1], ST_linear_linear[2], w = 2.0, c = :blue, s = :solid, label = "Bilinear FNO (linear, linear)")
    plot!(plt, ST_linear_linear[1], ST_linear_linear[3], w = 2.0, c = :blue, s = :dash, label = nothing)

    plot!(plt, ST_linear_nonlinear[1], ST_linear_nonlinear[2], w = 2.0, c = :red, s = :solid, label = "Bilinear FNO (linear, nonlinear)")
    plot!(plt, ST_linear_nonlinear[1], ST_linear_nonlinear[3], w = 2.0, c = :red, s = :dash, label = nothing)

    png(plt, joinpath(@__DIR__, "plt_train_bilin"))
end

nothing
#
