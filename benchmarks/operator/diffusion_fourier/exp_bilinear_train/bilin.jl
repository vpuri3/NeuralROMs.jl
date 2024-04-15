#
"""
Learn solution to diffusion equation

    -∇⋅ν∇u = f

for constant ν₀, and variable f

test bed for Fourier Neural Operator experiments where
forcing is learned separately.
"""

using NeuralROMs

# PDE stack
using FourierSpaces, LinearAlgebra

# ML stack
using Lux, Random, Optimisers

# vis/analysis, serialization
using Plots, BSON

using FFTW, LinearAlgebra
BLAS.set_num_threads(4)
FFTW.set_num_threads(4)

include("../datagen.jl")

# parameters
N  = 128  # problem size
K1 = 32   # ν-samples
K2 = 32   # f-samples
E  = 200  # epochs

rng = Random.default_rng()
Random.seed!(rng, 117)

# datagen
_V, _data, _, _ = datagen(rng, N, K1, K2) # train
V_, data_, _, _ = datagen(rng, N, K1, K2; mode = :test) # test

# Bilinear (linear / nonlin) model

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

NN = linear_nonlinear(nonlin, linear, bilin)


###
# ADAM
###

if false

opt = Optimisers.Adam()
learning_rates = (1f-3,)
maxiters  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "opt_adam")

_, ADAM = train_model(rng, NN, __data, data__, _V, opt;
    learning_rates, maxiters, dir, cbstep = 1)

end


###
# ADAM
###

if true

# opt = Optimisers.Adam(1f0, (0.9f0, 0.999f0))
opt = Optimisers.Adam(1f0, (0.89f0, 0.900f0))
learning_rates = (1f-3,)
maxiters  = E .* (1.00,) .|> Int
dir = joinpath(@__DIR__, "opt_adam_modified")

_, ADAMOPT = train_model(rng, NN, __data, data__, _V, opt;
        learning_rates, maxiters, dir, cbstep = 1)

end

###
# DESCENT
###

if false
    
opt = Optimisers.Descent()
learning_rates = (1f-8, 1f-3,)
maxiters  = E .* (0.10, 0.90,) .|> Int
dir = joinpath(@__DIR__, "opt_desc")

_, DESC = train_model(rng, NN, __data, data__, _V, opt;
    learning_rates, maxiters, dir, cbstep = 1)

end

###
# DESCENT + ADAM
###

if false

opt = Optimisers.Descent()
learning_rates = (1f-8,)
maxiters  = E .* (0.20,) .|> Int
dir = joinpath(@__DIR__, "opt_AD_1")

(_, p, st), AD_1 = train_model(rng, NN, __data, data__, _V, opt;
    learning_rates, maxiters, dir, cbstep = 1)

opt = Optimisers.Adam()
learning_rates = (1f-4,)
maxiters  = E .* (0.80,) .|> Int
dir = joinpath(@__DIR__, "opt_AD_2")

(_, p, st), AD_2 = train_model(rng, NN, __data, data__, _V, opt;
    learning_rates, maxiters, dir, cbstep = 1, p, st)

end

################
# Visualization
################

if false

    plt = plot(title = "Training Plot", yaxis = :log,
               xlabel = "Epochs", ylabel = "Loss (MSE)",
    )

    plot!(plt, ADAM[1], ADAM[2], w = 2.0, c = :black, label = "Adam")
    plot!(plt, DESC[1], DESC[3], w = 2.0, c = :red  , label = "Descent")

    png(plt, joinpath(@__DIR__, "plt_train_bilin"))
end

nothing
#
