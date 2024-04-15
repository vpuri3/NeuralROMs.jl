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

include("../datagen.jl")

# parameters
N  = 128  # problem size
K1 = 25   # ν-samples
K2 = 25   # f-samples
E  = 200  # epochs

rng = Random.default_rng()
Random.seed!(rng, 123)

# datagen
_V, _data0, _data1, _ = datagen(rng, N, K1, K2) # train
V_, data0_, data1_, _ = datagen(rng, N, K1, K2) # test

###
# train on data0
###
if false

__data0 = combine_data(_data0)
data0__ = combine_data(data0_)

w = 16    # width
m = (32,) # modes
c = size(__data0[1], 1) # in  channels
o = size(__data0[2], 1) # out channels

NN0 = Lux.Chain(
    PermutedBatchNorm(c, 3),
    Dense(c , w, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    Dense(w , o)
)

opt = Optimisers.Adam()
learning_rates = (1f-2, 1f-3, 1f-3)
maxiters  = E .* (0.10, 0.20, 0.70) .|> Int
dir = joinpath(@__DIR__, "exp_split0")
mkpath(dir)

model0 = train_model(rng, NN0, __data0, data0__, _V, opt;
                     learning_rates, maxiters, dir, cbstep = 5)

NN0, p0, st0 = model0[1]

end

###
# train on data1
###
if true
    
__data1 = combine_data(_data1)
data1__ = combine_data(data1_)

w = 16    # width
m = (32,) # modes
c = size(__data1[1], 1) # in  channels
o = size(__data1[2], 1) # out channels

NN1 = Lux.Chain(
    PermutedBatchNorm(c, 3),
    Dense(c , w, tanh),
    OpKernel(w, w, m, tanh),
    OpKernel(w, w, m, tanh),
    Dense(w , o)
)

# E = 1000
# NN1 = OpConv(c, o, m)
# learning_rates = (1f-3,)
# maxiters  = E .* (1.00,) .|> Int

opt = Optimisers.Adam()
learning_rates = (1f-2, 1f-3, 1f-3)
maxiters  = E .* (0.10, 0.20, 0.70) .|> Int
dir = joinpath(@__DIR__, "exp_split1")
mkpath(dir)

model1 = train_model(rng, NN1, __data1, data1__, _V, opt;
                     learning_rates, maxiters, dir, cbstep = 5)

NN1, p1, st1 = model1[1]

end
#####

function make_data(data0, data1)

    x_, ν_, f_, u_ = data0
    f0 = _data1[3]
    @assert f0[:, 1] ≈ f0[:, end]
    f0 = f0[:, 1]

    d0 = (x_, ν_, f_ .- f0, u_)
    d1 = _data1

    D0 = combine_data(d0)
    D1 = combine_data(d1)

end

function NNsplit(data0, data1, model0, model1; make_data = nothing)

    isnothing(make_data) && @error()

    NN0, p0, st0 = model0
    NN1, p1, st1 = model1

    x0, x1 = make_data(data0, data1)

    y0, st0 = NN0(x0, p0, st0)[1]
    y1, st1 = NN1(x1, p1, st1)[1]

    y0 + y1
end

#####
nothing
#
