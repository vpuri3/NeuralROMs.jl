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
using Plots, BSON, Printf

using FFTW, LinearAlgebra
BLAS.set_num_threads(4)
FFTW.set_num_threads(4)

include("../datagen.jl")

# parameters
N  = 128  # problem size
K1 = 1    # ν-samples
K2 = 1024 # f-samples
E  = 200  # epochs

rng = Random.default_rng()
Random.seed!(rng, 213)

###
# TODO - generate linear data to assess performance
###

# datagen
_V, _data, _, _ = datagen(rng, N, K1, K2) # train
V_, data_, _, _ = datagen(rng, N, K1, K2) # test

###
# linear DL model
###

__data = combine_data(_data)
data__ = combine_data(data_)

w = 16 # width
c = size(__data[1], 1) # in  channels
o = size(__data[2], 1) # out channels

for (Opt, name, learning_rates, maxiters) in (
    (Adam,     "Adam"        , (1f-2,)      , E .* (1.00,      ) .|> Int),
    (Adam,     "Adam_warmup1", (1f-6, 1f-2,), E .* (0.10, 0.90,) .|> Int),
    (Adam,     "Adam_warmup2", (1f-4, 1f-2,), E .* (0.01, 0.99,) .|> Int),
    (Adam,     "Adam_warmup3", (1f-3, 1f-2,), E .* (0.05, 0.99,) .|> Int),
    (Momentum, "Momentum"    , (1f-6, 1f-2,), E .* (0.10, 0.90,) .|> Int),
    (Descent,  "Descent"     , (1f-6, 1f-2,), E .* (0.10, 0.90,) .|> Int),
)
    opt = Opt()

    global STATS = []
    global plt = plot(
        title = "Training Plot, $Opt Optimizer", yaxis = :log,
        xlabel = "Epochs", ylabel = "Loss (MSE)",
        palette = :tab10, legend = :topright,
        )

    cmap = range(HSV(0,1,1), stop=HSV(-360,1,1), length = 6 + 1)

    NN = Lux.Chain(
        Dense(c, w),
        Dense(w, o),
    )

    for L in 0: 5
        NN = Lux.Chain(
            Dense(c, w),
            (Dense(w, w) for _ in 1:L)...,
            Dense(w, o),
        )

        dir = joinpath(@__DIR__, "opt_$(name)_L$(L)")

        _, ST = train_model(rng, NN, __data, data__, _V, opt;
            learning_rates, maxiters, dir, cbstep = 1)

        push!(STATS, ST)

        plot!(plt, ST[1], ST[2], w = 2.0, label = "$L Layers")#, c = cmap[L+1])
    end

    if length(maxiters) >= 2
        LRs = Tuple(@sprintf "%.1e" lr for lr in learning_rates)
        vline!(plt, [maxiters[1:end-1]...], c = :black, w = 3.0, label = "LRs=$(LRs)")
    end
    png(plt, joinpath(@__DIR__, "plt_$name"))
end

plt

# nothing
#
