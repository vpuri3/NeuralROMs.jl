#
using GeometryLearning

# PDE stack
using FourierSpaces, LinearAlgebra

# ML stack
using Lux, Random, Optimisers

# vis/analysis, serialization
using Plots, BSON

# accelerator
using CUDA, KernelAbstractions
CUDA.allowscalar(false)
import Lux: cpu, gpu

# misc
using Tullio, Zygote

using FFTW, LinearAlgebra
BLAS.set_num_threads(2)
FFTW.set_num_threads(8)

function main()

    rng = Random.default_rng()
    Random.seed!(rng, 117)

    N = 128
    E = 100

    w1 = 32
    w2 = 16
    wo = 2
    m = (24,)
    K = 1000

    # dev = Lux.cpu
    dev = Lux.gpu

    # NN = OpConvBilinear(w1, w2, wo, m)
    # data = ((rand(w1, N, K), rand(w2, N, K)), rand(wo, N, K)) |> dev

    # NN = OpConv(w1, w2, m)
    # NN = OpKernel(w1, w2, m)
    # NN = Dense(w1, w2)
    data = (rand(w1, N, K), rand(w2, N, K)) |> dev

    p, st = Lux.setup(rng, NN) |> dev
    _, _loss, _ = model_setup(NN, data)
    cb = (p, st, iter, maxiter) -> GeometryLearning.callback(p, st; _loss, iter, maxiter, step = 1)

    CUDA.@time p, st, _ = optimize(_loss, p, st, E; cb)
    # CUDA.@profile CUDA.@time p, st, _ = optimize(_loss, p, st, E; cb)

    (NN, p, st) |> cpu
end

model = main()

nothing

