#
using GeometryLearning

# PDE stack
using LinearAlgebra, FourierSpaces

# ML stack
using Lux, Random, Optimisers, MLUtils

# vis/analysis, serialization
using Plots, BSON

# accelerator
using CUDA, LuxCUDA #, KernelAbstractions
CUDA.allowscalar(false)
import Lux: cpu, gpu

# misc
using Tullio, Zygote

using FFTW, LinearAlgebra
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    FFTW.set_num_threads(nt)
end

rng = Random.default_rng()
Random.seed!(rng, 199)

w = 64 # width
l = 32 # latent
act = relu

NN = Chain(
    Conv((9,), 1 => 1, act; stride = 5, pad = 0),
    Conv((9,), 1 => 1, act; stride = 5, pad = 0),
    Conv((9,), 1 => 1, act; stride = 5, pad = 0),
    Conv((7,), 1 => 1, act; stride = 1, pad = 0),

    ConvTranspose((7,), 1 => 1, act; stride = 1, pad = 0), # 7
    ConvTranspose((10,), 1 => 1, act; stride = 5, pad = 0), # 40
    ConvTranspose((9,), 1 => 1, act; stride = 5, pad = 0), # 204
    ConvTranspose((9,), 1 => 1, act; stride = 5, pad = 0), # 1024
)


u = rand(Float32, 1024, 1, 10)
# u = rand(Float32, 1, 1, 10)
p, st = Lux.setup(rng, NN)
@show u |> size
@show NN(u, p, st)[1] |> size
