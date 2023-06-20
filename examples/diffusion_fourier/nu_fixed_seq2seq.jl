#
"""
Learn solution to diffusion equation

    -∇⋅ν₀∇ u = f

for constant ν₀, and variable f

test bed for Fourier Neural Operator experiments where
forcing is learned separately.
"""

using GeometryLearning

# PDE stack
using FourierSpaces, LinearAlgebra # PDE

# ML stack
using Zygote, Lux, Random, ComponentArrays, Optimisers

# vis
using Plots

""" data """
function datagen(rng, V, discr, K, ν0 = nothing)

    N = size(V, 1)

    # constant viscosity
    ν0 = isnothing(ν0) ? ones(Float32, N) : ν0
    ν0 = kron(ν0, ones(K)')

    f = 20 * rand(rng, Float32, N, K)

    @assert size(f) == size(ν0)

    V = make_transform(V, f)
    F = transformOp(V)

    # rm high freq modes
    Tr = truncationOp(V, (0.5,))
    ν0 = Tr * ν0
    f  = Tr * f

    # true sol
    A = diffusionOp(ν0, V, discr)
    u = A \ f

    V, f, u
end

""" main program """

# parameters
N = 128    # problem size
K = 100    # X-samples
E = 1000  # epochs

rng = Random.default_rng()
Random.seed!(rng, 0)

# space discr
V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
discr = Collocation()

# datagen
Random.seed!(rng, 917)

ν0 = 1 .+ 1 * rand(Float32, N)
_V, _f, _u = datagen(rng, V, discr, K, ν0) # train
V_, f_, u_ = datagen(rng, V, discr, K, ν0) # test

# model setup
NN = Lux.Chain(
    Lux.Dense(N , N, tanh),
    Lux.Dense(N , N, tanh),
    Lux.Dense(N , N, tanh),
    Lux.Dense(N , N, tanh),
    Lux.Dense(N , N),
)

p, st = Lux.setup(rng, NN)
p = p |> ComponentArray

model, loss, stats = model_setup(NN, st) # stateless models

# train setup
_stats = (p) -> stats(p, _f, _u)
_loss  = (p) -> loss( p, _f, _u)

stats_ = (p) -> stats(p, f_, u_)
loss_  = (p) -> loss( p, f_, u_)

cb = (p) -> callback(p; _loss, _stats, loss_, stats_)

# print stats before training
cb(p)

# training callback
ITER  = Int[]
_LOSS = Float32[]
LOSS_ = Float32[]

CB = (p, iter, maxiter) -> callback(p; _loss, _LOSS, loss_, LOSS_,
                                    ITER, iter, maxiter, step = 1)

# training loop
@time p = train(_loss, p, Int(E*0.05); opt = Optimisers.Adam(1f-5), cb = CB)
@time p = train(_loss, p, Int(E*0.05); opt = Optimisers.Adam(1f-3), cb = CB)
@time p = train(_loss, p, Int(E*0.70); opt = Optimisers.Adam(1f-4), cb = CB)
@time p = train(_loss, p, Int(E*0.25); opt = Optimisers.Adam(1f-5), cb = CB)

# test stats
cb(p)

# visualization
plt = plot_training(ITER, _LOSS, LOSS_)
plts = visualize(V, (_f, _u), (f_, u_), model, p)
#
