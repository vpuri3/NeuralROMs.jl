#
"""
Learn solution to diffusion equation

    -∇⋅ν∇ u = f₀

for variable ν, and constant f₀

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
function datagen(rng, V, discr, K, f0 = nothing)

    N = size(V, 1)

    # constant forcing
    f0 = isnothing(f0) ? ones(Float32, N) : f0
    f0 = kron(f0, ones(K)')

    ν = 1 .+ 1 * rand(rng, Float32, N, K)

    @assert size(f0) == size(ν)

    V = make_transform(V, ν)
    F = transformOp(V)

    # rm high freq modes
    Tr = truncationOp(V, (0.5,))
    ν  = Tr * ν
    f0  = Tr * f0

    # true sol
    A = diffusionOp(ν, V, discr)
    u = A \ f0
    u = u

    V, ν, u
end

""" main program """

# parameters
N = 128    # problem size
K = 100    # X-samples
E = 200  # epochs

# function main(N, K, E)
#     return NN, p, model
# end

rng = Random.default_rng()
Random.seed!(rng, 0)

# space discr
V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
discr = Collocation()

# datagen
Random.seed!(rng, 917)

f0 = 20 * rand(Float32, N)
_V, _ν, _u = datagen(rng, V, discr, K, f0) # train
V_, ν_, u_ = datagen(rng, V, discr, K, f0) # test

# model setup
NN = Lux.Chain(
    Lux.Dense(N , N, tanh),
    Lux.Dense(N , N, tanh),
    Lux.Dense(N , N, tanh),
    Lux.Dense(N , N),
)

p, st = Lux.setup(rng, NN)
p = p |> ComponentArray

model, loss, stats = model_setup(NN, st) # stateless models

# set up stats, loss, callback
_stats = (p) -> stats(p, _ν, _u)
_loss  = (p) -> loss( p, _ν, _u)

stats_ = (p) -> stats(p, ν_, u_)
loss_  = (p) -> loss( p, ν_, u_)

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
@time p = train(_loss, p, Int(E*0.05); opt = Optimisers.Adam(1f-2), cb = CB)
@time p = train(_loss, p, Int(E*0.70); opt = Optimisers.Adam(1f-3), cb = CB)
@time p = train(_loss, p, Int(E*0.20); opt = Optimisers.Adam(1f-4), cb = CB)

# stats after training
cb(p)

# visualization
plt = plot_training(ITER, _LOSS, LOSS_)
plts = visualize(V, (_ν, _u), (ν_, u_), model, p)
#
