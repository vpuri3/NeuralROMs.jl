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
using Plots, BSON

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

    V, (ν, u)
end

""" main program """

function main(N, K, E; name = "f_fixed_seq2seq")

    # space discr
    V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
    discr = Collocation()

    # datagen
    f0 = 20 * rand(Float32, N)
    _V, _data = datagen(rng, V, discr, K, f0) # train
    V_, data_ = datagen(rng, V, discr, K, f0) # test

    # model setup
    NN = Lux.Chain(
        Lux.Dense(N , N, tanh),
        Lux.Dense(N , N, tanh),
        Lux.Dense(N , N, tanh),
        Lux.Dense(N , N),
    )

    p, st = Lux.setup(rng, NN)
    p = p |> ComponentArray

    _model, _loss, _stats = model_setup(NN, _data)
    model_, loss_, stats_ = model_setup(NN, data_)

    # analysis callback
    cb = (p, st) -> callback(p, st; _loss, _stats, loss_, stats_)
    cb(p, st)

    # training callback
    ITER  = Int[]
    _LOSS = Float32[]
    LOSS_ = Float32[]

    CB = (p, st, iter, maxiter) -> callback(p, st; _loss, _LOSS, loss_, LOSS_,
                                            ITER, iter, maxiter, step = 1)

    # training loop
    @time p, st = train(_loss, p, st, Int(E*0.05); opt = Optimisers.Adam(1f-5), cb = CB)
    @time p, st = train(_loss, p, st, Int(E*0.05); opt = Optimisers.Adam(1f-2), cb = CB)
    @time p, st = train(_loss, p, st, Int(E*0.70); opt = Optimisers.Adam(1f-3), cb = CB)
    @time p, st = train(_loss, p, st, Int(E*0.20); opt = Optimisers.Adam(1f-4), cb = CB)

    # print stats
    cb(p, st)

    # visualization
    plt_train = plot_training(ITER, _LOSS, LOSS_)
    plts = visualize(V, _data, data_, NN, p, st)

    dir = @__DIR__
    png(plt_train, joinpath(dir, "plt_training"))
    png(plts[1],   joinpath(dir, "plt_traj_train"))
    png(plts[2],   joinpath(dir, "plt_traj_test"))
    png(plts[3],   joinpath(dir, "plt_r2_train"))
    png(plts[4],   joinpath(dir, "plt_r2_test"))

    model = NN, p, st
 
    BSON.@save joinpath(@__DIR__, "$name.bson") _data data_ model

    V, _data, data_, model
end

# parameters
N = 128    # problem size
K = 100    # X-samples
E = 200  # epochs

rng = Random.default_rng()
Random.seed!(rng, 917)

V, _data, data_, model = main(N, K, E)

_f, _u = _data
f_, u_ = data_
NN, p, st = model

nothing
#
