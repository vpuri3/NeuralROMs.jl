#
"""
Geometry learning sandbox

Want to solve `A(x) * u = M(x) * f` for varying `x` by
learning the mapping `x -> (A^{-1} M)(x)`.

# Reference
* https://slides.com/vedantpuri/project-discussion-2023-05-10

"""

using GeometryLearning
using SciMLOperators, LinearAlgebra, Random
using Zygote, Lux, ComponentArrays, Optimisers

""" data """
function datagen(rng, N, Kx, Ku)

    u = rand(rng, Float32, N, Ku)
    x = rand(rng, Float32, N, Kx)

    u0 = kron(u, ones(Kx)')
    x  = kron(ones(Ku)', x)

    # true sol
    D = DiagonalOperator(rand(rng, Float32, N))
    X = DiagonalOperator(x)

    Lt = X \ D * X \ D
    ut = Lt * u0

    x, u0, ut
end

""" model """
function model_setup(N, K)

    NN = Chain(
        Dense(N, N, tanh),
        Dense(N, N, tanh),
        Dense(N, N, tanh),
        Dense(N, N),
    )

    p, st = Lux.setup(rng, NN)
    p = p |> ComponentArray

    # wait for https://github.com/SciML/SciMLOperators.jl/pull/143
    # model_update_func(D, u, p, t; x = x) = NN(x, p, st)[1]
    # Lmodel = DiagonalOperator(zeros(N, K); update_func = model_update_func)
    # model = (p, x, u0) -> Lmodel(u0, p, 0.0; x = x) 

    function model(p, x, u0)
        d = NN(x, p, st)[1]
        D = DiagonalOperator(d)
        D * u0
    end

    function loss(p, x, u0, ut)
        upred = model(p, x, u0)

        norm(upred - ut, 2)
    end

    function errors(p, x, u0, ut)
        upred = model(p, x, u0)
        udiff = upred - ut

        meanAE = norm(udiff, 1) / length(ut)
        maxAE  = norm(udiff, Inf)

        meanAE, maxAE
    end

    model, p, loss, errors
end

function callback(iter, p, loss, errors)

    if iter % 100 == 0 || iter == 1

        (meanAE, maxAE) = isnothing(errors) ? (missing, missing) : errors(p)

        str  = string("Iter $iter: LOSS: ", round(loss(p), digits=8))
        str *= string(", meanAE: ", round(meanAE, digits=8))
        str *= string(", maxAE: " , round(maxAE , digits=8))

        println(str)
    end

    nothing
end

""" training """
function train(loss, p; opt = Optimisers.Adam(), E = 5_000, cb = nothing)

    # dry run
    l, pb = Zygote.pullback(loss, p)
    gr = pb(one.(l))[1]

    println("Loss with initial parameter set: ", l)

    # init optimizer
    opt_st = Optimisers.setup(opt, p)

    for iter in 1:E
        l, pb = Zygote.pullback(loss, p)
        gr = pb(one.(l))[1]

        opt_st, p = Optimisers.update(opt_st, p, gr)

        !isnothing(cb) && cb(iter, p)

    end

    p
end

""" main program """
function main(rng, N, Kx, Ku)
    K = Kx * Ku # samples

    model, p, loss, errors = model_setup(N, K)

    _x, _u0, _ut = datagen(rng, N, Kx, Ku) # train
    x_, u0_, ut_ = datagen(rng, N, Kx, Ku) # test

    # train setup
    _errors = (p,) -> errors(p, _x, _u0, _ut)
    _loss   = (p,) -> loss(p, _x, _u0, _ut)
    _cb  = (i, p,) -> callback(i, p, _loss, _errors)

    # test setup
    errors_ = (p,) -> errors(p, x_, u0_, ut_)
    loss_   = (p,) -> loss(p, x_, u0_, ut_)
    cb_  = (i, p,) -> callback(i, p, loss_, errors_)

    # test stats
    println("### Test stats ###")
    cb_(0, p)

    # training loop
    @time p = train(_loss, p; opt = Adam(1f-2), E = 1000, cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-3), E = 7000, cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-4), E = 2000, cb = _cb)

    # test stats
    println("### Test stats ###")
    cb_(0, p)

    p, model
end

# run this

rng = Random.default_rng()
Random.seed!(rng, 0)

N = 16      # problem size
Kx = 100    # X-samples
Ku = 100    # U-samples
E = 10_000  # epochs

ps, model = main(rng, N, Kx, Ku, E)

nothing
#
