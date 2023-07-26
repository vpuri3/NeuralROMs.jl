#
"""
Solve poisson equation in fourier space with transformer model.
Should observe lienar structure in attention (QK') matrix

EQN

-Δ u = f, x ∈ [-π, π), periodic

SOL

û(k) = f̂ / (1 + k²)

MODEL:
single layer model with no nonlinearity (single head linear attention)

û = NN(f̂)
q = Wq * f̂
k = Wk * f̂
v = Wv * f̂

û = (q * k') * v

# Reference
* https://slides.com/vedantpuri/coadvisor_meeting
"""

using GeometryLearning, FourierSpaces
using SciMLOperators, LinearAlgebra, Random
using Zygote, NNlib, Lux, ComponentArrays, Optimisers

""" data """
function datagen(rng, V, discr, K, D)

    # N = size(V, 1)

    u = rand(rng, Float32, N, K)

    # new function space
    V = make_transform(V, u)
    F = transformOp(V)
    V̂ = FourierSpaces.transform(V)
    
    # rm high freq
    # Tr = truncationOp(V̂, (0.5,))
    # û  = Tr * F * u

    # true sol
    Â = laplaceOp(V̂, discr)

    û = F * u
    f̂ = Â * û
    
    V, f̂, û

    # u = rand(rng, Float32, N, K)
    # f = D * u
    #
    # V, f, u
end

""" model """
function model_setup(V, f)

    N, K  = size(f)

    NN = Chain(
        Dense(N, N, use_bias = false),
    )

    # NN = Atten(N, N)
    NN = Diag(N)

    p, st = Lux.setup(rng, NN)
    p = p |> ComponentArray

    function model(p, f) # full
        NN(f, p, st)[1]
    end

    function loss(p, f, utrue)
        upred = model(p, f)
        udiff = upred - utrue

        norm(udiff, 2) #/ length(utrue)
    end

    function errors(p, f, utrue)
        upred = model(p, f)
        udiff = upred - utrue

        meanAE = norm(udiff, 1) / length(utrue)
        maxAE  = norm(udiff, Inf)

        meanAE, maxAE
    end

    model, p, loss, errors
end

function callback(iter, E, p, loss, errors)

    a = iszero(E) ? 1 : round(0.05 * E) |> Int
    if iter % a == 0 || iter == 1 || iter == E

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

    # init optimizer
    opt_st = Optimisers.setup(opt, p)

    !isnothing(cb) && cb(0, E, p)

    for iter in 1:E
        l, pb = Zygote.pullback(loss, p)
        gr = pb(one.(l))[1]

        opt_st, p = Optimisers.update(opt_st, p, gr)

        !isnothing(cb) && cb(iter, E, p)

    end

    p
end

""" main program """
function main(rng, N, K, E)

    """ space discr """
    V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
    Vh = CalculustCore.transform(V)
    discr = Collocation()

    # datagen
    D = Diagonal(rand(N))
    _V, _f, _u = datagen(rng, V, discr, K, D) # train
    V_, f_, u_ = datagen(rng, V, discr, K, D) # test

    # model setup
    model, p, loss, errors = model_setup(_V, _f)

    # train setup
    _errors = (p,) -> errors(p, _f, _u)
    _loss   = (p,) -> loss(p, _f, _u)
    _cb  = (i, E, p,) -> callback(i, E, p, _loss, _errors)

    # test setup
    errors_ = (p,) -> errors(p, f_, u_)
    loss_   = (p,) -> loss(p, f_, u_)
    cb_  = (i, E, p,) -> callback(i, E, p, loss_, errors_)

    # test stats
    println("### Test stats ###")
    cb_(0, 0, p)

    # training loop
    @time p = train(_loss, p; opt = Adam(1f+1), E = Int(E/2), cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-0), E = Int(E/2), cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-1), E = Int(E/2), cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-2), E = Int(E/2), cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-3), E = Int(E/2), cb = _cb)

    # test stats
    println("### Test stats ###")
    cb_(0, 0, p)

    p, model, D
end

# run this

rng = Random.default_rng()
Random.seed!(rng, 0)

N = 16     # problem size
K = 100    # U-samples
E = 10_000 # epochs

ps, model, D = main(rng, N, K, E)

nothing
#
