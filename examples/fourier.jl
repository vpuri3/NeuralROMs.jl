#
"""
Geometry learning sandbox

Want to solve `A(x) * u = M(x) * f` for varying `x` by
learning the mapping `x -> (A^{-1} M)(x)`.

# Reference
* https://slides.com/vedantpuri/project-discussion-2023-05-10
"""

using GeometryLearning, FourierSpaces, Tullio
using SciMLOperators, LinearAlgebra, Random
using Zygote, Lux, ComponentArrays, Optimisers

""" data """
function datagen(rng, V, discr, Kx, Ku)

    N = size(V, 1)

    u = rand(rng, Float32, N, Ku)
    x = rand(rng, Float32, N, Kx)
    x = cumsum(x, dims = 1)

    u0 = kron(u, ones(Kx)')
    x  = kron(ones(Ku)', x)

    V = make_transform(V, u0)
    F = transformOp(V)

    # rm high freq
    Tr = truncationOp(V, (0.5,))
    x  = Tr * x
    u0 = Tr * u0

    # true sol
    D = gradientOp(V, discr)[1]
    M = massOp(V, discr)
    A = laplaceOp(V, discr)

    J = DiagonalOperator(D * x)
    Ji = inv(J)

    Lt = (Ji * Ji * A) \ (M * J)
    ut = Lt * u0

    V, x, u0, ut
end

""" model """
function model_setup(V, x, u0)

    D = gradientOp(V)[1]
    F = transformOp(V)

    N, K  = size(x)
    N2 = N * N

    NN = Chain(
        Dense(N , N2, tanh),
        Dense(N2, N2, tanh),
        Dense(N2, N2, tanh),
        Dense(N2, N2),
    )

    # wait for https://github.com/SciML/SciMLOperators.jl/pull/143
    # model_update_func(D, u, p, t; x = x) = NN(x, p, st)[1]
    # Lmodel = DiagonalOperator(zeros(N, K); update_func = model_update_func)
    # model = (p, x, u0) -> Lmodel(u0, p, 0.0; x = x) 

    p, st = Lux.setup(rng, NN)
    p = p |> ComponentArray

    function model(p, x, u0) # full
        j = D * x
        m = NN(j, p, st)[1]
        M = reshape(m, (N, N, K))

        @tullio u[i, k] := M[i, j, k] * u0[j, k]
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

function callback(iter, E, p, loss, errors)

    a = iszero(E) ? 1 : round(0.01 * E) |> Int
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

    println("Loss with initial parameter set: ", l)

    # init optimizer
    opt_st = Optimisers.setup(opt, p)

    for iter in 1:E
        l, pb = Zygote.pullback(loss, p)
        gr = pb(one.(l))[1]

        opt_st, p = Optimisers.update(opt_st, p, gr)

        !isnothing(cb) && cb(iter, E, p)

    end

    p
end

""" main program """
function main(rng, N, Kx, Ku, E)

    """ space discr """
    V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
    Vh = CalculustCore.transform(V)
    discr = Collocation()

    # datagen
    _V, _x, _u0, _ut = datagen(rng, V, discr, Kx, Ku) # train
    V_, x_, u0_, ut_ = datagen(rng, V, discr, Kx, Ku) # test

    # model setup
    K = Kx * Ku
    model, p, loss, errors = model_setup(_V, _x, _u0)

    # train setup
    _errors = (p,) -> errors(p, _x, _u0, _ut)
    _loss   = (p,) -> loss(p, _x, _u0, _ut)
    _cb  = (i, E, p,) -> callback(i, E, p, _loss, _errors)

    # test setup
    errors_ = (p,) -> errors(p, x_, u0_, ut_)
    loss_   = (p,) -> loss(p, x_, u0_, ut_)
    cb_  = (i, E, p,) -> callback(i, E, p, loss_, errors_)

    # test stats
    println("### Test stats ###")
    cb_(0, 0, p)

    # training loop
    @time p = train(_loss, p; opt = Adam(1f-2), E = Int(E*0.05), cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-3), E = Int(E*0.70), cb = _cb)
    @time p = train(_loss, p; opt = Adam(1f-4), E = Int(E*0.25), cb = _cb)

    # test stats
    println("### Test stats ###")
    cb_(0, 0, p)

    p, model
end

# run this

rng = Random.default_rng()
Random.seed!(rng, 0)

N  = 8      # problem size
Kx = 100    # X-samples
Ku = 100    # U-samples
E = 10_000  # epochs

ps, model = main(rng, N, Kx, Ku, E)

nothing
#
