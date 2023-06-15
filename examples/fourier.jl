#
"""
Geometry learning sandbox

Want to solve `A(x) * u = M(x) * f` for varying `x` by
learning the mapping `x -> (A^{-1} M)(x)`.

# Reference
* https://slides.com/vedantpuri/project-discussion-2023-05-10
"""

using GeometryLearning, FourierSpaces, NNlib
using SciMLOperators, LinearAlgebra, Random
using Zygote, Lux, ComponentArrays, Optimisers

""" data """
function datagen(rng, V, discr, Kx, Ku)

    N = size(V, 1)

    f = rand(rng, Float32, N, Ku)
    x = rand(rng, Float32, N, Kx)
    x = cumsum(x, dims = 1)

    f = kron(f, ones(Kx)')
    x = kron(ones(Ku)', x)

    V = make_transform(V, f)
    F = transformOp(V)

    # rm high freq
    Tr = truncationOp(V, (0.5,))
    x  = Tr * x
    f  = Tr * f

    # true sol
    D = gradientOp(V, discr)[1]
    M = massOp(V, discr)
    A = laplaceOp(V, discr)

    J = DiagonalOperator(D * x)
    Ji = inv(J)

    Lt = (Ji * Ji * A) \ (M * J)
    ut = Lt * f

    V, x, f, ut
end

""" model """
function model_setup(V, x, f)

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
    # model = (p, x, f) -> Lmodel(f, p, 0.0; x = x) 

    p, st = Lux.setup(rng, NN)
    p = p |> ComponentArray

    function model(p, x, f) # full
        j = D * x
        m = NN(j, p, st)[1]
        M = reshape(m, (N, N, K))

        # @tullio u[i, k] := M[i, j, k] * f[j, k]
        batched_vec(M, f)
    end

    function loss(p, x, f, ut)
        upred = model(p, x, f)

        norm(upred - ut, 2)
    end

    function errors(p, x, f, ut)
        upred = model(p, x, f)
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
    _V, _x, _f, _ut = datagen(rng, V, discr, Kx, Ku) # train
    V_, x_, f_, ut_ = datagen(rng, V, discr, Kx, Ku) # test

    # model setup
    K = Kx * Ku
    model, p, loss, errors = model_setup(_V, _x, _f)

    # train setup
    _errors = (p,) -> errors(p, _x, _f, _ut)
    _loss   = (p,) -> loss(p, _x, _f, _ut)
    _cb  = (i, E, p,) -> callback(i, E, p, _loss, _errors)

    # test setup
    errors_ = (p,) -> errors(p, x_, f_, ut_)
    loss_   = (p,) -> loss(p, x_, f_, ut_)
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
