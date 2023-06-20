#
"""
Learn solution to diffusion equation

∇⋅ν∇ u = f₀

for variable ν, and constant f₀

test bed for Fourier Neural Operator experiments where
forcing is learned separately.
"""

using GeometryLearning, FourierSpaces, NNlib
using SciMLOperators, LinearAlgebra, Random
using Zygote, Lux, ComponentArrays, Optimisers

using Plots, Colors

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
    u = u .+ 1.0

    V, ν, u
end

""" model """
function model_setup() # input

    function model(p, ν)
        NN(ν, p, st)[1]
    end

    function loss(p, ν, utrue)
        upred = model(p, ν)

        norm(upred - utrue, 2)
    end

    function errors(p, ν, utrue)
        upred = model(p, ν)
        udiff = upred - utrue

        meanAE = norm(udiff, 1) / length(utrue)
        maxAE  = norm(udiff, Inf)

        urel   = udiff ./ utrue
        meanRE = norm(urel, 1) / length(utrue)
        maxRE  = norm(urel, Inf)

        meanAE, maxAE, meanRE, maxRE
    end

    model, loss, errors
end

function callback(p, iter, E, loss, errors)

    a = iszero(E) ? 1 : round(0.1 * E) |> Int
    if iter % a == 0 || iter == 1 || iter == E

        if !isnothing(errors)
            err = errors(p)

            str  = string("Iter $iter: LOSS: ", round(loss(p), digits=8))
            str *= string(", meanAE: ", round(err[1], digits=8))
            str *= string(", maxAE: " , round(err[2] , digits=8))
            str *= string(", meanRE: ", round(err[3], digits=8))
            str *= string(", maxRE: " , round(err[4] , digits=8))
        end

        println(str)
    end

    nothing
end

""" training """
function train(loss, p; opt = Optimisers.Adam(), E = 5000, cb = nothing)

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

        !isnothing(cb) && cb(p, iter, E)
    end

    p
end

""" visualize """
function visualize(V, test, train, model; nsamples = 5)
    _ν, _u = test
    ν_, u_ = train

    _v = model(_ν)
    v_ = model(ν_)

    x, = points(V)

    N = size(_u, 1)
    K = size(_u, 2)
    I = rand(axes(_u, 2), nsamples)
    n = 4
    ms = 4

    cmap = range(HSV(0,1,1), stop=HSV(-360,1,1), length = nsamples) 

    _plt = plot(title = "Training Comparison, N=$N points, $K trajectories", legend = false)
    for i in 1:nsamples
        c = cmap[i]

        ii = I[i]
        u = _u[:, ii]
        v = _v[:, ii]
        scatter!(_plt, x[begin:n:end], u[begin:n:end], markersize = ms, c = c)
        plot!(_plt, x, v, s = :solid, w = 2.0, c = c)
    end

    plt_ = plot(title = "Testing Comparison, N=$N points, $K trajectories", legend = false)
    for i in 1:nsamples
        c = cmap[i]

        ii = I[i]
        u = u_[:, ii]
        v = v_[:, ii]
        scatter!(plt_, x[begin:n:end], u[begin:n:end], markersize = ms, c = c)
        plot!(plt_, x, v, s = :solid, w = 2.0, c = c)
    end

    _plt, plt_
end

""" main program """

# parameters
N = 128    # problem size
K = 100    # X-samples
E = 20000  # epochs

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

model, loss, errors = model_setup()

# train setup
_errors = (p,) -> errors(p, _ν, _u)
_loss   = (p,) -> loss(p, _ν, _u)
_cb  = (p, i = 0, E = 0) -> callback(p, i, E, _loss, _errors)

# test setup
errors_ = (p,) -> errors(p, ν_, u_)
loss_   = (p,) -> loss(p, ν_, u_)
cb_  = (p, i = 0, E = 0) -> callback(p, i, E, loss_, errors_)

# test stats
println("### Test stats ###")
cb_(p)

# training loop
@time p = train(_loss, p; opt = Optimisers.Adam(1f-2), E = Int(E*0.05), cb = _cb)
@time p = train(_loss, p; opt = Optimisers.Adam(1f-3), E = Int(E*0.70), cb = _cb)
@time p = train(_loss, p; opt = Optimisers.Adam(1f-4), E = Int(E*0.25), cb = _cb)

# test stats
println("### Test stats ###")
cb_(p)

# visualization
_plt, plt_ = visualize(V, (_ν, _u), (ν_, u_), x -> model(p, x))
display(plt_)
#
