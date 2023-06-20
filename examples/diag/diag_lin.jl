#
"""
Geometry learning sandbox

Learning a diagonal matrix
"""

using SciMLOperators, LinearAlgebra, Random
using Zygote, Lux, Optimisers

rng = Random.default_rng()
Random.seed!(rng, 0)

""" data """

N = 32  # problem size
K = 20_000 # number of samples
E = 3_000 # number of epochs

u0 = rand(rng, N, K)
x  = rand(rng, N, K)

# true sol
D = DiagonalOperator(rand(rng, N))
X = DiagonalOperator(x)
Ltrue = X * D #* X * D # linear for now
utrue = Ltrue * u0

""" model """
function model_update_func(D, u, p, t; x = x)
    # replace with Lux NN model
    P = reshape(p, N, N)
    P * x
end

Lmodel = DiagonalOperator(zeros(N, K); update_func = model_update_func)

function model(p; u0 = u0, L = Lmodel)
    L(u0, p, 0.0)
end

function loss(p; u0 = u0, utrue = utrue)
    upred = model(p; u0 = u0)

    norm(upred - utrue, 2) / length(upred)
end

""" training """
ps = rand(rng, N * N)

# dry run
l, pb = Zygote.pullback(loss, ps)
gr = pb(one.(l))[1]

opt = Optimisers.Adam()
opt_st = Optimisers.setup(opt, ps)

println("Loss with initial parameter set: ", l)

for i in 1:E
    global ps, opt_st, gr, pb, l

    l, pb = Zygote.pullback(loss, ps)
    gr = pb(one.(l))[1]

    opt_st, ps = Optimisers.update(opt_st, ps, gr)

    if i % 100 == 0 || i == E
        println("Iter $i: loss: ", l)
    end
end

P = reshape(ps, N, N)

nothing
#
