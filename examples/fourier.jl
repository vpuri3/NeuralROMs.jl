#
"""
Geometry learning sandbox

Want to solve `A(x) * u = M(x) * f` for varying `x` by
learning the mapping `x -> (A \ M)(x)`.

# Reference
* https://slides.com/vedantpuri/project-discussion-2023-05-10
"""

using LinearAlgebra, Random
using Zygote, Lux

Random.seed!(123)

N = 4  # problem size
K = 10 # number of samples

# """ space discr """
# V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
# Vh = transform(V)
# discr = Collocation()
#
# (x,) = points(V)
# (k,) = modes(V)
# F = transformOp(V)
#
# u0 = @. sin(x)
#
# A = laplaceOp(V, discr)
# A = cache_operator(A, x)
#
# Ah = laplaceOp(Vh, discr)
# Ah = cache_operator(Ah, k)

u0 = rand(N, K)
x  = rand(N, K)

D = DiagonalOperator(rand(N))
J = DiagonalOperator(x)
Ltrue  = J * D #* J * D # linear for now

function ut(u)
    Ltrue * u
end

function model_update_func(D, u, p, t; x = x)
    # replace with Lux NN model
    P = reshape(p, N, N)
    P * x
end

Lmodel = DiagonalOperator(zeros(N, K); update_func = model_update_func)

function model(p; u = u0, L = L)
    L(u, p, 0.0)
end

function loss(p; u0 = u0, utrue = utrue)
    upred = model(p; u0 = u0)

    norm(upred - utrue, 2)
end
#
