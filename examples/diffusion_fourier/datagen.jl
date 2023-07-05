"""
Generate solution data to diffusion equation

    -∇⋅ν∇u = f

for changing ν, f
"""

using FourierSpaces, LinearAlgebra, Plots, Random

""" data """
function datagen(rng, N, K1, K2)

    K0 = K1 * K2

    V = FourierSpace(N; domain = IntervalDomain(0, 2pi))
    x = points(V)[1]

    # get points
    x0 = kron(x, ones(K0)')
    x1 = kron(x, ones(K1)')
    x2 = kron(x, ones(K2)')

    # make function spaces
    V0 = make_transform(V, x0)
    V1 = make_transform(V, x1)
    V2 = make_transform(V, x2)

    F0 = transformOp(V0)
    F1 = transformOp(V1)
    F2 = transformOp(V2)

    ν = 1 .+ 50 .^ rand(rng, Float32, N, K1)
    # ν = 1 .+  32 .^ rand(Float32, N, K1)
    f = 0 .+ 400 * rand(rng, Float32, N, K2)

    # f = 0 .+ 20 * rand(Float32, N, K2)

    # truncation op
    # X0 = truncationOp(V0, (0.50,))
    X1 = truncationOp(V1, (0.15,))
    X2 = truncationOp(V2, (0.50,))

    # rm high freq modes
    ν = X1 * ν
    f = X2 * f

    # get arrays
    ν0 = kron(ν, ones(K2)')
    ν1 = ν
    ν2 = kron(ν[:, 1], ones(K2)')

    f0 = kron(ones(K1)', f)
    f1 = kron(f[:, 1], ones(K1)')
    f2 = f

    # diffusion op
    discr = Collocation()
    A0 = diffusionOp(ν0, V0, discr)
    A1 = diffusionOp(ν1, V1, discr)
    A2 = diffusionOp(ν2, V2, discr)

    # true solution
    u0 = A0 \ f0
    u1 = A1 \ f1
    u2 = A2 \ f2

    data0 = (x0, ν0, f0, u0) # var   ν, var   f
    data1 = (x1, ν1, f1, u1) # var   ν, fixed f
    data2 = (x2, ν2, f2, u2) # fixed ν, var   f

    @assert all(isequal((N, K0)), size.(data0))
    @assert all(isequal((N, K1)), size.(data1))
    @assert all(isequal((N, K2)), size.(data2))

    @assert f1[:, 1] == f1[:, end]
    @assert ν2[:, 1] == ν2[:, end]

    V, data0, data1, data2
end

function combine_data(data)
    x, ν, f, u = data

    N, K = size(x)

    x1 = zeros(3, N, K) # x, ν, f
    y  = zeros(1, N, K) # u

    x1[1, :, :] = x
    x1[2, :, :] = ν
    x1[3, :, :] = f

    y[1, :, :] = u

    (x1, y)
end

function split_data(data)
    x, ν, f, u = data

    N, K = size(x)

    x1 = zeros(2, N, K) # ν, x
    x2 = zeros(1, N, K) # f
    y  = zeros(1, N, K) # u

    x1[1, :, :] = x
    x1[2, :, :] = ν

    x2[1, :, :] = f

    y[1, :, :] = u

    ((x1, x2), y)
end

#=
N  = 128    # problem size
K1 = 100     # X-samples
K2 = 100     # X-samples
K0 = K1 * K2

rng = Random.default_rng()
Random.seed!(rng, 123)

# datagen
V, data0, data1, data2 = datagen(rng, N, K1, K2)

x0, ν0, f0, u0 = data0
x1, ν1, f1, u1 = data1
x2, ν2, f2, u2 = data2

nplts = 5

dir = @__DIR__

x  = points(V)[1]

I0 = rand(1:K0, nplts)
p0 = plot(x, u0[:, I0], w = 2.0, legend = nothing, title = "u(ν, f)")
png(p0, joinpath(dir, "plt_u0"))

##########
I1 = rand(1:K1, nplts)
p1 = plot(x, u1[:, I1], w = 2.0, legend = nothing, title = "u(ν, f₀)")
png(p1, joinpath(dir, "plt_u1"))

p1 = plot(x, f1[:, I1], w = 2.0, legend = nothing, title = "f₀(x)")
png(p1, joinpath(dir, "plt_f1"))

p1 = plot(x, ν1[:, I1], w = 2.0, legend = nothing, title = "ν(x)")
png(p1, joinpath(dir, "plt_nu1"))

##########
I2 = rand(1:K2, nplts)
p2 = plot(x, u2[:, I2], w = 2.0, legend = nothing, title = "u(ν₀, f)")
png(p2, joinpath(dir, "plt_u2"))

p2 = plot(x, f2[:, I2], w = 2.0, legend = nothing, title = "f(x)")
png(p2, joinpath(dir, "plt_f2"))

p2 = plot(x, ν2[:, I2], w = 2.0, legend = nothing, title = "ν₀(x)")
png(p2, joinpath(dir, "plt_nu2"))

##########
nothing
=#
