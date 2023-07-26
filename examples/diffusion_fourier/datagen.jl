"""
Generate solution data to diffusion equation

    -∇⋅ν∇u = f

for changing ν, f
"""

using FourierSpaces, LinearAlgebra, LinearSolve, BSON

""" data """
function datagen1D(rng, N, K1, K2; mode = :train)

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

    # F0 = transformOp(V0)
    # F1 = transformOp(V1)
    # F2 = transformOp(V2)

    ν = 10 .+ 100 .^ rand(rng, Float32, N, K1)
    f = 0 .+ 1000 .^ rand(rng, Float32, N, K2)

    # truncation op
    # X0 = truncationOp(V0, (0.50,))
    X1 = truncationOp(V1, (0.15,))
    X2 = truncationOp(V2, (0.50,))

    # rm high freq modes
    ν = X1 * ν
    f = X2 * f

    @assert all(>(1f0), ν)

    # get arrays
    ν0 = kron(ν, ones(K2)')
    ν1 = ν
    ν2 = kron(ν[:, 1], ones(K2)')

    f0 = kron(ones(K1)', f)
    f1 = kron(f[:, 1], ones(K1)')
    f2 = f

    @assert f1[:, 1] == f1[:, end]
    @assert ν2[:, 1] == ν2[:, end]

    # arbitrarily scale forcing
    if mode == :test
        fscale0 = 10 * rand(rng, 1, K0)
        fscale1 = 10 * rand(rng, 1, K1)
        fscale2 = 10 * rand(rng, 1, K2)

        f0 = f0 .* fscale0 .* exp.(sin.(x0 .- 10 * rand(N, K0)) .* 5 .* rand(N, K0))
        f1 = f1 .* fscale1 .* exp.(sin.(x1 .- 10 * rand(N, K1)) .* 5 .* rand(N, K1))
        f2 = f2 .* fscale2 .* exp.(sin.(x2 .- 10 * rand(N, K2)) .* 5 .* rand(N, K2))
    end

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

    V, data0, data1, data2
end

function datagen2D(rng, N, K1, K2; mode = :train)

    K0 = K1 * K2

    V = FourierSpace(N, N; domain = FourierDomain(2)) |> Float32
    x, y = points(V)

    # get ν, f
    ν = 10 .+ 100  .^ rand(rng, Float32, N * N, K1)
    f = 0  .+ 1000 .^ rand(rng, Float32, N * N, K2)
    f = f .- sum(f, dims = 1) / (N * N)

    # make spaces
    V1 = make_transform(V, similar(ν))
    V2 = make_transform(V, similar(f))

    F1 = transformOp(V1)
    F2 = transformOp(V2)

    # rm high-freq modes
    ν = truncationOp(V1, (0.15, 0.15)) * ν
    f = truncationOp(V2, (0.50, 0.50)) * f

    @assert all(>(1f0), ν)
    @assert all(<(1f2), ν)

    # data arrays
    x0 = kron(x, ones(Float32, K0)')
    y0 = kron(y, ones(Float32, K0)')
    ν0 = kron(ν, ones(Float32, K2)')
    f0 = kron(ones(Float32, K1)', f)
    u0 = ones(Float32, N * N, K0)

    # diffusion op
    discr = Collocation()

    for i in axes(u0, 2)
        _ν = ν0[:, i]
        _f = f0[:, i]

        A = diffusionOp(_ν, V, discr)
        A = cache_operator(A, x)

        # prob = LinearProblem(A, _f; u0 = view(u0, :, i))
        prob = LinearProblem(A, _f)
        sol = solve(prob, KrylovJL_GMRES(), reltol = 1f-4)
        @show sol.iters
        @show resid = norm(A * sol.u - _f, 2) / (N * N)

        u0[:, i] = sol.u
    end

    # V0 = make_transform(V, x0)
    # A0 = diffusionOp(ν0, V, discr)

    data0 = (x0, y0, ν0, f0, u0) # var   ν, var   f

    @assert all(isequal((N * N, K0)), size.(data0)) "got $(size.(data0))"

    V, data0
end

function combine_data1D(data)
    x, ν, f, u = data

    N, K = size(x)

    x1 = zeros(3, N, K) # x, ν, f
    u1 = zeros(1, N, K) # u

    x1[1, :, :] = x
    x1[2, :, :] = ν
    x1[3, :, :] = f

    u1[1, :, :] = u

    (x1, u1)
end

function combine_data2D(data, K = size(data[1], 2))
    x, y, ν, f, u = data

    N, Kmax = size(x)
    n = sqrt(N) |> Integer
    K = min(K, Kmax)

    Ks = if K == Kmax
        1:K
    else
        rand(1:Kmax, K)
    end

    x1 = zeros(4, n, n, K) # x, y, ν, f
    u1 = zeros(1, n, n, K) # u

    x1[1, :, :, :] = x[:, Ks] |> vec
    x1[2, :, :, :] = y[:, Ks] |> vec
    x1[3, :, :, :] = ν[:, Ks] |> vec
    x1[4, :, :, :] = f[:, Ks] |> vec
    #
    u1[1, :, :, :] = u[:, Ks] |> vec

    (x1, u1)
end

#=
using Plots, Random

N  = 32
K1 = 32
K2 = 32

rng = Random.default_rng()
Random.seed!(rng, 127)

V, _data = datagen2D(rng, N, K1, K2)
V, data_ = datagen2D(rng, N, K1, K2)

# x, y, ν, f, u = data
# plot(u[:, 10], V)

BSON.@save joinpath(@__DIR__, "data2D_N$(N).bson") _data data_
=#

#=
using Plots, Random

N  = 128    # problem size
K1 = 100     # X-samples
K2 = 100     # X-samples
K0 = K1 * K2

rng = Random.default_rng()
Random.seed!(rng, 127)

V, data0, data1, data2 = datagen1D(rng, N, K1, K2)

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
