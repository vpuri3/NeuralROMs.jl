#
using FourierSpaces
using GeometryLearning

let
    # add test dependencies to env stack
    pkgpath = dirname(dirname(pathof(GeometryLearning)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using CUDA, LuxCUDA, LuxDeviceUtils
using OrdinaryDiffEq, LinearSolve, LinearAlgebra
using Plots, JLD2

CUDA.allowscalar(false)

"""
Kuramoto-Sivashinsky equation

∂ₜu + Δu + Δ²u + 1/2 |∇u|² = 0

x ∈ [0, L)ᵈ (periodic)

TODO: Compute Lyapunov exponent (maybe sensitivities) in 1D/ 2D

https://en.wikipedia.org/wiki/Kuramoto%E2%80%93Sivashinsky_equation
https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2014.0932
"""

T = Float32
len = T(10pi)

function uData(x; σ = 0.3f0)
    u = @. sin(1f1 * Float32(pi) * x) * exp(-(x/σ)^2)
    reshape(u, :, 1)
end

function uIC(x; μ=zero(T), σ=T(0.5))
    u = @. exp(-T(1/10) * ((x-μ)/σ)^2)
    reshape(u, :, 1)
end

odecb = begin
    function affect!(int)
        if int.iter % 100 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function ks1D(N, len, mu = nothing, p = nothing;
    tspan=(T(0), T(5)),
    ntsave=5000,
    odealg=SSPRK43(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
)

    # space discr
    domain = IntervalDomain(-len, len; periodic = true)
    V = FourierSpace(N; domain) |> Float32
    Vh = transform(V)
    discr = Collocation()

    (x,) = points(V)

    # get initial condition
    u0 = uIC(x) # mu parameter dependence
    # u0 = uData(x) # mu parameter dependence
    V  = make_transform(V, u0; p)
    û0 = transformOp(V) * u0

    function convect!(v, u, p, t)
        copy!(v, u)
    end

    # move to device
    V  = V  |> device
    u0 = u0 |> device
    û0 = û0 |> device

    V  = make_transform(V, u0; p)

    # operators
    Â = laplaceOp(Vh, discr) # -Δ
    B̂ = biharmonicOp(Vh, discr) # Δ²
    Ĉ = advectionOp((zero(û0),), Vh, discr; vel_update_funcs! =(convect!,)) # uuₓ
    F̂ = SciMLOperators.NullOperator(Vh) # F = 0

    L = cache_operator(Â - B̂, û0)
    N = cache_operator(-Ĉ + F̂, û0)

    # time discr
    tsave = range(tspan...; length = ntsave)
    prob = SplitODEProblem(L, N, û0, tspan, p)

    # solve
    @time sol = solve(prob, odealg, saveat=tsave, abstol = T(1e-3), callback = odecb)
    @show sol.retcode

    # move back to device
    x = x     |> cpu_device()
    û = sol   |> Array
    t = sol.t |> cpu_device()
    V = V     |> cpu_device()

    # compute derivatives
    u, udx, ud2x, ud3x, ud4x = begin
        Nk, Nx = size(transformOp(V))
        _, Nb, Nt = size(û)
        û_re = reshape(û, Nk, Nb * Nt)
        u_re = similar(û_re, T, Nx, Nb * Nt)

        V = make_transform(V, u_re; p=p)
        Dx = gradientOp(V)[1]

        ldiv!(u_re, transformOp(V), û_re)

        du_re  = Dx * u_re
        d2u_re = Dx * du_re
        d3u_re = Dx * d2u_re
        d4u_re = Dx * d3u_re

        reshape.((u_re, du_re, d2u_re, d3u_re, d4u_re), Nx, Nb, Nt)
    end

    # V, x, u, udx, ud2x = begin
    #     N1 = 1024
    #
    #     V1 = FourierSpace(N1; domain) |> Float32
    #     V1 = make_transform(V1, similar(u, N1, Nb * Nt))
    #     J = interpOp(V1, V)
    #
    #     N, Nb, Nt = size(u)
    #
    #     u    = J * reshape(u   , N, Nb * Nt)
    #     udx  = J * reshape(udx , N, Nb * Nt)
    #     ud2x = J * reshape(ud2x, N, Nb * Nt)
    #
    #     sz = N1, Nb, Nt
    #
    #     V1, points(V1)[1], reshape(u, sz), reshape(udx, sz), reshape(ud2x, sz)
    # end

    mu = isnothing(mu) ? fill(nothing, size(u, 2)) |> Tuple : mu

    if !isnothing(dir)
        mkpath(dir)
        metadata = (; len, readme = "u [Nx, Nbatch, Nt]")

        filename = joinpath(dir, "data.jld2")
        jldsave(filename; x, u, udx, ud2x, t, mu, metadata)

        for k in 1:size(u, 2)
            Itplt = LinRange(1, size(u, 3), 100) .|> Base.Fix1(round, Int)

            _t = @view t[Itplt]
            _u = @view u[:, k, Itplt]
            _udx = @view udx[:, k, Itplt]
            _ud2x = @view ud2x[:, k, Itplt]

            xlabel = "x"
            ylabel = "u(x, t)"
            title = isnothing(mu[k]) ? "" : "μ = $(round(mu[k]; digits = 2)), "

            anim = animate1D(_u, x, _t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_$(k).gif"), fps = 30)

            anim = animate1D(_udx, x, _t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_dx_$(k).gif"), fps = 30)

            anim = animate1D(_ud2x, x, _t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_d2x_$(k).gif"), fps = 30)

            begin
                idx = LinRange(1f0, size(_u, 2), 11) .|> Base.Fix1(round, Int)
                plt = plot(;title = "Data", xlabel, ylabel, legend = false)
                plot!(plt, x, _u[:, idx])
                png(plt, joinpath(dir, "traj_$(k)"))
                display(plt)
            end
        end
    end

    (sol, V), (x, u, t, mu)
end

N = 256
mu = nothing
tspan=(T(0), T(10))
ntsave=1000

device = cpu_device()
dir = joinpath(@__DIR__, "data_ks")
(sol, V), (x, u, t, mu) = ks1D(N, len; tspan, ntsave, device, dir)

nothing
#