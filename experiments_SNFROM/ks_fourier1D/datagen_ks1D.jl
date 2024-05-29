#
using FourierSpaces
using NeuralROMs

let
    # add test dependencies to env stack
    pkgpath = dirname(dirname(pathof(NeuralROMs)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using CUDA, LuxCUDA, LuxDeviceUtils
using OrdinaryDiffEq, LinearSolve, LinearAlgebra
using Plots, JLD2

CUDA.allowscalar(false)

"""
Kuramoto-Sivashinsky equation (normalized)

∂ₜu + Δu + νΔ²u + 1/2 |∇u|² = 0

x ∈ [-π, π)ᵈ (periodic)

https://apps.dtic.mil/sti/tr/pdf/ADA306758.pdf
https://en.wikipedia.org/wiki/Kuramoto%E2%80%93Sivashinsky_equation
https://royalsocietypublishing.org/doi/epdf/10.1098/rspa.2014.0932
"""

T = Float32

function uIC(x; μ=zero(T), σ=T(0.20)) # 0.30
    u = @. exp(-T(1/2) * ((x-μ)/σ)^2)
    # u = @. sin(T(0.5) * T(pi) * x) * exp(-((x-μ)/σ)^2)
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

function ks1D(N, len = T(2pi), ν = T(1), mu = nothing, p = nothing;
    tspan=(T(0), T(5)),
    ntsave=100,
    odealg=SSPRK43(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
)

    # space discr
    domain = IntervalDomain(-len/2, len/2; periodic = true)
    V = FourierSpace(N; domain) |> Float32
    discr = Collocation()

    (x,) = points(V)

    # get initial condition
    u0 = uIC(x) # mu parameter dependence
    V  = make_transform(V, u0; p)

    function convect!(v, u, p, t)
        copy!(v, u)
    end

    # move to device
    V  = V  |> device
    u0 = u0 |> device

    V  = make_transform(V, u0; p)

    # operators
    A = laplaceOp(V, discr) # -Δ
    B = biharmonicOp(V, discr) # Δ²
    C = advectionOp((zero(u0),), V, discr; vel_update_funcs! =(convect!,)) # uuₓ

    L = cache_operator(A - ν * B - C, u0)

    # time discr
    tsave = range(tspan...; length = ntsave)
    prob = ODEProblem(L, u0, tspan, p)

    # solve
    @time sol = solve(prob, odealg, saveat=tsave, abstol = T(1e-3), callback = odecb)
    @show sol.retcode

    # move back to device
    x = x     |> cpu_device()
    u = sol   |> Array
    t = sol.t |> cpu_device()
    V = V     |> cpu_device()

    # compute derivatives
    udx, ud2x, ud3x, ud4x = begin
        Nx, Nb, Nt = size(u)
        u_re = reshape(u, Nx, Nb * Nt)

        V = make_transform(V, u_re; p=p)
        Dx = gradientOp(V)[1]

        du_re  = Dx * u_re
        d2u_re = Dx * du_re
        d3u_re = Dx * d2u_re
        d4u_re = Dx * d3u_re

        udervs = (du_re, d2u_re, d3u_re, d4u_re,)

        reshape.(udervs, Nx, Nb, Nt)
    end

    mu = isnothing(mu) ? fill(nothing, size(u, 2)) |> Tuple : mu

    if !isnothing(dir)
        mkpath(dir)
        metadata = (; ν, L = len, readme = "u [Nx, Nbatch, Nt]")

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
            gif(anim, joinpath(dir, "traj_$(k).gif"), fps = 250)

            anim = animate1D(_udx, x, _t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_dx_$(k).gif"), fps = 30)
            
            anim = animate1D(_ud2x, x, _t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_d2x_$(k).gif"), fps = 30)

            begin
                idx = LinRange(1f0, size(_u, 2), 11) .|> Base.Fix1(round, Int)
                plt = plot(;title = "Data", xlabel, ylabel, legend = true)
                plot!(plt, x, _u[:, idx])
                png(plt, joinpath(dir, "traj_$(k)"))
                display(plt)
            end
        end
    end

    (sol, V), (x, u, t, mu)
end

N = 256
ν = 0.01f0
# ν = 0.1f0
mu = nothing
L = T(2pi)
tspan=(T(0), T(0.1))
ntsave=1000

device = cpu_device()
dir = joinpath(@__DIR__, "data_ks")
(sol, V), (x, u, t, mu) = ks1D(N, L, ν; tspan, ntsave, device, dir)

# x = LinRange(-L/2, L/2, 1000)
# plot(x, vec(uIC(x)), w = 2, c = :black) |> display

nothing
#
