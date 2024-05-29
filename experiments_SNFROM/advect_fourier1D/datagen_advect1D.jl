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

function uIC(x; μ = -0.5f0, σ = 0.1f0)
    u = @. exp(-1f0/2f0 * ((x-μ)/σ)^2)
    reshape(u, :, 1)
end

function advect1D(N, ν, c, mu = nothing, p = nothing;
    tspan=(0.f0, 4.0f0),
    ntsave=500,
    odealg=Tsit5(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
)

    # space discr
    domain = IntervalDomain(-1f0, 1f0; periodic = true)
    V = FourierSpace(N; domain) |> Float32
    discr = Collocation()

    (x,) = points(V)

    # get initial condition
    u0 = uIC(x) # mu parameter dependence
    V  = make_transform(V, u0; p)

    # move to device
    V  = V  |> device
    u0 = u0 |> device

    # operators
    A = -diffusionOp(ν, V, discr)
    C = advectionOp((fill!(similar(u0), c),), V, discr)
    odefunc = cache_operator(A - C, u0)

    # time discr
    tsave = range(tspan...; length = ntsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-6)

    # solve
    @time sol = solve(prob, odealg, saveat=tsave)
    @show sol.retcode

    # move back to device
    x = x     |> cpu_device()
    u = sol   |> Array
    t = sol.t |> cpu_device()
    V = V     |> cpu_device()

    # compute derivatives
    udx, ud2x = begin
        Nx, Nb, Nt = size(u)
        u_re = reshape(u, Nx, Nb * Nt)

        V  = make_transform(V, u_re; p=p)
        Dx = gradientOp(V)[1]

        du_re  = Dx * u_re
        d2u_re = Dx * du_re

        reshape(du_re, Nx, Nb, Nt), reshape(d2u_re, Nx, Nb, Nt)
    end

    mu = isnothing(mu) ? fill(nothing, size(u, 2)) |> Tuple : mu

    if !isnothing(dir)
        mkpath(dir)
        metadata = (; c, ν, readme = "u [Nx, Nbatch, Nt]")

        filename = joinpath(dir, "data.jld2")
        jldsave(filename; x, u, udx, ud2x, t, mu, metadata)

        for k in 1:size(u, 2)
            _u = @view u[:, k, :]
            _udx = @view udx[:, k, :]
            _ud2x = @view ud2x[:, k, :]

            xlabel = "x"
            ylabel = "u(x, t)"
            title = isnothing(mu[k]) ? "" : "μ = $(round(mu[k]; digits = 2)), "

            anim = animate1D(_u, x, t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_$(k).gif"), fps=30)

            anim = animate1D(_udx, x, t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_dx_$(k).gif"), fps=30)

            anim = animate1D(_ud2x, x, t; linewidth=2, xlabel, ylabel, title)
            gif(anim, joinpath(dir, "traj_d2x_$(k).gif"), fps=30)

            begin
                idx = LinRange(1f0, ntsave, 11) .|> Base.Fix1(round, Int)
                plt = plot(;title = "Data", xlabel, ylabel, legend = false)
                plot!(plt, x, _u[:, idx])
                png(plt, joinpath(dir, "traj_$(k)"))
                display(plt)
            end
        end
    end

    (sol, V), (x, u, t, mu)
end

N = 128
ν = 0f0
c = 0.25f0
mu = nothing # parameters

device = cpu_device()
dir = joinpath(@__DIR__, "data_advect")
(sol, V), (x, u, t, mu) = advect1D(N, ν, c, mu; device, dir)

nothing
#
