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

function uIC(x, y; μ = (-0.50f0, -0.50f0), σ = 0.1f0)
    r2 = @. (x - μ[1])^2 + (y - μ[2])^2
    u0 = @. exp(-1f0/2f0 * r2/(σ^2))
    reshape(u0, :, 1)
end

function advect1D(Nx, Ny, ν, cx, cy, mu = nothing, p = nothing;
    domain = FourierDomain(2),
    tspan=(0.f0, 4.0f0),
    ntsave=1000,
    odealg=Tsit5(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
)

    # space discr
    V = FourierSpace(Nx, Ny; domain) |> Float32
    discr = Collocation()

    (x, y) = points(V)

    # get initial condition
    u0 = uIC(x, y) # mu parameter dependence

    V  = make_transform(V, u0; p)

    # move to device
    V  = V  |> device
    u0 = u0 |> device

    vel = begin
        o = fill!(similar(u0), 1)

        (cx * o, cy * o)
    end

    # operators
    A = -diffusionOp(ν, V, discr)
    C = advectionOp(vel, V, discr)
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

    Nxy, Nb, Nt = size(u)

    # compute derivatives
    udx, udy = begin
        u_re = reshape(u, Nxy, Nb * Nt)

        V  = make_transform(V, u_re; p=p)
        Dx, Dy = gradientOp(V)

        udx_re = Dx * u_re
        udy_re = Dy * u_re

        sz = (Nxy, Nb, Nt)

        reshape(udx_re, sz), reshape(udy_re, sz)
    end

    if !isnothing(dir)

        mkpath(dir)

        # saving
        mu = isnothing(mu) ? fill(nothing, size(u, 2)) |> Tuple : mu
        x_save = reshape(vcat(x', y'), 2, Nxy)
        grid = (Nx, Ny)
        metadata = (; grid, cx, cy, ν, readme = "u [Nx, Nbatch, Nt]")

        filename = joinpath(dir, "data.jld2")
        jldsave(filename; x = x_save, u, udx, udy, t, mu, metadata)

        # visualization
        x_re = reshape(x, Nx, Ny)
        y_re = reshape(y, Nx, Ny)

        _x = x_re[:, 1]
        _y = y_re[1, :]

        for k in 1:size(u, 2)
            _u = @view u[:, k, :]
            _udx = @view udx[:, k, :]
            _udy = @view udy[:, k, :]
        
            xlabel = "x"
            ylabel = "u(x, t)"
            title = isnothing(mu[k]) ? "" : "μ = $(round(mu[k]; digits = 2)), "
        
            # anim = animate2D(_u, x, t; linewidth=2, xlabel, ylabel, title)
            # gif(anim, joinpath(dir, "traj_$(k).gif"), fps=30)
            #        
            # anim = animate2D(_udx, x, t; linewidth=2, xlabel, ylabel, title)
            # gif(anim, joinpath(dir, "traj_dx_$(k).gif"), fps=30)
            #        
            # anim = animate2D(_udy, x, t; linewidth=2, xlabel, ylabel, title)
            # gif(anim, joinpath(dir, "traj_dy_$(k).gif"), fps=30)

            anim = animate(_u, V)
            gif(anim, joinpath(dir, "traj_$k.gif"), fps = 10)

            idx = LinRange(1, ntsave, 6) .|> Base.Fix1(round, Int)

            for (i, id) in enumerate(idx)
                _u_re = reshape(_u, Nx, Ny, :)

                p1 = heatmap(_x, _y, _u_re[:, :, id]';
                    title = "Advection 2D Data", xlabel, ylabel)

                p2 = meshplt(x_re, y_re, _u_re[:, :, id])

                png(p1, joinpath(dir, "heatmap_$i"))
                png(p2, joinpath(dir, "meshplt_$i"))
            end
        end
    end

    (sol, V), (x, u, t, mu)
end

Nx, Ny = 128, 128

intx = IntervalDomain(-1.0f0, 1.0f0; periodic = true)
inty = IntervalDomain(-1.0f0, 1.0f0; periodic = true)
domain = intx × inty

ν = 0f0
cx, cy = 0.25f0, 0.25f0
tspan = (0f0, 4.0f0)
ntsave = 500
mu = nothing

device = cpu_device()
dir = joinpath(@__DIR__, "data_advect")
(sol, V), (x, u, t, mu) = advect1D(Nx, Ny, ν, cx, cy, mu;
    domain, tspan, ntsave, device, dir,)

nothing
#
