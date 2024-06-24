#
using FourierSpaces
using NeuralROMs

let
    # add test dependencies to env stack
    pkgpath = dirname(dirname(pathof(NeuralROMs)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using CUDA, LuxCUDA, LuxDeviceUtils, ComponentArrays
using OrdinaryDiffEq, LinearSolve, LinearAlgebra, Random
using Plots, JLD2, LaTeXStrings

Random.seed!(0)
CUDA.allowscalar(false)

T = Float32
_pi = Float32(pi)

odecb = begin
    function affect!(int)
        if int.iter % 1 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function uIC(x, y, μ) # 0.9 - 1.1
    u = @. μ * sin(2π * x) * sin(2π * y) # [N, K]

    u[x .< 0.0f0, :] .= 0
    u[y .< 0.0f0, :] .= 0

    u[x .> 0.5f0, :] .= 0
    u[y .> 0.5f0, :] .= 0

    ComponentArray(; vx = u, vy = copy(u))
end

function burgers2D(Nx, Ny, ν, mu = nothing, p = nothing;
    tspan=(0.f0, 0.5f0),
    ntsave=500,
    odealg=SSPRK43(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
)
    if isnothing(mu)
        mu = LinRange(0.9f0, 1.1f0, 7)
    end
    mu = reshape(mu, 1, :)

    abstol = reltol = 1e-6 |> T

    """ space discr """
    interval = IntervalDomain(-0.25f0, 0.75f0; periodic = true)
    domain = interval × interval
    V = FourierSpace(Nx, Ny; domain) |> T
    discr = Collocation()
    x, y = points(V)

    """ IC """
    u0 = uIC(x,y, mu) .|> T
    ps = ComponentArray(vel=u0)
    V = make_transform(V, u0.vx; p=ps)

    """ move to device """
    V  = V  |> device
    u0 = u0 |> device
    ps = ps |> device

    x, y = points(V)

    """ operators """
    Ax = -diffusionOp(ν, V, discr)
    Ay = -diffusionOp(ν, V, discr)

    Cx = advectionOp((zero(u0.vx), zero(u0.vy)), V, discr;
        vel_update_funcs = (
            (v,u,p,t) -> copy!(v, p.vel.vx),
            (v,u,p,t) -> copy!(v, p.vel.vy),
        ),
    )

    Cy = advectionOp((zero(u0.vx), zero(u0.vy)), V, discr;
        vel_update_funcs = (
            (v,u,p,t) -> copy!(v, p.vel.vx),
            (v,u,p,t) -> copy!(v, p.vel.vy),
        ),
    )

    Dtx = cache_operator(Ax - Cx, u0.vx)
    Dty = cache_operator(Ay - Cy, u0.vy)

    function ddt(u, p, t)
        ps = ComponentArray(vel=u)

        dvx = Dtx(u.vx, ps, t)
        dvy = Dty(u.vy, ps, t)

        ComponentArray(; vx = dvx, vy = dvy)
    end

    """ time discr """
    tspan = tspan .|> T
    tsave = range(tspan...; length = ntsave) .|> T
    prob  = ODEProblem(ddt, u0, tspan, p)

    @time sol = solve(prob, odealg; saveat = tsave, abstol, reltol,
        callback = odecb,)
    @show sol.retcode

    u = sol.u |> cpu_device()
    t = sol.t |> cpu_device()
    V = V     |> cpu_device()
    x, y = points(V)

    u = hcat(u...)

    Np = Nx * Ny
    Nt = length(t)
    Nb = length(mu)

    Ng = Np * Nb
    vx = reshape(u[1:Ng    , :], 1, Np, Nb, Nt)
    vy = reshape(u[1+Ng:end, :], 1, Np, Nb, Nt)

    u = cat(vx, vy; dims = 1)

    if !isnothing(dir)
        mkpath(dir)

        mu = isnothing(mu) ? fill(nothing, size(u, 2)) |> Tuple : mu
        x_save = reshape(vcat(x', y'), 2, Np)
        grid = (Nx, Ny,)
        metadata = (; Nx, Ny, ν, grid, readme = "u [Nx, Nbatch, Nt]")

        filename = joinpath(dir, "data.jld2")
        jldsave(filename; x = x_save, u, t, mu, metadata)

        for case in 1:size(u, 3)
            x_re = reshape(x , Nx, Ny)
            y_re = reshape(y , Nx, Ny)

            _x = x_re[:, 1]
            _y = y_re[1, :]

            It = LinRange(1, ntsave, 90) .|> Base.Fix1(round, Int)

            vx_re = reshape(u[1,:,case,:], Nx, Ny, :)
            vy_re = reshape(u[2,:,case,:], Nx, Ny, :)

            V1D = FourierSpace(Nx; domain = interval)
            vx_slice = Tuple(diag(vx_re[: ,:, i]) for i in It)
            vx_slice = hcat(vx_slice...)
            
            anim = animate(vx_slice, V1D, t; w = 2.0, title = "vx(x=y)")
            filename = joinpath(dir, "burgers_case$(case)_slice3" * ".gif")
            gif(anim, filename, fps = 30)

            ############

            # anim = animate(reshape(vx_re, N, :), V, sol.t)
            # filename = joinpath(dir, "burgers_vx" * ".gif")
            # gif(anim, filename, fps=100)
            #
            # anim = animate(reshape(vy_re, N, :), V, sol.t)
            # filename = joinpath(dir, "burgers_vy" * ".gif")
            # gif(anim, filename, fps=100)

            It = LinRange(1, ntsave, 4) .|> Base.Fix1(round, Int)

            for (i, id) in enumerate(It)
                _u_re = vx_re[:, :, id]

                p1 = heatmap(_x, _y, _u_re'; xlabel = L"x", ylabel = L"y", cmap = :viridis,)
                png(p1, joinpath(dir, "heatmap_case$(case)_$i"))
            end
        end
    end

    (sol, V), (x, u, t, mu)
end

Nx = Ny = 512
ν = 1f-3
dir = joinpath(@__DIR__, "data_burgers2D")
device = gpu_device()
odealg = SSPRK43()
(sol, V), (x, u, t, mu) = burgers2D(Nx, Ny, ν; device, dir, odealg)

nothing
#
