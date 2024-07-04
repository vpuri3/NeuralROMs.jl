#
using FourierSpaces
using NeuralROMs
using BenchmarkTools

let
    # add test dependencies to env stack
    pkgpath = dirname(dirname(pathof(NeuralROMs)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using CUDA, LuxCUDA, LuxDeviceUtils, ComponentArrays
using OrdinaryDiffEq, LinearSolve, LinearAlgebra, Random
using Plots, JLD2, LaTeXStrings, Setfield

Random.seed!(0)
CUDA.allowscalar(false)

T = Float32
_pi = Float32(pi)

function uIC(x, y; μ = 0.9) # 0.9 - 1.1
    u = @. μ * sin(2π * x) * sin(2π * y)

    u[x .< 0.0] .= 0
    u[y .< 0.0] .= 0

    u[x .> 0.5] .= 0
    u[y .> 0.5] .= 0

    ComponentArray(;vx = u, vy = copy(u))
end

function burgers2D(Nx, Ny, ν, mu = [0.9], p = nothing;
    tspan=(0.f0, 0.5f0),
    ntsave=500,
    odealg=SSPRK43(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
)
    N = Nx * Ny

    abstol = reltol = 1e-6 |> T

    """ space discr """
    interval = IntervalDomain(-0.25f0, 0.75f0; periodic = true)
    domain = interval × interval
    V = FourierSpace(Nx, Ny; domain) |> T
    discr = Collocation()
    x, y = points(V)

    """ IC """
    u0 = uIC(x,y) .|> T
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

    Cx = advectionOp((zero(x), zero(x)), V, discr;
        vel_update_funcs = (
            (v,u,p,t) -> copy!(v, p.vel.vx),
            (v,u,p,t) -> copy!(v, p.vel.vy),
        ),
    )

    Cy = advectionOp((zero(x), zero(x)), V, discr;
        vel_update_funcs = (
            (v,u,p,t) -> copy!(v, p.vel.vx),
            (v,u,p,t) -> copy!(v, p.vel.vy),
        ),
    )

    Dtx = cache_operator(Ax - Cx, x)
    Dty = cache_operator(Ay - Cy, x)

    function ddt(u, p, t)
        ps = ComponentArray(vel=u)

        dvx = Dtx(u.vx, ps, t)
        dvy = Dty(u.vy, ps, t)

        ComponentArray(; vx = dvx, vy = dvy)
    end

    """ time discr """
    tspan = tspan .|> T
    prob  = ODEProblem(ddt, u0, tspan, p; abstol, reltol)

    # solve
    timeFOM  = @belapsed CUDA.@sync $solve($prob, $odealg)
    statsFOM = CUDA.@timed solve(prob, odealg)

    @set! statsFOM.value = nothing
    @set! statsFOM.time  = timeFOM
    @show statsFOM.time

    statsFOM
end

FOMstats = begin
    Nx = Ny = 512
    ν = 1f-3
    device = gpu_device()
    odealg = SSPRK43()
    burgers2D(Nx, Ny, ν; device, odealg)
end
#
