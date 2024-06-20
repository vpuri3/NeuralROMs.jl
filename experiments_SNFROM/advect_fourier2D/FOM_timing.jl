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
    # odealg=SSPRK43(),
    odekw = (;),
    device = gpu_device(),
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
    prob = ODEProblem{false}(odefunc, u0, tspan, p)
    # prob = ODEProblem{false}(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-6)

    # solve
    sol = if device isa LuxDeviceUtils.AbstractLuxGPUDevice
        CUDA.@time sol = solve(prob, odealg)
    else
        @time sol = solve(prob, odealg)
    end

    nothing
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

device = gpu_device()
advect1D(Nx, Ny, ν, cx, cy, mu; domain, tspan, ntsave, device)

nothing
#
