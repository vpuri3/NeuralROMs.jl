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
using OrdinaryDiffEq, LinearSolve, LinearAlgebra, Random
using Plots, JLD2

Random.seed!(0)
CUDA.allowscalar(false)

_pi = Float32(pi)

odecb = begin
    function affect!(int)
        if int.iter % 1 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function burgers_inviscid(N, mu = LinRange(0.6, 1.2, 7)', p = nothing;
    tspan=(0.f0, 0.5f0),
    ntsave=1000,
    odealg=SSPRK43(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
)

    """ space discr """
    domain = IntervalDomain(0, 2; periodic = true)
    V = FourierSpace(N; domain) |> Float32
    discr = Collocation()

    (k,) = modes(V)
    (x,) = points(V)

    """ IC """
    u0 = begin
        u = @. 1 + mu/2 * (sin(2_pi * x - _pi/2) + 1)
        u[x .> 1f0, :] .= 1
        u
    end
    V  = make_transform(V, u0; p=p)

    """ move to device """
    V  = V  |> device
    u0 = u0 |> device

    """ operators """
    function burgers!(v, u, p, t)
        copyto!(v, u)
    end

    function forcing!(f, u, p, t)
        # f .= (x .+ _pi) ./ 2_pi
        # lmul!(false, f)
    end

    # model setup
    A = -diffusionOp(1f-4, V, discr)
    C = advectionOp((zero(u0),), V, discr;
        vel_update_funcs! = (burgers!,), truncation_fracs = (2//3,))
    F = forcingOp(zero(u0), V, discr; f_update_func! = forcing!)

    odefunc = cache_operator(A-C+F, u0)

    """ time discr """
    tsave = range(tspan...; length=ntsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-5)

    """ solve """
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)#, dt=1f-5)
    @show sol.retcode

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

    if !isnothing(dir)
        mkpath(dir)
        metadata = (; readme = "u [Nx, Nbatch, Nt]")

        # filename = joinpath(dir, "data.bson")
        # BSON.@save filename x u t mu metadata

        filename = joinpath(dir, "data.jld2")
        jldsave(filename; x, u, udx, ud2x, t, mu, metadata)

        for k in 1:size(u, 2)
            It = LinRange(1, ntsave, 100) .|> Base.Fix1(round, Int)
            _u = @view u[:, k, It]
            _udx = @view udx[:, k, It]
            _ud2x = @view ud2x[:, k, It]

            anim = animate1D(_u, x, t; linewidth=2, xlabel="x", ylabel="u(x,t)", title = "μ = $(round(mu[k]; digits = 2)), ")
            gif(anim, joinpath(dir, "traj_$(k).gif"), fps=30)

            anim = animate1D(_udx, x, t; linewidth=2, xlabel="x", ylabel="u(x,t)", title = "μ = $(round(mu[k]; digits = 2)), ")
            gif(anim, joinpath(dir, "traj_dx_$(k).gif"), fps=30)

            anim = animate1D(_ud2x, x, t; linewidth=2, xlabel="x", ylabel="u(x,t)", title = "μ = $(round(mu[k]; digits = 2)), ")
            gif(anim, joinpath(dir, "traj_d2x_$(k).gif"), fps=30)
        end
    end

    (sol, V), (x, u, t, mu)
end

N = 8192
dir = joinpath(@__DIR__, "visc_burg_param_ic", "burg_visc_re10k")
device = gpu_device()
linsolve = KrylovJL_GMRES()
# odealg = # ImplicitEuler(; linsolve) # Tsit5()
odealg = SSPRK43()
(sol, V), (x, u, t, mu) = burgers_inviscid(N; device, dir, odealg)

nothing
#
