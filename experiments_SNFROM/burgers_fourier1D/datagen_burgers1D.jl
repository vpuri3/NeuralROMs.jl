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

function uic(x, mu)
    x = vec(x)
    mu = reshape(mu, 1, :)

    u = @. 1 + mu/2 * (sin(2_pi * x - _pi/2) + 1)
    u[x .> 1f0, :] .= 1

    reshape(u, length(x), length(mu))
end

function burgers_inviscid(N, mu, tspan;
    p = nothing,
    ntsave=500,
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
    u0 = uic(x, mu)
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
    ν = 1f-4
    A = -diffusionOp(ν, V, discr)
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

        if isdir(dir)
            rm(dir; recursive = true)
        end
        mkdir(dir)

        readme = joinpath(dir, "README")
        readmeio = open(readme, "w")
        write(readmeio, "N = $N, tspan = $tspan, mu = $mu.")
        close(readmeio)

        metadata = (; ν, readme = "u [Nx, Nbatch, Nt]")

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

N = 1024
mu = [0.500, 0.525, 0.550, 0.575, 0.600, 0.625]
tspan = (0.f0, 0.5f0)
dir = joinpath(@__DIR__, "data_burg1D")
device = gpu_device()
(sol, V), (x, u, t, mu) = burgers_inviscid(N, mu, tspan; device, dir)

nothing
#
