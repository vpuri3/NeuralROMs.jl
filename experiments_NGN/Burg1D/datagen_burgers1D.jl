#
using FourierSpaces
using NeuralROMs

using CUDA, LuxCUDA, LuxDeviceUtils
using OrdinaryDiffEq, LinearSolve, LinearAlgebra, Random
using Plots, JLD2
using BenchmarkTools

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
    @btime solve($prob, $odealg, saveat=$tsave)

    x = x     |> cpu_device()
    u = sol   |> Array
    t = sol.t |> cpu_device()
    V = V     |> cpu_device()

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
        jldsave(filename; x, u, t, mu, metadata)

        for k in 1:size(u, 2)
            _u = @view u[:, k, :]

            xlabel = "x"
            ylabel = "u(x, t)"
            title = "Case $k"

            # anim = animate1D(_u, x, t; linewidth=2, xlabel, ylabel, title)
            # gif(anim, joinpath(dir, "traj_$(k).gif"), fps=30)

            begin
                idx = LinRange(1f0, ntsave, 11) .|> Base.Fix1(round, Int)
                plt = plot(;title, xlabel, ylabel, legend = false)
                plot!(plt, x, _u[:, idx])
                png(plt, joinpath(dir, "traj_$(k)"))
                display(plt)
            end
        end
    end

    (sol, V), (x, u, t, mu)
end

N = 8192
mu = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
tspan = (0.f0, 0.5f0)
dir = joinpath(@__DIR__, "data_burg1D")
device = gpu_device()
(sol, V), (x, u, t, mu) = burgers_inviscid(N, mu, tspan; device, dir)

nothing
#
