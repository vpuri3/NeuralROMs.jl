#
using FourierSpaces

using CUDA, OrdinaryDiffEq, LinearSolve, LinearAlgebra, Random
using Plots, BSON, GeometryLearning

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
    tsave=100,
    odealg=SSPRK43(),
    odekw = (;),
    device = cpu,
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
    tsave = range(tspan...; length=tsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-5)

    """ solve """
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)#, dt=1f-5)
    @show sol.retcode

    x = x     |> cpu
    u = sol   |> Array
    t = sol.t |> cpu
    V = V     |> cpu

    if !isnothing(dir)
        mkpath(dir)
        name = joinpath(dir, "data.bson")
        metadata = (; readme = "u [Nx, Nbatch, Nt]")
        BSON.@save name x u t mu metadata

        for k in 1:size(u, 2)
            uk = @view u[:, k, :]
            anim = animate1D(uk, x, t; linewidth=2, xlabel="x", ylabel="u(x,t)", title = "μ = $(round(mu[k]; digits = 2)), ")
            gif(anim, joinpath(dir, "traj_$(k).gif"), fps=30)
        end
    end

    (sol, V), (x, u, t, mu)
end

N = 8192
dir = joinpath(@__DIR__, "visc_burg_param_ic", "burg_visc_re10k")
device = gpu
linsolve = KrylovJL_GMRES()
odealg = # ImplicitEuler(; linsolve) # SSPRK83() # Tsit5()
odealg = SSPRK43()
(sol, V), (x, u, t, mu) = burgers_inviscid(N; device, dir, odealg)

nothing
#
