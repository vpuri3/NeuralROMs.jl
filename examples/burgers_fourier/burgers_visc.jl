#
using FourierSpaces

using CUDA, OrdinaryDiffEq, LinearAlgebra, Random
using Plots, BSON

Random.seed!(1)
CUDA.allowscalar(false)

N = 1024
K = 100
ν = 1f-4
p = nothing

odecb = begin
    function affect!(int)
        if int.iter % 10 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function burgers_visc1d(N, K, ν, p;
    tspan=(0.f0, 10.f0),
    tsave=100,
    odealg=SSPRK43(),
    device = cpu,
)

    """ space discr """
    domain = IntervalDomain(-1, 1; periodic = true)
    V = FourierSpace(N; domain) |> Float32
    discr = Collocation()

    (k,) = modes(V)
    (x,) = points(V)

    """ IC """
    u0 = rand(Float32, N, K)
    V = make_transform(V, u0; p=p)
    u0 = begin
        F = transformOp(V)
        u0h = F * u0
        # u0h[1, :] .= 0
        u0h[10:end, :] .= 0
        F \ u0h
    end

    """ move to device """
    V  = V  |> device
    u0 = u0 |> device

    """ operators """
    function burgers!(v, u, p, t)
        copyto!(v, u)
    end

    function forcing!(f, u, p, t)
#       f .= 1e-2*rand(length(f))
        lmul!(false, f)
    end

    # model setup
    A = -diffusionOp(ν, V, discr)
    C = advectionOp((zero(u0),), V, discr; vel_update_funcs! = (burgers!,))
    F = forcingOp(zero(u0), V, discr; f_update_func! = forcing!)

    odefunc = cache_operator(A-C+F, u0)

    """ time discr """
    tsave = range(tspan...; length=tsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-6)

    """ solve """
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)
    @show sol.retcode

    x = points(V)[1] |> cpu
    u = sol   |> Array
    t = sol.t |> cpu
    V = V     |> cpu

    (sol, V), (x, u, t,)
end

(sol, V), (x, u, t) = burgers_visc1d(N, K, ν, p; device = gpu)

dir = joinpath(@__DIR__, "burg_visc_re10k_traveling")
mkpath(dir)

name = joinpath(dir, "data.bson")
BSON.@save name x u

for k in 1:10
    uk = @view u[:, k, :]
    anim = animate(uk, V, t, legend=false, linewidth=2, color=:black, xlabel="x", ylabel="u(x,t)")
    gif(anim, joinpath(dir, "traj_$(k).gif"), fps=15)
end

nothing
#
