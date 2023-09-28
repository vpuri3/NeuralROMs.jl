#
using FourierSpaces

using CUDA, OrdinaryDiffEq, LinearAlgebra, Random
using Plots, BSON

Random.seed!(0)
CUDA.allowscalar(false)

N = 1024
Nmodes = 3
ν = 1f-3
p = nothing

function uIC(V::FourierSpace)
    x = points(V)[1]
    X = truncationOp(V, (Nmodes / N,))

    u0 = if x isa CUDA.CuArray
        X * CUDA.rand(size(x)...)
    else
        X * rand(size(x)...)
    end

    u0
end

odecb = begin
    function affect!(int)
        if int.iter % 10 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function burgers_visc1d(N, ν, p;
    uIC=uIC,
    tspan=(0.f0, 10.f0),
    nsims=10,
    nsave=100,
    odealg=SSPRK43(),
    device = cpu,
)

    """ space discr """
    V = FourierSpace(N) |> device
    discr = Collocation()

    (x,) = points(V)

    """ IC """
    u0 = [uIC(V) for i=1:nsims]
    u0 = hcat(u0...)
    V = make_transform(V, u0; p=p)

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
    tsave = range(tspan...; length=nsave)
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

(sol, V), (x, u, t) = burgers_visc1d(N, ν, p; device = gpu, nsims = 100)

dir = joinpath(@__DIR__, "visc_burg_re01k")
mkpath(dir)

name = joinpath(dir, "data.bson")
BSON.@save name x u

for k in 1:10
    uk = @view u[:, k, :]
    anim = animate(uk, V, t, legend=false, linewidth=2, color=:black, xlabel="x", ylabel="u(x,t)")
    gif(anim, joinpath(dir, "traj_$(k).gif"), fps=20)
end

nothing
#
