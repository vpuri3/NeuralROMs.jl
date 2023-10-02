#
using FourierSpaces

using CUDA, OrdinaryDiffEq, LinearAlgebra, Random
using Plots, BSON

Random.seed!(0)
CUDA.allowscalar(false)

N = 1024
K = 1
p = nothing

function uIC(V::FourierSpace; mu = 0.5)
    N = length(V)
    x = points(V)[1]

    u = @. mu / 2 * (sin(2pi * x))

    u = x * 0 .+ 1
end

odecb = begin
    function affect!(int)
        if int.iter % 1 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function burgers_visc1d(N, K, p;
    tspan=(0.f0, 0.f5),
    tsave=100,
    odealg=SSPRK43(),
    device = cpu,
)

    """ space discr """
    domain = InvervalDomain(0, 2; periodic = true)
    V = FourierSpace(N; domain) |> Float32
    discr = Collocation()

    (k,) = modes(V)
    (x,) = points(V)

    """ IC """
    u0 = [uIC(V) for i=1:nsims]
    u0 = hcat(u0...)
    V = make_transform(V, u0; p=p)

    """ operators """
    function burgers!(v, u, p, t)
        copyto!(v, u)
    end

    _pi = Float32(pi)
    function forcing!(f, u, p, t)
        f .= (x .+ _pi) ./ 2_pi
        # lmul!(false, f)
    end

    # model setup
    C = advectionOp((zero(u0),), V, discr; vel_update_funcs! = (burgers!,))
    F = forcingOp(zero(u0), V, discr; f_update_func! = forcing!)

    odefunc = cache_operator(-C+F, u0)

    V  = V  |> device
    u0 = u0 |> device

    """ time discr """
    tsave = range(tspan...; length=nsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=1f-6, abstol=1f-6)

    """ solve """
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)
    @show sol.retcode

    x = x     |> cpu
    u = sol   |> Array
    t = sol.t |> cpu
    V = V     |> cpu

    (sol, V), (x, u, t,)
end

(sol, V), (x, u, t) = burgers_visc1d(N, p; device = cpu, nsims = 2)

plt = plot(x, @view u[:, 1, :])
display(plt)

# dir = joinpath(@__DIR__, "invisc_burg_re01k")
# mkpath(dir)
#
# name = joinpath(dir, "data.bson")
# BSON.@save name x u
#
# for k in 1:10
#     uk = @view u[:, k, :]
#     anim = animate(uk, V, t, legend=false, linewidth=2, color=:black, xlabel="x", ylabel="u(x,t)")
#     gif(anim, joinpath(dir, "traj_$(k).gif"), fps=20)
# end

nothing
#
