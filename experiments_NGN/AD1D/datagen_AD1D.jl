#
using FourierSpaces, OrdinaryDiffEq

using CUDA, LuxCUDA, LuxDeviceUtils
using OrdinaryDiffEq, LinearSolve, LinearAlgebra
using Plots, JLD2

CUDA.allowscalar(false)

function uIC_gaussian(x)
    μ = -0.5
    σ = 0.1

    @. exp(-0.5 * ((x-μ)/σ)^2)
end

function scaled_tanh(x, a, b, ω, x̄)
    u = tanh(ω * (x - x̄)) # [-1, 1]
    scale = (b - a) / 2
    shift = (b + a) / 2
    scale * u + shift
end

function uIC_square(x)
    μ = -0.5
    ω = 50
    w = 0.35

    u1 = @.  scaled_tanh(x, -0.5, 0.5, ω, μ-w)
    u2 = @. -scaled_tanh(x, -0.5, 0.5, ω, μ+w)
    u1 + u2
end

function uIC(x; mu = nothing)
    u1 = uIC_gaussian(x)
    u2 = uIC_square(x)
    
    N = isnothing(mu) ? 1 : length(mu)
    u1 = repeat(u1, 1, N)
    u2 = repeat(u2, 1, N)

    hcat(u1, u2)
end

odecb = begin
    function affect!(int)
        if int.iter % 1 == 0
            println("[$(int.iter)] \t Time $(round(int.t; digits=8))s")
        end
    end

    DiscreteCallback((u,t,int) -> true, affect!, save_positions=(false,false))
end

function advectionDiffusion1D(
    N, mu, p = nothing;
    tspan=(0.f0, 1.0f0),
    ntsave=500,
    odealg=Tsit5(),
    odekw = (;),
    device = cpu_device(),
    dir = nothing,
    T = Float32,
)

    # space discr
    domain = IntervalDomain(-T(1.5), T(1.5); periodic = true)
    V = FourierSpace(N; domain) |> T
    discr = Collocation()

    (x,) = points(V)

    # get initial condition
    u0 = uIC(x; mu) .|> T
    V  = make_transform(V, u0; p)

    # assign c, ν
    ν = zero(u0)
    c = zero(u0)

    mu = repeat(mu, 2)
    for k in eachindex(mu)
        c[:, k] .= mu[k][1]
        ν[:, k] .= mu[k][2]
    end

    # move to device
    V  = V  |> device
    c  = c  |> device
    ν  = ν  |> device
    u0 = u0 |> device

    # operators
    A = -diffusionOp(ν, V, discr)
    C = advectionOp((c,), V, discr)
    odefunc = cache_operator(A - C, u0)

    # time discr
    tol = if T === Float32
        10 * eps(T)    # ≈ 10^-6
    elseif T === Float64
        10000 * eps(T) # ≈ 10^-12
    end

    tsave = range(tspan...; length = ntsave)
    prob = ODEProblem(odefunc, u0, tspan, p; reltol=tol, abstol=tol)

    # solve
    @time sol = solve(prob, odealg, saveat=tsave, callback=odecb)
    @show sol.retcode

    # move back to device
    x = x     |> cpu_device()
    u = sol   |> Array
    t = sol.t |> cpu_device()
    V = V     |> cpu_device()

    mu = isnothing(mu) ? fill(nothing, size(u, 2)) |> Tuple : mu

    if !isnothing(dir)
        mkpath(dir)
        metadata = (; readme = "")

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

N = 512
ν = 0f0
c = 1f0
T = Float32
mu = nothing # parameters

cs = T[1,    0,    1, 2.5]
νs = T[0, 0.01, 0.01, 0]
mu = zip(cs, νs) |> collect
device = cpu_device()

dir = joinpath(@__DIR__, "data_AD1D")
(sol, V), (x, u, t, mu) = advectionDiffusion1D(N, mu; T, device, dir)

# x = LinRange(-1.5, 1.5, 256) |> Array
# u = uIC_square(x) |> vec
# scatter(x, u, ms = 5) |> display

nothing
#
