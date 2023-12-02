#
using FourierSpaces
using GeometryLearning

let
    # add dependencies to env stack
    pkgpath = dirname(dirname(pathof(GeometryLearning)))
    tstpath = joinpath(pkgpath, "test")
    !(tstpath in LOAD_PATH) && push!(LOAD_PATH, tstpath)
end

using OrdinaryDiffEq, LinearAlgebra, Plots
using Plots

N = 128
ν = 0f0
c = 0.25f0
p = nothing

""" space discr """
domain = IntervalDomain(-1f0, 1f0; periodic = true)
V = FourierSpace(N; domain) |> Float32
discr = Collocation()

(x,) = points(V)
(k,) = modes(V)

""" operators """
A = -diffusionOp(ν, V, discr)
C = advectionOp((fill(c, N),), V, discr)
L = cache_operator(A - C, x)

""" IC """
function uIC(x; μ=-0.5f0, σ=0.1f0)
    @. exp(-1f0/2f0 * ((x-μ)/σ)^2)
end
u0 = uIC(x)

""" time discr """
tspan = (0.0f0, 4.0f0)
tsave = LinRange(tspan..., 11)
odealg = Tsit5()
prob = ODEProblem(L, u0, tspan, p)
@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-9, reltol=1e-9)

""" analysis """
pred = Array(sol)
plt = plot()
for i=1:length(sol.u)
    plot!(plt, x, sol.u[i], legend=true)
end
display(plt)
#
