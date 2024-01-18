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
using OrdinaryDiffEq, LinearSolve, LinearAlgebra
using Plots, JLD2
using Test

CUDA.allowscalar(false)

nx = 32
ny = 32
ν = 0e0
p = nothing

""" space discr """
V = FourierSpace(nx, ny)
discr = Collocation()

(x,y) = points(V)

""" operators """
A = -diffusionOp(ν, V, discr)

vx = 1.0; velx = @. x*0 + vx
vy = 1.0; vely = @. x*0 + vy
C = advectionOp((velx, vely), V, discr)
F = -C

A = cache_operator(A, x)
F = cache_operator(F, x)

""" IC """
uIC(x,y) = @. sin(1x) * sin(1y)
u0 = uIC(x,y)

""" time discr """
tspan = (0.0, 10.0)
tsave = (0, π/4, π/2, 3π/4, 2π,)
odealg = Tsit5()
prob = SplitODEProblem(A, F, u0, tspan, p)
@time sol = solve(prob, odealg, saveat=tsave, abstol=1e-8, reltol=1e-8)

""" analysis """
pred = Array(sol)

utrue(x, y, vx, vy, t) = uIC(x .- vx*t, y .- vy*t)
utr = utrue(x,y,vx,vy,sol.t[1])
for i=2:length(sol.u)
    ut = utrue(x, y, vx, vy, sol.t[i])
    global utr = hcat(utr, ut)
end

function meshplt(x,y,u;a=45,b=30)
    p = plot(x,y,u,legend=false,c=:grays,camera=(a,b))
    p = plot!(x',y',u',legend=false,c=:grays,camera=(a,b))
    return p
end

x_re = reshape(x, nx, ny)
y_re = reshape(y, nx, ny)
ut_re = reshape(utr, nx, ny, :)

plt = meshplt(x_re, y_re, ut_re[:,:,1])
display(plt)

# anim = animate(pred, V)
# filename = joinpath(dirname(@__FILE__), "advect" * ".gif")
# gif(anim, filename, fps=5)

err = norm(pred .- utr,Inf)
@test err < 1e-8
#
