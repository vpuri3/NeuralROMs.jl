#
using LinearAlgebra, Plots

kw = (; grid = false, legend = false, xaxis = false, yaxis = false)

x  = LinRange(-1, 1, 10_000) |> Array

lims = (-1.3, 1.3,)
u1 = @. sin(π * x)
p1 = plot(x, u1; c = :black, w = 48, kw..., xlims = lims, ylims = lims)
f1 = joinpath(@__DIR__, "sine")
png(p1, f1)

σ  = 0.25
u2 = @. exp(-1/2 * (x/σ)^2)

p2 = plot(x, u2; c = :green, w = 24, kw...)

p2 = plot!([σ, σ], [0, 0.6]; c = :black, w = 16, kw...)
p2 = plot!([-σ, -σ], [0, 0.6]; c = :black, w = 16, kw...)

p2 = plot!([2σ, 2σ], [0, 0.2]; c = :black, w = 16, kw...)
p2 = plot!([-2σ, -2σ], [0, 0.2]; c = :black, w = 16, kw...)

f2 = joinpath(@__DIR__, "gaussian")
png(p2, f2)

display(p2)

nothing
