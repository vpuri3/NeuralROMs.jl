#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#

function scaled_tanh(x, a, b, ω, x̄)
    u = tanh_fast(ω * (x - x̄)) # [-1, 1]
    scale = (b - a) / 2
    shift = (b + a) / 2
    scale * u + shift
end

function rswaf(x, x̄, w, ω0, ω1, α)
    x0 = x̄ - w
    x1 = x̄ + w

    u1 = @. tanh_fast(ω0 * (x - x0))
    u2 = @. tanh_fast(ω1 * (x - x1))

    0.5 * (u1 - u2) * α
end

#======================================================#
# Splitting Kernels in two
#======================================================#
x = LinRange(-1, 1, 1024)

plts = ()

for (w, ω1, ω2) in (
    (0.5, 10, 50),
    (0.5,  2, 50),
    (0.1,  2, 50),
    (0.1, 10, 10),
)
    local c  = 1.0
    local x0 = 0.0
    local ω = min(ω1, ω2)
    
    local y1 = rswaf(x, x0, w, ω1, ω2, c)
    local y2 = rswaf(x, x0-w/2, w/2, ω1, ω, c) # left
    local y3 = rswaf(x, x0+w/2, w/2, ω, ω2, c) # right
    local y4 = y2 + y3

    local plt = plot(; legend = false)
    plot!(plt, x, y1, w = 4, c = :black)
    plot!(plt, x, y2, w = 2, c = :red)
    plot!(plt, x, y3, w = 2, c = :red)
    plot!(plt, x, y4, w = 2, c = :magenta)

    @show sum(abs2, y1 - y4)

    global plts = (plts..., plt)
end

p = plot(plts...)
display(p)

#======================================================#
# Expressivity of Tanh kernels
#======================================================#
# x = LinRange(-1, 1, 1024)
# y1 = rswaf(x, -0.5, 0.3, 20, 20, 1.0)
# y2 = rswaf(x,    0, 0.1, 20, 30, 0.8)
# y3 = rswaf(x,  0.8, 0.1, 50, 50, 1.0)
# y4 = rswaf(x,    0, 0.5,  1.8, 50, 0.5)
#
# plt = plot(; legend = false)
# plot!(plt, x, y1, w = 4)
# plot!(plt, x, y2, w = 4)
# plot!(plt, x, y3, w = 4)
# plot!(plt, x, y4, w = 4)
# display(plt)

#======================================================#
# Expanse of Tanh kernels
#======================================================#
# x = LinRange(-1, 1, 1024)
# ω = [5 10 20 50] # LinRange(0, 50, N)'
# y = tanh_fast.(x * ω)
#
# plt = plot()
# cs = [:red, :green, :blue, :black, :brown, :cyan, :magenta]
# for i in reverse(1:length(ω))
#     w = ω[i]
#     s = 4 / w
#
#     c = cs[i]
#     plot!(plt, x, y[:, i]; w = 2, label = "ω = $w", c)
#     vline!(plt, [-s, s]; w = 1, s = :dash, c, label = nothing)
# end
# display(plt)

#======================================================#
nothing
