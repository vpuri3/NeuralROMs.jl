#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#

function gabor(x, α, x̄, σ, ω, ϕ,)
    z = @. (x - x̄) / abs(σ)
    gaussian  = @. exp(-0.5 * z^2)
    sinusodal = @. cos(2 * pi * ω * z + ϕ)

    α * gaussian .* sinusodal
end

function scaled_tanh(x, a, b, ω, x̄)
    u = tanh(ω * (x - x̄)) # [-1, 1]
    scale = (b - a) / 2
    shift = (b + a) / 2
    scale * u + shift
end

function squarewave(x, x0, x1, ω0, ω1)
    # a = x̄ - w
    # b = x̄ + w

    u1 = @.  scaled_tanh(x, -0.5, 0.5, ω0, x0)
    u2 = @. -scaled_tanh(x, -0.5, 0.5, ω1, x1)
    u1 + u2
end

function rswaf(x, x̄, w, ω0, ω1, α)
    x0 = x̄ - w
    x1 = x̄ + w

    u1 = @. tanh(ω0 * (x - x0))
    u2 = @. tanh(ω1 * (x - x1))

    0.5 * (u1 - u2) * α
end

#======================================================#
x = LinRange(-1, 1, 1024)

ω = 20
y = 0
y1 = rswaf(x, 0, 0.5, 20, 20, 0.8)


# y1 = rswaf(x, -0.5, 0.4, 10, 50)
# y2 = rswaf(x,  0.1, 0.2, 10, 20)
# y3 = rswaf(x, -0.1, 0.1, 50, 50)
# y4 = rswaf(x, -0.5, 0.5, 50, 50) + rswaf(x, -0.25, 0.75, 50, 50)
# y5 = rswaf(x,  0.5, -0.4, 10, 50)
# y6 = rswaf(x,  0.0, 0.0, 20, 10)
#
plt = plot()
plot!(plt, x, y1, w = 4)
# plot!(plt, x, y2, w = 4)
# plot!(plt, x, y3, w = 4)
# # plot!(plt, x, y4, w = 4)
# # plot!(plt, x, y5, w = 4)
# plot!(plt, x, y6, w = 4)
#
display(plt)
#======================================================#

# function plot_multigabor(x, Ng, Nf, σfactor)
#     x0, x1 = extrema(x)
#     span = (x1-x0) / Ng
#
#     x̄ = LinRange(-1 + span/2, 1-span/2, Ng)
#
#     plt = plot()
#     gs = zeros(length(x), Nf, Ng)
#
#     for j in 1:Ng
#         for i in 1:Nf
#             α = 1 / Nf
#             # α = 2 * rand() - 1
#             ω = (i - 1)
#             ϕ = 0
#             gs[:, i, j] = gabor(x, α, x̄[j], span / σfactor, ω, ϕ)
#             plot!(plt, x, gs[:, i, j], c = :black, w = 2)
#         end
#     end
#
#     plot!(plt, x, sum(gs, dims = 2:3) |> vec;
#         c = :red, w = 2, ylims = (-1,1), legend = false
#     )
#     plt
# end

#======================================================#
nothing
