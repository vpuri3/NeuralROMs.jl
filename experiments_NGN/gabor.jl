#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#

function gaussian(x, α, x̄, σ)
    z = @. (x - x̄) / abs(σ)
    @. α * exp(-0.5 * z^2)
end

function gabor(x, α, x̄, σ, ω, ϕ,)
    z = @. (x - x̄) / abs(σ)
    gaussian  = @. exp(-0.5 * z^2)
    sinusodal = @. cos(2 * pi * ω * z + ϕ)

    α * gaussian .* sinusodal
end
#------------------------------------#

function plot_multigabor(x, Ng, Nf, σfactor)
    x0, x1 = extrema(x)
    span = (x1-x0) / Ng

    x̄ = LinRange(-1 + span/2, 1-span/2, Ng)

    plt = plot()
    gs = zeros(length(x), Nf, Ng)

    for j in 1:Ng
        for i in 1:Nf
            α = 1 / Nf
            # α = 2 * rand() - 1
            ω = (i - 1)
            ϕ = 0
            gs[:, i, j] = gabor(x, α, x̄[j], span / σfactor, ω, ϕ)
            plot!(plt, x, gs[:, i, j], c = :black, w = 2)
        end
    end

    plot!(plt, x, sum(gs, dims = 2:3) |> vec;
        c = :red, w = 2, ylims = (-1,1), legend = false
    )
    plt
end

x = LinRange(-1, 1, 1024)

p1 = plot_multigabor(x, 1, 2, 4) |> display

# plot(p1, p2, p3, p4, size = (1200, 400)) |> display

#------------------------------------#
function plot_multigaussian(x, N, σfactor)
    x0, x1 = extrema(x)
    span = (x1-x0) / N

    x̄ = LinRange(-1 + span/2, 1-span/2, N)

    plt = plot()
    gs = zeros(length(x), N)

    for i in 1:N
        gs[:, i] = gaussian(x, 1, x̄[i], span / σfactor)
        plot!(plt, x, gs[:, i], c = :black, w = 2)
    end

    plot!(plt, x, sum(gs, dims = 2); c = :red, w = 2, ylims = (0,Inf), legend = false)
    plt
end

# x = LinRange(-1, 1, 1024)
#
# p1 = plot_multigaussian(x, 3, 1)
# p2 = plot_multigaussian(x, 3, 2)
# p3 = plot_multigaussian(x, 3, 3)
# p4 = plot_multigaussian(x, 3, 4)
#
# plot(p1, p2, p3, p4, size = (1200, 400)) |> display
#------------------------------------#

#------------------------------------#
# x = LinRange(-2, 2, 1024)
#
# α, x̄, σ, ω, ϕ = 1, 0, 1/2, 0, 0
# u0 = gabor(x, α, x̄, σ, ω, ϕ)
#
# α, x̄, σ, ω, ϕ = 1, 0, 1/2, 0.2, pi/2
# u1 = gabor(x, α, x̄, σ, ω, ϕ)
#
# α, x̄, σ, ω, ϕ = 1, 0, 1/2, 0.5, pi/2
# u2 = gabor(x, α, x̄, σ, ω, ϕ)
#
# plt = plot()
# plot!(plt, x, u0, w=3, c = :black)
# plot!(plt, x, u1, w=3, c = :blue)
# plot!(plt, x, u2, w=3, c = :red)
# display(plt)
#------------------------------------#

nothing
