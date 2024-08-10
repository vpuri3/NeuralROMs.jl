#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#
function gabor(
    x::AbstractArray,
    # scaling
    α,
    # gaussian mean, var
    x̄,
    σ,
    # sinusodal freq, shift
    ω,
    ϕ,
)
    z = @. (x - x̄) / abs(σ)
    gaussian  = @. exp(-0.5 * z^2)
    sinusodal = @. cos(2 * pi * ω * z + ϕ)

    α * gaussian .* sinusodal
end


x = LinRange(-2, 2, 1024)

α, x̄, σ, ω, ϕ = 1, 0, 1/2, 0, 0
u0 = gabor(x, α, x̄, σ, ω, ϕ)

α, x̄, σ, ω, ϕ = 1, 0, 1/2, 0.2, pi/2
u1 = gabor(x, α, x̄, σ, ω, ϕ)

α, x̄, σ, ω, ϕ = 1, 0, 1/2, 0.5, pi/2
u2 = gabor(x, α, x̄, σ, ω, ϕ)

plt = plot()
plot!(plt, x, u0, w=3, c = :black)
plot!(plt, x, u1, w=3, c = :blue)
plot!(plt, x, u2, w=3, c = :red)

display(plt)
nothing
