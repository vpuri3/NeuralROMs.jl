using Plots, NNlib

"""
Support of function is `r ∈ [0, 1/ϵ]`
"""
function wendland4(r, ϵ)
	ϵr = @. ϵ * r
	return @. relu(1 - ϵr)^6 * (35*ϵr^2 + 18*ϵr + 3)
end

N = 1000
x = LinRange(0, 3, N)

plt = plot(; title = "Wendland C4")

for ϵ in [5, 2, 1, 0.5, 0.2, 0.1, 0.05]
	plot!(plt, x, wendland4(x, ϵ), label = "ϵ=$(ϵ)", w = 4)
end

plt
