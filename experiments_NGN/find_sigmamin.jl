#
using NeuralROMs

joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_cases.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_evolve.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_NGN", "ng_models.jl") |> include

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

T = Float32
Ng = Nf = 1
σmin = 1e-4
σsplit = false
train_freq = false

NN = GaussianLayer1D(1, 1, Ng, Nf; T, σmin, σsplit, train_freq)
p, st = Lux.setup(rng, NN)
p = ComponentArray(p)

p.c .= 1.0
p.x̄ .= 0.0
p.σ .= 10^-2

md = (; x̄ = T[0], σx = T[1], ū = T[0], σu = T[1])
model = NeuralModel(NN, st, md)

x = LinRange(-1, 1, 1024)
x = reshape(x, 1, :) |> Array

ud0, ud1, ud2 = dudx2_1D(model, x, p)

@show any(isnan, ud0)
@show any(isnan, ud1)
@show any(isnan, ud2)

plt = plot()
plot!(plt, vec(x), vec(ud0); w = 3, label = "ud0")
plot!(plt, vec(x), vec(ud1); w = 3, label = "ud1")
plot!(plt, vec(x), vec(ud2); w = 3, label = "ud2")

display(plt)

nothing
