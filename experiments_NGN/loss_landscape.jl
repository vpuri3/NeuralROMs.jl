using LinearAlgebra
using Plots

function loss(upred, utrue)
    sum(abs2, utrue - upred) / length(utrue)
end

N  = 256
Np = 128

#====================================================#
x   = LinRange(-1.0, 1.0 , N ) |> Array
μs  = LinRange(-1.0, 1.0 , Np) |> Array
σs  = LinRange( 0.0, 0.5 , Np) |> Array
σis = LinRange( 0.0, 30.0, Np) |> Array

function f1(x, μ, σ)
    @. exp(-0.5 * ((x - μ) / σ)^2)
end

function f2(x, μ, σi)
    @. exp(-0.5 * ((x - μ) * σi)^2)
end

l1 = zeros(Np, Np)
l2 = zeros(Np, Np)
utrue = f1(x, 0, 0.1)

for j in axes(l1, 2)
    for i in axes(l1, 1)

        μ = μs[i]
        σ = σs[j]
        σi = σis[j]

        l1[i, j] = loss(f1(x, μ, σ ), utrue)
        l2[i, j] = loss(f2(x, μ, σi), utrue)
    end
end

#====================================================#
p1 = contourf(μs, σs, l1'; xlabel = "μ", ylabel = "σ",)
p2 = contourf(μs, σis, l2'; xlabel = "μ", ylabel = "1/σ",)
plt = plot(p1, p2; size = (800, 400))

# Looks like learning σ is easier than learning (1/σ) based on the loss landscape
#====================================================#
