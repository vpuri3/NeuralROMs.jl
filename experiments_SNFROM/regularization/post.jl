#
using LinearAlgebra, LaTeXStrings, JLD2
using NeuralROMs

import CairoMakie
import CairoMakie: Makie

function f(x; σ = 1.0f0)
    pi32 = Float32(pi)

    @. (x - pi32/2f0) * sin(x) * exp(-(x/σ)^2)
end

function makemodel(modelfile::String)
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    ST = model["STATS"]
    close(model)
    NeuralModel(NN, st, md), p, ST
end

datafile = joinpath(@__DIR__, "data_reg.jld2")
modelfile1 = joinpath(@__DIR__, "model1", "model_08.jld2") # vanilla
modelfile2 = joinpath(@__DIR__, "model2", "model_08.jld2") # L2
modelfile3 = joinpath(@__DIR__, "model3", "model_08.jld2") # lipschitz
modelfile4 = joinpath(@__DIR__, "model4", "model_08.jld2") # weight

data = jldopen(datafile)
x, _ = data["data_"]
close(data)

# x = x[1:16:end]
x = x[1:1:end]

model1, p1, ST1 = makemodel(modelfile1)
model2, p2, ST2 = makemodel(modelfile2)
model3, p3, ST3 = makemodel(modelfile3)
model4, p4, ST4 = makemodel(modelfile4)

xbatch = reshape(x, 1, :)
autodiff = AutoForwardDiff()
ϵ = nothing

u , ud1 , ud2  = forwarddiff_deriv2(f, x)
u1, u1d1, u1d2 = dudx2_1D(model1, xbatch, p1; autodiff, ϵ) .|> vec
u2, u2d1, u2d2 = dudx2_1D(model2, xbatch, p2; autodiff, ϵ) .|> vec
u3, u3d1, u3d2 = dudx2_1D(model3, xbatch, p3; autodiff, ϵ) .|> vec
u4, u4d1, u4d2 = dudx2_1D(model4, xbatch, p4; autodiff, ϵ) .|> vec

N = length(u)
n   = sum(abs2, u)   / N |> sqrt
nd1 = sum(abs2, ud1) / N |> sqrt
nd2 = sum(abs2, ud2) / N |> sqrt

e1 = abs.(u1 - u) ./ n .+ 1f-12
e2 = abs.(u2 - u) ./ n .+ 1f-12
e3 = abs.(u3 - u) ./ n .+ 1f-12
e4 = abs.(u4 - u) ./ n .+ 1f-12

e1d1 = abs.(u1d1 - ud1) ./ n .+ 1f-12
e2d1 = abs.(u2d1 - ud1) ./ n .+ 1f-12
e3d1 = abs.(u3d1 - ud1) ./ n .+ 1f-12
e4d1 = abs.(u4d1 - ud1) ./ n .+ 1f-12

e1d2 = abs.(u1d2 - ud2) ./ n .+ 1f-12
e2d2 = abs.(u2d2 - ud2) ./ n .+ 1f-12
e3d2 = abs.(u3d2 - ud2) ./ n .+ 1f-12
e4d2 = abs.(u4d2 - ud2) ./ n .+ 1f-12

e1_s = e1' * e1 / N # |> sqrt
e2_s = e2' * e2 / N # |> sqrt
e3_s = e3' * e3 / N # |> sqrt
e4_s = e4' * e4 / N # |> sqrt

e1d1_s = e1d1' * e1d1 / N # |> sqrt
e2d1_s = e2d1' * e2d1 / N # |> sqrt
e3d1_s = e3d1' * e3d1 / N # |> sqrt
e4d1_s = e4d1' * e4d1 / N # |> sqrt

e1d2_s = e1d2' * e1d2 / N # |> sqrt
e2d2_s = e2d2' * e2d2 / N # |> sqrt
e3d2_s = e3d2' * e3d2 / N # |> sqrt
e4d2_s = e4d2' * e4d2 / N # |> sqrt

println()
println("0th derivative")

println("Zero: $e1_s")
println("L2  : $e2_s")
println("SNFL: $e3_s")
println("SNFW: $e4_s")

println()
println("1st derivative")

println("Zero: $e1d1_s")
println("L2  : $e2d1_s")
println("SNFL: $e3d1_s")
println("SNFW: $e4d1_s")

println()
println("2nd derivative")

println("Zero: $e1d2_s")
println("L2  : $e2d2_s")
println("SNFL: $e3d2_s")
println("SNFW: $e4d2_s")

#==============================================================#

xlabel = L"x"
xlabelsize = ylabelsize = 16

fig = Makie.Figure(; size = (500, 400), backgroundcolor = :white, grid = :off)
# fig = Makie.Figure(; size = (800, 400), backgroundcolor = :white, grid = :off)

ax1 = Makie.Axis(fig[1,1]; xlabel, ylabel = L"u(x)"  , xlabelsize, ylabelsize)
ax2 = Makie.Axis(fig[2,1]; xlabel, ylabel = L"u'(x)" , xlabelsize, ylabelsize)
ax3 = Makie.Axis(fig[3,1]; xlabel, ylabel = L"u''(x)", xlabelsize, ylabelsize)

colors = [:black, :orange, :green, :blue, :red,]
styles = [:solid, :solid, :dash, :dashdot, :dashdotdot,]
labels = [L"Ground truth$$", L"No regularization$$", L"$L_2$ regularization $(γ=10^{-1})$", L"Lipschitz regularization $(α=5⋅10^{-5})$", L"Weight regularization $(γ=5⋅10^{-2})$",]

kws = Tuple(
    (; color = colors[i], linestyle = styles[i], label = labels[i], linewidth = 2)
    for i in 1:5
)

Makie.lines!(ax1, x,  u; kws[1]...)
Makie.lines!(ax1, x, u1; kws[2]...)
# Makie.lines!(ax1, x, u2; kws[3]...)
Makie.lines!(ax1, x, u3; kws[4]...)
Makie.lines!(ax1, x, u4; kws[5]...)

Makie.lines!(ax2, x,  ud1; kws[1]...)
Makie.lines!(ax2, x, u1d1; kws[2]...)
# Makie.lines!(ax2, x, u2d1; kws[3]...)
Makie.lines!(ax2, x, u3d1; kws[4]...)
Makie.lines!(ax2, x, u4d1; kws[5]...)

Makie.lines!(ax3, x,  ud2; kws[1]...)
Makie.lines!(ax3, x, u1d2; kws[2]...)
# Makie.lines!(ax3, x, u2d2; kws[3]...)
Makie.lines!(ax3, x, u3d2; kws[4]...)
Makie.lines!(ax3, x, u4d2; kws[5]...)

Makie.Legend(fig[0,:], ax1; orientation = :horizontal, framevisible = false, nbanks = 2, patchsize = (30, 25))

# Makie.Legend(fig[:,2], ax1; orientation = :vertical, framevisible = false, patchsize = (30,20))

# y axes
Makie.hideydecorations!(ax1; label = false, grid = false)
Makie.hideydecorations!(ax2; label = false, grid = false)
Makie.hideydecorations!(ax3; label = false, grid = false)

Makie.ylims!(ax3, -5, 5)

# x axes
Makie.linkxaxes!(ax1, ax2, ax3)
Makie.hidexdecorations!(ax1)
Makie.hidexdecorations!(ax2)

display(fig)
regpath = joinpath(pkgdir(NeuralROMs), "figs", "method", "exp_reg.pdf")
save(regpath, fig)
#==============================================================#
