#
using LinearAlgebra, LaTeXStrings, JLD2
using CairoMakie

let
    pkgpath = dirname(dirname(pathof(NeuralROMs)))
    figpath = joinpath(pkgpath, "figs")
    !(figpath in LOAD_PATH) && push!(LOAD_PATH, figpath)
end

using CairoMakie
>>>>>>> 9854823 (updated regularization)

function f(x; σ = 1.0f0)
    pi32 = Float32(pi)

    # @. (x - pi32/2f0) * sin(x) * exp(-(x/σ)^2)
    @. sin(abs(x/2))
end

function makemodel(modelfile::String)
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    close(model)
    NeuralModel(NN, st, md), p
end

datafile = joinpath(@__DIR__, "data_reg.jld2")
modelfile1 = joinpath(@__DIR__, "model1", "model_08.jld2") # vanilla
modelfile2 = joinpath(@__DIR__, "model2", "model_08.jld2") # L2
modelfile3 = joinpath(@__DIR__, "model3", "model_08.jld2") # lipschitz
modelfile4 = joinpath(@__DIR__, "model4", "model_08.jld2") # weight

data = jldopen(datafile)
x, _ = data["data_"]
close(data)

x = x[1:128:end]

model1, p1 = makemodel(modelfile1)
model2, p2 = makemodel(modelfile2)
model3, p3 = makemodel(modelfile3)
model4, p4 = makemodel(modelfile4)

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

e1_s = e1' * e1 / N |> sqrt
e2_s = e2' * e2 / N |> sqrt
e3_s = e3' * e3 / N |> sqrt
e4_s = e4' * e4 / N |> sqrt

e1d1_s = e1d1' * e1d1 / N |> sqrt
e2d1_s = e2d1' * e2d1 / N |> sqrt
e3d1_s = e3d1' * e3d1 / N |> sqrt
e4d1_s = e4d1' * e4d1 / N |> sqrt

e1d2_s = e1d2' * e1d2 / N |> sqrt
e2d2_s = e2d2' * e2d2 / N |> sqrt
e3d2_s = e3d2' * e3d2 / N |> sqrt
e4d2_s = e4d2' * e4d2 / N |> sqrt

println("Zero: $e1_s")
println("L2  : $e2_s")
println("SNFL: $e3_s")
println("SNFW: $e4_s")

println()
println("1st derivative")
println()

println("Zero: $e1d1_s")
println("L2  : $e2d1_s")
println("SNFL: $e3d1_s")
println("SNFW: $e4d1_s")

#==============================================================#

fig = Figure(; size = (900, 400), backgroundcolor = :white, grid = :off)

ax1 = Makie.Axis(fig[1,1]; xlabel = L"x", ylabel = L"u(x)" , xlabelsize = 16, ylabelsize = 16)
ax2 = Makie.Axis(fig[1,2]; xlabel = L"x", ylabel = L"u'(x)", xlabelsize = 16, ylabelsize = 16)
ax3 = Makie.Axis(fig[1,3]; xlabel = L"x", ylabel = L"e'(x)", xlabelsize = 16, ylabelsize = 16, yscale = log10)

colors = (:orange, :green, :blue, :red,)
styles = (:solid, :dot, :dashdot, :dashdotdot,)
labels = (L"No regularization$$", L"$L_2$ reg.", L"Lipschitz reg.$$", L"Weight reg.$$",)

lines!(ax1, x,  u, color = :black, linestyle = :solid, label = L"Data$$", linewidth = 2)
lines!(ax1, x, u1, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
lines!(ax1, x, u2, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
lines!(ax1, x, u3, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
lines!(ax1, x, u4, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)

lines!(ax2, x,  ud1, color = :black, linestyle = :solid, label = labels[1], linewidth = 2)
lines!(ax2, x, u1d1, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
lines!(ax2, x, u2d1, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
lines!(ax2, x, u3d1, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
lines!(ax2, x, u4d1, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)

lines!(ax3, x, e1d1, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
lines!(ax3, x, e2d1, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
lines!(ax3, x, e3d1, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
lines!(ax3, x, e4d1, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)

Legend(fig[0,:], ax1; orientation = :horizontal, framevisible = false)

display(fig)
#==============================================================#
nothing
