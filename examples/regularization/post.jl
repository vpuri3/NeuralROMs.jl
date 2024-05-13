#
using LinearAlgebra, LaTeXStrings, JLD2
using CairoMakie

include(joinpath(@__DIR__, "reg.jl"))

datafile = joinpath(@__DIR__, "data_reg.jld2")

modelfile1 = joinpath(@__DIR__, "model1", "model_08.jld2") # vanilla
modelfile2 = joinpath(@__DIR__, "model2", "model_08.jld2") # L2
modelfile3 = joinpath(@__DIR__, "model3", "model_08.jld2") # lipschitz
modelfile4 = joinpath(@__DIR__, "model4", "model_08.jld2") # weight

data = jldopen(datafile)
x, _ = data["data_"]
close(data)

function makemodel(modelfile::String)
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    close(model)
    NeuralModel(NN, st, md), p
end

model1, p1 = makemodel(modelfile1)
model2, p2 = makemodel(modelfile2)
model3, p3 = makemodel(modelfile3)
model4, p4 = makemodel(modelfile4)

xbatch = reshape(x, 1, :)
autodiff = AutoForwardDiff()
ϵ = nothing

u , ud1 , ud2  = forwarddiff_deriv2(uData, x)
u1, u1d1, u1d2 = dudx2_1D(model1, xbatch, p1; autodiff, ϵ) .|> vec
u2, u2d1, u2d2 = dudx2_1D(model2, xbatch, p2; autodiff, ϵ) .|> vec
u3, u3d1, u3d2 = dudx2_1D(model3, xbatch, p3; autodiff, ϵ) .|> vec
u4, u4d1, u4d2 = dudx2_1D(model4, xbatch, p4; autodiff, ϵ) .|> vec

N = length(u)
n   = sum(abs2, u)   / N |> sqrt
nd1 = sum(abs2, ud1) / N |> sqrt
nd2 = sum(abs2, ud2) / N |> sqrt

e1 = abs.(u1 - u) ./ n
e2 = abs.(u2 - u) ./ n
e3 = abs.(u3 - u) ./ n
e4 = abs.(u4 - u) ./ n

e1d1 = abs.(u1d1 - ud1) ./ n
e2d1 = abs.(u2d1 - ud1) ./ n
e3d1 = abs.(u3d1 - ud1) ./ n
e4d1 = abs.(u4d1 - ud1) ./ n

e1d2 = abs.(u1d2 - ud2) ./ n
e2d2 = abs.(u2d2 - ud2) ./ n
e3d2 = abs.(u3d2 - ud2) ./ n
e4d2 = abs.(u4d2 - ud2) ./ n

#==============================================================#

fig = Figure(; size = ( 900, 900), backgroundcolor = :white, grid = :off)

ax1 = Makie.Axis(fig[1,:]; xlabel = L"x", ylabel = L"u(x, t)"  , xlabelsize = 16, ylabelsize = 16)
ax2 = Makie.Axis(fig[2,:]; xlabel = L"x", ylabel = L"u'(x, t)" , xlabelsize = 16, ylabelsize = 16)
ax3 = Makie.Axis(fig[3,:]; xlabel = L"x", ylabel = L"u''(x, t)", xlabelsize = 16, ylabelsize = 16)

colors = (:orange, :green, :blue, :red,)
styles = (:solid, :dash, :dashdot, :dashdotdot,)
labels = (L"No regularization$$", L"$L_2$ regularization", L"Lipschitz regularization ($α=10^{-5}$)", L"Weight regularization ($γ = 10^{-2}$)",)

lines!(ax1, x,  u, color = :black, linestyle = :solid, label = labels[1], linewidth = 2)
lines!(ax1, x, u1, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
lines!(ax1, x, u2, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
lines!(ax1, x, u3, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
lines!(ax1, x, u4, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)

lines!(ax2, x,  ud1, color = :black, linestyle = :solid, label = labels[1], linewidth = 2)
lines!(ax2, x, u1d1, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
lines!(ax2, x, u2d1, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
lines!(ax2, x, u3d1, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
lines!(ax2, x, u4d1, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)

lines!(ax3, x,  ud2, color = :black, linestyle = :solid, label = labels[1], linewidth = 2)
lines!(ax3, x, u1d2, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
lines!(ax3, x, u2d2, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
lines!(ax3, x, u3d2, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
lines!(ax3, x, u4d2, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)

Legend(fig[0,:], ax1; orientation = :horizontal, patchsize = (50, 10), framevisible = false)

#==============================================================#
# lines!(ax1, x, u  , color = :black, linestyle = :solid, label = labels[1], linewidth = 2)
# lines!(ax1, x, ud1, color = :black, linestyle = :solid, label = labels[1], linewidth = 2)
# lines!(ax1, x, ud2, color = :black, linestyle = :solid, label = labels[1], linewidth = 2)
#
# lines!(ax2, x, e1d1, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
# lines!(ax2, x, e2d1, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
# lines!(ax2, x, e3d1, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
# lines!(ax2, x, e4d1, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)
#
# lines!(ax3, x, e1d2, color = colors[1], linestyle = styles[1], label = labels[1], linewidth = 2)
# lines!(ax3, x, e2d2, color = colors[2], linestyle = styles[2], label = labels[2], linewidth = 2)
# lines!(ax3, x, e3d2, color = colors[3], linestyle = styles[3], label = labels[3], linewidth = 2)
# lines!(ax3, x, e4d2, color = colors[4], linestyle = styles[4], label = labels[4], linewidth = 2)

display(fig)

#==============================================================#
nothing
