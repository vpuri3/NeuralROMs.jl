#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using JLD2, Plots                                 # vis/ save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using LaTeXStrings

import CairoMakie
import CairoMakie: Makie

CUDA.allowscalar(false)

begin
    nc = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
    BLAS.set_num_threads(nc)
end

include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "cases.jl"))

#======================================================#
function uData(x; σ = 1.0f0)
    pi32 = Float32(pi)

    # @. tanh(2f0 * x)

    # @. sin(5f0 * x^1) * exp(-(x/σ)^2)
    # @. sin(3f0 * x^2) * exp(-(x/σ)^2)

    # @. exp(sin(x))
    # @. sin(abs(x))

    @. (x - pi32/2f0) * sin(x) * exp(-(x/σ)^2)
end

function datagen_reg(datafile; _N = 1024, N_ = 16384)
    pi32 = Float32(pi)
    L = 2pi32

    _x = LinRange(-L, L, _N) |> Array
    x_ = LinRange(-L, L, N_) |> Array

    _u = uData(_x)
    u_ = uData(x_)
    metadata = (;)

    _data = (_x, _u)
    data_ = (x_, u_)

    jldsave(datafile; _data, data_, metadata)

    filename = joinpath(dirname(datafile), "plt_data")

    plt = plot(_x, _u, w = 3)
    png(plt, filename)

    plt
end

#======================================================#
function train_reg(
    datafile::String,
    dir::String,
    E, l, h, w;
    λ1::Real = 0f0,
    λ2::Real = 0f0,
    α::Real = 0f0,
    weight_decays::Union{Real,NTuple{M,<:Real}} = 0f0,
	_batchsize = nothing,
	warmup::Bool = true,
	early_stopping::Bool = true,
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
) where{M}

    #--------------------------------------------#
    # get data and normalize
    #--------------------------------------------#
    data = jldopen(datafile)
    _data = data["_data"]
    data_ = data["data_"]
    md_data = data["metadata"]
    close(data)

    _x, _u = reshape.(_data, 1, :)
    x_, u_ = reshape.(data_, 1, :)

    # normalize
    _x, x̄, σx = normalize_x(_x)
    _u, ū, σu = normalize_u(_u)

    x_, x̄, σx = normalize_x(x_)
    u_, ū, σu = normalize_u(u_)

    # metadata
    metadata = (; md_data, x̄, ū, σx, σu)

    _data = (_x, _u)
    data_ = (x_, u_)

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#

    NN = begin
        init_wt_in = scaled_siren_init(1f1)
        init_wt_hd = scaled_siren_init(1f0)
        init_wt_fn = glorot_uniform

        init_bias = rand32 # zeros32
        use_bias_fn = false

        act = sin

        wi = 1
        wd = w
        wo = 1

        in_layer = Dense(wi, wd, act; init_weight = init_wt_in, init_bias)
        hd_layer = Dense(wd, wd, act; init_weight = init_wt_hd, init_bias)
        fn_layer = Dense(wd, wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

        Chain(in_layer, fill(hd_layer, h)..., fn_layer)
    end

    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#

    lossfun = NeuralROMs.regularize_decoder(mse; α, λ1, λ2)
    _batchsize = isnothing(_batchsize) ? numobs(data) : _batchsize

    idx = ps_W_indices(NN; rng)
    weightdecay = IdxWeightDecay(0f0, idx)
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, warmup, weightdecay; early_stopping)

    #--------------------------------------------#

    train_args = (; l, h, w, E, _batchsize, λ1, λ2, weight_decays, α)
    metadata = (; metadata..., train_args)

    @show metadata

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, weight_decays,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    @show metadata

    model, ST
end

#======================================================#
# post process
#======================================================#
function makemodel(modelfile::String)
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    ST = model["STATS"]
    close(model)
    NeuralModel(NN, st, md), p, ST
end

#======================================================#
# main
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 474)

datafile = joinpath(@__DIR__, "data_reg.jld2")
device = Lux.gpu_device()

E = 1400
_N, N_ = 1024, 8192 # 512, 32768
_batchsize = 32
early_stopping = false

## weight norm experiment
l, h, w = 1, 5, 64

datagen_reg(datafile; _N, N_)

modeldir1 = joinpath(@__DIR__, "model1") # vanilla
modeldir2 = joinpath(@__DIR__, "model2") # L2
modeldir3 = joinpath(@__DIR__, "model3") # lipschitz
modeldir4 = joinpath(@__DIR__, "model4") # weight

#############
# TRAIN
#############

# α, weight_decays, λ2 = 0f-5, 0f-2, 0f-2 # vanilla
# isdir(modeldir1) && rm(modeldir1, recursive = true)
# _, ST1 = train_reg(datafile, modeldir1, E, l, h, w; rng, λ2, α, weight_decays, _batchsize, early_stopping, device,)
#
# α, weight_decays, λ2 = 0f-5, 0f-2, 5f-2 # L2
# isdir(modeldir2) && rm(modeldir2, recursive = true)
# train_reg(datafile, modeldir2, E, l, h, w; rng, λ2, α, weight_decays, _batchsize, early_stopping, device,)
#
# α, weight_decays, λ2 = 5f-5, 0f-2, 0f-2 # Lipschitz
# isdir(modeldir3) && rm(modeldir3, recursive = true)
# train_reg(datafile, modeldir3, E, l, h, w; rng, λ2, α, weight_decays, _batchsize, early_stopping, device,)
#
# α, weight_decays, λ2 = 0f-5, 5f-2, 0f-0 # Weight
# isdir(modeldir4) && rm(modeldir4, recursive = true)
# train_reg(datafile, modeldir4, E, l, h, w; rng, λ2, α, weight_decays, _batchsize, early_stopping, device,)

#======================================================#
# Tabulate errors
#======================================================#

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

u , ud1 , ud2  = forwarddiff_deriv2(uData, x)
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

#======================================================#
# Paper figure
#======================================================#

xlabel = L"x"
xlabelsize = ylabelsize = 16

fig = Makie.Figure(; size = (600, 500), backgroundcolor = :white, grid = :off)

ax1 = Makie.Axis(fig[1,1]; xlabel, ylabel = L"u(x)"  , xlabelsize, ylabelsize)
ax2 = Makie.Axis(fig[2,1]; xlabel, ylabel = L"u'(x)" , xlabelsize, ylabelsize)
ax3 = Makie.Axis(fig[3,1]; xlabel, ylabel = L"u''(x)", xlabelsize, ylabelsize)

colors = [:black, :orange, :green, :blue, :red,]
styles = [:solid, :solid, :dash, :dashdot, :dashdotdot,]
labels = [L"Ground truth$$", L"No regularization$$", L"$L_2$ regularization $(γ=5\cdot10^{-2})$", L"Lipschitz regularization $(α=5⋅10^{-5})$", L"Weight regularization $(γ=5⋅10^{-2})$",]

kws = Tuple(
    (; color = colors[i], linestyle = styles[i], label = labels[i], linewidth = 2)
    for i in 1:5
)

Makie.lines!(ax1, x,  u; kws[1]...)
Makie.lines!(ax1, x, u1; kws[2]...)
Makie.lines!(ax1, x, u2; kws[3]...)
Makie.lines!(ax1, x, u3; kws[4]...)
Makie.lines!(ax1, x, u4; kws[5]...)

Makie.lines!(ax2, x,  ud1; kws[1]...)
Makie.lines!(ax2, x, u1d1; kws[2]...)
Makie.lines!(ax2, x, u2d1; kws[3]...)
Makie.lines!(ax2, x, u3d1; kws[4]...)
Makie.lines!(ax2, x, u4d1; kws[5]...)

Makie.lines!(ax3, x,  ud2; kws[1]...)
Makie.lines!(ax3, x, u1d2; kws[2]...)
Makie.lines!(ax3, x, u2d2; kws[3]...)
Makie.lines!(ax3, x, u3d2; kws[4]...)
Makie.lines!(ax3, x, u4d2; kws[5]...)

Makie.Legend(fig[0,:], ax1; orientation = :horizontal, framevisible = false, nbanks = 3, patchsize = (30, 25))

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
save(joinpath(pkgdir(NeuralROMs), "figs", "method", "exp_reg.pdf"), fig)
save(joinpath(@__DIR__, "exp_reg.pdf"), fig)

#======================================================#
# Presentation figure
#======================================================#
# xlabel = L"x"
# xlabelsize = ylabelsize = 16
#
# fig = Makie.Figure(; size = (1000, 600), backgroundcolor = :white, grid = :off)
#
# ax1 = Makie.Axis(fig[1,1]; xlabel, ylabel = L"u(x)"  , xlabelsize, ylabelsize)
# ax2 = Makie.Axis(fig[2,1]; xlabel, ylabel = L"u(x)"  , xlabelsize, ylabelsize)
# ax3 = Makie.Axis(fig[3,1]; xlabel, ylabel = L"u(x)"  , xlabelsize, ylabelsize)
#
# colors = [:black, :orange, :green, :blue, :red,]
# styles = [:solid, :solid, :dash, :dashdot, :dashdotdot,]
# labels = [L"Ground truth$$", L"No regularization$$", L"$L_2$ regularization $(γ=10^{-1})$", L"Lipschitz regularization $(α=5⋅10^{-5})$", L"Weight regularization $(γ=5⋅10^{-2})$",]
#
# kws = Tuple(
#     (; color = colors[i], linestyle = styles[i], label = labels[i], linewidth = 12)
#     for i in 1:5
# )
#
# # Fig 1
# Makie.lines!(ax1, x,  u; kws[1]..., linewidth = 20)
# Makie.lines!(ax1, x, u1; kws[2]...)
# Makie.lines!(ax1, x, u3; kws[4]...)
# Makie.lines!(ax1, x, u4; kws[5]...)
#
# # Fig 2
# Makie.lines!(ax2, x,  ud1; kws[1]..., linewidth = 20)
# Makie.lines!(ax2, x, u1d1; kws[2]...)
# Makie.lines!(ax2, x, u3d1; kws[4]...)
# Makie.lines!(ax2, x, u4d1; kws[5]...)
#
# # Fig 3
# Makie.lines!(ax3, x,  ud2; kws[1]..., linewidth = 20)
# Makie.lines!(ax3, x, u1d2; kws[2]...)
# Makie.lines!(ax3, x, u3d2; kws[4]...)
# Makie.lines!(ax3, x, u4d2; kws[5]...)
#
# Makie.hidedecorations!(ax1)
# Makie.hidedecorations!(ax2)
# Makie.hidedecorations!(ax3)
#
# Makie.xlims!(ax1, -1, 1)
# Makie.xlims!(ax2, -1, 1)
# Makie.xlims!(ax3, -1, 1)
#
# Makie.ylims!(ax3, -6, 7)
#
# display(fig)
# regpath = joinpath(pkgdir(NeuralROMs), "figs", "presentation", "method", "exp_reg_full.svg")
# save(regpath, fig)

#======================================================#
nothing
#
