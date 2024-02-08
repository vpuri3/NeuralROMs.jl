#===============================================================#
function animate1D(
    u::AbstractMatrix,
    x::AbstractVector,
    t::AbstractVector = [];
    title = "",
    kwargs...
)
    if isempty(t)
        t = 1:size(u, 2)
    end

    ylims = begin
        mi = minimum(u)
        ma = maximum(u)
        buf = false #(ma - mi) / 5
        (mi - buf, ma + buf)
    end
    kw = (; ylims, kwargs...)
    anim = @animate for i in 1:size(u, 2)
        titlestr = title * "time = $(round(t[i], digits=8))"
        plt = plot(title = titlestr)
        plot!(plt, x, u[:, i]; kw..., label = "Ground Truth", c = :black)
    end
end

function animate1D(
    u::AbstractMatrix,
    v::AbstractMatrix,
    x::AbstractVector,
    t::AbstractVector = [];
    title = "",
    kwargs...
)
    if isempty(t)
        t = 1:size(u, 2)
    end

    ylims = begin
        mi = minimum(u)
        ma = maximum(u)
        buf = (ma - mi) / 5
    (mi - buf, ma + buf)
    end
    kw = (; ylims, kwargs...)
    anim = @animate for i in 1:size(u, 2)
        titlestr = title * "time = $(round(t[i], digits=8))"
        plt = plot(title = titlestr)
        plot!(plt, x, u[:, i]; kw..., label = "Ground Truth", c = :black)
        plot!(plt, x, v[:, i]; kw..., label = "Prediction"  , c = :red  )
    end
end

#===============================================================#

function animate2D(
    u::AbstractArray{T,3},
    x::AbstractMatrix,
    y::AbstractMatrix,
    t::AbstractVector = [];
    title = "",
    kwargs...
) where{T <: Real}

    if isempty(t)
        t = 1:size(u, 2)
    end

    zlims = begin
        mi = minimum(u)
        ma = maximum(u)
        buf = false #(ma - mi) / 5
        (mi - buf, ma + buf)
    end
    kw = (; zlims, kwargs...)

    anim = @animate for i in 1:size(u, 3)
        titlestr = title * "time = $(round(t[i], digits=8))"
        meshplt(x, y, u[:, :, i], ; title = titlestr, kw...)
    end
end

function animate2D(
    u::AbstractArray{T,3},
    v::AbstractArray{T,3},
    x::AbstractMatrix,
    y::AbstractMatrix,
    t::AbstractVector = [];
    title = "",
    kwargs...
) where{T <: Real}

    if isempty(t)
        t = 1:size(u, 2)
    end

    zlims = begin
        mi = minimum(u)
        ma = maximum(u)
        buf = false #(ma - mi) / 5
        (mi - buf, ma + buf)
    end
    kw = (; zlims, kwargs...)

    anim = @animate for i in 1:size(u, 3)
        titlestr = title * "time = $(round(t[i], digits=8))"
        p1 = plot(; title = titlestr)
        p1 = meshplt(x, y, u[:, :, i]; plt = p1, c=:black, w = 1.0, kw...,)
        p1 = meshplt(x, y, v[:, :, i]; plt = p1, c=:red  , w = 0.2, kw...,)
    end
end

function meshplt(
    x::AbstractArray,
    y::AbstractArray,
    u::AbstractArray;
    plt::Union{Nothing, Plots.Plot} = nothing,
    a::Real = 45, b::Real = 30, c = :grays, legend = false,
    kwargs...,
)
    plt = isnothing(plt) ? plot() : plt

    plot!(plt, x , y , u ; c, camera = (a,b), legend, kwargs...)
    plot!(plt, x', y', u'; c, camera = (a,b), legend, kwargs...)
end

#===============================================================#

function plot_derivatives1D(
    model::NTuple{3, Any},
    x::AbstractVector,
    md::NamedTuple;
    second_derv::Bool = false,
    third_derv::Bool  = false,
    fourth_derv::Bool = false,
    autodiff = AutoForwardDiff(),
    ϵ = nothing,
)
    NN, p, st = model
    xbatch = reshape(x, 1, :)

    model = NeuralModel(NN, st, md)
    u, ud1x, ud2x, ud3x, ud4x = dudx4(model, xbatch, p; autodiff, ϵ) .|> vec

    plt = plot(xabel = "x", ylabel = "u(x,t)")

    plot!(plt, x, u, label = "u"  , w = 3.0)
    # plot!(plt, x, ud1x, label = "udx", w = 3.0)

    second_derv && plot!(plt, x, ud2x, label = "ud2x", w = 3.0)
    third_derv  && plot!(plt, x, ud3x, label = "ud3x", w = 3.0)
    fourth_derv && plot!(plt, x, ud4x, label = "ud4x", w = 3.0)

    return plt
end

function plot_derivatives1D_autodecoder(
    decoder::NTuple{3, Any},
    x::AbstractVector,
    p0::AbstractVector,
    md = nothing;
    second_derv::Bool = false,
    third_derv::Bool  = false,
    fourth_derv::Bool = false,
    autodiff = AutoForwardDiff(),
    ϵ = nothing,
)
    NN, p, st = freeze_autodecoder(decoder, p0)

    Nx = length(x)
    xbatch = reshape(x, 1, Nx)
    Icode = ones(Int32, 1, Nx)

    model = NeuralEmbeddingModel(NN, st, Icode, md.x̄, md.σx, md.ū, md.σu)
    u, ud1x, ud2x, ud3x, ud4x = dudx4_1D(model, xbatch, p; autodiff, ϵ) .|> vec

    plt = plot(xabel = "x", ylabel = "u(x,t)")

    plot!(plt, x, u, label = "u"  , w = 3.0)
    plot!(plt, x, ud1x, label = "udx", w = 3.0)

    second_derv && plot!(plt, x, ud2x, label = "ud2x", w = 3.0)
    third_derv  && plot!(plt, x, ud3x, label = "ud3x", w = 3.0)
    fourth_derv && plot!(plt, x, ud4x, label = "ud4x", w = 3.0)

    return plt
end

#===============================================================#
"""
$SIGNATURES

"""
function plot_1D_surrogate_steady(V::Spaces.AbstractSpace{<:Any, 1},
    _data::NTuple{2, Any},
    data_::NTuple{2, Any},
    NN::Lux.AbstractExplicitLayer,
    p,
    st;
    nsamples = 5,
    dir = nothing,
    format = :CNB,
)
    x, = points(V)

    _x, _ŷ = _data
    x_, ŷ_ = data_

    _y = NN(_x, p, st)[1]
    y_ = NN(x_, p, st)[1]

    N, _K = size(_y)[end-1:end]
    N, K_ = size(y_)[end-1:end]

    _I = rand(1:_K, nsamples)
    I_ = rand(1:K_, nsamples)
    n = 4
    ms = 4

    cmap = range(HSV(0,1,1), stop=HSV(-360,1,1), length = nsamples + 1)

    # Trajectory plots
    kw = (; legend = false, xlabel = "x", ylabel = "u(x)")

    _p0 = plot(;title = "Training Comparison", kw...)
    p0_ = plot(;title = "Testing Comparison" , kw...)

    for i in 1:nsamples
        c = cmap[i]
        _i = _I[i]
        i_ = I_[i]

        kw_data = (; markersize = ms, c = c,)
        kw_pred = (; s = :solid, w = 2.0, c = c)

        _idx, idx_ = if _y isa AbstractMatrix
            (Colon(), _i), (Colon(), i_)
        elseif _y isa AbstractArray{<:Any, 3}
            # plot only the first output channel
            # make separate dispatches for visualize later
            (1, Colon(), _i), (1, Colon(), i_)
        end

        # training
        __y = _y[_idx...]
        __ŷ = _ŷ[_idx...]
        scatter!(_p0, x[begin:n:end], __ŷ[begin:n:end]; kw_data...)
        plot!(_p0, x, __y; kw_pred...)

        # testing
        y__ = y_[idx_...]
        ŷ__ = ŷ_[idx_...]
        scatter!(p0_, x[begin:n:end], ŷ__[begin:n:end]; kw_data...)
        plot!(p0_, x, y__; kw_pred...)
    end

    # R2 plots

    _R2 = round(rsquare(_y, _ŷ), digits = 8)
    R2_ = round(rsquare(y_, ŷ_), digits = 8)

    kw = (; legend = false, xlabel = "Data", ylabel = "Prediction", aspect_ratio = :equal)

    _p1 = plot(; title = "Training R² = $_R2", kw...)
    p1_ = plot(; title = "Testing R² = $R2_", kw...)

    scatter!(_p1, vec(_y), vec(_ŷ), ms = 1)
    scatter!(p1_, vec(y_), vec(ŷ_), ms = 1)

    _l = [extrema(_y)...]
    l_ = [extrema(y_)...]
    plot!(_p1, _l, _l, w = 4.0, c = :red)
    plot!(p1_, l_, l_, w = 4.0, c = :red)

    plts = _p0, p0_, _p1, p1_

    if !isnothing(dir)
        png(plts[1],   joinpath(dir, "plt_traj_train"))
        png(plts[2],   joinpath(dir, "plt_traj_test"))
        png(plts[3],   joinpath(dir, "plt_r2_train"))
        png(plts[4],   joinpath(dir, "plt_r2_test"))
    end

    plts
end
#===============================================================#
#
