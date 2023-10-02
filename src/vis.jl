function animate1D(u::AbstractMatrix,
    x::AbstractVector,
    t::AbstractVector;
    title = "",
    kwargs...
)

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
    end
end

function animate1D(u::AbstractMatrix,
    v::AbstractMatrix,
    x::AbstractVector,
    t::AbstractVector;
    title = "",
    kwargs...
)

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
#
