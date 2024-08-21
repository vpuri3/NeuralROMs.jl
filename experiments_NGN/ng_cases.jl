#
using NeuralROMs
using Random, Lux, NNlib, MLUtils, JLD2
using Plots, ColorSchemes, LaTeXStrings

#======================================================#

get_prob_grid(::AdvectionDiffusion1D) = (256,)
get_prob_domain(::AdvectionDiffusion1D) = (-1f0, 1f0)

get_prob_grid(::BurgersViscous1D) = (8192,)
get_prob_domain(::BurgersViscous1D) = (0f0, 2f0)

#===================================================#
function fieldplot(
    Xdata::AbstractArray,
    Tdata::AbstractArray,
    Udata::AbstractArray,
    Upred::AbstractArray,
    ps::AbstractArray,
    grid::Tuple,
    outdir::String,
    prefix::String,
    case::Integer;
    fps::Integer = 30,
)
    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    linewidth = 2.0

    for od in 1:out_dim
        up = Upred[od, :, :]
        ud = Udata[od, :, :]
        nr = sum(abs2, ud) / length(ud) |> sqrt
        er = (up - ud) / nr
        er = sum(abs2, er; dims = 1) / size(ud, 1) .|> sqrt |> vec

        Nx, Nt = size(Xdata, 2), length(Tdata)

        if in_dim == 1
            xd = vec(Xdata)

            Ixplt = LinRange(1, Nx, 32) .|> Base.Fix1(round, Int)
            Itplt = LinRange(1, Nt,  4) .|> Base.Fix1(round, Int)

            # u(x, t)
            plt = plot(;
                # title = "Ambient space evolution, case = $(case)",
                xlabel = L"x", ylabel = L"u(x,t)", legend = false,
            )

            colors = [:black, :blue, :magenta, :red]

            for (i,it) in enumerate(Itplt)
                c = colors[i]
                plot!(plt, xd, up[:, it]; linewidth, c)
                scatter!(plt, xd[Ixplt], ud[Ixplt, it]; w = 1, c)
            end

            png(plt, joinpath(outdir, "$(prefix)_u$(od)_case$(case)"))

            # # make gif
            # anim = animate1D(ud, up, xd, Tdata; w = 2, xlabel = L"x", ylabel = L"u(x,t)", title = "Case $case ")
            # gif(anim, joinpath(outdir, "evolve$(case).gif"); fps)

        elseif in_dim == 2
            xlabel = L"x"
            ylabel = L"y"
            zlabel = L"u$(od)(x, t)"

            kw = (; xlabel, ylabel, zlabel,)

            x_re = reshape(Xdata[1, :], grid)
            y_re = reshape(Xdata[2, :], grid)

            xline = x_re[:, 1]
            yline = x_re[1, :]

            upred_re = reshape(up, grid..., :)
            udata_re = reshape(ud, grid..., :)

            Itplt = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int)

            for (i, idx) in enumerate(Itplt)
                up_re = upred_re[:, :, idx]
                ud_re = udata_re[:, :, idx]

                # p1 = plot()
                # p1 = meshplt(x_re, y_re, up_re; plt = p1, c=:black, w = 1.0, kw...,)
                # p1 = meshplt(x_re, y_re, up_re - ud_re; plt = p1, c=:red  , w = 0.2, kw...,)
                #
                # p2 = meshplt(x_re, y_re, ud_re - up_re; title = "error", kw...,)
                #
                # png(p1, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)"))
                # png(p2, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)_error"))

                p3 = heatmap(up_re) #; title = "u$(od)(x, y)")
                p4 = heatmap(abs.(up_re - ud_re)) #; title = "u$(od)(x, y)")

                png(p3, joinpath(outdir, "$(prefix)_u$(od)_$(case)_time_$(i)"))
                png(p4, joinpath(outdir, "$(prefix)_u$(od)_$(case)_time_$(i)_error"))

            end
        else
            throw(ErrorException("in_dim = $in_dim not supported."))
        end

        # e(t)
        plt = plot(;
            title = "Error evolution, case = $(case)",
            xlabel = L"Time ($s$)",
            ylabel = L"ε(t)", legend = false,
            yaxis = :log,
            ylims = (10^-8, 1.0),
        )

        plot!(plt, Tdata, er; linewidth, ylabel = "ε(t)")
        png(plt, joinpath(outdir, "$(prefix)_e$(od)_case$(case)"))
    end

    if size(ps, 1) < 5
        psdir = joinpath(outdir, "plt_param$case")
        mkpath(psdir)

        for i in axes(ps, 1)
            p = @view ps[i, :]
            plt = plot(; xlabel = "Time (t)", ylabel = "Param", title = "Param $(i)")
            plot!(plt, Tdata, p; w = 2, c = :black,)
            png(plt, joinpath(psdir, "plt_p$(i)"))
        end
    end

    nothing
end

#======================================================#

