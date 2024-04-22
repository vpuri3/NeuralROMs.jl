#
using LinearAlgebra, HDF5, LaTeXStrings
# using Plots
using CairoMakie

function makeplots(
    datafile,
    outdir::String,
    casename::AbstractString;
    ifcrom::Bool = false,
    ifdt::Bool = false,
)

    data = h5open(datafile)
    xFOM = data["xFOM"] |> Array # [in_dim, grid...]
    tFOM = data["tFOM"] |> Array # [Nt]
    uFOM = data["uFOM"] |> Array # [out_dim, grid..., Nt]
    #
    uPCA = data["uPCA"] |> Array
    uCAE = data["uCAE"] |> Array
    uSNL = data["uSNL"] |> Array
    uSNW = data["uSNW"] |> Array
    #
    pCAE = data["pCAE"] |> Array # dynamics solve
    pSNL = data["pSNL"] |> Array
    pSNW = data["pSNW"] |> Array
    #
    qCAE = data["qCAE"] |> Array # encder prediction
    qSNL = data["qSNL"] |> Array
    qSNW = data["qSNW"] |> Array

    # C-ROM
    uCRM = uFOM * NaN
    pCRM = pSNW * NaN
    qCRM = qSNW * NaN

    if ifcrom
        uCRM = data["uCRM"] |> Array
        pCRM = data["pCRM"] |> Array
        qCRM = data["qCRM"] |> Array
    end

    # DT
    pdtCAE, pdtSNL, pdtSNW = if ifdt
        pdtCAE = data["pdtCAE"] |> Array
        pdtSNL = data["pdtSNL"] |> Array
        pdtSNW = data["pdtSNW"] |> Array

        pdtCAE, pdtSNL, pdtSNW
    else
        nothing, nothing, nothing
    end

    #======================================================#

    in_dim  = size(xFOM, 1)
    out_dim = size(uFOM, 1)
    Nt = length(tFOM)

    grid = size(uFOM)[2:in_dim+1]
    Nxyz = prod(grid)

    @assert in_dim == ndims(xFOM) - 1
    @assert size(xFOM)[2:end] == size(uFOM)[2:end-1]
    @assert size(uFOM)[end] == length(tFOM)

    Itplt = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int)
    i1, i2 = Itplt[2], Itplt[5]

    ## grab the first output dimension
    ii = Tuple(Colon() for _ in 1:in_dim + 1)
    uFOM = uFOM[1, ii...]
    uPCA = uPCA[1, ii...]
    uCAE = uCAE[1, ii...]
    uSNL = uSNL[1, ii...]
    uSNW = uSNW[1, ii...]
    uCRM = uCRM[1, ii...]

    ## normalize
    nr = sum(abs2, uFOM; dims = 1:in_dim) ./ prod(size(uFOM)[1:in_dim]) .|> sqrt

    ePCA = (uFOM - uPCA) ./ nr
    eCAE = (uFOM - uCAE) ./ nr
    eSNL = (uFOM - uSNL) ./ nr
    eSNW = (uFOM - uSNW) ./ nr
    eCRM = (uFOM - uCRM) ./ nr

    e2tPCA = sum(abs2, ePCA; dims = 1:in_dim) / Nxyz |> vec
    e2tCAE = sum(abs2, eCAE; dims = 1:in_dim) / Nxyz |> vec
    e2tSNL = sum(abs2, eSNL; dims = 1:in_dim) / Nxyz |> vec
    e2tSNW = sum(abs2, eSNW; dims = 1:in_dim) / Nxyz |> vec
    e2tCRM = sum(abs2, eCRM; dims = 1:in_dim) / Nxyz |> vec

    e2tPCA = sqrt.(e2tPCA) .+ 1f-12
    e2tCAE = sqrt.(e2tCAE) .+ 1f-12
    e2tSNL = sqrt.(e2tSNL) .+ 1f-12
    e2tSNW = sqrt.(e2tSNW) .+ 1f-12
    e2tCRM = sqrt.(e2tCRM) .+ 1f-12

    idx = collect(Colon() for _ in 1:in_dim)
    eitPCA = collect(norm(ePCA[idx..., i]) for i in 1:Nt)
    eitCAE = collect(norm(eCAE[idx..., i]) for i in 1:Nt)
    eitSNL = collect(norm(eSNL[idx..., i]) for i in 1:Nt)
    eitSNW = collect(norm(eSNW[idx..., i]) for i in 1:Nt)
    eitCRM = collect(norm(eCRM[idx..., i]) for i in 1:Nt)

    upreds   = (uPCA, uCAE, uSNL, uSNW,)
    epreds   = (ePCA, eCAE, eSNL, eSNW,)
    eitpreds = (eitPCA, eitCAE, eitSNL, eitSNW,)
    e2tpreds = (e2tPCA, e2tCAE, e2tSNL, e2tSNW,)

    if ifcrom
        upreds   = (upreds..., uCRM,)
        epreds   = (epreds..., uCRM,)
        eitpreds = (eitpreds..., eitCRM,)
        e2tpreds = (e2tpreds..., e2tCRM,)
    end

    figt = Figure(; size = ( 900, 400), backgroundcolor = :white, grid = :off)
    figc = Figure(; size = (1000, 800), backgroundcolor = :white, grid = :off)
    fige = Figure(; size = ( 600, 400), backgroundcolor = :white, grid = :off)
    figp = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)

    axt0 = Axis(figt[1,1]; xlabel = L"x", ylabel = L"u(x, t)", xlabelsize = 20, ylabelsize = 20)
    axt1 = Axis(figt[1,2]; xlabel = L"x", ylabel = L"u(x, t)", xlabelsize = 20, ylabelsize = 20)

    axe1 = Axis(fige[1,1]; xlabel = L"t", ylabel = L"ε(t)", yscale = log10, xlabelsize = 20, ylabelsize = 20)
    # axe2 = Axis(fige[1,2]; xlabel = L"t", ylabel = L"ε_\infty(t)", yscale = log10, xlabelsize = 20, ylabelsize = 20)
    axe2 = Axis(Figure()[1,1])

    #===============================#
    # FIGP
    #===============================#

    sckwq = (; color = :black, markersize = 25, marker = :star5 , label = L"$\tilde{u}(t=0)$ prediction")
    lnkwq = (; color = :red  , linewidth = 4, linestyle = :solid, label = L"\text{Prediction}")
    lnkwp = (; color = :blue , linewidth = 6, linestyle = :dot  , label = L"\text{Dynamics solve}")
    lnkwt = (; color = :green, linewidth = 6, linestyle = :dash , label = L"\text{Dynamics solve (Large }\Delta t)")

    axkwp = if size(pCAE, 1) == 2
        (; xlabel = L"\tilde{u}_1(t)", ylabel = L"\tilde{u}_2(t)", xlabelsize = 20, ylabelsize = 20)
    elseif size(pCAE, 1) == 1
        axkwp = (; xlabel = L"t", ylabel = L"\tilde{u}(t)", xlabelsize = 20, ylabelsize = 20)
    end

    axp1 = Axis(figp[1,1]; axkwp...)
    axp2 = Axis(figp[1,2]; axkwp...)
    axp3 = Axis(figp[1,3]; axkwp...)

    pplot!(axp1, tFOM, pCAE, qCAE, pdtCAE; ifdt, sckwq, lnkwq, lnkwp, lnkwt)
    pplot!(axp2, tFOM, pSNL, qSNL, pdtSNL; ifdt, sckwq, lnkwq, lnkwp, lnkwt)
    pplot!(axp3, tFOM, pSNW, qSNW, pdtSNW; ifdt, sckwq, lnkwq, lnkwp, lnkwt)

    Label(figp[2,1], L"(a)")
    Label(figp[2,2], L"(b)")
    Label(figp[2,3], L"(c)")

    colsize!(figp.layout, 1, Relative(0.33))
    colsize!(figp.layout, 2, Relative(0.33))
    colsize!(figp.layout, 3, Relative(0.33))

    # axislegend(axp1)
    # axislegend(axp2)
    # axislegend(axp3)

    figp[3,:] = Legend(figp, axp1; orientation = :horizontal, patchsize = (50, 10))

    #===============================#
    # FIGT, FIGE, FIGC
    #===============================#

    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = ("POD", "CAE", "SNFL", "SNFW", "CROM")

    levels = if occursin("exp2", casename)
        n = 11

        l1 = range(-0.2, 1.2, n)
        l2 = range(-0.2, 1.2, n)
        l3 = 10.0 .^ range(-4, 0, n)

        l1, l2, l3
    elseif occursin("exp4", casename)
        n = 11

        l1 = range(-0.2, 1.0, n)
        l2 = range(-0.2, 1.0, n)
        l3 = 10.0 .^ range(-5, 0, n)

        l1, l2, l3
    end

    l1 = (L"(a)", L"(b)", L"(c)", L"(d)")
    l2 = (L"(e)", L"(f)", L"(g)", L"(h)")
    l3 = (L"(i)", L"(j)", L"(k)", L"(l)")

    for (i, (up, ep, eit, e2t)) in enumerate(zip(upreds, epreds, eitpreds, e2tpreds))

        color = colors[i]
        label = labels[i]
        linestyle = styles[i]

        plt_kw = (; color, label, linewidth = 3, linestyle)
        cax_kw = (; aspect = 1, xlabel = L"x", ylabel = L"y")
        ctr_kw = (; extendlow = :cyan, extendhigh = :magenta,)

        if in_dim == 1
            x = vec(xFOM)

            if i == 1
                lines!(axt0, x, uFOM[:, i1]; linewidth = 5, label = "FOM", color = :black)
                lines!(axt1, x, uFOM[:, i2]; linewidth = 5, label = "FOM", color = :black)
            end

            lines!(axt0, x, up[:, i1]; plt_kw...)
            lines!(axt1, x, up[:, i2]; plt_kw...)

        elseif in_dim == 2
            x, y = xFOM[1,:, :], xFOM[2, :, :]
            xdiag = diag(x)

            if i == 1
                ## color plots
                axc1 = Axis(figc[1,1]; cax_kw...)
                axc2 = Axis(figc[3,1]; cax_kw...)

                cf1 = contourf!(axc1, xdiag, xdiag, uFOM[:, :, i1]; ctr_kw..., levels = levels[1])
                cf2 = contourf!(axc2, xdiag, xdiag, uFOM[:, :, i2]; ctr_kw..., levels = levels[2])

                Colorbar(figc[1,5], cf1)
                Colorbar(figc[3,5], cf2)

                Label(figc[2,1], l1[1])
                Label(figc[4,1], l2[1])

                ## diagonal line plots
                uddiag1 = diag(uFOM[:, :, i1])
                uddiag2 = diag(uFOM[:, :, i2])

                axt0.xlabel = L"x = y"
                axt1.xlabel = L"x = y"

                axt0.ylabel = L"u(x = y, t)"
                axt1.ylabel = L"u(x = y, t)"

                lines!(axt0, xdiag, uddiag1; linewidth = 5, label = "FOM", color = :black)
                lines!(axt1, xdiag, uddiag2; linewidth = 5, label = "FOM", color = :black)
            end

            ## color plot (t0, t1)
            if i != 3
                j = i != 4 ? i+1 : i

                axc1 = Axis(figc[1,j]; cax_kw...)
                axc2 = Axis(figc[3,j]; cax_kw...)

                cf1 = contourf!(axc1, xdiag, xdiag, up[:, :, i1]; ctr_kw..., levels = levels[1])
                cf2 = contourf!(axc2, xdiag, xdiag, up[:, :, i2]; ctr_kw..., levels = levels[2])

                Label(figc[2,j], l1[j])
                Label(figc[4,j], l2[j])
            end

            ## color plot (error)
            axc3 = Axis(figc[5,i]; cax_kw...,)
            cf3 = contourf!(axc3, xdiag, xdiag, abs.(ep[:, :, i2]); ctr_kw...,
                colorscale = log10, levels = levels[3])

            Label(figc[6,i], l3[i])

            if i == 1
                Colorbar(figc[5,5], cf3)
            end

            # diagonal plots
            up1 = diag(up[:, :, i1])
            up2 = diag(up[:, :, i2])

            lines!(axt0, xdiag, up1; plt_kw...)
            lines!(axt1, xdiag, up2; plt_kw...)
        end

        ## error plot
        lines!(axe1, tFOM, e2t; linewidth = 3, label = labels[i], plt_kw...)
        lines!(axe2, tFOM, eit; linewidth = 3, label = labels[i], plt_kw...)
    end

    axislegend(axe1; position = :lt, patchsize = (30, 10), orientation = :horizontal)
    figt[2,:] = Legend(figt, axt1; patchsize = (30, 10), orientation = :horizontal)

    if occursin("exp3", casename)
        ylims!(fige.content[1], 10^-5, 10^-1)
    end

    if in_dim == 2
        for ax in figc.content
            if ax isa Axis
                tightlimits!(ax)
                hidedecorations!(ax; label = false)
            end
        end

        colsize!(figc.layout, 1, Relative(0.25))
        colsize!(figc.layout, 2, Relative(0.25))
        colsize!(figc.layout, 3, Relative(0.25))
        colsize!(figc.layout, 4, Relative(0.25))
    end

    linkaxes!(axt0, axt1)

    # save(joinpath(outdir, casename * "p1.eps"), figt)
    # save(joinpath(outdir, casename * "p2.eps"), fige)
    #
    # if in_dim == 2
    #     save(joinpath(outdir, casename * "p3.eps"), figc)
    # end

    save(joinpath(outdir, casename * "p4.png"), figp)

    nothing
end
#======================================================#

function makeplot_exp3(
    datafiles::String...;
    outdir::String,
    ifdt::Bool = false,
)
    figp = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)

    axkwp = (;
        xlabel = L"\tilde{u}_1(t)",
        ylabel = L"\tilde{u}_2(t)",
        xlabelsize = 20,
        ylabelsize = 20,
    )

    axp1 = Axis(figp[1,1]; axkwp...)
    axp2 = Axis(figp[1,2]; axkwp...)
    axp3 = Axis(figp[1,3]; axkwp...)

    # is there a 2D colormap ??
    labels = (
        L"$\mu = 0.500$ (Training)",
        L"$\mu = 0.525$ (Interpolation)",
        L"$\mu = 0.550$ (Training)",
        L"$\mu = 0.575$ (Interpolation)",
        L"$\mu = 0.600$ (Training)",
        L"$\mu = 0.625$ (Extrapolation)",
    )

    colors = (:blue, :orange, :green, :red, :purple, :brown,)

    for (i, datafile) in enumerate(datafiles)
        data = h5open(datafile)
        tFOM = data["tFOM"] |> Array # [Nt]
        #
        pCAE = data["pCAE"] |> Array
        pSNL = data["pSNL"] |> Array
        pSNW = data["pSNW"] |> Array
        #
        qCAE = data["qCAE"] |> Array # encder prediction
        qSNL = data["qSNL"] |> Array
        qSNW = data["qSNW"] |> Array

        color = colors[i]
        label = labels[i]
        sckwq = (; color = :black, markersize = 15, marker = :star5,)
        lnkwq = (; color, label, linewidth = 4, linestyle = :solid,)
        lnkwp = (; color, linewidth = 6, linestyle = :dot,)

        pplot!(axp1, tFOM, pCAE, qCAE; sckwq, lnkwq, lnkwp)
        pplot!(axp2, tFOM, pSNL, qSNL; sckwq, lnkwq, lnkwp)
        pplot!(axp3, tFOM, pSNW, qSNW; sckwq, lnkwq, lnkwp)
    end

    Label(figp[2,1], L"(a)")
    Label(figp[2,2], L"(b)")
    Label(figp[2,3], L"(c)")

    colsize!(figp.layout, 1, Relative(0.33))
    colsize!(figp.layout, 2, Relative(0.33))
    colsize!(figp.layout, 3, Relative(0.33))

    figp[3,:] = Legend(figp, axp1; orientation = :horizontal,) # patchsize = (100, 10))

    save(joinpath(outdir, "exp3p.eps"), figp)
end

#======================================================#
function pplot!(ax, t, p, q, pdt = nothing;
    ifdt = false, 
    sckwq = (;),
    lnkwq = (;),
    lnkwp = (;),
    lnkwt = (;),
)
    if size(p, 1) == 2
        scatter!(ax, q[:, 1:1]; sckwq...)
        lines!(ax, q; lnkwq...)
        lines!(ax, p; lnkwp...)
        if ifdt
            lines!(ax, pdt; lnkwt...)
        end
    elseif size(p, 1) == 1
        scatter!(ax, [0f0,], q[:, 1]; sckwq...)
        lines!(ax, t, vec(q); lnkwq...)
        lines!(ax, t, vec(p); lnkwp...)

        if ifdt
            lines!(ax, t, vec(pdt); lnkwt...)
        end
    else
        @error "latent size size(p, 1) == $(size(p, 1)) not supported."
    end
end

#======================================================#

"""
    sync_colorranges!(plotobjects::MakieCore.ScenePlot...)

Set the colorrange of all `plotobjects` to the same value,
namely the extrema of all z-values of all plotobjects in `plotobjects`.

https://discourse.julialang.org/t/one-colorbar-for-multiple-axes/77021/8
"""
# function sync_colorranges!(plotobjects::Makie.MakieCore.ScenePlot...)
function sync_colorranges!(plotobjects::Makie.Heatmap...)
    for plt in plotobjects 
        haskey(plt.attributes.attributes, :colorrange) || error("This function syncronizes the color range of the given plotobjects. One of the plotobjects passed has no color range.")
    end
    possible_extremas = [extrema(to_value(plt[3])) for plt in plotobjects]
    global_extremas = extrema(vcat(collect.(possible_extremas)...))
    for plt in plotobjects
        plt.attributes.colorrange[] = global_extremas
    end
    return nothing
end

#======================================================#
h5dir  = joinpath(@__DIR__, "h5files")
outdir = joinpath(@__DIR__, "results")

e1file = joinpath(h5dir, "advect1d.h5")
e2file = joinpath(h5dir, "advect2d.h5")
e4file = joinpath(h5dir, "burgers2d.h5")
e5file = joinpath(h5dir, "ks1d.h5")

e3file1 = joinpath(h5dir, "burgers1dcase1.h5")
e3file2 = joinpath(h5dir, "burgers1dcase2.h5")
e3file3 = joinpath(h5dir, "burgers1dcase3.h5")
e3file4 = joinpath(h5dir, "burgers1dcase4.h5")
e3file5 = joinpath(h5dir, "burgers1dcase5.h5")
e3file6 = joinpath(h5dir, "burgers1dcase6.h5")

makeplots(e1file, outdir, "exp1"; ifdt = true)
# makeplots(e2file, outdir, "exp2")
# makeplots(e4file, outdir, "exp4")
# makeplots(e5file, outdir, "exp5")
#
# # makeplots(e3file1, outdir, "exp3case1")
# # makeplots(e3file3, outdir, "exp3case3")
# # makeplots(e3file2, outdir, "exp3case2")
# makeplots(e3file4, outdir, "exp3case4")
# makeplots(e3file5, outdir, "exp3case5")
# makeplots(e3file6, outdir, "exp3case6")

makeplot_exp3(e3file1, e3file2, e3file3, e3file4, e3file5, e3file6; outdir)

#======================================================#
nothing
