#
using LinearAlgebra, HDF5, JLD2, LaTeXStrings
using CairoMakie

function makeplots(
    datafile,
    outdir::String,
    casename::String;
    ifcrom::Bool = false,
    ifdt::Bool = false,
)

    mkpath(outdir)

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

    tdtFOM, udtFOM, udtCAE, udtSNL, udtSNW = if ifdt
        tdtFOM = data["tdtFOM"] |> Array
        udtFOM = data["udtFOM"] |> Array
        udtCAE = data["udtCAE"] |> Array
        udtCAE = data["udtCAE"] |> Array
        udtSNL = data["udtSNL"] |> Array
        udtSNW = data["udtSNW"] |> Array

        tdtFOM, udtFOM, udtCAE, udtSNL, udtSNW
    else
        nothing, nothing, nothing, nothing, nothing
    end

    close(data)

    #======================================================#

    in_dim  = size(xFOM, 1)
    out_dim = size(uFOM, 1)
    Nt = length(tFOM)

    grid = size(uFOM)[2:in_dim+1]
    Nxyz = prod(grid)
    Nfom = Nxyz * out_dim

    @assert in_dim == ndims(xFOM) - 1
    @assert size(xFOM)[2:end] == size(uFOM)[2:end-1]
    @assert size(uFOM)[end] == length(tFOM)

    Itplt = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int)
    i1, i2 = Itplt[2], Itplt[5]

    ## normalize
    nr = sum(abs2, uFOM; dims = 1:in_dim+1) ./ Nfom .|> sqrt

    ePCA = (uFOM - uPCA) ./ nr
    eCAE = (uFOM - uCAE) ./ nr
    eSNL = (uFOM - uSNL) ./ nr
    eSNW = (uFOM - uSNW) ./ nr
    eCRM = (uFOM - uCRM) ./ nr

    e2tPCA = sum(abs2, ePCA; dims = 1:in_dim+1) / Nfom |> vec
    e2tCAE = sum(abs2, eCAE; dims = 1:in_dim+1) / Nfom |> vec
    e2tSNL = sum(abs2, eSNL; dims = 1:in_dim+1) / Nfom |> vec
    e2tSNW = sum(abs2, eSNW; dims = 1:in_dim+1) / Nfom |> vec
    e2tCRM = sum(abs2, eCRM; dims = 1:in_dim+1) / Nfom |> vec

    e2tPCA = sqrt.(e2tPCA) .+ 1f-12
    e2tCAE = sqrt.(e2tCAE) .+ 1f-12
    e2tSNL = sqrt.(e2tSNL) .+ 1f-12
    e2tSNW = sqrt.(e2tSNW) .+ 1f-12
    e2tCRM = sqrt.(e2tCRM) .+ 1f-12

    idx = collect(Colon() for _ in 1:in_dim+1)
    eitPCA = collect(norm(ePCA[idx..., i]) for i in 1:Nt)
    eitCAE = collect(norm(eCAE[idx..., i]) for i in 1:Nt)
    eitSNL = collect(norm(eSNL[idx..., i]) for i in 1:Nt)
    eitSNW = collect(norm(eSNW[idx..., i]) for i in 1:Nt)
    eitCRM = collect(norm(eCRM[idx..., i]) for i in 1:Nt)

    if ifdt
        nrdt = sum(abs2, udtFOM; dims = 2:in_dim+1) ./ Nxyz .|> sqrt

        edtCAE = (udtFOM - udtCAE) ./ nrdt
        edtSNL = (udtFOM - udtSNL) ./ nrdt
        edtSNW = (udtFOM - udtSNW) ./ nrdt

        e2tdtCAE = sum(abs2, edtCAE; dims = 1:in_dim+1) / Nfom |> vec
        e2tdtSNL = sum(abs2, edtSNL; dims = 1:in_dim+1) / Nfom |> vec
        e2tdtSNW = sum(abs2, edtSNW; dims = 1:in_dim+1) / Nfom |> vec

        e2tdtCAE = sqrt.(e2tdtCAE) .+ 1f-12
        e2tdtSNL = sqrt.(e2tdtSNL) .+ 1f-12
        e2tdtSNW = sqrt.(e2tdtSNW) .+ 1f-12
    end

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

    # grab the first output dimension for plotting
    ii = Tuple(Colon() for _ in 1:in_dim + 1)
    upreds = getindex.(upreds, 1, ii...)
    epreds = getindex.(epreds, 1, ii...)
    uFOM = getindex(uFOM, 1, ii...)

    #======================================================#

    figt = Figure(; size = ( 900, 400), backgroundcolor = :white, grid = :off)
    figc = Figure(; size = (1000, 800), backgroundcolor = :white, grid = :off)
    fige = Figure(; size = ( 600, 400), backgroundcolor = :white, grid = :off)
    figp = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)
    figq = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)

    ylabel_t, ylabel_e = if occursin("exp3", casename)
        L"u(x, t; \mathbf{μ})", L"ε(t; \mathbf{μ})"
    else
        L"u(x, t)", L"ε(t)"
    end

    axt0 = Axis(figt[1,1]; xlabel = L"x", ylabel = ylabel_t, xlabelsize = 16, ylabelsize = 16)
    axt1 = Axis(figt[1,2]; xlabel = L"x", ylabel = ylabel_t, xlabelsize = 16, ylabelsize = 16)
    axe1 = Axis(fige[1,1]; xlabel = L"t", ylabel = ylabel_e, yscale = log10, xlabelsize = 16, ylabelsize = 16)

    if ifdt
        fige = Figure(; size = (900, 400), backgroundcolor = :white, grid = :off)
        axe1 = Axis(fige[1,1]; xlabel = L"t", ylabel = ylabel_e, yscale = log10, xlabelsize = 16, ylabelsize = 16)
        axe2 = Axis(fige[1,2]; xlabel = L"t", ylabel = ylabel_e, yscale = log10, xlabelsize = 16, ylabelsize = 16)
    end

    #===============================#
    # FIGP
    #===============================#

    if size(pCAE, 1) == 2
        axkwp = (; xlabel = L"\tilde{u}_1(t)", ylabel = L"\tilde{u}_2(t)", xlabelsize = 16, ylabelsize = 16)

        axp1 = Axis(figp[1,1]; axkwp...)
        axp2 = Axis(figp[1,2]; axkwp...)
        axp3 = Axis(figp[1,3]; axkwp...)

        sckwq = (; color = :red  , markersize = 20,)
        lnkwq = (; color = :red  , linewidth = 2.5,)
        lnkwp = (; color = :blue , linewidth = 4.0,)
        lnkwt = (; color = :green, linewidth = 4.0,)

        kwCAE = (; ifdt, sckwq, lnkwq, lnkwp, lnkwt)
        kwSNF = (; ifdt, sckwq, lnkwq, lnkwp, lnkwt)

        sq, lq, lp, lt = pplot!(axp1, tFOM, pCAE, qCAE, pdtCAE; kwCAE...)
        sq, lq, lp, lt = pplot!(axp2, tFOM, pSNL, qSNL, pdtSNL; kwSNF...)
        sq, lq, lp, lt = pplot!(axp3, tFOM, pSNW, qSNW, pdtSNW; kwSNF...)

        Label(figp[2,1], L"(a) CAE-ROM$$", fontsize = 16)
        Label(figp[2,2], L"(b) SNFL-ROM$$", fontsize = 16)
        Label(figp[2,3], L"(c) SNFW-ROM$$", fontsize = 16)

        colsize!(figp.layout, 1, Relative(0.33))
        colsize!(figp.layout, 2, Relative(0.33))
        colsize!(figp.layout, 3, Relative(0.33))

        eq = [
            LineElement(; linestyle = :solid, lnkwq...),
            MarkerElement(; marker = :star5, sckwq..., points = Point2f[(0.05,0.5)])
        ]

        if ifdt
            elems  = [eq, lp, lt]
            labels = [L"\text{Learned prediction}", L"\text{Dynamics evaluation }(Δt = Δt_0)", L"\text{Dynamics evaluation }(Δt = 10Δt_0)"]
        else
            elems  = [eq, lp]
            labels = [L"\text{Learned prediction}", L"\text{Dynamics evaluation}"]
        end

        Legend(figp[0,:], elems, labels; orientation = :horizontal, patchsize = (50, 10), framevisible = false)
    end

    #===============================#
    # FIGQ
    #===============================#

    axkwp = (; xlabel = L"t", ylabel = L"\tilde{u}(t)", xlabelsize = 16, ylabelsize = 16)

    axp1 = Axis(figq[1,1]; axkwp...)
    axp2 = Axis(figq[1,2]; axkwp...)
    axp3 = Axis(figq[1,3]; axkwp...)

    lnkwq = (; linewidth = 2.5, solid_color = [:red    , :blue  ])
    lnkwp = (; linewidth = 4.0, solid_color = [:magenta, :cyan  ])
    lnkwt = (; linewidth = 4.0, solid_color = [:pink   , :purple])

    kwCAE = (; ifdt, lnkwq, lnkwp, lnkwt)
    kwSNF = (; ifdt, lnkwq, lnkwp, lnkwt)

    lq, lp, lt = ptplot!(axp1, tFOM, pCAE, qCAE, pdtCAE; kwCAE...)
    lq, lp, lt = ptplot!(axp2, tFOM, pSNL, qSNL, pdtSNL; kwSNF...)
    lq, lp, lt = ptplot!(axp3, tFOM, pSNW, qSNW, pdtSNW; kwSNF...)

    Label(figq[2,1], L"(a) CAE-ROM$$", fontsize = 16)
    Label(figq[2,2], L"(b) SNFL-ROM$$", fontsize = 16)
    Label(figq[2,3], L"(c) SNFW-ROM$$", fontsize = 16)

    colsize!(figq.layout, 1, Relative(0.33))
    colsize!(figq.layout, 2, Relative(0.33))
    colsize!(figq.layout, 3, Relative(0.33))

    elems = [
        LineElement(; linestyle = :solid, linewidth = 2.5, color = :black,),
        LineElement(; linestyle = :dot  , linewidth = 4.0, color = :black,),
        LineElement(; linestyle = :dash , linewidth = 4.0, color = :black,),
    ]

    if ifdt
        labels = [L"\text{Dynamics evaluation }(Δt = Δt_0)", L"\text{Dynamics evaluation }(Δt = 10Δt_0)"]
    else
        elems  = elems[1:2]
        labels = [L"\text{Dynamics evaluation}",]
    end

    if occursin("exp3", casename)
        l1 = L"Learned prediction $e_{θ_e}(ū(t; \mathbf{\mu}))$"
        l2 = L"Learned prediction $\Xi_\varrho(t, \mathbf{\mu})$"
    else
        l1 = L"Learned prediction $e_{θ_e}(ū(t))$"
        l2 = L"Learned prediction $\Xi_\varrho(t)$"
    end

    axislegend(axp1, elems, [l1, labels...]; position = :lt, patchsize = (40, 10))
    axislegend(axp2, elems, [l2, labels...]; position = :lb, patchsize = (40, 10))
    axislegend(axp3, elems, [l2, labels...]; position = :lb, patchsize = (40, 10))

    if casename == "exp1" # hack to make legend fit
        ylims!(axp1, -30, 35)
    end

    #===============================#
    # FIGT, FIGE, FIGC
    #===============================#

    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = (L"POD-ROM$$", L"CAE-ROM$$", L"SNFL-ROM$$", L"SNFW-ROM$$",)

    levels = if occursin("exp2", casename)
        n = 11

        l1 = range(-0.2, 1.2, n)
        l2 = range(-0.2, 1.2, n)
        l3 = 10.0 .^ range(-4, 0, n)

        l1, l2, l3
    elseif occursin("exp4", casename)
        n = 11

        l1 = range(-0.2, 1.1, n)
        l2 = range(-0.2, 1.1, n)
        l3 = 10.0 .^ range(-5, 0, n)

        l1, l2, l3
    end

    l1 = (L"(a) FOM$$"    , L"(b) POD-ROM$$", L"(c) CAE-ROM$$" , L"(d) SNFW-ROM$$")
    l2 = (L"(e) FOM$$"    , L"(f) POD-ROM$$", L"(g) CAE-ROM$$" , L"(h) SNFW-ROM$$")
    l3 = (L"(i) POD-ROM$$", L"(j) CAE-ROM$$", L"(k) SNFL-ROM$$", L"(l) SNFW-ROM$$")

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
                lines!(axt0, x, uFOM[:, i1]; linewidth = 3, label = L"FOM$$", color = :black)
                lines!(axt1, x, uFOM[:, i2]; linewidth = 3, label = L"FOM$$", color = :black)
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

                Label(figc[2,1], l1[1], fontsize = 16)
                Label(figc[4,1], l2[1], fontsize = 16)

                ## diagonal line plots
                uddiag1 = diag(uFOM[:, :, i1])
                uddiag2 = diag(uFOM[:, :, i2])

                axt0.xlabel = L"x = y"
                axt1.xlabel = L"x = y"

                axt0.ylabel = L"u(x = y, t)"
                axt1.ylabel = L"u(x = y, t)"

                lines!(axt0, xdiag, uddiag1; linewidth = 3, label = L"FOM$$", color = :black)
                lines!(axt1, xdiag, uddiag2; linewidth = 3, label = L"FOM$$", color = :black)
            end

            ## color plot (t0, t1)
            if i != 3
                j = i != 4 ? i+1 : i

                axc1 = Axis(figc[1,j]; cax_kw...)
                axc2 = Axis(figc[3,j]; cax_kw...)

                cf1 = contourf!(axc1, xdiag, xdiag, up[:, :, i1]; ctr_kw..., levels = levels[1])
                cf2 = contourf!(axc2, xdiag, xdiag, up[:, :, i2]; ctr_kw..., levels = levels[2])

                Label(figc[2,j], l1[j], fontsize = 16)
                Label(figc[4,j], l2[j], fontsize = 16)
            end

            ## color plot (error)
            axc3 = Axis(figc[5,i]; cax_kw...,)
            cf3 = contourf!(axc3, xdiag, xdiag, abs.(ep[:, :, i2]); ctr_kw...,
                colorscale = log10, levels = levels[3])

            Label(figc[6,i], l3[i], fontsize = 16)

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
    end

    # FIGT
    if ifdt
        lines!(axe2, tdtFOM, e2tdtCAE; linewidth = 3, label = labels[2], color = colors[2], linestyle = styles[2])
        lines!(axe2, tdtFOM, e2tdtSNL; linewidth = 3, label = labels[3], color = colors[3], linestyle = styles[3])
        lines!(axe2, tdtFOM, e2tdtSNW; linewidth = 3, label = labels[4], color = colors[4], linestyle = styles[4])

        linkaxes!(axe1, axe2)

        Label(fige[2,1], L"(a) Dynamics evaluation $(Δt = Δt_0)$", fontsize = 16)
        Label(fige[2,2], L"(b) Dynamics evaluation $(Δt = 10Δt_0)$", fontsize = 16)
        colsize!(fige.layout, 1, Relative(0.50))
        colsize!(fige.layout, 2, Relative(0.50))
    end

    # axislegend(axe1; position = :lt, patchsize = (30, 10), orientation = :horizontal)
    fige[0,:] = Legend(fige, axe1, patchsize = (30, 10), orientation = :horizontal, framevisible = false)
    figt[0,:] = Legend(figt, axt1; patchsize = (30, 10), orientation = :horizontal, framevisible = false)

    t1 = round(sigdigits=3, tFOM[i1])
    t2 = round(sigdigits=3, tFOM[i2])

    Label(figt[2,1], L"(a) $t = %$(t1)$", fontsize = 16)
    Label(figt[2,2], L"(b) $t = %$(t2)$", fontsize = 16)
    colsize!(figt.layout, 1, Relative(0.50))
    colsize!(figt.layout, 2, Relative(0.50))

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

    save(joinpath(outdir, casename * "p1.pdf"), figt) # T
    save(joinpath(outdir, casename * "p2.pdf"), fige) # E
    in_dim == 2 && save(joinpath(outdir, casename * "p3.pdf"), figc) # C
    size(pCAE, 1) == 2 && save(joinpath(outdir, casename * "p4.pdf"), figp) # P vs P
    save(joinpath(outdir, casename * "p5.pdf"), figq) # P vs T

    nothing
end
#======================================================#

function makeplots_parametric(
    datafiles::NTuple{N, String},
    outdir::String,
    casename::String,
) where{N}

    mkpath(outdir)

    figp = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)
    fige = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)

    axkwp = (;
        xlabel = L"\tilde{u}_1(t; \mathbf{μ})",
        ylabel = L"\tilde{u}_2(t; \mathbf{μ})",
        xlabelsize = 16,
        ylabelsize = 16,
    )

    axp1 = Axis(figp[1,1]; axkwp...)
    axp2 = Axis(figp[1,2]; axkwp...)
    axp3 = Axis(figp[1,3]; axkwp...)

    axkwe = (;
        xlabel = L"t",
        ylabel = L"ε(t; \mathbf{μ})",
        yscale = log10,
        xlabelsize = 16,
        ylabelsize = 16,
    )

    axe1 = Axis(fige[1,1]; axkwe...)
    axe2 = Axis(fige[1,2]; axkwe...)
    axe3 = Axis(fige[1,3]; axkwe...)

    #===============================#
    # FIGP
    #===============================#

    labels = if occursin("exp3", casename)
        (
            L"$\mu = 0.500$ (Training)",
            L"$\mu = 0.525$ (Interpolation)",
            L"$\mu = 0.550$ (Training)",
            L"$\mu = 0.575$ (Interpolation)",
            L"$\mu = 0.600$ (Training)",
            L"$\mu = 0.625$ (Extrapolation)",
        )
    elseif occursin("exp4", casename)
        (
            L"$\mu = 0.900$ (Training)",
            L"$\mu = 0.933$ (Training)",
            L"$\mu = 0.966$ (Training)",
            L"$\mu = 1.000$ (Interpolation)",
            L"$\mu = 1.033$ (Training)",
            L"$\mu = 1.066$ (Training)",
            L"$\mu = 1.100$ (Training)",
        )
    end

    colors = (:blue, :orange, :green, :red, :purple, :brown, :magenta)

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

        @show size(pCAE), size(qCAE)
        @show size(pSNL), size(qSNL) # wrong size.
        @show size(pSNW), size(qSNW) # wrong size.

        color = colors[i]
        label = labels[i]
        sckwq = (; color, markersize = 15, marker = :star5,)
        lnkwq = (; color, label, linewidth = 2.5, linestyle = :solid,)
        lnkwp = (; color, linewidth = 4, linestyle = :dot,)

        pplot!(axp1, tFOM, pCAE, qCAE; sckwq, lnkwq, lnkwp)
        pplot!(axp2, tFOM, pSNL, qSNL; sckwq, lnkwq, lnkwp)
        pplot!(axp3, tFOM, pSNW, qSNW; sckwq, lnkwq, lnkwp)
    end

    Label(figp[2,1], L"(a) CAE-ROM$$" , fontsize = 16)
    Label(figp[2,2], L"(b) SNFL-ROM$$", fontsize = 16)
    Label(figp[2,3], L"(c) SNFW-ROM$$", fontsize = 16)

    colsize!(figp.layout, 1, Relative(0.33))
    colsize!(figp.layout, 2, Relative(0.33))
    colsize!(figp.layout, 3, Relative(0.33))

    figp[0,:] = Legend(figp, axp1; orientation = :horizontal, framevisible = false)

    elems = [
        [
            LineElement(; linestyle = :solid, color = :black, linewidth = 3),
            MarkerElement(; marker = :star5, color = :black, markersize = 15, points = Point2f[(0.05,0.5)])
        ],
        LineElement(; linestyle = :dot, color = :black, linewidth = 6),
    ]

    l1 = [L"Learned prediction $e_{θ_e}(ū(t; \mathbf{\mu}))$"  , L"\text{Dynamics evaluation}"]
    l2 = [L"Learned prediction $\Xi_\varrho(t, \mathbf{\mu})$" , L"\text{Dynamics evaluation}"]

    axislegend(axp1, elems, l1; position = :lt, patchsize = (40, 10))
    axislegend(axp2, elems, l2; position = :lt, patchsize = (40, 10))
    axislegend(axp3, elems, l2; position = :lt, patchsize = (40, 10))

    ylims!(axp1, -8, 20)

    save(joinpath(outdir, "$(casename)p.pdf"), figp)

    #===============================#
    # FIGE
    #===============================#

    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = (L"POD-ROM$$", L"CAE-ROM$$", L"SNFL-ROM$$", L"SNFW-ROM$$",)

    in_dim, out_dim, grid = if occursin("exp3", casename)
        1, 1, (1024,)
    elseif occursin("exp4", casename)
        2, 2, (512, 512)
    end

    Nxyz = prod(grid)
    Nfom = Nxyz * out_dim

    for (j, datafile) in enumerate(datafiles[4:6])
        data = h5open(datafile)

        tFOM = data["tFOM"] |> Array
        uFOM = data["uFOM"] |> Array
        uPCA = data["uPCA"] |> Array
        uCAE = data["uCAE"] |> Array
        uSNL = data["uSNL"] |> Array
        uSNW = data["uSNW"] |> Array

        close(data)

        nr = sum(abs2, uFOM; dims = 1:in_dim+1) ./ Nfom .|> sqrt

        ePCA = (uFOM - uPCA) ./ nr
        eCAE = (uFOM - uCAE) ./ nr
        eSNL = (uFOM - uSNL) ./ nr
        eSNW = (uFOM - uSNW) ./ nr

        e2tPCA = sum(abs2, ePCA; dims = 1:in_dim+1) / Nfom |> vec
        e2tCAE = sum(abs2, eCAE; dims = 1:in_dim+1) / Nfom |> vec
        e2tSNL = sum(abs2, eSNL; dims = 1:in_dim+1) / Nfom |> vec
        e2tSNW = sum(abs2, eSNW; dims = 1:in_dim+1) / Nfom |> vec

        e2tPCA = sqrt.(e2tPCA) .+ 1f-12
        e2tCAE = sqrt.(e2tCAE) .+ 1f-12
        e2tSNL = sqrt.(e2tSNL) .+ 1f-12
        e2tSNW = sqrt.(e2tSNW) .+ 1f-12

        plt_kw = Tuple(
            (; linewidth = 3, color = colors[i], linestyle = styles[i], label = labels[i])
            for i in 1:4
        )

        if j == 1     # interpolation
            lines!(axe2, tFOM, e2tPCA; plt_kw[1]...)
            lines!(axe2, tFOM, e2tCAE; plt_kw[2]...)
            lines!(axe2, tFOM, e2tSNL; plt_kw[3]...)
            lines!(axe2, tFOM, e2tSNW; plt_kw[4]...)
        elseif j == 2 # training
            lines!(axe1, tFOM, e2tPCA; plt_kw[1]...)
            lines!(axe1, tFOM, e2tCAE; plt_kw[2]...)
            lines!(axe1, tFOM, e2tSNL; plt_kw[3]...)
            lines!(axe1, tFOM, e2tSNW; plt_kw[4]...)
        elseif j == 3 # extrapolation
            lines!(axe3, tFOM, e2tPCA; plt_kw[1]...)
            lines!(axe3, tFOM, e2tCAE; plt_kw[2]...)
            lines!(axe3, tFOM, e2tSNL; plt_kw[3]...)
            lines!(axe3, tFOM, e2tSNW; plt_kw[4]...)
        end
    end

    linkaxes!(axe1, axe2, axe3)
    fige[0,:] = Legend(fige, axe1, patchsize = (30, 10), orientation = :horizontal, framevisible = false)

    Label(fige[2,1], L"(a) $μ=0.600$ (training)"     , fontsize = 16)
    Label(fige[2,2], L"(b) $μ=0.575$ (interpolation)", fontsize = 16)
    Label(fige[2,3], L"(c) $μ=0.625$ (extrapolation)", fontsize = 16)

    colsize!(fige.layout, 1, Relative(0.33))
    colsize!(fige.layout, 2, Relative(0.33))
    colsize!(fige.layout, 3, Relative(0.33))

    save(joinpath(outdir, "$(casename)e.pdf"), fige)

    nothing
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
        sq = scatter!(ax, q[:, 1:1]; marker = :star5, sckwq...)
        lq = lines!(ax, q; linestyle = :solid, lnkwq...)
        lp = lines!(ax, p; linestyle = :dot  , lnkwp...)
        lt = ifdt ? lines!(ax, pdt; linestyle = :dash, lnkwt...) : nothing
    else
        @warn "latent size size(p, 1) == $(size(p, 1)) not supported."
        return nothing, nothing, nothing, nothing
    end

    sq, lq, lp, lt
end

function ptplot!(ax, t, p, q, pdt = nothing;
    ifdt = false, 
    lnkwq = (;),
    lnkwp = (;),
    lnkwt = (;),
)
    lq = series!(ax, t, q; linestyle = :solid, lnkwq...)
    lp = series!(ax, t, p; linestyle = :dot  , lnkwp...)
    lt = if ifdt
        idt = LinRange(1, length(t), size(pdt, 2)) .|> Base.Fix1(round, Int)
        tdt = t[idt]
        series!(ax, tdt, pdt; linestyle = :dash, lnkwt...)
    else
        nothing
    end

    lq, lp, lt
end

#======================================================#
function makeplots_hyper(
    e2hyper::String,
    outdir::String,
    casename::String,
)

    mkpath(outdir)

    # Get fields
    grid = if occursin("exp2", casename)
        128, 128
    elseif occursin("exp4", casename)
        512, 512
    end

    file = jldopen(e2hyper)
    xdata = file["xdata"]
    udata = file["udata"]
    upreds = file["uprs"]
    times = file["tims"]
    mems  = file["mems"]
    close(file)

    xdiag = diag(reshape(xdata[1, :, :], grid))
    udata = if occursin("exp2", casename)
        reshape(udata, grid)
    elseif occursin("exp4", casename)
        reshape(udata[1, :], grid)
    end

    ups = collect(reshape(upred[1, :], grid) for upred in upreds)
    ups = cat(ups...; dims = 3)

    nr  = sqrt(sum(abs2, udata) / length(udata))
    eps = @. (ups - udata) / nr

    icases = keys(upreds)
    ncases = length(upreds)

    epreds = NamedTuple{icases}(eps[:, :, i] for i in 1:ncases)
    ediags = NamedTuple{icases}(diag(eps[:, :, i]) for i in 1:ncases)
    udiags = NamedTuple{icases}(diag(ups[:, :, i]) for i in 1:ncases)
    uddiag = diag(udata)

    k1 = (:N16384_dt1 , :N4096_dt1 , :N1024_dt1 , :N256_dt1 , :N64_dt1 )
    k2 = (:N16384_dt2 , :N4096_dt2 , :N1024_dt2 , :N256_dt2 , :N64_dt2 )
    k3 = (:N16384_dt5 , :N4096_dt5 , :N1024_dt5 , :N256_dt5 , :N64_dt5 )
    k4 = (:N16384_dt10, :N4096_dt10, :N1024_dt10, :N256_dt10, :N64_dt10)

    # create dataframe / table
    df = nothing

    tFOM = times.FOM
    mFOM = mems.FOM

    e1 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k1)
    e2 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k2)
    e3 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k3)
    e4 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k4)

    # @show round.(e1 .* 100, sigdigits = 4)
    # @show round.(e2 .* 100, sigdigits = 4)
    # @show round.(e3 .* 100, sigdigits = 4)
    # @show round.(e4 .* 100, sigdigits = 4)
    
    println()

    t1 = Tuple(times[k] for k in k1)
    t2 = Tuple(times[k] for k in k2)
    t3 = Tuple(times[k] for k in k3)
    t4 = Tuple(times[k] for k in k4)

    nn = 1024^3
    
    m1 = Tuple(mems[k] for k in k1) ./ nn
    m2 = Tuple(mems[k] for k in k2) ./ nn
    m3 = Tuple(mems[k] for k in k3) ./ nn
    m4 = Tuple(mems[k] for k in k4) ./ nn
    
    s1 = tFOM ./ t1
    s2 = tFOM ./ t2
    s3 = tFOM ./ t3
    s4 = tFOM ./ t4

    # @show tFOM
    # @show round.(s1, sigdigits = 4)
    # @show round.(s2, sigdigits = 4)
    # @show round.(s3, sigdigits = 4)
    # @show round.(s4, sigdigits = 4)
    #
    # println()
    #
    # @show mFOM / nn
    # @show round.(m1, sigdigits = 4)
    # @show round.(m2, sigdigits = 4)
    # @show round.(m3, sigdigits = 4)
    # @show round.(m4, sigdigits = 4)
    
    # println()
    #
    # return df

    ### Make plots

    figc = Figure(; size = (750, 750), backgroundcolor = :white, grid = :off)
    fige = Figure(; size = (750, 850), backgroundcolor = :white, grid = :off)
    figp = Figure(; size = (750, 850), backgroundcolor = :white, grid = :off)

    # FIGC
    nlevels = 11
    levels = if occursin("exp2", casename)
        10.0 .^ range(-4, 0, nlevels)
    elseif occursin("exp4", casename)
        10.0 .^ range(-5, 0, nlevels)
    end

    kw_axc = (; aspect = 1, xlabel = L"x", ylabel = L"y")
    kw_ctr = (; extendlow = :cyan, extendhigh = :magenta, colorscale = log10, levels)

    # FIGE, FIGP

    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = (L"$|X_\text{proj}|=16384$", L"$|X_\text{proj}|=4096$", L"$|X_\text{proj}|=1024$", L"$|X_\text{proj}|=256$", L"$|X_\text{proj}|=64$",)

    kw_axe = (;
        xlabel = L"y = x",
        ylabel = L"ε(t; \mathbf{μ})",
        yscale = log10,
        xlabelsize = 16,
        ylabelsize = 16,
    )

    kw_axp = (;
        xlabel = L"y = x",
        ylabel = L"u(x, t; \mathbf{μ})",
        xlabelsize = 16,
        ylabelsize = 16,
    )

    kw_fom = (; linewidth = 3, color = :black, linestyle = :solid, label = L"FOM$$")

    kw_lin = Tuple(
        (; linewidth = 3, color = colors[j], linestyle = styles[j], label = labels[j])
        for j in 1:5
    )

    axe1 = Axis(fige[1, 1]; kw_axe...)
    axe2 = Axis(fige[1, 2]; kw_axe...)
    axe3 = Axis(fige[3, 1]; kw_axe...)
    axe4 = Axis(fige[3, 2]; kw_axe...)

    axp1 = Axis(figp[1, 1]; kw_axp...)
    axp2 = Axis(figp[1, 2]; kw_axp...)
    axp3 = Axis(figp[3, 1]; kw_axp...)
    axp4 = Axis(figp[3, 2]; kw_axp...)

    for j in 1:5
        ep1 = epreds[k1[j]] .|> abs
        ep2 = epreds[k2[j]] .|> abs
        ep3 = epreds[k3[j]] .|> abs
        ep4 = epreds[k4[j]] .|> abs

        ed1 = ediags[k1[j]] .|> abs
        ed2 = ediags[k2[j]] .|> abs
        ed3 = ediags[k3[j]] .|> abs
        ed4 = ediags[k4[j]] .|> abs

        ud1 = udiags[k1[j]]
        ud2 = udiags[k2[j]]
        ud3 = udiags[k3[j]]
        ud4 = udiags[k4[j]]

        # FIGE
        lines!(axe1, xdiag, ed1; kw_lin[j]...)
        lines!(axe2, xdiag, ed2; kw_lin[j]...)
        lines!(axe3, xdiag, ed3; kw_lin[j]...)
        lines!(axe4, xdiag, ed4; kw_lin[j]...)

        # FIGP

        if j == 1
            # FOM
            lines!(axp1, xdiag, uddiag; kw_fom...)
            lines!(axp2, xdiag, uddiag; kw_fom...)
            lines!(axp3, xdiag, uddiag; kw_fom...)
            lines!(axp4, xdiag, uddiag; kw_fom...)
        end

        lines!(axp1, xdiag, ud1; kw_lin[j]...)
        lines!(axp2, xdiag, ud2; kw_lin[j]...)
        lines!(axp3, xdiag, ud3; kw_lin[j]...)
        lines!(axp4, xdiag, ud4; kw_lin[j]...)

        # FIGC
        ax1j = Axis(figc[1,j]; kw_axc...)
        ax2j = Axis(figc[2,j]; kw_axc...)
        ax3j = Axis(figc[3,j]; kw_axc...)
        ax4j = Axis(figc[4,j]; kw_axc...)

        cf1j = contourf!(ax1j, xdiag, xdiag, ep1; kw_ctr...)
        cf2j = contourf!(ax2j, xdiag, xdiag, ep2; kw_ctr...)
        cf3j = contourf!(ax3j, xdiag, xdiag, ep3; kw_ctr...)
        cf4j = contourf!(ax4j, xdiag, xdiag, ep4; kw_ctr...)

        tightlimits!(ax1j)
        tightlimits!(ax2j)
        tightlimits!(ax3j)
        tightlimits!(ax4j)

        hidedecorations!(ax1j; label = false)
        hidedecorations!(ax2j; label = false)
        hidedecorations!(ax3j; label = false)
        hidedecorations!(ax4j; label = false)

        if j == 5
            # ticks = [1f-5, 1f-4, 1f-3, 1f-2, 1f-1,]
            # Colorbar(figc[4, :], cf1j; ticks, vertical = false)
            Colorbar(figc[5, :], cf1j; vertical = false)
        end
    end

    # FIGE, FIGP
    linkaxes!(axe1, axe2, axe3, axe4)
    linkaxes!(axp1, axp2, axp3, axp4)

    fige[0,:] = Legend(fige, axe1, patchsize = (30, 10), orientation = :horizontal, framevisible = false)
    figp[0,:] = Legend(figp, axp1, patchsize = (30, 10), orientation = :horizontal, framevisible = false)

    Label(fige[2,1], L"(a) $Δt =  1Δt_0$", fontsize = 16)
    Label(fige[2,2], L"(b) $Δt =  2Δt_0$", fontsize = 16)
    Label(fige[4,1], L"(c) $Δt =  5Δt_0$", fontsize = 16)
    Label(fige[4,2], L"(d) $Δt = 10Δt_0$", fontsize = 16)

    Label(figp[2,1], L"(a) $Δt =  1Δt_0$", fontsize = 16)
    Label(figp[2,2], L"(b) $Δt =  2Δt_0$", fontsize = 16)
    Label(figp[4,1], L"(c) $Δt =  5Δt_0$", fontsize = 16)
    Label(figp[4,2], L"(d) $Δt = 10Δt_0$", fontsize = 16)

    colsize!(fige.layout, 1, Relative(0.50))
    colsize!(fige.layout, 2, Relative(0.50))
    
    colsize!(figp.layout, 1, Relative(0.50))
    colsize!(figp.layout, 2, Relative(0.50))

    # FIGC
    Label(figc[1:4, -1][1,1], L"Time-step size $(Δt)$"; rotation = pi/2, fontsize = 16)
    Label(figc[-1, 1:4][1,1], L"Number of hyper-reduction points $(|X_\text{proj}|)$"; fontsize = 16)

    Label(figc[0,1], L"$|X_\text{proj}| = 16384$", fontsize = 16)
    Label(figc[0,2], L"$|X_\text{proj}| = 4096$" , fontsize = 16)
    Label(figc[0,3], L"$|X_\text{proj}| = 1024$" , fontsize = 16)
    Label(figc[0,4], L"$|X_\text{proj}| = 256$"  , fontsize = 16)
    Label(figc[0,5], L"$|X_\text{proj}| = 64$"   , fontsize = 16)

    Label(figc[1,0], L"$Δt =  1Δt_0$"; fontsize = 16, rotation = pi/2)
    Label(figc[2,0], L"$Δt =  2Δt_0$"; fontsize = 16, rotation = pi/2)
    Label(figc[3,0], L"$Δt =  5Δt_0$"; fontsize = 16, rotation = pi/2)
    Label(figc[4,0], L"$Δt = 10Δt_0$"; fontsize = 16, rotation = pi/2)

    rowsize!(figc.layout, 1, Relative(0.22))
    rowsize!(figc.layout, 2, Relative(0.22))
    rowsize!(figc.layout, 3, Relative(0.22))
    rowsize!(figc.layout, 4, Relative(0.22))

    colsize!(figc.layout, 1, Relative(0.19))
    colsize!(figc.layout, 2, Relative(0.19))
    colsize!(figc.layout, 3, Relative(0.19))
    colsize!(figc.layout, 4, Relative(0.19))
    colsize!(figc.layout, 5, Relative(0.19))

    # save
    # save(joinpath(outdir, "hyper_$(casename)_c.pdf"), figc)
    # save(joinpath(outdir, "hyper_$(casename)_e.pdf"), fige)
    save(joinpath(outdir, "hyper_$(casename)_p.pdf"), figp)

    return df
end

#======================================================#
outdir = joinpath(@__DIR__, "results")
datadir = joinpath(@__DIR__, "datafiles")

# EXP 1, 2, 5
e1file = joinpath(datadir, "advect1d.h5")
e2file = joinpath(datadir, "advect2d.h5")
e5file = joinpath(datadir, "ks1d.h5")

# EXP 3
e3file1 = joinpath(datadir, "burgers1dcase1.h5")
e3file2 = joinpath(datadir, "burgers1dcase2.h5")
e3file3 = joinpath(datadir, "burgers1dcase3.h5")
e3file4 = joinpath(datadir, "burgers1dcase4.h5")
e3file5 = joinpath(datadir, "burgers1dcase5.h5")
e3file6 = joinpath(datadir, "burgers1dcase6.h5")
e3files = (e3file1, e3file2, e3file3, e3file4, e3file5, e3file6)

# EXP 4
e4file1 = joinpath(datadir, "burgers2dcase1.h5")
e4file2 = joinpath(datadir, "burgers2dcase2.h5")
e4file3 = joinpath(datadir, "burgers2dcase3.h5")
e4file4 = joinpath(datadir, "burgers2dcase4.h5")
e4file5 = joinpath(datadir, "burgers2dcase5.h5")
e4file6 = joinpath(datadir, "burgers2dcase6.h5")
e4file7 = joinpath(datadir, "burgers2dcase7.h5")
e4files = (e4file1, e4file2, e4file3, e4file4, e4file5, e4file6, e4file7)

# # EXP 1, 2, 5
# makeplots(e1file, outdir, "exp1"; ifdt = true)
# makeplots(e2file, outdir, "exp2"; ifdt = false)
# makeplots(e5file, outdir, "exp5"; ifdt = false)
#
# # EXP 3
# makeplots(e3file4, outdir, "exp3case4")
# makeplots(e3file5, outdir, "exp3case5")
# makeplots_parametric(e3files, outdir, "exp3")

# # EXP 4
# makeplots(e4file4, outdir, "exp4case4")
makeplots_parametric(e4files, outdir, "exp4")

# e2hyper = joinpath(@__DIR__, "..", "experiments_SNFROM", "advect_fourier2D", "dump", "hypercompiled.jld2")
# e4hyper = joinpath(@__DIR__, "..", "experiments_SNFROM", "burgers_fourier2D", "dump", "hypercompiled.jld2")
#
# df_exp2 = makeplots_hyper(e2hyper, outdir, "exp2")
# df_exp4 = makeplots_hyper(e4hyper, outdir, "exp4")
#======================================================#
nothing
