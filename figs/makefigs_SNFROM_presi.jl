#
using GLMakie
using CairoMakie
using Random, LinearAlgebra, HDF5, JLD2, LaTeXStrings

rng = Random.default_rng()
Random.seed!(rng, 0)

#======================================================#

function activate_backend(backend::Symbol)
    if backend === :GLMakie
        @info "Activating GLMakie"
        GLMakie.activate!()
    elseif backend === :CairoMakie
        @info "Activating CairoMakie"
        CairoMakie.activate!()
    end
    nothing
end

#======================================================#

function makeplots(
    datafile::String,
    outdir::String,
    casename::String;
    ifdt::Bool = false,
    ifcrom::Bool = false,
    ifFOM::Bool = true,
    ifPCA::Bool = true,
    ifCAE::Bool = true,
    ifSNL::Bool = true,
    ifSNW::Bool = true,

    backend::Symbol = :CairoMakie,
)
    activate_backend(backend)

    framerate = 24
    imgext = backend === :GLMakie ? ".png" : ".svg"

    #======================================================#
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
    uCRM = uFOM * NaN32
    pCRM = pSNW * NaN32
    qCRM = qSNW * NaN32

    if ifcrom
        uCRM = data["uCRM"] |> Array
        pCRM = data["pCRM"] |> Array
        qCRM = data["qCRM"] |> Array
    end

    # DT
    tdtFOM = tFOM * NaN32
    udtFOM = uFOM * NaN32

    udtCAE = uSNW * NaN32
    udtSNL = uSNW * NaN32
    udtSNW = uSNW * NaN32

    pdtCAE = pSNW * NaN32
    pdtSNL = pSNW * NaN32
    pdtSNW = pSNW * NaN32

    if ifdt
        tdtFOM = data["tdtFOM"] |> Array
        udtFOM = data["udtFOM"] |> Array

        pdtCAE = data["pdtCAE"] |> Array
        pdtSNL = data["pdtSNL"] |> Array
        pdtSNW = data["pdtSNW"] |> Array

        udtCAE = data["udtCAE"] |> Array
        udtSNL = data["udtSNL"] |> Array
        udtSNW = data["udtSNW"] |> Array
    end

    close(data)

    #======================================================#

    in_dim  = size(xFOM, 1)
    out_dim = size(uFOM, 1)
    Nt = length(tFOM)
    Ntdt = length(tdtFOM)

    grid = size(uFOM)[2:in_dim+1]
    Nxyz = prod(grid)
    Nfom = Nxyz * out_dim

    @assert in_dim == ndims(xFOM) - 1
    @assert size(xFOM)[2:end] == size(uFOM)[2:end-1]
    @assert size(uFOM)[end] == length(tFOM)

    if in_dim == 1
        xFOM = vec(xFOM)
    elseif in_dim == 2
        xdiagFOM = diag(xFOM[1, :, :])
        udiagFOM = hcat(Tuple(diag(uFOM[1, :, :, i]) for i in 1:Nt)...)
           
        udiagPCA = hcat(Tuple(diag(uPCA[1, :, :, i]) for i in 1:Nt)...)
        udiagCAE = hcat(Tuple(diag(uCAE[1, :, :, i]) for i in 1:Nt)...)
        udiagSNL = hcat(Tuple(diag(uSNL[1, :, :, i]) for i in 1:Nt)...)
        udiagSNW = hcat(Tuple(diag(uSNW[1, :, :, i]) for i in 1:Nt)...)
    end

    #======================================================#
    # normalize

    nr = sum(abs2, uFOM; dims = 2:in_dim+1) ./ Nxyz .|> sqrt

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

    # grab the first output dimension for plotting
    begin
        ii = Tuple(Colon() for _ in 1:in_dim + 1)

        uFOM = uFOM[1, ii...]

        uPCA = uPCA[1, ii...]
        uCAE = uCAE[1, ii...]
        uSNL = uSNL[1, ii...]
        uSNW = uSNW[1, ii...]

        ePCA = ePCA[1, ii...]
        eCAE = eCAE[1, ii...]
        eSNL = eSNL[1, ii...]
        eSNW = eSNW[1, ii...]
    end

    #======================================================#
    # Blob animation
    #======================================================#

    if occursin("exp3", casename) | occursin("exp3", casename)

        Ngif  = framerate * 2
        Itgif = LinRange(1, Nt, Ngif) .|> Base.Fix1(round, Int)
        Ix = LinRange(1, grid[1], 12) .|> Base.Fix1(round, Int)
        
        figb = Figure(; size = (600, 200), backgroundcolor = :white, grid = :off)
        kw_ax = (;)

        kw_sc = (;
            color = :red,
            strokewidth = 0, 
            markersize = 40,
        )

        kw_ln = (;
            linewidth = 5.0,
            color = :red,
        )

        axb = Axis(figb[1,1]; kw_ax...)

        hidedecorations!(axb)
        hidespines!(axb)

        if in_dim == 1
            y1 = Observable(uFOM[Ix, Itgif[1]])
            y2 = Observable(uFOM[ :, Itgif[1]])
            scatter!(axb, xFOM[Ix], y1; kw_sc...)
            lines!(  axb, xFOM[ :], y2; kw_ln...)
        elseif in_dim == 2

            y1 = Observable(udiagFOM[Ix, Itgif[1]])
            y2 = Observable(udiagFOM[ :, Itgif[1]])

            scatter!(axb, xdiagFOM[Ix], y1; kw_sc...)
            lines!(  axb, xdiagFOM[ :], y2; kw_ln...)
        end

        if occursin("exp5", casename)
            ylims!(axb, -2.5, 4.5)
        elseif occursin("exp3", casename)
            ylims!(axb, 0.9, 1.7)
        elseif occursin("exp4", casename)
            ylims!(axb, -0.1, 1.1)
        end

        function anim_blob(i)
            if in_dim == 1
                y1[] = uFOM[Ix, Itgif[i]]
                y2[] = uFOM[ :, Itgif[i]]
            elseif in_dim == 2
                y1[] = udiagFOM[Ix, Itgif[i]]
                y2[] = udiagFOM[ :, Itgif[i]]
            end
        end

        # p1 = scatterlines!(axb, xFOM[Ix], uFOM[Ix, Nt]; kw_sc...)

        gifb = joinpath(outdir, casename * "-figb.gif")
        save(joinpath(outdir, casename * "-figb0" * imgext), figb)
        record(anim_blob, figb, gifb, 1:Ngif; framerate,)
        save(joinpath(outdir, casename * "-figb1" * imgext), figb)

        println("$casename: FIGB done")
    end

    #======================================================#
    # make things observables
    #======================================================#

    i1, i2 = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int) |> extrema

    # placeholder to initialize observables
    ih = if occursin("exp3", casename) | occursin("exp4", casename)
        i1
    else
        i2
    end

    ii = Tuple(Colon() for _ in 1:in_dim)

    # T
    obs_tFOM = Observable(tFOM) # [Nt]

    # U
    obs_uFOM = Observable(uFOM[ii..., ih]) # [grid..., Nt]

    obs_uPCA = Observable(uPCA[ii..., ih])
    obs_uCAE = Observable(uCAE[ii..., ih])
    obs_uSNL = Observable(uSNL[ii..., ih])
    obs_uSNW = Observable(uSNW[ii..., ih])

    if in_dim == 2
        obs_udiagFOM = Observable(udiagFOM[:, ih])

        obs_udiagPCA = Observable(udiagPCA[:, ih])
        obs_udiagCAE = Observable(udiagCAE[:, ih])
        obs_udiagSNL = Observable(udiagSNL[:, ih])
        obs_udiagSNW = Observable(udiagSNW[:, ih])
    end

    # E
    obs_ePCA = Observable(ePCA[ii..., ih])
    obs_eCAE = Observable(eCAE[ii..., ih])
    obs_eSNL = Observable(eSNL[ii..., ih])
    obs_eSNW = Observable(eSNW[ii..., ih])

    obs_e2tPCA = Observable(e2tPCA) # [Nt]
    obs_e2tCAE = Observable(e2tCAE)
    obs_e2tSNL = Observable(e2tSNL)
    obs_e2tSNW = Observable(e2tSNW)

    # Q
    obs_qCAE = Observable(qCAE) # [Nrom, Nt]
    obs_qSNL = Observable(qSNL)
    obs_qSNW = Observable(qSNW)

    # P
    obs_pCAE = Observable(pCAE)
    obs_pSNL = Observable(pSNL)
    obs_pSNW = Observable(pSNW)

    # Pdt
    obs_tdtFOM = Observable(tdtFOM) # [Nt]

    obs_pdtCAE = Observable(pdtCAE)
    obs_pdtSNL = Observable(pdtSNL)
    obs_pdtSNW = Observable(pdtSNW)

    if ifcrom
        obs_uCRM = Observable(uCRM[ii..., ih])
        obs_pCRM = Observable(pCRM)
        obs_qCRM = Observable(qCRM)
    end

    # GIF parameters
    Ngif  = 24 * 5
    It   = LinRange(1, Nt  , Ngif) .|> Base.Fix1(round, Int)
    Itdt = LinRange(1, Ntdt, Ngif) .|> Base.Fix1(round, Int)

    function anim_func(step)
        it   = It[step]
        itdt = Itdt[step]

        # does not retrigger plotting

        # T
        obs_tFOM.val = @view tFOM[1:it]

        # U
        obs_uFOM.val = @view uFOM[ii..., it]

        obs_uPCA.val = @view uPCA[ii..., it]
        obs_uCAE.val = @view uCAE[ii..., it]
        obs_uSNL.val = @view uSNL[ii..., it]
        obs_uSNW.val = @view uSNW[ii..., it]

        if in_dim == 2
            obs_udiagFOM.val = @view udiagFOM[:, it]

            obs_udiagPCA.val = @view udiagPCA[:, it]
            obs_udiagCAE.val = @view udiagCAE[:, it]
            obs_udiagSNL.val = @view udiagSNL[:, it]
            obs_udiagSNW.val = @view udiagSNW[:, it]
        end

        # E
        obs_ePCA.val = @view ePCA[ii..., it]
        obs_eCAE.val = @view eCAE[ii..., it]
        obs_eSNL.val = @view eSNL[ii..., it]
        obs_eSNW.val = @view eSNW[ii..., it]

        obs_e2tPCA.val = @view e2tPCA[1:it]
        obs_e2tCAE.val = @view e2tCAE[1:it]
        obs_e2tSNL.val = @view e2tSNL[1:it]
        obs_e2tSNW.val = @view e2tSNW[1:it]

        # Q
        obs_qCAE.val = @view qCAE[:, 1:it]

        if !occursin("exp4", casename) # TODO: rerun on Eagle
            obs_qSNL.val = @view qSNL[:, 1:it]
            obs_qSNW.val = @view qSNW[:, 1:it]
        end

        # P
        obs_pCAE.val = @view pCAE[:, 1:it]
        obs_pSNL.val = @view pSNL[:, 1:it]
        obs_pSNW.val = @view pSNW[:, 1:it]

        # Pdt
        obs_tdtFOM.val = @view tdtFOM[1:itdt]

        obs_pdtCAE.val = @view pdtCAE[:, 1:itdt]
        obs_pdtSNL.val = @view pdtSNL[:, 1:itdt]
        obs_pdtSNW.val = @view pdtSNW[:, 1:itdt]

        # trigger plotting for each component

        # T
        notify(obs_tFOM)

        # U
        notify(obs_uFOM)

        notify(obs_uPCA)
        notify(obs_uCAE)
        notify(obs_uSNL)
        notify(obs_uSNW)

        if in_dim == 2
            notify(obs_udiagFOM)

            notify(obs_udiagPCA)
            notify(obs_udiagCAE)
            notify(obs_udiagSNL)
            notify(obs_udiagSNW)
        end

        # E
        notify(obs_ePCA)
        notify(obs_eCAE)
        notify(obs_eSNL)
        notify(obs_eSNW)

        notify(obs_e2tPCA)
        notify(obs_e2tCAE)
        notify(obs_e2tSNL)
        notify(obs_e2tSNW)

        # Q
        notify(obs_qCAE)
        notify(obs_qSNL)
        notify(obs_qSNW)

        # P
        notify(obs_pCAE)
        notify(obs_pSNL)
        notify(obs_pSNW)

        # Pdt
        notify(obs_tdtFOM)

        notify(obs_pdtCAE)
        notify(obs_pdtSNL)
        notify(obs_pdtSNW)
    end


    #======================================================#
    # everything
    #======================================================#

    figt = Figure(; size = ( 600, 400), backgroundcolor = :white, grid = :off)
    fige = Figure(; size = ( 600, 400), backgroundcolor = :white, grid = :off)

    figp = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)

    ylabel_t, ylabel_e = if occursin("exp3", casename) | occursin("exp4", casename)
        L"u(x, t; \mathbf{μ})", L"ε(t; \mathbf{μ})"
    else
        L"u(x, t)", L"ε(t)"
    end

    axt = Axis(figt[1,1]; xlabel = L"x", ylabel = ylabel_t, xlabelsize = 16, ylabelsize = 16)
    axe = Axis(fige[1,1]; xlabel = L"t", ylabel = ylabel_e, yscale = log10, xlabelsize = 16, ylabelsize = 16)

    #==========================#
    # FIGP
    #==========================#

    if size(pCAE, 1) == 2 & !(occursin("exp4", casename)) # TODO: get exp4 from eagle

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

        sq, lq, lp, lt = pplot!(axp1, obs_tFOM, obs_pCAE, obs_qCAE, obs_pdtCAE; kwCAE...)
        sq, lq, lp, lt = pplot!(axp2, obs_tFOM, obs_pSNL, obs_qSNL, obs_pdtSNL; kwSNF...)
        sq, lq, lp, lt = pplot!(axp3, obs_tFOM, obs_pSNW, obs_qSNW, obs_pdtSNW; kwSNF...)

        Label(figp[2,1], L"CAE‐ROM$$", fontsize = 16)
        Label(figp[2,2], L"SNFL‐ROM (ours)$$", fontsize = 16)
        Label(figp[2,3], L"SNFW‐ROM (ours)$$", fontsize = 16)

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

        gifp = joinpath(outdir, casename * "-figp.gif")
        record(anim_func, figp, gifp, 1:Ngif; framerate)
        save(joinpath(outdir, casename * "-figp" * imgext), figp)
    end

    println("$casename: FIGP done")

    #==========================#
    # FIGT, FIGE
    #==========================#

    colors = (:black, :orange, :green, :blue, :red, :brown,)
    styles = (:solid, :solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = (L"FOM$$", L"POD‐ROM$$", L"CAE‐ROM$$", L"SNFL‐ROM (ours)$$", L"SNFW‐ROM (ours)$$")

    ln_kw = Tuple(
        (; linewidth = 3, label = labels[i], color = colors[i], linestyle = styles[i],)
        for i in 1:5
    )

    # FIGT
    if in_dim == 1
        ifFOM && lines!(axt, xFOM, obs_uFOM; ln_kw[1]...)

        ifPCA && lines!(axt, xFOM, obs_uPCA; ln_kw[2]...)
        ifCAE && lines!(axt, xFOM, obs_uCAE; ln_kw[3]...)
        ifSNL && lines!(axt, xFOM, obs_uSNL; ln_kw[4]...)
        ifSNW && lines!(axt, xFOM, obs_uSNW; ln_kw[5]...)

    elseif in_dim == 2
        ifFOM && lines!(axt, xdiagFOM, obs_udiagFOM; ln_kw[1]...)
                 
        ifPCA && lines!(axt, xdiagFOM, obs_udiagPCA; ln_kw[2]...)
        ifCAE && lines!(axt, xdiagFOM, obs_udiagCAE; ln_kw[3]...)
        ifSNL && lines!(axt, xdiagFOM, obs_udiagSNL; ln_kw[4]...)
        ifSNW && lines!(axt, xdiagFOM, obs_udiagSNW; ln_kw[5]...)

        axt.xlabel = L"x = y"
    end

    # FIGE
    lines!(axe, obs_tFOM, obs_e2tPCA; ln_kw[2]...)
    lines!(axe, obs_tFOM, obs_e2tCAE; ln_kw[3]...)
    lines!(axe, obs_tFOM, obs_e2tSNL; ln_kw[4]...)
    lines!(axe, obs_tFOM, obs_e2tSNW; ln_kw[5]...)

    # legends
    if occursin("exp1", casename)
        figl1 = Figure(; size = ( 800, 100), backgroundcolor = :white, grid = :off)
        Legend(figl1[1,1], axe, patchsize = (30, 10), orientation = :horizontal, framevisible = false)

        figl2 = Figure(; size = ( 200, 200), backgroundcolor = :white, grid = :off)
        Legend(figl2[1,1], axe; patchsize = (30, 30), orientation = :vertical, framevisible = false)

        save(joinpath(outdir, "legend1" * imgext), figl1)
        save(joinpath(outdir, "legend2" * imgext), figl2)
    end
    
    if occursin("exp3", casename)
        ylims!(fige.content[1], 10^-5, 10^-1)
    end

    gift = joinpath(outdir, casename * "-figt.gif")
    gife = joinpath(outdir, casename * "-fige.gif")

    record(anim_func, figt, gift, 1:Ngif; framerate)
    record(anim_func, fige, gife, 1:Ngif; framerate)

    println("$casename: FIGT done")
    println("$casename: FIGE done")

    #==========================#
    # FIGC
    #==========================#

    figc1 = Figure(; size = (500, 500), backgroundcolor = :white, grid = :off)
    figc2 = Figure(; size = (500, 500), backgroundcolor = :white, grid = :off)
    figc3 = Figure(; size = (500, 500), backgroundcolor = :white, grid = :off)
    figc4 = Figure(; size = (500, 500), backgroundcolor = :white, grid = :off)
    figc5 = Figure(; size = (500, 500), backgroundcolor = :white, grid = :off)

    if in_dim == 2
        levels = if occursin("exp2", casename)
            n = 11

            l1 = range(-0.2, 1.2, n)     # FOM
            l2 = range(-0.2, 1.2, n)     #
            l3 = 10.0 .^ range(-4, 0, n) # ER

            l1, l2, l3
        elseif occursin("exp4", casename)
            n = 11

            l1 = range(-0.2, 1.1, n)     # FOM
            l2 = range(-0.2, 1.1, n)     #
            l3 = 10.0 .^ range(-5, 0, n) # ER

            l1, l2, l3
        end

        cax_kw = (; aspect = 1, xlabel = L"x", ylabel = L"y")
        ctr_kw = (; extendlow = :cyan, extendhigh = :magenta,)

        axc1 = Axis(figc1[1,1]; cax_kw...)
        axc2 = Axis(figc2[1,1]; cax_kw...)
        axc3 = Axis(figc3[1,1]; cax_kw...)
        axc4 = Axis(figc4[1,1]; cax_kw...)
        axc5 = Axis(figc5[1,1]; cax_kw...)

        # cf1 = contourf!(axc1, xdiagFOM, xdiagFOM, obs_uFOM; ctr_kw..., levels = levels[1])
        # cf2 = contourf!(axc2, xdiagFOM, xdiagFOM, obs_ePCA; ctr_kw..., levels = levels[3], colorscale = log10,)
        # cf3 = contourf!(axc3, xdiagFOM, xdiagFOM, obs_eCAE; ctr_kw..., levels = levels[3], colorscale = log10,)
        # cf4 = contourf!(axc4, xdiagFOM, xdiagFOM, obs_eSNL; ctr_kw..., levels = levels[3], colorscale = log10,)
        # cf5 = contourf!(axc5, xdiagFOM, xdiagFOM, obs_eSNW; ctr_kw..., levels = levels[3], colorscale = log10,)

        hmp_kw = (; lowclip = :cyan, highclip = :magenta,)

        cf1 = heatmap!(axc1, xdiagFOM, xdiagFOM, obs_uFOM; hmp_kw..., colorrange = extrema(levels[1]))
        cf2 = heatmap!(axc2, xdiagFOM, xdiagFOM, @lift(abs.($obs_ePCA)); hmp_kw..., colorrange = extrema(levels[3]), colorscale = log10,)
        cf3 = heatmap!(axc3, xdiagFOM, xdiagFOM, @lift(abs.($obs_eCAE)); hmp_kw..., colorrange = extrema(levels[3]), colorscale = log10,)
        cf4 = heatmap!(axc4, xdiagFOM, xdiagFOM, @lift(abs.($obs_eSNL)); hmp_kw..., colorrange = extrema(levels[3]), colorscale = log10,)
        cf5 = heatmap!(axc5, xdiagFOM, xdiagFOM, @lift(abs.($obs_eSNW)); hmp_kw..., colorrange = extrema(levels[3]), colorscale = log10,)

        for ax in (axc1, axc2, axc3, axc4, axc5)
            tightlimits!(ax)
            hidedecorations!(ax, label = false)
        end

        Colorbar(figc1[1,2], cf1) # FOM
        Colorbar(figc2[1,2], cf2) # ER
        Colorbar(figc3[1,2], cf3)
        Colorbar(figc4[1,2], cf4)
        Colorbar(figc5[1,2], cf5)

        gifc1 = joinpath(outdir, casename * "-figc1.mkv")

        gifc2 = joinpath(outdir, casename * "-figc2.mkv")
        gifc3 = joinpath(outdir, casename * "-figc3.mkv")
        gifc4 = joinpath(outdir, casename * "-figc4.mkv")
        gifc5 = joinpath(outdir, casename * "-figc5.mkv")

        record(anim_func, figc1, gifc1, 1:Ngif; framerate)
        save(joinpath(outdir, casename * "-figc1" * ".png"), figc1)
        println("$casename: FIGC1 done")

        record(anim_func, figc2, gifc2, 1:Ngif; framerate)
        save(joinpath(outdir, casename * "-figc2" * ".png"), figc2)
        println("$casename: FIGC2 done")

        record(anim_func, figc3, gifc3, 1:Ngif; framerate)
        save(joinpath(outdir, casename * "-figc3" * ".png"), figc3)
        println("$casename: FIGC3 done")

        record(anim_func, figc4, gifc4, 1:Ngif; framerate)
        save(joinpath(outdir, casename * "-figc4" * ".png"), figc4)
        println("$casename: FIGC4 done")

        record(anim_func, figc5, gifc5, 1:Ngif; framerate)
        save(joinpath(outdir, casename * "-figc5" * ".png"), figc5)
        println("$casename: FIGC5 done")

        println("$casename: FIGC done")
    end

    #==========================#
    # DONE
    #==========================#
    return
end
#======================================================#

function rom_schematic(
    outdir::String;
    backend::Symbol = :GLMakie,
)
    mkpath(outdir)
    activate_backend(backend)
    #===============================#

    N = 1000

    t = LinRange(0, 2, N) |> Array

    # points
    xyz = map(t) do t
        [t, 1 * sinpi(-1.5t), 1.5 * cospi(2t)] |> Point3f
    end

    # PCA
    X = hcat(map(a -> [a.data...], xyz)...) # [3, N]
    x̄ = vec(sum(X, dims = 2)) ./ N

    U = svd(X .- x̄).U
    u1, u2, u3 = U[:, 1], U[:, 2], U[:, 3]
    U = hcat(u2, u3)

    rPCA, sPCA = makegrid(10, 10)
    xPCA = @. U[1,1] * rPCA + U[1,2] * sPCA .+ x̄[1]
    yPCA = @. U[2,1] * rPCA + U[2,2] * sPCA .+ x̄[2]
    zPCA = @. U[3,1] * rPCA + U[3,2] * sPCA .+ x̄[3]

    # CAE
    xCAE = xyz .+ 0.005 * collect(Point3f(rand(3)) for _ in 1:N)

    ## FIG
    fig = Figure(; size = (1000, 700), backgroundcolor = :white, grid = :off)
    ax  = Axis3(fig[1,1];
        azimuth = 0.3 * pi,
        elevation = 0.0625 * pi,
        aspect = (4,1,1),
    )

    hidedecorations!(ax)
    hidespines!(ax)

    ## FOM curve

    ln_kw = (; linewidth = 6, color = :red, linestyle = :solid,)
    sc_kw = (; color = :white, strokewidth = 2, markersize = 20)
    Isc = LinRange(1, N, 8) .|> Base.Fix1(round, Int)

    lines!(ax, xyz; ln_kw...,)
    scatter!(ax, xyz[Isc]; sc_kw...)

    save(joinpath(outdir, "schematic1.png"), fig)

    ## AE line
    lFOM = lines!(ax, xCAE; linewidth = 6, color = :green, linestyle = :dash,)

    ## SVD projection
    Xproj = U * (U' * (X .- x̄)) .+ x̄ # [3, N]
    Xproj = map(x -> Point3f(x), eachcol(Xproj))
    
    lines!(ax, Xproj; linewidth = 6, color = :orange, linestyle = :dash)
    # scatter!(ax, Xproj[Isc]; sc_kw..., alpha = 0.5)

    ## SVD plane
    sf_kw = (; colormap = [:black, :black], alpha = 0.5)
    surface!(ax, xPCA, yPCA, zPCA; sf_kw...)

    # save(joinpath(outdir, "schematic2.png"), fig)
    save(joinpath(outdir, "schematic3.png"), fig)

    ## DONE
    fig
end

function makegrid(Nx, Ny;
    x0 = -1f0,
    x1 =  1f0,
    y0 = -1f0,
    y1 =  1f0,
)
    rx = LinRange(x0, x1, Nx)
    ry = LinRange(y0, y1, Ny)
    ox = ones(Nx)
    oy = ones(Ny)

    x = rx .* oy'
    y = ox .* ry'

    x, y
end

#======================================================#

# P vs P
function pplot!(ax, t, p, q, pdt = nothing;
    ifdt = false, 
    sckwq = (;),
    lnkwq = (;),
    lnkwp = (;),
    lnkwt = (;),
)
    if size(p[], 1) == 2
        sq = scatter!(ax, q[][:, 1:1]; marker = :star5, sckwq...)
        lq = lines!(ax, q; linestyle = :solid, lnkwq...)
        lp = lines!(ax, p; linestyle = :dot  , lnkwp...)
        lt = ifdt ? lines!(ax, pdt; linestyle = :dash, lnkwt...) : nothing
    else
        @warn "latent size size(p, 1) == $(size(p, 1)) not supported."
        return nothing, nothing, nothing, nothing
    end

    sq, lq, lp, lt
end

# P vs T
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
# main
#======================================================#
outdir = joinpath(@__DIR__, "presentation")
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
e4file4 = joinpath(datadir, "burgers2dcase4.h5")
e4file7 = joinpath(datadir, "burgers2dcase7.h5")
e4files = (e4file1, e4file4, e4file7)

#======================================================#

# makeplots(e1file , outdir, "exp1", ifdt = true)
# makeplots(e2file , outdir, "exp2")
# makeplots(e3file4, outdir, "exp3case4")
# makeplots(e4file4, outdir, "exp4case4")
# makeplots(e5file , outdir, "exp5")

# rom_schematic(joinpath(outdir, "schematic/"))

#======================================================#
nothing
