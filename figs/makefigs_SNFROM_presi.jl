#
# using GLMakie
using CairoMakie
using LinearAlgebra, HDF5, JLD2, LaTeXStrings

#======================================================#

function makeplots(
    datafile::String,
    outdir::String,
    casename::String;
    ifdt::Bool = false,
    ifcrom::Bool = false,
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
        # get diagonal
        # xdiag = diag(xFOM[1, :, :])
        # uidagFOM = hcat(Tuple(diag(uFOM[1, :, :, i]) for i in 1:Nt)...)
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

    if occursin("exp3", casename) #| occursin("exp4", casename)

        Ngif  = 24 * 2
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

        y1 = Observable(uFOM[Ix, Itgif[1]])
        y2 = Observable(uFOM[ :, Itgif[1]])
        scatter!(axb, xFOM[Ix], y1; kw_sc...)
        lines!(  axb, xFOM[ :], y2; kw_ln...)

        if occursin("exp5", casename)
            ylims!(axb, -2.5, 4.5)
        elseif occursin("exp3", casename)
            ylims!(axb, 0.9, 1.7)
        elseif occursin("exp4", casename)
            ylims!(axb, -0.1, 1.1)
        end

        function anim_blob(i)
            y1[] = uFOM[Ix, Itgif[i]]
            y2[] = uFOM[ :, Itgif[i]]
        end

        # p1 = scatterlines!(axb, xFOM[Ix], uFOM[Ix, Nt]; kw_sc...)

        gifb = joinpath(outdir, casename * "-blob-FOM.gif")
        save(joinpath(outdir, casename * "-blob-FOM0.svg"), figb)
        record(anim_blob, figb, gifb, 1:Ngif; framerate = Ngif ÷ 2,)
        save(joinpath(outdir, casename * "-blob-FOM1.svg"), figb)
    end

    #======================================================#
    # make things observables
    #======================================================#

    i1, i2 = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int) |> extrema

    # placeholder
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
        obs_qSNL.val = @view qSNL[:, 1:it]
        obs_qSNW.val = @view qSNW[:, 1:it]

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
    # line plot animation + img
    #======================================================#

    figt = Figure(; size = ( 600, 400), backgroundcolor = :white, grid = :off)
    fige = Figure(; size = ( 600, 400), backgroundcolor = :white, grid = :off)

    figp = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)
    figc = Figure(; size = (1200, 400), backgroundcolor = :white, grid = :off)

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
        record(anim_func, figp, gifp, 1:Ngif; framerate = 24)
        save(joinpath(outdir, casename * "-figp.svg"), figp)
    end

    #==========================#
    # FIGT, FIGC
    #==========================#

    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = (L"POD‐ROM$$", L"CAE‐ROM$$", L"SNFL‐ROM (ours)$$", L"SNFW‐ROM (ours)$$",)

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

    labels = (L"FOM$$", L"POD‐ROM$$", L"CAE‐ROM$$", L"SNFL‐ROM (ours)$$", L"SNFW‐ROM (ours)$$")

    lines!(axt, xFOM, uFOM[])

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
makeplots(e3file1, outdir, "exp3case1")
# makeplots(e4file1, outdir, "exp4case1")
# makeplots(e5file , outdir, "exp5")

#======================================================#
nothing
