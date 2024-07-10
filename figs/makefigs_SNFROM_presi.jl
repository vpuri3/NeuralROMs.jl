#
using GLMakie
using LinearAlgebra, HDF5, JLD2, LaTeXStrings

#======================================================#

function makeplots(
    datafile::String,
    outdir::String,
    casename::String;
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
    
    #======================================================#

    in_dim  = size(xFOM, 1)
    out_dim = size(uFOM, 1)

    Nt   = length(tFOM)
    grid = size(uFOM)[2:in_dim+1]
    Nxyz = prod(grid)
    Nfom = Nxyz * out_dim

    if in_dim == 1
        xFOM = vec(xFOM)
        uFOM = reshape(uFOM, Nxyz, Nt)
    elseif in_dim == 2
        # get diagonal
        xFOM = diag(xFOM[1, :, :])
        uFOM = hcat(Tuple(diag(uFOM[1, :, :, i]) for i in 1:Nt)...)
    end

    #======================================================#

    fig1 = Figure(; size = (600, 400), backgroundcolor = :white, grid = :off)

    kw_ax = (;
    )
    kw_sc = (;
        linewidth = 0.5,
        color = :black,
        strokewidth = 0, 
        markersize = 7.5,
    )

    Ngif  = 250
    Itgif = LinRange(1, Nt, Ngif) .|> Base.Fix1(round, Int)
    Ix = LinRange(1, grid[1], 128) .|> Base.Fix1(round, Int)

    ax1 = Axis(fig1[1,1]; kw_ax...)

    hidedecorations!(ax1)

    y = Observable(uFOM[Ix, Itgif[1]])
    p = scatterlines!(ax1, xFOM[Ix], y; kw_sc...)

    if occursin(casename, "exp5")
        ylims!(ax1, -2.5, 4.5)
    end

    function animstep(i)
        y[] = uFOM[Ix, Itgif[i]]
    end

    giffile = joinpath(outdir, casename * ".gif")
    record(animstep, fig1, giffile, 1:length(Itgif); framerate = Ngif ÷ 5,)

    # Itplt = [1, Nt ÷ 2, Nt]
    # p1 = scatterlines!(ax1, xFOM[Ix], uFOM[Ix, Itplt[1]]; kw_sc...)
    # p2 = scatterlines!(ax1, xFOM[Ix], uFOM[Ix, Itplt[2]]; kw_sc...)
    # p3 = scatterlines!(ax1, xFOM[Ix], uFOM[Ix, Itplt[3]]; kw_sc...)
    # imgfile = joinpath(outdir, casename * "p1.png")
    # save(imgfile, fig1)

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

# makeplots(e1file , outdir, "exp1")
# makeplots(e2file , outdir, "exp2")
# makeplots(e3file1, outdir, "exp3case1")
# makeplots(e4file1, outdir, "exp4case1")
# makeplots(e5file , outdir, "exp5")

#======================================================#
