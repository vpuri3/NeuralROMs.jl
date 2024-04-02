#
using LinearAlgebra, HDF5, LaTeXStrings
# using Plots
using CairoMakie

function makeplots(
    datafile,
    outdir::String,
    casename::AbstractString,
)

    data = h5open(datafile)
    xFOM = data["xFOM"] |> Array # [in_dim, grid...]
    tFOM = data["tFOM"] |> Array # [Nt]
    uFOM = data["uFOM"] |> Array # [out_dim, grid..., Nt]
    uPCA = data["uPCA"] |> Array
    uCAE = data["uCAE"] |> Array
    uSNL = data["uSNL"] |> Array
    uSNW = data["uSNW"] |> Array

    in_dim  = size(xFOM, 1)
    out_dim = size(uFOM, 1)
    Nt = length(tFOM)

    @assert in_dim == ndims(xFOM) - 1
    @assert size(xFOM)[2:end] == size(uFOM)[2:end-1]
    @assert size(uFOM)[end] == length(tFOM)

    Itplt = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int)
    i1, i2 = Itplt[2], Itplt[5]

    # grab the first output dimension
    ii = Tuple(Colon() for _ in 1:in_dim + 1)
    uFOM = uFOM[1, ii...]
    uPCA = uPCA[1, ii...]
    uCAE = uCAE[1, ii...]
    uSNL = uSNL[1, ii...]
    uSNW = uSNW[1, ii...]

    # normalize
    nr = sum(abs2, uFOM) / length(uFOM) |> sqrt

    ePCA = (uFOM - uPCA) / nr
    eCAE = (uFOM - uCAE) / nr
    eSNL = (uFOM - uSNL) / nr
    eSNW = (uFOM - uSNW) / nr

    etPCA = sum(abs2, ePCA; dims = 1:in_dim) / prod(size(uFOM)[1:in_dim]) |> vec
    etCAE = sum(abs2, eCAE; dims = 1:in_dim) / prod(size(uFOM)[1:in_dim]) |> vec
    etSNL = sum(abs2, eSNL; dims = 1:in_dim) / prod(size(uFOM)[1:in_dim]) |> vec
    etSNW = sum(abs2, eSNW; dims = 1:in_dim) / prod(size(uFOM)[1:in_dim]) |> vec

    upreds  = (uPCA, uCAE, uSNL, uSNW,)
    epreds  = (ePCA, eCAE, eSNL, eSNW,)
    etpreds = (etPCA, etCAE, etSNL, etSNW,)

    figt = Figure(; size = (900, 400), backgroundcolor = :white, grid = :off)
    fige = Figure(; backgroundcolor = :white, grid = :off)
    figc = Figure(; size = (1000, 800), backgroundcolor = :white, grid = :off)

    axt0 = Axis(figt[1,1]; xlabel = L"x", ylabel = L"u(x, t)")
    axt1 = Axis(figt[1,2]; xlabel = L"x", ylabel = L"u(x, t)")
    axer = Axis(fige[1,1]; xlabel = L"t", ylabel = L"ε^2(t)", yscale = log10)
    
    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot,)
    mshape = (:circle, :utriangle, :diamond, :dtriangle, :star5,)
    labels = ("POD", "CAE", "SNFL)", "SNFW")

    for (i, (up, ep, et)) in enumerate(zip(upreds, epreds, etpreds))

        color = colors[i]
        label = labels[i]
        linestyle = styles[i]

        plt_kw = (; color, label, linewidth = 3, linestyle)
        cax_kw = (; aspect = 1, xlabel = L"x", ylabel = L"y")

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
                # contour plots
                ax1 = Axis(figc[1,1]; cax_kw...)
                ax2 = Axis(figc[2,1]; cax_kw...)

                contourf!(ax1, xdiag, xdiag, uFOM[:, :, i1])
                contourf!(ax2, xdiag, xdiag, uFOM[:, :, i2])

                # diagonal plots
                uddiag1 = diag(uFOM[:, :, i1])
                uddiag2 = diag(uFOM[:, :, i2])

                axt0.xlabel = L"x = y"
                axt1.xlabel = L"x = y"

                axt0.ylabel = L"u(x = y, t)"
                axt1.ylabel = L"u(x = y, t)"

                lines!(axt0, xdiag, uddiag1; linewidth = 5, label = "FOM", color = :black)
                lines!(axt1, xdiag, uddiag2; linewidth = 5, label = "FOM", color = :black)
            end

            # t0, t1
            if i != 3
                j = i != 4 ? i+1 : i
                ax1 = Axis(figc[1,j]; cax_kw...)
                ax2 = Axis(figc[2,j]; cax_kw...)

                contourf!(ax1, xdiag, xdiag, up[:, :, i1])
                contourf!(ax2, xdiag, xdiag, up[:, :, i2])
            end

            # error
            ax3 = Axis(figc[3,i]; cax_kw...)
            contourf!(ax3, xdiag, xdiag, ep[:, :, i2])

            # diagonal plots
            up1 = diag(up[:, :, i1])
            up2 = diag(up[:, :, i2])

            lines!(axt0, xdiag, up1; plt_kw...)
            lines!(axt1, xdiag, up2; plt_kw...)
        end

        lines!(axer, tFOM, et; linewidth = 3, label = labels[i], plt_kw...)
    end

    axislegend(axer; position = :lt, patchsize = (30, 10), orientation = :horizontal)
    figt[2,:] = Legend(figt, axt1; patchsize = (35, 10), orientation = :horizontal)

    if in_dim == 2
        Colorbar(figc[1,5])# limits = clims1)
        Colorbar(figc[2,5])# limits = clims2)
        Colorbar(figc[3,5])# limits = climse)
    end

    linkaxes!(axt0, axt1)
    # linkaxes!(figc.content[:]...)

    # synchronize color range

    save(joinpath(outdir, casename * "p1.png"), figt)
    save(joinpath(outdir, casename * "p2.png"), fige)

    if in_dim == 2
        save(joinpath(outdir, casename * "p3.png"), figc)
    end

    nothing
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

# makeplots(e1file, outdir, "exp1")
makeplots(e2file, outdir, "exp2")
# makeplots(e4file, outdir, "exp4")
# makeplots(e5file, outdir, "exp5")

# makeplots(e3file1, outdir, "exp3case1")
# makeplots(e3file2, outdir, "exp3case2")
# makeplots(e3file3, outdir, "exp3case3")
# makeplots(e3file4, outdir, "exp3case4")
# makeplots(e3file5, outdir, "exp3case5")

#======================================================#
nothing
