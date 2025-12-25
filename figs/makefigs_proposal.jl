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
function rom_schematic(
    outdir::String;
    backend::Symbol = :GLMakie,
)
    mkpath(outdir)
    activate_backend(backend)
    #===============================#

    N = 1000

    t = LinRange(0, 2, N) |> Array

    #----------------------------------------#
    # target curve
    #----------------------------------------#
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

    #----------------------------------------#
    # CAE curve
    #----------------------------------------#
    xyzCAE = map(t) do t
        [t, 0.9 * sinpi(-1.48 * t), 1.4 * cospi(1.97 * t)] |> Point3f
    end |> collect
    xyzCAE_array = hcat(map(a -> [a.data...], xyzCAE)...)

    #----------------------------------------#
    # CAE manifold
    #----------------------------------------#
    xCAE = t .* ones(1000)'
    L = 5.0
    yCAE = ones(1000) * LinRange(-L, L, N)'
    r2 = @. (xCAE - xyzCAE_array[1,:])^2 + (yCAE - xyzCAE_array[2,:])^2
    zCAE = @. 1.5 * cospi(1.97 * xCAE) * exp(-1.0 * r2)
    zCAE[r2 .> 1.0] .= NaN
    
    #----------------------------------------#
    ## FIG
    #----------------------------------------#
    fig = Figure(; size = (1000, 700), backgroundcolor = :white, grid = :off, padding = 0)
    ax  = Axis3(fig[1,1];
        azimuth = 0.3 * pi,
        elevation = 0.0625 * pi,
        aspect = (4,1,1),
    )
    
    colsize!(fig.layout, 1, Relative(1))
    rowsize!(fig.layout, 1, Relative(1))

    hidedecorations!(ax)
    hidespines!(ax)

    ## FOM curve

    ln_kw = (; linewidth = 6, color = :red, linestyle = :solid,)
    sc_kw = (; color = :white, strokewidth = 2, markersize = 20)
    Isc = LinRange(1, N, 8) .|> Base.Fix1(round, Int)

    lines!(ax, xyz; ln_kw...,)
    scatter!(ax, xyz[Isc]; sc_kw...)

    save(joinpath(outdir, "schematic1.png"), fig)

    ## SVD projection
    Xproj = U * (U' * (X .- x̄)) .+ x̄ # [3, N]
    Xproj = map(x -> Point3f(x), eachcol(Xproj))

    lines!(ax, Xproj; linewidth = 6, color = :orange, linestyle = :dash)

    ## SVD plane
    sf_kw = (; colormap = [:black, :black], alpha = 0.5)
    surface!(ax, xPCA, yPCA, zPCA; sf_kw...)

    save(joinpath(outdir, "schematic2.png"), fig)

    ## AE line
    lCAE = lines!(ax, xyzCAE; linewidth = 6, color = :blue, linestyle = :dash,)

    ## AE manifold
    sf_kw = (; colormap = [:blue, :blue], alpha = 0.2)
    surface!(ax, xCAE, yCAE, zCAE; sf_kw...)

    save(joinpath(outdir, "schematic3.png"), fig)

    ## DONE
    fig
end

function makegrid(
    Nx, Ny;
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
# main
#======================================================#
outdir = joinpath(@__DIR__, "proposal")
rom_schematic(outdir)

#======================================================#
nothing
