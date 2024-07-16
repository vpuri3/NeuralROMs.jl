#
"""
make manifold learning plot
"""

using Random
using GLMakie
using LinearAlgebra

rng = Random.default_rng()
Random.seed!(rng, 123)

function zfunc(x, y; ϵ = 0.0, rng = Random.default_rng())
    1 - x^2 + ϵ * randn(rng)
end

function makegrid(Nx, Ny)
    rx = LinRange(-1, 1, Nx)
    ry = LinRange(-1, 1, Ny)
    ox = ones(Nx)
    oy = ones(Ny)

    x = rx .* oy'
    y = ox .* ry'

    x, y
end

#===============================#

Nxpts, Nypts = 20, 10
Nxsrf, Nysrf = 200, 100

xpts, ypts = makegrid(Nxpts, Nypts)
xsrf, ysrf = makegrid(Nxsrf, Nysrf)

zpts = @. zfunc(xpts, ypts; ϵ = 1e-2, rng)
zpca = 0.5 .+ 0 * xsrf
zcrm = @. zfunc(xsrf, ysrf; ϵ = 2e-2)
zsnf = @. zfunc(xsrf, ysrf)
#===============================#

xy_label_kw = (;
    xlabelvisible = false,
    ylabelvisible = false,

    xticksvisible = false,
    yticksvisible = false,

    xticklabelsvisible = false,
    yticklabelsvisible = false,
)

z_label_kw = (;
    zlabelvisible = false,
    zticksvisible = false,
    zticklabelsvisible = false,
)

#===============================#

fig1 = Figure(
    size = (800, 800),
    backgroundcolor = :white,
    grid = :off,
)

ax1 = Axis3(
    fig1[1,1];
    title = L"Simulation snapshots embedded in parametric space$$",
    titlesize = 32,
    azimuth = pi/4,
    elevation = 0.3,
    # aspect = (1, 1, 1),
    limits = (-1.2, 1.2, -1.2, 1.2, 0, 1.1),

    xy_label_kw...,
    z_label_kw...,
)

l1 = scatter!(ax1, xpts, ypts, zpts;
    # color = :black,
    color = vec(xpts .+ ypts),
    colormap = [:red, :blue],
    markersize = 15,
)

t1 = text!(ax1, 1, 1, -0.2; text = L"\mathbb{R}^N", fontsize = 40)

#===============================#

fig2 = Figure(
    size = (800, 800),
    backgroundcolor = :white,
    grid = :off,
)

ax2 = Axis(
    fig2[1,1];
    title = L"Snapshot coordinates on reduced manifold$$",
    titlesize = 32,
    limits = (-1.2, 1.2, -1.2, 1.2),
    xy_label_kw...,
)

l2 = scatter!(ax2, vec(xpts), vec(ypts);
    color = vec(xpts .+ ypts),
    colormap = [:red, :blue],
    markersize = 15,
)

t2 = text!(ax2, 0, -1.2; text = L"\mathbb{R}^r", fontsize = 40)

#===============================#

fig3 = Figure(
    size = (800, 800),
    backgroundcolor = :white,
    grid = :off,
)

ax3 = Axis3(
    fig3[1,1];
    title = L"PCA fits hyper-plane to data$$",
    titlesize = 32,
    azimuth = pi/4,
    elevation = 0.3,
    # aspect = (1, 1, 1),
    limits = (-1.2, 1.2, -1.2, 1.2, 0, 1.1),

    xy_label_kw...,
    z_label_kw...,
)

l3 = scatter!(ax3, xpts, ypts, zpts;
    # color = :black,
    color = vec(xpts .+ ypts),
    colormap = [:red, :blue],
    markersize = 15,
)

surface!(ax3, xsrf, ysrf, zpca;
    color = 0*vec(xsrf),
    colormap = [:black, :black],
    alpha = 0.5,
    transparent = true,
)

#===============================#

fig4 = Figure(
    size = (800, 800),
    backgroundcolor = :white,
    grid = :off,
)

ax4 = Axis3(
    fig4[1,1];
    title = L"We fit a nonlinear manifold to data$$",
    titlesize = 32,
    azimuth = pi/4,
    elevation = 0.3,
    # aspect = (1, 1, 1),
    limits = (-1.2, 1.2, -1.2, 1.2, 0, 1.1),

    xy_label_kw...,
    z_label_kw...,
)

l4 = scatter!(ax4, xpts, ypts, zpts;
    color = vec(xpts .+ ypts),
    colormap = [:red, :blue],
    markersize = 15,
)

surface!(ax4, xsrf, ysrf, zsnf;
    color = 0*vec(xsrf),
    colormap = [:green, :green],
    alpha = 0.5,
    transparent = true,
)

#===============================#

fig5 = Figure(
    size = (800, 800),
    backgroundcolor = :white,
    grid = :off,
)

ax5 = Axis3(
    fig5[1,1];
    title = L"We fit a nonlinear manifold to data$$",
    titlesize = 32,
    azimuth = pi/4,
    elevation = 0.3,
    # aspect = (1, 1, 1),
    limits = (-1.2, 1.2, -1.2, 1.2, 0, 1.1),

    xy_label_kw...,
    z_label_kw...,
)

l5 = scatter!(ax5, xpts, ypts, zpts;
    color = vec(xpts .+ ypts),
    colormap = [:red, :blue],
    markersize = 15,
)

surface!(ax5, xsrf, ysrf, zcrm;
    color = 0*vec(xsrf),
    colormap = [:black, :black],
    alpha = 0.3,
    transparent = true,
)

#===============================#
save(joinpath(@__DIR__, "snapshot_ful.png"), fig1)
save(joinpath(@__DIR__, "snapshot_red.png"), fig2)
save(joinpath(@__DIR__, "snapshot_pca.png"), fig3)
save(joinpath(@__DIR__, "snapshot_snf.png"), fig4)
save(joinpath(@__DIR__, "snapshot_crm.png"), fig5)
#===============================#
nothing
