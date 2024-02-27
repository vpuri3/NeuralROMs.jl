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

Nxpts, Nypts = 20, 10
Nxsrf, Nysrf = 200, 100

xpts, ypts = makegrid(Nxpts, Nypts)
xsrf, ysrf = makegrid(Nxsrf, Nysrf)

zpts = @. zfunc(xpts, ypts; ϵ = 1e-2, rng)
zpca = 0.5 .+ 0 * xsrf
zcrm = @. zfunc(xsrf, ysrf; ϵ = 2e-2)
zsnf = @. zfunc(xsrf, ysrf)
#===============================#

fig = Figure(
    size = (800, 800),
    backgroundcolor = :white,
    grid = :off,
)

ax1 = Axis3(fig[1,1];
    title = L"Simulation snapshots embedded in parametric space $$",
    titlesize = 32,

    # viewmode = :fit,
    azimuth = pi/4,
    elevation = 0.3,
    # aspect = (1, 1, 1),
    limits = (-1.2, 1.2, -1.2, 1.2, 0, 1.1),

    xlabelvisible = false,
    ylabelvisible = false,
    zlabelvisible = false,

    xticksvisible = false,
    yticksvisible = false,
    zticksvisible = false,

    xticklabelsvisible = false,
    yticklabelsvisible = false,
    zticklabelsvisible = false,
)

t = text!(ax1, 1, 1, -0.2; text = L"\mathbb{R}^N", fontsize = 40)

l = scatter!(ax1, xpts, ypts, zpts;
    # color = :black,
    color = vec(xpts .+ ypts),
    colormap = [:red, :blue],
    markersize = 15,
)

# surface!(ax1, xsrf, ysrf, zpca;
#     color = 0*vec(xsrf),
#     colormap = [:black, :black],
#     alpha = 0.5,
#     transparent = true,
# )

# surface!(ax1, xsrf, ysrf, zsnf;
#     color = 0*vec(xsrf),
#     colormap = [:green, :green],
#     alpha = 0.5,
#     transparent = true,
# )

surface!(ax1, xsrf, ysrf, zcrm;
    color = 0*vec(xsrf),
    colormap = [:black, :black],
    alpha = 0.3,
    transparent = true,
)

fig

