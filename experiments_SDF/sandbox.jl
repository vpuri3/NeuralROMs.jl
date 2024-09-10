using LinearAlgebra, Plots

#
# Hourglass SDF
# 
#   ---------------------
#   |                   |
#   |    ___________    |
#   |    \         /    |
#   |     \       /     |
#   |      \     /      |
#   |      |     |      |
#   |      |     |      |
#   |      /     \      |
#   |     /       \     |  ^ z
#   |    /         \    |  |
#   ---------------------   --> x
# 
# `x ∈ [-1, 1]`
# `z ∈ [ 0, 1]`
# `t ∈ [ 0, 1]`
#

pi32 = Float32(pi)

function fields(x, z, t)
    r = @. 0.5f0 * (1.3f0 - sin(pi32 * z)) # radius
    s = abs.(x) .≤ r                       # full SDF
    M = @. z ≤ t                           # time mask
    s = s .* M

    d = rand(size(s)...)
    T = @. 1f0 + z - sin(t)

    s, d .* s, T .* s
end

function plot_sdf(Nx, Nz, Nt)
    xx = LinRange(-1f0, 1f0, Nx) |> Array
    zz = LinRange( 0f0, 1f0, Nz) |> Array
    tt = LinRange( 0f0, 1f0, Nt) |> Array

    x = zeros(Nx, Nz, Nt)
    z = zeros(Nx, Nz, Nt)
    t = zeros(Nx, Nz, Nt)

    x[:, :, :] .= reshape(xx, (Nx, 1, 1))
    z[:, :, :] .= reshape(zz, (1, Nz, 1))
    t[:, :, :] .= reshape(tt, (1, 1, Nt))

    s, d, T = fields(x, z, t)

    plt_kw = (;
        legend = false,
        xlims = (-1, 1), xlabel = "x",
        ylims = ( 0, 1), ylabel = "z",
        aspect_ratio = 1,
    )

    ii = LinRange(1, Nt, 4) .|> Base.Fix1(round, Int)

    pss = []
    pdd = []
    ptt = []

    for i in ii
        title = "Time $(tt[i])"
        ps = heatmap(xx, zz, s[:, :, i]'; plt_kw..., title, c = :grays,)
        pd = heatmap(xx, zz, d[:, :, i]'; plt_kw..., title)
        pt = heatmap(xx, zz, T[:, :, i]'; plt_kw..., title, colorbar = true)

        push!(pss, ps)
        push!(pdd, pd)
        push!(ptt, pt)
    end

    size = (1400, 800)
    ps = plot(pss...; size)#, title = "SDF"         )
    pd = plot(pdd...; size)#, title = "Displacement")
    pt = plot(ptt...; size)#, title = "Temperature" )

    imx, imz = Nx ÷ 2, Nz ÷ 3
    title_hist = "Temperature history"
    label_hist = "($(xx[imx]), $(zz[imz]))"
    ph = plot(tt, T[imx, imz, :]; w = 2, title = title_hist, label = label_hist, legend = :bottomright)

    ps, pd, pt, ph
end

Nx = Ny = 512
Nt = 51
ps, pd, pt, ph = plot_sdf(Nx, Ny, Nt)
display(pt)

nothing
#
