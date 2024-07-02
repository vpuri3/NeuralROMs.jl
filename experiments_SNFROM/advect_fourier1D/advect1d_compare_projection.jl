#
using NeuralROMs, LaTeXStrings
import CairoMakie
import CairoMakie.Makie

joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "compare.jl") |> include
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

prob = Advection1D(0.25f0)
datafile = joinpath(@__DIR__, "data_advect/", "data.jld2")
device = Lux.gpu_device()

makedata_kws = (; Ix = :, _Ib = [1,], Ib_ = [1,], _It = :, It_ = :)

function compare_advect1d_l()
    latents = [1, 2, 3, 4, 8, 12, 16, 20]#, 24]

    for latent in latents
        ll = lpad(latent, 2, "0")

        #==================#
        # train
        #==================#

        modeldir_PCA = joinpath(@__DIR__, "model_PCA$(ll)") # traditional
        modeldir_CAE = joinpath(@__DIR__, "model_CAE$(ll)") # Lee, Carlberg
        modeldir_SNW = joinpath(@__DIR__, "model_SNW$(ll)") # us (Weight decay)
        modeldir_SNL = joinpath(@__DIR__, "model_SNL$(ll)") # us (Lipschitz)

        # # train PCA
        # train_PCA(datafile, modeldir_PCA, latent; makedata_kws, rng, device)
        #
        # # train_CAE
        # train_params_CAE = (; E = 1400, w = 32, makedata_kws, act = elu)
        # train_CAE_compare(prob, latent, datafile, modeldir_CAE, train_params_CAE; rng, device)
        #
        # # train_SNL
        # train_params_SNL = (; E = 1400, wd = 64, α = 1f-4, γ = 0f-0, makedata_kws,)
        # train_SNF_compare(latent, datafile, modeldir_SNL, train_params_SNL; rng, device)
        #
        # # train_SNW
        # train_params_SNW = (; E = 1400, wd = 64, α = 0f-0, γ = 1f-2, makedata_kws,)
        # train_SNF_compare(latent, datafile, modeldir_SNW, train_params_SNW; rng, device)

        #==================#
        # postprocess
        #==================#

        modelfile_PCA = joinpath(modeldir_PCA, "model.jld2")
        modelfile_CAE = joinpath(modeldir_CAE, "model_07.jld2")
        modelfile_SNL = joinpath(modeldir_SNL, "model_08.jld2")
        modelfile_SNW = joinpath(modeldir_SNW, "model_08.jld2")

        postprocess_PCA(prob, datafile, modelfile_PCA; rng, device)
        postprocess_CAE(prob, datafile, modelfile_CAE; rng)
        postprocess_SNF(prob, datafile, modelfile_SNL; rng, device)
        postprocess_SNF(prob, datafile, modelfile_SNW; rng, device)
    end

    nothing
end

#======================================================#
function plot_compare_advect1d_l()
    latents = [1, 2, 3, 4, 8, 12, 16, 20]#, 24]

    case = 1

    e_PCA = []
    e_CAE = []
    e_SNL = []
    e_SNW = []

    for latent in latents

        @show latent

        ll = lpad(latent, 2, "0")

        modeldir_PCA = joinpath(@__DIR__, "model_PCA$(ll)") # traditional
        modeldir_CAE = joinpath(@__DIR__, "model_CAE$(ll)") # Lee, Carlberg
        modeldir_SNL = joinpath(@__DIR__, "model_SNL$(ll)") # us (Lipschitz)
        modeldir_SNW = joinpath(@__DIR__, "model_SNW$(ll)") # us (Weight decay)

        file_PCA = joinpath(modeldir_PCA, "results", "evolve$(case).jld2")
        file_CAE = joinpath(modeldir_CAE, "results", "evolve$(case).jld2")
        file_SNL = joinpath(modeldir_SNL, "results", "evolve$(case).jld2")
        file_SNW = joinpath(modeldir_SNW, "results", "evolve$(case).jld2")

        file_PCA = jldopen(file_PCA)
        file_CAE = jldopen(file_CAE)
        file_SNL = jldopen(file_SNL)
        file_SNW = jldopen(file_SNW)

        # Ulrnd
        Ud = file_PCA["Udata"]

        U_PCA = file_PCA["Ulrnd"]
        U_CAE = file_CAE["Ulrnd"]
        U_SNL = file_SNL["Ulrnd"]
        U_SNW = file_SNW["Ulrnd"]

        N = length(Ud)
        Nr = sum(abs2, Ud) / N |> sqrt

        er_PCA = sum(abs2, U_PCA - Ud) / N / Nr |> sqrt
        er_CAE = sum(abs2, U_CAE - Ud) / N / Nr |> sqrt
        er_SNL = sum(abs2, U_SNL - Ud) / N / Nr |> sqrt
        er_SNW = sum(abs2, U_SNW - Ud) / N / Nr |> sqrt

        push!(e_PCA, er_PCA)
        push!(e_CAE, er_CAE)
        push!(e_SNL, er_SNL)
        push!(e_SNW, er_SNW)
    end

    xlabel = L"N_{ROM}"
    ylabel = L"$e_\text{proj}$"
    xlabelsize = ylabelsize = 16

    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = (L"POD‐ROM$$", L"CAE‐ROM$$", L"SNFL‐ROM$$", L"SNFW‐ROM$$",)

    kwl = Tuple(
        (; color = colors[i], linestyle = styles[i], label = labels[i], linewidth = 2)
        for i in 1:4
    )

    kws = Tuple(
        (; color = colors[i],)
        for i in 1:4
    )

    fig = Makie.Figure(; size = (600, 400), backgroundcolor = :white, grid = :off)
    ax  = Makie.Axis( fig[1,1]; xlabel, ylabel, xlabelsize, ylabelsize,
        # xscale = log2,
        yscale = log10,
    )

    Makie.lines!(ax, latents, e_PCA; kwl[1]...)
    Makie.lines!(ax, latents, e_CAE; kwl[2]...)
    Makie.lines!(ax, latents, e_SNL; kwl[3]...)
    Makie.lines!(ax, latents, e_SNW; kwl[4]...)

    Makie.scatter!(ax, latents, e_PCA; kws[1]...)
    Makie.scatter!(ax, latents, e_CAE; kws[2]...)
    Makie.scatter!(ax, latents, e_SNL; kws[3]...)
    Makie.scatter!(ax, latents, e_SNW; kws[4]...)

    Makie.Legend(fig[0,1], ax; orientation = :horizontal, framevisible = false, unique = true)

    path1 = joinpath(pkgdir(NeuralROMs), "figs", "method", "exp_nrom.pdf")
    path2 = joinpath(@__DIR__, "compare_ll_case1.pdf")

    save(path1, fig)
    save(path2, fig)

    fig
end

#======================================================#
# compare_advect1d_l()
plot_compare_advect1d_l()
#======================================================#
nothing
