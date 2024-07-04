#
using NeuralROMs
using LinearAlgebra, Plots, LaTeXStrings
using JLD2, HDF5

import CairoMakie
import CairoMakie: Makie

joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "PCA.jl")      |> include
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "convAE.jl")   |> include
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "convINR.jl")  |> include
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "smoothNF.jl") |> include
joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "cases.jl")    |> include

# joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "autodecode.jl")  |> include

#======================================================#
function get_makedata_kws(train_params)
    if haskey(train_params, :makedata_kws)
        train_params.makedata_kws
    else
        (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :)
    end
end

function get_batchsizes(train_params)
    bsz = (;)

    if haskey(train_params, :_batchsize)
        bsz = (; bsz..., _batchsize = train_params._batchsize,)
    end

    if haskey(train_params, :batchsize_)
        bsz = (; bsz..., batchsize_ = train_params.batchsize_,)
    end

    bsz
end

function train_CAE_compare(
    prob::NeuralROMs.AbstractPDEProblem,
    l::Integer, 
    datafile::String,
    modeldir::String,
    train_params = (;);
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.gpu_device(),
)
    E   = haskey(train_params, :E  ) ? train_params.E   : 1400
    w   = haskey(train_params, :w  ) ? train_params.w   : 32
    act = haskey(train_params, :act) ? train_params.act : tanh # relu, tanh, elu

    NN = cae_network(prob, l, w, act)

    ### size debugging
    # p, st = Lux.setup(rng, NN)
    # x = rand(Float32, 512, 512, 1, 5,)
    # y = NN(x, p, st)[1]
    # @show size(y)
    # @assert false

    # misc
    batchsizes = get_batchsizes(train_params)
    makedata_kws = get_makedata_kws(train_params)

    isdir(modeldir) && rm(modeldir, recursive = true)
    train_CAE(datafile, modeldir, NN, E; rng,
        makedata_kws, warmup = false, device, batchsizes...,
    )
end

function train_CINR_compare(
    prob::NeuralROMs.AbstractPDEProblem,
    l::Integer, 
    datafile::String,
    modeldir::String,
    train_params = (;);
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.gpu_device(),
)
    E   = haskey(train_params, :E  ) ? train_params.E   : 1400
    h   = haskey(train_params, :h  ) ? train_params.we  : 5
    we  = haskey(train_params, :we ) ? train_params.we  : 32
    wd  = haskey(train_params, :wd ) ? train_params.we  : 64
    act = haskey(train_params, :act) ? train_params.act : tanh # relu, tanh, elu

    NN = convINR_network(prob, l, h, we, wd, act)

    ### size debugging
    # p, st = Lux.setup(rng, NN)
    # x = rand(Float32, 512, 512, 1, 5,)
    # y = NN(x, p, st)[1]
    # @show size(y)
    # @assert false

    # misc
    batchsizes = get_batchsizes(train_params)
    makedata_kws = get_makedata_kws(train_params)

    isdir(modeldir) && rm(modeldir, recursive = true)
    train_CINR(datafile, modeldir, NN, E; rng,
        makedata_kws, warmup = true, device, batchsizes...,
    )
end

function train_SNF_compare(
    l::Integer, 
    datafile::String,
    modeldir::String,
    train_params = (;);
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.gpu_device(),
)
    E = haskey(train_params, :E) ? train_params.E : 1400

    # hyper-network
    hh = haskey(train_params, :hh) ? train_params.hh : 0
    wh = haskey(train_params, :wh) ? train_params.wh : 8
    λ2 = haskey(train_params, :λ2) ? train_params.λ2 : 1f-3

    # deocder
    hd = haskey(train_params, :hd) ? train_params.hd : 5
    wd = haskey(train_params, :wd) ? train_params.wd : 128

    # decoder smoothing
    α = haskey(train_params, :α) ? train_params.α : 0f-0 # Lipschitz reg
    γ = haskey(train_params, :γ) ? train_params.γ : 0f-0 # Weight decay

    if iszero(α) & iszero(γ)
        @warn "Got γ = 0, α = 0"
    end

    # batchsize
    batchsizes = get_batchsizes(train_params)
    makedata_kws = get_makedata_kws(train_params)

    isdir(modeldir) && rm(modeldir, recursive = true)
    train_SNF(datafile, modeldir, l, hh, hd, wh, wd, E;
        rng, makedata_kws, λ2, α, weight_decays = γ, device,
        batchsizes...,
    )
end

#======================================================#
# compare cases (accuracy)
#======================================================#

function compare_plots(
    modeldirs,
    labels,
    outdir::String,
    casename::AbstractString,
    case::Integer,
    grid;
    ifdt::Bool = false,
)

    p1 = plot(; xlabel = L"x", ylabel = L"u(x, t)", legend = :topleft, framestyle = :box)
    p2 = plot(; xlabel = L"x", ylabel = L"u(x, t)", legend = :topleft, framestyle = :box)
    p3 = plot(; xlabel = L"t", ylabel = L"ε^2(t)" , legend = :topleft, framestyle = :box, yaxis = :log)
    p4 = nothing

    suffix = ("PCA", "CAE", "SNL", "SNW", "CRM")
    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :solid, :solid, :solid, :solid,)

    h5dict = Dict()
    h5path = joinpath(outdir, "$(casename).h5")

    for (i, modeldir) in enumerate(modeldirs)
        ev = jldopen(joinpath(modeldir, "results", "evolve$(case).jld2"))

        Xd = ev["Xdata"]
        Td = ev["Tdata"]
        Ud = ev["Udata"]
        Up = ev["Upred"]
        Pp = ev["Ppred"]
        Ue = ev["Ulrnd"]
        Pe = isone(i) ? Pp : ev["Plrnd"]

        @show size(Pp), size(Pp)

        in_dim  = size(Xd, 1)
        out_dim = size(Ud, 1)
        Nx, Nt = size(Xd, 2), length(Td)

        Itplt = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int)
        i1, i2 = Itplt[2], Itplt[5]

        up = Up[1, :, :] # Nx, Nt
        ud = Ud[1, :, :]
        nr = sum(abs2, ud) / length(ud) |> sqrt
        er = (up - ud) / nr
        er = sum(abs2, er; dims = 1) / size(ud, 1) |> vec

        c = colors[i]
        s = styles[i]
        label = labels[i]

        plt_kw = (; c, s, label, w = 3)
        ctr_kw = (; cmap = :viridis, aspect_ratio = :equal, xlabel = L"x", ylabel = L"y", cbar = false)

        if in_dim == 1
            xd = vec(Xd)
            Nx, = grid

            if i == 1
                plot!(p1, xd, ud[:, i1]; w = 5, label = "FOM", c = :black)
                plot!(p2, xd, ud[:, i2]; w = 5, label = "FOM", c = :black)
            end

            plot!(p1, xd, up[:, i1]; plt_kw...)
            plot!(p2, xd, up[:, i2]; plt_kw...)

        elseif in_dim == 2
            x_re = reshape(Xd[1,:], grid)
            y_re = reshape(Xd[2,:], grid)
            xdiag = diag(x_re)

            ud_re = reshape(ud, grid..., Nt)

            Nx, Ny = grid
            @assert Nx == Ny

            if i == 1
                # contour plots
                p4 = plot(layout = (4, 1), size = (2000, 500))
                contourf!(p4[1,1], xdiag, xdiag, ud; ctr_kw...)

                # diagonal plots
                uddiag1 = diag(ud_re[:, :, i1])
                uddiag2 = diag(ud_re[:, :, i2])

                plot!(p1, xlabel = L"y=x", ylabel = L"u(y=x, t)")

                plot!(p1, xdiag, uddiag1, w = 5, label = "FOM", c = :black)
                plot!(p2, xdiag, uddiag2, w = 5, label = "FOM", c = :black)
            end

            up1 = diag(reshape(up[:, i1], grid))
            up2 = diag(reshape(up[:, i2], grid))

            plot!(p1, xdiag, up1; plt_kw...)
            plot!(p2, xdiag, up2; plt_kw...)
        end

        plot!(p3, Td, er; w = 3, label = labels[i], c, s)

        ylm = extrema((ylims(p1)..., ylims(p2)...,))
        plot!(p1, ylims = ylm)
        plot!(p2, ylims = ylm)

        plot!(p3, ytick = 10.0 .^ .-(0:9), ylims = (10^-9, 1))

        #save stuff to HDF5 files
        td = vec(Td)
        xd = reshape(Xd, in_dim , grid...)
        ud = reshape(Ud, out_dim, grid..., Nt)
        up = reshape(Up, out_dim, grid..., Nt)
        ue = reshape(Ue, out_dim, grid..., Nt)
        
        if i == 1
            h5dict = Dict(h5dict...,
                "xFOM" => xd, "tFOM" => td, "uFOM" => ud,
            )
        end

        h5dict = Dict(h5dict..., "u$(suffix[i])" => up, "v$(suffix[i])" => ue)

        # everything but PCA

        # save p
        if i != 1
            h5dict = Dict(h5dict..., "p$(suffix[i])" => Pp)
            h5dict = Dict(h5dict..., "q$(suffix[i])" => Pe)

            # large dt
            if ifdt
                e = jldopen(joinpath(modeldir, "dt", "evolve$(case).jld2"))
                pdt = e["Ppred"]
                udt = e["Upred"]
                Ntdt = size(udt)[3]
                udt = reshape(udt, (out_dim, grid..., Ntdt))

                if i == 2
                    tdtFOM = e["Tdata"] |> Array
                    udtFOM = reshape(e["Udata"], (out_dim, grid..., Ntdt))

                    h5dict = Dict(h5dict..., "tdtFOM" => tdtFOM)
                    h5dict = Dict(h5dict..., "udtFOM" => udtFOM)
                end

                h5dict = Dict(h5dict..., "pdt$(suffix[i])" => pdt)
                h5dict = Dict(h5dict..., "udt$(suffix[i])" => udt)
            end
        end
    end

    png(p1, joinpath(outdir, "compare_t0_case$(case)"))
    png(p2, joinpath(outdir, "compare_t1_case$(case)"))
    png(p3, joinpath(outdir, "compare_er_case$(case)"))

    file = h5open(h5path, "w")
    for (k, v) in h5dict
        write(file, k, v)
    end
    close(file)

    p1, p2, p3
end

#======================================================#
# hyper-reduction experiments
#======================================================#

function hyper_timings(
    prob::NeuralROMs.AbstractPDEProblem,
    datafile::String,
    modelfile::String,
    casename::String,
    casenum::Integer,
    fomfile::String,
)
    grid = get_prob_grid(prob)
    modeldir = dirname(modelfile)

    statsROM = (;)

    dt_mults = [1, 2, 5, 10]
    iskips = if occursin("exp2", casename)
        [1, 2, 4, 8, 16]
    elseif occursin("exp4", casename)
        [4, 8, 16, 32, 64]
    end

    for dt_mult in reverse(dt_mults)
        for iskip in reverse(iskips)

            # hyper-indices
            ids = zeros(Bool, grid...)
            @views ids[1:iskip:end, 1:iskip:end] .= true
            hyper_indices = findall(isone, vec(ids))
            hyper_reduction_path = joinpath(modeldir, "hyper.jld2")

            # time-step
            It = LinRange(1, 500, 500 ÷ dt_mult) .|> Base.Fix1(round, Int)
            data_kws = (; Ix = :, It)

            evolve_kw = (;
                Δt = 10f0,
                data_kws,
                hyper_reduction_path,
                hyper_indices,
                learn_ic = false,
                verbose = false,
                benchmark = true,
                adaptive = false
            )

            # directory
            N = length(hyper_indices)
            casename = "N$(N)_dt$(dt_mult)"
            outdir = joinpath(modeldir, "hyper_$(casename)")

            # run
            _, stats = evolve_SNF(prob, datafile, modelfile, casenum; rng, outdir, evolve_kw..., device)
            statsROM = (; statsROM..., Symbol(casename) => stats)
        end
    end

    # FOM stats
    statsFOM = include(fomfile)

    statsfile = joinpath(modeldir, "hyperstats.jld2")
    jldsave(statsfile; statsROM, statsFOM)

    statsROM, statsFOM, statsfile
end


function hyper_plots(
    datafile::String,
    modeldir::String,
    outdir::String,
    casename::String,
    casenum::Integer;
    makefigs::Bool = true,
)
    mkpath(outdir)

    # get data

    statsfile = joinpath(modeldir, "hyperstats.jld2")
    statsROM = jldopen(statsfile)["statsROM"]
    statsFOM = jldopen(statsfile)["statsFOM"]

    # get FOM data
    grid = if occursin("exp2", casename)
        128, 128
    elseif occursin("exp4", casename)
        512, 512
    end

    xdata, _, _, udata, _ = loaddata(datafile)
    udata = udata[:, :, casenum, end]

    udata = if occursin("exp2", casename)
        reshape(udata, grid)
    elseif occursin("exp4", casename)
        reshape(udata[1, :], grid)
    end

    # get stats, predictions

    times  = (;)
    mems   = (;) # GPU memory
    upreds = (;) # at final time-step

    for case in keys(statsROM)
        evolvefile = joinpath(modeldir, "hyper_" * String(case), "evolve$(casenum).jld2")
        ev = jldopen(evolvefile)

        times  = (; times... , case => getproperty(statsROM, case).time)
        mems   = (; mems...  , case => getproperty(statsROM, case).gpu_bytes)
        upreds = (; upreds..., case => ev["Upred"][:, :, end])

        close(ev)
    end

    times = (; times..., FOM = statsFOM.time)
    mems  = (; mems... , FOM = statsFOM.gpu_bytes)

    # compute errors

    ups = collect(reshape(upred[1, :], grid) for upred in upreds)
    ups = cat(ups...; dims = 3)

    nr  = sqrt(sum(abs2, udata) / length(udata))
    ers = @. (ups - udata) / nr

    icases = keys(upreds)
    ncases = length(upreds)

    epreds = NamedTuple{icases}(ers[:, :, i] for i in 1:ncases)
    ediags = NamedTuple{icases}(diag(ers[:, :, i]) for i in 1:ncases)
    udiags = NamedTuple{icases}(diag(ups[:, :, i]) for i in 1:ncases)
    xdiag  = diag(reshape(xdata[1, :, :], grid))
    uddiag = diag(udata)

    Ns = [16384, 4096, 1024, 256, 64]

    k1 = (:N16384_dt1 , :N4096_dt1 , :N1024_dt1 , :N256_dt1 , :N64_dt1 )
    k2 = (:N16384_dt2 , :N4096_dt2 , :N1024_dt2 , :N256_dt2 , :N64_dt2 )
    k3 = (:N16384_dt5 , :N4096_dt5 , :N1024_dt5 , :N256_dt5 , :N64_dt5 )
    k4 = (:N16384_dt10, :N4096_dt10, :N1024_dt10, :N256_dt10, :N64_dt10)

    # create dataframe / table

    tFOM = times.FOM
    mFOM = mems.FOM

    e1 = collect(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k1)
    e2 = collect(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k2)
    e3 = collect(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k3)
    e4 = collect(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k4)

    @show round.(e1 .* 100, sigdigits = 4)
    @show round.(e2 .* 100, sigdigits = 4)
    @show round.(e3 .* 100, sigdigits = 4)
    @show round.(e4 .* 100, sigdigits = 4)

    println()

    t1 = collect(times[k] for k in k1)
    t2 = collect(times[k] for k in k2)
    t3 = collect(times[k] for k in k3)
    t4 = collect(times[k] for k in k4)

    nn = 1024^3

    m1 = collect(mems[k] for k in k1) ./ nn
    m2 = collect(mems[k] for k in k2) ./ nn
    m3 = collect(mems[k] for k in k3) ./ nn
    m4 = collect(mems[k] for k in k4) ./ nn

    s1 = tFOM ./ t1
    s2 = tFOM ./ t2
    s3 = tFOM ./ t3
    s4 = tFOM ./ t4

    @show tFOM
    @show round.(t1, sigdigits = 4)
    @show round.(t2, sigdigits = 4)
    @show round.(t3, sigdigits = 4)
    @show round.(t4, sigdigits = 4)

    println()

    @show round.(s1, sigdigits = 4)
    @show round.(s2, sigdigits = 4)
    @show round.(s3, sigdigits = 4)
    @show round.(s4, sigdigits = 4)
    
    println()
    
    @show mFOM / nn
    @show round.(m1, sigdigits = 4)
    @show round.(m2, sigdigits = 4)
    @show round.(m3, sigdigits = 4)
    @show round.(m4, sigdigits = 4)

    ### Make plots

    makefigs || return

    # fontsize = 16
    #
    # # FIGE, FIGM
    # fige = Makie.Figure(; size = (900, 400), backgroundcolor = :white, grid = :off)
    #
    # # styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    # colors = (:orange, :green, :blue, :red, :brown,)
    # styles = (:solid, :solid, :solid, :solid, :solid)
    # labels = (L"Δt=Δt₀", L"Δt=2Δt₀", L"Δt=5Δt₀", L"Δt=10Δt₀",)
    #
    # xlabel = L"Number of hyper‐reduction points $(X_\text{proj})$"
    #
    # kw_e1 = (;
    #     xlabel,
    #     ylabel = L"Speedup$$",
    #     xscale = log2,
    #     yscale = log10,
    #     xlabelsize = fontsize,
    #     ylabelsize = fontsize,
    # )
    #
    # kw_e2 = (;
    #     xlabel,
    #     ylabel = L"ε(t=T; μ)",
    #     xscale = log2,
    #     yscale = log10,
    #     xlabelsize = fontsize,
    #     ylabelsize = fontsize,
    # )
    #
    # axe1 = Makie.Axis(fige[1,1]; kw_e1...)
    # axe2 = Makie.Axis(fige[1,2]; kw_e2...)
    #
    # kw_l = Tuple(
    #     (; linewidth = 2, color = colors[i], linestyle = styles[i], label = labels[i],) for i in 1:4
    # )
    #
    # Makie.scatterlines!(axe1, Ns, s1; kw_l[1]...)
    # Makie.scatterlines!(axe1, Ns, s2; kw_l[2]...)
    # Makie.scatterlines!(axe1, Ns, s3; kw_l[3]...)
    # Makie.scatterlines!(axe1, Ns, s4; kw_l[4]...)
    #
    # Makie.scatterlines!(axe2, Ns, e1; kw_l[1]...)
    # Makie.scatterlines!(axe2, Ns, e2; kw_l[2]...)
    # Makie.scatterlines!(axe2, Ns, e3; kw_l[3]...)
    # Makie.scatterlines!(axe2, Ns, e4; kw_l[4]...)
    #
    # save(joinpath(outdir, "hyper_$(casename)_s.pdf"), fige)
    
    # # FIGE, FIGP
    #
    # fige = Makie.Figure(; size = (600, 300), backgroundcolor = :white, grid = :off)
    # figp = Makie.Figure(; size = (600, 300), backgroundcolor = :white, grid = :off)
    #
    # kw_axe = (;
    #     xlabel = L"y = x",
    #     ylabel = L"ε(t; \mathbf{μ})",
    #     yscale = log10,
    #     xlabelsize = fontsize,
    #     ylabelsize = fontsize,
    # )
    #
    # kw_axp = (;
    #     xlabel = L"y = x",
    #     ylabel = L"u(x, t; \mathbf{μ})",
    #     xlabelsize = fontsize,
    #     ylabelsize = fontsize,
    # )
    #
    # kw_fom = (; linewidth = 3, color = :black, linestyle = :solid, label = L"FOM$$")
    #
    # kw_lin = Tuple(
    #     (; linewidth = 2, color = colors[j], linestyle = styles[j], label = labels[j])
    #     for j in 1:5
    # )
    #
    # axe1 = Makie.Axis(fige[1, 1]; kw_axe...)
    # axe2 = Makie.Axis(fige[1, 2]; kw_axe...)
    # axe3 = Makie.Axis(fige[1, 3]; kw_axe...)
    # axe4 = Makie.Axis(fige[1, 4]; kw_axe...)
    #
    # axp1 = Makie.Axis(figp[1, 1]; kw_axp...)
    # axp2 = Makie.Axis(figp[1, 2]; kw_axp...)
    # axp3 = Makie.Axis(figp[1, 3]; kw_axp...)
    # axp4 = Makie.Axis(figp[1, 4]; kw_axp...)
    #
    # for j in 1:5
    #     ed1 = ediags[k1[j]] .|> abs
    #     ed2 = ediags[k2[j]] .|> abs
    #     ed3 = ediags[k3[j]] .|> abs
    #     ed4 = ediags[k4[j]] .|> abs
    #
    #     ud1 = udiags[k1[j]]
    #     ud2 = udiags[k2[j]]
    #     ud3 = udiags[k3[j]]
    #     ud4 = udiags[k4[j]]
    #
    #     # FIGE
    #     Makie.lines!(axe1, xdiag, ed1; kw_lin[j]...)
    #     Makie.lines!(axe2, xdiag, ed2; kw_lin[j]...)
    #     Makie.lines!(axe3, xdiag, ed3; kw_lin[j]...)
    #     Makie.lines!(axe4, xdiag, ed4; kw_lin[j]...)
    #
    #     # FIGP
    #
    #     if j == 1
    #         # FOM
    #         Makie.lines!(axp1, xdiag, uddiag; kw_fom...)
    #         Makie.lines!(axp2, xdiag, uddiag; kw_fom...)
    #         Makie.lines!(axp3, xdiag, uddiag; kw_fom...)
    #         Makie.lines!(axp4, xdiag, uddiag; kw_fom...)
    #     end
    #
    #     Makie.lines!(axp1, xdiag, ud1; kw_lin[j]...)
    #     Makie.lines!(axp2, xdiag, ud2; kw_lin[j]...)
    #     Makie.lines!(axp3, xdiag, ud3; kw_lin[j]...)
    #     Makie.lines!(axp4, xdiag, ud4; kw_lin[j]...)
    # end
    #
    # # FIGE, FIGP
    # Makie.linkaxes!(axe1, axe2, axe3, axe4)
    # Makie.linkaxes!(axp1, axp2, axp3, axp4)
    #
    # fige[0,:] = Makie.Legend(fige, axe1, patchsize = (30, 5), orientation = :vertical, framevisible = false, nbanks = 3)
    # figp[0,:] = Makie.Legend(figp, axp1, patchsize = (30, 5), orientation = :vertical, framevisible = false, nbanks = 3)
    #
    # Makie.Label(fige[2,1], L"(a) $Δt =  1Δt_0$"; fontsize)
    # Makie.Label(fige[2,2], L"(b) $Δt =  2Δt_0$"; fontsize)
    # Makie.Label(fige[2,3], L"(c) $Δt =  5Δt_0$"; fontsize)
    # Makie.Label(fige[2,4], L"(d) $Δt = 10Δt_0$"; fontsize)
    #
    # Makie.Label(figp[2,1], L"(a) $Δt =  1Δt_0$"; fontsize)
    # Makie.Label(figp[2,2], L"(b) $Δt =  2Δt_0$"; fontsize)
    # Makie.Label(figp[2,3], L"(c) $Δt =  5Δt_0$"; fontsize)
    # Makie.Label(figp[2,4], L"(d) $Δt = 10Δt_0$"; fontsize)
    #
    # Makie.colsize!(fige.layout, 1, Makie.Relative(0.25))
    # Makie.colsize!(fige.layout, 2, Makie.Relative(0.25))
    # Makie.colsize!(fige.layout, 3, Makie.Relative(0.25))
    # Makie.colsize!(fige.layout, 4, Makie.Relative(0.25))
    #
    # Makie.colsize!(figp.layout, 1, Makie.Relative(0.25))
    # Makie.colsize!(figp.layout, 2, Makie.Relative(0.25))
    # Makie.colsize!(figp.layout, 3, Makie.Relative(0.25))
    # Makie.colsize!(figp.layout, 4, Makie.Relative(0.25))
    #
    # Makie.hideydecorations!(axe2; grid = false)
    # Makie.hideydecorations!(axe3; grid = false)
    # Makie.hideydecorations!(axe4; grid = false)
    #
    # Makie.hideydecorations!(axp2; grid = false)
    # Makie.hideydecorations!(axp3; grid = false)
    # Makie.hideydecorations!(axp4; grid = false)
    #
    # # save
    # save(joinpath(outdir, "hyper_$(casename)_e.pdf"), fige)
    # save(joinpath(outdir, "hyper_$(casename)_p.pdf"), figp)

    return nothing
end

#======================================================#
nothing
