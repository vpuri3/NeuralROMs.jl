#
using NeuralROMs
using LinearAlgebra, Plots, LaTeXStrings
using JLD2, HDF5

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
    device = Lux.cpu_device(),
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
    device = Lux.cpu_device(),
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
    device = Lux.cpu_device(),
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

function hyper_plots(
    datafile::String,
    modeldir::String,
    casenum::Integer,
)

    statsfile = joinpath(modeldir, "hyperstats.jld2")
    statsROM = jldopen(statsfile)["statsROM"]
    statsFOM = jldopen(statsfile)["statsFOM"]

    tims = (;)
    mems = (;) # GPU memory
    uprs = (;) # at final time-step

    xdata, _, _, udata, _ = loaddata(datafile)
    udata = udata[:, :, casenum, end]

    for case in keys(statsROM)
        evolvefile = joinpath(modeldir, "hyper_" * String(case), "evolve$(casenum).jld2")
        ev = jldopen(evolvefile)

        tims = (; tims..., case => getproperty(statsROM, case).time)
        mems = (; mems..., case => getproperty(statsROM, case).gpu_bytes)
        uprs = (; uprs..., case => ev["Upred"][:, :, end])

        close(ev)
    end

    tims = (; tims..., FOM = statsFOM.time)
    mems = (; mems..., FOM = statsFOM.gpu_bytes)

    times    = round.(values(tims), sigdigits = 4)
    speedups = round.(statsFOM.time ./ values(tims), sigdigits = 4)
    @show times
    @show speedups

    savefile = joinpath(modeldir, "hypercompiled.jld2")
    jldsave(savefile; xdata, udata, tims, mems, uprs)
end

#======================================================#

function makeplots_hyper(
    e2hyper::String,
    outdir::String,
    casename::String,
)

    mkpath(outdir)

    # Get fields
    grid = if occursin("exp2", casename)
        128, 128
    elseif occursin("exp4", casename)
        512, 512
    end

    file = jldopen(e2hyper)
    xdata = file["xdata"]
    udata = file["udata"]
    upreds = file["uprs"]
    times = file["tims"]
    mems  = file["mems"]
    close(file)

    xdiag = diag(reshape(xdata[1, :, :], grid))
    udata = if occursin("exp2", casename)
        reshape(udata, grid)
    elseif occursin("exp4", casename)
        reshape(udata[1, :], grid)
    end

    ups = collect(reshape(upred[1, :], grid) for upred in upreds)
    ups = cat(ups...; dims = 3)

    nr  = sqrt(sum(abs2, udata) / length(udata))
    eps = @. (ups - udata) / nr

    icases = keys(upreds)
    ncases = length(upreds)

    epreds = NamedTuple{icases}(eps[:, :, i] for i in 1:ncases)
    ediags = NamedTuple{icases}(diag(eps[:, :, i]) for i in 1:ncases)
    udiags = NamedTuple{icases}(diag(ups[:, :, i]) for i in 1:ncases)
    uddiag = diag(udata)

    k1 = (:N16384_dt1 , :N4096_dt1 , :N1024_dt1 , :N256_dt1 , :N64_dt1 )
    k2 = (:N16384_dt2 , :N4096_dt2 , :N1024_dt2 , :N256_dt2 , :N64_dt2 )
    k3 = (:N16384_dt5 , :N4096_dt5 , :N1024_dt5 , :N256_dt5 , :N64_dt5 )
    k4 = (:N16384_dt10, :N4096_dt10, :N1024_dt10, :N256_dt10, :N64_dt10)

    # create dataframe / table
    df = nothing

    tFOM = times.FOM
    mFOM = mems.FOM

    e1 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k1)
    e2 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k2)
    e3 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k3)
    e4 = Tuple(sqrt(sum(abs2, epreds[k]) / length(udata)) for k in k4)

    # @show round.(e1 .* 100, sigdigits = 4)
    # @show round.(e2 .* 100, sigdigits = 4)
    # @show round.(e3 .* 100, sigdigits = 4)
    # @show round.(e4 .* 100, sigdigits = 4)
    
    println()

    t1 = Tuple(times[k] for k in k1)
    t2 = Tuple(times[k] for k in k2)
    t3 = Tuple(times[k] for k in k3)
    t4 = Tuple(times[k] for k in k4)

    nn = 1024^3
    
    m1 = Tuple(mems[k] for k in k1) ./ nn
    m2 = Tuple(mems[k] for k in k2) ./ nn
    m3 = Tuple(mems[k] for k in k3) ./ nn
    m4 = Tuple(mems[k] for k in k4) ./ nn
    
    s1 = tFOM ./ t1
    s2 = tFOM ./ t2
    s3 = tFOM ./ t3
    s4 = tFOM ./ t4

    # @show tFOM
    # @show round.(s1, sigdigits = 4)
    # @show round.(s2, sigdigits = 4)
    # @show round.(s3, sigdigits = 4)
    # @show round.(s4, sigdigits = 4)
    #
    # println()
    #
    # @show mFOM / nn
    # @show round.(m1, sigdigits = 4)
    # @show round.(m2, sigdigits = 4)
    # @show round.(m3, sigdigits = 4)
    # @show round.(m4, sigdigits = 4)
    
    # println()
    #
    # return df

    ### Make plots

    figc = Figure(; size = (750, 750), backgroundcolor = :white, grid = :off)
    fige = Figure(; size = (750, 850), backgroundcolor = :white, grid = :off)
    figp = Figure(; size = (750, 850), backgroundcolor = :white, grid = :off)

    # FIGC
    nlevels = 11
    levels = if occursin("exp2", casename)
        10.0 .^ range(-4, 0, nlevels)
    elseif occursin("exp4", casename)
        10.0 .^ range(-5, 0, nlevels)
    end

    kw_axc = (; aspect = 1, xlabel = L"x", ylabel = L"y")
    kw_ctr = (; extendlow = :cyan, extendhigh = :magenta, colorscale = log10, levels)

    # FIGE, FIGP

    colors = (:orange, :green, :blue, :red, :brown,)
    styles = (:solid, :dash, :dashdot, :dashdotdot, :dot)
    labels = (L"$|X_\text{proj}|=16384$", L"$|X_\text{proj}|=4096$", L"$|X_\text{proj}|=1024$", L"$|X_\text{proj}|=256$", L"$|X_\text{proj}|=64$",)

    kw_axe = (;
        xlabel = L"y = x",
        ylabel = L"ε(t; \mathbf{μ})",
        yscale = log10,
        xlabelsize = 16,
        ylabelsize = 16,
    )

    kw_axp = (;
        xlabel = L"y = x",
        ylabel = L"u(x, t; \mathbf{μ})",
        xlabelsize = 16,
        ylabelsize = 16,
    )

    kw_fom = (; linewidth = 3, color = :black, linestyle = :solid, label = L"FOM$$")

    kw_lin = Tuple(
        (; linewidth = 3, color = colors[j], linestyle = styles[j], label = labels[j])
        for j in 1:5
    )

    axe1 = Axis(fige[1, 1]; kw_axe...)
    axe2 = Axis(fige[1, 2]; kw_axe...)
    axe3 = Axis(fige[3, 1]; kw_axe...)
    axe4 = Axis(fige[3, 2]; kw_axe...)

    axp1 = Axis(figp[1, 1]; kw_axp...)
    axp2 = Axis(figp[1, 2]; kw_axp...)
    axp3 = Axis(figp[3, 1]; kw_axp...)
    axp4 = Axis(figp[3, 2]; kw_axp...)

    for j in 1:5
        ep1 = epreds[k1[j]] .|> abs
        ep2 = epreds[k2[j]] .|> abs
        ep3 = epreds[k3[j]] .|> abs
        ep4 = epreds[k4[j]] .|> abs

        ed1 = ediags[k1[j]] .|> abs
        ed2 = ediags[k2[j]] .|> abs
        ed3 = ediags[k3[j]] .|> abs
        ed4 = ediags[k4[j]] .|> abs

        ud1 = udiags[k1[j]]
        ud2 = udiags[k2[j]]
        ud3 = udiags[k3[j]]
        ud4 = udiags[k4[j]]

        # FIGE
        lines!(axe1, xdiag, ed1; kw_lin[j]...)
        lines!(axe2, xdiag, ed2; kw_lin[j]...)
        lines!(axe3, xdiag, ed3; kw_lin[j]...)
        lines!(axe4, xdiag, ed4; kw_lin[j]...)

        # FIGP

        if j == 1
            # FOM
            lines!(axp1, xdiag, uddiag; kw_fom...)
            lines!(axp2, xdiag, uddiag; kw_fom...)
            lines!(axp3, xdiag, uddiag; kw_fom...)
            lines!(axp4, xdiag, uddiag; kw_fom...)
        end

        lines!(axp1, xdiag, ud1; kw_lin[j]...)
        lines!(axp2, xdiag, ud2; kw_lin[j]...)
        lines!(axp3, xdiag, ud3; kw_lin[j]...)
        lines!(axp4, xdiag, ud4; kw_lin[j]...)

        # FIGC
        ax1j = Axis(figc[1,j]; kw_axc...)
        ax2j = Axis(figc[2,j]; kw_axc...)
        ax3j = Axis(figc[3,j]; kw_axc...)
        ax4j = Axis(figc[4,j]; kw_axc...)

        cf1j = contourf!(ax1j, xdiag, xdiag, ep1; kw_ctr...)
        cf2j = contourf!(ax2j, xdiag, xdiag, ep2; kw_ctr...)
        cf3j = contourf!(ax3j, xdiag, xdiag, ep3; kw_ctr...)
        cf4j = contourf!(ax4j, xdiag, xdiag, ep4; kw_ctr...)

        tightlimits!(ax1j)
        tightlimits!(ax2j)
        tightlimits!(ax3j)
        tightlimits!(ax4j)

        hidedecorations!(ax1j; label = false)
        hidedecorations!(ax2j; label = false)
        hidedecorations!(ax3j; label = false)
        hidedecorations!(ax4j; label = false)

        if j == 5
            # ticks = [1f-5, 1f-4, 1f-3, 1f-2, 1f-1,]
            # Colorbar(figc[4, :], cf1j; ticks, vertical = false)
            Colorbar(figc[5, :], cf1j; vertical = false)
        end
    end

    # FIGE, FIGP
    linkaxes!(axe1, axe2, axe3, axe4)
    linkaxes!(axp1, axp2, axp3, axp4)

    fige[0,:] = Legend(fige, axe1, patchsize = (30, 10), orientation = :horizontal, framevisible = false)
    figp[0,:] = Legend(figp, axp1, patchsize = (30, 10), orientation = :horizontal, framevisible = false)

    Label(fige[2,1], L"(a) $Δt =  1Δt_0$", fontsize = 16)
    Label(fige[2,2], L"(b) $Δt =  2Δt_0$", fontsize = 16)
    Label(fige[4,1], L"(c) $Δt =  5Δt_0$", fontsize = 16)
    Label(fige[4,2], L"(d) $Δt = 10Δt_0$", fontsize = 16)

    Label(figp[2,1], L"(a) $Δt =  1Δt_0$", fontsize = 16)
    Label(figp[2,2], L"(b) $Δt =  2Δt_0$", fontsize = 16)
    Label(figp[4,1], L"(c) $Δt =  5Δt_0$", fontsize = 16)
    Label(figp[4,2], L"(d) $Δt = 10Δt_0$", fontsize = 16)

    colsize!(fige.layout, 1, Relative(0.50))
    colsize!(fige.layout, 2, Relative(0.50))
    
    colsize!(figp.layout, 1, Relative(0.50))
    colsize!(figp.layout, 2, Relative(0.50))

    # FIGC
    Label(figc[1:4, -1][1,1], L"Time‐step size $(Δt)$"; rotation = pi/2, fontsize = 16)
    Label(figc[-1, 1:4][1,1], L"Number of hyper‐reduction points $(|X_\text{proj}|)$"; fontsize = 16)

    Label(figc[0,1], L"$|X_\text{proj}| = 16384$", fontsize = 16)
    Label(figc[0,2], L"$|X_\text{proj}| = 4096$" , fontsize = 16)
    Label(figc[0,3], L"$|X_\text{proj}| = 1024$" , fontsize = 16)
    Label(figc[0,4], L"$|X_\text{proj}| = 256$"  , fontsize = 16)
    Label(figc[0,5], L"$|X_\text{proj}| = 64$"   , fontsize = 16)

    Label(figc[1,0], L"$Δt =  1Δt_0$"; fontsize = 16, rotation = pi/2)
    Label(figc[2,0], L"$Δt =  2Δt_0$"; fontsize = 16, rotation = pi/2)
    Label(figc[3,0], L"$Δt =  5Δt_0$"; fontsize = 16, rotation = pi/2)
    Label(figc[4,0], L"$Δt = 10Δt_0$"; fontsize = 16, rotation = pi/2)

    rowsize!(figc.layout, 1, Relative(0.22))
    rowsize!(figc.layout, 2, Relative(0.22))
    rowsize!(figc.layout, 3, Relative(0.22))
    rowsize!(figc.layout, 4, Relative(0.22))

    colsize!(figc.layout, 1, Relative(0.19))
    colsize!(figc.layout, 2, Relative(0.19))
    colsize!(figc.layout, 3, Relative(0.19))
    colsize!(figc.layout, 4, Relative(0.19))
    colsize!(figc.layout, 5, Relative(0.19))

    # save
    # save(joinpath(outdir, "hyper_$(casename)_c.pdf"), figc)
    # save(joinpath(outdir, "hyper_$(casename)_e.pdf"), fige)
    save(joinpath(outdir, "hyper_$(casename)_p.pdf"), figp)

    return df
end

#======================================================#
nothing
