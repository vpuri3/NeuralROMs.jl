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
        # Ue = ev["Ulrnd"]
        Pe = isone(i) ? Pp : ev["Plrnd"]

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
        
        if i == 1
            h5dict = Dict(h5dict...,
                "xFOM" => xd, "tFOM" => td, "uFOM" => ud,
            )
        end
        h5dict = Dict(h5dict..., "u$(suffix[i])" => up)

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
nothing
