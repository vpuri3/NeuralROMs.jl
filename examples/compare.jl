#
using GeometryLearning
using Plots, LaTeXStrings
using NPZ

joinpath(pkgdir(GeometryLearning), "examples", "PCA.jl")      |> include
joinpath(pkgdir(GeometryLearning), "examples", "convAE.jl")   |> include
joinpath(pkgdir(GeometryLearning), "examples", "smoothNF.jl") |> include
joinpath(pkgdir(GeometryLearning), "examples", "problems.jl") |> include

## comparison untenable with C-ROM due to large num-epochs
# joinpath(pkgdir(GeometryLearning), "examples", "convINR.jl")  |> include
# joinpath(pkgdir(GeometryLearning), "examples", "autodecode.jl")  |> include

#======================================================#
function get_makedata_kws(train_params)
    if haskey(train_params, :makedata_kws)
        train_params.makedata_kws
    else
        (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :)
    end
end

function train_CAE_compare(
    prob::GeometryLearning.AbstractPDEProblem,
    l::Integer, 
    datafile::String,
    modeldir::String,
    train_params = (;);
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
)
    E   = haskey(train_params, :E  ) ? train_params.E : 1400
    w   = haskey(train_params, :w  ) ? train_params.w : 32
    act = haskey(train_params, :act) ? train_params.E : tanh # relu, tanh
    makedata_kws = get_makedata_kws(train_params)
    bsz = haskey(train_params, :batchsize_) ? (; batchsize_ = train_params.batchsize_,) : (;)

    NN = cae_network(prob, l, w, act)

    ### size debugging
    # p, st = Lux.setup(rng, NN)
    # x = rand(Float32, 1024, 1, 5,)
    # y = NN(x, p, st)[1]
    # @show size(y)
    # @assert false

    isdir(modeldir) && rm(modeldir, recursive = true)
    train_CAE(datafile, modeldir, NN, E; rng,
        makedata_kws, warmup = false, device, bsz...,
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
    bsz = haskey(train_params, :batchsize_) ? (; batchsize_ = train_params.batchsize_,) : (;)

    # makedata_kws
    makedata_kws = get_makedata_kws(train_params)

    isdir(modeldir) && rm(modeldir, recursive = true)
    train_SNF(datafile, modeldir, l, hh, hd, wh, wd, E;
        rng, makedata_kws, λ2, α, weight_decays = γ, device,
        bsz...,
    )
end

#======================================================#
# main
#======================================================#

function compare_plots(
    modeldirs,
    labels,
    outdir::String,
    casename::String,
    case::Integer,
    grid,
)

    p1 = plot(; xlabel = L"x", ylabel = L"u(x, t)", title = "$(casename) at time T/4") # t = T/4
    p2 = plot(; xlabel = L"x", ylabel = L"u(x, t)", title = "$(casename) at time T") # t = T
    p3 = plot(; xlabel = L"t", ylabel = L"ε(t)"   , title = "$(casename) error vs time") # Error

    plot!(p3, yaxis = :log)

    colors = (:orange, :green, :blue, :red)

    for (i, modeldir) in enumerate(modeldirs)
        ev = jldopen(joinpath(modeldir, "results", "evolve$(case).jld2"))

        Xd = ev["Xdata"]
        Td = ev["Tdata"]
        Ud = ev["Udata"]
        Up = ev["Upred"]

        in_dim  = size(Xd, 1)
        out_dim = size(Ud, 1)
        Nx, Nt = size(Xd, 2), length(Td)

        Itplt = LinRange(1, Nt, 4) .|> Base.Fix1(round, Int)
        i1, i2 = Itplt[2], Itplt[4]

        up = Up[1, :, :] # Nx, Nt
        ud = Ud[1, :, :]
        nr = sum(abs2, ud) / length(ud) |> sqrt
        er = (up - ud) / nr
        er = sum(abs2, er; dims = 1) / size(ud, 1) |> vec

        palette = :tab10

        if in_dim == 1
            xd = vec(Xd)

            if i == 1
                plot!(p1, xd, ud[:, i1]; w = 5, palette, label = "FOM", c = :black)
                plot!(p2, xd, ud[:, i2]; w = 5, palette, label = "FOM", c = :black)
            end

            plot!(p1, xd, up[:, i1]; w = 3, palette, label = labels[i], c = colors[i], s = :dash)
            plot!(p2, xd, up[:, i2]; w = 3, palette, label = labels[i], c = colors[i], s = :dash)

        elseif in_dim == 2
            grid
        end

        plot!(p3, Td, er; w = 3, palette, label = labels[i], c = colors[i])
    end

    png(p1, joinpath(outdir, "compare_t0_case$(case)"))
    png(p2, joinpath(outdir, "compare_t1_case$(case)"))
    png(p3, joinpath(outdir, "compare_er_case$(case)"))

    p1, p2, p3
end

#======================================================#
nothing
