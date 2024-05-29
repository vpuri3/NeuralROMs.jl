#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2                                 # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using LaTeXStrings

CUDA.allowscalar(false)

# using FFTW
begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
    # FFTW.set_num_threads(nt)
end

include(joinpath(pkgdir(NeuralROMs), "experiments_SNFROM", "cases.jl"))

#======================================================#
function makedata_SNF(
    datafile::String;
    Ix = Colon(),  # subsample in space
    _Ib = Colon(), # train/test split in batches
    Ib_ = Colon(), # disregard. set to everything but _Ib
    _It = Colon(), # train/test split in time
    It_ = Colon(),
)
    # load data
    x, t, mu, u, md_data = loaddata(datafile)

    _Ib = isa(_Ib, Colon) ? (1:size(u, 3)) : _Ib
    Ib_ = setdiff(1:size(u, 3), _Ib)
    Ib_ = isempty(Ib_) ? _Ib : Ib_

    # normalize
    x, x̄, σx = normalize_x(x)
    u, ū, σu = normalize_u(u)
    t, t̄, σt = normalize_t(t)

    # subsample, test/train split
    _x = @view x[:, Ix]
    x_ = @view x[:, Ix]

    _t = @view t[_It]
    t_ = @view t[It_]

    _u = @view u[:, Ix, _Ib, _It]
    u_ = @view u[:, Ix, Ib_, It_]

    # get dimensions
    in_dim  = size(x, 1)
    out_dim = size(u, 1)
    prm_dim = 1

    if !isnothing(mu[1])
        prm_dim += length(mu[1])
    end

    _, Nx, _Nb, _Nt = size(_u)
    _, Nx, Nb_, Nt_ = size(u_)

    _Ns = _Nb * _Nt # num trajectories
    Ns_ = Nb_ * Nt_

    println("Using $Nx sample points per trajectory.")
    println("$_Ns / $Ns_ trajectories in train/test sets.")

    # make arrays

    # space
    _xyz = zeros(Float32, in_dim, Nx, _Ns)
    xyz_ = zeros(Float32, in_dim, Nx, Ns_)

    _xyz[:, :, :] .= _x # [in_dim, Nx]
    xyz_[:, :, :] .= x_

    # parameters
    _prm = zeros(Float32, prm_dim, Nx, _Nb, _Nt)
    prm_ = zeros(Float32, prm_dim, Nx, Nb_, Nt_)

    reshape(view(_prm, 1, :, :, :), Nx * _Nb, _Nt) .= reshape(_t, 1, _Nt)
    reshape(view(prm_, 1, :, :, :), Nx * Nb_, Nt_) .= reshape(t_, 1, Nt_)

    if !isnothing(mu[1])
        _mu = hcat(mu[_Ib]...)
        mu_ = hcat(mu[Ib_]...)

        _prm[2:prm_dim, :, :, :] .= reshape(_mu, prm_dim-1, 1, _Nb, 1)
        prm_[2:prm_dim, :, :, :] .= reshape(mu_, prm_dim-1, 1, Nb_, 1)
    end

    # solution
    _y = reshape(_u, out_dim, :)
    y_ = reshape(u_, out_dim, :)

    _x = (reshape(_xyz, in_dim, :), reshape(_prm, prm_dim, :))
    x_ = (reshape(xyz_, in_dim, :), reshape(prm_, prm_dim, :))

    readme = ""

    makedata_kws = (; Ix, _Ib, Ib_, _It, It_)

    metadata = (; ū, σu, x̄, σx, t̄, σt,
        Nx, _Ns, Ns_,
        makedata_kws, md_data, readme,
    )

    (_x, _y), (x_, y_), metadata
end

#===========================================================#

function train_SNF(
    datafile::String,
    modeldir::String,
    l::Int, # latent space size
    hh::Int, # num hidden layers
    hd::Int, # num hidden layers
    wh::Int, # hidden layer width
    wd::Int, # hidden layer width
    E::Int; # num epochs
    rng::Random.AbstractRNG = Random.default_rng(),
    warmup::Bool = true,
    _batchsize = nothing,
    batchsize_ = nothing,
    λ2::Real = 0f0,
    σ2inv::Real = 0f0,
    α::Real = 0f0,
    weight_decays::Union{Real, NTuple{M, <:Real}} = 0f0,
    makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :,),
    device = Lux.cpu_device(),
) where{M}

    _data, data_, metadata = makedata_SNF(datafile; makedata_kws...)
    dir = modeldir

    in_dim  = size(_data[1][1], 1)
    prm_dim = size(_data[1][2], 1)
    out_dim = size(_data[2], 1)

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    println("input size: $in_dim")
    println("param size: $prm_dim")
    println("output size: $out_dim")

    hyper = begin
        wi = prm_dim
        wo = l

        act = tanh
        in_layer = Dense(wi, wh, act)
        hd_layer = Dense(wh, wh, act)
        fn_layer = Dense(wh, wo; use_bias = false)

        Chain(in_layer, fill(hd_layer, hh)..., fn_layer)
    end

    decoder = begin
        init_wt_in = scaled_siren_init(1f1)
        init_wt_hd = scaled_siren_init(1f0)
        init_wt_fn = glorot_uniform

        init_bias = rand32 # zeros32
        use_bias_fn = false

        act = sin

        wi = l + in_dim
        wo = out_dim

        in_layer = Dense(wi, wd, act; init_weight = init_wt_in, init_bias)
        hd_layer = Dense(wd, wd, act; init_weight = init_wt_hd, init_bias)
        fn_layer = Dense(wd, wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

        Chain(in_layer, fill(hd_layer, hd)..., fn_layer)
    end

    #-------------------------------------------#
    # training hyper-params
    #-------------------------------------------#

    NN = FlatDecoder(hyper, decoder)

    _batchsize = isnothing(_batchsize) ? numobs(_data) ÷ 100 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data) ÷ 1   : batchsize_

    lossfun = NeuralROMs.regularize_flatdecoder(mse; α, λ2)

    idx = ps_W_indices(NN, :decoder; rng)
    weightdecay = IdxWeightDecay(0f0, idx)
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, warmup, weightdecay)

    #-------------------------------------------#

    train_args = (; l, hh, hd, wh, wd, E, _batchsize, λ2, σ2inv, α, weight_decays)
    metadata   = (; metadata..., train_args)

    display(NN)
    displaymetadata(metadata)

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_, weight_decays,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    displaymetadata(metadata)

    plot_training(ST...) |> display

    model, ST, metadata
end

#======================================================#
function evolve_SNF(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String,
    case::Integer; # batch
    rng::Random.AbstractRNG = Random.default_rng(),
    outdir::String = joinpath(dirname(modelfile), "results"),
    data_kws = (; Ix = :, It = :),
    Δt::Union{Real, Nothing} = nothing,
    timealg::NeuralROMs.AbstractTimeAlg = EulerForward(),
    adaptive::Bool = false,
    scheme::Union{Nothing, NeuralROMs.AbstractSolveScheme} = nothing,
    autodiff_xyz::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ_xyz::Union{Real, Nothing} = nothing,
    learn_ic::Bool = true,
    verbose::Bool = true,
    device = Lux.cpu_device(),
)
    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile)

    # load model
    (NN, p, st), md = loadmodel(modelfile)

    #==============#
    # subsample in space
    #==============#
    Udata = @view Udata[:, data_kws.Ix, :, data_kws.It]
    Xdata = @view Xdata[:, data_kws.Ix]
    Tdata = @view Tdata[data_kws.It]

    Ud = Udata[:, :, case, :]
    U0 = Ud[:, :, 1]

    data = (Xdata, U0, Tdata)
    data = copy.(data) # ensure no SubArrays

    #==============#
    # get hyper-decoer
    #==============#
    hyper, decoder = get_flatdecoder(NN, p, st)

    #==============#
    # get p0
    #==============#
    _data, _, _ = makedata_SNF(datafile; Ix = [1,], _Ib = [case])
    _prm = _data[1][2]
    _ps = hyper[1](_prm, hyper[2], hyper[3])[1]
    p0 = _ps[:, 1]

    #==============#
    # make model
    #==============#
    NN, p0, st = freeze_decoder(decoder, length(p0); rng, p0)
    model = NeuralModel(NN, st, md)

    #==============#
    # evolve
    #==============#
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()
    linesearch = LineSearch()
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    nlsmaxiters = 20

    Δt = isnothing(Δt) ? -(-(extrema(Tdata)...)) / 100.0f0 : Δt

    if isnothing(scheme)
        scheme  = GalerkinProjection(linsolve, 1f-3, 1f-6) # abstol_inf, abstol_mse
        # scheme = LeastSqPetrovGalerkin(nlssolve, nlsmaxiters, 1f-6, 1f-3, 1f-6)
    end

    @time _, ps, Up = evolve_model(
        prob, model, timealg, scheme, data, p0, Δt;
        adaptive, autodiff_xyz, ϵ_xyz, learn_ic, verbose, device,
    )

    #==============#
    # visualization
    #==============#

    mkpath(outdir)

    # field visualizations
    grid = get_prob_grid(prob)
    fieldplot(Xdata, Tdata, Ud, Up, grid, outdir, "evolve", case)

    # parameter plots
    plt = plot(; title = L"$\tilde{u}$ distribution, case " * "$(case)")
    plt = make_param_scatterplot(ps, Tdata; plt, label = "Dynamics solve", color = :blues, cbar = false)
    png(plt, joinpath(outdir, "evolve_p_scatter_case$(case)"))

    plt = plot(; title = L"$\tilde{u}$ evolution, case " * "$(case)")
    plot!(plt, Tdata, ps', w = 3.0, label = "Dynamics solve")
    png(plt, joinpath(outdir, "evolve_p_case$(case)"))

    # save files
    filename = joinpath(outdir, "evolve$(case).jld2")
    jldsave(filename; Xdata, Tdata, Udata = Ud, Upred = Up, Ppred = ps, Plrnd = _ps)

    # print error metrics
    @show sqrt(mse(Up, Ud) / mse(Ud, 0 * Ud))
    @show norm(Up - Ud, Inf) / sqrt(mse(Ud, 0 * Ud))

    Xdata, Tdata, Ud, Up, ps
end

#===========================================================#
function postprocess_SNF(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    makeplot::Bool = true,
    evolve_kw::NamedTuple = (;),
    outdir::String = joinpath(dirname(modelfile), "results"),
    device = Lux.cpu_device(),
)
    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile)

    # load model
    model, md = loadmodel(modelfile)

    #==============#
    # set up train/test split
    #==============#
    _Ib = isa(md.makedata_kws._Ib, Colon) ? (1:size(Udata, 3)) : md.makedata_kws._Ib
    _It = isa(md.makedata_kws._It, Colon) ? (1:size(Udata, 4)) : md.makedata_kws._It

    Ib_ = setdiff(1:size(Udata, 3), _Ib)
    Ib_ = isempty(Ib_) ? _Ib : Ib_
    It_ = 1:size(Udata, 4)

    displaymetadata(md)

    #==============#
    # train/test split
    #==============#
    _Udata = @view Udata[:, :, _Ib, _It] # un-normalized
    Udata_ = @view Udata[:, :, Ib_, It_]

    _Tdata = @view Tdata[_It]
    Tdata_ = @view Tdata[It_]

    #==============#
    # Get model
    #==============#
    hyper, decoder = get_flatdecoder(model...)
    model = NeuralModel(decoder[1], decoder[3], md)

    #==============#
    # evaluate model
    #==============#
    _data, data_, _ = makedata_SNF(datafile; md.makedata_kws...)

    in_dim  = size(Xdata, 1)
    out_dim, Nx, Nb, Nt = size(Udata)

    _code = eval_model(hyper, _data[1][2]; device)
    code_ = eval_model(hyper, data_[1][2]; device)

    _xc = vcat(_data[1][1], _code)
    xc_ = vcat(data_[1][1], code_)

    _upred = eval_model(decoder, _xc; device)
    upred_ = eval_model(decoder, xc_; device)

    _upred = reshape(_upred, out_dim, Nx, length(_Ib), length(_It))
    upred_ = reshape(upred_, out_dim, Nx, length(Ib_), length(It_))

    _Upred = unnormalizedata(_upred, md.ū, md.σu)
    Upred_ = unnormalizedata(upred_, md.ū, md.σu)

    @show mse(_Upred, _Udata) / mse(_Udata, 0 * _Udata)
    @show mse(Upred_, Udata_) / mse(Udata_, 0 * Udata_)

    #==============#
    # save codes
    #==============#
    _code = reshape(_code, size(_code, 1), Nx, length(_Ib), length(_It))
    code_ = reshape(code_, size(code_, 1), Nx, length(Ib_), length(It_))

    _ps = _code[:, 1, :, :] # [code_len, _Nb, _Nt]
    ps_ = code_[:, 1, :, :] # [code_len, Nb_, Nt_]

    isdir(outdir) && rm(outdir; recursive = true)
    mkpath(outdir)

    if makeplot
        grid = get_prob_grid(prob)

        # field plots
        for case in axes(_Ib, 1)
            Ud = _Udata[:, :, case, :]
            Up = _Upred[:, :, case, :]
            fieldplot(Xdata, _Tdata, Ud, Up, grid, outdir, "train", case)
        end

        # parameter plots
        linewidth = 2.0
        palette = :tab10
        colors = (:reds, :greens, :blues, cgrad(:viridis), cgrad(:inferno), cgrad(:thermal))

        plt = plot(; title = "Parameter scatter plot")

        for (i, case) in enumerate(_Ib)
            _p = _ps[:, i, :]
            color = colors[i]
            plt = make_param_scatterplot(_p, _Tdata; plt,
                label = "Case $(case)", color, cbar = false)

            # parameter evolution plot
            p2 = plot(;
                title =  "Learned parameter evolution, case $(case)",
                xlabel = L"Time ($s$)", ylabel = L"\tilde{u}(t)", legend = false
            )
            plot!(p2, _Tdata, _p'; linewidth, palette)
            png(p2, joinpath(outdir, "train_p_case$(case)"))
        end

        for (i, case) in enumerate(Ib_)
            if case ∉ _Ib
                p_ = ps_[:, i, :]
                color = colors[i + length(_Ib)]
                plt = make_param_scatterplot(p_, Tdata_; plt,
                    label = "Case $(case) (Testing)", color, cbar = false)

                # parameter evolution plot
                p2 = plot(;
                    title =  "Trained parameter evolution, case $(case)",
                    xlabel = L"Time ($s$)", ylabel = L"\tilde{u}(t)", legend = false
                )
                plot!(p2, Tdata_, p_'; linewidth, palette)
                png(p2, joinpath(outdir, "test_p_case$(case)"))
            end
        end

        png(plt, joinpath(outdir, "train_p_scatter"))

    end # makeplot

    #==============#
    # Evolve
    #==============#
    for case in union(_Ib, Ib_)
        evolve_SNF(prob, datafile, modelfile, case; rng, outdir, evolve_kw..., device)
    end

    #==============#
    # Compare evolution with training plots
    #==============#

    for (i, case) in  enumerate(_Ib)
        ev = jldopen(joinpath(outdir, "evolve$(case).jld2"))

        ps = ev["Ppred"]
        _p = _ps[:, i, :]

        plt = plot(; title =  L"$\tilde{u}$ distribution, case " * "$(case)")
        plt = make_param_scatterplot(_p, _Tdata; plt, label = "HyperNet prediction", color = :reds, cbar = false)
        plt = make_param_scatterplot(ps, Tdata; plt, label = "Dynamics solve", color = :blues, cbar = false)
        png(plt, joinpath(outdir, "compare_p_scatter_case$(case)"))

        plt = plot(; title = L"$\tilde{u}$ evolution, case " * "$(case)")
        plot!(plt, Tdata, ps'; w = 3.0, label = "Dynamics solve", palette = :tab10)
        plot!(plt, _Tdata, _p'; w = 4.0, label = "HyperNet prediction", style = :dash, palette = :tab10)
        png(plt, joinpath(outdir, "compare_p_case$(case)"))
    end

    for (i, case) in  enumerate(Ib_)
        ev = jldopen(joinpath(outdir, "evolve$(case).jld2"))

        ps = ev["Ppred"]
        p_ = ps_[:, i, :]

        plt = plot(; title = L"$\tilde{u}$ distribution, case " * "$(case)")
        plt = make_param_scatterplot(p_, Tdata_; plt, label = "HyperNet prediction", color = :reds, cbar = false)
        plt = make_param_scatterplot(ps, Tdata; plt, label = "Dynamics solve", color = :blues, cbar = false)
        png(plt, joinpath(outdir, "compare_p_scatter_case$(case)"))

        plt = plot(; title = L"$\tilde{u}$ evolution, case " * "$(case)")
        plot!(plt, Tdata, ps'; w = 3.0, label = "Dynamics solve", palette = :tab10)
        plot!(plt, Tdata_, p_'; w = 4.0, label = "HyperNet prediction", style = :dash, palette = :tab10)
        png(plt, joinpath(outdir, "compare_p_case$(case)"))
    end

    #==============#
    # Done
    #==============#
    if haskey(md, :readme)
        RM = joinpath(outdir, "README.md")
        RM = open(RM, "w")
        write(RM, md.readme)
        close(RM)
    end

    nothing
end
#===========================================================#
#
