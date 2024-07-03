#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2                                 # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU

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
function makedata_CAE(
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

    _u = @view u[:, Ix, _Ib, _It]
    u_ = @view u[:, Ix, Ib_, It_]

    Nx = size(_x, 2)
    @assert size(_u, 2) == size(_x, 2)

    # get dimensinos
    in_dim  = size(x, 1)
    out_dim = size(u, 1)

    _Ns = size(_u, 3) * size(_u, 4)
    Ns_ = size(u_, 3) * size(u_, 4)

    println("Using $Nx sample points per trajectory.")
    println("$_Ns / $Ns_ trajectories in train/test sets.")

    grid = if in_dim == 1
        (Nx,)
    elseif in_dim == 2
        md_data.grid
    end

    # make arrays
    _u = permutedims(_u, (2, 1, 3, 4)) # [Nx, out_dim, Nbatch, Ntime]
    u_ = permutedims(u_, (2, 1, 3, 4))

    _u = reshape(_u, grid..., out_dim, _Ns)
    u_ = reshape(u_, grid..., out_dim, Ns_)

    readme = ""
    makedata_kws = (; Ix, _Ib, Ib_, _It, It_,)
    metadata = (; ū, σu, x̄, σx,
        Nx, _Ns, Ns_,
        makedata_kws, md_data, readme,
    )

    (_u, _u), (u_, u_), metadata
end

#======================================================#
function train_CAE(
    datafile::String,
    modeldir::String,
    NN::Lux.AbstractExplicitLayer,
    E::Int; # num epochs
    rng::Random.AbstractRNG = Random.default_rng(),
    warmup::Bool = false,
    _batchsize = nothing,
    batchsize_ = nothing,
    makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :,),
    cb_epoch = nothing,
    device = Lux.cpu_device(),
)

    _data, data_, metadata = makedata_CAE(datafile; makedata_kws...)
    dir = modeldir

    lossfun = function(NN, p, st, batch)
        x, ŷ = batch
        y, st = NN(x, p, st)
        loss = sum(abs2, ŷ - y) / length(ŷ)

        loss, st, (;)
    end

    _batchsize = isnothing(_batchsize) ? numobs(_data) ÷ 50 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data)      : batchsize_

    #--------------------------------------------#
    # optimizer
    #--------------------------------------------#
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, warmup)

    train_args = (; E, _batchsize,)
    metadata   = (; metadata..., train_args)

    display(NN)
    displaymetadata(metadata)

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_, 
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
        cb_epoch,
    )

    displaymetadata(metadata)

    plot_training!(ST...) |> display

    model, ST, metadata
end

#======================================================#
function postprocess_CAE(
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
    (NN, p, st), md = loadmodel(modelfile)

    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    #==============#
    # setup train/test split
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

    #==============#
    # get encoder / decoer
    #==============#
    encoder = NN.layers.encoder, p.encoder, st.encoder
    encoder = NeuralROMs.remake_ca_in_model(encoder...)

    decoder = NN.layers.decoder, p.decoder, st.decoder
    decoder = NeuralROMs.remake_ca_in_model(decoder...)

    grid = get_prob_grid(prob)

    #==============#
    # evaluate model
    #==============#
    _udata = normalizedata(_Udata, md.ū, md.σu)
    udata_ = normalizedata(Udata_, md.ū, md.σu)

    _udataperm = permutedims(_udata, (2, 1, 3, 4))
    udataperm_ = permutedims(udata_, (2, 1, 3, 4))

    _udataresh = reshape(_udataperm, grid..., out_dim, :)
    udataresh_ = reshape(udataperm_, grid..., out_dim, :)

    _code = encoder[1](_udataresh, encoder[2], encoder[3])[1]
    code_ = encoder[1](udataresh_, encoder[2], encoder[3])[1]

    _upredresh = decoder[1](_code, decoder[2], decoder[3])[1]
    upredresh_ = decoder[1](code_, decoder[2], decoder[3])[1]

    _upredperm = reshape(_upredresh, prod(grid), out_dim, length(_Ib), length(_It))
    upredperm_ = reshape(upredresh_, prod(grid), out_dim, length(Ib_), length(It_))

    _upred = permutedims(_upredperm, (2, 1, 3, 4))
    upred_ = permutedims(upredperm_, (2, 1, 3, 4))

    _Upred = unnormalizedata(_upred, md.ū, md.σu)
    Upred_ = unnormalizedata(upred_, md.ū, md.σu)

    @show mse(_Upred, _Udata) / mse(_Udata, 0 * _Udata)
    @show mse(Upred_, Udata_) / mse(Udata_, 0 * Udata_)

    _ps = reshape(_code, size(_code, 1), length(_Ib), length(_It)) # [code_len, _Nb, _Nt]
    ps_ = reshape(code_, size(code_, 1), length(Ib_), length(It_)) # [code_len, Nb_, Nt_]

    isdir(outdir) && rm(outdir; recursive = true)
    mkpath(outdir)

    if makeplot

        # field plots
        for case in axes(_Ib, 1)
            Ud = _Udata[:, :, case, :]
            Up = _Upred[:, :, case, :]
            fieldplot(Xdata, Tdata, Ud, Up, grid, outdir, "train", case)
        end

        # parameter plots
        linewidth = 2.0
        palette = :tab10
        colors = (:reds, :greens, :blues, cgrad(:thermal), cgrad(:acton), cgrad(:viridis), cgrad(:thermal), cgrad(:viridis))
        shapes = (:circle, :square, :star,)

        plt = plot(; title = "Parameter scatter plot")

        for (i, case) in enumerate(_Ib)
            _p = _ps[:, i, :]
            color = colors[i]
            plt = make_param_scatterplot(_p, Tdata; plt,
                label = "Case $(case) (Training)", color, cbar = false)

            # parameter evolution plot
            p2 = plot(;
                title =  "Learned parameter evolution, case $(case)",
                xlabel = L"Time ($s$)", ylabel = L"\tilde{u}(t)", legend = false
            )
            plot!(p2, Tdata, _p'; linewidth, palette)
            png(p2, joinpath(outdir, "train_p_case$(case)"))
        end

        for (i, case) in enumerate(Ib_)
            if case ∉ _Ib
                p_ = ps_[:, i, :]
                color = colors[i + length(_Ib)]
                plt = make_param_scatterplot(p_, Tdata; plt,
                    label = "Case $(case) (Testing)", color, cbar = false)

                # parameter evolution plot
                p2 = plot(;
                    title =  "Trained parameter evolution, case $(case)",
                    xlabel = L"Time ($s$)", ylabel = L"\tilde{u}(t)", legend = false
                )
                plot!(p2, Tdata, p_'; linewidth, palette)
                png(p2, joinpath(outdir, "test_p_case$(case)"))
            end
        end

        png(plt, joinpath(outdir, "train_p_scatter"))

    end # makeplot

    #==============#
    # Evolve
    #==============#
    for case in union(_Ib, Ib_)
        evolve_CAE(prob, datafile, modelfile, case; rng, outdir, evolve_kw..., device)
    end

    #==============#
    # Compare evolution with training plots
    #==============#

    # for (i, case) in  enumerate(_Ib)
    #     ev = jldopen(joinpath(outdir, "evolve$(case).jld2"))
    #
    #     ps = ev["Ppred"]
    #     Ue = ev["Upred"]
    #
    #     _p = _ps[:, i, :]
    #     Uh = _Upred[:, :, i, :] # encoder/decoder prediction
    #     Ud = _Udata[:, :, i, :]
    #
    #     fieldplot(Xdata, Tdata, Uh, Ue, grid, outdir, "compare", case)
    #
    #     # Compare u
    #     println("#=======================#")
    #     println("Dynamics Solve")
    #     @show norm(Ue - Ud, 2) / length(Ud)
    #     @show norm(Ue - Ud, Inf)
    #
    #     println("#=======================#")
    #     println("Encoder Prediction")
    #     @show norm(Uh - Ud, 2) / length(Ud)
    #     @show norm(Uh - Ud, Inf)
    #
    #     ###
    #     # Compare ũ
    #     ###
    #
    #     println("#=======================#")
    #     println("Dynamics Solve vs Encoder Prediction")
    #     @show norm(ps - _p, 2) / length(ps)
    #     @show norm(ps - _p, Inf)
    #
    #     plt = plot(; title = L"$\tilde{u}$ distribution, case " * "$(case)")
    #     plt = make_param_scatterplot(_p, Tdata; plt, label = "Encoder prediction", color = :reds, cbar = false)
    #     plt = make_param_scatterplot(ps, Tdata; plt, label = "Dynamics solve", color = :blues, cbar = false)
    #     png(plt, joinpath(outdir, "compare_p_scatter_case$(case)"))
    #
    #     plt = plot(; title = L"$\tilde{u}$ evolution, case " * "$(case)")
    #     plot!(plt, Tdata, ps'; w = 3.0, label = "Dynamics solve", palette = :tab10)
    #     plot!(plt, Tdata, _p'; w = 4.0, label = "Encoder prediction", style = :dash, palette = :tab10)
    #     png(plt, joinpath(outdir, "compare_p_case$(case)"))
    # end
    #
    # for (i, case) in  enumerate(Ib_)
    #     ev = jldopen(joinpath(outdir, "evolve$(case).jld2"))
    #
    #     ps = ev["Ppred"]
    #     Ue = ev["Upred"]
    #
    #     p_ = ps_[:, i, :]
    #     Uh = Upred_[:, :, i, :] # encoder/decoder prediction
    #     Ud = Udata_[:, :, i, :]
    #
    #     fieldplot(Xdata, Tdata, Uh, Ue, grid, outdir, "compare", case)
    #
    #     # Compare u
    #     println("#=======================#")
    #     println("Dynamics Solve")
    #     @show norm(Ue - Ud, 2) / length(Ud)
    #     @show norm(Ue - Ud, Inf)
    #
    #     println("#=======================#")
    #     println("Encoder Prediction")
    #     @show norm(Uh - Ud, 2) / length(Ud)
    #     @show norm(Uh - Ud, Inf)
    #
    #     ###
    #     # Compare ũ
    #     ###
    #
    #     println("#=======================#")
    #     println("Dynamics Solve vs Encoder Prediction")
    #     @show norm(ps - p_, 2) / length(ps)
    #     @show norm(ps - p_, Inf)
    #
    #     plt = plot(; title = L"$\tilde{u}$ distribution, case " * "$(case)")
    #     plt = make_param_scatterplot(p_, Tdata; plt, label = "Encoder prediction", color = :reds, cbar = false)
    #     plt = make_param_scatterplot(ps, Tdata; plt, label = "Dynamics solve", color = :blues, cbar = false)
    #     png(plt, joinpath(outdir, "compare_p_scatter_case$(case)"))
    #
    #     plt = plot(; title =  L"$\tilde{u}$ evolution, case " * "$(case)")
    #     plot!(plt, Tdata, ps'; w = 3.0, label = "Dynamics solve", palette = :tab10)
    #     plot!(plt, Tdata, p_'; w = 4.0, label = "Encoder prediction", style = :dash, palette = :tab10)
    #     png(plt, joinpath(outdir, "compare_p_case$(case)"))
    # end

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

#======================================================#
function evolve_CAE(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String,
    case::Integer;
    rng::Random.AbstractRNG = Random.default_rng(),
    outdir::String = joinpath(dirname(modelfile), "results"),
    data_kws = (; Ix = :, It = :),
    Δt::Union{Real, Nothing} = nothing,
    timealg::NeuralROMs.AbstractTimeAlg = EulerForward(),
    adaptive::Bool = false,
    scheme::Union{Nothing, NeuralROMs.AbstractSolveScheme} = nothing,
    autodiff_xyz::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ_xyz::Union{Real, Nothing} = 1f-2,
    learn_ic::Bool = true,
    verbose::Bool = true,
    device = Lux.cpu_device(),
)
    mkpath(outdir)

    # load data
    Xdata, Tdata, mu, Udata, md_data = loaddata(datafile)

    # load model
    (NN, p, st), md = loadmodel(modelfile)

    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    #==============#
    # subsample in space
    #==============#
    Udata = @view Udata[:, data_kws.Ix, :, data_kws.It]
    Xdata = @view Xdata[:, data_kws.Ix]
    Tdata = @view Tdata[data_kws.It]

    Nx = size(Xdata, 2)
    Nt = length(Tdata)

    Ud = Udata[:, :, case, :]
    U0 = Ud[:, :, 1]

    data = (Xdata, U0, Tdata)
    data = copy.(data) # ensure no SubArrays

    #==============#
    # get encoder / decoer
    #==============#
    encoder = NN.layers.encoder, p.encoder, st.encoder
    encoder = NeuralROMs.remake_ca_in_model(encoder...)

    decoder = NN.layers.decoder, p.decoder, st.decoder
    decoder = NeuralROMs.remake_ca_in_model(decoder...)

    #==============#
    # get p0
    #==============#
    grid = get_prob_grid(prob)

    Ud_norm = normalizedata(Ud, md.ū, md.σu)
    Ud_perm = permutedims(Ud_norm, (2, 1, 3))
    Ud_resh = reshape(Ud_perm, grid..., out_dim, Nt) # [Nx, Ny, out_dim, Nt]

    pl = encoder[1](Ud_resh, encoder[2], encoder[3])[1]
    p0 = pl[:, 1]

    #==============#
    # make model
    #==============#
    model = CAEModel(decoder..., Xdata, grid, md)

    #==============#
    # evolve
    #==============#
    linsolve = QRFactorization()
    autodiff = AutoForwardDiff()
    linesearch = LineSearch()
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    nlsmaxiters = 10

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
    # analysis
    #==============#

    Ul = eval_model(model, Xdata, pl, getaxes(p0); device)

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
    filename = joinpath(outdir, "evolve$case.jld2")
    jldsave(filename; Xdata, Tdata, Udata = Ud, Upred = Up, Ppred = ps, Plrnd = pl, Ulrnd = Ul)

    Xdata, Tdata, Ud, Up, ps
end
#======================================================#
