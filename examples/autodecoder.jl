#
using GeometryLearning
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

#======================================================#
function makedata_autodecoder(datafile::String;
    Ix = Colon(), # subsample in space
    _Ib = Colon(), # train/test split in batches
    Ib_ = Colon(),
    _It = Colon(), # train/test split in time
    It_ = Colon(),
)

    #==============#
    # load data
    #==============#
    data = jldopen(datafile)
    x = data["x"]
    u = data["u"] # [Nx, Nb, Nt]
    md_data = data["metadata"]
    close(data)

    @assert x isa AbstractVecOrMat
    x = x isa AbstractVector ? reshape(x, 1, :) : x # (Dim, Npoints)

    #==============#
    # normalize
    #==============#

    # single field prediction
    ū  = sum(u) / length(u)
    σu = sum(abs2, u .- ū) / length(u) |> sqrt
    u  = normalizedata(u, ū, σu)

    # noramlize x, y separately
    x̄  = sum(x, dims = 2) / size(x, 2) |> vec
    σx = sum(abs2, x .- x̄, dims = 2) / size(x, 2) .|> sqrt |> vec
    x  = normalizedata(x, x̄, σx)

    #==============#
    # subsample, test/train split
    #==============#
    x  = @view x[:, Ix]
    _u = @view u[Ix, _Ib, _It]
    u_ = @view u[Ix, Ib_, It_]

    Nx = size(x, 2)

    _u = reshape(_u, Nx, :)
    u_ = reshape(u_, Nx, :)

    _Ns = size(_u, 2) # number of codes i.e. # trajectories
    Ns_ = size(u_, 2)

    # indices
    _idx = zeros(Int32, Nx, _Ns)
    idx_ = zeros(Int32, Nx, Ns_)

    _idx[:, :] .= 1:_Ns |> adjoint
    idx_[:, :] .= 1:Ns_ |> adjoint

    # space
    _xyz = zeros(Float32, Nx, _Ns)
    xyz_ = zeros(Float32, Nx, Ns_)

    _xyz[:, :] .= x'
    xyz_[:, :] .= x'

    # solution
    _y = reshape(_u, 1, :)
    y_ = reshape(u_, 1, :)

    _x = (reshape(_xyz, 1, :), reshape(_idx, 1, :))
    x_ = (reshape(xyz_, 1, :), reshape(idx_, 1, :))

    readme = "Train/test on the same trajectory."

    makedata_kws = (; Ix, _Ib, Ib_, _It, It_)

    metadata = (; ū, σu, x̄, σx,
        Nx, _Ns, Ns_,
        makedata_kws, md_data, readme,
    )

    (_x, _y), (x_, y_), metadata
end

#===========================================================#

function train_autodecoder(
    datafile::String,
    modeldir::String,
    l::Int, # latent space size
    h::Int, # num hidden layers
    w::Int, # hidden layer width
    E::Int; # num epochs
    rng::Random.AbstractRNG = Random.default_rng(),
    _batchsize = nothing,
    batchsize_ = nothing,
    λ::Real = 0f0,
    makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :,),
    cb_epoch = nothing,
    device = Lux.cpu_device(),
)
    _data, _, metadata = makedata_autodecoder(datafile; makedata_kws...)
    dir = modeldir

    Nx = isa(makedata_kws.Ix, Colon) ? metadata.Nx : length(makedata_kws.Ix)

    #--------------------------------------------#
    # training hyper-params
    #--------------------------------------------#
    lrs = (1f-2, 1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    opts = Optimisers.Adam.(lrs)
    nepochs = (20, round.(Int, E / 7 * ones(7))...,)
    schedules = Step.(lrs, 1f0, Inf32)

    ################
    # nepochs = (50, E,)
    # schedules = (
    #     Step(1f-2, 1f0, Inf32),                        # constant warmup
    #     # Triangle(λ0 = 1f-5, λ1 = 1f-3, period = 20), # gradual warmup
    #     # Exp(1f-3, 0.996f0),
    #     SinExp(λ0 = 1f-5, λ1 = 1f-3, period = 50, γ = 0.995f0),
    #     # CosAnneal(λ0 = 1f-5, λ1 = 1f-3, period = 50),
    # )
    # opts = fill(Optimisers.Adam(), length(nepochs)) |> Tuple
    ################

    #--------------------------------------------#
    # architecture hyper-params
    #--------------------------------------------#
    act = sin

    init_wt_in = scaled_siren_init(3f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    # lossfun = mse
    lossfun = iszero(λ) ? mse : l2reg(mse, λ; property = :decoder)

    _batchsize = isnothing(_batchsize) ? Nx * 10       : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data) : batchsize_

    #----------------------#----------------------#

    in_layer = Dense(l+1, w, act; init_weight = init_wt_in, init_bias)
    hd_layer = Dense(w  , w, act; init_weight = init_wt_hd, init_bias)
    fn_layer = Dense(w  , 1; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

    decoder = Chain(
        in_layer,
        fill(hd_layer, h)...,
        fn_layer,
    )

    NN = AutoDecoder(decoder, metadata._Ns, l)

    # # second order optimizer
    # # if it can fit on 11 gb vRAM on 2080Ti
    # if Lux.parameterlength(decoder) < 35_000
    #     opts = (opts..., LBFGS(),)
    #     nepochs = (nepochs..., round(Int, E / 10))
    #     schedules = (schedules..., Step(1f0, 1f0, Inf32))
    # end

    train_args = (; l, h, w, E, _batchsize, batchsize_, λ)
    metadata = (; metadata..., train_args)
    #----------------------#----------------------#

    @show train_args

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_,
        opts, nepochs, schedules,
        device, dir, metadata, lossfun,
        cb_epoch,
    )

    plot_training(ST...) |> display

    model
end

#======================================================#

function infer_autodecoder(
    model::AbstractNeuralModel,
    data::Tuple, # (X, U, T)
    p0::AbstractVector;
    device = Lux.cpu_device(),
    learn_init::Bool = false,
    verbose::Bool = false,
)
    # make data
    xdata, udata, _ = data
    Nx, Nt = size(udata)

    model = model |> device # problem here.
    xdata = xdata |> device
    udata = udata |> device
    p = p0 |> device

    # optimizer
    autodiff = AutoForwardDiff()
    linsolve = QRFactorization()
    linesearch = LineSearch()

    # linesearchalg = Static()
    # linesearchalg = BackTracking()
    # linesearchalg = HagerZhang()
    # linesearch = LineSearch(; method = linesearchalg, autodiff = AutoZygote())
    # linesearch = LineSearch(; method = linesearchalg, autodiff = AutoFiniteDiff())

    # nlssolve = BFGS()
    # nlssolve = LevenbergMarquardt(; autodiff, linsolve)
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)

    codes  = ()
    upreds = ()
    MSEs   = []

    for iter in 1:Nt

        xbatch = reshape(xdata, 1, Nx)
        ubatch = reshape(udata[:, iter], 1, Nx)
        batch  = xbatch, ubatch

        if learn_init & (iter == 1)
            p, _ = nonlinleastsq(model, p, batch, Optimisers.Adam(1f-1); verbose)
            p, _ = nonlinleastsq(model, p, batch, Optimisers.Adam(1f-2); verbose)
            p, _ = nonlinleastsq(model, p, batch, Optimisers.Adam(1f-3); verbose)
        end

        p, _ = nonlinleastsq(model, p, batch, nlssolve; maxiters = 20, verbose)

        # eval
        upred = model(xbatch, p)
        l = round(mse(upred, ubatch); sigdigits = 8)

        codes  = push(codes, p)
        upreds = push(upreds, upred)
        push!(MSEs, l)

        if verbose
            println("Iter $iter, MSE: $l")
            iter += 1
        end
    end

    code  = mapreduce(getdata, hcat, codes ) |> Lux.cpu_device()
    upred = mapreduce(adjoint, hcat, upreds) |> Lux.cpu_device()

    return code, upred, MSEs
end

#======================================================#
function evolve_autodecoder(
    prob::AbstractPDEProblem,
    decoder::NTuple{3, Any},
    metadata::NamedTuple,
    data::NTuple{3, AbstractVecOrMat},
    p0::AbstractVector;
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
    verbose::Bool = true,
)

    # data, model
    x, _, _ = data

    NN, p0, st = freeze_autodecoder(decoder, p0; rng)
    model = NeuralEmbeddingModel(NN, st, x, metadata)

    # solvers
    linsolve = QRFactorization()
    autodiff = AutoForwardDiff()
    linesearch = LineSearch() # TODO
    nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    nlsmaxiters = 10

    # linesearch = LineSearch(method = BackTracking(), autodiff = AutoZygote())
    # nlssolve = GaussNewton(;autodiff = AutoZygote(), linsolve, linesearch)
    # nlsmaxiters = 20

    autodiff_space = AutoForwardDiff()
    ϵ_space = nothing

    # autodiff_space = AutoFiniteDiff()
    # ϵ_space = 0.005f0

    timealg = EulerForward()
    # timealg = EulerBackward()

    Δt = 1f-3
    time_adaptive = true

    scheme = Galerkin(linsolve, 1f-3, 1f-6) # abstol_inf, abstol_mse
    scheme = Galerkin(linsolve, 1f-0, 1f-2) # abstol_inf, abstol_mse

    # residual = make_residual(prob, timealg; autodiff = autodiff_space, ϵ = ϵ_space)
    # scheme = LeastSqPetrovGalerkin(nlssolve, residual, nlsmaxiters, 1f-6, 1f-3, 1f-6)

    evolve_model(prob, model, timealg, scheme, data, p0, Δt;
        nlssolve, time_adaptive, autodiff_space, ϵ_space, device, verbose,
    )
end

#===========================================================#
function postprocess_autodecoder(
    prob::AbstractPDEProblem,
    datafile::String,
    modelfile::String,
    outdir::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    device = Lux.cpu_device(),
    makeplot::Bool = true,
    verbose::Bool = true,
    fps::Int = 300,
)

    #==============#
    # load data
    #==============#
    data = jldopen(datafile)
    Tdata = data["t"]
    Xdata = data["x"]
    Udata = data["u"]
    mu = data["mu"]

    close(data)

    # data sizes
    Nx, Nb, Nt = size(Udata)

    mu = isnothing(mu) ? fill(nothing, Nb) |> Tuple : mu
    mu = isa(mu, AbstractArray) ? vec(mu) : mu

    #==============#
    # load model
    #==============#
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    close(model)

    #==============#
    mkpath(outdir)
    #==============#

    #==============#
    # subsample in space
    #==============#
    Udata = @view Udata[md.makedata_kws.Ix, :, :]
    Xdata = @view Xdata[md.makedata_kws.Ix]
    Nx = length(Xdata)

    #==============#
    # train/test split
    #==============#
    _Udata = @view Udata[:, md.makedata_kws._Ib, :] # un-normalized
    Udata_ = @view Udata[:, md.makedata_kws.Ib_, :]

    #==============#
    # from training data
    #==============#
    _Ib = isa(md.makedata_kws._Ib, Colon) ? (1:size(Udata, 2)) : md.makedata_kws._Ib
    Ib_ = isa(md.makedata_kws.Ib_, Colon) ? (1:size(Udata, 2)) : md.makedata_kws.Ib_

    _data, _, _ = makedata_autodecoder(datafile; md.makedata_kws...)
    _xdata, _Icode = _data[1]
    _xdata = unnormalizedata(_xdata, md.x̄, md.σx)

    @show md
    
    # model = NeuralEmbeddingModel(NN, st, _Icode, md.x̄, md.σx, md.ū, md.σu) |> device
    # _Upred = model(_xdata |> device, p |> device) |> Lux.cpu_device()
    # _Upred = reshape(_Upred, Nx, _Ib, Nt)
    #
    # for k in _Ib
    #     Ud = @view _Udata[:, k, :]
    #     Up = @view _Upred[:, k, :]
    #
    #     if makeplot
    #         xlabel = "x"
    #         ylabel = "u(x, t)"
    # 
    #         _mu = mu[_Ib[k]]
    #         title  = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"
    #
    #         idx_pred = LinRange(1, size(Ud, 2), 10) .|> Base.Fix1(round, Int)
    #         idx_data = idx_pred
    #
    #         upred = Up[:, idx_pred]
    #         udata = Ud[:, idx_data]
    #
    #         Iplot = 1:8:Nx
    #
    #         plt = plot(xlabel = "x", ylabel = "u(x, t)", legend = false)
    #         plot!(plt, Xdata, upred, w = 2, palette = :tab10)
    #         scatter!(plt, Xdata[Iplot], udata[Iplot, :], w = 1, palette = :tab10)
    #         png(plt, joinpath(outdir, "train$(k)"))
    #
    #         Itplt = LinRange(1, size(Ud, 2), 100) .|> Base.Fix1(round, Int)
    #
    #         anim = animate1D(Ud[:, Itplt], Up[:, Itplt], Xdata, Tdata[Itplt];
    #             w = 2, xlabel, ylabel, title)
    #         gif(anim, joinpath(outdir, "train$(k).gif"); fps)
    #     end
    # end

    #==============#
    # inference (via data regression)
    #==============#

    # decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    #
    # p0 = _code[2].weight[:, 1]
    # Icode = ones(Int32, 1, Nx)
    #
    # NN, p0, st = freeze_autodecoder(decoder, p0; rng)
    # model = NeuralEmbeddingModel(NN, st, Icode, md.x̄, md.σx, md.ū, md.σu)
    #
    # for k in axes(mu)[1]
    #     Ud = Udata[:, k, :]
    #     data = (Xdata, Ud, Tdata)
    #
    #     _, Up, _ = infer_autodecoder(model, data, p0; device, verbose)
    #
    #     if makeplot
    #         xlabel = "x"
    #         ylabel = "u(x, t)"
    #
    #         _mu   = mu[k]
    #         title = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"
    #         _name = k in _Ib ? "infer_train$(k)" : "infer_test$(k)"
    #
    #         idx = LinRange(1, size(Ud, 2), 101) .|> Base.Fix1(round, Int)
    #         plt = plot(;title, xlabel, ylabel, legend = false)
    #         plot!(plt, Xdata, Up[:, idx], w = 2.0,  s = :solid)
    #         plot!(plt, Xdata, Ud[:, idx], w = 4.0,  s = :dash)
    #         png(plt, joinpath(outdir, _name))
    #         display(plt)
    #
    #         anim = animate1D(Ud, Up, Xdata, Tdata;
    #             w = 2, xlabel, ylabel, title)
    #         gif(anim, joinpath(outdir, "$(_name).gif"); fps)
    #     end
    # end

    #==============#
    # evolve
    #==============#
    
    # decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
    # p0 = _code[2].weight[:, 1]
    #
    # for k in axes(mu)[1]
    #     Ud = Udata[:, k, :]
    #     data = (Xdata, Ud, Tdata)
    #     @time _, Up, Tpred = evolve_autodecoder(prob, decoder, md, data, p0;
    #         rng, device, verbose)
    #
    #     if makeplot
    #         xlabel = "x"
    #         ylabel = "u(x, t)"
    #
    #         _mu   = mu[k]
    #         title = isnothing(_mu) ? "" : "μ = $(round(_mu, digits = 2))"
    #         _name = k in _Ib ? "evolve_train$(k)" : "evolve_test$(k)"
    #
    #         plt = plot(; title, xlabel, ylabel, legend = false)
    #         plot!(plt, Xdata, Up[:, idx], w = 2.0,  s = :solid)
    #         plot!(plt, Xdata, Ud[:, idx], w = 4.0,  s = :dash)
    #         png(plt, joinpath(outdir, _name))
    #         display(plt)
    #
    #         anim = animate1D(Ud, Up, Xdata, Tdata;
    #             w = 2, xlabel, ylabel, title)
    #         gif(anim, joinpath(outdir, "$(_name).gif"); fps)
    #     end
    # end

    #==============#
    # check derivative
    #==============#
    begin
        second_derv = false
        third_derv = false
        fourth_derv = true

        decoder, _code = GeometryLearning.get_autodecoder(NN, p, st)
        ncodes = size(_code[2].weight, 2)
        idx = LinRange(1, ncodes, 10) .|> Base.Fix1(round, Int)

        for i in idx
            p0 = _code[2].weight[:, i]

            p1 = plot_derivatives1D_autodecoder(
                decoder, Xdata, p0, md;
                second_derv,
                autodiff = AutoFiniteDiff(), ϵ=1f-2,
            )
            png(p1, joinpath(outdir, "derv_$(i)_FD_AD"))

            p2 = plot_derivatives1D_autodecoder(
                decoder, Xdata, p0, md;
                second_derv, third_derv, fourth_derv,
                autodiff = AutoForwardDiff(),
            )
            png(p2, joinpath(outdir, "derv_$(i)_FWD_AD"))
            display(p2)
        end
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

nothing
#
