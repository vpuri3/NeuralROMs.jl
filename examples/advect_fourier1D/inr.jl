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

function inr_decoder(l, h, w, in_dim, out_dim)
    init_wt_in = scaled_siren_init(3f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    act = sin

    wi = l + in_dim
    wo = out_dim

    in_layer = Dense(wi, w , act; init_weight = init_wt_in, init_bias)
    hd_layer = Dense(w , w , act; init_weight = init_wt_hd, init_bias)
    fn_layer = Dense(w , wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

    Chain(in_layer, fill(hd_layer, h)..., fn_layer)
end

function inr_network(
    prob::GeometryLearning.AbstractPDEProblem,
    l::Integer,
    h::Integer,
    we::Integer,
    wd::Integer,
    act,
)

    if prob isa Advection1D
        Ns = (128,)
        in_dim  = 1
        out_dim = 1

        wi = in_dim

        encoder = Chain(
            Conv((8,), wi  => we, act; stride = 4, pad = 2), # /4
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4
            Conv((2,), we  => we, act; stride = 1, pad = 0), # /2
            flatten,
            Dense(we, l),
        )

        decoder = inr_decoder(l, h, wd, in_dim, out_dim)
        
        ImplicitEncoderDecoder(encoder, decoder, Ns, out_dim)

    elseif prob isa ViscousBurgers1D
    end
end

#======================================================#
function makedata_INR(
    datafile::String;
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
    t  = data["t"]
    x  = data["x"]
    u  = data["u"] # [Nx, Nb, Nt] or [out_dim, Nx, Nb, Nt]
    mu = data["mu"]
    md_data = data["metadata"]
    close(data)
    
    @assert ndims(u) ∈ (3,4,)
    @assert x isa AbstractVecOrMat
    x = x isa AbstractVector ? reshape(x, 1, :) : x # (Dim, Npoints)
    
    if ndims(u) == 3 # [Nx, Nb, Nt]
        u = reshape(u, 1, size(u)...) # [1, Nx, Nb, Nt]
    end
    
    in_dim  = size(x, 1)
    out_dim = size(u, 1)
    
    println("input size $in_dim with $(size(x, 2)) points per trajectory.")
    println("output size $out_dim.")
    
    @assert eltype(x) === Float32
    @assert eltype(u) === Float32
    
    #==============#
    # normalize
    #==============#
    
    ū  = sum(u, dims = (2,3,4)) / (length(u) ÷ out_dim) |> vec
    σu = sum(abs2, u .- ū, dims = (2,3,4)) / (length(u) ÷ out_dim) .|> sqrt |> vec
    u  = normalizedata(u, ū, σu)
    
    x̄  = sum(x, dims = 2) / size(x, 2) |> vec
    σx = sum(abs2, x .- x̄, dims = 2) / size(x, 2) .|> sqrt |> vec
    x  = normalizedata(x, x̄, σx)

    #==============#
    # subsample, test/train split
    #==============#
    _x = @view x[:, Ix]
    x_ = @view x[:, Ix]

    _u = @view u[:, Ix, _Ib, _It]
    u_ = @view u[:, Ix, Ib_, It_]

    Nx = size(_x, 2)
    @assert size(_u, 2) == size(_x, 2) "size(_u): $(size(_u)), size(_x): $(size(_x))"

    println("Using $Nx sample points per trajectory.")

    _Ns = size(_u, 3) * size(_u, 4) # number of codes i.e. # trajectories
    Ns_ = size(u_, 3) * size(u_, 4)

    println("$_Ns / $Ns_ trajectories in train/test sets.")

    readme = "Train/test on the same trajectory."

    #==============#
    # make arrays
    #==============#

    @assert in_dim == 1 "work on Burgers 2D later"

    _uperm = permutedims(_u, (2, 1, 3, 4)) # [Nx, out_dim, Nb, Nt]
    uperm_ = permutedims(u_, (2, 1, 3, 4))

    _uperm = reshape(_uperm, Nx, in_dim, _Ns)
    uperm_ = reshape(uperm_, Nx, in_dim, Ns_)

    _xperm = permutedims(_x, (2, 1))
    xperm_ = permutedims(x_, (2, 1))

    _X = zeros(Float32, Nx, in_dim + out_dim, _Ns)
    X_ = zeros(Float32, Nx, in_dim + out_dim, Ns_)

    _X[:, begin:out_dim, :] = _uperm # [x; u]
    X_[:, begin:out_dim, :] = uperm_

    _X[:, out_dim+1:end, :] .= _xperm
    X_[:, out_dim+1:end, :] .= xperm_

    _U = reshape(_u, out_dim, Nx, _Ns)
    U_ = reshape(u_, out_dim, Nx, _Ns)

    readme = "Train/test on 0.0-0.5."
    makedata_kws = (; Ix, _Ib, Ib_, _It, It_,)
    metadata = (; ū, σu, x̄, σx,
        Nx, _Ns, Ns_,
        makedata_kws, md_data, readme,
    )

    (_X, _U), (X_, U_), metadata
end

#======================================================#

function train_INR(
    datafile::String,
    modeldir::String,
    NN::Lux.AbstractExplicitLayer,
    E::Int; # num epochs
    rng::Random.AbstractRNG = Random.default_rng(),
    warmup::Bool = true,
    _batchsize = nothing,
    batchsize_ = nothing,
    makedata_kws = (; Ix = :, _Ib = :, Ib_ = :, _It = :, It_ = :,),
    cb_epoch = nothing,
    device = Lux.cpu_device(),
)

    _data, _, metadata = makedata_INR(datafile; makedata_kws...)
    dir = modeldir

    lossfun = function(NN, p, st, batch)
        x, ŷ = batch
        y, st = NN(x, p, st)
        loss = sum(abs2, ŷ - y) / length(ŷ)

        loss, st, ()
    end

    _batchsize = isnothing(_batchsize) ? numobs(_data) ÷ 50 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data)      : batchsize_

    #--------------------------------------------#
    # optimizer
    #--------------------------------------------#
    lrs = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    Nlrs = length(lrs)

    opts = Tuple(Optimisers.Adam(lr) for lr in lrs)

    nepochs = (round.(Int, E / (Nlrs) * ones(Nlrs))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, Nlrs)...,)

    if warmup
        opt_warmup = Optimisers.Adam(1f-2)
        nepochs_warmup = 10
        schedule_warmup = Step(1f-2, 1f0, Inf32)
        early_stopping_warmup = true
        
        ######################
        opts = (opt_warmup, opts...,)
        nepochs = (nepochs_warmup, nepochs...,)
        schedules = (schedule_warmup, schedules...,)
        early_stoppings = (early_stopping_warmup, early_stoppings...,)
    end

    #--------------------------------------------#

    train_args = (; E, _batchsize,)
    metadata   = (; metadata..., train_args)

    @show metadata

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_, 
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
        cb_epoch,
    )

    @show metadata

    plot_training(ST...) |> display

    model, ST, metadata
end

#======================================================#
rng = Random.default_rng()
Random.seed!(rng, 199)

# parameters
E   = 700  # epochs
l   = 1    # latent
h   = 5    # hidden
we  = 32   # width
wd  = 32   # width
act = tanh # relu, tanh

prob = Advection1D(0.25f0)
datafile  = joinpath(@__DIR__, "data_advect/", "data.jld2")
modeldir  = joinpath(@__DIR__, "CAE")
modelfile = joinpath(modeldir, "model_07.jld2")
outdir    = joinpath(modeldir, "results")
device = Lux.gpu_device()

NN = inr_network(prob, l, h, we, wd, act)

p, st = Lux.setup(rng, NN)
p = ComponentArray(p)
_data, _, _ = makedata_INR(datafile)
@show _data[1] |> size
@show _data[2] |> size
@show NN(_data[1], p, st)[1] |> size
@show length(p.decoder)

# ## train
isdir(modeldir) && rm(modeldir, recursive = true)
model, ST, metadata = train_INR(datafile, modeldir, NN, E; rng, warmup = false, device)
#======================================================#
nothing
#
