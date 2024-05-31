#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2, NPZ                            # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using LaTeXStrings

using MeshCat
using GeometryBasics: Mesh, HyperRectangle, Vec, SVector
using Meshing: isosurface, MarchingCubes, MarchingTetrahedra, NaiveSurfaceNets

CUDA.allowscalar(false)

begin
    nt = Sys.CPU_THREADS
    nc = min(nt, length(Sys.cpu_info()))

    BLAS.set_num_threads(nc)
end

# include(joinpath(pkgdir(NeuralROMs), "examples", "cases.jl"))

#======================================================#
function makedata_SDF(
    casename::String,
    δ::Real;
    _Ix = Colon(),  # subsample in space
)
    # load data

    basedir = joinpath(pkgdir(NeuralROMs), "examples", "sdf_dataset")

    datafiles = (;
        near = joinpath(basedir, "near", casename),
        rand = joinpath(basedir, "rand", casename),
        surf = joinpath(basedir, "surface", casename),
    )

    datas = Tuple(
        npzread(datafile) for datafile in datafiles
    )

    xs = Tuple(
        data["position"]' for data in datas # [D, N]
    )

    us = Tuple(
        data["distance"]' for data in datas # [1, N]
    )

    # subsample
    xs = Tuple(x[:, 1:10:end] for x in xs)
    us = Tuple(u[:, 1:10:end] for u in us)

    println("Using $(length.(us)) sample points from subsets $(keys(datafiles)).")

    # make arrays
    x = hcat(xs...)
    u = hcat(us...)

    # clamp RHS
    clamp!(u, -δ, δ)

    # remove NaNs
    __Ix = findall(!isnan, x[1, :])
    __Iy = findall(!isnan, x[2, :])
    __Iz = findall(!isnan, x[3, :])
    __Iu = findall(!isnan, u[1, :])
    __I  = intersect(__Ix, __Iy, __Iz, __Iu)

    x = x[:, __I]
    u = u[:, __I]

    # metadata
    readme = ""

    makedata_kws = (; _Ix,)

    metadata = (;
        casename, readme, δ, makedata_kws,
    )

    (x, u), (x, u), metadata
end

#===========================================================#

function train_SDF(
    casename::String,
    modeldir::String,
    h::Int, # num hidden layers
    w::Int, # hidden layer width
    E::Int; # num epochs
    rng::Random.AbstractRNG = Random.default_rng(),
    δ::Real = 0.1f0,
    warmup::Bool = true,
    _batchsize = nothing,
    batchsize_ = nothing,
    weight_decays::Union{Real, NTuple{M, <:Real}} = 0f0,
    makedata_kws = (; _Ix = :,),
    device = Lux.gpu_device(),
) where{M}

    _data, _, metadata = makedata_SDF(casename, δ; makedata_kws...)
    dir = modeldir

    in_dim  = size(_data[1], 1)
    out_dim = size(_data[2], 1)

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    println("input size: $(in_dim)")
    println("output size: $(out_dim)")
    println("Data points: $(numobs(_data))")

    NN = begin
        # DeepSDF paper recommends weight normalization

        init_wt_in = scaled_siren_init(1f1)
        init_wt_hd = scaled_siren_init(1f0)
        init_wt_fn = glorot_uniform

        init_bias = rand32 # zeros32
        use_bias_fn = false

        act = sin

        wi = in_dim
        wo = out_dim

        in_layer = Dense(wi, w , act; init_weight = init_wt_in, init_bias)
        hd_layer = Dense(w , w , act; init_weight = init_wt_hd, init_bias)
        fn_layer = Dense(w , wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)
        finalize = _clamp_tanh(δ) |> WrappedFunction

        Chain(in_layer, fill(hd_layer, h)..., fn_layer, finalize)
    end

    #-------------------------------------------#
    # training hyper-params
    #-------------------------------------------#

    _batchsize = isnothing(_batchsize) ? numobs(_data) ÷ 100 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data) ÷ 1   : batchsize_

    lossfun = mae # mae_clamped(δ)
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, warmup)

    #-------------------------------------------#

    train_args = (; h, w, E, _batchsize, batchsize_)
    metadata   = (; metadata..., δ, train_args)

    display(NN)

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_, weight_decays,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    plot_training(ST...) |> display

    model, ST, metadata
end
#===========================================================#

function make_optimizer(
    E::Integer,
    warmup::Bool,
    weightdecay = nothing,
)
    lrs = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    Nlrs = length(lrs)

    # Grokking (https://arxiv.org/abs/2201.02177)
    # Optimisers.Adam(lr, (0.9f0, 0.95f0)), # 0.999 (default), 0.98, 0.95
    # https://www.youtube.com/watch?v=IHikLL8ULa4&ab_channel=NeelNanda
    opts = if isnothing(weightdecay)
        Tuple(
            Optimisers.Adam(lr) for lr in lrs
        )
    else
        Tuple(
            OptimiserChain(
                Optimisers.Adam(lr),
                weightdecay,
            )
            for lr in lrs
        )
    end

    nepochs = (round.(Int, E / (Nlrs) * ones(Nlrs))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, Nlrs)...,)

    if warmup
        opt_warmup = if isnothing(weightdecay)
            Optimisers.Adam(1f-2)
        else
            OptimiserChain(Optimisers.Adam(1f-2), weightdecay,)
        end
        nepochs_warmup = 10
        schedule_warmup = Step(1f-2, 1f0, Inf32)
        early_stopping_warmup = true

        ######################
        opts = (opt_warmup, opts...,)
        nepochs = (nepochs_warmup, nepochs...,)
        schedules = (schedule_warmup, schedules...,)
        early_stoppings = (early_stopping_warmup, early_stoppings...,)
    end
    
    opts, nepochs, schedules, early_stoppings
end
#===========================================================#

function postprocess_SDF(
    casename::String,
    modelfile::String;
    vis = Visualizer(),
    samples = (500, 500, 500),
)
    # load model
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]

    # sample points
    u = begin
        X = LinRange(-1.0f0, 1.0f0, samples[1])
        Y = LinRange(-1.0f0, 1.0f0, samples[2])
        Z = LinRange(-1.0f0, 1.0f0, samples[3])

        Ix = ones(Float32, samples[1])
        Iy = ones(Float32, samples[2])
        Iz = ones(Float32, samples[3])

        xx = kron(Iz, Iy, X )
        yy = kron(Iz, Y , Ix)
        zz = kron(Z , Iy, Ix)

        x = vcat(xx', yy', zz')
        u = eval_model((NN, p, st), x; device)
        reshape(u, samples...)
    end

    @time mesh = Mesh(
        u, MarchingCubes();
        origin = SVector(-1f0,-1f0,-1f0), widths = SVector(2f0,2f0,2f0),
    )

    setobject!(vis, mesh)
    vis
end

#===========================================================#
function eval_model(
    model::NTuple{3, Any},
    x;
    batchsize = numobs(x) ÷ 10,
    device = Lux.gpu_device(),
)
    NN, p, st = model

    loader = MLUtils.DataLoader(x; batchsize, shuffle = false, partial = true)

    p, st = (p, st) |> device
    st = Lux.testmode(st)

    if device isa Lux.LuxCUDADevice
        loader = CuIterator(loader)
    end

    y = ()
    for batch in loader
        yy = NN(batch, p, st)[1]
        y = (y..., yy)
    end

    hcat(y...) |> Lux.cpu_device()
end
#===========================================================#

_clamp_vanilla(δ)  = x -> @. clamp(x, -δ, δ)
_clamp_tanh(δ)     = x -> @. δ * tanh_fast(x)
_clamp_sigmoid(δ)  = x -> @. δ * (2 * sigmoid_fast(x) - 1)
_clamp_softsign(δ) = x -> @. δ * softsign(x)

#===========================================================#

#
