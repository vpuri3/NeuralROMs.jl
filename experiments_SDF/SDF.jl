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

nc = min(Sys.CPU_THREADS, length(Sys.cpu_info()))
BLAS.set_num_threads(nc)

include(joinpath(pkgdir(NeuralROMs), "experiments_SDF", "utils.jl"))

#======================================================#
function makedata_SDF(
    casename::String,
    δ::Real;
    _Ix = Colon(),  # subsample in space
)
    # load data
    basedir = joinpath(pkgdir(NeuralROMs), "experiments_SDF", "dataset_sdf-explorer")

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
    warmup::Bool = false,
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
        finalize = clamp_tanh(δ) |> WrappedFunction

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

    mesh = begin
        Xs = Tuple(LinRange(-1.0f0, 1.0f0, samples[i]) for i in 1:3)
        Is = Tuple(ones(Float32, samples[i]) for i in 1:3)

        xx = kron(Is[3], Is[2], Xs[1])
        yy = kron(Is[3], Xs[2], Is[1])
        zz = kron(Xs[3], Is[2], Is[1])

        x = vcat(xx', zz', yy')
        u = eval_model((NN, p, st), x; device)
        u = reshape(u, samples...)

        Mesh(u, MarchingCubes())
    end

    setobject!(vis, mesh)
    vis
end
#===========================================================#
#
