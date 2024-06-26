#
using NeuralROMs
using LinearAlgebra, ComponentArrays              # arrays
using Random, Lux, MLUtils, ParameterSchedulers   # ML
using OptimizationOptimJL, OptimizationOptimisers # opt
using LinearSolve, NonlinearSolve, LineSearches   # num
using Plots, JLD2, HDF5, NPZ                      # vis / save
using CUDA, LuxCUDA, KernelAbstractions           # GPU
using LaTeXStrings

let
    pkgpath = dirname(dirname(pathof(NeuralROMs)))
    sdfpath = joinpath(pkgpath, "experiments_SDF")
    !(sdfpath in LOAD_PATH) && push!(LOAD_PATH, sdfpath)
end

# using MeshCat
# using GeometryBasics: Mesh, HyperRectangle, Vec, SVector
# using Meshing: isosurface, MarchingCubes, MarchingTetrahedra, NaiveSurfaceNets

using WGLMakie, Bonito

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

    datas = Tuple(npzread(datafile) for datafile in datafiles)
    xs    = Tuple(data["position"]' for data in datas) # [D, N]
    us    = Tuple(data["distance"]' for data in datas) # [1, N]

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
    NN::Lux.AbstractExplicitLayer,
    casename::String,
    modeldir::String,
    E::Int; # num epochs

    rng::Random.AbstractRNG = Random.default_rng(),

    δ::Real = 0.01f0, # 0.05f0

    lrs = nothing,
    warmup::Bool = false,
    weight_decays::Union{Real, NTuple{M, <:Real}} = 0f-5,
    beta = nothing,
    epsilon = nothing,

    _batchsize = nothing,
    batchsize_ = nothing,

    precompute_mlh::Bool = false,

    makedata_kws = (; _Ix = :,),
    device = Lux.gpu_device(),
) where{M}

    _data, _, metadata = makedata_SDF(casename, δ; makedata_kws...)
    dir = modeldir

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    in_dim  = size(_data[1], 1)
    out_dim = size(_data[2], 1)

    println("input size: $(in_dim)")
    println("output size: $(out_dim)")
    println("Data points: $(numobs(_data))")

    #-------------------------------------------#
    # training hyper-params
    #-------------------------------------------#

    _batchsize = isnothing(_batchsize) ? numobs(_data) ÷ 100 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(_data) ÷ 1   : batchsize_

    lossfun = mae
    weightdecay = WeightDecay(weight_decays)
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, lrs; warmup, weightdecay, beta, epsilon)

    #-------------------------------------------#

    # precompute MLH indices
    if precompute_mlh
        _x  = _data[1]
        MLH = NN.layers.MLH
        _i, MLH = precompute_MLH(_x, MLH)

        NN = Chain(; MLH, NN.layers.MLP)
        _data = (_i, _data[2])
    end

    #-------------------------------------------#
    train_args = (; E, _batchsize, batchsize_)
    metadata   = (; metadata..., δ, train_args)

    display(NN)

    @show metadata

    @time model, ST = train_model(NN, _data; rng,
        _batchsize, batchsize_, weight_decays,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    @show metadata

    plot_training(ST...) |> display

    model, ST, metadata
end
#===========================================================#

function postprocess_SDF(
    modelfile::String;
    samples = (256, 256, 256),
    device = Lux.gpu_device(),
)
    # load model
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    md = model["metadata"]
    δ = md.δ

    display(NN)
    @show md

    sdffile = joinpath(modeldir, "sdf.h5")

    u = if ispath(sdffile)
        file = h5open(sdffile)
        u = file["u"] |> Array
        close(file)
        u
    else
        Xs = Tuple(LinRange(-1.0f0, 1.0f0, samples[i]) for i in 1:3)
        Is = Tuple(ones(Float32, samples[i]) for i in 1:3)

        xx = kron(Is[3], Is[2], Xs[1])
        yy = kron(Is[3], Xs[2], Is[1])
        zz = kron(Xs[3], Is[2], Is[1])

        x = vcat(xx', zz', yy')
        @time u = eval_model((NN, p, st), x; device) #, batchsize = 100)
        @time u = reshape(u, samples...)
        # u = Float16.(u)

        file = h5open(sdffile, "w")
        write(file, "u", u)
        close(file)

        u
    end

    # # MeshCat workflow

    # # RUN.JL
    # isdefined(Main, :vis) && MeshCat.close_server!(vis.core)
    # vis = Visualizer()
    # vis = postprocess_SDF(casename, modelfile; vis, device)
    # open(vis; start_browser = false)

    # # HERE
    # mesh = Mesh(u, MarchingCubes())
    # vis = isnothing(vis) ? Visualizer() : vis
    # setobject!(vis, mesh)
    # return vis

    # # WGLMakie workflow

    # plotsdf() = volume(u;
    #     algorithm = :iso,
    #     isovalue = zero(Float16),
    #     isorange = Float16(0.10δ),
    #     nan_color = :transparent,
    #     colormap = [:transparent, :gray]
    # )

    u = map(x -> x ≤ 0 ? 1 : 0, u) |> BitArray
    plotsdf() = volume(u;
        absorption = 50,
        algorithm = :absorption,
        nan_color = :transparent,
        colormap = [:transparent, :gray]
    )

    makeapp(sess, req) = DOM.div(plotsdf())

    app = App(makeapp)
    server = Bonito.Server(app, "0.0.0.0", 8700)
    return server
end
#===========================================================#
#
