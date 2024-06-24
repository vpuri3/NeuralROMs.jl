#
using NeuralROMs
include(joinpath(pkgdir(NeuralROMs), "experiments_SDF", "SDF.jl"))

#======================================================#
# Select case
#======================================================#
# casename = "Gear.npz"
# modeldir  = joinpath(@__DIR__, "dump1")

# casename = "Temple.npz"
# modeldir  = joinpath(@__DIR__, "dump2")

# casename = "Burger.npz"
# modeldir  = joinpath(@__DIR__, "dump3")

casename = "HumanSkull.npz"
modeldir  = joinpath(@__DIR__, "dump4")

# casename = "Cybertruck.npz"
# modeldir  = joinpath(@__DIR__, "dump5")
#======================================================#

rng = Random.default_rng()
Random.seed!(rng, 199)

modelfile = joinpath(modeldir, "model_05.jld2")
device = Lux.gpu_device()

δ = 0.01f0
#======================================================#
# Train MLP
#======================================================#

# NN = begin
#     h, w = 5, 512
#     # DeepSDF paper recommends weight normalization
#
#     init_wt_in = scaled_siren_init(1f1)
#     init_wt_hd = scaled_siren_init(1f0)
#     init_wt_fn = glorot_uniform
#
#     init_bias = rand32 # zeros32
#     use_bias_fn = false
#
#     act = sin
#
#     wi, wo = 3, 1
#
#     in_layer = Dense(wi, w , act; init_weight = init_wt_in, init_bias)
#     hd_layer = Dense(w , w , act; init_weight = init_wt_hd, init_bias)
#     fn_layer = Dense(w , wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)
#     finalize = ClampTanh(δ)
#
#     Chain(in_layer, fill(hd_layer, h)..., fn_layer, finalize)
# end
#
# E = 490
# isdir(modeldir) && rm(modeldir, recursive = true)
# model, ST, md = train_SDF(NN, casename, modeldir, E; rng, warmup = true, δ, device)

#======================================================#
# Hash Encoding
#======================================================#

NN = begin
    nLevels = 8
    out_dims = 2
    nEmbeddings = 2^10

    MLH = MultiLevelSpatialHash(; out_dims, nEmbeddings, nLevels, min_res = 8)

    mlp_w  = 64
    mlp_in = out_dims * nLevels #+ 3

    MLP = Chain(
        Dense(mlp_in, mlp_w, relu),
        Dense(mlp_w, 1; use_bias = false),
        ClampTanh(δ),
    )

    Chain(; MLH, MLP)
end

E = 500
warmup = false
lrs = (1f-3, 5f-4, 1f-4, 5f-5, 1f-5,)
beta = (0.9f0, 0.99f0)
epsilon = 1f-15
ifmlh = true


isdir(modeldir) && rm(modeldir, recursive = true)
model, ST, md = train_SDF(NN, casename, modeldir, E; rng, δ,
    lrs, warmup, beta, epsilon, ifmlh, device,
)

#======================================================#
# process visualization
#======================================================#
# isdefined(Main, :server) && close(server)
# server = postprocess_SDF(modelfile; device)
#======================================================#
nothing
#
