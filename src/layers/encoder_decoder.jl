#
#======================================================#
using Lux: initialstates
"""
    ImplicitEncoderDecoder

Composition of a (possibly convolutional) encoder and an implicit
neural network decoder.

The input array `[Nx, Ny, C, B]` or `[C, Nx, Ny, B]` is expected to
contain `XYZ` coordinates in the last `dim` entries of the channel
dimension which is dictated by `channel_dim`. The number of channels in
the input array must match `encoder_width + D`, where `encoder_width`
is the expected input width of your encoder. The `encoder` network
is expected to work with whatever `channel_dim`, `encoder_channels`
you choose.

NOTE: `channel_dim` is set to 1. So the assumption is `[C, Nx, Ny, B]`

The coordinates are split and the remaining channels are
passed to `encoder` which compresses each `[:, :, :, 1]` slice into
a latent vector of length `L`. The output of the encoder is of size
`[L, B]`.

With a compressed representation of each `image`, we are ready to apply the
decoder mapping. The decoder is an implicit neural network which expects
as input the concatenation of the latent vector and a query point.
The decoder returns the value of the target field at that point.

The decoder is usually a deep neural network and expects the channel
dimension to be the leading dimension. The decoder expects input with
size of leading dimension `L+dim`, and returns an array with leading
size `out_dim`.

Here, we feed it an array of size `[L+2, Nx, Ny, B]`, where the input
`Npoints` equal to `(Nx, Ny,)` is the number of training points in each
trajectory.

"""
function ImplicitEncoderDecoder(
    encoder::Lux.AbstractExplicitLayer,
    decoder::Lux.AbstractExplicitLayer,
    Npoints::NTuple{D, Integer},
    out_dim::Integer,
) where{D}

    in_dim = D
    channel_dim = D + 1 # in_dim

    # channel length
    channel_split = 1:in_dim, (in_dim+1):(in_dim + out_dim)

    repeat = WrappedFunction(Base.Fix2(_ntimes, Npoints))
    noop = NoOpLayer()

    PERM1 = (D+1, 1:D..., D+2) # [Nx, Ny, C, B] -> [C, Nx, Ny, B]
    PERM2 = (2:D+1..., 1, D+2) # [C, Nx, Ny, B] -> [Nx, Ny, C, B]

    perm1 = PermuteLayer(PERM1)
    perm2 = PermuteLayer(PERM2)

    Chain(;
        split   = SplitRows(channel_split...; channel_dim), # x[N, in_dim, B], u[N, out_dim, B] 
        encode  = Parallel(nothing; noop, encoder),         # x[N, in_dim, B], ũ[L, B]
        assem   = Parallel(vcat, perm1, repeat),            # x[in_dim, N, B], ũ[L, N, B]  -> [L + in_dim, N, B]
        decoder = decoder,                                  # [L + in_dim, N, B] -> [out_dim, N, B]
    )
end

function get_encoder_decoder(NN::Lux.AbstractExplicitLayer, p, st)
    encoder = (NN.layers.encode.layers.encoder, p.encode.encoder, st.encode.encoder)
    decoder = (NN.layers.decoder, p.decoder, st.decoder)
    
    remake_ca_in_model(encoder...), remake_ca_in_model(decoder...)
end

#======================================================#
# Auto Decoder
#======================================================#
"""
    AutoDecoder

Assumes input is `(xyz, idx)` of sizes `[in_dim, K]`, `[1, K]` respectively
"""
function AutoDecoder(
    decoder::Lux.AbstractExplicitLayer,
    num_batches::Int,
    code_len::Int;
    init_weight = randn32, # scale_init(randn32, 1f-1, 0f0) # N(μ = 0, σ2 = 0.1^2)
    code = nothing,
)
    code = Embedding(num_batches => code_len; init_weight)
    noop = NoOpLayer()

    codex = Chain(;
        vec = WrappedFunction(vec),
        code = code,
    )

    Chain(;
        assem   = Parallel(vcat; noop, codex), # [D+L, K] (x, code)
        decoder = decoder,                     # [out, K] (out)
    )
end

function get_autodecoder(
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, AbstractArray},
    st::NamedTuple,
)
    decoder = (NN.layers.decoder, p.decoder, st.decoder)
    code    = (NN.layers.assem.layers.codex.layers.code, p.assem.codex.code, st.assem.codex.code)

    remake_ca_in_model(decoder...), remake_ca_in_model(code...)
end

#======================================================#
# FlatDecoder
#======================================================#
"""
    FlatDecoder

Input: `(x, param)` of sizes `[x_dim, K]`, and `[p_dim, K]` respectively.
Output: solution field `u` of size `[out_dim, K]`.
"""
function FlatDecoder(
    hyper::Lux.AbstractExplicitLayer,
    decoder::Lux.AbstractExplicitLayer,
)
    noop = NoOpLayer()

    Chain(;
        assem   = Parallel(vcat; noop, hyper), # [D+L, K] (x, code)
        decoder = decoder,                     # [out, K] (out)
    )
end

function get_flatdecoder(
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, AbstractArray},
    st::NamedTuple,
)
    hyper   = (NN.layers.assem.layers.hyper, p.assem.hyper, st.assem.hyper)
    decoder = (NN.layers.decoder, p.decoder, st.decoder)

    remake_ca_in_model(hyper...), remake_ca_in_model(decoder...)
end

#======================================================#
# OneEmbedding, freeze_decoder
#======================================================#

struct OneEmbedding{F} <: Lux.AbstractExplicitLayer
    len::Int
    init::F
end

OneEmbedding(len::Int; init_weight = zeros32) = OneEmbedding(len, init_weight)

function Lux.initialparameters(rng::AbstractRNG, e::OneEmbedding)
    return (; weight = e.init(rng, e.len),)
end

Lux.initialstates(::Random.AbstractRNG, ::OneEmbedding) = (;)

function (e::OneEmbedding)(x::AbstractVecOrMat, ps, st)
    code = repeat(ps.weight, 1, size(x, 2)) # [x_in, K]
    return code, st
end

function freeze_decoder(
    decoder::NTuple{3, Any},
    code_len::Integer;
    rng::Random.AbstractRNG = Random.default_rng(),
    p0::Union{AbstractVector, Nothing} = nothing,
)
    # freeze decoder
    decoder_frozen = Lux.Experimental.freeze(decoder...)

    # make NN
    branch = BranchLayer(NoOpLayer(), OneEmbedding(code_len))
    parallel = Parallel(vcat, NoOpLayer(), NoOpLayer())
    NN = Chain(; branch, parallel, decoder = decoder_frozen[1])

    # setup NN
    p, st = Lux.setup(rng, NN)
    st = Lux.testmode(st)
    p = ComponentArray(p)

    @set! st.decoder.frozen_params = decoder[2]

    if !isnothing(p0)
        @assert length(p0) == code_len
        copy!(p, p0)
    end

    NN, p, st
end

#======================================================#
# Hyper Decoder
#======================================================#
"""
    HyperDecoder

Assumes input is `(xyz, idx)` of sizes `[D, K]`, `[1, K]` respectively
"""
function HyperDecoder(
    weight_gen::Lux.AbstractExplicitLayer,
    evaluator::Lux.AbstractExplicitLayer,
    num_batches::Int,
    code_len::Int;
    init_weight = randn32,
    code = nothing,
)
    code = if isnothing(code)
        Embedding(num_batches => code_len; init_weight)
    else
        code
    end

    code_gen = Chain(;
        vec  = WrappedFunction(vec),
        code = code,
        gen  = weight_gen,
    )

    HyperNet(code_gen, evaluator)
end

function get_hyperdecoder(NN::Lux.AbstractExplicitLayer, p, st)
    NN.weight_generator.code, p.weight_generator.code, st.weight_generator.code
end
#======================================================#
#
