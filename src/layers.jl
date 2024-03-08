#
#======================================================#
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

Assumes input is `(xyz, idx)` of sizes `[D, K]`, `[1, K]` respectively
"""
function AutoDecoder(
    decoder::Lux.AbstractExplicitLayer,
    num_batches::Int,
    code_len::Int;
    init_weight = randn32, # scale_init(randn32, 1f-1, 0f0) # N(μ = 0, σ2 = 0.1^2)
    code = nothing,
    EmbeddingType::Type{<:Lux.AbstractExplicitLayer} = Lux.Embedding
)
    code = if isnothing(code)
        if isone(num_batches)
            # TODO - scatter doesn't work for Zygote over ForwardDiff on GPU.
            # OneEmbedding avoids calls to scatter
            EmbeddingType(num_batches => code_len; init_weight)
            # OneEmbedding(code_len; init_weight)
        else
            Embedding(num_batches => code_len; init_weight)
        end
    else
        code
    end

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

function freeze_autodecoder(
    decoder::NTuple{3, Any},
    p0::AbstractVector;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    num_batches = 1
    code_len = length(p0)
    decoder_frozen = Lux.Experimental.freeze(decoder...)

    NN = AutoDecoder(decoder_frozen[1], num_batches, code_len)
    p, st = Lux.setup(rng, NN)
    st = Lux.testmode(st)
    p = ComponentArray(p)

    copy!(p, p0)
    @set! st.decoder.frozen_params = decoder[2]
    
    NN, p, st
end

#======================================================#
# FrozenDecoderModel
#======================================================#

# function FrozenDecoderModel(
#     decoder::NTuple{3, Any},
#     p0::AbstractVector;
#     rng::Random.AbstractRNG = Random.default_rng(),
# )
#     noop = NoOpLayer()
#     decoder_frozen = Lux.Experimental.freeze(decoder...)
#     parallel = Parallel(vcat, noop, noop)
#
#     NN = Chain(; parallel, decoder_frozen,)
#     p, st = Lux.setup(rng, NN)
#     st = Lux.testmode(st)
#     p = ComponentArray(p)
#    
#     copy!(p, p0)
#     @set! st.decoder_frozen.frozen_params = decoder[2]
#
#     NN, p, st
# end

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

function (e::OneEmbedding)(x::AbstractArray{<:Integer}, ps, st)
    @assert all(isequal(true), x)

    o = Zygote.ignore() do 
        o = similar(x, Bool, length(x))
        fill!(o, true)
    end

    code = ps.weight * o'
    code_re = reshape(code, e.len, size(x)...)

    return code_re, st
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
# Hyper Network
#======================================================#
struct HyperNet{W <: Lux.AbstractExplicitLayer, C <: Lux.AbstractExplicitLayer, A} <:
       Lux.AbstractExplicitContainerLayer{(:weight_generator, :evaluator)}
    weight_generator::W
    evaluator::C
    ca_axes::A
end

function HyperNet(wt_gen::Lux.AbstractExplicitLayer, evltr::Lux.AbstractExplicitLayer)
    rng = Random.default_rng()
    ca_axes = Lux.initialparameters(rng, evltr) |> ComponentArray |> getaxes
    return HyperNet(wt_gen, evltr, ca_axes)
end

function Lux.initialparameters(rng::AbstractRNG, hn::HyperNet)
    return (weight_generator=Lux.initialparameters(rng, hn.weight_generator),)
end

function (hn::HyperNet)(x, ps, st::NamedTuple)
    ps_new, st_ = hn.weight_generator(x, ps.weight_generator, st.weight_generator)
    @set! st.weight_generator = st_
    return ComponentArray(vec(ps_new), hn.ca_axes), st
end

function (hn::HyperNet)((x, y)::T, ps, st::NamedTuple) where {T <: Tuple}
    ps_ca, st = hn(x, ps, st)
    pred, st_ = hn.evaluator(y, ps_ca, st.evaluator)
    @set! st.evaluator = st_
    return pred, st
end

#======================================================#
# Sine Layer
# https://github.com/vsitzmann/siren?tab=readme-ov-file
#======================================================#
export SDense
@concrete struct SDense{use_bias} <: Lux.AbstractExplicitLayer
    activation
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias
    ω0
    is_first
end

function SDense(
    in_dims::Int,
    out_dims::Int,
    activation=identity;
    init_weight=glorot_uniform,
    init_bias=zeros32,
    use_bias::Bool=true,
    allow_fast_activation::Bool=true,
    ω0::Real = 30f0,
    is_first::Bool = false,
)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return SDense{use_bias}(activation, in_dims, out_dims, init_weight, init_bias, ω0, is_first)
end

function Lux.initialparameters(
    rng::AbstractRNG,
    d::SDense{use_bias}
) where {use_bias}

    bias = d.init_bias(rng, d.out_dims, 1)
    weight = d.init_weight(rng, d.out_dims, d.in_dims)

    T = eltype(weight)
    ω = [T(d.ω0),]

    ## scale so that weight * ω ~ unit
    if !d.is_first
        weight = weight ./ ω
        bias   = bias   ./ ω
    end

    if use_bias
        return (; weight, bias, ω)
    else
        return (; weight, ω)
    end
end

function Lux.parameterlength(d::SDense{use_bias}) where {use_bias}
    return use_bias ? d.out_dims * (d.in_dims + 1) + 1 : d.out_dims * d.in_dims + 1
end

Lux.statelength(d::SDense) = 0

@inline function (d::SDense{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    z = ps.weight * x
    return Lux.__apply_activation(d.activation, ps.ω .* z), st
end

@inline function (d::SDense{true})(x::AbstractVector, ps, st::NamedTuple)
    z = ps.weight * x .+ vec(ps.bias)
    y = Lux.__apply_activation(d.activation, ps.ω .* z)
    return y, st
end

@inline function (d::SDense{true})(x::AbstractMatrix, ps, st::NamedTuple)
    z = ps.weight * x .+ ps.bias
    y = Lux.__apply_activation(d.activation, ps.ω .* z)
    return y, st
end

#======================================================#
# Lipschitz Layer
# https://arxiv.org/pdf/2202.08345.pdf
#======================================================#
export LDense
@concrete struct LDense{use_bias} <: Lux.AbstractExplicitLayer
    activation
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias
    ω0
end

function LDense(
    in_dims::Int,
    out_dims::Int,
    activation=identity;
    init_weight=glorot_uniform,
    init_bias=zeros32,
    use_bias::Bool=true,
    allow_fast_activation::Bool=true,
    ω0::Real = 1f0,
)
    activation = allow_fast_activation ? NNlib.fast_act(activation) : activation
    return LDense{use_bias}(activation, in_dims, out_dims, init_weight, init_bias, ω0)
end

function Lux.initialparameters(
    rng::AbstractRNG,
    d::LDense{use_bias}
) where {use_bias}

    bias = d.init_bias(rng, d.out_dims, 1)
    weight = d.init_weight(rng, d.out_dims, d.in_dims)

    T = eltype(weight)
    ω = [T(d.ω0),]

    if use_bias
        return (; weight, bias, ω)
    else
        return (; weight, ω)
    end
end

function Lux.parameterlength(d::LDense{use_bias}) where {use_bias}
    return use_bias ? d.out_dims * (d.in_dims + 1) + 1 : d.out_dims * d.in_dims + 1
end

Lux.statelength(d::LDense) = 0

@inline function (d::LDense{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    w_hat = normalize_lipschitz(ps.weight) .* ps.ω
    z = w_hat * x
    return Lux.__apply_activation(d.activation, z), st
end

@inline function (d::LDense{true})(x::AbstractVector, ps, st::NamedTuple)
    w_hat = normalize_lipschitz(ps.weight) .* ps.ω
    z = w_hat * x .+ vec(ps.bias)
    y = Lux.__apply_activation(d.activation, z)
    return y, st
end

@inline function (d::LDense{true})(x::AbstractMatrix, ps, st::NamedTuple)
    w_hat = normalize_lipschitz(ps.weight) .* ps.ω
    z = w_hat * x .+ ps.bias
    y = Lux.__apply_activation(d.activation, z)
    return y, st
end

function normalize_lipschitz(W::AbstractMatrix) # ||W||_∞
    return W / compute_cbound(W)
end

function compute_cbound(::LDense, p, st)
    prod(p.ω)
end

#===========================================================#
# HOSC activation
# https://arxiv.org/pdf/2401.10967.pdf
#===========================================================#
export hosc, HDense

export HDense
@concrete struct HDense{use_bias} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    init_weight
    init_bias
    ω0
end

function HDense(
    in_dims::Int,
    out_dims::Int;
    init_weight=glorot_uniform,
    init_bias=zeros32,
    use_bias::Bool=true,
    allow_fast_activation::Bool=true,
    ω0::Real = 1f0,
)
    return HDense{use_bias}(in_dims, out_dims, init_weight, init_bias, ω0)
end

function Lux.initialparameters(
    rng::AbstractRNG,
    d::HDense{use_bias}
) where {use_bias}

    bias = d.init_bias(rng, d.out_dims, 1)
    weight = d.init_weight(rng, d.out_dims, d.in_dims)

    T = eltype(weight)
    ω = [T(d.ω0),]

    if use_bias
        return (; weight, bias, ω)
    else
        return (; weight, ω)
    end
end

function Lux.parameterlength(d::HDense{use_bias}) where {use_bias}
    return use_bias ? d.out_dims * (d.in_dims + 1) + 1 : d.out_dims * d.in_dims + 1
end

Lux.statelength(d::HDense) = 0

@inline function (d::HDense{false})(x::AbstractVecOrMat, ps, st::NamedTuple)
    y = ps.weight * x
    z = Lux.__apply_activation(sin, y)
    return Lux.__apply_activation(tanh_fast, ps.ω .* z), st
end

@inline function (d::HDense{true})(x::AbstractVector, ps, st::NamedTuple)
    y = ps.weight * x .+ vec(ps.bias)
    z = Lux.__apply_activation(sin, y)
    return Lux.__apply_activation(tanh_fast, ps.ω .* z), st
end

@inline function (d::HDense{true})(x::AbstractMatrix, ps, st::NamedTuple)
    y = ps.weight * x .+ ps.bias
    z = Lux.__apply_activation(sin, y)
    return Lux.__apply_activation(tanh_fast, ps.ω .* z), st
end

function compute_cbound(::HDense, p, st)
    compute_cbound(p.weight) * prod(p.ω) 
end

#======================================================#
function PermuteLayer(perm::NTuple{D, Int}) where{D}
    WrappedFunction(Base.Fix2(permutedims, perm))
end

#======================================================#
"""
$SIGNATURES

Assumes channel dimension is 1
"""
function PermutedBatchNorm(c, num_dims) # assumes channel_dim = 1

    perm0 = ((2:num_dims-1)..., 1, num_dims)
    perm1 = (num_dims-1, (1:num_dims-2)..., num_dims)

    Chain(
        Base.Fix2(permutedims, perm0),
        BatchNorm(c),
        Base.Fix2(permutedims, perm1),
    )
end

#======================================================#
"""
SplitRows

Split rows of ND array, into `Tuple` of ND arrays.
"""
struct SplitRows{T} <: Lux.AbstractExplicitLayer
    splits::T
    channel_dim::Int
end
function SplitRows(splits...; channel_dim = 1)
    SplitRows(splits, channel_dim)
end
Lux.initialparameters(::Random.AbstractRNG, ::SplitRows) = (;)
Lux.initialstates(::Random.AbstractRNG, ::SplitRows) = (;)
Lux.parameterlength(::SplitRows) = 0
Lux.statelength(::SplitRows) = 0

function (l::SplitRows)(x::AbstractArray, _, st)
    len = sum(length.(l.splits))

    C = l.channel_dim
    N = ndims(x)

    @assert eltype(len) === Int64 "SplitRows only accepts splits eltype Int64"
    @assert len == size(x, C) "Cannot split array of size $(size(x)) with
        splits $(l.splits) along dimension $(C)."

    xs = ()
    _cols = Tuple(Colon() for _ in 1:(C-1))
    cols_ = Tuple(Colon() for _ in (C+1):N)

    for split in l.splits
        if split isa Integer
            split = split:split
        end

        # https://github.com/JuliaGPU/CUDA.jl/issues/2009
        # _x = view(x, _cols..., split, cols_...)
        _x = getindex(x, _cols..., split, cols_...)

        @assert size(_x) == (size(x)[1:C-1]..., length(split), size(x)[C+1:end]...) "got $(size(_x))"

        xs = (xs..., _x)
    end

    xs, st
end

#======================================================#
"""
Diagonal layer
"""
struct Diag{L, I} <: Lux.AbstractExplicitLayer
    len::L
    init::I
end

Diag(len::Int; init = Lux.glorot_uniform) = Diag(len, init)

function Lux.initialparameters(rng::Random.AbstractRNG, l::Diag)
    (;
     diag = l.init(rng, l.len),
    )
end

Lux.initialstates(::Random.AbstractRNG, l::Diag) = (;)
Lux.parameterlength(l::Diag) = l.len
Lux.statelength(::Diag) = 0

function (l::Diag)(x::AbstractArray, ps, st::NamedTuple)

    D = Diagonal(ps.diag)
    y = D * x

    return y, st
end

#======================================================#
"""
Attention Layer

single layer model with no nonlinearity (single head linear attention)

u = NN(f)
q = Wq * u
k = Wk * u
v = Wv * u

v = activation(q * k') * u
"""
struct Atten{I, TI, F} <: Lux.AbstractExplicitLayer
    in_dims::I
    out_dims::I
    init::TI
    activation::F
end

function Atten(in_dims::Int, out_dims::Int; init = Lux.glorot_uniform,
               activation = identity)
    Atten(in_dims, out_dims, init, activation)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::Atten)
    (;
     Wq = l.init(rng, l.out_dims, l.in_dims),
     Wk = l.init(rng, l.out_dims, l.in_dims),
     Wv = l.init(rng, l.out_dims, l.in_dims),
    )
end

Lux.initialstates(::Random.AbstractRNG, l::Atten) = (;)
Lux.parameterlength(l::Atten) = 3 * l.in_dims * l.out_dims
Lux.statelength(::Atten) = 0

function (l::Atten)(x::AbstractArray, ps, st::NamedTuple)
    q = ps.Wq * x
    k = ps.Wk * x
    v = ps.Wv * x

    y = l.activation(q * k') * v

    return y, st
end
#======================================================#
#
