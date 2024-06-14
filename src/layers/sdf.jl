#
#===========================================================#
clamp_vanilla(x::AbstractArray, δ) = @. clamp(x, -δ, δ)
clamp_tanh(x::AbstractArray, δ)    = @. δ * tanh_fast(x)
clamp_sigmoid(x::AbstractArray, δ) = @. δ * (2 * sigmoid_fast(x) - 1)
clamp_softsign(x::AbstractArray,δ) = @. δ * softsign(x)

struct ClampVanilla{T <: Real}  <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampTanh{T <: Real}     <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampSigmoid{T <: Real}  <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampSoftsign{T <: Real} <: Lux.AbstractExplicitLayer; δ::T; end

Lux.initialstates(::ClampVanilla ) = (;)
Lux.initialstates(::ClampTanh    ) = (;)
Lux.initialstates(::ClampSigmoid ) = (;)
Lux.initialstates(::ClampSoftsign) = (;)

Lux.initialparameters(::ClampVanilla ) = (;)
Lux.initialparameters(::ClampTanh    ) = (;)
Lux.initialparameters(::ClampSigmoid ) = (;)
Lux.initialparameters(::ClampSoftsign) = (;)

Lux.statelength(::ClampVanilla ) = 0
Lux.statelength(::ClampTanh    ) = 0
Lux.statelength(::ClampSigmoid ) = 0
Lux.statelength(::ClampSoftsign) = 0

Lux.parameterlength(::ClampVanilla ) = 0
Lux.parameterlength(::ClampTanh    ) = 0
Lux.parameterlength(::ClampSigmoid ) = 0
Lux.parameterlength(::ClampSoftsign) = 0

(l::ClampVanilla )(x::AbstractArray, p, st) = clamp_vanilla(x, l.δ), st
(l::ClampTanh    )(x::AbstractArray, p, st) = clamp_tanh(x, l.δ), st
(l::ClampSigmoid )(x::AbstractArray, p, st) = clamp_sigmoid(x, l.δ), st
(l::ClampSoftsign)(x::AbstractArray, p, st) = clamp_softsign(x, l.δ), st

#===========================================================#
# Multiresolution hash encoding based on
# https://github.com/cheind/pure-torch-ngp
# torchngp/modules/encoding.py
#===========================================================#

""" Cubic hermite interpolation """
cubic_interp(f0, f1, w) = @. f0 + (f1 - f0) * (3 - 2w) * w^2
lerp(f0, f1, w) = @. w * (f1 - f0) + f0

function get_bottomleft(
    xyz::AbstractMatrix{T},
    shape::NTuple{D, Ti}
) where{T, D, Ti}
    @assert size(xyz, 1) == D

    # normalize from [-1, 1] to [0, 1]
    xyz  = (xyz .+ 1f0) .* 0.5f0

    XYZ  = xyz .* shape
    Ixyz = floor.(Int32, XYZ)
    wxyz = XYZ - Ixyz

    Ixyz, wxyz
end

function index_hash(
    Ix::AbstractArray{Ti},
    Iy::AbstractArray{Ti},
    Iz::AbstractArray{Ti},
    nEmbeddings::Integer,
) where{Ti <: Integer}

    # https://bigprimes.org/
    primes = Int32[1, 481549, 928079]
    val = xor.(Ix * primes[1], Iy * primes[2], Iz * primes[3])
    mod1.(val, Ti(nEmbeddings))
end

function index_grid(
    Ix::AbstractArray{Ti},
    Iy::AbstractArray{Ti},
    Iz::AbstractArray{Ti},
    shape::NTuple{3, <:Integer},
) where{Ti}
    @. (Iz - 1) * (shape[1] * shape[2]) + (Iy - 1) * shape[1] + Ix
end

function get_boundingbox(
    xyz::AbstractArray{T},
    shape::NTuple{D, Ti},
    nEmbeddings,
    indexfun,
) where{T, D, Ti}

    Ixyz0, wxyz = get_bottomleft(xyz, shape)

    Ix0, Iy0, Iz0 = collect(getindex(Ixyz0, d, :) for d in axes(Ixyz0, 1))
    wx , wy , wz  = collect(getindex(wxyz , d, :) for d in axes(wxyz , 1))

    Ix1, Iy1, Iz1 = map(II -> II .+ true, (Ix0, Iy0, Iz0))
    wx , wy , wz  = map(w -> reshape(w, 1, size(w)...), (wx, wy, wz))

    I000 = indexfun(Ix0, Iy0, Iz0, nEmbeddings)
    I100 = indexfun(Ix1, Iy0, Iz0, nEmbeddings)
    I010 = indexfun(Ix0, Iy1, Iz0, nEmbeddings)
    I110 = indexfun(Ix1, Iy1, Iz0, nEmbeddings)
    I001 = indexfun(Ix0, Iy0, Iz1, nEmbeddings)
    I101 = indexfun(Ix1, Iy0, Iz1, nEmbeddings)
    I011 = indexfun(Ix0, Iy1, Iz1, nEmbeddings)
    I111 = indexfun(Ix1, Iy1, Iz1, nEmbeddings)

    (wx, wy, wz), (I000, I100, I010, I110, I001, I101, I011, I111)
end
#===========================================================#
# Spatial hash grid
#===========================================================#

@concrete struct SpatialHash{D} <: Lux.AbstractExplicitContainerLayer{(:embedding,)} where{D}
    shape
    embedding
    interpfun
end

function SpatialHash(
    out_dims::Integer,
    nEmbeddings::Integer,
    shape::NTuple{D, Integer};
    init_weight = randn32,
    interpfun = lerp,
) where{D}

    embedding = Embedding(nEmbeddings => out_dims; init_weight)

    SpatialHash{D}(shape, embedding, interpfun)
end

function (l::SpatialHash)(xyz::AbstractArray{T}, p, st) where{T}
    @assert size(xyz, 1) == length(l.shape)
    wxyz, Ibbox = Zygote.@ignore get_boundingbox(xyz, l.shape, l.embedding.in_dims, index_hash)

    l((wxyz, Ibbox), p, st)
end

function (l::SpatialHash)((wxyz, Ibbox), p, st)
    wx, wy, wz = wxyz
    I000, I100, I010, I110, I001, I101, I011, I111 = Ibbox

    f000, _ = l.embedding(I000, p, st)
    f100, _ = l.embedding(I100, p, st)
    f010, _ = l.embedding(I010, p, st)
    f110, _ = l.embedding(I110, p, st)
    f001, _ = l.embedding(I001, p, st)
    f101, _ = l.embedding(I101, p, st)
    f011, _ = l.embedding(I011, p, st)
    f111, _ = l.embedding(I111, p, st)

    # bottom (z=0)
    fx00 = l.interpfun(f000, f100, wx)
    fx10 = l.interpfun(f010, f110, wx)
    fxy0 = l.interpfun(fx00, fx10, wy)

    # top (z=1)
    fx01 = l.interpfun(f001, f101, wx)
    fx11 = l.interpfun(f011, f111, wx)
    fxy1 = l.interpfun(fx01, fx11, wy)

    f = l.interpfun(fxy0, fxy1, wz)

    f, st
end

#===========================================================#
# Spatial grid
#===========================================================#

# @concrete struct FeatureGrid{D} <: Lux.AbstractExplicitContainerLayer{(:embedding,)} where{D}
#     shape
#     embedding
#     interpfun
# end
#
# function FeatureGrid(
#     out_dims::Integer,
#     shape::NTuple{D, Integer};
#     init_weight = randn32,
#     interpfun = lerp,
# ) where{D}
#
#     nEmbeddings = prod(shape)
#     embedding = Embedding(nEmbeddings => out_dims; init_weight)
#
#     FeatureGrid{D}(shape, embedding, interpfun)
# end
#
# function (l::FeatureGrid)(xyz::AbstractArray{T}, p, st) where{T}
#     @assert size(xyz, 1) == length(l.shape)
#     wxyz, Ibbox = Zygote.@ignore get_boundingbox(xyz, l.shape, l.embedding.in_dims, index_grid)
#
#     l((wxyz, Ibbox), p, st)
# end
#
# function (l::FeatureGrid)((wxyz, Ibbox), p, st)
#     wx, wy, wz = Zygote.@ignore wxyz
#     I000, I100, I010, I110, I001, I101, I011, I111 = Zygote.@ignore Ibbox
#
#     f000, _ = l.embedding(I000, p, st)
#     f100, _ = l.embedding(I100, p, st)
#     f010, _ = l.embedding(I010, p, st)
#     f110, _ = l.embedding(I110, p, st)
#     f001, _ = l.embedding(I001, p, st)
#     f101, _ = l.embedding(I101, p, st)
#     f011, _ = l.embedding(I011, p, st)
#     f111, _ = l.embedding(I111, p, st)
#
#     # bottom (z=0)
#     fx00 = l.interpfun(f000, f100, wx)
#     fx10 = l.interpfun(f010, f110, wx)
#     fxy0 = l.interpfun(fx00, fx10, wy)
#
#     # top (z=1)
#     fx01 = l.interpfun(f001, f101, wx)
#     fx11 = l.interpfun(f011, f111, wx)
#     fxy1 = l.interpfun(fx01, fx11, wy)
#
#     f = l.interpfun(fxy0, fxy1, wz)
#
#     f, st
# end
#===========================================================#

export MultiLevelSpatialHash
function MultiLevelSpatialHash(;
    out_dims::Integer = 2,
    nEmbeddings::Integer = 2^14,
    nLevels::Integer = 16,
    min_res::Integer = 16,  # shape of first level
    max_res::Integer = 512, # shape of final level
    init_weight = scale_init(randn32, 1f-4, 0f0),
    interpfun = lerp,
)
    gf = exp(log(max_res / min_res) / (nLevels - 1)) # growth factor
    Ns = min_res .* collect(gf ^ l for l in 0:nLevels-1)
    Ns = round.(Int32, Ns)

    println("Generating $(nLevels) spatial hashes with resolutions $(Ns)")

    levels = NamedTuple(
        Symbol("hash$l") => SpatialHash(
            out_dims, nEmbeddings, (Ns[l], Ns[l], Ns[l]);
            init_weight, interpfun,
        )
        for l in 1:nLevels
    )

    # Chain(;
    #     branch = BranchLayer(; noop = NoOpLayer(), levels...),
    #     Vcat = Parallel(vcat, (NoOpLayer() for _ in 1:nLevels+1)...),
    # )
    Chain(;
        branch = BranchLayer(; levels...),
        Vcat = Parallel(vcat, (NoOpLayer() for _ in 1:nLevels)...),
    )
end

function precompute_indices(xyz, MLH)
    w1 = (wx, wy, wz)
    I1 = (I000, I100)
    (
        (w1, I1),
        (w2, I2),
    )
end

#===========================================================#
#
