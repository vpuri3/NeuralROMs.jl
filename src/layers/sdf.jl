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
# Spatial grid
# https://github.com/cheind/pure-torch-ngp
#===========================================================#

# function index_grid(
#     Ix::AbstractVector{<:Integer},
#     Iy::AbstractVector{<:Integer},
#     Iz::AbstractVector{<:Integer},
#     shape::NTuple{3, <:Integer},
# )
#     @. (Iz - 1) * (shape[1] * shape[2]) + (Iy - 1) * shape[1] + Ix
# end
#
# @concrete struct FeatureGrid <: Lux.AbstractExplicitContainerLayer{(:embedding,)}
#     shape
#     embedding
# end
#
# function FeatureGrid(
#     out_dims::Integer,
#     shape::Integer...;
#     init_weight = randn32,
# )
#     embedding = Embedding(prod(shape) => out_dims; init_weight)
#     FeatureGrid(shape, embedding)
# end
#
# function (l::FeatureGrid)(x::AbstractArray, p, st)
#     # get bounding box indices
#     Ixyz000 = get_bbox(xyz, l.shape)
#
#     idx = index_grid()
# end

#===========================================================#
# Multiresolution hash encoding based on
# https://github.com/cheind/pure-torch-ngp
#===========================================================#

""" Cubic hermite interpolation """
cubic_interp(f0, f1, w) = @. f0 + (f1 - f0) * (3 - 2w) * w^2
lerp(f0, f1, w) = @. w * (f1 - f0) + f0

function get_bottomleft(
    xyz::AbstractMatrix{T}, # normalized to [0, 1]
    shape::NTuple{D, Ti}
) where{T, D, Ti}
    @assert size(xy, 1) == D

    # same as
    # Ixyz = round.(Ti, xyz .* dxy, RoundDown)

    dxyz = one(T) ./ (shape .- true)
    Ixyz = fld.(xyz, dxyz) .|> Int32
    wxyz = mod.(xyz, dxyz)

    Ixyz, wxyz
end

"""
torchngp/modules/encoding.py
"""
function index_hash(
    Ix::AbstractVector{Integer}, # [K]
    Iy::AbstractVector{Integer},
    Iz::AbstractVector{Integer},
    nEmbeddings::Integer,
)
    primes = [1, 2654435761, 805459861]
    val = xor.(Ix * primes[1], Iy * primes[2], Iz * primes[3])
    mod1.(val, nEmbeddings)
end

export SpatialHash
@concrete struct SpatialHash <: Lux.AbstractExplicitContainerLayer{(:embedding,)}
    shape
    embedding
    interpfun
end

function SpatialHash(
    out_dims::Integer,
    nEmbeddings::Integer,
    shape::NTuple{D, Integer};
    init_weight = randn32,
    interpfun = lerp
) where{D}
    embedding = Embedding(nEmbeddings => out_dims; init_weight)
    SpatialHash(shape, embedding, interpfun)
end

function (l::SpatialHash)(xyz::AbstractArray{T}, p, st) where{T}
    @assert size(xyz, 1) == length(l.shape)

    # xyz = clamp(xyz, -1 + T(1e-6), 1 - T(1e-6))

    Ixyz0, wxyz = get_bottomleft(xyz, l.shape)
    Ix0, Iy0, Iz0 = collect(getindex(d, :) for d in axes(Ixyz0, 1))
    wx , wy , wz  = collect(getindex(d, :) for d in axes(wxyz , 1))

    Ix1, Iy1, Iz1 = map(II -> II .+ true, (Ix, Iy, Iz))

    I000 = index_hash(Ix0, Iy0, Iz0 ,l.nEmbeddings)
    I100 = index_hash(Ix1, Iy0, Iz0 ,l.nEmbeddings)
    I010 = index_hash(Ix0, Iy1, Iz0 ,l.nEmbeddings)
    I110 = index_hash(Ix1, Iy1, Iz0 ,l.nEmbeddings)
    I001 = index_hash(Ix0, Iy0, Iz1 ,l.nEmbeddings)
    I101 = index_hash(Ix1, Iy0, Iz1 ,l.nEmbeddings)
    I011 = index_hash(Ix0, Iy1, Iz1 ,l.nEmbeddings)
    I111 = index_hash(Ix1, Iy1, Iz1 ,l.nEmbeddings)

    f000, _ = l.embedding(I000, p.embedding, st.embedding)
    f100, _ = l.embedding(I100, p.embedding, st.embedding)
    f010, _ = l.embedding(I010, p.embedding, st.embedding)
    f110, _ = l.embedding(I110, p.embedding, st.embedding)
    f001, _ = l.embedding(I001, p.embedding, st.embedding)
    f101, _ = l.embedding(I101, p.embedding, st.embedding)
    f011, _ = l.embedding(I011, p.embedding, st.embedding)
    f111, _ = l.embedding(I111, p.embedding, st.embedding)

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
#
