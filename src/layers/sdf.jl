#
#===========================================================#
clamp_tanh(x::AbstractArray    , δ) = @. δ * tanh_fast(x)
clamp_vanilla(x::AbstractArray , δ) = @. clamp(x, -δ, δ)
clamp_sigmoid(x::AbstractArray , δ) = @. δ * (2 * sigmoid_fast(x) - 1)
clamp_softsign(x::AbstractArray, δ) = @. δ * softsign(x)

struct ClampTanh{T <: Real}     <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampVanilla{T <: Real}  <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampSigmoid{T <: Real}  <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampSoftsign{T <: Real} <: Lux.AbstractExplicitLayer; δ::T; end

Lux.initialstates(::ClampTanh    ) = (;)
Lux.initialstates(::ClampVanilla ) = (;)
Lux.initialstates(::ClampSigmoid ) = (;)
Lux.initialstates(::ClampSoftsign) = (;)

Lux.initialparameters(::ClampTanh    ) = (;)
Lux.initialparameters(::ClampVanilla ) = (;)
Lux.initialparameters(::ClampSigmoid ) = (;)
Lux.initialparameters(::ClampSoftsign) = (;)

Lux.statelength(::ClampTanh    ) = 0
Lux.statelength(::ClampVanilla ) = 0
Lux.statelength(::ClampSigmoid ) = 0
Lux.statelength(::ClampSoftsign) = 0

Lux.parameterlength(::ClampTanh    ) = 0
Lux.parameterlength(::ClampVanilla ) = 0
Lux.parameterlength(::ClampSigmoid ) = 0
Lux.parameterlength(::ClampSoftsign) = 0

(l::ClampTanh    )(x::AbstractArray, p, st) = clamp_tanh(x, l.δ), st
(l::ClampVanilla )(x::AbstractArray, p, st) = clamp_vanilla(x, l.δ), st
(l::ClampSigmoid )(x::AbstractArray, p, st) = clamp_sigmoid(x, l.δ), st
(l::ClampSoftsign)(x::AbstractArray, p, st) = clamp_softsign(x, l.δ), st

#===========================================================#
# Multiresolution hash encoding based on
# https://github.com/cheind/pure-torch-ngp
# torchngp/modules/encoding.py
#===========================================================#

""" Cubic hermite interpolation """
interp_cubic( f0, f1, w) = @. f0 + (f1 - f0) * (3 - 2w) * w^2
interp_linear(f0, f1, w) = @. w * (f1 - f0) + f0

function get_boundingbox_indices(
    xyz::AbstractMatrix{T},
    shape::NTuple{D, Ti}
) where{T, D, Ti}
    @assert size(xyz, 1) == D

    # normalize from [-1, 1] to [0, 1]
    xyz = (xyz .+ 1f0) .* 0.5f0

    XYZ   = xyz .* shape
    Ixyz0 = floor.(Int32, XYZ)
    Ixyz1 = ceil.( Int32, XYZ)

    wxyz = XYZ - Ixyz0

    Ixyz0, Ixyz1, wxyz
end

function get_boundingbox(
    xyz::AbstractArray{T},
    shape::NTuple{2, Ti},
    nEmbeddings,
    indexfun,
) where{T, Ti}

    @assert size(xyz, 1) == 2

    Ixyz0, Ixyz1, wxyz = get_boundingbox_indices(xyz, shape)

    Ix0, Iy0 = collect(getindex(Ixyz0, d, :) for d in axes(Ixyz0, 1))
    Ix1, Iy1 = collect(getindex(Ixyz1, d, :) for d in axes(Ixyz1, 1))
    wx , wy  = collect(getindex(wxyz , d, :) for d in axes(wxyz , 1))

    wx , wy  = map(w -> reshape(w, 1, size(w)...), (wx, wy))

    I00 = indexfun(Ix0, Iy0, nEmbeddings, shape)
    I10 = indexfun(Ix1, Iy0, nEmbeddings, shape)
    I01 = indexfun(Ix0, Iy1, nEmbeddings, shape)
    I11 = indexfun(Ix1, Iy1, nEmbeddings, shape)

    (wx, wy), (I00, I10, I01, I11,)
end

function get_boundingbox(
    xyz::AbstractArray{T},
    shape::NTuple{3, Ti},
    nEmbeddings,
    indexfun,
) where{T, Ti}

    @assert size(xyz, 1) == 3

    Ixyz0, Ixyz1, wxyz = get_boundingbox_indices(xyz, shape)

    Ix0, Iy0, Iz0 = collect(getindex(Ixyz0, d, :) for d in axes(Ixyz0, 1))
    Ix1, Iy1, Iz1 = collect(getindex(Ixyz1, d, :) for d in axes(Ixyz1, 1))
    wx , wy , wz  = collect(getindex(wxyz , d, :) for d in axes(wxyz , 1))

    wx , wy , wz  = map(w -> reshape(w, 1, size(w)...), (wx, wy, wz))

    I000 = indexfun(Ix0, Iy0, Iz0, nEmbeddings, shape)
    I100 = indexfun(Ix1, Iy0, Iz0, nEmbeddings, shape)
    I010 = indexfun(Ix0, Iy1, Iz0, nEmbeddings, shape)
    I110 = indexfun(Ix1, Iy1, Iz0, nEmbeddings, shape)
    I001 = indexfun(Ix0, Iy0, Iz1, nEmbeddings, shape)
    I101 = indexfun(Ix1, Iy0, Iz1, nEmbeddings, shape)
    I011 = indexfun(Ix0, Iy1, Iz1, nEmbeddings, shape)
    I111 = indexfun(Ix1, Iy1, Iz1, nEmbeddings, shape)

    (wx, wy, wz), (I000, I100, I010, I110, I001, I101, I011, I111,)
end

#===========================================================#
# index schemes
#===========================================================#

function index_hash(
    Ix::AbstractArray{Ti},
    Iy::AbstractArray{Ti},
    nEmbeddings::Integer,
    shape::NTuple{2, <:Integer},
) where{Ti <: Integer}

    # https://bigprimes.org/
    primes = Int32[1, 481549]
    val = xor.(Ix * primes[1], Iy * primes[2])
    mod1.(val, Ti(nEmbeddings))
end

function index_hash(
    Ix::AbstractArray{Ti},
    Iy::AbstractArray{Ti},
    Iz::AbstractArray{Ti},
    nEmbeddings::Integer,
    shape::NTuple{3, <:Integer},
) where{Ti <: Integer}

    # https://bigprimes.org/
    primes = Int32[1, 481549, 928079]
    val = xor.(Ix * primes[1], Iy * primes[2], Iz * primes[3])
    mod1.(val, Ti(nEmbeddings))
end

function index_grid(
    Ix::AbstractArray{Ti},
    Iy::AbstractArray{Ti},
    nEmbeddings::Integer,
    shape::NTuple{2, <:Integer},
) where{Ti}
    @. (Iy - 1) * shape[1] + Ix
end

function index_grid(
    Ix::AbstractArray{Ti},
    Iy::AbstractArray{Ti},
    Iz::AbstractArray{Ti},
    nEmbeddings::Integer,
    shape::NTuple{3, <:Integer},
) where{Ti}
    @. (Iz - 1) * (shape[1] * shape[2]) + (Iy - 1) * shape[1] + Ix
end

#===========================================================#
# Abstract feature grid types
#===========================================================#

export SpatialHash, SpatialGrid

@concrete struct FeatureGrid{D} <: Lux.AbstractExplicitContainerLayer{(:embedding,)}
    shape
    indexfun
    interpfun
    embedding
end

function SpatialHash(
    out_dims::Integer,
    nEmbeddings::Integer,
    shape::NTuple{D, Integer};
    init_weight = randn32,
    interpfun = interp_linear,
) where{D}

    embedding = Embedding(nEmbeddings => out_dims; init_weight)

    FeatureGrid{D}(shape, index_hash, interpfun, embedding)
end

function SpatialGrid(
    out_dims::Integer,
    shape::NTuple{D, Integer};
    init_weight = randn32,
    interpfun = interp_linear,
) where{D}

    nEmbeddings = prod(shape)
    embedding = Embedding(nEmbeddings => out_dims; init_weight)

    FeatureGrid{D}(shape, index_grid, interpfun, embedding)
end

function (l::FeatureGrid)(xyz::AbstractArray, p, st)
    @assert size(xyz, 1) == length(l.shape)
    wxyz, Ibbox = Zygote.@ignore get_boundingbox(xyz, l.shape, l.embedding.in_dims, l.indexfun)

    l((wxyz, Ibbox), p, st)
end

function (l::FeatureGrid{2})((wxyz, Ibbox)::Tuple, p, st)
    wx, wy = wxyz
    I00, I10, I01, I11 = Ibbox

    f00, _ = l.embedding(I00, p, st)
    f10, _ = l.embedding(I10, p, st)
    f01, _ = l.embedding(I01, p, st)
    f11, _ = l.embedding(I11, p, st)

    fx0 = l.interpfun(f00, f10, wx) # y = 0
    fx1 = l.interpfun(f01, f11, wx) # y = 1
    f   = l.interpfun(fx0, fx1, wy) # interp

    f, st
end

function (l::FeatureGrid{3})((wxyz, Ibbox)::Tuple, p, st)
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

export MultiLevelSpatialHash, precompute_MLH

function MultiLevelSpatialHash(;
    out_dims::Integer = 2,
    nEmbeddings::Integer = 2^14,
    nLevels::Integer = 16,
    min_res::Integer = 16,  # shape of first level
    max_res::Integer = 512, # shape of final level
    init_weight = scale_init(randn32, 1f-4, 0f0),
    interpfun = interp_linear,
)
    gf = exp(log(max_res / min_res) / (nLevels - 1)) # growth factor
    Ns = min_res .* collect(gf ^ l for l in 0:nLevels-1)
    Ns = round.(Int32, Ns)

    println("Generating $(nLevels) spatial hashes with resolutions $(Ns)")

    levels = (;)
    for l in 1:nLevels
        N = Ns[l]
        shape = (N, N, N,)
        layer = if prod(shape) <= nEmbeddings
            SpatialGrid(out_dims, shape; init_weight, interpfun)
        else
            SpatialHash(out_dims, nEmbeddings, shape; init_weight, interpfun)
        end
        levels = (; levels..., Symbol("level$l") => layer)
    end

    Chain(;
        grids = BranchLayer(; levels...),
        vcat  = WrappedFunction(__vcat),
    )
end

__vcat(x::NTuple{N,AbstractArray}) where{N} = vcat(x...)

function precompute_MLH(xyz, MLH)
    ids = ()

    layers = MLH.layers.grids.layers

    for layer in layers
        bbox = get_boundingbox(xyz, layer.shape, layer.embedding.in_dims, layer.indexfun)
        ids = (ids..., bbox)
    end

    NN = Parallel(vcat; layers...)
    NN = Chain(;
        grids = Parallel(vcat; layers...),
    )

    ids, NN
end

#===========================================================#
#
