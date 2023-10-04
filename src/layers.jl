#
"""
$SIGNATURES

Assumes channel dimension is 1
"""
function PermutedBatchNorm(c, num_dims) # assumes channel_dim = 1

    perm0 = ((2:num_dims-1)..., 1, num_dims)
    perm1 = (num_dims-1, (1:num_dims-2)..., num_dims)

    # Chain(
    #     Base.Fix2(PermutedDimsArray, perm0),
    #     BatchNorm(c),
    #     Base.Fix2(PermutedDimsArray, perm1),
    # )
    Chain(
        Base.Fix2(permutedims, perm0),
        BatchNorm(c),
        Base.Fix2(permutedims, perm1),
    )
end

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
#
