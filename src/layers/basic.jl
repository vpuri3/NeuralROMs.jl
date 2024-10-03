#
#======================================================#
# Hyper Network
#======================================================#

struct HyperNet{W <: AbstractLuxLayer, C <: AbstractLuxLayer, A} <:
       Lux.AbstractLuxContainerLayer{(:weight_generator, :evaluator)}
    weight_generator::W
    evaluator::C
    ca_axes::A
end

function HyperNet(
    weight_gen::AbstractLuxLayer,
    evaluator::AbstractLuxLayer
)
    rng = Random.default_rng()
    ca_axes = Lux.initialparameters(rng, evaluator) |> ComponentArray |> getaxes

    return HyperNet(weight_gen, evaluator, ca_axes)
end

function Lux.initialparameters(rng::AbstractRNG, hn::HyperNet)
    return (;
        weight_generator = Lux.initialparameters(rng, hn.weight_generator),
    )
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
# Permute Layer
#======================================================#

function PermuteLayer(perm::NTuple{D, Int}) where{D}
    WrappedFunction(Base.Fix2(permutedims, perm))
end

#======================================================#
# PermutedBatchNorm
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
# SPLIT ROWS
#======================================================#

"""
SplitRows

Split rows of ND array, into `Tuple` of ND arrays.
"""
struct SplitRows{T} <: AbstractLuxLayer
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
#
