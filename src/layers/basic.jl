#
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
# Periodic BC layer
#======================================================#

"""
Periodic BC layer (For 1D only rn)
based on
- https://github.com/julesberman/RSNG/blob/main/rsng/dnn.py
- https://github.com/Algopaul/ng_embeddings/blob/main/embedded_ng.ipynb
"""
struct PeriodicLayer{I,L,P} <: Lux.AbstractExplicitLayer
    init::I # should be U(0, 2)
    width::L
    periods::P
end

function PeriodicLayer(width::Integer, periods; init = rand32)
    PeriodicLayer(init, width, periods)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::PeriodicLayer)
    (;
        ϕ = l.init(rng, l.width, length(l.periods)),
    )
end

function Lux.initialstates(::Random.AbstractRNG, l::PeriodicLayer)
    T = eltype(l.periods)
    (;
        ω = @. T(2) / [l.periods...]
    )
end

Lux.statelength(l::PeriodicLayer) = length(l.periods)
Lux.parameterlength(l::PeriodicLayer) = l.width * length(l.periods)

function (l::PeriodicLayer)(x::AbstractArray, ps, st::NamedTuple)
    y = @. cospi(st.ω * x + ps.ϕ)
    return y, st
end

#======================================================#
# Gaussian Layer (only 1D for now)
#======================================================#

export GaussianLayer

struct GaussianLayer{F,I} <: Lux.AbstractExplicitLayer
    init::F
    in_dim::I
    out_dim::I
end

function GaussianLayer(in_dim::Integer, out_dim::Integer; init = rand32)
    @assert in_dim == out_dim == 1
    GaussianLayer(init, in_dim, out_dim)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::GaussianLayer)
    T = l.init(rng, 1) |> eltype
    (;
        c  = T[1],
        x̄  = l.init(rng, l.in_dim),
        σi = l.init(rng, l.in_dim),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, l::GaussianLayer)
    T = l.init(rng, 1) |> eltype
    (;
        oneby2 = T[-0.5],
    )
end

function (l::GaussianLayer)(x::AbstractArray, ps, st::NamedTuple)
    z = @. (x - ps.x̄) * ps.σi
    # y = ps.c * _rbf(z)
    y = @. ps.c * exp(st.oneby2 * z^2)
    return y, st
end

# @inline _rbf(x) = @. exp(-x^2)
# function ChainRulesCore.rrule(::typeof(_rbf), x)
#     T = eltype(x)
#     y = _rbf(x)
#     @inline ∇_rbf(ȳ) = ChainRulesCore.NoTangent(), @fastmath(@. -T(2) * x * y * ȳ)
#
#     y, ∇_rbf
# end

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
#
