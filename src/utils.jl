#
#===========================================================#
fastify(act) = act
fastify(::typeof(tanh)) = Lux.tanh_fast
fastify(::typeof(Lux.sigmoid)) = Lux.sigmoid_fast

#===========================================================#

function pad_array(x::AbstractArray{<:Any, N}, dims::NTuple{N}) where{N}
    X = similar(x, dims)
    fill!(X, 0)

    idx = map(d -> 1:d, size(x))
    X[idx...] .= x

    X
end

function ChainRulesCore.rrule(::typeof(pad_array), x, dims)

    function pad_array_pullback(X̄)
        idx = map(d -> 1:d, size(x))

        return NoTangent(), view(X̄, idx...), NoTangent()
    end

    return pad_array(x, dims), pad_array_pullback
end

#===========================================================#

""", FUNC_CACHE_PREFER_NONE
    _ntimes(x, (Nx, Ny)): x [L, B] --> [L, Nx, Ny, B]

Make `Nx ⋅ Ny` copies of the first dimension and store it in the following
dimensions. Works for any `(Nx, Ny, ...)`.
"""
function _ntimes(x::AbstractMatrix, Ns::Union{Int,NTuple{D,Int}}) where{D}
    L, B = size(x)
    y = repeat(x; outer = (prod(Ns), 1))
    reshape(y, L, Ns..., B)
end

function ChainRulesCore.rrule(::typeof(_ntimes), x, N)
    y = _ntimes(x, N)

    function ntimes_pb(ȳ)
        x̄ = sum(ȳ, dims = 2)
        x̄ = reshape(x̄, size(x))
        NoTangent(), x̄, NoTangent()
    end

    y, ntimes_pb
end

#===========================================================#

c_glorot_uniform(dims...) = Lux.glorot_uniform(dims...) + Lux.glorot_uniform(dims...) * im
# Lux.glorot_uniform(rng::AbstractRNG, ::Type{<:Real}, dims...) = Lux.glorot_uniform(rng, dims...)
Lux.glorot_uniform(rng::AbstractRNG, ::Type{<:Complex}, dims...) = c_glorot_uniform(rng, dims...)

#===========================================================#

function init_siren(
    rng::AbstractRNG,
    ::Type{T},
    dims::Integer...;
    scale::Real = 1
) where{T <: Real}

    scale = T(scale) * sqrt(T(24) / T(_nfan(dims...)[1]))
    return (rand(rng, T, dims...) .- T(1//2)) * scale
end

init_siren(dims::Integer...; kw...) = init_siren(Random.default_rng(), Float32, dims...; kw...)
init_siren(rng::AbstractRNG, dims::Integer...; kw...) = init_siren(rng, Float32, dims...; kw...)
init_siren(::Type{T}, dims::Integer...; kw...) where{T<:Real} = init_siren(Random.default_rng(), T, dims...; kw...)

function scaled_siren_init(scale::Real)
    (args...; kwargs...) -> init_siren(args...; kwargs..., scale)
end

function scale_init(init, scale::Real, shift::Real)
    function scale_init_internal(args...; kwargs...)
        x = init(args...; kwargs...)
        T = eltype(x)
        (init(args...; kwargs...) .* T(scale)) .+ T(shift)
    end
end
#===========================================================#
function remake_ca(
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, AbstractArray},
    st::NamedTuple,
)
    if p isa NamedTuple
        NN, p, st
    else
        p = ComponentArray(getdata(p) |> copy, getaxes(p))
        NN, p, st
    end
end
#===========================================================#

_getdata(loader::MLUtils.DataLoader) = loader.data
_getdata(loader::CuIterator) = _getdata(loader.batches)
#===========================================================#
#
