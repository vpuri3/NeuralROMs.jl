#
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

c_glorot_uniform(dims...) = Lux.glorot_uniform(dims...) + Lux.glorot_uniform(dims...) * im
Lux.glorot_uniform(rng::AbstractRNG, ::Type{<:Real}, dims...) = Lux.glorot_uniform(rng, dims...)
Lux.glorot_uniform(rng::AbstractRNG, ::Type{<:Complex}, dims...) = c_glorot_uniform(rng, dims...)
#
