#
#======================================================#
struct DecoderWeightDecay{Tg, Ta} <: Optimisers.AbstractRule
    gamma::Tg
    ca_axes::Ta
end

Optimisers.init(o::DecoderWeightDecay, x::AbstractArray) = nothing

function Optimisers.apply!(
    o::DecoderWeightDecay,
    state,
    x::AbstractArray{T},
    dx,
) where{T}
    γ = T(o.gamma)

    # dx′ = @lazy dx + γ * x
    # return state, dx′

    dx  = Base.materialize(dx)
    dx′ = deepcopy(dx)

    x_ca   = ComponentArray(x  , o.ca_axes)
    dx′_ca = ComponentArray(dx′, o.ca_axes)

    dx′_ca.decoder += γ * x_ca.decoder

    return state, getdata(dx′)
end
#======================================================#
