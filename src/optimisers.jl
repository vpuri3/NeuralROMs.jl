#
#======================================================#
struct PartWeightDecay{Tg, Ta, Tn} <: Optimisers.AbstractRule
    gamma::Tg
    ca_axes::Ta
    name::Tn
end

PartWeightDecay(gamma, ca_axes) = PartWeightDecay(gamma, ca_axes, "")

Optimisers.init(o::PartWeightDecay, x::AbstractArray) = nothing

function Optimisers.apply!(
    o::PartWeightDecay,
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

Base.show(io::IO, o::PartWeightDecay) = print(io, "PartWeightDecay($(o.name)))")
#======================================================#
