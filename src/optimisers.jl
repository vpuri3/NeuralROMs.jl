#
#======================================================#
struct DecoderWeightDecay{Tg, Ta} <: Optimisers.AbstractRule
    lambda::Tg
    ca_axes::Ta
end
Optimisers.init(::DecoderWeightDecay, ::AbstractArray) = nothing
Base.show(io::IO, ::DecoderWeightDecay) = print(io, "DecoderWeightDecay())")

function Optimisers.apply!(
    o::DecoderWeightDecay,
    state,
    x::AbstractArray{T},
    dx,
) where{T}
    γ = T(o.lambda)

    ### Optimisers.WeightDecay
    # dx′ = @lazy dx + γ * x
    # return state, dx′

    ### in place
    dx = Base.materialize(dx)
    x_ca  = ComponentArray(x , o.ca_axes)
    dx_ca = ComponentArray(dx, o.ca_axes)
    dx_ca.decoder += γ * x_ca.decoder
    return state, dx

    ### out of place
    # dx  = Base.materialize(dx)
    # dx′ = deepcopy(dx)
    # x_ca   = ComponentArray(x  , o.ca_axes)
    # dx′_ca = ComponentArray(dx′, o.ca_axes)
    # dx′_ca.decoder += γ * x_ca.decoder # in-place updates dx′
    # return state, getdata(dx′)
end

#======================================================#
struct IdxWeightDecay{Tg, Ti} <: Optimisers.AbstractRule
    lambda::Tg
    index::Ti
end
Optimisers.init(::IdxWeightDecay, ::AbstractArray) = nothing
Base.show(io::IO, o::IdxWeightDecay) = print(io, "IdxWeightDecay($(o.lambda), $(length(o.index)))")

function Optimisers.apply!(
    o::IdxWeightDecay,
    state,
    x::AbstractArray{T},
    dx,
) where{T}
    γ = T(o.lambda)
    i = o.index

    ### in place
    dx = Base.materialize(dx)
    dx[i] += γ * x[i]
    return state, dx

    ### out of place
    # dx  = Base.materialize(dx)
    # dx′ = deepcopy(dx)
    # dx′[i] += γ * x[i]
    # return state, dx′ # getdata(dx′)
end
#======================================================#
