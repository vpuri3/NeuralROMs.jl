#======================================================#
# Periodic BC layer
#======================================================#

export PeriodicLayer

"""
x -> sin(π⋅x/L)

Works when input is symmetric around 0, i.e., x ∈ [-1, 1).
If working with something like [0, 1], use cosines instead.
"""
@concrete struct PeriodicLayer <: AbstractLuxLayer
    idxs
    periods
end

Lux.initialstates(::Random.AbstractRNG, l::PeriodicLayer) = (; k = 1 ./ l.periods)

function (l::PeriodicLayer)(x::AbstractMatrix, ps, st::NamedTuple)
    other_idxs = ChainRulesCore.@ignore_derivatives setdiff(axes(x, 1), l.idxs)
    y = vcat(x[other_idxs, :], @. sinpi(st.k * x[l.idxs, :]))
    y, st
end

function Base.show(io::IO, l::PeriodicLayer)
    println(io, "PeriodicLayer($(l.idxs), $(l.periods))")
end

#======================================================#
#
