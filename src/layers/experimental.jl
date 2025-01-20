#======================================================#
# Periodic BC layer
#======================================================#

export PeriodicLayer

#
# https://github.com/julesberman/CoLoRA/blob/main/colora/layers.py#L9
#

"""
x -> sin(π⋅x/L)

Works when input is symmetric around 0, i.e., x ∈ [-1, 1).
If working with something like [0, 1], use cosines instead.
"""
@concrete struct PeriodicLayer <: AbstractLuxLayer
    idxs
    periods
	width
end

function Lux.initialstates(::Random.AbstractRNG, l::PeriodicLayer)
	T = Float32
	(;
		k = T(2) ./ T.(l.periods),
        b = T.(LinRange(0f0, 2f0, l.width)),
	)
end

# function Lux.initialparameters(rng::Random.AbstractRNG, l::PeriodicLayer)
# 	d = length(l.periods)
# 	w = l.width
#
# 	(;
# 		a = ones32(rng, w, d),
# 		b = rand32(rng, w, d) * 2 .- 1,
#       b = T.(LinRange(0f0, 2f0, l.width)),
# 		c = zeros32(rng, w, d),
# 	)
# end

function (l::PeriodicLayer)(x::AbstractMatrix, ps, st::NamedTuple)
	# other indices
    io = ChainRulesCore.@ignore_derivatives setdiff(axes(x, 1), l.idxs)
	yo = x[io, :]

	# periodic indices
	w = l.width
	d = length(l.periods)
	k = reshape(st.k, (1, d))
    b = reshape(st.b, (w, 1)) # ps.b

	xp = x[l.idxs, :]
	xp = reshape(xp, 1, size(xp)...)
    yp = @. sinpi(k * xp + b)
	# yp = @. ps.a * xp + ps.c
	# yp = dropdims(sum(yp; dims = 1); dims = 1) ./ d
	yp = reshape(yp, w * d, :)

	# combine
    y = vcat(yo, yp)
    y, st
end

function Base.show(io::IO, l::PeriodicLayer)
	println(io, "PeriodicLayer($(l.idxs), $(l.periods), $(l.width))")
end
#======================================================#
#
