#
#======================================================#
# Tanh Kernels 1D
#
# Q. parameterize (x̄, w) or (x0, x1) directly?
# - x̄ ,  w: compact support, fixed orientation 
# - x0, x1: orientation switches if x0 < x1. Why is that a problem?
#   potentially gives more flexibility with gradient descent ??
#
# Q. Adding Tanh kernel with x1=x0 (resultant zero func) and fitting
#    with GD as a form of mesh refinement.
#    Test this idea out with initializing w as [1, 0, ..., 0]
#    and seeing if the last zeros change.
#    How does it compare with setting c = [1, 0, ..., 0]
#
# Q. We can also switch between the two during training/ online solve
#
# - parameterize x0, x1 and force them to be in [-1, 1]
# - try gradient boosting type approach

#======================================================#
export TK1D

@concrete struct TK1D{I <: Integer} <: AbstractLuxLayer
	n::I   # active kernels
	N::I   # total kernels
	domain # (xmin, xmax)
	periodic::Bool
	T::Type{<:Real}
end

function TK1D(n, N; domain = (-1f0, 1f0), periodic = true, T = Float32)
    @assert length(domain) == 2
	@assert 0 ≤ n ≤ N
    TK1D(n, N, T.(domain), periodic, T)
end

function Lux.initialstates(::Random.AbstractRNG, l::TK1D)
	# can add additional mask for frequency / polynomial order

	half = l.T[0.5]
	mask = zeros(Bool, l.N)
	mask[1:l.n] .= true
	(; half, mask)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::TK1D)
	left, right = l.domain
	middle = (left + right) / 2
	width  = (right - left) / 2

	# initialize kernels to cover the entire domain ± 10% noise
	# if this doesn't work. try splitting em up in the domain later
	# but i think global initialization + splitting during training should work
	x̄ = randn(rng, l.T, l.N) * (width / 10) .+ middle
	w = randn(rng, l.T, l.N) * (width / 10) .+ width

	# steepness ω ∈ [5w, 15w]
	ω0 = rand(rng, l.T, l.N) * (width * 10) .+ (width * 5)
	ω1 = rand(rng, l.T, l.N) * (width * 10) .+ (width * 5)

	# magnitude ~1 after summing
	c = (rand(rng, l.T, l.N) .+ l.T(0.5)) / l.n

    # global shift
    b = zeros(l.T, 1)

    (; x̄, w, ω0, ω1, b, c)
end

function (l::TK1D)(x::AbstractMatrix{T}, ps, st::NamedTuple) where{T}
	mask = st.mask
	x̄  = ps.x̄[mask]
	w  = ps.w[mask]
	ω0 = ps.ω0[mask]
	ω1 = ps.ω1[mask]
	b  = ps.b
	c  = ps.c[mask]

	tanh_kernel1d(x, x̄, w, ω0, ω1, b, c, st)
end
#======================================================#

function tanh_kernel1d(x, x̄, w, ω0, ω1, b, c, st)
	# kernal expanse
	x0 = @. x̄ - w
	x1 = @. x̄ + w

    # make kernels
    y0 = @. tanh_fast(ω0 * (x - x0))
    y1 = @. tanh_fast(ω1 * (x - x1))
	y  = @. (y0 - y1) * c * st.half # move below (later)

    # sum kernels
    y = sum(y; dims = 1)

    # add global shift
    y = @. y + b

    y, st
end

function _movetoend(x::AbstractVector, mask::AbstractVector{Bool})
	@assert length(mask) == size(x, 1)
	return vcat(x[.!mask], x[mask])
end

function prune_kernels(NN::TK1D, p, st, mask::AbstractVector{Bool})
	# handle trivial case
	n = sum(mask)
	if n == 0
		return NN, p, st
	end

	# update NN
	@set! NN.n = max(0, NN.n - n)

	# update mask
	st = deepcopy(st)
	st.mask[mask] .= false

	# move dead kernels to the end
	@set! st.mask = _movetoend(st.mask, mask)

	@set! p.x̄  = _movetoend(p.x̄ , mask)
	@set! p.w  = _movetoend(p.w , mask)
	@set! p.ω0 = _movetoend(p.ω0, mask)
	@set! p.ω1 = _movetoend(p.ω1, mask)
	@set! p.c  = _movetoend(p.c , mask)
	# for k in (:x̄, :w, :ω0, :ω1, :c)
	# 	@eval @set! p.$(k) = _movetoend(p.$(k), mask)
	# end

	return NN, p, st
end

function clone_kernels(NN::TK1D, p, st, id0)
	# handle trivial case
	if isempty(id0)
		return NN, p, st, 1:0
	end

	# find indices
	i0 = sum(st.mask)
	i1 = i0 + length(id0)
	if i1 > NN.N
		i1 = NN.N
		id0 = id0[1:(i1-i0)]
	end

	id1 = (i0+1):i1
	@assert all(iszero, st.mask[id1]) "$(id1), $(st.mask[id1])"

	# update NN
	@set! NN.n = NN.n + (i1 - i0)

	# update mask
	st = deepcopy(st)
	st.mask[id1] .= true

	# clone params
	@set! p.x̄[id1]  = p.x̄[id0]
	@set! p.w[id1]  = p.w[id0]
	@set! p.ω0[id1] = p.ω0[id0]
	@set! p.ω1[id1] = p.ω1[id0]
	@set! p.c[id1]  = p.c[id0]
	# for k in (:x̄, :w, :ω0, :ω1, :c)
	# 	@eval p.$(k)[id1] = p.$(k)[id0]
	# end

	return NN, p, st, id1
end

function split_kernels(NN::TK1D, p, st, id0)
	# handle trivial case
	if isempty(id0)
		return NN, p, st, 1:0
	end

	NN, p, st, id1 = clone_kernels(NN, p, st, id0)

	if length(id1) != length(id0)
		id0 = id0[1:length(id1)]
	end

	# exact split
	halfw = p.w[id0] ./ 2
	ω_min = min.(p.ω0[id0], p.ω1[id0])

	p.x̄[id0] .= p.x̄[id0] - halfw # left side
    p.x̄[id1] .= p.x̄[id1] + halfw # right side

    p.w[id0] .= halfw
    p.w[id1] .= halfw

    p.ω0[id1] .= ω_min
    p.ω1[id0] .= ω_min

	return NN, p, st, id1
end
#======================================================#
#
