#
#======================================================#
# Tanh Kernels 1D
#
#  - Adding Tanh kernel with x1=x0 (resultant zero func) and fitting
#    with GD as a form of mesh refinement.
#    Test this idea out with initializing w as [1, 0, ..., 0]
#    and seeing if the last zeros change.
#    How does it compare with setting c = [1, 0, ..., 0]
#
#  - We can also switch between the two during training/ online solve
#
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
	@assert 0 ≤ n ≤ N
    @assert length(domain) == 2
    @assert T ∈ (Float32, Float64)
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
	@unpack n, N, T = l
	left, right = l.domain
	middle = (left + right) / T(2)
	width  = (right - left) / T(2)

	# split the first n kernels around ± 20% noise
	x̄ = randn(rng, T, N) * (width / T(5))
	w = randn(rng, T, N) * (width / T(5))

	x̄[1:n]  += LinRange(left, right, n + 2)[2:end-1]
	w[1:n] .+= width / T(n + 1)

	# steepness ω ∈ [5w, 15w]
	ω0 = rand(rng, T, N) * (width * 10) .+ (width * 5)
	ω1 = rand(rng, T, N) * (width * 10) .+ (width * 5)

	# magnitude ~1 after summing
	c = (rand(rng, T, N) .+ T(0.5)) / n

    # global shift
    b = zeros(T, 1)

    (; x̄, w, ω0, ω1, b, c)
end

function (l::TK1D)(x::AbstractMatrix, ps, st::NamedTuple)
	mask = st.mask
	x̄  = ps.x̄[mask]
	w  = ps.w[mask]
	ω0 = ps.ω0[mask]
	ω1 = ps.ω1[mask]
	b  = ps.b
	c  = ps.c[mask]

	tanh_kernel1d(x, x̄, w, ω0, ω1, b, c, st)
end

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

#======================================================#
# Gaussian Kernel 1D
#======================================================#
export GK1D

@concrete struct GK1D{T, I <: Integer} <: AbstractLuxLayer
	n::I   # active kernels
	N::I   # total kernels
	domain # (xmin, xmax)
	periodic::Bool
	split::Bool
	σmin::T
end

function GK1D(n, N; domain = (-1f0, 1f0), periodic = true, T = Float32,
	split::Bool = false, σmin::Real = 1f-4,
)
	@assert 0 ≤ n ≤ N
    @assert length(domain) == 2
    @assert T ∈ (Float32, Float64)
	GK1D(n, N, T.(domain), periodic, split, T(σmin))
end

function Lux.initialstates(::Random.AbstractRNG, l::GK1D{T}) where{T}
	σmin = T[l.σmin]
	mask = zeros(Bool, l.N)
	mask[1:l.n] .= true
	minushalf = T[-0.5]

	(; minushalf, mask, σmin)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::GK1D{T}) where{T}
	@unpack n, N = l
	left, right = l.domain
	middle = (left + right) / 2
	width  = (right - left) / 2

	# split the first n kernels around ± 20% noise
	x̄ = randn(rng, T, N) * (width / T(5))
	σ = randn(rng, T, N) * (width / T(5))

	x̄[1:n]  += LinRange(left, right, n + 2)[2:end-1]
	σ[1:n] .+= width / T(n + 1)

	# magnitude ~1 after summing
	c = (rand(rng, T, N) .+ T(0.5)) / n

    # global shift
    b = zeros(T, 1)

	if l.split
		(; x̄, b, c, σ0 = σ, σ1 = σ)
	else
		(; x̄, b, c, σ)
	end
end

function (l::GK1D{T})(x::AbstractMatrix, ps, st::NamedTuple) where{T}
	mask = st.mask
	x̄ = ps.x̄[mask] # [n,]
	c = ps.c[mask]
	b = ps.b
	σ = if l.split
		σ0 = ps.σ0[mask]
		σ1 = ps.σ1[mask]
		scaled_tanh(x, σ0, σ1, T(50), T(0)) # [n, K]
	else
		ps.σ[mask] # [n,]
	end
	σ = @. softplus(σ) + st.σmin

	gaussian_kernel1d(x, x̄, σ, b, c, st)
end

function gaussian_kernel1d(x, x̄, σ, b, c, st)
	z = @. (x - x̄) / σ
    y = @. exp(st.minushalf * z^2)
	y = @. y * c
	y = sum(y; dims = 1)
    y = @. y + b

    y, st
end

function scaled_tanh(x, a, b, w, x̄)
    u = @. tanh_fast(w * (x - x̄)) # [-1, 1]
    scale = @. (b - a) / 2
    shift = @. (b + a) / 2
    @. scale * u + shift
end

#======================================================#
# interface
#======================================================#
const Kernel1DLayer = Union{TK1D, GK1D,}

function evaluate_kernels(NN::Kernel1DLayer, p, st, x::AbstractMatrix)
	ys = []
	for k in 1:NN.n
		NN_ = @set NN.n = 1
		st_ = @set st.mask = zeros(Bool, NN.N)
		st_.mask[k] = true
		y = NN_(x, p, st_)[1]
		push!(ys, y)
	end
	ys
end

function _movetoend(param::AbstractVector, mask::AbstractVector{Bool})
	@assert length(mask) == size(param, 1)
	return vcat(param[.!mask], param[mask])
end

function prune_kernels(NN::Kernel1DLayer, p, st, mask::AbstractVector{Bool})
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

	@show keys(p)

	# update params
	p = deepcopy(p)
	for k in keys(p) # (:x̄, :w, :ω0, :ω1, :c)
		k == :b && continue
		param = getproperty(p, k)
		param .= _movetoend(param, mask)
	end

	return NN, p, st
end

function activate_kernels(NN::Kernel1DLayer, p, st, n::Integer)
	if (NN.n + n) > NN.N
		n = NN.N - NN.n
	end
	id1 = (NN.n+1):(NN.n+n)

	@assert all(iszero, st.mask[id1]) "$(id1), $(st.mask[id1])"

	# update NN
	@set! NN.n = NN.n + n

	# update mask
	st = deepcopy(st)
	st.mask[id1] .= true

	# update params
	p = deepcopy(p)

	return NN, p, st, id1
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
