#======================================================#
# Periodic BC layer
#======================================================#

export PeriodicLayer

"""
x -> sin(π⋅x/L)

Works when input is symmetric around 0, i.e., x ∈ [-1, 1).
If working with something like [0, 1], use cosines instead.
"""
@concrete struct PeriodicLayer <: Lux.AbstractExplicitLayer
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
# 1D Tanh kernel layer
#======================================================#

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

export TanhKernel1D

@concrete struct TanhKernel1D{I<:Integer} <: Lux.AbstractExplicitLayer
    in_dim::I
    out_dim::I
    num_kernels::I
    domain
    linear::Bool
    periodic::Bool
    T
end

function TanhKernel1D(
    in_dim::Integer,
    out_dim::Integer,
    num_kernels::Integer;
    T = Float32,
    domain = [-1, 1],
    order::Integer = 0,
    periodic::Bool = false,
)
    @assert length(domain) == 2
    @assert in_dim == out_dim == 1

    TanhKernel1D(in_dim, out_dim, num_kernels, T.(domain), periodic, T)
end

function Base.show(io::IO, l::TanhKernel1D)
    println(io, "TanhKernel1D($(l.in_dim), $(l.out_dim), $(l.num_kernels))")
end

function Lux.initialstates(rng::Random.AbstractRNG, l::TanhKernel1D)
    (;
        half = l.T[0.5],

        # xdom0 = l.T[l.domain[1]],
        # xdom1 = l.T[l.domain[2]],
        # xmean = l.T[0.5 * (l.domain[1] + l.domain[2])],
        # xspan = l.T[l.domain[2] - l.domain[1]],
    )
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::TanhKernel1D)

    # # principled initialization
    # x0, x1 = l.domain
    # xspan = (x1 - x0) / l.num_kernels
    # x̄ = LinRange(x0 + xspan / 2, x1 - xspan / 2, l.num_kernels) .|> l.T
    # w = zeros(l.num_kernels)                                    .|> l.T
    # c = rand32(rng, l.num_kernels) .* 2 .- 1                    .|> l.T
    # w[l.num_kernels ÷ 2] = 1

    # [1, 0, ..., 0]
    x̄ = zeros(l.T, l.num_kernels)
    w = ones( l.T, l.num_kernels)
    c = fill(l.T(1), l.num_kernels)

    w[2:end] .= 0
    x̄[2:end] .+= randn(rng, l.num_kernels - 1) * 10^(-2)

    #------------------------------#

    # steepness
    ω0 = fill(l.T(10), l.num_kernels)
    ω1 = fill(l.T(10), l.num_kernels)

    # global shift
    b = zeros(l.T, 1)

    (; x̄, w, ω0, ω1, b, c)
end

function (l::TanhKernel1D)(x::AbstractMatrix{T}, ps, st::NamedTuple) where{T}
    x0 = @. ps.x̄ - ps.w
    x1 = @. ps.x̄ + ps.w

    # if l.periodic
    # end

    # make kernels
    y0 = @. tanh_fast(ps.ω0 * (x - x0))
    y1 = @. tanh_fast(ps.ω1 * (x - x1))
    y  = @. (y0 - y1) * ps.c * st.half # (move to below)

    # sum kernels
    y = sum(y; dims = 1)

    # add global shift
    y = @. y + ps.b

    y, st
end

export split_TanhKernel1D

function split_TanhKernel1D(
    NN::TanhKernel1D,
    p,
    st::NamedTuple;
    rng::AbstractRNG = Random.default_rng(),
    debug::Bool = false,
)
    # Kernels
    # [1, 2, 3] -> [_1, _2, _3, 1_, 2_, 3_]

    Nk = NN.num_kernels
    NN_ = @set NN.num_kernels = 2 * Nk
    p_, st_ = Lux.setup(copy(rng), NN_)
    p_ = ComponentArray(p_)

    halfw = p.w ./ 2
    ω_min = min.(p.ω0, p.ω1)

    p_.x̄[begin:Nk] .= p.x̄ - halfw
    p_.x̄[Nk+1:end] .= p.x̄ + halfw

    p_.w[begin:Nk] .= halfw
    p_.w[Nk+1:end] .= halfw

    p_.ω0[begin:Nk] .= p.ω0
    p_.ω0[Nk+1:end] .= ω_min

    p_.ω1[begin:Nk] .= ω_min
    p_.ω1[Nk+1:end] .= p.ω1

    p_.b .= p.b

    p_.c[begin:Nk] .= p.c
    p_.c[Nk+1:end] .= p.c

    if debug
        N = 1000
        x = LinRange(NN.domain..., N)
        x = reshape(x, 1, N) .|> NN.T

        u  =  NN(x,  p,  st)[1]
        u_ = NN_(x, p_, st_)[1]

        e = sum(abs2, u - u_) / length(N)

        @assert e < eps(NN.T) "Got error $e"
    end

    NN_, p_, st_
end

export merge_TanhKernel1D

"""
Input should be:
`((NN1, p1, st1), (NN2, p2, st2), ...)`
where each `NN` is a `TanhKernel1D` and `p`, `st` its parameters and states
"""
function merge_TanhKernel1D(
    models;
    rng::AbstractRNG = Random.default_rng(),
    debug::Bool = false,
)
    if isempty(models)
        @error "models is empty: $(models)"
    end

    if length(models) == 1
        return models[1]
    end

    NNs = map(x -> x[1], models)
    pps = map(x -> x[2], models)

    Nk = sum(x -> x.num_kernels, NNs)

    NN = NNs[1]
    NN = @set NN.num_kernels = Nk
    p, st = Lux.setup(copy(rng), NN)
    p = ComponentArray(p)

    # for x in (:x̄, :w, :c, :ω0, :ω1)
    #     dst = getproperty(p, x)
    #
    #     src = Base.Fix2(getproperty, x).(pps)
    #     src = vcat(src...)
    #
    #     copy!(dst, src)
    # end

    p.x̄ .= vcat(map(p -> p.x̄, pps)...)
    p.w .= vcat(map(p -> p.w, pps)...)
    p.c .= vcat(map(p -> p.c, pps)...)

    p.ω0 .= vcat(map(p -> p.ω0, pps)...)
    p.ω1 .= vcat(map(p -> p.ω1, pps)...)

    p.b .= sum(p -> p.b, pps)

    if debug
        N = 1000
        x = LinRange(NN.domain..., N)
        x = reshape(x, 1, N) .|> NN.T

        u  = NN(x,  p,  st)[1]
        _u = sum(m -> m[1](x, m[2], m[3])[1], models)

        e = sum(abs2, u - _u) / length(N)

        @assert e < eps(NN.T) "Got error $e"
    end

    NN, p, st
end
#======================================================#
#
