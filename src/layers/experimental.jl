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
# 1D Gaussian Layer
#======================================================#

export Gaussian1D

# Assumes x ∈ [-1, 1], u ∈ [ 0, 1]
# What normalization to assume for u?
# Would [0, 1] work for Gabor (must have -ve vals )
# should initialization of c include negative values?
# no if Gabor freqs can take care of that??
###
# QUESTIONS
###
# - is it better to optimize 1/σ ? No based on loss_landscape.jl
####
# PRUNING/ CULLING CRITERIA:
####
# - if σ too small, set c = 0 σ = 1.
# - if c too small, set c = 0, σ = 1.
# - merge Gaussians if x̄ close?? Check Gaussian splatting paper.
# - refinement: add Gaussians if error/residual too large somewhere.
####
# Q: HOW TO CAPTURE SHOCKS?
####
# A: Have a σleft and σr trainable and cacluate sigma as
#      σ = scaled_tanh(x, σleft, σright, ω, x̄)
#
#              /|
#             / |
#            /  |
#           /   |
# _________/    |_______
#
####
# EXTENSION TO 2D:
####
# For 2D Gaussian/Gabor, let the sinusodal be in the periodic (angular) direction
# of the Gaussian. E.g. https://en.wikipedia.org/wiki/Gabor_filter
# [Gabor Splatting for High-Quality Gigapixel Image Representations]

@concrete struct Gaussian1D{I<:Integer} <: Lux.AbstractExplicitLayer
    in_dim::I
    out_dim::I
    num_gauss::I
    num_freqs::I

    T
    domain
    periodic::Bool

    σmin
    σfactor
    σsplit::Bool
    σinvert::Bool

    train_freq::Bool
end

function Gaussian1D(
    in_dim::Integer,
    out_dim::Integer,
    num_gauss::Integer,
    num_freqs::Integer;
    T = Float32,
    domain = [-1, 1],
    periodic::Bool = false,
    σmin = T(1e-3), # consult find_sigmamin.jl. Could be higher...
    σfactor = T(4),
    σsplit::Bool = false,
    σinvert::Bool = false,
    train_freq::Bool = true,
)
    @assert in_dim == 1
    @assert T ∈ (Float32, Float64)

    Gaussian1D(
        in_dim, out_dim, num_gauss, num_freqs,
        T, T.(domain), periodic, T(σmin), T(σfactor),
        σsplit, σinvert, train_freq,
    )
end

function init_ω_ϕ(l::Gaussian1D)

    # frequencies
    ω = range(0, l.num_freqs-1)
    ω = reshape(ω, 1, l.num_freqs) .|> l.T

    # phase shifts ∈ [-1, 1]
    ϕ = zeros(l.num_freqs)
    ϕ = reshape(ϕ, 1, l.num_freqs) .|> l.T

    ω = repeat(ω, 1, 1, l.num_gauss)
    ϕ = repeat(ϕ, 1, 1, l.num_gauss)

    ω, ϕ
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::Gaussian1D)

    x0, x1 = l.domain
    span = (x1 - x0) / l.num_gauss

    # global shift
    b = zeros32(rng, 1) .|> l.T

    # scaling
    c0 = l.T(1 / l.num_freqs)
    c = rand(rng, 1, l.num_freqs, l.num_gauss) .* 2 .- 1 .|> l.T

    # mean
    x̄ = LinRange(x0 + span/2, x1 - span/2, l.num_gauss) 
    x̄ = reshape(x̄, 1, 1, l.num_gauss) .|> l.T

    # variance
    σ = fill(span/l.σfactor, (1, 1, l.num_gauss)) .|> l.T

    x̄ = repeat(x̄, 1, l.num_freqs)
    σ = repeat(σ, 1, l.num_freqs)

    ps = (; b, c, x̄)

    if l.σinvert
        σi = inv.(σ)
        if l.σsplit
            ps = (; ps..., σil = σi, σir = σi, w = l.T[50])
        else
            ps = (; ps..., σi)
        end
    else
        if l.σsplit
            ps = (; ps..., σl = σ, σr = σ, w = l.T[50])
        else
            ps = (; ps..., σ)
        end
    end

    if l.train_freq
        ω, ϕ  = init_ω_ϕ(l)
        ps = (; ps..., ω, ϕ)
    end

    ps
end

function Lux.initialstates(rng::Random.AbstractRNG, l::Gaussian1D)
    σϵ = l.T[l.σmin]
    two = l.T[2]
    minushalf = l.T[-0.5]

    st = (; σϵ, two, minushalf)

    if !l.train_freq
        ω, ϕ  = init_ω_ϕ(l)
        st = (; st..., ω, ϕ)
    end

    # periodic kernel in
    # https://www.cs.toronto.edu/~duvenaud/cookbook/index.html
    if l.periodic
        ω_domain = l.T[1 ./ (l.domain[2] - l.domain[1])]
        st = (; st..., ω_domain)
    end

    st
end

function (l::Gaussian1D)(x::AbstractMatrix{T}, ps, st::NamedTuple) where{T}

    # reshape for broadcasting
    x_re = reshape(x, l.in_dim, 1, 1, size(x, 2))   # [D, 1, 1, K]

    # get ω, ϕ
    ω, ϕ = if l.train_freq
        ps.ω, ps.ϕ
    else
        st.ω, st.ϕ
    end

    # rescale with (x̄, σ)
    xd = @. x_re - ps.x̄ 

    if l.periodic
        xd = @. sinpi(xd * st.ω_domain)
    end

    z = if l.σinvert                        # [1, 1, Ng, K]
        σi = if l.σsplit
            σil = @. abs(ps.σil)
            σir = @. abs(ps.σir)
            σ = scaled_tanh(xd, σil, σir, ps.w, zero(T))
        else
            @. abs(ps.σi)
        end
        @. xd * σi
    else
        σ = if l.σsplit
            σl = @. abs(ps.σl) + st.σϵ
            σr = @. abs(ps.σr) + st.σϵ
            σ = scaled_tanh(xd, σl, σr, ps.w, zero(T))
            # σ = scaled_tanh(x_re, σl, σr, ps.w, ps.x̄)
        else
            @. abs(ps.σ) + st.σϵ
        end
        @. xd / σ
    end

    # apply Gaussian, sinusodal
    y_gauss = @. exp(st.minushalf * z^2)    # [1, 1 , Ng, K]
    y_sin   = @. cospi(ω * z + ϕ)           # [1, Nf,  1, K]

    # scale, multiply, add
    y = @. ps.c * y_gauss * y_sin
    y = sum(y; dims = 2:3)                  # [D, 1, 1, K]
    y = reshape(y, (l.out_dim, size(x, 2))) # [D, K]

    # add global shift
    y = @. y + ps.b

    return y, st
end

function scaled_tanh(x, a, b, w, x̄)
    u = @. tanh_fast(w * (x - x̄)) # [-1, 1]
    scale = @. (b - a) / 2
    shift = @. (b + a) / 2
    @. scale * u + shift
end

# @inline _rbf(x) = @. exp(-x^2)
# function ChainRulesCore.rrule(::typeof(_rbf), x)
#     T = eltype(x)
#     y = _rbf(x)
#     @inline ∇_rbf(ȳ) = ChainRulesCore.NoTangent(), @fastmath(@. -T(2) * x * y * ȳ)
#
#     y, ∇_rbf
# end
#
#======================================================#
# 1D Tanh kernel layer
#======================================================#

# Q. parameterize x̄, w or x0, x1 directly?
# - x̄ ,  w: compact support, orientation fixed
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
    periodic::Bool
    T
end

function TanhKernel1D(
    in_dim::Integer,
    out_dim::Integer,
    num_kernels::Integer;
    T = Float32,
    domain = [-1, 1],
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
    # debug::Bool = false,
)
    # Kernels
    # [1, 2, 3] -> [_1, _2, _3, 1_, 2_, 3_]

    Nk = NN.num_kernels
    NN_ = @set NN.num_kernels = 2 * Nk
    p_, st_ = Lux.setup(copy(rng), NN_)

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

    # if debug
    #     N = 1000
    #     x = LinRange(NN.domain..., N)
    #     x = reshape(x, 1, N) .|> NN.T
    #
    #     u  =  NN(x,  p,  st)[1]
    #     u_ = NN_(x, p_, st_)[1]
    #
    #     e = sum(abs2, u - u_) / length(N)
    #
    #     @assert e < eps(NN.T) "Got error $e"
    # end

    NN_, p_, st_
end

#======================================================#
#
