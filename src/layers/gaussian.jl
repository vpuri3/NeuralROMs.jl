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

@concrete struct Gaussian1D{I<:Integer} <: AbstractLuxLayer
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
