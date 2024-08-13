#======================================================#
# Multiplicative Filter Networks
# https://openreview.net/pdf?id=OmtmcPkkhT
# https://github.com/boschresearch/multiplicative-filter-networks/tree/main
#======================================================#

export FourierMFN, GaborMFN

@concrete struct MFN{I} <: Lux.AbstractExplicitContainerLayer{(:filters, :linears)}
    in_dim::I
    hd_dim::I
    out_dim::I

    filters <: NamedTuple
    linears <: NamedTuple
end

function (l::MFN)(x::AbstractArray, ps, st)
    keysf = keys(st.filters)
    keysl = keys(st.linears)

    kf = keysf[1]
    y, st_f = l.filters[kf](x, ps.filters[kf], st.filters[kf])
    @set! st.filters[kf] = st_f

    for i in 2:length(l.filters)
        kf = keysf[i]
        kl = keysl[i-1]

        y_f, st_f = l.filters[kf](x, ps.filters[kf], st.filters[kf])
        y_d, st_l = l.linears[kl](y, ps.linears[kl], st.linears[kl])

        y = y_f .* y_d

        @set! st.filters[kf] = st_f
        @set! st.linears[kl] = st_l
    end

    kl = keysl[end]
    y, st_l = l.linears[kl](y, ps.linears[kl], st.linears[kl])
    @set! st.linears[kl] = st_l

    return y, st
end

function MFN(
    in_dim::Integer,
    hd_dim::Integer,
    out_dim::Integer,
    filters;
    linear_kws = (;),
    out_kws = (;),
)
    Nf = length(filters)
    linears = Tuple( Dense(hd_dim, hd_dim; linear_kws...) for _ in 1:Nf-1)
    linears = (linears..., Dense(hd_dim, out_dim; out_kws...))

    filters = Lux.__named_tuple_layers(filters...)
    linears = Lux.__named_tuple_layers(linears...)

    MFN(in_dim, hd_dim, out_dim, filters, linears)
end

function FourierMFN(
    in_dim::Integer,
    hd_dim::Integer,
    out_dim::Integer,
    num_filters::Integer = 3;

    init_bias = nothing,
    init_weight = scaled_siren_init(1f1),

    linear_kws = (;),
    out_kws = (;),
)
    pi32 = Float32(pi)

    if isnothing(init_bias)
        init_bias = scale_init(rand32, 2 * pi32, -pi32) # U(-π, π)
    end
    
    if isnothing(init_weight)
        init_weight = scale_init(glorot_uniform, Float32(1/sqrt(num_filters)), Float32(0))
    end
    
    filters = Tuple(
        Dense(in_dim, hd_dim, sin; init_weight, init_bias,)
        for _ in 1:num_filters
    )
    
    MFN(in_dim, hd_dim, out_dim, filters; linear_kws, out_kws)
end

function GaborMFN(
    in_dim::Integer,
    hd_dim::Integer,
    out_dim::Integer,
    num_filters::Integer = 3;

    init_kws = (;),
    linear_kws = (;),
    out_kws = (;),
)
    init_W = scale_init(glorot_uniform, Float32(1/sqrt(num_filters)), Float32(0))

    filters = Tuple(
        GaborLayer(in_dim, hd_dim; init_W, init_kws...)
        for _ in 1:num_filters
    )

    MFN(in_dim, hd_dim, out_dim, filters; linear_kws, out_kws)
end

#======================================================#
# GaborLayer
#======================================================#

export GaborLayer

# @concrete struct GaborLayer{I} <: Lux.AbstractExplicitContainerLayer{(:dense,)}
@concrete struct GaborLayer{I} <: Lux.AbstractExplicitLayer
    in_dim::I
    out_dim::I

    init_W
    init_b

    init_μ
    init_γ
end

function GaborLayer(
    in_dim::Integer,
    out_dim::Integer;
    init_W = nothing, init_b = nothing,
    init_μ = nothing, init_γ = nothing,
)
    pi32 = Float32(pi)
    
    if isnothing(init_W)
        init_W = glorot_uniform
    end

    if isnothing(init_b)
        init_b = scale_init(rand32, 2 * pi32, -pi32) # U(-π, π)
    end

    if isnothing(init_μ)
        init_μ = rand32
    end

    if isnothing(init_γ)
        init_γ = rand32
    end

    GaborLayer(in_dim, out_dim, init_W, init_b, init_μ, init_γ)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::GaborLayer)
    (;
        x̄ = l.init_μ(rng, l.in_dim),
        γ = l.init_γ(rng, l.out_dim),

        bias = l.init_b(rng, l.out_dim),
        weight = l.init_W(rng, l.out_dim, l.in_dim),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, l::GaborLayer)
    T = l.init_W(rng, 1) |> eltype
    (;
        minushalf = T[-0.5],
    )
end

function (l::GaborLayer)(x::AbstractArray, ps, st)

    # should apply softplus to γ

    y_sin = sin.(ps.weight * x .+ ps.bias)
    dist2 = sum(abs2, (x .- ps.x̄); dims = 1)
    y_exp = exp.(st.minushalf * dist2 .* ps.γ)

    y = y_sin .* y_exp

    return y, st
end

#======================================================#
# Gaussian Layer (only 1D for now)
#======================================================#

export GaussianLayer1D

# Assumes x ∈ [-1, 1], u ∈ [ 0, 1]
# What normalization to assume for u?
# Would [0, 1] work for Gabor (must have -ve vals )
# should initialization of c include negative values?
# no if Gabor freqs can take care of that??
###
# Questions
###
# - is it better to optimize 1/σ ? No based on loss_landscape.jl
####
# Pruning/ Culling criteria:
####
# - if σ too small, set c = 0 σ = 1.
# - if c too small, set c = 0, σ = 1.
# - merge Gaussians if x̄ close?? Check Gaussian splatting paper.
# - refinement: add Gaussians if error/residual too large somewhere.
####
# Q: How to capture shocks?
# A: Have a σleft and σr trainable and cacluate sigma as
#      σ = scaled_tanh(x, σleft, σright, ω, x̄)
#
#              /|
#             / |
#            /  |
#           /   |
# _________/    |_______

@concrete struct GaussianLayer1D{I<:Integer} <: Lux.AbstractExplicitLayer
    in_dim::I
    out_dim::I
    num_gauss::I
    num_freqs::I

    T
    domain

    σmin
    σfactor
    σsplit::Bool

    train_freq::Bool
end

function GaussianLayer1D(
    in_dim::Integer,
    out_dim::Integer,
    num_gauss::Integer,
    num_freqs::Integer;
    T = Float32,
    domain = [-1f0, 1f0],
    σmin = T(1e-3), # consult find_sigmamin.jl. Could be higher...
    σfactor = T(4),
    σsplit::Bool = false,
    train_freq::Bool = true,
)
    @assert in_dim == 1
    @assert T ∈ (Float32, Float64)

    GaussianLayer1D(
        in_dim, out_dim, num_gauss, num_freqs,
        T, T.(domain), T(σmin), T(σfactor), σsplit, train_freq,
    )
end

function init_ω_ϕ(l::GaussianLayer1D)

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

function Lux.initialparameters(rng::Random.AbstractRNG, l::GaussianLayer1D)

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

    if l.σsplit
        ps = (; ps..., σl = σ, σr = σ, w = l.T[50])
    else
        ps = (; ps..., σ)
    end

    if l.train_freq
        ω, ϕ  = init_ω_ϕ(l)
        ps = (; ps..., ω, ϕ)
    end

    ps
end

function Lux.initialstates(rng::Random.AbstractRNG, l::GaussianLayer1D)
    σϵ = l.T[l.σmin]
    two = l.T[2]
    minushalf = l.T[-0.5]

    st = (; σϵ, two, minushalf)

    if !l.train_freq
        ω, ϕ  = init_ω_ϕ(l)
        st = (; st..., ω, ϕ)
    end

    # if l.σsplit
    #     w = l.T[50]
    #     st = (;st..., w)
    # end

    st
end

function (l::GaussianLayer1D)(x::AbstractMatrix, ps, st::NamedTuple)

    # reshape for broadcasting
    x_re = reshape(x, l.in_dim, 1, 1, size(x, 2))   # [D, 1, 1, K]

    # get ω, ϕ
    ω, ϕ = if l.train_freq
        ps.ω, ps.ϕ
    else
        st.ω, st.ϕ
    end

    # get σ
    σ = if l.σsplit
        σl = @. abs(ps.σl) + st.σϵ
        σr = @. abs(ps.σr) + st.σϵ
        σ = scaled_tanh(x_re, σl, σr, ps.w, ps.x̄)
    else
        @. abs(ps.σ) + st.σϵ
    end

    # rescale with (x̄, σ)
    z = @. (x_re - ps.x̄) / σ                # [1, 1, Ng, K]

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
    u = @. tanh(w * (x - x̄)) # [-1, 1]
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

# NOTES:
# For 2D Gabor, let the sinusodal be in the periodic (angular) direction
# of the Gaussian. E.g.
# https://en.wikipedia.org/wiki/Gabor_filter
# Gabor Splatting for High-Quality Gigapixel Image Representations

#======================================================#
# Periodic BC layer
#======================================================#

export PeriodicLayer

"""
Periodic BC layer (For 1D only rn)
based on
- https://github.com/julesberman/RSNG/blob/main/rsng/dnn.py
- https://github.com/Algopaul/ng_embeddings/blob/main/embedded_ng.ipynb
"""
struct PeriodicLayer{I,L,P} <: Lux.AbstractExplicitLayer
    init::I # should be U(0, 2)
    width::L
    periods::P
end

function PeriodicLayer(width::Integer, periods; init = rand32)
    PeriodicLayer(init, width, periods)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::PeriodicLayer)
    (;
        ϕ = l.init(rng, l.width, length(l.periods)),
    )
end

function Lux.initialstates(::Random.AbstractRNG, l::PeriodicLayer)
    T = eltype(l.periods)
    (;
        ω = @. T(2) / [l.periods...]
    )
end

Lux.statelength(l::PeriodicLayer) = length(l.periods)
Lux.parameterlength(l::PeriodicLayer) = l.width * length(l.periods)

function (l::PeriodicLayer)(x::AbstractArray, ps, st::NamedTuple)
    y = @. cospi(st.ω * x + ps.ϕ)
    y, st
end

