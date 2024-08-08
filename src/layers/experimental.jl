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
        init_weight = scale_init(glorot_uniform, Float32(1/sqrt(num_filters)), 0)
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

    init_mu = nothing,
    init_gamma = nothing,
    init_bias = nothing,
    init_weight = nothing,

    linear_kws = (;),
    out_kws = (;),
)
    pi32 = Float32(pi)
    
    if isnothing(init_bias)
        init_bias = scale_init(rand32, 2 * pi32, -pi32) # U(-π, π)
    end

    # if isnothing(init_weight)
    #     init_weight = scale_init(glorot_uniform, Float32(1/sqrt(num_filters)), 0)
    # end
    #
    # filters = Tuple(
    #     Dense(in_dim, hd_dim, sin; init_weight, init_bias,)
    #     for _ in 1:num_filters
    # )

    @error()
    
    MFN(in_dim, hd_dim, out_dim, filters; linear_kws, out_kws)
end

#======================================================#
# GaborLayer
#======================================================#

export GaborLayer

@concrete struct GaborLayer{I} <: Lux.AbstractExplicitLayer
# @concrete struct GaborLayer{I} <: Lux.AbstractExplicitContainerLayer{(:dense,)}
    in_dim::I
    out_dim::I

    init_b
    init_W

    init_μ
    init_γ
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::GaborLayer)
    (;
        x̄ = l.init_μ(rng, l.in_dim),
        γ = l.init_γ(rng, l.out_dim),

        W = l.init_W(rng, l.out_dim, l.in_dim),
        b = l.init_b(rng, l.out_dim),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, l::GaborLayer)
    T = l.init(rng, 1) |> eltype
    (;
        minushalf = T[-0.5],
    )
end

function (l::GaborLayer)(x::AbstractArray, ps, st)

    # should apply softplus to γ

    y_sin = sin.(ps.W * x + ps.b)
    dist2 = sum(abs2, (x .- ps.x̄); dims = 1)
    y_exp = exp.(st.minushalf * dist2 .* ps.γ)

    y = y_sin .* y_exp

    return y, st
end

#======================================================#
# Gaussian Layer (only 1D for now)
#======================================================#

export GaussianLayer

struct GaussianLayer{F,I} <: Lux.AbstractExplicitLayer
    init::F
    in_dim::I
    out_dim::I
end

function GaussianLayer(in_dim::Integer, out_dim::Integer; init = rand32)
    @assert in_dim == out_dim == 1
    GaussianLayer(init, in_dim, out_dim)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::GaussianLayer)
    T = l.init(rng, 1) |> eltype
    (;
        c  = T[1],
        x̄  = l.init(rng, l.in_dim),
        σi = l.init(rng, l.in_dim),
    )
end

function Lux.initialstates(rng::Random.AbstractRNG, l::GaussianLayer)
    T = l.init(rng, 1) |> eltype
    (;
        minushalf = T[-0.5],
    )
end

function (l::GaussianLayer)(x::AbstractArray, ps, st::NamedTuple)
    z = @. (x - ps.x̄) * ps.σi
    # y = ps.c * _rbf(z)
    y = @. ps.c * exp(st.minushalf * z^2)
    return y, st
end

# @inline _rbf(x) = @. exp(-x^2)
# function ChainRulesCore.rrule(::typeof(_rbf), x)
#     T = eltype(x)
#     y = _rbf(x)
#     @inline ∇_rbf(ȳ) = ChainRulesCore.NoTangent(), @fastmath(@. -T(2) * x * y * ȳ)
#
#     y, ∇_rbf
# end

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

