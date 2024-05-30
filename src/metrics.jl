#
#=====================================================================#
"""
    mae(ypred, ytrue) -> l
    mae(NN, p, st, batch) -> l, st, stats

Mean squared error
"""
mae(y, ŷ) = sum(abs, ŷ - y) / numobs(ŷ)

function mae(NN, p, st, batch)
    x, ŷ = batch
    y, st = NN(x, p, st)
    mae(y, ŷ), st, (;)
end

"""
    mae_clamped(δ)(NN, p, st, batch) -> l, st, stats

Clamped mean squared error
"""
function mae_clamped(
    δ::Real;
    clamp_true::Bool = true,
    clamp_pred::Bool = true,
)
    function mae_clamped_internal(NN, p, st, batch)
        x, ŷ = batch
        y, st = NN(x, p, st)

        y = clamp_true ? clamp.(y, -δ, δ) : y
        ŷ = clamp_pred ? clamp.(ŷ, -δ, δ) : ŷ

        mae(y, ŷ), st, (;)
    end
end

"""
    mse(ypred, ytrue) -> l
    mse(NN, p, st, batch) -> l, st, stats

Mean squared error
"""
mse(y, ŷ) = sum(abs2, ŷ - y) / numobs(ŷ)

function mse(NN, p, st, batch)
    x, ŷ = batch
    y, st = NN(x, p, st)

    mse(y, ŷ), st, (;)
end

"""
    pnorm(p)(y, ŷ) -> l
    pnorm(p)(NN, p, st, batch) -> l, st, stats

P-Norm
"""
function pnorm(p::Real)
    function pnorm_internal(y, ŷ)
        sum(abs ∘ Base.Fix2(^, p), ŷ - y) / numobs(ŷ)
    end
    
    function pnorm_internal(NN, p, st, batch)
        x, ŷ = batch
        y, st = NN(x, p, st)
        pnorm_internal(y, ŷ), st, ()
    end

    pnorm_internal
end

"""
    rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)

Calculuate r2 (coefficient of determination) score.
"""
function rsquare(y, ŷ)
    @assert size(y) == size(ŷ)

    y = vec(y)
    ŷ = vec(ŷ)

    ȳ = sum(ŷ) / length(ŷ)   # mean
    MSE  = sum(abs2, ŷ  - y) # mse  (sum of squares of residuals)
    VAR  = sum(abs2, ŷ .- ȳ) # var  (sum of squares of data)

    rsq =  1 - MSE / (VAR + eps(eltype(y)))

    return rsq
end
#=====================================================================#

"""
    elasticreg(lossfun, λ1, λ2)(NN, p, st, batch) -> l, st, stats

Elastic Regularization (L1 + L2)
"""
function elasticreg(
    lossfun,
    λ1::Real,
    λ2::Real;
    property = nothing,
    lname::Symbol = :mse,
)

    function elasticreg_internal(NN, p, st::NamedTuple, batch::Tuple)
        T  = eltype(p)
        l, st_new, stats = lossfun(NN, p, st, batch)

        lstats = NamedTuple{(lname,)}((l,))

        _p = isnothing(property) ? p : getproperty(p, property)
        _N = length(_p)

        l1 = if iszero(λ1)
            zero(T)
        else
            λ1 * sum(abs , _p) / _N 
        end

        l2 = if iszero(λ2)
            zero(T)
        else
             T(0.5) * λ2 * sum(abs2, _p) / _N
        end

        lreg = l + l1 + l2

        lreg, st_new, (; stats..., lstats..., l1, l2)
    end
end

#=====================================================================#

export compute_cbound

function compute_cbound(NN, p, st)
    true
end

function compute_cbound(NN::Chain, p, st)
    cbound = true

    for layername in propertynames(NN.layers)
        layer    = getproperty(NN.layers, layername)
        p_layer  = getproperty(p, layername)
        st_layer = getproperty(st, layername)

        cbound *= compute_cbound(layer, p_layer, st_layer)
    end

    cbound
end

function compute_cbound(::Dense, p, st)
    compute_cbound(p.weight)
end

function compute_cbound(W::AbstractMatrix) # ∞ norm
    rsum = sum(abs, W, dims = 2)
    maximum(rsum)
end

#=====================================================================#

"""
    codereg_autodecoder(lossfun, σ; property)(NN, p, st, batch) -> l, st, stats

code regularized loss: `lossfun(..) + 1/σ² ||ũ||₂²`
"""
function codereg_autodecoder(lossfun, σ2inv::Real; lname::Symbol = :mse)

    function codereg_internal(NN, p, st::NamedTuple, batch::Tuple)
        T = eltype(p)
        N = numobs(batch)
        l, st_new, stats = lossfun(NN, p, st, batch)

        lstats = NamedTuple{(lname,)}((l,))

        lcode = if iszero(σ2inv)
             zero(T)
        else
            _, code = get_autodecoder(NN, p, st)
            pcode, _ = code[1](batch[1][2], code[2], code[3])
            lcode = sum(abs2, pcode)

            σ2inv * lcode / N
        end

        loss = l + lcode

        loss, st_new, (; stats..., lstats..., lcode)
    end
end

#=====================================================================#

"""
    regularize_decoder(lossfun, σ, λ1, λ2, property)(NN, p, st, batch) -> l, st, stats

code reg loss, L1/L2 on decoder
`lossfun(..) + 1/σ² ||ũ||₂² + L1/L2 on decoder + Lipschitz reg. on decoder`
"""
function regularize_decoder(
    lossfun;
    α::T = 0f0,
    λ1::T = 0f0,
    λ2::T = 0f0,
    lname::Symbol = :mse,
) where{T<:Real}

    function regularize_decoder_internal(NN, p, st::NamedTuple, batch::Tuple)
        @assert eltype(p) == T
        N = numobs(batch)
        l, st_new, stats = lossfun(NN, p, st, batch)

        if p isa AbstractArray
            @assert eltype(p) === T "got $(eltype(p)) === $(T)."
            @assert eltype(l) === T "got $(eltype(l)) === $(T)."
        end

        lstats = NamedTuple{(lname,)}((l,))
        stats = (; stats..., lstats...)

        ###
        # Lipschitz reg
        ###

        lcond, stats = if iszero(α)
            zero(T), stats
        else
            cbound = compute_cbound(NN, p, st)
            lcond  = cbound * α / N
            lcond, (; stats..., cbound, lcond)
        end

        ###
        # elastic reg
        ###

        N = length(p)

        l1, stats = if iszero(λ1)
            zero(T), stats
        else
            l1 = λ1 * sum(abs , p) / N 
            l1, (; stats..., l1,)
        end

        l2, stats = if iszero(λ2)
            zero(T), stats
        else
            l2 = T(0.5) * λ2 * sum(abs2, p) / N
            l2, (; stats..., l2,)
        end

        ###
        # sum
        ###

        loss = l + lcond + l1 + l2

        loss, st_new, stats
    end
end
#=====================================================================#

"""
    regularize_autodecoder(lossfun, σ, λ1, λ2, property)(NN, p, st, batch) -> l, st, stats

code reg loss, L1/L2 on decoder
`lossfun(..) + 1/σ² ||ũ||₂² + L1/L2 on decoder + Lipschitz reg. on decoder`
"""
function regularize_autodecoder(
    lossfun;
    σ2inv::T = Inf32,
    α::T = 0f0,
    λ1::T = 0f0,
    λ2::T = 0f0,
    lname::Symbol = :mse,
) where{T<:Real}

    function regularize_autodecoder_internal(NN, p, st::NamedTuple, batch::Tuple)
        @assert eltype(p) == T
        N = numobs(batch)
        l, st_new, stats = lossfun(NN, p, st, batch)

        if p isa AbstractArray
            @assert eltype(p) === T "got $(eltype(p)) === $(T)."
            @assert eltype(l) === T "got $(eltype(l)) === $(T)."
        end

        lstats = NamedTuple{(lname,)}((l,))
        stats = (; stats..., lstats...)

        decoder, code = get_autodecoder(NN, p, st)

        ###
        # code regularization
        ###

        lcode, stats = if iszero(σ2inv)
             zero(T), stats
        else
            pcode, _ = code[1](batch[1][2], code[2], code[3])
            lcode = sum(abs2, pcode)

            lcode = σ2inv * lcode / N

            lcode, (; stats..., lcode)
        end

        ###
        # Lipschitz reg on decoder
        ###

        lcond, stats = if iszero(α)
            zero(T), stats
        else
            cbound = compute_cbound(decoder...)
            lcond  = cbound * α / N
            lcond, (; stats..., cbound, lcond)
        end

        ###
        # elastic reg on decoder
        ###

        _p = decoder[2]
        _N = length(_p)

        l1, stats = if iszero(λ1)
            zero(T), stats
        else
            l1 = λ1 * sum(abs , _p) / _N 
            l1, (; stats..., l1,)
        end

        l2, stats = if iszero(λ2)
            zero(T), stats
        else
            l2 = T(0.5) * λ2 * sum(abs2, _p) / _N
            l2, (; stats..., l2,)
        end

        ###
        # sum
        ###

        loss = l + lcode + lcond + l1 + l2

        loss, st_new, stats
    end
end
#=====================================================================#

"""
    regularize_flatdecoder(lossfun, σ, λ1, λ2, property)(NN, p, st, batch) -> l, st, stats

`lossfun(..) + L2 (on hyper) + Lipschitz (on decoder)`
"""
function regularize_flatdecoder(
    lossfun;
    α::T = 0f0,  # on decoder
    λ2::T = 0f0, # on hyper
    lname::Symbol = :mse,
) where{T<:Real}

    function regularize_flatdecoder_internal(NN, p, st::NamedTuple, batch::Tuple)
        @assert eltype(p) == T
        N = numobs(batch)
        l, st_new, stats = lossfun(NN, p, st, batch)

        if p isa AbstractArray
            @assert eltype(p) === T "got $(eltype(p)) === $(T)."
            @assert eltype(l) === T "got $(eltype(l)) === $(T)."
        end

        lstats = NamedTuple{(lname,)}((l,))
        stats = (; stats..., lstats...)

        hyper, decoder = get_flatdecoder(NN, p, st)

        ###
        # Lipschitz reg on decoder
        ###

        lcond, stats = if iszero(α)
            zero(T), stats
        else
            cbound = compute_cbound(decoder...)
            lcond  = cbound * α / N
            lcond, (; stats..., cbound, lcond)
        end

        ###
        # L2 reg on decoder
        ###

        _p = hyper[2]
        _N = length(_p)

        l2, stats = if iszero(λ2)
            zero(T), stats
        else
            l2 = T(0.5) * λ2 * sum(abs2, _p) / _N
            l2, (; stats..., l2_hyper = l2,)
        end

        ###
        # sum
        ###

        loss = l + lcond + l2

        loss, st_new, stats
    end
end
#=====================================================================#
#
