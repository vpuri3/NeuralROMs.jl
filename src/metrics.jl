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
    mae(y, ŷ), st, ()
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

    mse(y, ŷ), st, ()
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
function elasticreg(lossfun, λ1::Real, λ2::Real; property = nothing)
    function elasticreg_internal(NN, p, st::NamedTuple, batch::Tuple)
        T  = eltype(p)
        l, st_new, stats = lossfun(NN, p, st, batch)

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

        lreg, st_new, (; stats..., l, l1, l2)
    end
end

#=====================================================================#

"""
    codereg(lossfun, σ; property)(NN, p, st, batch) -> l, st, stats

code regularized loss: `lossfun(..) + 1/σ² ||ũ||₂²`
"""
function codereg(lossfun, σ2inv::Real)

    function codereg_internal(NN, p, st::NamedTuple, batch::Tuple)
        T = eltype(p)
        N = numobs(batch)
        l, st_new, stats = lossfun(NN, p, st, batch)

        lcode = if iszero(σ2inv)
             zero(T)
        else
            _, code = get_autodecoder(NN, p, st)
            pcode, _ = code[1](batch[1][2], code[2], code[3])
            lcode = sum(abs2, pcode)

            σ2inv * lcode / N
        end

        loss = l + lcode

        loss, st_new, (; stats..., l, lcode)
    end
end

#=====================================================================#

export compute_cbound

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
) where{T<:Real}

    function regularize_autodecoder_internal(NN, p, st::NamedTuple, batch::Tuple)
        @assert eltype(p) == T
        N = numobs(batch)
        l, st_new, stats = lossfun(NN, p, st, batch)

        decoder, code = get_autodecoder(NN, p, st)

        ###
        # code regularization
        ###

        lcode = if iszero(σ2inv)
             zero(T)
        else
            pcode, _ = code[1](batch[1][2], code[2], code[3])
            lcode = sum(abs2, pcode)

            σ2inv * lcode / N
        end

        ###
        # Lipschitz reg on decoder
        ###

        cbound = compute_cbound(decoder...)
        lcond  =  iszero(α) ? T(0) : cbound * α / N

        ###
        # elastic reg on decoder
        ###

        _p = decoder[2]
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

        ###
        # sum
        ###

        loss = l + lcode + lcond + l1 + l2
    
        loss, st_new, (; stats..., l, lcode, lcond, cbound, l1, l2)
    end
end
#=====================================================================#
#
