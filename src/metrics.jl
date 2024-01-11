#
"""
    mae(ypred, ytrue) -> l
    mae(NN, p, st, batch) -> l, st, stats

Mean squared error
"""
mae(y, ŷ) = sum(abs, ŷ - y) / length(ŷ)

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
mse(y, ŷ) = sum(abs2, ŷ - y) / length(ŷ)

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
        sum(abs ∘ Base.Fix2(^, p), ŷ - y) / length(ŷ)
    end
    
    function pnorm_internal(NN, p, st, batch)
        x, ŷ = batch
        y, st = NN(x, p, st)
        pnorm_internal(y, ŷ), st, ()
    end

    pnorm_internal
end

"""
    l1reg(lossfun, λ)(NN, p, st, batch) -> l, st, stats

L1 Regularization
"""
function l1reg(lossfun, λ::Real; property = nothing)
    function l1reg_internal(NN, p, st::NamedTuple, batch::Tuple)
        l, st, stats = lossfun(NN, p, st, batch)

        _p = isnothing(property) ? p : getproperty(p, property)

        r1 = λ * sum(abs, _p) / length(_p)
        l1 = l + r1

        l1, st, (; stats, l)
    end
end

"""
    l2reg(lossfun, λ)(NN, p, st, batch) -> l, st, stats

L2 Regularization
"""
function l2reg(lossfun, λ::Real; property = nothing)
    function l2reg_internal(NN, p, st::NamedTuple, batch::Tuple)
        l, st, stats = lossfun(NN, p, st, batch)

        T  = eltype(p)
        _p = isnothing(property) ? p : getproperty(p, property)

        r2 = λ * sum(abs2, _p) / length(_p) * T(0.5)
        l2 = l + r2

        l2, st, (; stats, l, r2)
    end
end

"""
    elasticreg(lossfun, λ1, λ2)(NN, p, st, batch) -> l, st, stats

Elastic Regularization (L1 + L2)
"""
function elasticreg(lossfun, λ1::Real, λ2::Real; property = nothing)
    function elasticreg_internal(NN, p, st::NamedTuple, batch::Tuple)
        l, st, stats = lossfun(NN, p, st, batch)

        T  = eltype(p)
        _p = isnothing(property) ? p : getproperty(p, property)
        N  = length(_p)

        r1 = iszero(λ1) ? zero(T) : λ1 * sum(abs , _p) / N
        r2 = iszero(λ2) ? zero(T) : λ2 * sum(abs2, _p) / N * T(0.5)

        lreg = l + r1 + r2

        lreg, st, (; stats, l, r1, r2)
    end
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
#
