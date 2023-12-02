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
    l2reg(lossfun, λ)(NN, p, st, batch) -> l, st, stats

L2-Regularization
"""
function l2reg(lossfun, λ::Real; property = nothing)
    function l2reg_internal(NN, p, st::NamedTuple, batch::Tuple)
        l, st, stats = lossfun(NN, p, st, batch)

        preg = isnothing(property) ? p : getproperty(p, property)
        reg  = λ * (preg' * preg) * λ * 0.5 / length(preg)
        lreg = l + reg

        lreg, st, (stats, l)
    end

    l2reg_internal
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
