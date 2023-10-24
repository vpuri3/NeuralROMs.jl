#
"""
    mse(ypred, ytrue)

Mean squared error
"""
mae(y, ŷ) = sum(abs, ŷ - y) / length(ŷ)

"""
    mse(ypred, ytrue)

Mean squared error
"""
mse(y, ŷ) = sum(abs2, ŷ - y) / length(ŷ)

"""
    pnorm(p)

pnorm
"""
pnorm(p) = (y, ŷ) -> sum(abs ∘ Base.Fix2(^, p), ŷ - y) / length(ŷ)

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
