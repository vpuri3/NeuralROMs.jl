#
"""
    mse(ypred, ytrue)

Mean squared error
"""
mse(y, ŷ) = sum(abs2, ŷ - y) / length(ŷ)

"""
    rel_mse(ypred, ytrue)

relative mean square error
"""
function rel_mse(y, ŷ)
    m = minimum(ŷ)

    ŷ_rel = ŷ - (m - 1f0)
    y_rel = y - (m - 1f0)

    mse(y_rel, ŷ_rel)
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
