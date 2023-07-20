#
function normalize_trajectories(y, ŷ)

    @assert ndims(y) == ndims(ŷ) ≥ 3

    C, B = size(ŷ)[1, end]
    Ns = size(ŷ[2:end-1])
    N = prod(Ns)

    Y = reshape(y, (C, N, B))
    Ŷ = reshape(ŷ, (C, N, B))

    # normalize every [c, :, b] trajectory
    Ŷm = 1 # size [C, B]

    Y = Y ./ Ŷm
    Ŷ = Ŷ ./ Ŷm

    y = reshape(Y, (C, Ns..., B))
    ŷ = reshape(Ŷ, (C, Ns..., B))

    y, ŷ
end

"""
    mse_norm(ypred, ytrue)

Mean squared error where each trajectory is normalized to have unit norm
trajectories of size
"""
function mse_norm(y, ŷ)
    y_nzd, ŷ_nzd = normalize_trajectories(y, ŷ)
    mse(y_nzd, ŷ_nzd)
end

"""
    mse(ypred, ytrue)

Mean squared error
"""
mse(y, ŷ) = sum(abs2, ŷ - y) / length(ŷ)

"""
    rel_mse(ypred, ytrue)

Normalize each trajectory to have unit norm, then take relative error
"""
function rel_mse(y, ŷ, ϵ = eps(eltype(ŷ)),)

    ŷ_abs = abs.(ŷ)
    y_abs = abs.(y)

    sum((ŷ_abs - y_abs) ./ (ŷ_abs .+ ϵ)) / length(ŷ)
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
