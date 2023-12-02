#======================================================#
# ForwardDiff
#======================================================#
# struct FDDeriv1Tag end
# struct FDDeriv2Tag end
# struct FDDeriv2TagInternal end

"""
Based on SparseDiffTools.auto_jacvec

MWE:

```julia
f = x -> exp.(x)
f = x -> x .^ 2
x = [1.0, 2.0, 3.0, 4.0]

forwarddiff_deriv1(f, x)
forwarddiff_deriv2(f, x)
```
"""
function forwarddiff_deriv1(f, x)
    T = eltype(x)
    tag = ForwardDiff.Tag(f, T)
    y = Dual{typeof(tag)}.(x, one(T))

    fy = f(y)
    fx = value.(fy)
    df = partials.(fy, 1)
    
    fx, df
end

# SparseDiffTools.SparseDiffToolsTag()
# SparseDiffTools.DeivVecTag()
# ForwardDiff.Tag(FDDeriv1Tag(), eltype(x))

# function ForwardDiff.checktag(
#     ::Type{<:ForwardDiff.Tag{<:SparseDiffToolsTag, <:T}},
#     f::F, x::AbstractArray{T}) where {T, F}
#     return true
# end

function forwarddiff_deriv2(f, x)
    T = eltype(x)
    tag1 = ForwardDiff.Tag(f, T)
    tag2 = ForwardDiff.Tag(f, T)
    z = Dual{typeof(tag1)}.(Dual{typeof(tag2)}.(x, one(T)), one(T))

    fz = f(z)
    fx = value.(value.(fz))
    df = value.(partials.(fz, 1))
    d2f = partials.(partials.(fz, 1), 1)

    fx, df, d2f
end

function forwarddiff_jacobian(f, x)
    ForwardDiff.jacobian(f, x)
end

#======================================================#
# FiniteDiff
#======================================================#
function finitediff_deriv1(f, x; ϵ = cbrt(eps(eltype(x))))
    _fx = f(x .- ϵ)
    fx  = f(x)
    fx_ = f(x .+ ϵ)

    T = eltype(x)
    ϵinv = inv(ϵ)

    df  = T(0.5) * ϵinv   * (fx_ - _fx)

    fx, df
end

function finitediff_deriv2(f, x; ϵ = cbrt(eps(eltype(x))))
    _fx = f(x .- ϵ)
    fx  = f(x)
    fx_ = f(x .+ ϵ)

    T = eltype(x)
    ϵinv = inv(ϵ)

    df  = T(0.5) * ϵinv   * (fx_ - _fx)
    d2f = T(1.0) * ϵinv^2 * (fx_ + _fx - 2fx)

    fx, df, d2f
end

function finitediff_jacobian(f, x)
    FiniteDiff.finite_difference_jacobian(f, x, Val{:central})
end

#======================================================#
