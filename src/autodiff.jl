#
const ADInputTypes{T} = Union{T, AbstractArray{T}} where{T <: Number}

#======================================================#
#======================================================#
#======================================================#
# ForwardDiff
#======================================================#
#======================================================#
#======================================================#
"""
Based on SparseDiffTools.auto_jacvec

MWE:

```julia
f = x -> exp.(x)
f = x -> x .^ 2
x = [1.0, 2.0, 3.0, 4.0]

forwarddiff_deriv1(f, x)
forwarddiff_deriv2(f, x)
forwarddiff_deriv4(f, x)
```
"""
function forwarddiff_deriv1(f,
    x::ADInputTypes{T},
) where{T}

    tag = ForwardDiff.Tag(f, T)
    y = Dual{typeof(tag)}.(x, one(T))

    fy = f(y)
    fx = value.(fy)
    fdx = partials.(fy, 1)
    
    fx, fdx
end

function forwarddiff_deriv1(f,
    (x, y)::NTuple{2, ADInputTypes{T}},
) where{T}
    error("TODO. What should signature of f be? f(x) ?")
end
#======================================================#

function forwarddiff_deriv2(f,
    x::ADInputTypes{T},
) where{T}

    tag1 = ForwardDiff.Tag(f, T)
    tag2 = ForwardDiff.Tag(f, T)

    z = Dual{typeof(tag1)}.(Dual{typeof(tag2)}.(x, one(T)), one(T))

    fz = f(z)
    fx = value.(value.(fz))
    df = value.(partials.(fz, 1))
    d2f = partials.(partials.(fz, 1), 1)

    fx, df, d2f
end
#======================================================#

function forwarddiff_deriv4(f,
    x::ADInputTypes{T},
) where{T}

    tag = ForwardDiff.Tag(f, T)

    z = x
    z = Dual{typeof(tag)}.(z, one(T))
    z = Dual{typeof(tag)}.(z, one(T))
    z = Dual{typeof(tag)}.(z, one(T))
    z = Dual{typeof(tag)}.(z, one(T))

    fz  = f(z)
    fx  = value.(value.(value.(value.(fz))))
    d1f = partials.(value.(value.(value.(fz))), 1)
    d2f = partials.(partials.(value.(value.(fz)), 1), 1)
    d3f = partials.(partials.(partials.(value.(fz), 1), 1), 1)
    d4f = partials.(partials.(partials.(partials.(fz, 1), 1), 1), 1)

    fx, d1f, d2f, d3f, d4f
end

#======================================================#
function forwarddiff_jacobian(f,
    x::ADInputTypes{T},
) where{T}
    ForwardDiff.jacobian(f, x) # TODO: scalar indexing on GPU :/
end

# SparseDiffTools.SparseDiffToolsTag()
# SparseDiffTools.DeivVecTag()
# ForwardDiff.Tag(FDDeriv1Tag(), eltype(x))

# struct FDDeriv1Tag end
# struct FDDeriv2Tag end
# struct FDDeriv2TagInternal end

# function ForwardDiff.checktag(
#     ::Type{<:ForwardDiff.Tag{<:SparseDiffToolsTag, <:T}},
#     f::F, x::AbstractArray{T}) where {T, F}
#     return true
# end

#======================================================#
#======================================================#
#======================================================#
# FiniteDiff
#======================================================#
#======================================================#
#======================================================#

function finitediff_deriv1(f,
    x::ADInputTypes{T};
    ϵ = nothing,
) where{T}

    ϵ = isnothing(ϵ) ? cbrt(eps(T)) : T.(ϵ)
    ϵinv = inv(ϵ)

    _fx = f(x .- ϵ)
    fx  = f(x)
    fx_ = f(x .+ ϵ)

    fdx = T(0.5) * ϵinv   * (fx_ - _fx)

    fx, fdx
end
#======================================================#

function finitediff_deriv2(f,
    x::ADInputTypes{T};
    ϵ = nothing,
) where{T}

    ϵ = isnothing(ϵ) ? cbrt(eps(T)) : T.(ϵ)
    ϵinv = inv(ϵ)

    _fx = f(x .- ϵ)
    fx  = f(x)
    fx_ = f(x .+ ϵ)

    fdx  = T(0.5) * ϵinv   * (fx_ - _fx)
    fdxx = T(1.0) * ϵinv^2 * (fx_ + _fx - 2fx)

    fx, fdx, fdxx
end
#======================================================#

function finitediff_deriv4(f,
    x::ADInputTypes{T};
    ϵ = nothing,
) where{T}

    ϵ = isnothing(ϵ) ? cbrt(eps(T)) : T.(ϵ)
    ϵinv = inv(ϵ)
    
    __fx = f(x .- 2ϵ)
    _fx  = f(x .- ϵ)
    fx   = f(x)
    fx_  = f(x .+ ϵ)
    fx__ = f(x .+ 2ϵ)
    
    # https://www.mech.kth.se/~ardeshir/courses/literature/fd.pdf
    d1f = T(0.5) * ϵinv   * (fx_ - _fx)
    d2f = T(1.0) * ϵinv^2 * (fx_ + _fx - 2fx)
    d3f = T(0.5) * ϵinv^3 * (fx__ - 2fx_ + 2_fx - __fx)
    d4f = T(1.0) * ϵinv^4 * (fx__ - 4fx_ + 6fx - 4_fx + __fx)
    
    @warn "3rd, 4th order formulae finitediff derivative are very inaccurate" maxlog = 1

    fx, d1f, d2f, d3f, d4f
end

#======================================================#
function finitediff_jacobian(f,
    x::ADInputTypes{T};
    ϵ = nothing,
) where{T}

    FiniteDiff.finite_difference_jacobian(f, x, Val{:central})
end

#======================================================#
