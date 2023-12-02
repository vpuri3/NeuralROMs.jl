#
using LineSearches
using LinearAlgebra: norm, dot

function gdoptimize(f, g!, fg!, x0::AbstractArray{T}, linesearch,
                    maxiter::Int = 10000,
                    g_rtol::T = sqrt(eps(T)), g_atol::T = eps(T)) where T <: Number
    x = copy(x0)
    gvec = similar(x)
    g!(gvec, x)
    fx = f(x)

    gnorm = norm(gvec)
    gtol = max(g_rtol*gnorm, g_atol)

    # Univariate line search functions
    ϕ(α) = f(x .+ α.*s)
    function dϕ(α)
        g!(gvec, x .+ α.*s)
        return dot(gvec, s)
    end
    function ϕdϕ(α)
        phi = fg!(gvec, x .+ α.*s)
        dphi = dot(gvec, s)
        return (phi, dphi)
    end

    s = similar(gvec) # Step direction

    iter = 0
    while iter < maxiter && gnorm > gtol
        iter += 1
        s .= -gvec # gradient dir

        dϕ_0 = dot(s, gvec)
        α, fx = linesearch(ϕ, dϕ, ϕdϕ, 1.0, fx, dϕ_0)

        @. x = x + α*s # in place
        g!(gvec, x)
        gnorm = norm(gvec)
    end

    return (fx, x, iter)
end

# rosenbrock function
f(x) = (1.0 - x[1])^2 + 100.0 * (x[2] - x[1]^2)^2

function g!(gvec, x)
    gvec[1] = -2.0 * (1.0 - x[1]) - 400.0 * (x[2] - x[1]^2) * x[1]
    gvec[2] = 200.0 * (x[2] - x[1]^2)
    gvec
end

function fg!(gvec, x)
    g!(gvec, x)
    f(x)
end

x0 = [-1., 1.0]

linesearch = BackTracking(order=3)
@show fx_bt3, x_bt3, iter_bt3 = gdoptimize(f, g!, fg!, x0, linesearch)

linesearch = StrongWolfe()
@show fx_sw, x_sw, iter_sw = gdoptimize(f, g!, fg!, x0, linesearch)
#======================================================#
nothing
#
