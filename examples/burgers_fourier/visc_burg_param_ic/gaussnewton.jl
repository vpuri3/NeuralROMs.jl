#
using LinearAlgebra, Zygote
using Lux, LinearSolve, LineSearches

#======================================================#
function nlsq(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{AbstractVector, NamedTuple},
    st::NamedTuple,
    data::Tuple;
    device = Lux.cpu_device(),
    kwargs...,
)
    st = Lux.testmode(st)
    p0 = ComponentArray(p0)

    xdata, ydata = data |> device
    p0, st = (p0, st) |> device

    function f(p)
        NN(xdata, p, st)[1] - ydata |> vec
    end

    nlsq(f, p0; kwargs...)
end
#======================================================#
"""
# Arguments
- `alpha`: step size at previous iteration
- `x`: current x
- `x_ls`: new guess
- `f_x_previous`: previous objective value
- `s`: search direction
"""
struct NLSQState{T<:Real, Tx<:AbstractVector{T}, Ts<:AbstractVector{T}}
    alpha::T
    x::Tx
    x_ls::Tx
    f_x_previous::T
    s::Ts
end
#======================================================#
"""
    nlsq(f, x0; maxiters, abstol, α0, linesearch, method) -> x

Nonlinear least square
"""
function nlsq(
    f,
    x0::AbstractArray;
    maxiters::Int = 50,
    abstol::Real = 1f-6,
    α0 = 1.0f0,
    alphaguess = InitialStatic(),
    linesearch = BackTracking(),
    method = Val(:GaussNewton),
)
    x = x0
    α = α0 # initial_previous

    obj(x) = (y = f(x); y' * y / length(y)) # MSE
    obj_grad(x) = Zygote.gradient(obj, x)[1]

    iter = 0
    o = obj(x)
    oprev = o

    while iter < maxiters && o > abstol
        iter += 1 

        s = step_direction(x, f, obj, method)
        α, o = dolinesearch(obj, obj_grad, linesearch, α, x, s, o)
        x += α * s

        _o = round(o; sigdigits = 8)
        println("$(typeof(method)) Iter $iter || MSE: $_o")

        if isapprox(oprev, o, atol = 1f-5)
            println("no longer going down")
            break
        end
        oprev = o
    end

    x
end

function step_direction(x, f, obj, method)

    method = isnothing(method) ? Val(:GaussNewton) : method

    if method isa Val{:GaussNewton}
        y = f(x)
        J = Zygote.jacobian(f, x)[1]
        -leastsquare(J, y)
    elseif method isa Val{:GradientDescent}
        -Zygote.gradient(obj, x)[1] # Gradient Descent
    end
end

function leastsquare(A, b)
    A \ b # (A'*A) \ A'*y TODO replace with LinearSolve QR
end

function dolinesearch(obj, obj_grad, linesearch, α0, x, s,
    ϕ0 = nothing, dϕ0 = nothing,
)
    T = eltype(x)

    ϕ(α)   = obj(x + α * s)
    dϕ(α)  = dot(obj_grad(x + α * s)[:], s) # TODO why is [:] still needed?
    ϕdϕ(α) = ϕ(α), dϕ(α)

    # @show obj_grad(x) |> typeof
    # @show s |> typeof

    ϕ0  = isnothing(ϕ0)  ? obj(x)      : ϕ0
    dϕ0 = isnothing(dϕ0) ? dϕ(zero(T)) : dϕ0

    linesearch(ϕ, dϕ, ϕdϕ, α0, ϕ0, dϕ0)
end

#======================================================#

# # rosenbrock
# rosenbrock(x) = (1.f0 - x[1])^2 + 100.f0 * (x[2] - x[1]^2)^2
# f(x) = [rosenbrock(x), (x[1] - 1.f0), (x[2]-1.f0), x[1]*x[2] - 1]
# x0 = zeros(Float32, 2)
#
# for linesearch in (
#     Static, BackTracking, HagerZhang, MoreThuente, StrongWolfe,
# )
#     println(linesearch)
#     linesearch = linesearch === Static ? Static() : linesearch{Float32}()
#     nlsq(f, x0; linesearch)
#     println()
# end
nothing
#
