#
using LinearAlgebra, ComponentArrays
using Zygote, ForwardDiff, Lux
using LineSearches, NLSolversBase

#======================================================#
"""
# Arguments
- `alpha`: step size at previous iteration
- `x`: current x
- `x_ls`: new guess
- `f_x_previous`: previous objective value
- `s`: search direction
"""
mutable struct NLSQState{T<:Real,
    Tx<:AbstractVector{T},
    Ts<:AbstractVector{T},
    }
    alpha::T
    x::Tx
    x_ls::Tx
    f_x_previous::T
    s::Ts
end
#======================================================#
function step_direction(x, f, loss, method)
    method = isnothing(method) ? Val(:GaussNewton) : method
    if method isa Val{:GaussNewton}
        y = f(x)
        J = Zygote.jacobian(f, x)[1]
        -leastsquare(J, y)
    elseif method isa Val{:GradientDescent}
        -Zygote.gradient(loss, x)[1] # Gradient Descent
    end
end

function leastsquare(A, b)
    A \ b # (A'*A) \ A'*y TODO replace with LinearSolve QR
end

function nlsq(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{AbstractVector, NamedTuple},
    st::NamedTuple,
    data::Tuple,
    method;
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

    nlsq(f, p0, method; kwargs...)
end
#======================================================#
"""
    nlsq(f, x0; maxiters, abstol, α0, linesearch, method) -> x

Nonlinear least square
"""
function nlsq(
    f,
    x0::AbstractArray,
    method = Val(:GaussNewton);
    maxiters::Int = 50,
    abstol::Real = 1f-6,
    reltol::Real = 1f-6,
    alphaguess = InitialStatic(),
    linesearch = BackTracking(),
)

    loss(x) = (y = f(x); y' * y / length(y)) # MSE
    grad(x) = Zygote.gradient(loss, x)[1]
    function lossgrad(x)
        (l,), pb = Zygote.pullback(loss, x)
        g = pb(one.(l))[1]
        l, g
    end

    obj = OnceDifferentiable(loss, grad, lossgrad, x0; inplace = false)

    x1  = copy(x0)

    iter = 0
    _l  = loss(x0)
    l = _l

    state = NLSQState(1f0, x0, x1, l, x0)

    @time while iter < maxiters && l > abstol
        iter += 1 

        state.s = step_direction(x0, f, loss, method)
        dϕ0 = dot(grad(x0)[:], state.s[:]) # TODO [:]
        state.alpha = alphaguess(linesearch, state, l, dϕ0, obj)

        α, l = linesearch(obj, x0, state.s, state.alpha, x1, l, dϕ0)

        lprint = round(l; sigdigits = 8)
        println("$(typeof(method)) Iter $iter || MSE: $lprint")

        if isapprox(_l, l, atol = 1f-5)
            break
        end

        _l = l
        copy!(x0, x1)
        state.alpha = α
        state.f_x_previous = l
    end

    x1
end

#======================================================#
function datafit(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    data::Tuple,
    opt::Optim.AbstractOptimizer;
    maxiters::Integer = 50,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    device = cpu_device(),
    io::Union{Nothing, IO} = stdout,
)
    st = Lux.testmode(st)
    p0 = ComponentArray(p0)

    xdata, ydata = data |> device
    p0, st = (p0, st)   |> device

    function optloss(optx, optp)
        ypred, _ = NN(xdata, optx, st)

        mse(ydata, ypred)
    end

    iter = Ref(0)
    function callback(p, l)
        println(io, "[$(iter[]) / $maxiters] MSE: $l")
        iter[] += 1
        return false
    end

    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, p0)

    @time optres = solve(optprob, opt; maxiters, callback) #, abstol, reltol)

    obj = round(optres.objective; sigdigits = 8)
    tim = round(optres.solve_time; sigdigits = 8)
    println(io, "#=======================#")
    @show optres.retcode
    println(io, "Achieved objective value $(obj) in time $(tim)s.")
    println(io, "#=======================#")

    optres.u
end

#======================================================#


# rosenbrock(x) = (1.f0 - x[1])^2 + 100.f0 * (x[2] - x[1]^2)^2
# f(x) = [rosenbrock(x), (x[1] - 1.f0), (x[2]-1.f0), x[1]*x[2] - 1]
# x0 = zeros(Float32, 2)
#
# for line in (
#     Static,
#     # BackTracking,
#     # HagerZhang,
#     # MoreThuente,
#     # StrongWolfe,
# )
#     for alpha in (
#         InitialStatic,
#         # InitialPrevious,
#         # InitialQuadratic,
#         # InitialHagerZhang,
#         # InitialConstantChange,
#     )
#         println(line, "\t", alpha)
#
#         linesearch = line === Static ? Static() : line{Float32}()
#         alphaguess = alpha{Float32}()
#
#         nlsq(f, x0; linesearch, alphaguess)
#         println()
#     end
# end
nothing
#
