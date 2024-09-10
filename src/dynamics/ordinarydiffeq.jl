#
#===========================================================#
export GalerkinCollocation

@concrete mutable struct GalerkinCollocation{mass_matrix, normal} <: AbstractSolveScheme where{mass_matrix, normal}
    prob
    model
    xyz
    α # regularization
    ca_axes
    linalg
    debug::Bool
end

function GalerkinCollocation(
    prob::AbstractPDEProblem,
    model::AbstractNeuralModel,
    p0::AbstractVector{T},
    xyz::AbstractMatrix{T};
    α::T = zero(T),
    linalg::SciMLBase.AbstractLinearAlgorithm = KrylovJL_GMRES(),
    mass_matrix::Bool = false,
    normal::Bool = false,
    debug::Bool = false,
) where{T<:Real}

    # https://jso.dev/Krylov.jl/stable/solvers/ls/
    # linalg = QRFactorization(ColumnNorm())
    # linalg = KrylovJL_GMRES() on normal = true
    # linalg = KrylovJL_LSMR() on normal = false

    # # reuse lincache
    # f = model(xyz, p0)
    # J = dudp(model, xyz, p0)
    # linprob = LinearProblem(J, vec(f))
    # lincache = SciMLBase.init(linprob, linalg)

    ca_axes = getaxes(p0)

    GalerkinCollocation{
        mass_matrix, normal,
    }(
        prob, model, xyz, α, ca_axes, linalg, debug,
    )
end

# function Adapt.adapt_structure(l::GalerkinCollocation)
# end

function (l::GalerkinCollocation{false, false})(
    p::AbstractVector,
    params,
    t::Number,
)
    ps = ComponentArray(p, l.ca_axes)

    J = dudp(l.model, l.xyz, ps)
    f = dudtRHS(l.prob, l.model, l.xyz, ps, t) |> vec

    # A = iszero(l.α) ? J : (J + I * l.α)
    A = J
    b = f

    linprob = LinearProblem(A,  b)
    linsol  = solve(linprob, l.linalg)
    check_linsol_retcode(J, f, linsol; debug = l.debug)

    dp = linsol.u

    getdata(dp)
end

function (l::GalerkinCollocation{false, true})(
    p::AbstractVector,
    params,
    t::Number,
)
    ps = ComponentArray(p, l.ca_axes)

    J = dudp(l.model, l.xyz, ps)
    f = dudtRHS(l.prob, l.model, l.xyz, ps, t) |> vec

    A = J' * J
    A = iszero(l.α) ? A : (A + I * l.α)
    b = J' * f

    linprob = LinearProblem(A,  b)
    linsol  = solve(linprob, l.linalg)
    check_linsol_retcode(J, f, linsol; debug = l.debug)

    dp = linsol.u

    getdata(dp)
end

##################
# in place ODE problem
# pointless unless we're reusuing lincache
# and computing J, f in-place
##################

# function (l::GalerkinCollocation{false, false})(
#     dp::AbstractVector,
#     p::AbstractVector,
#     params,
#     t::Number,
# )
#     ps = ComponentArray(p, l.ca_axes)
#
#     J = dudp(l.model, l.xyz, ps)
#     f = dudtRHS(l.prob, l.model, l.xyz, ps, t)
#
#     linprob = LinearProblem(J' * J, J' * vec(f); u0 = dp)
#     # linprob = LinearProblem(J, vec(f); u0 = dp)
#     linsol  = solve(linprob, l.linalg)
#     check_linsol_retcode(linsol)
#
#     nothing
# end

##################
# mass matrix form
##################

# want tighter integration between linear solve and ODE solve.
# Make J the mass-matrix and have scheme(u, p, t) -> f
# looks like varying mass matrices are not fully supported
# odefunc = ODEFunction{iip}(scheme; mass_matrix = J)
# https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/

# TODO: write custom Jacobian
# J = d/dp (rhs(p, params, t))

function (l::GalerkinCollocation{true})(
    dpdt::AbstractVector,
    p::AbstractVector,
    params,
    t::Number,
)
    ps = ComponentArray(p, l.ca_axes)
    f = dudtRHS(l.prob, l.model, l.xyz, ps, t) |> vec
    J = dudp(l.model, l.xyz, ps)
    J * dpdt - f
end
#===========================================================#
#
