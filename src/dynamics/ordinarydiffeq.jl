#
#===========================================================#
export GalerkinCollocation

@concrete mutable struct GalerkinCollocation{mm} <: AbstractSolveScheme where{mm}
    prob
    model
    xyz
    ca_axes
    linalg
end

function GalerkinCollocation(
    prob::AbstractPDEProblem,
    model::AbstractNeuralModel,
    p0::AbstractVector{T},
    xyz::AbstractMatrix{T};
    linalg::SciMLBase.AbstractLinearAlgorithm = QRFactorization(),
    mass_matrix::Bool = false,
) where{T<:Number}

    # # reuse lincache
    # f = model(xyz, p0)
    # J = dudp(model, xyz, p0)
    # linprob = LinearProblem(J, vec(f))
    # lincache = SciMLBase.init(linprob, linalg)

    ca_axes = getaxes(p0)
    GalerkinCollocation{mass_matrix}(prob, model, xyz, ca_axes, linalg)
end

# function Adapt.adapt_structure(l::GalerkinCollocation)
# end

function (l::GalerkinCollocation{false})(
    p::AbstractVector,
    params,
    t::Number,
)
    ps = ComponentArray(p, l.ca_axes)

    J = dudp(l.model, l.xyz, ps)
    f = dudtRHS(l.prob, l.model, l.xyz, ps, t)

    linprob = LinearProblem(J, vec(f))
    linsol  = solve(linprob, l.linalg)
    dp = linsol.u

    getdata(dp)
end

function (l::GalerkinCollocation{false})(
    dp::AbstractVector,
    p::AbstractVector,
    params,
    t::Number,
)
    ps = ComponentArray(p, l.ca_axes)

    J = dudp(l.model, l.xyz, ps)
    f = dudtRHS(l.prob, l.model, l.xyz, ps, t)

    linprob = LinearProblem(J, vec(f); u0 = dp)
    solve(linprob, l.linalg)

    nothing
end

# want tighter integration between linear solve and ODE solve.
# Make J the mass-matrix and have scheme(u, p, t) -> f
# looks like varying mass matrices are not fully supported
# odefunc = ODEFunction{iip}(scheme; mass_matrix = J)
# https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/

# TODO: write custom Jacobian
#
# J = d/dp (rhs(p, params, t))
#

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
