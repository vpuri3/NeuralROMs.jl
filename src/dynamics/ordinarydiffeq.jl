#
#===========================================================#
export GalerkinCollocation

mutable struct GalerkinCollocation{
    Tprob,Tmodel,Ttimealg,Txyz,Taxes,Tlincache
} <: AbstractSolveScheme

    prob::Tprob
    model::Tmodel
    timealg::Ttimealg

    xyz::Txyz
    ca_axes::Taxes
    lincache::Tlincache

    function GalerkinCollocation(
        prob::AbstractPDEProblem,
        model::AbstractNeuralModel,
        timealg::SciMLBase.AbstractODEAlgorithm,
        p0::AbstractVector{T},
        xyz::AbstractMatrix{T};
        linalg::SciMLBase.AbstractLinearAlgorithm = QRFactorization(),
    ) where{T<:Number}

        f = model(xyz, p0)
        J = dudp(model, xyz, p0)

        ca_axes = getaxes(p0)

        linprob = LinearProblem(J, vec(f))
        lincache = SciMLBase.init(linprob, linalg)

        new{
            typeof(prob), typeof(model), typeof(timealg), typeof(xyz),
            typeof(ca_axes), typeof(lincache),
        }(
            prob, model, timealg, xyz, ca_axes, lincache,
        )
    end
end

# function Adapt.adapt_structure(l::GalerkinCollocation)
# end

function (l::GalerkinCollocation)(
    p::AbstractVector,
    params,
    t::Number,
)
    ps = ComponentArray(p, l.ca_axes)

    f = dudtRHS(l.prob, l.model, l.xyz, ps, t)
    J = dudp(l.model, l.xyz, ps)

    # # reuse cache
    # l.lincache.A = J
    # l.lincache.b = vec(f)
    # dp = solve(l.lincache).u

    # dont reuse cache
    linprob = LinearProblem(J, vec(f))
    linsol  = solve(linprob, l.lincache.alg)
    dp = linsol.u

    getdata(dp)
end

function (l::GalerkinCollocation)(
    dp::AbstractVector,
    p::AbstractVector,
    params,
    t::Number,
)
    ps = ComponentArray(p, l.ca_axes)

    f = dudtRHS(l.prob, l.model, l.xyz, ps, t)
    J = dudp(l.model, l.xyz, ps)

    linprob = LinearProblem(J, vec(f); u0 = dp)
    solve(linprob, l.lincache.alg)
end

#===========================================================#
# function evolve_model(
#     scheme::GalerkinCollocation,
#     data::NTuple{3, AbstractVecOrMat},
#     p0::AbstractVector,
#     Δt::Union{Real,Nothing} = nothing;
#     device = Lux.cpu_device(),
#     verbose::Bool = true,
# )
#     odeprob = ODEProblem(scheme, p0, tspan)
#     integrator = SciMLBase.init(odeprob, scheme.odealg)
#
#     # move to device
#     p0 = p0 |> device
#     scheme = scheme |> device
#
#     Δt = isnothing(Δt) ? -(reverse(extrema(tsave))...) / 200 |> T : T(Δt)
#
#     solve(integrator)
# end
#===========================================================#
#
