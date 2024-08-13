#
#===========================================================#
export GalerkinCollocation

mutable struct GalerkinCollocation{Tprob,Tmodel,Ttimealg,Txyz,Tlincache} <: AbstractSolveScheme
    prob::Tprob
    model::Tmodel
    timealg::Ttimealg

    xyz::Txyz
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

        linprob = LinearProblem(J, vec(f))
        lincache = SciMLBase.init(linprob, linalg)

        new{
            typeof(prob), typeof(model), typeof(timealg), typeof(xyz),
            typeof(lincache),
        }(
            prob, model, timealg, xyz, lincache,
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
    f = dudtRHS(l.prob, l.model, l.xyz, p, t)
    J = dudp(l.model, l.xyz, p)

    l.lincache.A = J
    l.lincache.b = vec(f)
    l.lincache.u = similar(l.lincache.u)

    # print linsolve stats

    solve!(l.lincache)
    l.lincache.u
end

# function (l::GalerkinCollocation)(
#     dp::AbstractVector,
#     p::AbstractVector,
#     params,
#     t::Number,
# )
# end

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
