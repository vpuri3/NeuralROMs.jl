struct Advection1D{T} <: AbstractPDEProblem
    c::T # make it function of time c(x, t)
end

struct AdvectionDiffusion1D{T} <: AbstractPDEProblem
    c::T
    ν::T
end

struct BurgersInviscid1D{T} <: AbstractPDEProblem
    ν::T
end

struct BurgersViscous1D{T} <: AbstractPDEProblem
    ν::T
end

function dudtRHS(
    prob::Advection1D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    c = prob.c
    _, udx = dudx1(model, x, p; autodiff, ϵ)

    @. -c * udx
end

function dudtRHS(
    prob::AdvectionDiffusion1D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    c = prob.c
    ν = prob.ν

    _, udx, udxx = dudx2(model, x, p; autodiff, ϵ)

    @. -c * udx + ν * udxx
end

