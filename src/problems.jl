#
#===================================================#
struct Advection1D{T} <: AbstractPDEProblem
    c::T # make it function of time c(x, t)
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

#===================================================#
struct AdvectionDiffusion1D{T} <: AbstractPDEProblem
    c::T
    ν::T
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

#===================================================#
struct BurgersInviscid1D <: AbstractPDEProblem
end

function dudtRHS(
    prob::BurgersInviscid1D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    u, udx = dudx1(model, x, p; autodiff, ϵ)

    @. -u * udx
end

#===================================================#
struct BurgersViscous1D{T} <: AbstractPDEProblem
    ν::T
end
#===================================================#

