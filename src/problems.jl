#
#===================================================#
struct Advection1D{T} <: AbstractPDEProblem
    c::T
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
    _, udx = dudx1_1D(model, x, p; autodiff, ϵ)

    @. -c * udx
end

#===================================================#
struct Advection2D{T} <: AbstractPDEProblem
    cx::T
    cy::T
end

function dudtRHS(
    prob::Advection2D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    cx, cy = prob.cx, prob.cy

    _, udx, udy = dudx1_2D(model, x, p; autodiff, ϵ)

    @. -(cx * udx + cy * udy)
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

    _, udx, udxx = dudx2_1D(model, x, p; autodiff, ϵ)

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
    u, udx = dudx1_1D(model, x, p; autodiff, ϵ)

    @. -u * udx
end

#===================================================#
struct BurgersViscous1D{T} <: AbstractPDEProblem
    ν::T
end

function dudtRHS(
    prob::BurgersViscous1D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    ν = prob.ν

    u, udx, udxx = dudx2_1D(model, x, p; autodiff, ϵ)

    @. -u * udx + ν * udxx
end

#===================================================#
struct KuramotoSivashinsky1D <: AbstractPDEProblem
end

function dudtRHS(
    ::KuramotoSivashinsky1D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    u, ud1x, ud2x, _, ud4x = dudx4_1D(model, x, p; autodiff, ϵ)

    @. -ud2x - ud4x - (u * ud1x) # -lapl (anti-diffusion), biharmonic, convection
end

#===================================================#

