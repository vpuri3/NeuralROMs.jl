#
#===================================================#
indims(::AbstractPDEProblem{D}) where{D} = D

#===================================================#
struct Advection1D{T} <: AbstractPDEProblem{1}
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

    @. -(c * udx)
end

#===================================================#
struct Advection2D{T} <: AbstractPDEProblem{2}
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

    _, (udx, udy) = dudx1_2D(model, x, p; autodiff, ϵ)

    @. -(cx * udx + cy * udy)
end

#===================================================#
struct AdvectionDiffusion1D{T} <: AbstractPDEProblem{1}
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
struct BurgersInviscid1D <: AbstractPDEProblem{1}
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
struct BurgersViscous1D{T} <: AbstractPDEProblem{1}
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
struct BurgersViscous2D{T} <: AbstractPDEProblem{2}
    ν::T
end

function dudtRHS(
    prob::BurgersViscous2D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    ν = prob.ν

    uv, (uvdx, uvdy), (uvdxx, uvdyy) = dudx2_2D(model, x, p; autodiff, ϵ)

    u = getindex(uv, 1:1, :)
    v = getindex(uv, 2:2, :)

    udx = getindex(uvdx, 1:1, :)
    vdx = getindex(uvdx, 2:2, :)

    udy = getindex(uvdy, 1:1, :)
    vdy = getindex(uvdy, 2:2, :)

    # diffusion:
    # ν * (udxx + udyy)
    # ν * (vdxx + vdyy)

    diffusion = ν * (uvdxx + uvdyy)

    # convection:
    #   u * udx + v * udy
    #   u * vdx + v * vdy

    conv_x = @. u * udx + v * udy
    conv_y = @. u * vdx + v * vdy

    convection = vcat(conv_x, conv_y)

    diffusion - convection
end

#===================================================#
struct KuramotoSivashinsky1D{T} <: AbstractPDEProblem{1}
    ν::T
end

function dudtRHS(
    prob::KuramotoSivashinsky1D,
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector,
    t::Real;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    ν = prob.ν
    u, ud1x, ud2x, _, ud4x = dudx4_1D(model, x, p; autodiff, ϵ)

    #  -lapl (anti-diffusion) + biharmonic + convection
    @. -ud2x - ν * ud4x - (u * ud1x)
end

#===================================================#

