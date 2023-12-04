#
function make_residual(
    prob::AbstractPDEProblem,
    timestepper::AbstractTimeStepper;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function make_residual_internal(
        model::AbstractNeuralModel,
        p::AbstractVector,
        batch::NTuple{2, Any},
        nlsp,
    )
        x, _ = batch
        t1, Δt, t0, p0, u0 = nlsp

        _p, _t = isimplicit(timestepper) ? (p, t1) : (p0, t0)

        rhs = dudtRHS(prob, model, x, _p, _t; autodiff, ϵ)
        u1  = model(x, p)
        compute_residual(timestepper, Δt, u0, u1, rhs)
    end
end

function residual_learn(
        model::AbstractNeuralModel,
        p::AbstractVector,
        batch::NTuple{2, Any},
        nlsp,
    )
    x, û = batch
    u = model(x, p)
    vec(û - u)
end

#===========================================================#
struct EulerForward <: AbstractTimeStepper end
struct EulerBackward <: AbstractTimeStepper end

isimplicit(::EulerForward) = false
isimplicit(::EulerBackward) = true

function compute_residual(
    ::Union{EulerForward, EulerBackward},
    Δt::Real,
    u0::AbstractArray,
    u1::AbstractArray,
    rhs::AbstractArray,
)
    u1 - u0 - Δt * rhs |> vec
end

function apply_timestep(
    ::Union{EulerForward, EulerBackward},
    Δt::Real,
    u0::AbstractArray,
    rhs::AbstractArray,
)
    @. u0 + Δt * rhs
end

#===========================================================#
abstract type AbstractSolveScheme end

mutable struct PODGalerkin{T} <: AbstractSolveScheme
    nlsolve
    abstol::T
    reltol::T
end

implicit_timestepper(::PODGalerkin) = false

@concrete mutable struct LeastSqPetrovGalerkin{T} <: AbstractSolveScheme # init cache
    linsolve
    implicit_timestepper::Bool
end

implicit_timestepper(scheme::LeastSqPetrovGalerkin) = scheme.implicit_timestepper

@concrete mutable struct TimeIntegrator
    scheme
    tsteper

    Δt
    times
    ps
    Us
end

#===========================================================#

#===========================================================#
#
