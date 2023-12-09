#
#===========================================================#
#=
# timealg interface

make_RHS_term(timealg, Δt, fprevs)

compute_residual(timealg, Δt, unew, fnew, uprevs, fprevs)

apply_timestep(timealg, uprevs, fprevs, Δt) -> unew
apply_timestep(timealg, uprevs, rhs_term) -> unew

=#

# function apply_timestep(
#     timealg::AbstractTimeStepper,
#     uprevs::NTuple{N, Tv},
#     fprevs::NTuple{N, Tv},
#     f::Tv, # unused for explicit
#     Δt::T,
# ) where{N,T,Tv<:AbstractArray{T}}
#     rhs_term = make_RHS_term(timealg, Δt, f, fprevs)
#     apply_timestep(timealg, uprevs, rhs_term)
# end

abstract type AbstractRKMethod <: AbstractTimeStepper end

struct EulerForward <: AbstractTimeStepper end
struct EulerBackward <: AbstractTimeStepper end

struct AB2 <: AbstractTimeStepper end

struct RK2 <: AbstractRKMethod end # midpoint
struct RK4 <: AbstractRKMethod end

isimplicit(::EulerForward) = false
isimplicit(::EulerBackward) = true

adaptiveΔt(::AbstractTimeStepper) = true
adaptiveΔt(::AB2) = false

nsavedstates(::Union{EulerForward, EulerBackward}) = 1
nsavedstates(::AbstractRKMethod) = 1

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

function print_resid_stats(r::AbstractArray, abstolMSE, abstolInf)
    mse_r = sum(abs2, r) / length(r)
    inf_r = norm(r, Inf)

    color_mse = mse_r <= abstolMSE ? :green : :red
    color_inf = inf_r <= abstolInf ? :green : :red

    printstyled("MSE: $(round(mse_r, sigdigits = 8)) ", color = color_mse)
    printstyled("||∞: $(round(inf_r, sigdigits = 8))", color = color_inf)
    return
end

#===========================================================#

@concrete mutable struct Galerkin{T} <: AbstractSolveScheme
    linsolve
    abstolInf::T # ||∞ # TODO - switch to reltol
    abstolMSE::T # ||₂
end

function solve_timestep(
    integrator::TimeIntegrator{T},
    scheme::Galerkin;
    verbose::Bool = true,
) where{T}

    @unpack timealg, autodiff, ϵ = integrator
    @unpack Δt, tprevs, pprevs, uprevs, fprevs = integrator
    @unpack prob, model, x = integrator

    t0, p0, u0, f0 = get_state(integrator)

    #=
    rhs_term = J0 \ make_RHS_term(timealg, fprevs, Δt)
    p1 = apply_timestep(timealg, pprevs, rhs_term)

    # is it Euler FWD
    (J*u)_n+1 - (J*u)_n = Δt * (f_n + f_n-1 + ...)

    OR

    J_n+1 * (u_n+1 - u_n) = Δt * (f_n + f_n-1 + ...)
    =#

    # du/dp (N, n), du/dt (N,)
    J0 = dudp(model, x, p0; autodiff, ϵ)

    # solv7
    linprob = LinearProblem(J0, vec(f0))
    dpdt0 = solve(linprob, scheme.linsolve).u

    # get new states
    t1 = t0 + Δt
    p1 = apply_timestep(timealg, Δt, p0, dpdt0)
    u1 = model(x, p1)
    f1 = dudtRHS(prob, model, x, p1, t1)

    # compute residual stats
    r1 = compute_residual(timealg, Δt, u0, u1, f1)

    # print message
    steps = 0
    if verbose
        print("Linear Steps: $steps, ")
        print_resid_stats(r1, scheme.abstolMSE, scheme.abstolInf)
        println()
    end

    t1, p1, u1, f1, r1
end

@concrete mutable struct LeastSqPetrovGalerkin{T} <: AbstractSolveScheme
    nlssolve
    nlsresidual
    nlsmaxiters
    abstolnls::T
    abstolInf::T # ||∞
    abstolMSE::T # ||₂
end

function solve_timestep(
    integrator::TimeIntegrator{T},
    scheme::LeastSqPetrovGalerkin;
    verbose::Bool = true,
) where{T}

    @unpack timealg, autodiff, ϵ = integrator
    @unpack Δt, tprevs, pprevs, uprevs, fprevs = integrator
    @unpack prob, model, x = integrator

    t0, p0, u0, f0 = get_state(integrator)
    batch = (x, u0)

    #=
    Next steps: Do LineSearch in GaussNewton solve. Follow CROM implementation.
    =#

    t1 = t0 + Δt
    nlsp = t1, Δt, t0, p0, u0

    # solve
    p1, nlssol = nonlinleastsq(
        model, p0, batch, scheme.nlssolve;
        nlsp,
        residual = scheme.nlsresidual,
        maxiters = scheme.nlsmaxiters,
        abstol = scheme.abstolnls,
    )

    # get new states
    u1 = model(x, p1)
    f1 = dudtRHS(prob, model, x, p1, t1)

    # compute residual stats
    r1 = nlssol.resid # compute_residual(timealg, Δt, u0, u1, f1)

    # print message
    steps = nlssol.stats.nsteps
    if verbose
        print("Nonlinear Steps: $steps, ")
        print_resid_stats(r1, scheme.abstolMSE, scheme.abstolInf)
        print(" , RetCode: $(nlssol.retcode)\n")
    end

    t1, p1, u1, f1, r1
end

#===========================================================#
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
#
