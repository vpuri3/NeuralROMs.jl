#
#===========================================================#
"""
ODE Algorithms for solving

```math
du/dt = f(u, t)
```

to be discretized as
```math
a_-1 * u_n+1 + ... + a_k * u_n-k = Δt ⋅ (b_-1 + f(u_n+1) + ... + b_k * f(u_n-k))

⟺
∑_{i=-1}^k a_i *  u_n+i = Δt * ∑_{i=-1}^k b_i f(u_i)
```

# Interface

## overload
- `isimplicit(timealg)`: if `b_-1 = 0`
- `adaptiveΔt(timealg)`: if adaptive `Δt` is supported
- `nsavedsteps(timealg)`: number of saved time-steps
- `make_f_term(timealg, Δt, fprevs, f)`
- `make_uprev_term(timealg, uprevs)`
- `make_unew_term(timealg, u)`

## implemented
- `compute_residual(timealg, Δt, fprevs, uprevs, f, u) -> LHS - RHS`
- `apply_timestep(timealg, Δt, fprevs, uprevs, f)`:
- `apply_timestep(timealg, fterm, uprevs)`:
"""

adaptiveΔt(::AbstractTimeAlg) = true # default

function compute_residual(
    timealg::AbstractTimeAlg,
    Δt::T,
    fprevs::NTuple{N, Tv},
    uprevs::NTuple{N, Tv},
    f::AbstractArray{T},
    u::AbstractArray{T},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    lhs = make_unew_term(timealg, u) + make_uprev_term(timealg, uprevs)
    rhs = make_f_term(timealg, Δt, fprevs, f)
    rhs - lhs
end

function apply_timestep(
    timealg::AbstractTimeAlg,
    Δt::T,
    fprevs::NTuple{N, Tv},
    uprevs::NTuple{N, Tv},
    f::Union{Nothing, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    fterm = make_f_term(timealg, Δt, fprevs, f)
    apply_timestep(timealg, fterm, uprevs)
end

function apply_timestep(
    timealg::AbstractTimeAlg,
    fterm::AbstractArray{T},
    uprevs::NTuple{N, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    fterm - make_uprev_term(timealg, uprevs)
end

#===========================================================#
# Euler FWD/BWD
#===========================================================#
struct EulerForward <: AbstractTimeAlg end
struct EulerBackward <: AbstractTimeAlg end

isimplicit(::EulerForward) = false
isimplicit(::EulerBackward) = true

nsavedstates(::Union{EulerForward, EulerBackward}) = 1

function make_f_term(
    ::EulerForward,
    Δt::T,
    fprevs::NTuple{N, Tv},
    f::Union{Nothing, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    Δt * (fprevs[1])
end

function make_f_term(
    ::EulerBackward,
    Δt::T,
    fprevs::NTuple{N, Tv},
    f::Union{Nothing, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    Δt * f
end

function make_uprev_term(
    ::Union{EulerForward, EulerBackward},
    uprevs::NTuple{N, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    - uprevs[1]
end

function make_unew_term(
    ::Union{EulerForward, EulerBackward},
    u::Tv,
) where{T<:Number,Tv<:AbstractArray{T}}
    u
end

#===========================================================#
# AB
#===========================================================#
abstract type AbstractABMethod <: AbstractTimeAlg end
struct AB2 <: AbstractTimeAlg end

adaptiveΔt(::AB2) = false

#===========================================================#
# RK
#===========================================================#
abstract type AbstractRKMethod <: AbstractTimeAlg end
struct RK2 <: AbstractRKMethod end # midpoint
struct RK4 <: AbstractRKMethod end

nsavedstates(::AbstractRKMethod) = 1

#===========================================================#
#===========================================================#
# Solve Schemes
#===========================================================#
#===========================================================#

@concrete mutable struct GalerkinProjection{T} <: AbstractSolveScheme
    linsolve
    abstolInf::T # ||∞ # TODO - switch to reltol
    abstolMSE::T # ||₂
end

function solve_timestep(
    integrator::TimeIntegrator{T},
    scheme::GalerkinProjection;
    verbose::Bool = true,
) where{T}

    @unpack timealg, autodiff, ϵ = integrator
    @unpack Δt, tprevs, pprevs, uprevs, fprevs = integrator
    @unpack prob, model, x = integrator

    t0, p0, _, _ = get_state(integrator)

    # explicit time-integrator
    # ΔuΔt_rhs: RHS of discretized ODE
    f1 = nothing # This will error with implicit integrator
    ΔuΔt_rhs = make_f_term(timealg, Δt, fprevs, f1) # du/dt (N,)
    J0 = dudp(model, x, p0; autodiff, ϵ)            # du/dp (N, n)

    # this is FWD Euler
    linprob = LinearProblem(J0, vec(ΔuΔt_rhs))
    ΔpΔt_rhs = solve(linprob, scheme.linsolve).u


    #= GalerkinProjection

    original: u' = f(u, t)
    ROM map : u = g(ũ)

    ⟹  J(ũ) *  ũ' = f(ũ, t)

    ⟹  ũ' = pinv(J)  (ũ) * f(ũ, t)

    solve with timestepper
    ⟹  ũ' = f̃(ũ, t)

    e.g.
    (J*u)_n+1 - (J*u)_n = Δt * (f_n + f_n-1 + ...)

    =#

    # get new states
    t1 = t0 + Δt
    p1 = apply_timestep(timealg, ΔpΔt_rhs, pprevs)
    u1 = model(x, p1)
    f1 = dudtRHS(prob, model, x, p1, t1)
    r1 = compute_residual(timealg, Δt, fprevs, uprevs, f1, u1)

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
    TODO LSPG: Do LineSearch in GaussNewton solve. Follow CROM implementation.
    TODO LSPG: See if regularizing codes improves performance.
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
    timestepper::AbstractTimeAlg;
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
#
