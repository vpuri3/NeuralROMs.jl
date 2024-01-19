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
- `apply_timestep(timealg, Δt, fprevs, uprevs, f) -> u_n+1`
- `apply_timestep(timealg, fterm, uprevs) -> u_n+1`
"""

adaptiveΔt(::AbstractTimeAlg) = true # default

function compute_residual(
    timealg::AbstractTimeAlg,
    Δt::T,
    fprevs::NTuple{N, Tf},
    uprevs::NTuple{N, Tu},
    f::Tf,
    u::Tu,
) where{N,T<:Number,Tf<:AbstractArray{T}, Tu<:AbstractArray{T}}
    lhs = make_unew_term(timealg, u) + make_uprev_term(timealg, uprevs)
    rhs = make_f_term(timealg, Δt, fprevs, f)
    rhs - lhs
end

function apply_timestep(
    timealg::AbstractTimeAlg,
    Δt::T,
    fprevs::NTuple{N, Tf},
    uprevs::NTuple{N, Tu},
    f::Union{Nothing, Tf},
) where{N,T<:Number,Tf<:AbstractArray{T}, Tu<:AbstractArray{T}}
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

# EulerFWD
function make_f_term(
    ::EulerForward,
    Δt::T,
    fprevs::NTuple{N, Tv},
    f::Union{Nothing, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    Δt * (fprevs[1])
end

# EulerBWD
function make_f_term(
    ::EulerBackward,
    Δt::T,
    fprevs::NTuple{N, Tv},
    f::Union{Nothing, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    Δt * f
end

# EulerFWD/BWD
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
isimplicit(::AbstractRKMethod) = false

function make_f_term(
    ::RK2,
    Δt::T,
    fprevs::NTuple{N, Tv},
    f::Union{Nothing, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    # Δt * (fprevs[1])
end

function make_uprev_term(
    ::RK2,
    uprevs::NTuple{N, Tv},
) where{N,T<:Number,Tv<:AbstractArray{T}}
    # - uprevs[1]
end

function make_unew_term(
    ::RK2,
    u::Tv,
) where{T<:Number,Tv<:AbstractArray{T}}
    # u
end

#===========================================================#
#===========================================================#
# Solve Schemes
#===========================================================#
#===========================================================#

"""
# GalerkinProjection

original: `u' = f(u, t)`
ROM map : `u = g(ũ)`

`⟹  J(ũ) *  ũ' = f(ũ, t)`

`⟹  ũ' = pinv(J)  (ũ) * f(ũ, t)`

solve with timestepper
`⟹  ũ' = f̃(ũ, t)`

e.g.
`(J*u)_n+1 - (J*u)_n = Δt * (f_n + f_n-1 + ...)`
"""
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
    @unpack Δt, pprevs, uprevs, fprevs, f̃prevs = integrator
    @unpack prob, model, x = integrator

    t0, _, _, _, _ = get_state(integrator)

    if isimplicit(timealg)
        @error """GalerkinProjection with implicit time-integrator is
        not implemetned"""
    else
        f1 = nothing
        p1 = apply_timestep(timealg, Δt, f̃prevs, pprevs, f1)
    end

    # get new states
    t1 = t0 + Δt
    u1 = model(x, p1)
    f1 = dudtRHS(prob, model, x, p1, t1)
    f̃1 = compute_f̃(f1, p1, x, model, scheme; autodiff, ϵ)
    r1 = compute_residual(timealg, Δt, fprevs, uprevs, f1, u1)

    # print message
    steps = 0
    if verbose
        print("Linear Steps: $steps, ")
        print_resid_stats(r1, scheme.abstolMSE, scheme.abstolInf)
        println()
    end

    t1, p1, u1, f1, f̃1, r1
end

function compute_f̃(
    f::AbstractVecOrMat,
    p::AbstractVector,
    x::AbstractArray,
    model::AbstractNeuralModel,
    scheme::GalerkinProjection;
    autodiff = AutoForwardDiff(),
    ϵ = nothing
)
    J = dudp(model, x, p; autodiff, ϵ) # du/dp (N, n)

    linprob = LinearProblem(J, vec(f))
    linsol  = solve(linprob, scheme.linsolve)

    # TODO compute_f̃: add stats?
    
    linsol.u
end

#===========================================================#

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

    t0, p0, u0, _, _ = get_state(integrator)
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

    t1, p1, u1, f1, nothing, r1
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
