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

Algorithm:
- `compute_residual(timealg, Δt, fprevs, uprevs, tprevs, f, u, f_func) -> LHS - RHS`
   `f, u` at postulated timestep
   `f_func: (u, t) -> f(u, t)` to be used in multistage methods like RK
   Pass in `f_func = nothing` if not to be used

- `apply_timestep(timealg, Δt, fprevs, uprevs, tprevs, f, f_func) -> u_n+1`
   `f` at postulated timestep. Pass in `f = nothing` for explicit timestepper
   `f_func: (u, t) -> f(u, t)` to be used in multistage methods like RK

Traits:
- `isimplicit(timealg)`: if `b_-1 = 0`
- `adaptiveΔt(timealg)`: if adaptive `Δt` is supported
- `nsavedstates(timealg)`: number of saved steps (not including present step)
"""

adaptiveΔt(::AbstractTimeAlg) = true # default

#===========================================================#
# Euler FWD/BWD
#===========================================================#
struct EulerForward <: AbstractTimeAlg end
struct EulerBackward <: AbstractTimeAlg end

isimplicit(::EulerForward) = false
isimplicit(::EulerBackward) = true

nsavedstates(::Union{EulerForward, EulerBackward}) = 0

###
# compute_residual
###

function compute_residual(
    ::EulerForward,
    Δt::T,
    fprevs::NTuple{N, AbstractArray{T}},
    uprevs::NTuple{N, AbstractArray{T}},
    tprevs::NTuple{N, T},
    f::AbstractArray,
    u::AbstractArray,
    f_func,
) where{N,T<:Number}
    lhs = u - uprevs[1]
    rhs = Δt * fprevs[1]
    lhs - rhs
end

function compute_residual(
    ::EulerBackward,
    Δt::T,
    fprevs::NTuple{N, AbstractArray{T}},
    uprevs::NTuple{N, AbstractArray{T}},
    tprevs::NTuple{N, T},
    f::AbstractArray,
    u::AbstractArray,
    f_func,
) where{N,T<:Number}
    lhs = u - uprevs[1]
    rhs = Δt * f
    lhs - rhs
end

###
# apply_timestep
###

function apply_timestep(
    ::EulerForward,
    Δt::T,
    fprevs::NTuple{N, AbstractArray{T}},
    uprevs::NTuple{N, AbstractArray{T}},
    tprevs::NTuple{N, T},
    f::Union{Nothing, AbstractArray},
    f_func,
) where{N,T<:Number}
    uprevs[1] + Δt * fprevs[1]
end

function apply_timestep(
    ::EulerBackward,
    Δt::T,
    fprevs::NTuple{N, AbstractArray{T}},
    uprevs::NTuple{N, AbstractArray{T}},
    tprevs::NTuple{N, T},
    f::Union{Nothing, AbstractArray},
    f_func,
) where{N,T<:Number}
    uprevs[1] + Δt * f
end

#===========================================================#
# RK
#===========================================================#
abstract type AbstractRKMethod <: AbstractTimeAlg end
struct RK2 <: AbstractRKMethod end # midpoint
struct RK4 <: AbstractRKMethod end

nsavedstates(::AbstractRKMethod) = 0
isimplicit(::AbstractRKMethod) = false

###
# apply_timestep
###

function apply_timestep(
    ::RK2,
    Δt::T,
    fprevs::NTuple{N, AbstractArray{T}},
    uprevs::NTuple{N, AbstractArray{T}},
    tprevs::NTuple{N, T},
    f::Union{Nothing, AbstractArray},
    f_func,
) where{N,T<:Number}
    k1 = Δt * fprevs[1]
    k2 = Δt * f_func(uprevs[1] + k1 / T(2), tprevs[1] + Δt / T(2))

    uprevs[1] + k2
end

function apply_timestep(
    ::RK4,
    Δt::T,
    fprevs::NTuple{N, AbstractArray{T}},
    uprevs::NTuple{N, AbstractArray{T}},
    tprevs::NTuple{N, T},
    f::Union{Nothing, AbstractArray},
    f_func,
) where{N,T<:Number}
    k1 = Δt * fprevs[1]
    k2 = Δt * f_func(uprevs[1] + k1 / T(2), tprevs[1] + Δt / T(2))
    k3 = Δt * f_func(uprevs[1] + k2 / T(2), tprevs[1] + Δt / T(2))
    k4 = Δt * f_func(uprevs[1] + k3       , tprevs[1] + Δt       )

    uprevs[1] + k2 / T(6) + k2 / T(3) + k3 / T(3) + k4 / T(6)
end

###
# compute_residual
###

function compute_residual(
    ::Union{RK2, RK4},
    Δt::T,
    fprevs::NTuple{N, AbstractArray{T}},
    uprevs::NTuple{N, AbstractArray{T}},
    tprevs::NTuple{N, T},
    f::AbstractArray,
    u::AbstractArray,
    f_func,
) where{N,T<:Number} # RK1 residual
    zero(T) * u
end

#===========================================================#
# AB
#===========================================================#
abstract type AbstractABMethod <: AbstractTimeAlg end
struct AB2 <: AbstractTimeAlg end

adaptiveΔt(::AB2) = false
nsavedstates(::AbstractABMethod) = 1
isimplicit(::AbstractABMethod) = false

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
    abstolInf::T # ||∞ # TODO GalerkinProjection - switch to reltol
    abstolMSE::T # ||₂
end

function solve_timestep(
    integrator::TimeIntegrator{T},
    scheme::GalerkinProjection;
    verbose::Bool = true,
) where{T}

    @unpack prob, model, x, timealg, lincache = integrator
    @unpack autodiff_jac, autodiff_xyz, ϵ_jac, ϵ_xyz = integrator
    @unpack Δt, pprevs, uprevs, tprevs, fprevs, f̃prevs = integrator

    p1 = if isimplicit(timealg)
        @error "GalerkinProjection with implicit time-integrator is not implemetned"
    else
        f̃1 = nothing
        kws = (; autodiff_jac, autodiff_xyz, ϵ_jac, ϵ_xyz)
        dpdt_rhs(p, t) = compute_f̃(p, x, t, prob, model, lincache; kws...)
        apply_timestep(timealg, Δt, f̃prevs, pprevs, tprevs, f̃1, dpdt_rhs)
    end

    # get new states
    t1 = get_time(integrator) + Δt
    u1 = model(x, p1)
    f1 = dudtRHS(prob, model, x, p1, t1; autodiff = autodiff_xyz, ϵ = ϵ_xyz)
    f̃1 = compute_f̃(f1, p1, x, model, lincache; autodiff_jac, ϵ_jac)
    r1 = compute_residual(timealg, Δt, fprevs, uprevs, tprevs, f1, u1, nothing) #TODO

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
    p::AbstractVector,
    x::AbstractArray,
    t::Real,
    prob::AbstractPDEProblem,
    model::AbstractNeuralModel,
    lincache::LinearSolve.LinearCache;
    autodiff_jac = AutoForwardDiff(),
    autodiff_xyz = AutoForwardDiff(),
    ϵ_jac = nothing,
    ϵ_xyz = nothing,
)
    f = dudtRHS(prob, model, x, p, t; autodiff = autodiff_xyz, ϵ = ϵ_xyz)
    compute_f̃(f, p, x, model, lincache; autodiff_jac, ϵ_jac,)
end

function compute_f̃(
    f::AbstractVecOrMat,
    p::AbstractVector,
    x::AbstractArray,
    model::AbstractNeuralModel,
    lincache::LinearSolve.LinearCache;
    autodiff_jac = AutoForwardDiff(),
    ϵ_jac = nothing,
)
    # du/dp (N, n)
    J = dudp(model, x, p; autodiff = autodiff_jac, ϵ = ϵ_jac)

    lincache.A = J
    lincache.b = vec(f)
    lincache.u = similar(lincache.u)

    solve!(lincache)
    lincache.u
end

#===========================================================#

@concrete mutable struct LeastSqPetrovGalerkin{T} <: AbstractSolveScheme
    nlssolve
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

    #=
    TODO LSPG: Do LineSearch in GaussNewton solve. Follow CROM implementation.
    TODO LSPG: See if regularizing codes improves performance.
    =#

    @unpack prob, model, x, timealg, Δt = integrator
    @unpack autodiff_nls, autodiff_xyz, ϵ_nls, ϵ_xyz = integrator

    t0, p0, u0, _, _ = get_state(integrator)

    batch = (x, u0)
    residual = make_residual(integrator)

    # solve
    p1, nlssol = nonlinleastsq(
        model, p0, batch, scheme.nlssolve;
        residual,
        maxiters = scheme.nlsmaxiters,
        abstol = scheme.abstolnls,
    )

    # get new states
    t1 = t0 + Δt
    u1 = model(x, p1)
    f1 = dudtRHS(prob, model, x, p1, t1; autodiff = autodiff_xyz, ϵ = ϵ_xyz)
    f̃1 = nothing
    r1 = nlssol.resid

    # print message
    steps = nlssol.stats.nsteps
    if verbose
        print("Nonlinear Steps: $steps, ")
        print_resid_stats(r1, scheme.abstolMSE, scheme.abstolInf)
        print(" , RetCode: $(nlssol.retcode)\n")
    end

    t1, p1, u1, f1, f̃1, r1
end

#===========================================================#
function make_residual(
    integrator::TimeIntegrator,
)
    @unpack Δt, uprevs, fprevs, tprevs = integrator
    @unpack prob, timealg, autodiff_xyz, ϵ_xyz = integrator

    function make_residual_internal(
        model::AbstractNeuralModel,
        p1::AbstractVector,
        batch::NTuple{2, Any},
        nlsp,
    )
        x, _ = batch
        t1 = get_time(integrator) + Δt

        u1 = model(x, p1)
        f1 = dudtRHS(prob, model, x, p1, t1; autodiff = autodiff_xyz, ϵ = ϵ_xyz)

        function dudt_rhs(u, t)
            if timealg isa AbstractRKMethod
                @error "Not implemented LSPG for multistage methods yet."
            end
        end

        compute_residual(timealg, Δt, fprevs, uprevs, tprevs, f1, u1, dudt_rhs)
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
    printstyled("||∞: $(round(inf_r, sigdigits = 8)) ", color = color_inf)

    if isnan(mse_r) | isnan(inf_r)
        println()
        throw(ErrorException("Residual has NaN"))
    end

    return
end

#===========================================================#
#
