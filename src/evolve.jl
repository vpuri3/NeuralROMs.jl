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

        _p, _t = timeinteg_isimplicit(timestepper) ? (p, t1) : (p0, t0)

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

timeinteg_isimplicit(::EulerForward) = false
timeinteg_isimplicit(::EulerBackward) = true

function compute_residual(
    ::Union{EulerForward, EulerBackward},
    Δt::Real,
    u0::AbstractArray,
    u1::AbstractArray,
    rhs::AbstractArray,
)
    u1 - u0 - Δt * rhs |> vec
end

timeinteg_order(::Union{EulerForward, EulerBackward}) = 1

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

@concrete mutable struct LeastSqPetrovGalerkin{T} <: AbstractSolveScheme
    nlsolve
    residual
    maxiters
    abstolInf::T # ||∞
    abstol2::T   # ||₂
end

@concrete mutable struct Galerkin{T} <: AbstractSolveScheme
    linsolve
    abstolInf::T # ||∞
    abstolMSE::T # ||₂
end

#===========================================================#
@concrete mutable struct TimeIntegrator
    prob
    model
    timealg
    scheme

    x
    tprevs
    pprevs
    uprevs
    fprevs

    adaptive::Bool
    autodiff
    ϵ
end

function TimeIntegrator(
    prob::AbstractPDEProblem,
    model::AbstractNeuralModel,
    timealg::AbstractTimeStepper,
    scheme::AbstractSolveScheme,
    x::AbstractArray{T}, # TODO
    t0::T,
    p0::AbstractVector;
    adaptive::Bool = true,
    autodiff = AutoForwardDiff(),
    ϵ = nothing,
) where{T}
    u0 = model(x, p0)
    f0 = dudtRHS(prob, model, x, p0, t0; autodiff, ϵ)

    torder = timeinteg_order(timealg)

    tprevs = (t0, (T(NaN) for _ in 1:torder-1)...)
    pprevs = (p0, (fill!(similar(p0), NaN) for _ in 1:torder-1)...)
    uprevs = (u0, (fill!(similar(u0), NaN) for _ in 1:torder-1)...)
    fprevs = (f0, (fill!(similar(f0), NaN) for _ in 1:torder-1)...)

    TimeIntegrator(
        prob, model, timealg, scheme,
        x, tprevs, pprevs, uprevs, fprevs,
        adaptive, autodiff, ϵ,
    )
end

function update_integrator!(int::TimeIntegrator,
    t::T,
    p::AbstractArray{T}, # TODO- AbstractVector
    u::AbstractArray{T},
    f::AbstractArray{T},
) where{T<:Real}

    int.tprevs = (t, int.tprevs[1:end-1]...)
    int.pprevs = (p, int.pprevs[1:end-1]...)
    int.uprevs = (u, int.uprevs[1:end-1]...)
    int.fprevs = (f, int.fprevs[1:end-1]...)

    int
end

function solve_timestep(int::TimeIntegrator, Δt::Real)
    solve_timestep(int, int.scheme, int.timealg, Δt)
end

function solve_timestep(
    int::TimeIntegrator,
    scheme::Galerkin,
    timealg::AbstractTimeStepper,
    Δt::T
) where{T}

    @unpack tprevs, pprevs, uprevs, fprevs = int
    @unpack prob, model, x = int
    @unpack autodiff, ϵ = int

    xbatch = reshape(x, 1, :)

    t0, p0, u0, f0 = getindex.((tprevs, pprevs, uprevs, fprevs), 1)

    # du/dp (N, n), du/dt (N,)
    J0 = dudp(model, xbatch, p0; autodiff, ϵ)
    f0 = dudtRHS(prob, model, xbatch, p0, t0; autodiff, ϵ)

    # solve
    dpdt0 = J0 \ vec(f0)

    # get new states
    t1 = t0 + Δt
    p1 = apply_timestep(timealg, Δt, p0, dpdt0)
    u1 = model(xbatch, p1)
    f1 = dudtRHS(prob, model, xbatch, p1, t1)

    # compute residual stats
    r1 = compute_residual(timealg, Δt, u0, u1, f1)

    # print message

    print("Linear Steps: $(0), ")
    print_resid_stats(r1, scheme.abstolMSE, scheme.abstolInf)

    t1, p1, u1, f1, r1, Δt
end

function print_resid_stats(r::AbstractArray, abstolMSE, abstolInf)
    mse_r = sum(abs2, r) / length(r)
    inf_r = norm(r, Inf)

    color_mse = mse_r <= abstolMSE ? :green : :red
    color_inf = inf_r <= abstolInf ? :green : :red

    printstyled("MSE: $(round(mse_r, sigdigits = 8)) ", color = color_mse)
    printstyled("||∞: $(round(inf_r, sigdigits = 8))\n", color = color_inf)
    return
end

function perform_timestep!(
    integrator::TimeIntegrator,
    Δt::T;
    Δt_min = T(1e-5),
) where{T}

    t1, p1, u1, f1, r1 = solve_timestep(integrator, Δt)

    @unpack scheme, timealg = integrator
    @unpack abstolInf, abstolMSE = scheme

    mse_r = sum(abs2, r1) / length(r1)
    inf_r = norm(r1, Inf)

    if integrator.adaptive
        while (mse_r > abstolMSE) | (inf_r > abstolInf)

            if Δt < Δt_min
                printstyled("MINIMUM Δt = $Δt reached.\n", color = :red)
                break
            end

            Δt /= T(2f0)

            t1, p1, u1, f1, r1 = solve_timestep(integrator, Δt)

            mse_r = sum(abs2, r1) / length(r1)
            inf_r = norm(r1, Inf)
        end
    else

    end

    update_integrator!(integrator, t1, p1, u1, f1)

    return t1, p1, u1, f1, r1, Δt
end

#===========================================================#
#
