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
@concrete mutable struct TimeIntegrator{T}
    prob
    model
    timealg
    scheme

    x
    Δt::T
    Δt_guess::T

    tspan
    tsave

    tstep
    isave

    # state
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
    x::AbstractArray{T},
    tsave::AbstractVector{T},
    p0::AbstractVector;
    Δt::Union{T,Nothing} = nothing,
    adaptive::Bool = true,
    autodiff = AutoForwardDiff(),
    ϵ = nothing,
) where{T<:Number}

    if isnothing(Δt) & !adaptive
        error("Must provide Δt if `adaptive = $(adaptive)`")
    end

    Δt = isnothing(Δt) ? 1f-2 : Δt

    @assert issorted(tsave) "`tsave` must be a vector of strictly increasing entries."
    @assert length(tsave) > 1
    tspan = extrema(tsave)

    # next step flags
    tstep = 1
    isave = 2

    # current state
    t0 = first(tspan)
    u0 = model(x, p0)
    f0 = dudtRHS(prob, model, x, p0, t0; autodiff, ϵ)

    # previous states
    torder = timeinteg_order(timealg)
    tprevs = (t0, (T(NaN) for _ in 1:torder-1)...)
    pprevs = (p0, (fill!(similar(p0), NaN) for _ in 1:torder-1)...)
    uprevs = (u0, (fill!(similar(u0), NaN) for _ in 1:torder-1)...)
    fprevs = (f0, (fill!(similar(f0), NaN) for _ in 1:torder-1)...)

    TimeIntegrator(
        prob, model, timealg, scheme,
        x, Δt, Δt,
        tspan, tsave, tstep, isave,
        tprevs, pprevs, uprevs, fprevs,
        adaptive, autodiff, ϵ,
    )
end

function get_time(integrator::TimeIntegrator)
    integrator.tprevs[1]
end

function get_nexttime(integrator::TimeIntegrator)
    get_time(integrator) + get_Δt(integrator)
end

function get_Δt(integrator::TimeIntegrator)
    integrator.Δt
end

function get_tspan(integrator::TimeIntegrator)
    integrator.tspan
end

function get_state(int::TimeIntegrator)
    getindex.((int.tprevs, int.pprevs, int.uprevs, int.fprevs), 1) # t, p, u, f
end

function get_next_savetime(integrator::TimeIntegrator{T}) where{T}
    @unpack isave, tsave = integrator

    if isave > length(tsave)
        return typemax(T)
    else
        return getindex(tsave, isave)
    end
end

# call before perform_timestep!
function update_Δt_for_saving!(
    integrator::TimeIntegrator{T};
    tol::T = T(1e-6)
) where{T}
    tsv = get_next_savetime(integrator)
    if (tsv - tol) < get_nexttime(integrator)
        integrator.Δt = tsv - get_time(integrator)
    end
end

# call after perform_timestep!
function savestep!(
    integrator::TimeIntegrator{T};
    tol::T = T(1e-6),
    verbose::Bool = true,
) where{T}
    t = get_time(integrator)
    tsv = get_next_savetime(integrator)

    if abs(t - tsv) < tol

        println("SAVING STATE")

        integrator.isave += 1
        integrator.Δt = integrator.Δt_guess
        return get_state(integrator)
    else
        integrator.Δt_guess = integrator.Δt
        return nothing
    end
end

function update_integrator!(integrator::TimeIntegrator,
    t::T,
    p::AbstractArray{T},
    u::AbstractArray{T},
    f::AbstractArray{T},
) where{T<:Real}

    integrator.tstep += 1

    integrator.tprevs = (t, integrator.tprevs[1:end-1]...)
    integrator.pprevs = (p, integrator.pprevs[1:end-1]...)
    integrator.uprevs = (u, integrator.uprevs[1:end-1]...)
    integrator.fprevs = (f, integrator.fprevs[1:end-1]...)

    integrator
end

function perform_timestep!(
    integrator::TimeIntegrator{T};
    Δt_min::T = T(1e-5),
    tol_scale::T = T(10),
    verbose::Bool = true,
) where{T}

    if verbose
        tstep = integrator.tstep
        t_print  = round(get_nexttime(integrator); sigdigits=6)
        Δt_print = round(get_Δt(integrator); sigdigits=6)
        println("#============================#")
        println("Time Step: $tstep, Time: $t_print, Δt: $Δt_print")
    end

    t1, p1, u1, f1, r1 = solve_timestep(integrator; verbose)

    @unpack scheme, timealg = integrator
    @unpack abstolInf, abstolMSE = scheme

    if integrator.adaptive
        mse_r = sum(abs2, r1) / length(r1)
        inf_r = norm(r1, Inf)

        while (mse_r > abstolMSE) | (inf_r > abstolInf)

            if integrator.Δt < Δt_min
                if verbose
                    printstyled("MINIMUM Δt = $(integrator.Δt) reached.\n",
                        color = :red)
                end
                break
            end

            integrator.Δt /= T(2f0)

            t1, p1, u1, f1, r1 = solve_timestep(integrator; verbose)

            mse_r = sum(abs2, r1) / length(r1)
            inf_r = norm(r1, Inf)
        end

        if (tol_scale * mse_r < abstolMSE) & (tol_scale * inf_r < abstolInf)
            if verbose
                printstyled("Tolerances are well satisfied. Bumping up Δt.\n",
                    color = :green)
            end
            integrator.Δt *= T(1.50)
        end
    end

    update_integrator!(integrator, t1, p1, u1, f1)

    return r1
end

function solve_timestep(
    integrator::TimeIntegrator;
    verbose::Bool = true
)
    solve_timestep(integrator, integrator.scheme; verbose)
end

function evolve_integrator(
    integrator::TimeIntegrator{T};
    verbose::Bool = true,
) where{T}

    ts = ()
    ps = ()
    us = ()

    begin
        t, p, u, _ = get_state(integrator)
        ts = (ts..., t)
        ps = (ps..., p)
        us = (us..., u)
    end

    # Time loop
    while get_time(integrator) <= get_tspan(integrator)[2]

        update_Δt_for_saving!(integrator)

        perform_timestep!(integrator; verbose)

        state = savestep!(integrator)

        if !isnothing(state)
            t, p, u, _ = state
            ts = (ts..., t)
            ps = (ps..., p)
            us = (us..., u)
        end
    end

    ts = [ts...]
    ps = mapreduce(getdata, hcat, ps) |> Lux.cpu_device()
    us = mapreduce(adjoint, hcat, us) |> Lux.cpu_device()

    @assert norm(integrator.tsave - ts, Inf) < 1e-6

    return ts, ps, us
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

    # du/dp (N, n), du/dt (N,)
    J0 = dudp(model, x, p0; autodiff, ϵ)
    # f0 = dudtRHS(prob, model, x, p0, t0; autodiff, ϵ) # already computed above

    # solve
    dpdt0 = J0 \ vec(f0)

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

    # get stuff jankily
    t0, p0, u0, f0 = getindex.((tprevs, pprevs, uprevs, fprevs), 1)
    batch = (x, u0)

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
#
