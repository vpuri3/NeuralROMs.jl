#===========================================================#
# type
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
    f̃prevs   # Galerkin only

    adaptive::Bool
    autodiff
    ϵ
end

function TimeIntegrator(
    prob::AbstractPDEProblem,
    model::AbstractNeuralModel,
    timealg::AbstractTimeAlg,
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

    if !adaptiveΔt(timealg) & adaptive
        error("Chosen timealg $timealg doesn't support adaptive Δt. \
            Pass in kwarg `adaptive=false`")
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

    f̃0 = if scheme isa GalerkinProjection
        compute_f̃(f0, p0, x, model, scheme; autodiff, ϵ)
    else
        p0 * NaN
    end

    # previous states
    nstates = nsavedstates(timealg) - 1
    tprevs = (t0, (T(NaN) for _ in 1:nstates)...)
    pprevs = (p0, (fill!(similar(p0), NaN) for _ in 1:nstates - 1)...)
    uprevs = (u0, (fill!(similar(u0), NaN) for _ in 1:nstates - 1)...)
    fprevs = (f0, (fill!(similar(f0), NaN) for _ in 1:nstates - 1)...)
    f̃prevs = (f̃0, (fill!(similar(f̃0), NaN) for _ in 1:nstates - 1)...)

    TimeIntegrator(
        prob, model, timealg, scheme,
        x, Δt, Δt,
        tspan, tsave, tstep, isave,
        tprevs, pprevs, uprevs, fprevs, f̃prevs,
        adaptive, autodiff, ϵ,
    )
end

#===========================================================#
# interface
#===========================================================#

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

"""
returns `t, p, u, f, f̃`
"""
function get_state(int::TimeIntegrator)
    getindex.((int.tprevs, int.pprevs, int.uprevs, int.fprevs, int.f̃prevs), 1)
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
    tol::T = T(1e-6),
    verbose::Bool = true,
) where{T}
    tsv = get_next_savetime(integrator)
    if (tsv - tol) < get_nexttime(integrator)
        if verbose
            printstyled("Reducing Δt for saving time-step.\n", color = :magenta)
        end
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

        if verbose
            time_print = round(get_time(integrator), sigdigits = 8)
            printstyled("Saving state at time $(time_print)\n",
                color = :magenta)
        end

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
    f̃::Union{AbstractArray{T}, Nothing},
) where{T<:Real}

    integrator.tstep += 1

    integrator.tprevs = (t, integrator.tprevs[1:end-1]...)
    integrator.pprevs = (p, integrator.pprevs[1:end-1]...)
    integrator.uprevs = (u, integrator.uprevs[1:end-1]...)
    integrator.fprevs = (f, integrator.fprevs[1:end-1]...)

    if !isnothing(f̃)
        integrator.f̃prevs = (f̃, integrator.f̃prevs[1:end-1]...)
    end

    integrator
end

function perform_timestep!(
    integrator::TimeIntegrator{T};
    Δt_min::T = T(5e-6),
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

    t1, p1, u1, f1, f̃1, r1 = solve_timestep(integrator; verbose)

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

    update_integrator!(integrator, t1, p1, u1, f1, f̃1)

    return r1
end

function solve_timestep(
    ::TimeIntegrator,
    scheme::AbstractSolveScheme;
    verbose::Bool = true,
)
    error("`solve_timestep` has not been implemented for type $scheme.")
end

function solve_timestep(
    integrator::TimeIntegrator;
    verbose::Bool = true
)
    solve_timestep(integrator, integrator.scheme; verbose)
end

function evolve_integrator!(
    integrator::TimeIntegrator{T};
    verbose::Bool = true,
) where{T}

    # save states at `integrator.tsave`
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

        state = savestep!(integrator; verbose)

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
function evolve_model(
    prob::AbstractPDEProblem,
    model::AbstractNeuralModel,
    timealg::AbstractTimeAlg,
    scheme::AbstractSolveScheme,
    data::NTuple{3, AbstractVecOrMat},
    p0::AbstractVector,
    Δt::Union{Real,Nothing} = nothing;
    nlssolve = nothing,
    nlsmaxiters = 10,
    nlsabstol = 1f-7,
    time_adaptive::Bool = true,
    autodiff_space = AutoForwardDiff(),
    ϵ_space = nothing,
    device = Lux.cpu_device(),
    verbose::Bool = true,
)
    # data
    x, u0, tsave = data

    # move to device
    (x, u0) = (x, u0) |> device
    model = model |> device
    p0 = p0 |> device
    T = eltype(p0)

    # solvers
    nlssolve = if isnothing(nlssolve)
        linsolve = QRFactorization()
        autodiff = AutoForwardDiff()
        linesearch = LineSearch() # TODO
        nlssolve = GaussNewton(;autodiff, linsolve, linesearch)
    else
        nlssolve
    end

    #============================#
    # learn IC
    #============================#
    if verbose
        println("#============================#")
        println("Time Step: $(0), Time: 0.0 - learn IC")
    end

    p0, _ = nonlinleastsq(model, p0, (x, u0), nlssolve;
        residual = residual_learn,
        maxiters = nlsmaxiters * 5,
        termination_condition = AbsTerminationMode(),
        abstol = nlsabstol,
        verbose,
    )
    u0 = model(x, p0)

    Δt = isnothing(Δt) ? -(reverse(extrema(tsave))...) / 200 |> T : T(Δt)

    integrator = TimeIntegrator(prob, model, timealg, scheme, x, tsave, p0;
        adaptive = time_adaptive, autodiff = autodiff_space, ϵ = ϵ_space,
    )

    evolve_integrator!(integrator; verbose)
end
#===========================================================#
