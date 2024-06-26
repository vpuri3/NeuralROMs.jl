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
    adaptive::Bool

    tspan
    tsave

    tstep
    isave

    # state
    tprevs
    pprevs # ũprevs
    uprevs
    fprevs
    f̃prevs # Galerkin only (rhs in reduced space)

    autodiff_nls
    autodiff_jac
    autodiff_xyz

    ϵ_nls
    ϵ_jac
    ϵ_xyz

    lincache # Galerkin only (linear solver cache)
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

    autodiff_nls = AutoForwardDiff(),
    autodiff_jac = AutoForwardDiff(),
    autodiff_xyz = AutoForwardDiff(),

    ϵ_nls = nothing,
    ϵ_jac = nothing,
    ϵ_xyz = nothing,

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
    f0 = dudtRHS(prob, model, x, p0, t0; autodiff = autodiff_xyz, ϵ = ϵ_xyz)

    lincache = if scheme isa GalerkinProjection
        J = dudp(model, x, p0; autodiff = autodiff_jac, ϵ = ϵ_jac)
        linprob = LinearProblem(J, vec(similar(f0)))
        LinearSolve.init(linprob, scheme.linsolve)
    else
        nothing
    end

    f̃0 = if scheme isa GalerkinProjection
        compute_f̃(f0, p0, x, model, lincache; autodiff_jac, ϵ_jac)
    else
        p0 * NaN
    end

    # previous states
    nstates = nsavedstates(timealg)
    tprevs = (t0, (T(NaN) for _ in 1:nstates)...)
    pprevs = (p0, (fill!(similar(p0), NaN) for _ in 1:nstates - 1)...)
    uprevs = (u0, (fill!(similar(u0), NaN) for _ in 1:nstates - 1)...)
    fprevs = (f0, (fill!(similar(f0), NaN) for _ in 1:nstates - 1)...)
    f̃prevs = (f̃0, (fill!(similar(f̃0), NaN) for _ in 1:nstates - 1)...)

    TimeIntegrator(
        prob, model, timealg, scheme,
        x, Δt, Δt, adaptive,
        tspan, tsave, tstep, isave,
        tprevs, pprevs, uprevs, fprevs, f̃prevs,
        autodiff_nls, autodiff_jac, autodiff_xyz,
        ϵ_nls, ϵ_jac, ϵ_xyz,
        lincache,
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
    fields = (int.tprevs, int.pprevs, int.uprevs, int.fprevs, int.f̃prevs)
    getindex.(fields, 1)
end

function get_saved_states(int::TimeIntegrator)
    fields = (int.tprevs, int.pprevs, int.uprevs, int.fprevs, int.f̃prevs)
    Tuple(f[2:end] for f in fields)
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
    tol::T = T(1e-6),
    verbose::Bool = true,
) where{T}

    # save states at `integrator.tsave`
    ts = ()
    ps = ()
    us = ()

    # save initial condition
    begin
        t, p, u, _ = get_state(integrator)
        ts = (ts..., t)
        ps = (ps..., p)
        us = (us..., u)
    end

    # Time loop
    _, Tfinal = get_tspan(integrator)

    while get_time(integrator) <= Tfinal

        if abs(get_time(integrator) - Tfinal) < tol
            break
        end

        update_Δt_for_saving!(integrator; tol, verbose)
        perform_timestep!(integrator; verbose)
        state = savestep!(integrator; verbose)

        if !isnothing(state)
            t, p, u, _ = state
            ts = (ts..., t)
            ps = (ps..., p)
            us = (us..., u)
        end
    end

    ps = ps |> Lux.cpu_device()
    us = us |> Lux.cpu_device()

    ts = [ts...]
    ps = mapreduce(getdata, hcat, ps)
    us = cat(us...; dims = 3)

    @assert norm(integrator.tsave - ts, Inf) < 1e-6

    return ts, ps, us
end

#===========================================================#

function descend_p0(
    model::AbstractNeuralModel,
    p0::AbstractVector,
    batch::NTuple{2, Any},
    lr::Real,
    round::Integer = 1;
    maxiters::Integer = 100,
    verbose::Bool = true,
)
    x, u0 = batch
    opt = Optimisers.Descent(lr)
    p0, _ = nonlinleastsq(model, p0, batch, opt; maxiters, verbose = false)
    mse_norm = mse(model(x, p0), u0) / mse(u0, 0 * u0)

    if verbose
        println("Descent round $(round): MSE (normalized): $(mse_norm)")
    end

    p0, mse_norm
end

function learn_p0(
    model::AbstractNeuralModel,
    p0::AbstractVector,
    batch::NTuple{2, Any};
    descend::Bool = true,
    nlssolve = nothing,
    verbose::Bool = true,
)
    x, u0 = batch

    if verbose
        println("#============================#")
        println("Time Step: $(0), Time: 0.0 - learn IC")
    end

    if descend
        p0, mse_norm = descend_p0(model, p0, batch, 1f-3, 1; verbose)
        p0, mse_norm = descend_p0(model, p0, batch, 1f-4, 2; verbose)
        p0, mse_norm = descend_p0(model, p0, batch, 1f-5, 3; verbose)
    end

    if isnothing(nlssolve)
        linesearch = LineSearch()
        autodiff = AutoForwardDiff()
        linsolve = QRFactorization()
        nlssolve = GaussNewton(; autodiff, linsolve, linesearch)
    end

    p0, _ = nonlinleastsq(model, p0, batch, nlssolve;
        residual = residual_learn,
        maxiters = 100,
        verbose,
    )

    mse_norm = mse(model(x, p0), u0) / mse(u0, 0 * u0)
    verbose && println("Gauss-Newton: MSE (normalized): $(mse_norm)")

    p0, mse_norm
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

    adaptive::Bool = true,
    learn_ic::Bool = false,
    descend_ic::Bool = false,
    nlssolve_ic = nothing,

    autodiff_nls = AutoForwardDiff(),
    autodiff_jac = AutoForwardDiff(),
    autodiff_xyz = AutoForwardDiff(),

    ϵ_nls = nothing,
    ϵ_jac = nothing,
    ϵ_xyz = nothing,

    IC_TOL::Real = 1f-4,

    verbose::Bool = true,
    device = Lux.cpu_device(),
)
    # data
    x, u0, tsave = data

    # move to device
    p0 = p0 |> device
    model = model |> device
    (x, u0) = (x, u0) |> device

    T = eltype(p0)

    #============================#
    # learn IC
    #============================#

    mse_norm = mse(model(x, p0), u0) / mse(u0, 0 * u0)

    if verbose
        println("#============================#")
        println("Initial Condition MSE: $mse_norm")
        println("Time Step: $(0), Time: 0.0")
    end

    if learn_ic
        p0, mse_norm = learn_p0(model, p0, (x, u0);
            descend = descend_ic, nlssolve = nlssolve_ic, verbose)
    end

    if (mse_norm > IC_TOL)
        @warn "MSE ($mse_norm) > $(IC_TOL)."
    end

    #============================#
    # time-marching
    #============================#

    Δt = isnothing(Δt) ? -(reverse(extrema(tsave))...) / 200 |> T : T(Δt)

    integrator = TimeIntegrator(
        prob, model, timealg, scheme, x, tsave, p0; Δt, adaptive,
        autodiff_nls, autodiff_jac, autodiff_xyz,
        ϵ_nls, ϵ_jac, ϵ_xyz,
    )

    evolve_integrator!(integrator; verbose)
end
#===========================================================#
