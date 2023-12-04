#
function make_residual(
    dudtRHS,
    timestepper_residual;
    implicit::Bool = false,
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
    md = nothing,
)
    function make_residual_internal(
        model::AbstractNeuralModel,
        p::AbstractVector,
        batch::NTuple{2, Any},
        nlsp,
    )
        x, _ = batch
        t1, Δt, t0, p0, u0 = nlsp

        _p = implicit ? p  : p0
        _t = implicit ? t1 : t0

        rhs = dudtRHS(model, x, _p, _t, md; autodiff, ϵ)
        u1  = model(x, p)
        timestepper_residual(Δt, u0, u1, rhs)
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

function timestepper_residual_euler(Δt, u0, u1, rhs)
    u1 - u0 - Δt * rhs |> vec
end

#===========================================================#
abstract type AbstractSolveScheme end

mutable struct PODGalerkin{T, S} <: AbstractSolveScheme
    solver::S
    abstol::T
    reltol::T
end

implicit_timestepper(::PODGalerkin) = false

mutable struct LeastSqPetrovGalerkin{T, S} <: AbstractSolveScheme # init cache
    solver::S
    implicit_timestepper::Bool
end

implicit_timestepper(scheme::LeastSqPetrovGalerkin) = scheme.implicit_timestepper

mutable struct TimeIntegrator{T, S}
    scheme::S

    Δt::T
    times
    ps
    Us
end


abstract type AbstractTimeSteper end

struct EulerForwad <: AbstractTimeSteper end
struct EulerBackward <: AbstractTimeSteper end

isimplicit_timestepper(::EulerForwad) = false
isimplicit_timestepper(::EulerBackward) = true

#===========================================================#

function do_timestep(
    solver::LeastSqPetrovGalerkin,
    adaptive::Bool,
    NN, p0, st, batch, nlssolver,
    residual, nlsp, maxiters, abstol;
    verbose = true,
)
    @time p1, nlssol = nonlinleastsq(
        NN, p0, st, batch, nlssolver;
        residual, nlsp, maxiters, abstol,
    )

    if verbose
        nlsmse = sum(abs2, nlssol.resid) / length(nlssol.resid)
        nlsinf = norm(nlssol.resid, Inf)

        println("Nonlinear Steps: $(nlssol.stats.nsteps), \
            MSE: $(round(nlsmse, sigdigits = 8)), \
            ||∞: $(round(nlsinf, sigdigits = 8)), \
            Ret: $(nlssol.retcode)"
        )
    end

    # adaptive

    p1
end

# function do_timestep(::Val{:PODGalerkin},
#     NN, p0, st, batch, dUdtRHS, md, t0, Δt;
#     verbose = true, adaptive = true,
# )
#     Xbatch, Icode = batch[1]
#
#     # dU/dp, dU/dt
#     J = dUdp(Xbatch[1], NN, p0, st, Icode, md)#; autodiff, ϵ) # (N, n)
#     r = dUdtRHS(Xbatch[1], NN, p0, st, Icode, md, t0) # (N,)
#
#     dpdt = J \ vec(r) # use `LinearSolve.jl`
#     p1 = p0 + Δt * dpdt # evolve with euler forward
#
#     if verbose # compute residual, print norms
#     end
#
#     # adaptive
#
#     p1
# end
#===========================================================#

#
