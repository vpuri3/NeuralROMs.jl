#
shiftdata(u::AbstractArray, μ::Number, σ::Number) = (u .- μ) / sqrt(σ)
unshiftdata(u::AbstractArray, μ::Number, σ::Number) = u * sqrt(σ) .+ μ

# For 2D, make X a tuple (X, Y). should work fine with dUdX, etc
# otherwise need `makeUfromXY`, `makeUfromX_newmodel` type functions

function makeUfromX(X, NN, p, st, Icode, md)
    x = shiftdata(X, md.x̄, md.σx)
    u = NN((x, Icode), p, st)[1]
    unshiftdata(u, md.ū, md.σu)
end

#===========================================================#

function dUdX1(X, NN, p, st, Icode, md;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function _makeUfromX(X; NN = NN, p = p, st = st, Icode = Icode, md = md)
        makeUfromX(X, NN, p, st, Icode, md)
    end

    if isa(autodiff, AutoFiniteDiff)
        finitediff_deriv1(_makeUfromX, X; ϵ)
    elseif isa(autodiff, AutoForwardDiff)
        forwarddiff_deriv1(_makeUfromX, X)
    end
end

function dUdX2(X, NN, p, st, Icode, md;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function _makeUfromX(X; NN = NN, p = p, st = st, Icode = Icode, md = md)
        makeUfromX(X, NN, p, st, Icode, md)
    end

    if isa(autodiff, AutoFiniteDiff)
        finitediff_deriv2(_makeUfromX, X; ϵ)
    elseif isa(autodiff, AutoForwardDiff)
        forwarddiff_deriv2(_makeUfromX, X)
    end
end

function dUdp(X, NN, p, st, Icode, md;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function _makeUfromX(p; X = X, NN = NN, st = st, Icode = Icode, md = md)
        makeUfromX(X, NN, p, st, Icode, md)
    end

    if isa(autodiff, AutoFiniteDiff)
        finitediff_jacobian(_makeUfromX, p; ϵ)
    elseif isa(autodiff, AutoForwardDiff)
        forwarddiff_jacobian(_makeUfromX, p)
    end
end

#===========================================================#
function make_residual(
    dUdtRHS,
    timestepper_residual;
    implicit::Bool = false,
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function make_residual_internal(NN, p, st, batch, nlsp)
        XI, U0 = batch
        t0, t1, Δt, p0, md = nlsp
        X, Icode = XI

        _p = implicit ? p  : p0
        _t = implicit ? t1 : t0

        Rhs = dUdtRHS(X, NN, _p, st, Icode, md, _t; autodiff, ϵ)
        U1  = makeUfromX(X, NN, p, st, Icode, md)

        timestepper_residual(Δt, U0, U1, Rhs) |> vec
    end
end

function timestepper_residual_euler(Δt, U0, U1, Rhs)
    U1 - U0 - Δt * Rhs |> vec
end

function residual_learn(NN, p, st, batch, nlsp)
    XI, U0 = batch
    X, Icode = XI
    md = nlsp

    U1 = makeUfromX(X, NN, p, st, Icode, md)

    vec(U1 - U0)
end

#===========================================================#
# function do_timestep(::Val{:LSPG},
#     NN, p0, st, batch, nlssolver,
#     residual, nlsp, maxiters, abstol,
# )
#     @time p1, nlssol = nonlinleastsq(
#         NN, p0, st, batch, nlssolver;
#         residual, nlsp, maxiters, abstol,
#     )
#
#     nlsmse = sum(abs2, nlssol.resid) / length(nlssol.resid)
#     nlsinf = norm(nlssol.resid, Inf)
#     println("Nonlinear Steps: $(nlssol.stats.nsteps), \
#         MSE: $(round(nlsmse, sigdigits = 8)), \
#         ||∞: $(round(nlsinf, sigdigits = 8)), \
#         Ret: $(nlssol.retcode)"
#     )
#
#     p1
# end
#
# function do_timestep(::Val{:PODGalerkin},
#     NN, p0, st, batch, nlssolver,
#     residual, nlsp, maxiters, abstol,
# )
#     Xbatch, Icode = batch
#     # dU/dp, dU/dt
#     J = dUdp(Xbatch[1], NN, p, st, Icode, md; autodiff, ϵ) # (N, n)
#     r = dUdtRHS_advection(Xbatch[1], NN, p0, st, Icode, md, t0) # (N,)
#     dpdt = J \ vec(r)
#     p1 = p0 + Δt * dpdt # evolve with euler forward
#     p1
# end
#===========================================================#

#
