#
#======================================================#
function nonlinleastsq(
    model::AbstractNeuralModel,
    p0::Union{NamedTuple, AbstractVector},
    data::NTuple{2, Any},
    nls::NonlinearSolve.AbstractNonlinearSolveAlgorithm;
    maxiters::Integer = 20,
    abstol = nothing,
    reltol = nothing,
    io::Union{Nothing, IO} = stdout,
    verbose::Bool = false,
    residual = nothing, # (NN, p, st, data, nlsp) -> resid
    nlsp = SciMLBase.NullParameters(),
    termination_condition = nothing,
    callback = nothing,
)
    p0 = ComponentArray(p0)

    residual = isnothing(residual) ? residual_learn : residual

    function nlsloss(nlsx, nlsp)
        r = residual(model, nlsx, data, nlsp)
        vec(r)
    end

    if verbose
        iter = Ref(0)
        callback = !isnothing(callback) ? callback : function(nlsx, l)
            iter[] += 1
            println(io, "NonlinearSolve [$(iter[]) / $maxiters] MSE: $l")
            return false
        end
    end

    nlsprob = NonlinearLeastSquaresProblem{false}(nlsloss, p0, nlsp)
    nlssol  = solve(nlsprob, nls;
        maxiters,
        callback,
        abstol,
        reltol,
        termination_condition,
    )

    if verbose
        mse = sum(abs2, nlssol.resid) / length(nlssol.resid)
        println("Steps: $(nlssol.stats.nsteps), \
            MSE: $(round(mse, sigdigits = 8)), \
            Ret: $(nlssol.retcode)"
        )
    end

    nlssol.u, nlssol
end
#======================================================#
function nonlinleastsq(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    data::NTuple{2, Any},
    nls::NonlinearSolve.AbstractNonlinearSolveAlgorithm;
    maxiters::Integer = 20,
    abstol = nothing,
    reltol = nothing,
    io::Union{Nothing, IO} = stdout,
    verbose::Bool = false,
    residual = nothing, # (NN, p, st, data, nlsp) -> resid
    nlsp = SciMLBase.NullParameters(),
    termination_condition = nothing,
    callback = nothing,
)
    st = Lux.testmode(st)
    p0 = ComponentArray(p0)

    residual = if isnothing(residual)
        # data regression
        (NN, p, st, data, nlsp) -> NN(data[1], p, st)[1] - data[2]
    else
        residual
    end

    function nlsloss(nlsx, nlsp)
        r = residual(NN, nlsx, st, data, nlsp)
        vec(r)
    end

    if verbose
        iter = Ref(0)
        callback = !isnothing(callback) ? callback : function(nlsx, l)
            iter[] += 1
            println(io, "NonlinearSolve [$(iter[]) / $maxiters] MSE: $l")
            return false
        end
    end

    nlsprob = NonlinearLeastSquaresProblem{false}(nlsloss, p0, nlsp)
    nlssol  = solve(nlsprob, nls;
        maxiters,
        callback,
        abstol,
        reltol,
        termination_condition,
    )

    if verbose
        mse = sum(abs2, nlssol.resid) / length(nlssol.resid)
        println("Steps: $(nlssol.stats.nsteps), \
            MSE: $(round(mse, sigdigits = 8)), \
            Ret: $(nlssol.retcode)"
        )
    end

    nlssol.u, nlssol
end
#======================================================#
#======================================================#

function nonlinleastsq(
    model::AbstractNeuralModel,
    p0::Union{NamedTuple, AbstractVector},
    data::NTuple{2, Any},
    opt::Union{Optim.AbstractOptimizer, Optimisers.AbstractRule};
    maxiters::Integer = 50,
    abstol = nothing,
    reltol = nothing,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    io::Union{Nothing, IO} = stdout,
    verbose::Bool = true,
    residual = nothing, # (NN, p, st, data) -> resid
    callback = nothing,
)
    p0 = ComponentArray(p0)

    residual = isnothing(residual) ? residual_learn : residual

    function optloss(optx, optp)
        r = residual(model, optx, data, optp)
        sum(abs2, r) / length(r)
    end

    iter = Ref(0)

    callback = !isnothing(callback) ? callback : function(optx, l)
        iter[] += 1
        if verbose
            println(io, "NonlinearSolve [$(iter[]) / $maxiters] MSE: $l")
        end
        return false
    end

    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, p0)
    optsol  = solve(optprob, opt;
        maxiters,
        callback,
        abstol,
        reltol,
    )

    if verbose
        obj = round(optsol.objective; sigdigits = 8)

        println(io, "#=======================#")
        @show optsol.retcode
        println(io, "Achieved objective value $(obj).")
        println(io, "#=======================#")
    end

    optsol.u, optsol
end
#======================================================#

function nonlinleastsq(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    data::NTuple{2, Any},
    opt::Union{Optim.AbstractOptimizer, Optimisers.AbstractRule};
    maxiters::Integer = 50,
    abstol = nothing,
    reltol = nothing,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    io::Union{Nothing, IO} = stdout,
    verbose::Bool = true,
    residual = nothing, # (NN, p, st, data) -> resid
    callback = nothing,
)
    st = Lux.testmode(st)
    p0 = ComponentArray(p0)

    residual = isnothing(residual) ? residual_learn : residual

    function optloss(optx, optp)
        r = residual(NN, optx, st, data)
        sum(abs2, r) / length(r)
    end

    if verbose
        iter = Ref(0)
        callback = !isnothing(callback) ? callback : function(nlsx, l)
            iter[] += 1
            println(io, "NonlinearSolve [$(iter[]) / $maxiters] MSE: $l")
            return false
        end
    end

    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, p0)
    optsol  = solve(optprob, opt;
        maxiters,
        callback,
        abstol,
        reltol,
    )

    if verbose
        obj = round(optsol.objective; sigdigits = 8)

        println(io, "#=======================#")
        @show optsol.retcode
        println(io, "Achieved objective value $(obj).")
        println(io, "#=======================#")
    end

    optsol.u, optsol
end
#======================================================#

# Linesearch / nonlinleastsq
# rosenbrock(x) = (1.f0 - x[1])^2 + 100.f0 * (x[2] - x[1]^2)^2
# f(x) = [rosenbrock(x), (x[1] - 1.f0), (x[2]-1.f0), x[1]*x[2] - 1]
# x0 = zeros(Float32, 2)
#
# for line in (
#     Static,
#     # BackTracking,
#     # HagerZhang,
#     # MoreThuente,
#     # StrongWolfe,
# )
#     for alpha in (
#         InitialStatic,
#         # InitialPrevious,
#         # InitialQuadratic,
#         # InitialHagerZhang,
#         # InitialConstantChange,
#     )
#         println(line, "\t", alpha)
#
#         linesearch = line === Static ? Static() : line{Float32}()
#         alphaguess = alpha{Float32}()
#
#         nlsq(f, x0; linesearch, alphaguess)
#         println()
#     end
# end

nothing
#
