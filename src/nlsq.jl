#
function nlsq(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    data::Tuple,
    nls::NonlinearSolve.AbstractNonlinearSolveAlgorithm;
    maxiters::Integer = 50,
    abstol::Real = 1f-5,
    io::Union{Nothing, IO} = stdout,
    verbose::Bool = true,
    residual = nothing, # (NN, p, st, data, nlsp) -> resid
    nlsp = SciMLBase.NullParameters()
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

    # TODO - callback not being called. Read NonlinearSolve docs
    iter = Ref(0)
    function callback(nlsx, l) # not being used
        if verbose
            iter[] += 1
            println(io, "[$(iter[]) / $maxiters] MSE: $l")
        end
        return false
    end

    nlsprob = NonlinearLeastSquaresProblem{false}(nlsloss, p0, nlsp)
    nlssol  = solve(nlsprob, nls; maxiters, callback, abstol)

    mse = sum(abs2, nlssol.resid) / length(nlssol.resid)
    steps = nlssol.stats.nsteps
    retcode = nlssol.retcode

    if verbose
        println(io, "Steps: $(steps), MSE: $(round(mse, sigdigits = 8)), Ret: $(retcode)")
    end

    nlssol.u, mse, steps, retcode
end

#======================================================#
function nlsq(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    data::Tuple,
    opt::Union{Optim.AbstractOptimizer, Optimisers.AbstractRule};
    maxiters::Integer = 50,
    adtype::ADTypes.AbstractADType = AutoZygote(),
    io::Union{Nothing, IO} = stdout,
    verbose::Bool = true,
    residual = nothing, # (NN, p, st, data) -> resid
)
    st = Lux.testmode(st)
    p0 = ComponentArray(p0)

    residual = if isnothing(residual)
        (NN, p, st, data) -> NN(data[1], p, st)[1] - data[2]
    else
        residual
    end

    function optloss(optx, optp)
        r = residual(NN, optx, st, data)
        sum(abs2, r) / length(r)
    end

    iter = Ref(0)
    function callback(optx, l)
        if verbose
            println(io, "[$(iter[]) / $maxiters] MSE: $l")
            iter[] += 1
        end
        return false
    end

    optfun  = OptimizationFunction(optloss, adtype)
    optprob = OptimizationProblem(optfun, p0)

    optsol = solve(optprob, opt; maxiters, callback) #, abstol, reltol)


    if verbose
        obj = round(optsol.objective; sigdigits = 8)
        tym = round(optsol.solve_time; sigdigits = 8)

        println(io, "#=======================#")
        @show optsol.retcode
        println(io, "Achieved objective value $(obj) in time $(tym)s.")
        println(io, "#=======================#")
    end

    optsol.u, optsol.objective
end

# Linesearch / nlsq
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
