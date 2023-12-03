#
function nonlinleastsq(
    NN::Lux.AbstractExplicitLayer,
    p0::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    data::Tuple,
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

    # TODO - callback not being called. Read NonlinearSolve docs
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
