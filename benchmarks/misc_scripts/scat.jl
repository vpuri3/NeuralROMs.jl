#
using CUDA, NNlib
using Zygote, ForwardDiff

CUDA.allowscalar(false)

#==========================#
function hess_gather(
    x::AbstractMatrix,
    i::AbstractVector{<:Integer};
    ifgpu = false,
)
    if ifgpu
        x = x  |> cu
        i = i .|> Int32 |> cu
    end

    function loss(x)
        y = NNlib.gather(x, i)

        sum(abs2, y)
    end

    g(x) = Zygote.gradient(loss, x)[1]
    H(x) = ForwardDiff.jacobian(g, x)

    # p0 = Zygote.seed(p, Val(length(p)))
    # @show loss(p0) |> typeof
    # @show g(p0)    |> typeof
    #
    # @show g(p)
    # @show H(p)

    # Zygote.hessian(loss, x)
    H(x)
end
#==========================#

E, O, K = 3, 5, 10
x = rand(O, E)
i = rand(1:E, K)

hess_gather(x, i; ifgpu = false)
#
