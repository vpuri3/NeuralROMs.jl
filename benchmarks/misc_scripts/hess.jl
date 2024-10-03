#
using Random
using Lux, CUDA, LuxCUDA, ComponentArrays
using Zygote, ForwardDiff

CUDA.allowscalar(false)

#==========================#
function testhessian(
    NN::AbstractLuxLayer,
    data::Tuple;
    device = cpu_device(),
)
    p, st = Lux.setup(Random.default_rng(), NN)

    st = Lux.testmode(st)
    p = ComponentArray(p)

    xdata, ydata = data |> device
    p, st = (p, st)     |> device

    function loss(optx)
        ypred, _ = NN(xdata, optx, st)

        sum(abs2, ydata - ypred)
    end

    g(p) = Zygote.gradient(loss, p)[1]
    H(p) = ForwardDiff.jacobian(g, p)

    # p0 = Zygote.seed(p, Val(length(p)))
    # @show loss(p0) |> typeof
    # @show g(p0)    |> typeof
    #
    # @show g(p)
    # @show H(p)

    Zygote.hessian(loss, p)
end
#==========================#

# NN = Chain(Dense(1, 3), Dense(3, 1))
# data = ntuple(_ -> rand(1, 10), 2)

E, K = 1, 10
NN = Chain(Embedding(E => 3), Dense(3, 1))
data = (ones(Int32, K), rand(1, K))

device = Lux.cpu_device()
testhessian(NN, data; device)
#

