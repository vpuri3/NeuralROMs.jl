
using CUDA, OptimizationOptimJL, Zygote

loss(x, p) = sum(abs2, x - p)
x0 = zeros(Float32, 2) |> cu
p = Float32[1.0, 100.0] |> cu

optf = OptimizationFunction(loss, AutoZygote())
prob = Optimization.OptimizationProblem(optf, x0, p)
@show sol = solve(prob, Optim.LBFGS())

