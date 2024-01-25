
################
# trainnig schedule
################
#
# nepochs = (50, E,)
# schedules = (
#     Step(1f-2, 1f0, Inf32),                        # constant warmup
#     # Triangle(λ0 = 1f-5, λ1 = 1f-3, period = 20), # gradual warmup
#     # Exp(1f-3, 0.996f0),
#     SinExp(λ0 = 1f-5, λ1 = 1f-3, period = 50, γ = 0.995f0),
#     # CosAnneal(λ0 = 1f-5, λ1 = 1f-3, period = 50),
# )
# opts = fill(Optimisers.Adam(), length(nepochs)) |> Tuple
#
################
# second order optimizer
################
#
# # if it can fit on 11 gb vRAM on 2080Ti
# if Lux.parameterlength(decoder) < 35_000
#     opts = (opts..., LBFGS(),)
#     nepochs = (nepochs..., round(Int, E / 10))
#     schedules = (schedules..., Step(1f0, 1f0, Inf32))
# end
#
################
# fourth order derivative
################
# using ForwardDiff
# using ForwardDiff: Dual, value, partials
#
# # f = x -> exp.(2x)
# f = x -> x .^ 5
# x = [1.0, 10.0]
#
# # using GeometryLearning
# # fwd = forwarddiff_deriv4(f, x)
# # fd = finitediff_deriv4(f, x)
#
# ## 4st order
# T = Float64
# tag = ForwardDiff.Tag(f, T)
#
# z = x
# z = Dual{typeof(tag)}.(z, one(T))
# z = Dual{typeof(tag)}.(z, one(T))
# z = Dual{typeof(tag)}.(z, one(T))
# z = Dual{typeof(tag)}.(z, one(T))
#
# fz  = f(z)
# fx  = value.(value.(value.(value.(fz))))
# d1f = partials.(value.(value.(value.(fz))), 1)
# d2f = partials.(partials.(value.(value.(fz)), 1), 1)
# d3f = partials.(partials.(partials.(value.(fz), 1), 1), 1)
# d4f = partials.(partials.(partials.(partials.(fz, 1), 1), 1), 1)
#
# fx, d1f, d2f, d3f, d4f
################
