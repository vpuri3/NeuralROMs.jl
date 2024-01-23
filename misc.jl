
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
