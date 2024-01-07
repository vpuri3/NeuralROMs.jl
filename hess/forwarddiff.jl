#
#======================================================#
using ForwardDiff
using ForwardDiff: Dual, value, partials

# f = x -> exp.(2x)
f = x -> x .^ 5
x = [1.0, 10.0]

# using GeometryLearning
# fwd = forwarddiff_deriv4(f, x)
# fd = finitediff_deriv4(f, x)
# nothing

## 4st order
T = Float64
tag = ForwardDiff.Tag(f, T)

z = x
z = Dual{typeof(tag)}.(z, one(T))
z = Dual{typeof(tag)}.(z, one(T))
z = Dual{typeof(tag)}.(z, one(T))
z = Dual{typeof(tag)}.(z, one(T))

fz  = f(z)
fx  = value.(value.(value.(value.(fz))))
d1f = partials.(value.(value.(value.(fz))), 1)
d2f = partials.(partials.(value.(value.(fz)), 1), 1)
d3f = partials.(partials.(partials.(value.(fz), 1), 1), 1)
d4f = partials.(partials.(partials.(partials.(fz, 1), 1), 1), 1)

fx, d1f, d2f, d3f, d4f
