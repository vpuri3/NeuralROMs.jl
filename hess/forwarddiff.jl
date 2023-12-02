#
#======================================================#
using ForwardDiff
using ForwardDiff: Dual, value, partials

# f = x -> exp.(x)
f = x -> x .^ 2
x = [1.0, 2.0, 3.0, 4.0]

# 2st order
z = Dual{:FD_D2Tag}.(Dual{:FD_D2TagInt}.(x, true), true)
fz = f(z)
fx = value.(value.(fz))
df = value.(partials.(fz, 1))
d2f = partials.(partials.(fz, 1), 1)
#======================================================#
fz
