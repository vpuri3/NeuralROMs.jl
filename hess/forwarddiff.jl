#
#======================================================#
using ForwardDiff, NeuralROMs
using ForwardDiff: Dual, value, partials

# # f = x -> exp.(2x)
# f = x -> x .^ 5
# x = [1.0, 10.0]
#
# fwd = forwarddiff_deriv4(f, x)
# fd = finitediff_deriv4(f, x)
# nothing


f = xy -> reshape(xy[1, :] .* xy[2, :], 1, :)

# xy = [2.0, 3.0]
# xy = reshape([2.0, 3.0], (2, 1))
xy = vcat(fill(2, (1, 2)), fill(3, (1, 2)))

x = reshape(xy[1, :], (1, :))
y = reshape(xy[2, :], (1, :))

f_dx = function(x_internal)
    xy_internal = vcat(x_internal, y)
    f(xy_internal)
end
f_dy = function(y_internal)
    xy_internal = vcat(x, y_internal)
    f(xy_internal)
end

u, udx = forwarddiff_deriv1(f_dx, x)
_, udy = forwarddiff_deriv1(f_dy, y)

u, udx, udy
