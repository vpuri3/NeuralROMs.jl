#
using HDF5, Random
using CalculustCore: ndgrid

function darcy2D(filename, K = 1024, rng = Random.default_rng())
    file = h5open(filename, "r")

    ν = read(file["nu"])           # [128, 128, 10000]
    x = read(file["x-coordinate"]) # [128]
    y = read(file["y-coordinate"]) # [128]
    t = read(file["t-coordinate"]) # [10]
    u = read(file["tensor"])       # [128, 128, 1, 10000]

    N = length(x)
    Kmax = size(u,)[end]
    Ks = rand(rng, 1:Kmax, 2K)
    _I = Ks[begin:K]
    I_ = Ks[K+1:end]

    x, y = ndgrid(x, y)

    _x = zeros(3, N, N, K)
    x_ = zeros(3, N, N, K)

    _u = zeros(1, N, N, K)
    u_ = zeros(1, N, N, K)

    _x[1, :, :, :] .= x
    _x[2, :, :, :] .= y
    _x[3, :, :, :] .= ν[:, :, _I]

    x_[1, :, :, :] .= x
    x_[2, :, :, :] .= y
    x_[3, :, :, :] .= ν[:, :, I_]

    _u[1, :, :, :] = u[:, :, 1, _I]
    u_[1, :, :, :] = u[:, :, 1, I_]

    (_x, _u), (x_, u_)
end
