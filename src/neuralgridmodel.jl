#
#===========================================================#
abstract type AbstractNeuralGridModel <: AbstractNeuralModel end
#===========================================================#

function dudx1_1D(
    model::AbstractNeuralGridModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ = nothing,
)
    @assert size(x, 1) == 1
    @assert ndims(x) == 2 "TODO: implement reshapes to handle ND arrays"

    u  = model(x, p)
    du = apply_matrix(u, model.grid, model.Dx[1])

    u, du
end

function dudx2_1D(
    model::AbstractNeuralGridModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ = nothing,
)
    @assert size(x, 1) == 1
    @assert ndims(x) == 2 "TODO: implement reshapes to handle ND arrays"

    u = model(x, p)
    d1u = apply_matrix(u, model.grid, model.Dx[1])
    d2u = apply_matrix(u, model.grid, model.Dx[2])

    u, d1u, d2u
end

function dudx4_1D(
    model::AbstractNeuralGridModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ = nothing,
)
    @assert size(x, 1) == 1
    @assert ndims(x) == 2 "TODO: implement reshapes to handle ND arrays"

    u = model(x, p)
    d1u = apply_matrix(u, model.grid, model.Dx[1])
    d2u = apply_matrix(u, model.grid, model.Dx[2])
    d3u = apply_matrix(u, model.grid, model.Dx[3])
    d4u = apply_matrix(u, model.grid, model.Dx[4])

    u, d1u, d2u, d3u, d4u
end

#===========================================================#

function dudx1_2D(
    model::AbstractNeuralGridModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ = nothing,
)
    @assert size(x, 1) == 2
    @assert ndims(x) == 2 "TODO: implement reshapes to handle ND arrays"

    u  = model(x, p)
    udx = apply_matrix(u, model.grid, model.Dx[1], I)
    udy = apply_matrix(u, model.grid, I, model.Dy[1])

    u, (udx, udy,)
end

function dudx2_2D(
    model::AbstractNeuralGridModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ = nothing,
)
    @assert size(x, 1) == 2
    @assert ndims(x) == 2 "TODO: implement reshapes to handle ND arrays"

    u = model(x, p)

    udx = apply_matrix(u, model.grid, model.Dx[1], I)
    udy = apply_matrix(u, model.grid, I, model.Dy[1])

    udxx = apply_matrix(u, model.grid, model.Dx[2], I)
    udyy = apply_matrix(u, model.grid, I, model.Dy[2])

    u, (udx, udy,), (udxx, udyy,)
end

#===========================================================#

# TODO: differentiate multiple fields in the same function call

function make_dx_mats(
    xy::AbstractArray,
    grid::NTuple{D, Integer},
) where{D}

    in_dim, N = size(xy)

    if D == 1
        x_vec = vec(xy)
        dxmats(x_vec), (I, I, I, I)
    elseif D == 2
        xy_re = reshape(xy, in_dim, grid...)
        x_vec = vec(xy_re[1, :, 1])
        y_vec = vec(xy_re[2, 1, :])
        dxmats(x_vec), dxmats(y_vec)
    else
        throw(ErrorException("Only support 1D/2D periodic grids"))
    end
end

function apply_matrix(u, grid, Dx)
    out_dim, N = size(u)
    @assert prod(grid) == N "grid size $grid doesn't match with field size $out_dim x $N."

    dus = ()
    for od in 1:out_dim
        u_re = reshape(u[od, :], grid...)
        du = Dx * u_re
        du = reshape(du, 1, N)
        dus = (dus..., du)
    end
    vcat(dus...)
end

function apply_matrix(u, grid, Dx, Dy)
    out_dim, N = size(u)
    @assert prod(grid) == N "grid size $grid doesn't match with field size $out_dim x $N."

    dus = ()
    for od in 1:out_dim
        u_re = reshape(u[od, :], grid...)
        du = Dx * u_re * Dy'
        du = reshape(du, 1, N)
        dus = (dus..., du)
    end
    vcat(dus...)
end

#===========================================================#
export PCAModel
@concrete mutable struct PCAModel{Tu} <: AbstractNeuralGridModel
    P
    Dx
    Dy
    grid
    out_dim

    ū::Tu
    σu::Tu
end

function PCAModel(
    P::AbstractMatrix,
    x::AbstractArray,
    grid::NTuple{D, Integer},
    out_dim::Integer,
    metadata::NamedTuple,
) where{D}

    ū = metadata.ū
    σu = metadata.σu

    if out_dim > 1
        @warn """
        out_dim > 1 not fully implemented in PCAModel.
        All output fields will be identical.
        """
    end

    Dx, Dy = make_dx_mats(x, grid)

    PCAModel(P, Dx, Dy, grid, out_dim, ū, σu,)
end

function Adapt.adapt_structure(to, model::PCAModel)
    P  = Adapt.adapt_structure(to, model.P)
    Dx = Adapt.adapt_structure(to, model.Dx)
    Dy = Adapt.adapt_structure(to, model.Dy)
    ū  = Adapt.adapt_structure(to, model.ū)
    σu = Adapt.adapt_structure(to, model.σu)

    PCAModel(P, Dx, Dy, model.grid, model.out_dim, ū, σu,)
end

function (model::PCAModel)(
    x::AbstractArray,
    p::AbstractVector,
)
    u_norm = model.P * p

    # hack
    if model.out_dim > 1
        @warn """
        PCAModel doesn't fully support out_dim > 1.
        All output fields will be identical.
        """ maxlog = 1
        u_norm = reshape(u_norm, 1, :)
        u_norm = repeat(u_norm, model.out_dim, 1)
    end

    u = unnormalizedata(u_norm, model.ū, model.σu)
    reshape(u, model.out_dim, :)
end

#===========================================================#

export INRModel
@concrete mutable struct INRModel{Tx, Tu} <: AbstractNeuralGridModel
    NN
    st

    x̄::Tx
    σx::Tx

    ū::Tu
    σu::Tu

    Dx
    Dy
    grid
end

function INRModel(
    NN::AbstractLuxLayer,
    st::NamedTuple,
    x::AbstractArray{T},
    grid::NTuple{D, Integer}, # (Nx, Ny)
    metadata::NamedTuple,
) where{T<:Number, D}

    Dx, Dy = make_dx_mats(x, grid)

    x̄ = metadata.x̄
    ū = metadata.ū

    σx = metadata.σx
    σu = metadata.σu

    INRModel(NN, st, x̄, σx, ū, σu, Dx, Dy, grid)
end

function Adapt.adapt_structure(to, model::INRModel)
    st = Adapt.adapt_structure(to, model.st)
    x̄  = Adapt.adapt_structure(to, model.x̄ )
    ū  = Adapt.adapt_structure(to, model.ū )
    σx = Adapt.adapt_structure(to, model.σx)
    σu = Adapt.adapt_structure(to, model.σu)
    Dx = Adapt.adapt_structure(to, model.Dx)
    Dy = Adapt.adapt_structure(to, model.Dy)

    INRModel(
        model.NN, st, x̄, σx, ū, σu, Dx, Dy, model.grid,
    )
end

function (model::INRModel)(
    x::AbstractArray,
    p::AbstractVector,
)
    x_norm = normalizedata(x, model.x̄, model.σx)
    u_norm = model.NN(x_norm, p, model.st)[1]

    unnormalizedata(u_norm, model.ū, model.σu)
end

#===========================================================#
export CAEModel
@concrete mutable struct CAEModel{Tu} <: AbstractNeuralGridModel
    NN
    p
    st

    ū::Tu
    σu::Tu

    Dx
    Dy
    grid
end

function CAEModel(
    NN::AbstractLuxLayer,
    p::Union{NamedTuple, ComponentVector},
    st::NamedTuple,
    x::AbstractArray{T},
    grid::NTuple{D, Integer},
    metadata::NamedTuple,
) where{T<:Number, D}

    Dx, Dy = make_dx_mats(x, grid)

    ū  = metadata.ū
    σu = metadata.σu

    CAEModel(NN, p, st, ū, σu, Dx, Dy, grid)
end

function Adapt.adapt_structure(to, model::CAEModel)
    p  = Adapt.adapt_structure(to, model.p )
    st = Adapt.adapt_structure(to, model.st)
    ū  = Adapt.adapt_structure(to, model.ū )
    σu = Adapt.adapt_structure(to, model.σu)
    Dx = Adapt.adapt_structure(to, model.Dx)
    Dy = Adapt.adapt_structure(to, model.Dy)

    CAEModel(model.NN, p, st, ū, σu, Dx, Dy, model.grid)
end

function (model::CAEModel)(
    x::AbstractArray,
    p::AbstractVector,
)
    p_resh = reshape(getdata(p), :, 1)
    u_norm = model.NN(p_resh, model.p, model.st)[1] # [grid..., out_dim]

    N = prod(model.grid)
    in_dim  = length(model.grid)
    out_dim = size(u_norm, in_dim + 1)

    u_norm = reshape(u_norm, N, out_dim) # [N, out_dim]
    u_norm = permutedims(u_norm, (2, 1)) # [out_dim, N]

    unnormalizedata(u_norm, model.ū, model.σu)
end
#===========================================================#
