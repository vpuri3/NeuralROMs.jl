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
    @assert out_dim == 1 "We apply Dx mats to individual fields."
    @assert prod(grid) == N "grid size $grid doesn't match with field size $out_dim x $N."

    u_re = reshape(u, grid...)

    du = Dx * u_re
    
    reshape(du, out_dim, N)
end

function apply_matrix(u, grid, Dx, Dy)
    out_dim, N = size(u)
    @assert out_dim == 1 "We apply Dx mats to individual fields."
    @assert prod(grid) == N "grid size $grid doesn't match with field size $out_dim x $N."

    u_re = reshape(u, grid...)

    du = Dx * u_re * Dy'
    
    reshape(du, out_dim, N)
end

#===========================================================#
export PCAModel
@concrete mutable struct PCAModel{Tu} <: AbstractNeuralGridModel
    P
    Dx
    Dy
    grid

    ū::Tu
    σu::Tu
end

function PCAModel(
    P::AbstractMatrix,
    x::AbstractArray,
    grid::NTuple{D, Integer},
    metadata::NamedTuple,
) where{D}

    ū = metadata.ū
    σu = metadata.σu

    Dx, Dy = make_dx_mats(x, grid)

    PCAModel(P, Dx, Dy, grid, ū, σu,)
end

function Adapt.adapt_structure(to, model::PCAModel)
    P  = Adapt.adapt_structure(to, model.P)
    Dx = Adapt.adapt_structure(to, model.Dx)
    Dy = Adapt.adapt_structure(to, model.Dy)
    ū  = Adapt.adapt_structure(to, model.ū)
    σu = Adapt.adapt_structure(to, model.σu)

    PCAModel(P, Dx, Dy, model.grid, ū, σu,)
end

function (model::PCAModel)(
    x::AbstractArray,
    p::AbstractVector,
)
    u_norm = model.P * p
    u = unnormalizedata(u_norm, model.ū, model.σu)
    reshape(u, 1, :)
end

#===========================================================#

export INRModel
@concrete mutable struct INRModel{Tx, Tu} <: AbstractNeuralGridModel
    NN
    st
    Icode

    x̄::Tx
    σx::Tx

    ū::Tu
    σu::Tu

    Dx
    Dy
    grid
end

function INRModel(
    NN::Lux.AbstractExplicitLayer,
    st::NamedTuple,
    x::AbstractArray{T},
    grid::NTuple{D, Integer}, # (Nx, Ny)
    metadata::NamedTuple,
) where{T<:Number, D}

    Icode = begin
        sz = (1, size(x)[2:end]...,)
        Icode = similar(x, Int32, sz)
        fill!(Icode, true)
    end

    Dx, Dy = make_dx_mats(x, grid)

    x̄ = metadata.x̄
    ū = metadata.ū

    σx = metadata.σx
    σu = metadata.σu

    INRModel(NN, st, Icode, x̄, σx, ū, σu, Dx, Dy, grid)
end

function Adapt.adapt_structure(to, model::INRModel)
    st = Adapt.adapt_structure(to, model.st)
    Icode = Adapt.adapt_structure(to, model.Icode)
    x̄  = Adapt.adapt_structure(to, model.x̄ )
    ū  = Adapt.adapt_structure(to, model.ū )
    σx = Adapt.adapt_structure(to, model.σx)
    σu = Adapt.adapt_structure(to, model.σu)
    Dx = Adapt.adapt_structure(to, model.Dx)
    Dy = Adapt.adapt_structure(to, model.Dy)

    INRModel(
        model.NN, st, Icode, x̄, σx, ū, σu, Dx, Dy, model.grid,
    )
end

function (model::INRModel)(
    x::AbstractArray,
    p::AbstractVector,
    Icode::Union{AbstractArray, Nothing} = nothing,
)

    Icode = if !isnothing(Icode)
        Icode
    else
        if !isnothing(model.Icode)
            model.Icode
        else
            Zygote.@ignore begin
                sz = (1, size(x)[2:end]...,)
                Icode = similar(x, Int32, sz)
                fill!(Icode, true)
            end
        end
    end

    x_norm = normalizedata(x, model.x̄, model.σx)
    batch  = (x_norm, Icode)
    u_norm = model.NN(batch, p, model.st)[1]

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
    NN::Lux.AbstractExplicitLayer,
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
    u_norm = model.NN(p_resh, model.p, model.st)[1]
    u = unnormalizedata(u_norm, model.ū, model.σu)
    reshape(u, 1, :)
end
#===========================================================#
