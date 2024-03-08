#
#===========================================================#
using SparseArrays: sparse

abstract type AbstractNeuralGridModel <: AbstractNeuralModel end

function dudx1_1D(
    model::AbstractNeuralGridModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoFiniteDiff(),
    ϵ = nothing,
)
    @assert size(x, 1) == 1
    @assert ndims(x) == 2 "TODO: implement reshapes to handle ND arrays"

    u = model(x, p)
    du = model.D1x * vec(u)
    du = reshape(du, 1, :)

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
    d1u = model.D1x * vec(u)
    d2u = model.D2x * vec(u)

    d1u = reshape(d1u, 1, :)
    d2u = reshape(d2u, 1, :)

    u, d1u, d2u
end

#===========================================================#

function d1x1dmat(x::AbstractArray)
    @assert size(x, 1) == 1
    @assert ndims(x) == 2

    N = length(x)
    T = eltype(x)

    x = vec(x)
    h = x[2] - x[1]

    u = zeros(T, N-1) .+ 1/(2h)
    d = zeros(T, N  )
    l = zeros(T, N-1) .- 1/(2h)

    Dx = Tridiagonal(l, d, u) |> Array
    Dx[1, end] = -1/(2h)
    Dx[end, 1] =  1/(2h)

    sparse(Dx)
end

function d2x1dmat(x::AbstractArray)
    @assert size(x, 1) == 1
    @assert ndims(x) == 2

    N = length(x)
    T = eltype(x)

    x = vec(x)
    h = x[2] - x[1]

    d = -2 * ones(T, N  ) ./ (h^2) # diagonal
    b =  1 * ones(T, N-1) ./ (h^2) # off diagonal

    Dx = Tridiagonal(b, d, b) |> Array
    Dx[1, end] = 1 / (h^2)
    Dx[end, 1] = 1 / (h^2)

    sparse(Dx)
end

#===========================================================#
export PCAModel
@concrete mutable struct PCAModel{Tu} <: AbstractNeuralGridModel
    P
    D1x
    D2x

    ū::Tu
    σu::Tu
end

function PCAModel(
    P::AbstractMatrix,
    x::AbstractArray,
    metadata::NamedTuple,
)

    ū = metadata.ū
    σu = metadata.σu

    D1x = d1x1dmat(x)
    D2x = d2x1dmat(x)

    PCAModel(P, D1x, D2x, ū, σu,)
end

function Adapt.adapt_structure(to, model::PCAModel)
    P   = Adapt.adapt_structure(to, model.P)
    D1x = Adapt.adapt_structure(to, model.D1x)
    D2x = Adapt.adapt_structure(to, model.D2x)
    ū   = Adapt.adapt_structure(to, model.ū)
    σu  = Adapt.adapt_structure(to, model.σu)

    PCAModel(P, D1x, D2x, ū, σu,)
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

    D1x
    D2x
end

function INRModel(
    NN::Lux.AbstractExplicitLayer,
    st::NamedTuple,
    x::AbstractArray{T},
    metadata::NamedTuple,
) where{T<:Number}

    Icode = begin
        sz = (1, size(x)[2:end]...,)
        Icode = similar(x, Int32, sz)
        fill!(Icode, true)
    end

    D1x = d1x1dmat(x)
    D2x = d2x1dmat(x)

    x̄ = metadata.x̄
    ū = metadata.ū

    σx = metadata.σx
    σu = metadata.σu

    INRModel(NN, st, Icode, x̄, σx, ū, σu, D1x, D2x,)
end

function Adapt.adapt_structure(to, model::INRModel)
    st = Adapt.adapt_structure(to, model.st)
    Icode = Adapt.adapt_structure(to, model.Icode)
    x̄  = Adapt.adapt_structure(to, model.x̄ )
    ū  = Adapt.adapt_structure(to, model.ū )
    σx = Adapt.adapt_structure(to, model.σx)
    σu = Adapt.adapt_structure(to, model.σu)
    D1x = Adapt.adapt_structure(to, model.D1x)
    D2x = Adapt.adapt_structure(to, model.D2x)

    INRModel(
        model.NN, st, Icode, x̄, σx, ū, σu, D1x, D2x,
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

    D1x
    D2x
end

function CAEModel(
    NN::Lux.AbstractExplicitLayer,
    p::Union{NamedTuple, ComponentVector},
    st::NamedTuple,
    x::AbstractArray{T},
    metadata::NamedTuple,
) where{T<:Number}

    D1x = d1x1dmat(x)
    D2x = d2x1dmat(x)

    ū  = metadata.ū
    σu = metadata.σu

    CAEModel(NN, p, st, ū, σu, D1x, D2x,)
end

function Adapt.adapt_structure(to, model::CAEModel)
    p  = Adapt.adapt_structure(to, model.p )
    st = Adapt.adapt_structure(to, model.st)
    ū  = Adapt.adapt_structure(to, model.ū )
    σu = Adapt.adapt_structure(to, model.σu)
    D1x = Adapt.adapt_structure(to, model.D1x)
    D2x = Adapt.adapt_structure(to, model.D2x)

    CAEModel(model.NN, p, st, ū, σu, D1x, D2x,)
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