
#===========================================================#
@concrete mutable struct NeuralModel{Tx, Tu} <: AbstractNeuralModel
    NN
    st

    x̄::Tx
    σx::Tx

    ū::Tu
    σu::Tu
end

function NeuralModel(
    NN::Lux.AbstractExplicitLayer,
    st::NamedTuple,
    metadata::NamedTuple,
)
    x̄ = metadata.x̄
    ū = metadata.ū

    σx = metadata.σx
    σu = metadata.σu

    NeuralModel(NN, st, x̄, σx, ū, σu,)
end

function Adapt.adapt_structure(to, model::NeuralModel)
    st = Adapt.adapt_structure(to, model.st)
    x̄  = Adapt.adapt_structure(to, model.x̄ )
    ū  = Adapt.adapt_structure(to, model.ū )
    σx = Adapt.adapt_structure(to, model.σx)
    σu = Adapt.adapt_structure(to, model.σu)

    NeuralModel(
        model.NN, st, x̄, σx, ū, σu,
    )
end

function (model::NeuralModel)(
    x::AbstractArray,
    p::AbstractVector,
)
    x_norm = normalizedata(x, model.x̄, model.σx)
    u_norm = model.NN(x_norm, p, model.st)[1]
    unnormalizedata(u_norm, model.ū, model.σu)
end

#===========================================================#
@concrete mutable struct NeuralEmbeddingModel{Tx, Tu} <: AbstractNeuralModel
    NN
    st
    Icode

    x̄::Tx
    σx::Tx

    ū::Tu
    σu::Tu
end

function NeuralEmbeddingModel(
    NN::Lux.AbstractExplicitLayer,
    st::NamedTuple,
    x::AbstractArray{T},
    metadata::NamedTuple,
    Icode::Union{Nothing,AbstractArray{<:Integer}} = nothing,
) where{T<:Number}

    Icode = if isnothing(Icode)
        sz = (1, size(x)[2:end]...,)
        Icode = similar(x, Int32, sz)
        fill!(Icode, true)
    end

    NeuralEmbeddingModel(NN, st, metadata, Icode,)
end

function NeuralEmbeddingModel(
    NN::Lux.AbstractExplicitLayer,
    st::NamedTuple,
    metadata::NamedTuple,
    Icode::AbstractArray{<:Integer},
)
    x̄ = metadata.x̄
    ū = metadata.ū

    σx = metadata.σx
    σu = metadata.σu

    NeuralEmbeddingModel(NN, st, Icode, x̄, σx, ū, σu,)
end

function Adapt.adapt_structure(to, model::NeuralEmbeddingModel)
    st = Adapt.adapt_structure(to, model.st)
    Icode = Adapt.adapt_structure(to, model.Icode)
    x̄  = Adapt.adapt_structure(to, model.x̄ )
    ū  = Adapt.adapt_structure(to, model.ū )
    σx = Adapt.adapt_structure(to, model.σx)
    σu = Adapt.adapt_structure(to, model.σu)

    NeuralEmbeddingModel(
        model.NN, st, Icode, x̄, σx, ū, σu,
    )
end

function (model::NeuralEmbeddingModel)(
    x::AbstractArray,
    p::AbstractVector,
)

    Zygote.@ignore Icode = if isnothing(model.Icode)
        sz = (1, size(x)[2:end]...,)
        Icode = similar(x, Int32, sz)
        fill!(Icode, true)
    end

    x_norm = normalizedata(x, model.x̄, model.σx)
    batch  = (x_norm, model.Icode)
    u_norm = model.NN(batch, p, model.st)[1]

    unnormalizedata(u_norm, model.ū, model.σu)
end

#===========================================================#

function dudx1_1D(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function dudx1_1D_internal(x)
        model(x, p)
    end

    doautodiff1(dudx1_1D_internal, x, autodiff, ϵ)
end

function dudx2_1D(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function dudx2_1D_internal(x)
        model(x, p)
    end

    doautodiff2(dudx2_1D_internal, x, autodiff, ϵ)
end

function dudx4_1D(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function dudx4_1D_internal(x)
        model(x, p)
    end

    doautodiff4(dudx4_1D_internal, x, autodiff, ϵ)
end

#===========================================================#

function dudx1_2D(
    model::AbstractNeuralModel,
    xy::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    @assert size(xy, 1) == 2
    @assert ndims(xy) == 2 "TODO: implement reshapes to handle ND arrays"

    x = getindex(xy, 1:1, :)
    y = getindex(xy, 2:2, :)

    function dudx1_2D_internal_dx(x_internal)
        xy_internal = vcat(x_internal, y)
        model(xy_internal, p)
    end

    function dudx1_2D_internal_dy(y_internal)
        xy_internal = vcat(x, y_internal)
        model(xy_internal, p)
    end

    u, udx = doautodiff1(dudx1_2D_internal_dx, x, autodiff, ϵ)
    _, udy = doautodiff1(dudx1_2D_internal_dy, y, autodiff, ϵ)

    u, (udx, udy,)
end

#===========================================================#

function dudp(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function dudp_internal(p)
        model(x, p)
    end

    doautodiff_jacobian(dudp_internal, p, autodiff, ϵ)
end
#===========================================================#

