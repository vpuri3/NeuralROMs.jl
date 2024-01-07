
#===========================================================#
function normalizedata(
    u::AbstractArray,
    μ::Union{Number, AbstractVecOrMat},
    σ::Union{Number, AbstractVecOrMat},
)
    (u .- μ) ./ σ
end

function unnormalizedata(
    u::AbstractArray,
    μ::Union{Number, AbstractVecOrMat},
    σ::Union{Number, AbstractVecOrMat},
)
    (u .* σ) .+ μ
end
#===========================================================#

# For 2D, make X a tuple (X, Y). should work fine with dUdX, etc
# otherwise need `makeUfromXY`, `makeUfromX_newmodel` type functions

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
    x̄ = metadata.x̄
    ū = metadata.ū

    σx = metadata.σx
    σu = metadata.σu

    Icode = if isnothing(Icode)
        IT = T isa Type{Float64} ? Int64 : Int32
        Icode = similar(x, IT)
        fill!(Icode, true)
    end
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
        IT = T isa Type{Float64} ? Int64 : Int32
        Icode = similar(x, IT)
        fill!(Icode, true)
    end

    x_norm = normalizedata(x, model.x̄, model.σx)
    batch  = (x_norm, model.Icode)
    u_norm = model.NN(batch, p, model.st)[1]

    unnormalizedata(u_norm, model.ū, model.σu)
end

function dudx1(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function dudx1_internal(x)
        model(x, p)
    end

    if isa(autodiff, AutoFiniteDiff)
        finitediff_deriv1(dudx1_internal, x; ϵ)
    elseif isa(autodiff, AutoForwardDiff)
        forwarddiff_deriv1(dudx1_internal, x)
    end
end

function dudx2(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function dudx2_internal(x)
        model(x, p)
    end

    if isa(autodiff, AutoFiniteDiff)
        finitediff_deriv2(dudx2_internal, x; ϵ)
    elseif isa(autodiff, AutoForwardDiff)
        forwarddiff_deriv2(dudx2_internal, x)
    end
end

function dudx4(
    model::AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
    ϵ = nothing,
)
    function dudx4_internal(x)
        model(x, p)
    end

    if isa(autodiff, AutoFiniteDiff)
        finitediff_deriv4(dudx4_internal, x; ϵ)
    elseif isa(autodiff, AutoForwardDiff)
        forwarddiff_deriv4(dudx4_internal, x)
    else
        error("Got unsupported `autodiff = `$autodiff")
    end
end

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

    if isa(autodiff, AutoFiniteDiff)
        finitediff_jacobian(dudp_internal, p; ϵ)
    elseif isa(autodiff, AutoForwardDiff)
        forwarddiff_jacobian(dudp_internal, p)
    end
end
#===========================================================#
