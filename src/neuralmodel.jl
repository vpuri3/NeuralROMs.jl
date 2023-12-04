
#===========================================================#
normalizedata(u::AbstractArray, μ::Number, σ::Number) = (u .- μ) / σ
unnormalizedata(u::AbstractArray, μ::Number, σ::Number) = (u * σ) .+ μ
#===========================================================#

# For 2D, make X a tuple (X, Y). should work fine with dUdX, etc
# otherwise need `makeUfromXY`, `makeUfromX_newmodel` type functions

@concrete mutable struct NeuralSpaceModel{T} <: AbstractNeuralModel
    NN
    st
    Icode

    x̄::T
    σx::T

    ū::T
    σu::T
end

function Adapt.adapt_structure(to, model::NeuralSpaceModel)
    st = Adapt.adapt_structure(to, model.st)
    Icode = Adapt.adapt_structure(to, model.Icode)

    NeuralSpaceModel(
        model.NN, st, Icode, model.x̄, model.σx, model.ū, model.σu,
    )
end

function (model::NeuralSpaceModel)(x::AbstractArray, p::AbstractVector)
    # assume x is of shape (1, N). may change to vec later.
    @assert size(model.Icode) == size(x)

    x_norm = normalizedata(x, model.x̄, model.σx)
    batch  = (x_norm, model.Icode)
    u_norm = model.NN(batch, p, model.st)[1]

    unnormalizedata(u_norm, model.ū, model.σu)
end

function dudx1(
    model::NeuralSpaceModel,
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
    model::NeuralSpaceModel,
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

function dudp(
    model::NeuralSpaceModel,
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

