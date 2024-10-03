#
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
    NN::AbstractLuxLayer,
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
#===========================================================#
#### NEURAL MODEL GRADIENTS, JACOBIAN
#===========================================================#
#===========================================================#

const DUDX_1D_FUNCS = (:dudx1_1D, :dudx2_1D, :dudx3_1D, :dudx4_1D,)
const DUDX_2D_FUNCS = (:dudx1_2D, :dudx2_2D,)

for (dudx_f, do_ad_f) in zip(DUDX_1D_FUNCS, DO_AD_FUNCS)
    @eval function $dudx_f(
        model::AbstractNeuralModel,
        x::AbstractArray,
        p::AbstractVector;
        autodiff::ADTypes.AbstractADType = AutoForwardDiff(),
        ϵ = nothing,
    )
        @assert size(x, 1) == 1 "input grid must be 1D. Got size(x) = $(size(x))"

        function dudx_1D_internal(x)
            model(x, p)
        end

        $do_ad_f(dudx_1D_internal, x, autodiff, ϵ)
    end
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

function dudx2_2D(
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

    function dudx2_2D_internal_dx(x_internal)
        xy_internal = vcat(x_internal, y)
        model(xy_internal, p)
    end

    function dudx2_2D_internal_dy(y_internal)
        xy_internal = vcat(x, y_internal)
        model(xy_internal, p)
    end

    u, udx, udxx = doautodiff2(dudx2_2D_internal_dx, x, autodiff, ϵ)
    _, udy, udyy = doautodiff2(dudx2_2D_internal_dy, y, autodiff, ϵ)

    u, (udx, udy,), (udxx, udyy,)
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
#
