#
#===========================================================#

clamp_vanilla(x::AbstractArray, δ) = @. clamp(x, -δ, δ)
clamp_tanh(x::AbstractArray, δ)    = @. δ * tanh_fast(x)
clamp_sigmoid(x::AbstractArray, δ) = @. δ * (2 * sigmoid_fast(x) - 1)
clamp_softsign(x::AbstractArray,δ) = @. δ * softsign(x)

struct ClampVanilla{T <: Real}  <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampTanh{T <: Real}     <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampSigmoid{T <: Real}  <: Lux.AbstractExplicitLayer; δ::T; end
struct ClampSoftsign{T <: Real} <: Lux.AbstractExplicitLayer; δ::T; end

Lux.initialstates(::ClampVanilla ) = (;)
Lux.initialstates(::ClampTanh    ) = (;)
Lux.initialstates(::ClampSigmoid ) = (;)
Lux.initialstates(::ClampSoftsign) = (;)

Lux.initialparameters(::ClampVanilla ) = (;)
Lux.initialparameters(::ClampTanh    ) = (;)
Lux.initialparameters(::ClampSigmoid ) = (;)
Lux.initialparameters(::ClampSoftsign) = (;)

Lux.statelength(::ClampVanilla ) = 0
Lux.statelength(::ClampTanh    ) = 0
Lux.statelength(::ClampSigmoid ) = 0
Lux.statelength(::ClampSoftsign) = 0

Lux.parameterlength(::ClampVanilla ) = 0
Lux.parameterlength(::ClampTanh    ) = 0
Lux.parameterlength(::ClampSigmoid ) = 0
Lux.parameterlength(::ClampSoftsign) = 0

(l::ClampVanilla )(x::AbstractArray, p, st) = clamp_vanilla(x, l.δ), st
(l::ClampTanh    )(x::AbstractArray, p, st) = clamp_tanh(x, l.δ), st
(l::ClampSigmoid )(x::AbstractArray, p, st) = clamp_sigmoid(x, l.δ), st
(l::ClampSoftsign)(x::AbstractArray, p, st) = clamp_softsign(x, l.δ), st
#===========================================================#
#
