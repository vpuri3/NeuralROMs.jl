#

###
# high level functions
###

"""
$SIGNATURES

accept data in shape (C, X1, ..., Xd, B)

"""
function OpKernel(ch_in::Int, ch_out::Int, modes::NTuple{D, Int},
    activation = identity;
    transform = nothing,
    init = Lux.glorot_uniform,
) where{D}

    conv = OpConv(ch_in, ch_out, modes; transform, init)
    # lin = Dense(ch_in, ch_out; use_bias = false, init_weight = init)
    lin = Dense(ch_in, ch_out; init_weight = init)

    Chain(
        Lux.Parallel(+, lin, conv),
        Lux.WrappedFunction(activation),
    )
end

function OpKernelBilinear(ch_in1::Int, ch_in2::Int, ch_out::Int,
    modes::NTuple{D, Int},
    activation = identity;
    init = Lux.glorot_uniform,
) where{D}

    conv = OpConvBilinear(ch_in1, ch_in2, ch_out, modes; transform, init)
    # lin  = Bilinear((ch_in1, ch_in2) => ch_out; init_weight = init, use_bias = false)
    lin  = Bilinear((ch_in1, ch_in2) => ch_out; init_weight = init) # doesnt accept abstractarray

    Chain(
        Lux.Parallel(.+, lin, conv),
        Lux.WrappedFunction(activation),
    )
end

"""
$SIGNATURES

if you have linear dependence on `x1`, and nonlinear on `x2`, then

```
x1 → nonlin → y1 ↘
                  bilinear → project → z
x2 → linear → y2 ↗
```

# Arguments
- Call `nonlin` as `nonlin(x1, p, st)`
- Call `linear` as `linear(x2, p, st)`
- Call `bilin`  as `bilin((y1, y2), p, st)`

"""
function linear_nonlinear(nonlinear, linear, bilinear, project = NoOpLayer())

    Chain(
        Parallel(nothing, nonlinear, linear),
        bilinear,
        project,
    )
end

###
# layers
###

"""
Neural Operator convolution layer

# TODO `OpConv` design consierations
- create AbstractTransform interface
- innitialize params W_re, W_imag if eltype(Transform) isn't isreal
so that eltype(params) is always real

"""
struct OpConv{D, F, I} <: Lux.AbstractExplicitLayer
    ch_in::Int
    ch_out::Int
    modes::NTuple{D, Int}
    transform::F
    init::I

    function OpConv(ch_in, ch_out, modes, transform, init)
        new{
            length(modes),
            typeof(transform),
            typeof(init),
            }(
              ch_in, ch_out, modes, transform, init,
             )
    end
end

function OpConv(ch_in::Int, ch_out::Int, modes::NTuple{D, Int};
    init = Lux.glorot_uniform,
    transform = nothing,
) where{D}

    transform = isnothing(transform) ? (FFTW.rfft, FFTW.irfft) : transform

    OpConv(ch_in, ch_out, modes, transform, init)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::OpConv)

    dims  = (l.ch_in, l.ch_out, prod(l.modes))
    scale = one(Float32) # / (l.ch_in * l.ch_out)

    (;
     W = scale * l.init(rng, ComplexF32, dims...),
    )
end

Lux.initialstates(::Random.AbstractRNG, l::OpConv) = (;)
Lux.parameterlength(l::OpConv) = prod(l.modes) * l.ch_in * l.ch_out
Lux.statelength(::OpConv) = 0

function (l::OpConv{D})(x::AbstractArray, p, st::NamedTuple) where{D}

    @assert D == ndims(x) - 2
    @assert size(x, 1) == l.ch_in
    Ns = size(x)[2:D+1] # transform dimensions

    # transform, truncate
    x̂_tr, Ks = __opconv(x, l.transform, l.modes)

    # apply weight
    ŷ_tr = opconv_wt(x̂_tr, p.W)

    # pad, inv-transform
    y = opconv__(ŷ_tr, l.transform, l.modes, Ks, Ns)

    return y, st
end

"""
Neural Operator bilinear convolution layer

"""
struct OpConvBilinear{D, F, I} <: Lux.AbstractExplicitLayer
    ch_in1::Int
    ch_in2::Int
    ch_out::Int
    modes::NTuple{D, Int}
    transform::F
    init::I

    function OpConvBilinear(ch_in1, ch_in2, ch_out, modes, transform, init)
        new{
            length(modes),
            typeof(transform),
            typeof(init),
            }(
              ch_in1, ch_in2, ch_out, modes, transform, init,
             )
    end
end

function OpConvBilinear(ch_in1::Int, ch_in2::Int, ch_out::Int,
    modes::NTuple{D, Int};
    init = Lux.glorot_uniform,
    transform = nothing,
) where{D}

    transform = isnothing(transform) ? (FFTW.rfft, FFTW.irfft) : transform

    OpConvBilinear(ch_in1, ch_in2, ch_out, modes, transform, init)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::OpConvBilinear)

    dims  = (l.ch_in1, l.ch_in2, l.ch_out, prod(l.modes))
    scale = one(Float32) / (l.ch_in1 * l.ch_in2 * l.ch_out)

    (;
     W = scale * l.init(rng, ComplexF32, dims...),
    )
end

Lux.initialstates(::Random.AbstractRNG, l::OpConvBilinear) = (;)
Lux.parameterlength(l::OpConvBilinear) = prod(l.modes) * l.ch_in1 * l.ch_in2 * l.ch_out
Lux.statelength(::OpConvBilinear) = 0

"""
Extend OpConv to accept two inputs

Like Lux.Bilinear in modal space

"""
function (l::OpConvBilinear{D})((x, y)::NTuple{2, AbstractArray}, p, st::NamedTuple) where{D}

    @assert ndims(x) == ndims(y)
    @assert D == ndims(x) - 2
    @assert size(x, 1) == l.ch_in1
    @assert size(y, 1) == l.ch_in2
    @assert size(x)[2:end] == size(y)[2:end]

    Ns = size(x)[2:D+1] # transform dimensions

    # transform, truncate
    x̂_tr, Ks = __opconv(x, l.transform, l.modes)
    ŷ_tr, Ks = __opconv(y, l.transform, l.modes)

    # apply weight
    ẑ_tr = opconv_wt(x̂_tr, ŷ_tr, p.W)

    # pad, inv-transform
    z = opconv__(ẑ_tr, l.transform, l.modes, Ks, Ns)

    return z, st
end

###
# helper functions
###

"""
$SIGNATURES

Accepts `x` [C, N1...Nd, B].
Returns `x̂` [C, M, B] where `M = prod(modes)`

# Operations
- apply transform to `N1...Nd`:       `[K1...Kd, C, B] <- [K1...Kd, C, B]`
- truncate (discard high-freq modes): `[M1...Md, C, B] <- [K1...Kd, C, B]` where `modes == (M1...Md)`
"""
function __opconv(x, transform, modes::NTuple{D, Int}) where{D}

    # transform
    x̂ = transform[1](x, 2:D+1) # [K1...Kd, Ci, B]

    # truncate
    x̂_tr = view(x̂, :, map(d -> 1:d, modes)..., :) # [M1...Md, Ci, B]

    x̂_tr, size(x̂)[2:D+1]
end

"""
$SIGNATURES

"""
function opconv__(ŷ_tr, transform, modes::NTuple{D, Int}, Ks, Ns) where{D}

    Co = size(ŷ_tr)[1]   # channel len
    B  = size(ŷ_tr)[end] # batch   len

    # pad frequency modes
    ŷ = pad_array(ŷ_tr, (Co, Ks..., B))

    # inverse transform
    transform[2](ŷ, Ns[1], 2:D+1)
end

"""
$SIGNATURES

Apply pointwise linear transform in mode space, i.e. no mode-mixing.
Unique linear transform for each mode.

# Operations
- reshape: `[Ci, M, B] <- [Ci, M1...Md, B]` where `M = prod(M1...Md)`
- apply weight:
- reshape: `[Co, M1...Md, B] <- [Co, M, B]`
"""
function opconv_wt(x, W)

    D = ndims(x) - 2

    modes = size(x)[2:D+1]
    Ci, Co = size(W)[1:2]
    B = size(x)[end]

    @assert size(x, 1) == Ci

    # reshape
    X = reshape(x, (Ci, prod(modes), B)) # [Ci, M, B]

    # apply weight
    @tullio Y[co, m, b] := W[ci, co, m] * X[ci, m, b] # [Co, M, B]

    # un-reshape
    reshape(Y, (Co, modes..., B))
end

function opconv_wt(x, y, W)

    D = ndims(x) - 2

    modes = size(x)[2:D+1]
    C1, C2, Co = size(W)[1:3]
    B = size(x)[end]

    @assert size(x, 1) == C1
    @assert size(y, 1) == C2
    @assert size(y)[end] == B

    # reshape
    X = reshape(x, (C1, prod(modes), B)) # [C1, M, B]
    Y = reshape(y, (C2, prod(modes), B)) # [C2, M, B]

    # apply weight to get [Co, M, B]
    @tullio Z[co, m, b] := X[c1, m, b] * W[c1, c2, co, m] * Y[c2, m, b]

    # un-reshape
    reshape(Z, (Co, modes..., B))
end
#
