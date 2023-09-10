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
    use_bias = false,
) where{D}

    activation = fastify(activation)

    conv = OpConv(ch_in, ch_out, modes; transform, init)
    loc = Dense(ch_in, ch_out; init_weight = init, use_bias)

    # Parallel(+, loc, conv), # x -> lox(x) + conv(x)

    Chain(
        BranchLayer(loc, conv),      # x        -> (loc(x), conv(x))
        WrappedFunction(sum),        # (x1, x2) -> x1 + x2
        WrappedFunction(activation), # x        -> act(x)
    )
end

function OpKernelBilinear(ch_in1::Int, ch_in2::Int, ch_out::Int,
    modes::NTuple{D, Int},
    activation = identity;
    transform = nothing,
    init = Lux.glorot_uniform,
) where{D}

    activation = fastify(activation)

    null = NoOpLayer()
    conv = OpConvBilinear(ch_in1, ch_in2, ch_out, modes; transform, init)
    loc  = Bilinear((ch_in1, ch_in2) => ch_out; init_weight = init, use_bias = false)

    # Parallel(+, null, null),
    Chain(
        BranchLayer(loc, conv),      # x        -> (loc(x), conv(x))
        WrappedFunction(sum),        # (x1, x2) -> x1 + x2
        WrappedFunction(activation), # x        -> act(x)
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
function linear_nonlinear(split, nonlin, linear, bilinear, project = NoOpLayer())

    Chain(
        split,                             # x -> (x1, x2)
        Parallel(nothing, nonlin, linear), # (x1, x2) -> (f(x1), g(x2))
        bilinear,                          # (f(x1), g(x2)) -> y
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

"""
$SIGNATURES
"""
function OpConv(ch_in::Int, ch_out::Int, modes::NTuple{D, Int};
    init = Lux.glorot_uniform,
    transform = nothing,
) where{D}

    transform = isnothing(transform) ? (FFTW.rfft, FFTW.irfft) : transform
    # TODO
    if isnothing(transform)
        transform = FourierTransform(mesh...)
    end

    OpConv(ch_in, ch_out, modes, transform, init)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::OpConv)

    dims  = (prod(l.modes), l.ch_in, l.ch_out)
    scale = one(Float32) / (l.ch_in * l.ch_out)

    (;
        W = scale * l.init(rng, ComplexF32, dims...), # TODO eltype(l.transform)
    )
end

Lux.initialstates(::Random.AbstractRNG, l::OpConv) = (;)
Lux.parameterlength(l::OpConv) = prod(l.modes) * l.ch_in * l.ch_out
Lux.statelength(::OpConv) = 0

function (l::OpConv{D})(x::AbstractArray, p, st::NamedTuple) where{D}

    @assert ndims(x) == D + 2     "got $(ndims(x)) == $(D+2)"
    @assert size(x, 1) == l.ch_in "got $(size(x, 1)) == $(l.ch_in)"
    Ns = size(x)[2:D+1] # transform dimensions

    # permute, transform, truncate
    x̂_tr, Ks = __opconv(x, l.transform, l.modes)

    # apply weight
    ŷ_tr = opconv_wt(x̂_tr, p.W)

    # pad, inv-transform, unpermute
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

    dims  = (prod(l.modes), l.ch_out, l.ch_in1, l.ch_in2)
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

    @assert ndims(x) == ndims(y) == D + 2
    @assert D == ndims(x) - 2      "got $(D) == $(ndims(x))"
    @assert size(x, 1) == l.ch_in1 "got $(size(x, 1)) == $(l.ch_in1)"
    @assert size(y, 1) == l.ch_in2 "got $(size(y, 1)) == $(l.ch_in2)"
    @assert size(x)[2:end] == size(y)[2:end]

    Ns = size(x)[2:D+1] # transform dimensions

    # permute, transform, truncate
    x̂_tr, Ks = __opconv(x, l.transform, l.modes)
    ŷ_tr, Ks = __opconv(y, l.transform, l.modes)

    # apply weight
    ẑ_tr = opconv_wt(x̂_tr, ŷ_tr, p.W)

    # pad, inv-transform, unpermute
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

    N = ndims(x)
    perm1 = ((2:D+1)..., 1, N)
    # perm2 = (D+1, (1:D)..., N)

    # permute (move transform dims to front)
    x = permutedims(x, perm1) # [N1...Nd, Ci, B] <- [Ci, N1...Nd, B]

    # transform
    x̂ = transform[1](x, 1:D)  # [K1...Kd, Ci, B]

    # truncate
    x̂_tr = view(x̂, map(d -> 1:d, modes)..., :, :) # [M1...Md, Ci, B]

    x̂_tr, size(x̂)[1:D]
end

"""
$SIGNATURES

"""
function opconv__(ŷ_tr, transform, modes::NTuple{D, Int}, Ks, Ns) where{D}

    Co = size(ŷ_tr)[D+1] # channel len
    B  = size(ŷ_tr)[end] # batch   len

    N = ndims(ŷ_tr)
    # perm1 = ((2:D+1)..., 1, N)
    perm2 = (D+1, (1:D)..., N)

    # pad frequency modes
    ŷ = pad_array(ŷ_tr, (Ks..., Co, B)) # [K1...Kd, Co, B] <- [M1...Md, Co, B]

    # inverse transform
    y = transform[2](ŷ, Ns[1], 1:D)     # [N1...Nd, Co, B]

    # unpermute
    permutedims(y, perm2)               # [C, N1...Nd, B] <- [N1...Nd, C, B]
end

"""
$SIGNATURES

Apply pointwise linear transform in mode space, i.e. no mode-mixing.
Unique linear transform for each mode.

# Operations
- reshape: `[Ci, M, B] <- [Ci, M1...Md, B]` where `M = prod(M1...Md)`
- apply weight
- reshape: `[Co, M1...Md, B] <- [Co, M, B]`
"""
function opconv_wt(x, W)

    D = ndims(x) - 2

    modes = size(x)[1:D]
    M, Ci, Co = size(W)
    B = size(x)[end]

    @assert size(x, D+1) == Ci "got $(size(x, 1)) == $Ci"
    @assert M == prod(modes)   "got $M == $(prod(modes))"

    # reshape
    X = reshape(x, (prod(modes), Ci, B))              # [M, Ci, B]

    # apply weight
    @tullio Y[m, co, b] := W[m, ci, co] * X[m, ci, b] # [M, Co, B]

    # un-reshape
    reshape(Y, (modes..., Co, B))
end

function opconv_wt(x, y, W)

    D = ndims(x) - 2

    modes = size(x)[1:D]
    M, Co, C1, C2 = size(W)
    B = size(x)[end]

    @assert size(x, D+1) == C1  "got $(size(x, D+1)) == $C1"
    @assert size(y, D+1) == C2  "got $(size(y, D+1)) == $C2"
    @assert size(y)[end] == B "got $(size(y)[end]) == $B"
    @assert M == prod(modes)  "got $M == $(prod(modes))"

    # reshape
    X = reshape(x, (prod(modes), C1, B)) # [M, C1, B]
    Y = reshape(y, (prod(modes), C2, B)) # [M, C2, B]

    # apply weight to get [Co, M, B]
    # see examples/cuda_perf.jl for Tullio kernel triage
    @tullio Z1[m, co, c2, b] := W[m, co, c1, c2] * X[m, c1, b]
    @tullio Z2[m, co, b]     := Z1[m, co, c2, b] * Y[m, c2, b] # [M, Co, B]

    # un-reshape
    reshape(Z2, (modes..., Co, B))
end
#
