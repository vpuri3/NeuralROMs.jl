#
"""
Diagonal layer
"""
struct Diag{L, I} <: Lux.AbstractExplicitLayer
    len::L
    init::I
end

Diag(len::Int; init = Lux.glorot_uniform) = Diag(len, init)

function Lux.initialparameters(rng::Random.AbstractRNG, l::Diag)
    (;
     diag = l.init(rng, l.len),
    )
end

Lux.initialstates(::Random.AbstractRNG, l::Diag) = (;)
Lux.parameterlength(l::Diag) = l.len
Lux.statelength(::Diag) = 0

function (l::Diag)(x::AbstractArray, ps, st::NamedTuple)

    D = Diagonal(ps.diag)
    y = D * x

    return y, st
end

"""
Attention Layer

single layer model with no nonlinearity (single head linear attention)

u = NN(f)
q = Wq * u
k = Wk * u
v = Wv * u

v = activation(q * k') * u
"""
struct Atten{I, TI, F} <: Lux.AbstractExplicitLayer
    in_dims::I
    out_dims::I
    init::TI
    activation::F
end

function Atten(in_dims::Int, out_dims::Int; init = Lux.glorot_uniform,
               activation = identity)
    Atten(in_dims, out_dims, init, activation)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::Atten)
    (;
     Wq = l.init(rng, l.out_dims, l.in_dims),
     Wk = l.init(rng, l.out_dims, l.in_dims),
     Wv = l.init(rng, l.out_dims, l.in_dims),
    )
end

Lux.initialstates(::Random.AbstractRNG, l::Atten) = (;)
Lux.parameterlength(l::Atten) = 3 * l.in_dims * l.out_dims
Lux.statelength(::Atten) = 0

function (l::Atten)(x::AbstractArray, ps, st::NamedTuple)
    q = ps.Wq * x
    k = ps.Wk * x
    v = ps.Wv * x

    y = l.activation(q * k') * v

    return y, st
end

"""
$SIGNATURES

accept data in shape (C, X1, ..., Xd, B)

TODO - Extend this to accept two inputs (optionally)
For example if you have linear dependence on `f`, and nonlinear on `ν`,
then

```
ν -> lifting -> OpKernel (nl) -> ... -> OpKernel -> project (lin) -> out
                                           /|\
                                            |
                                           forc
```

where OpKernel(ν, f) will perform something like `ν̂' * W * f̂` in
fourier space.

options: Bilinear, Branch/ Parallel, PairwiseFusion
"""
function OperatorKernel(in_dims::Int, out_dims::Int, modes::NTuple{D, Int};
    activation = identity,
    transform = nothing,
    init = Lux.glorot_uniform
) where{D}

    conv = OperatorConv(in_dims, out_dims, modes; transform, init)
    lin = Dense(in_dims, out_dims; use_bias = false, init_weight = init)

    Chain(
        Lux.Parallel(+, lin, conv),
        Lux.WrappedFunction(activation),
    )
end

"""
Neural Operator convolution layer

"""
struct OperatorConv{D, F, I} <: Lux.AbstractExplicitLayer
    in_dims::Int
    out_dims::Int
    modes::NTuple{D, Int}
    transform::F
    init::I

    function OperatorConv(in_dims, out_dims, modes, transform, init)
        new{
            length(modes),
            typeof(transform),
            typeof(init),
            }(
              in_dims, out_dims, modes, transform, init,
             )
    end
end

function OperatorConv(in_dims::Int, out_dims::Int, modes::NTuple{D, Int};
    init = Lux.glorot_uniform,
    transform = nothing,
) where{D}

    transform = isnothing(transform) ? (FFTW.rfft, FFTW.irfft) : transform

    OperatorConv(in_dims, out_dims, modes, transform, init)
end

function Lux.initialparameters(rng::Random.AbstractRNG, l::OperatorConv)
    scale = one(Float32) # / (l.in_dims * l.out_dims)

    # TODO - innitialize as W_re, W_imag if eltype(Transform) isreal
    #        so eltype(params) is always real
    # TODO - create AbstractTransform interface
    (;
     W = scale * l.init(rng, ComplexF32, l.out_dims, l.in_dims, prod(l.modes)),
    )
end

# put fft plan in state later. generate new if FFTW.assert_applicable is false
Lux.initialstates(::Random.AbstractRNG, l::OperatorConv) = (;)
Lux.parameterlength(l::OperatorConv) = prod(l.modes) * l.in_dims * l.out_dims
Lux.statelength(::OperatorConv) = 0

function (l::OperatorConv{D})(x::AbstractArray, p, st::NamedTuple) where{D}

    @assert D == ndims(x) - 2
    B = size(x)[end] # length of batch dim

    # permute so FFT dims in front: [N1...Nd, Ci, B] <- [Ci, N1...Nd, B]
    x_perm = permutedims(x, (2:D+1..., 1, D+2))

    # transform
    # F1 = l.transform(x_perm, 1:D)
    # x̂ = F1 * x_perm   # [K1...Kd, Ci, B]
    x̂ = l.transform[1](x_perm, 1:D)

    # truncate
    x̂_tr = view(x̂, map(d -> 1:d, l.modes)..., :, :)     # [M1...Md, Ci, B]
    x̂_re = reshape(x̂_tr, (prod(l.modes), l.in_dims, B)) # [M, Ci, B]

    # apply weights # try permutedims and use NNlib.batched_mul
    @tullio ŷ_re[m, co, b] := p.W[co, ci, m] * x̂_re[m, ci, b]

    ŷ_tr = reshape(ŷ_re, (l.modes..., l.out_dims, B))

    # pad frequency modes
    ŷ = pad_array(ŷ_tr, (size(x̂)[1:D]..., l.out_dims, B))

    # inverse transform
    # F2 = l.transform(x_perm, 1:D)
    # y_perm = F2 \ ŷ
    y_perm = l.transform[2](ŷ, size(x_perm, 1), 1:D)

    # unpermute
    y = permutedims(y_perm, (D+1, 1:D..., D+2)) # [Co, N1...Nd, B] <- [N1...Nd, Co, B]

    return y, st
end

function (l::OperatorConv{D})((x, z)::Tuple, p, st::NamedTuple) where{D}

end

"""
$SIGNATURES

make subtype of Lux.AbstractExplicitContainerLayer
Basically PairwiseFusionLayer where connection is also a layer
"""
struct FusionLayer
    layer1
    layer2
    connection
end

function lin_nonlin((x_ln, x_nl), p, st)

    # linear
    y_ln, st_ln = NN_ln(x_ln, p.ln, st.ln)

    # nonlinear
    y_nl, st_nl = NN_nl(x_nl, p.nl, st.nl)

    # connection
    y, st_cnn = NN_lin((y_ln, y_nl), p.cnn, st.cnn)

    st = (; ln = st_ln, nl = st_nl, cnn = st_cnn)

    y, st
end

#
