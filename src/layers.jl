#

"""
    implicit_encoder_decoder

Composition of a (possibly convolutional) encoder and an implicit
neural network decoder.

The input array `[Nx, Ny, C, B]` or `[C, Nx, Ny, B]` is expected to
contain `XYZ` coordinates in the last `dim` entries of the channel
dimension which is dictated by `channel_dim`. The number of channels in
the input array must match `encoder_width + D`, where `encoder_width`
is the expected input width of your encoder. The `encoder` network
is expected to work with whatever `channel_dim`, `encoder_channels`
you choose.

The coordinates are split and the remaining channels are
passed to `encoder` which compresses each `[:, :, :, 1]` slice into
a latent vector of length `L`. The output of the encoder is of size
`[L, B]`.

With a compressed mapping of each `image`, we are ready to apply the
decoder mapping. The decoder is an implicit neural network which expects
as input the concatenation of the latent vector and a query point.
The decoder returns the value of the target field at that point.

The decoder is usually a deep neural network and expects the channel
dimension to be the leading dimension. The decoder expects input with
size of leading dimension `L+dim`, and returns an array with leading
size `out_dim`.

Here, we feed it an array of size `[L+2, Nx, Ny, B]`, where the input
`Npoints` equal to `(Nx, Ny,)` is the number of training points in each
trajectory.

"""
function ImplicitEncoderDecoder(
    encoder::Lux.AbstractExplicitLayer,
    decoder::Lux.AbstractExplicitLayer,
    Npoints::NTuple{D, Int},
    encoder_width::Integer; # number of input channels to encoder
    channel_dim::Integer = D + 1,
    out_dim::Integer = 1,
) where{D}

    # channel length
    channel_split = 1:encoder_width, (encoder_width+1):(encoder_width+D)

    __ntimes = Base.Fix2(_ntimes, Npoints)

    Perm1, Perm2 = if (D > 1) | out_dim > 1
        @warn "haven't tested for D>1 yet. Possibly need to make some changes."
        @warn "like replacing the `ReshapeLayer`s with `permutedims`."

        # TODO - replace reshapes with permutedims ??
        PERM1 = ()
        PERM2 = ()
        PermLayer = WrappedFunction(Base.Fix2(permutedims, PERM))
    else
        ReshapeLayer((D, Npoints...)), ReshapeLayer(Npoints..., out_dim)
    end

    Chain(
        SplitRows(channel_split...; channel_dim), # u[N, 1, B], x[N, 1, B]
        Parallel(nothing, encoder, NoOpLayer()),  # ũ[L, B]   , x[N, 1, B]
        Parallel(vcat, WrappedFunction(__ntimes), ReshapeLayer((D, Npoints...))), # [L,N,B], [1,N,B] -> [L+1,N,B]
        decoder,  # [L+1,N,B] -> [out_dim, N, B]
        ReshapeLayer((Npoints..., out_dim)),
        # Base.Fix2(permutedims, PERM),
    )
end

"""
$SIGNATURES

Assumes channel dimension is 1
"""
function PermutedBatchNorm(c, num_dims) # assumes channel_dim = 1

    perm0 = ((2:num_dims-1)..., 1, num_dims)
    perm1 = (num_dims-1, (1:num_dims-2)..., num_dims)

    Chain(
        Base.Fix2(permutedims, perm0),
        BatchNorm(c),
        Base.Fix2(permutedims, perm1),
    )
end

"""
SplitRows

Split rows of ND array, into `Tuple` of ND arrays.
"""
struct SplitRows{T} <: Lux.AbstractExplicitLayer
    splits::T
    channel_dim::Int
end
function SplitRows(splits...; channel_dim = 1)
    SplitRows(splits, channel_dim)
end
Lux.initialparameters(::Random.AbstractRNG, ::SplitRows) = (;)
Lux.initialstates(::Random.AbstractRNG, ::SplitRows) = (;)
Lux.parameterlength(::SplitRows) = 0
Lux.statelength(::SplitRows) = 0

function (l::SplitRows)(x::AbstractArray, _, st)
    len = sum(length.(l.splits))

    C = l.channel_dim
    N = ndims(x)

    @assert eltype(len) === Int64 "SplitRows only accepts splits eltype Int64"
    @assert len == size(x, C) "Cannot split array of size $(size(x)) with
        splits $(l.splits) along dimension $(C)."

    xs = ()
    _cols = Tuple(Colon() for _ in 1:(C-1))
    cols_ = Tuple(Colon() for _ in (C+1):N)

    for split in l.splits
        if split isa Integer
            split = split:split
        end

        # https://github.com/JuliaGPU/CUDA.jl/issues/2009
        # _x = view(x, _cols..., split, cols_...)
        _x = getindex(x, _cols..., split, cols_...)

        @assert size(_x) == (size(x)[1:C-1]..., length(split), size(x)[C+1:end]...) "got $(size(_x))"

        xs = (xs..., _x)
    end

    xs, st
end

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
#
