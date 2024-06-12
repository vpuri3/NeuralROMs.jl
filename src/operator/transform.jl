#
# TODO: subtype AbstractTransform <: Lux.AbstractExplicitLayer and make
# TODO: OpConv(Bilinear) a hyper network/ Lux Container layer.
# TODO: then we can think about trainable transform types
abstract type AbstractTransform{D} end

"""
$TYPEDEF
"""
struct FourierTransform{D} <: AbstractTransform{D} # apply rfft on [1:D]
    mesh::NTuple{D, Int}
end

FourierTransform(mesh::Int...) = FourierTransform(mesh)
Base.eltype(::FourierTransform) = ComplexF32
Base.ndims(::FourierTransform{D}) where{D} = D

function Base.:*(F::FourierTransform{D}, x::AbstractArray) where{D}
    FFTW.rfft(x, 1:D)
end

function Base.:\(F::FourierTransform{D}, x::AbstractArray) where{D}
    FFTW.irfft(x, F.mesh[1], 1:D)
end

"""
$TYPEDEF

"""
struct CosineTransform{D} <: AbstractTransform{D}
    mesh::NTuple{D, Int}
end

CosineTransform(mesh::Int...) = FourierTransform(mesh)
Base.eltype(::CosineTransform) = Float32
Base.ndims(::CosineTransform{D}) where{D} = D

function Base.:*(F::CosineTransform{D}, x::AbstractArray) where{D}
    dct(x, 1:D)
end

function Base.:\(F::CosineTransform{D}, x::AbstractArray) where{D}
    idct(x, 1:D)
end
#
