
abstract type AbstractTransform{D} end

"""
$TYPEDEF

"""
struct FourierTransform{D} <: AbstractTransform{D}
    dims::NTuple{D, Int}
end

Base.eltype(::FourierTransform) = ComplexF32
Base.ndims(::FourierTransform{D}) where{D} = D
Base.size(F::FourierTransform) = 1

function Base.:*(F::FourierTransform, x::AbstractArray)
    FFTW.rfft(x, F.dims)
end

function Base.:\(F::FourierTransform, x::AbstractArray)
    FFTW.irfft(x, F.d, F.dims)
end

"""
$TYPEDEF

"""
struct CosineTransform{D} <: AbstractTransform{D}
    dims::NTuple{D, Int}
end

Base.eltype(::CosineTransform) = Float32
Base.ndims(::CosineTransform{D}) where{D} = D
Base.size(F::CosineTransform) = 1

function Base.:*(F::CosineTransform, x::AbstractArray)
    FFTW.dct(x, F.dims)
end

function Base.:\(F::CosineTransform, x::AbstractArray)
    FFTW.idct(x, F.dims)
end
#
