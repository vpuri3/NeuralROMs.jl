#
using BenchmarkTools, FFTW, CUDA

C  = 16
Ns = 128, 128
K  = 64

# x = rand(C, Ns..., K) |> cu
# dx = 2:3

x = rand(Ns..., K, C) |> cu
y = rand(Ns..., C, K) |> cu

dx = 1:2
dy = 1:2

println("### FFT dim $dx ###")
_x = rfft(x, dx)
@btime CUDA.@sync rfft($x, $dx)
@btime CUDA.@sync irfft($_x, $Ns[1], $dx)

println("### FFT dim $dy ###")
_y = rfft(y, dy)
@btime CUDA.@sync rfft($y, $dy)
@btime CUDA.@sync irfft($_y, $Ns[1], $dy)

println("### permutedims ###")
perm_x = (4, 1, 2, 3)
perm_y = (3, 1, 2, 4)
@btime CUDA.@sync permutedims($x, $perm_x)
@btime CUDA.@sync permutedims($y, $perm_y)

# println("FFT on PermutedDimsArray")
# @btime CUDA.@sync rfft(PermutedDimsArray($y, $perm_y), $dx) # very bad

"""
julia> include("examples/cuda_perf.jl")
### FFT dim 1:2 ###
  1.149 ms (36 allocations: 1.73 KiB)
  2.927 ms (37 allocations: 1.70 KiB)
### FFT dim 1:2 ###
  1.085 ms (36 allocations: 1.73 KiB)
  1.782 ms (44 allocations: 2.03 KiB)
### permutedims ###
  2.941 ms (70 allocations: 4.03 KiB)
  796.055 μs (24 allocations: 1.08 KiB)
"""

nothing

#=
# linear transform with Tullio
using NNlib, Tullio

Ci, Co = 32, 64
M = 1024
B = 100

X = rand(Ci, M, B)

W = rand(Co, Ci, M)
@btime @tullio Y[co, m, b] := W[co, ci, m] * X[ci, m, b] # current

W = rand(Ci, Co, M)
@btime @tullio Y[co, m, b] := W[ci, co, m] * X[ci, m, b] # winner

W = rand(Co, M, Ci)
@btime @tullio Y[co, m, b] := W[co, m, ci] * X[ci, m, b]

W = rand(Ci, M, Co)
@btime @tullio Y[co, m, b] := W[ci, m, co] * X[ci, m, b]

W = rand(M, Co, Ci)
@btime @tullio Y[co, m, b] := W[m, co, ci] * X[ci, m, b]

W = rand(M, Ci, Co)
@btime @tullio Y[co, m, b] := W[m, ci, co] * X[ci, m, b]
=#

"""
```
Ci, Co = 64, 32
M = 1024
B = 100

julia> include("examples/fft_perf.jl")
  19.883 ms (116 allocations: 25.01 MiB)
  8.989 ms (116 allocations: 25.01 MiB)  # winner
  43.560 ms (116 allocations: 25.01 MiB)
  11.161 ms (116 allocations: 25.01 MiB)
  67.615 ms (118 allocations: 25.01 MiB)
  72.830 ms (116 allocations: 25.01 MiB)

Ci, Co = 64, 32
M = 1024
B = 100

julia> include("examples/fft_perf.jl")
  39.987 ms (116 allocations: 50.01 MiB)
  18.441 ms (116 allocations: 50.01 MiB)  # winner
  109.079 ms (118 allocations: 50.01 MiB)
  20.383 ms (116 allocations: 50.01 MiB)
  130.341 ms (117 allocations: 50.01 MiB)
  140.711 ms (117 allocations: 50.01 MiB)

Ci, Co = 32, 64
M = 1024
B = 100

julia> include("examples/fft_perf.jl")
  19.794 ms (117 allocations: 50.01 MiB)
  9.101 ms (116 allocations: 50.01 MiB)  # winner
  47.273 ms (117 allocations: 50.01 MiB)
  11.320 ms (116 allocations: 50.01 MiB)
  64.177 ms (117 allocations: 50.01 MiB)
  66.034 ms (116 allocations: 50.01 MiB)
```
"""

nothing
#
