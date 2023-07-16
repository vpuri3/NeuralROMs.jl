
using BenchmarkTools, FFTW, LinearAlgebra

FFTW.set_num_threads(8)
BLAS.set_num_threads(2)

if true
    
C  = 16
Ns = 128, 128
K  = 100

x = rand(C, Ns..., K);
y = rand(Ns..., C, K)

dx = 2:3
dy = 1:2

println("### rFFT/irFFT dim $dx (C, Nx, Ny, B) ###")
_x = rfft(x, dx)
@btime rfft($x, $dx)
@btime irfft($_x, $Ns[1], $dx)

println("### rFFT/irFFT dim $dy (Nx, Ny, C, B) ###")
_y = rfft(y, dy)
@btime rfft($y, $dy)
@btime irfft($_y, $Ns[1], $dy)

println("### permutedims ###")
@btime permutedims($x, (2, 3, 1, 4))
@btime permutedims($y, (3, 1, 2, 4))

end

"""
Bringing FFT dimensions to the front is not advantageous

```
julia> include("examples/fft_perf.jl")
### FFT dim 2:3 ###
  61.609 ms (9 allocations: 203.13 MiB)
  108.098 ms (13 allocations: 606.25 MiB)
### FFT dim 1:2 ###
  56.772 ms (9 allocations: 203.13 MiB)
  100.064 ms (13 allocations: 606.25 MiB)
### permutedims ###
  41.810 ms (3 allocations: 200.00 MiB)
  25.929 ms (3 allocations: 200.00 MiB)
```
"""

# linear transform with Tullio
using Tullio

if false
    
Ci, Co = 32, 64
M = 1024
B = 100

X = rand(Ci, M, B)
W = rand(Co, Ci, M)

W = respahe(W, (Co, Ci, M))
@btime @tullio Y[co, m, b] := W[co, ci, m] * X[ci, m, b]

W = reshape(W, (Ci, Co, M))
@btime @tullio Y[co, m, b] := W[ci, co, m] * X[ci, m, b] # winner

W = reshape(W, (Co, M, Ci))
@btime @tullio Y[co, m, b] := W[co, m, ci] * X[ci, m, b] # worst - do not run

W = reshape(W, (Ci, M, Co))
@btime @tullio Y[co, m, b] := W[ci, m, co] * X[ci, m, b]

W = reshape(W, (M, Co, Ci))
@btime @tullio Y[co, m, b] := W[m, co, ci] * X[ci, m, b]

W = reshape(W, (M, Ci, Co))
@btime @tullio Y[co, m, b] := W[m, ci, co] * X[ci, m, b]

end

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
