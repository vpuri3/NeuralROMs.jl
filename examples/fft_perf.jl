
using BenchmarkTools, FFTW

# TODO
# - is it worthwhile to permute so FFT dimensions are first?
# - show we save plan_fft?
#
# - tullio performace impact for permutation applying weight kernel

C  = 16
Ns = 128, 128
K  = 100

x = rand(C, Ns..., K);
dx = 2:3

println("### FFT dim $dx ###")
_x = rfft(x, dx)
@btime rfft($x, $dx)
@btime irfft($_x, $Ns[1], $dx)

y = rand(Ns..., C, K)
dy = 1:2

println("### FFT dim $dy ###")
_y = rfft(y, dy)
@btime rfft($y, $dy)
@btime irfft($_y, $Ns[1], $dy)

println("### permutedims ###")
@btime permutedims($x, (2, 3, 1, 4))
@btime permutedims($y, (3, 1, 2, 4))

"""
Permutation is not advantageous

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

# tullio
using NNlib, Tullio

Ci, Co = 32, 32
M = 256
B = 100

@tullio Y[co, m, b] := W[co, ci, m] * X[ci, m, b]

#
