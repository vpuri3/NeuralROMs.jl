
using BenchmarkTools, FFTW, LinearAlgebra

BLAS.set_num_threads(2)
FFTW.set_num_threads(8)

if false

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
    
Ci, Co = 32, 32
M = 1024
B = 100

X = rand(Ci, M, B)
W = rand(Co, Ci, M)

println("#==================#")
println("### Tullio tests ###")
println("# with x[C, M, B]")
println("#==================#")

GC.gc()

println("# Y[co, m, b] = W[co, ci, m] * X[ci, m, b]")
W = reshape(W, (Co, Ci, M))
@btime @tullio Y[co, m, b] := W[co, ci, m] * X[ci, m, b]

GC.gc()

println("# Y[co, m, b] = W[ci, co, m] * X[ci, m, b]")
W = reshape(W, (Ci, Co, M))
@btime @tullio Y[co, m, b] := W[ci, co, m] * X[ci, m, b] # winner

GC.gc()

println("# Y[co, m, b] = W[co, m, ci] * X[ci, m, b]")
# W = reshape(W, (Co, M, Ci))
# @btime @tullio Y[co, m, b] := W[co, m, ci] * X[ci, m, b] # worst - do not run

GC.gc()

println("# Y[co, m, b] = W[ci, m, co] * X[ci, m, b]")
W = reshape(W, (Ci, M, Co))
@btime @tullio Y[co, m, b] := W[ci, m, co] * X[ci, m, b]

GC.gc()

println("# Y[co, m, b] = W[m, ci, co] * X[ci, m, b]")
W = reshape(W, (M, Co, Ci))
@btime @tullio Y[co, m, b] := W[m, co, ci] * X[ci, m, b]

GC.gc()

println("# Y[co, m, b] = W[m, co, ci] * X[ci, m, b]")
W = reshape(W, (M, Ci, Co))
@btime @tullio Y[co, m, b] := W[m, ci, co] * X[ci, m, b]

end

"""
#==================#
### Tullio tests ###
# with x[C, M, B]
#==================#
# Y[co, m, b] = W[co, ci, m] * X[ci, m, b]
  84.978 ms (2 allocations: 25.00 MiB)    
# Y[co, m, b] = W[ci, co, m] * X[ci, m, b]
  25.760 ms (2 allocations: 25.00 MiB)    
# Y[co, m, b] = W[co, m, ci] * X[ci, m, b]
# Y[co, m, b] = W[ci, m, co] * X[ci, m, b]
  37.652 ms (2 allocations: 25.00 MiB) 
# Y[co, m, b] = W[m, ci, co] * X[ci, m, b]
  608.589 ms (2 allocations: 25.00 MiB)
# Y[co, m, b] = W[m, co, ci] * X[ci, m, b]    
  421.808 ms (2 allocations: 25.00 MiB)
"""

if false
println("#==================#")
println("### Tullio tests ###")
println("# with x[M, C, B]")
println("#==================#")

Ci, Co = 32, 32
M = 1024
B = 100

X = rand(M, Ci, B)
W = rand(Co, Ci, M)

GC.gc()

println("# Y[m, co, b] = W[co, ci, m] * X[m, ci, b]")
W = reshape(W, (Co, Ci, M))
@btime @tullio Y[m, co, b] := W[co, ci, m] * X[m, ci, b]

GC.gc()

println("# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]")
W = reshape(W, (Ci, Co, M))
@btime @tullio Y[m, co, b] := W[ci, co, m] * X[m, ci, b]

GC.gc()

println("# Y[m, co, b] = W[m, ci, co] * X[m, ci, b]")
W = reshape(W, (M, Co, Ci))
@btime @tullio Y[m, co, b] := W[m, co, ci] * X[m, ci, b]

GC.gc()

println("# Y[m, co, b] = W[m, co, ci] * X[m, ci, b]")
W = reshape(W, (M, Ci, Co))
@btime @tullio Y[m, co, b] := W[m, ci, co] * X[m, ci, b]

end

"""
#==================#
### Tullio tests ###
# with x[M, C, B]
#==================#
# Y[m, co, b] = W[co, ci, m] * X[m, ci, b]
  276.022 ms (2 allocations: 25.00 MiB)
# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]
  165.038 ms (2 allocations: 25.00 MiB)
# Y[m, co, b] = W[m, ci, co] * X[m, ci, b]
  270.286 ms (2 allocations: 25.00 MiB)
# Y[m, co, b] = W[m, co, ci] * X[m, ci, b]
  320.016 ms (2 allocations: 25.00 MiB)
"""

if true
println("#==================#")
println("### Tullio Bilinear tests ###")
println("# with x/y[C, M, B]")
println("#==================#")

C1, C2, Co = 32, 32, 4
M = 1024
B = 100

X = rand(C1, M, B)
Y = rand(C2, M, B)
W = rand(Co, C1, C2, M)

GC.gc()
W = reshape(W, (Co, C1, C2, M))
@btime @tullio Z[co, m, b] := X[c1, m, b] * W[co, c1, c2, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (Co, C2, C1, M))
@btime @tullio Z[co, m, b] := X[c1, m, b] * W[co, c2, c1, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C1, Co, C2, M))
@btime @tullio Z[co, m, b] := X[c1, m, b] * W[c1, co, c2, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C2, Co, C1, M))
@btime @tullio Z[co, m, b] := X[c1, m, b] * W[c2, co, c1, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C1, C2, Co, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c1, c2, co, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C2, C1, Co, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c2, c1, co, m] * Y[c2, m, b]

end

"""
#==================#
### Tullio Bilinear tests ###
# with x/y[C, M, B]
#==================#
  487.817 ms (8 allocations: 3.13 MiB)
  529.306 ms (8 allocations: 3.13 MiB)
  137.640 ms (8 allocations: 3.13 MiB)
  528.762 ms (8 allocations: 3.13 MiB)
  130.426 ms (8 allocations: 3.13 MiB)
  506.867 ms (8 allocations: 3.13 MiB)
"""

nothing
#
