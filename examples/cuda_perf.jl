#
using BenchmarkTools, FFTW, CUDA

CUDA.allowscalar(false)

if false

C  = 16
Ns = 128, 128
K  = 100

x = CUDA.rand(Ns..., K, C)
y = CUDA.rand(Ns..., C, K)
z = CUDA.rand(C, Ns..., K)

dx = 1:2
dy = 1:2

GC.gc()

println("### rFFT/irFFT dims (Nx, Ny, K, C) ###") # winner
_x = rfft(x, dx)
@btime CUDA.@sync rfft($x, $dx)
@btime CUDA.@sync irfft($_x, $Ns[1], $dx)

GC.gc()

println("### rFFT/irFFT dims (Nx, Ny, C, K) ###")
_y = rfft(y, dy)
@btime CUDA.@sync rfft($y, $dy)
@btime CUDA.@sync irfft($_y, $Ns[1], $dy)

GC.gc()

println("### permute (C, Nx, Ny, K) <-> (Nx, Ny, K, C) ###") # winner
perm1 = (2, 3, 4, 1) # bwd
perm2 = (4, 1, 2, 3) # fwd
@btime CUDA.@sync permutedims($z, $perm1)
@btime CUDA.@sync permutedims($x, $perm2)

GC.gc()

println("### permute (C, Nx, Ny, K) <-> (Nx, Ny, C, K) ###")
perm1 = (2, 3, 1, 4) # bwd
perm2 = (3, 1, 2, 4) # fwd
@btime CUDA.@sync permutedims($z, $perm1)
@btime CUDA.@sync permutedims($y, $perm2)

# println("FFT on PermutedDimsArray")
# @btime CUDA.@sync rfft(PermutedDimsArray($y, $perm_y), $dx) # very bad

end

"""
julia> include("examples/cuda_perf.jl")                                                                                                                                                                 [4/1755]
### rFFT/irFFT dims (Nx, Ny, K, C) ###
  3.650 ms (76 allocations: 4.39 KiB)            
  5.937 ms (83 allocations: 4.66 KiB)            
### rFFT/irFFT dims (Nx, Ny, C, K) ###  
  3.601 ms (77 allocations: 4.66 KiB)               
  5.911 ms (83 allocations: 4.66 KiB)            
### permute (C, Nx, Ny, K) <-> (Nx, Ny, K, C) ###
  2.557 ms (70 allocations: 4.03 KiB)   
  9.399 ms (70 allocations: 4.03 KiB)   
### permute (C, Nx, Ny, K) <-> (Nx, Ny, C, K) ###
  2.276 ms (70 allocations: 4.03 KiB)               
  2.134 ms (70 allocations: 4.03 KiB)               
"""

nothing

using Tullio, CUDA, KernelAbstractions

if false
println("#==================#")
println("### Tullio tests ###")
println("# with x[C, M, B]")
println("#==================#")

Ci, Co = 32, 64
M = 1024
B = 100

X = CUDA.rand(Ci, M, B)
W = CUDA.rand(Co, Ci, M)

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
#==================#
# Y[co, m, b] = W[co, ci, m] * X[ci, m, b]
  110.170 μs (115 allocations: 4.94 KiB)
# Y[co, m, b] = W[ci, co, m] * X[ci, m, b]
  113.370 μs (116 allocations: 4.97 KiB)
# Y[co, m, b] = W[co, m, ci] * X[ci, m, b]
# Y[co, m, b] = W[ci, m, co] * X[ci, m, b]
  111.694 μs (115 allocations: 4.94 KiB)
# Y[co, m, b] = W[m, ci, co] * X[ci, m, b]
  111.174 μs (115 allocations: 4.94 KiB)
# Y[co, m, b] = W[m, co, ci] * X[ci, m, b]
  110.254 μs (115 allocations: 4.94 KiB)
"""

if true
println("#==================#")
println("### Tullio tests ###")
println("# with x[M, C, B]")
println("#==================#")

Ci, Co = 32, 64
M = 1024
B = 100

X = CUDA.rand(M, Ci, B)
W = CUDA.rand(Co, Ci, M)

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
  112.160 μs (117 allocations: 4.97 KiB)
# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]
  112.457 μs (117 allocations: 4.97 KiB)
# Y[m, co, b] = W[m, ci, co] * X[m, ci, b]
  112.313 μs (117 allocations: 4.97 KiB)
# Y[m, co, b] = W[m, co, ci] * X[m, ci, b]
  111.700 μs (117 allocations: 4.97 KiB)
"""

nothing
#
