#
using BenchmarkTools, FFTW, CUDA

CUDA.allowscalar(false)

if false

C  = 16
Ns = 128, 128
K  = 100

dims = 1:2

x = CUDA.rand(Ns..., K, C)
y = CUDA.rand(Ns..., C, K)
z = CUDA.rand(C, Ns..., K)

GC.gc()

println("### rFFT/irFFT dims (Nx, Ny, K, C) ###") # winner
_x = rfft(x, dims)
@btime CUDA.@sync rfft($x, $dims)
@btime CUDA.@sync irfft($_x, $Ns[1], $dims)

GC.gc()

println("### rFFT/irFFT dims (Nx, Ny, C, K) ###")
_y = rfft(y, dims)
@btime CUDA.@sync rfft($y, $dims)
@btime CUDA.@sync irfft($_y, $Ns[1], $dims)

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
# @btime CUDA.@sync rfft(PermutedDimsArray($y, $perm_y), $dims) # very bad

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

Ci, Co = 32, 32
M = 1024
B = 100

X = CUDA.rand(Ci, M, B)
W = CUDA.rand(Co, Ci, M)

GC.gc()

println("# Y[co, m, b] = W[co, ci, m] * X[ci, m, b]")
W = reshape(W, (Co, Ci, M))
@btime CUDA.@sync @tullio Y[co, m, b] := W[co, ci, m] * X[ci, m, b]

GC.gc()

println("# Y[co, m, b] = W[ci, co, m] * X[ci, m, b]")
W = reshape(W, (Ci, Co, M))
@btime CUDA.@sync @tullio Y[co, m, b] := W[ci, co, m] * X[ci, m, b] # winner

GC.gc()

println("# Y[co, m, b] = W[co, m, ci] * X[ci, m, b]")
# W = reshape(W, (Co, M, Ci))
# @btime CUDA.@sync @tullio Y[co, m, b] := W[co, m, ci] * X[ci, m, b] # worst - do not run

GC.gc()

println("# Y[co, m, b] = W[ci, m, co] * X[ci, m, b]")
W = reshape(W, (Ci, M, Co))
@btime CUDA.@sync @tullio Y[co, m, b] := W[ci, m, co] * X[ci, m, b]

GC.gc()

println("# Y[co, m, b] = W[m, ci, co] * X[ci, m, b]")
W = reshape(W, (M, Co, Ci))
@btime CUDA.@sync @tullio Y[co, m, b] := W[m, co, ci] * X[ci, m, b]

GC.gc()

println("# Y[co, m, b] = W[m, co, ci] * X[ci, m, b]")
W = reshape(W, (M, Ci, Co))
@btime CUDA.@sync @tullio Y[co, m, b] := W[m, ci, co] * X[ci, m, b]

end

"""
#==================#
### Tullio tests ###
# with x[C, M, B]   
#==================# 
# Y[co, m, b] = W[co, ci, m] * X[ci, m, b]
  3.828 ms (161 allocations: 7.89 KiB)
# Y[co, m, b] = W[ci, co, m] * X[ci, m, b] 
  17.044 ms (161 allocations: 7.89 KiB)   
# Y[co, m, b] = W[co, m, ci] * X[ci, m, b] 
# Y[co, m, b] = W[ci, m, co] * X[ci, m, b]
  42.634 ms (161 allocations: 7.89 KiB)   
# Y[co, m, b] = W[m, ci, co] * X[ci, m, b]
  6.181 ms (161 allocations: 7.89 KiB)    
# Y[co, m, b] = W[m, co, ci] * X[ci, m, b]
  7.353 ms (161 allocations: 7.89 KiB)   
"""

if false
println("#==================#")
println("### Tullio tests ###")
println("# with x[M, C, B]")
println("#==================#")

Ci, Co = 32, 32
M = 1024
B = 100

X = CUDA.rand(M, Ci, B)
W = CUDA.rand(Co, Ci, M)

GC.gc()

println("# Y[m, co, b] = W[co, ci, m] * X[m, ci, b]")
W = reshape(W, (Co, Ci, M))
@btime CUDA.@sync @tullio Y[m, co, b] := W[co, ci, m] * X[m, ci, b]

GC.gc()

println("# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]")
W = reshape(W, (Ci, Co, M))
@btime CUDA.@sync @tullio Y[m, co, b] := W[ci, co, m] * X[m, ci, b]

GC.gc()

println("# Y[m, co, b] = W[m, ci, co] * X[m, ci, b]")
W = reshape(W, (M, Co, Ci))
@btime CUDA.@sync @tullio Y[m, co, b] := W[m, co, ci] * X[m, ci, b]

GC.gc()

println("# Y[m, co, b] = W[m, co, ci] * X[m, ci, b]")
W = reshape(W, (M, Ci, Co))
@btime CUDA.@sync @tullio Y[m, co, b] := W[m, ci, co] * X[m, ci, b]

end

"""
#==================#
### Tullio tests ###
# with x[M, C, B]   
#==================#
# Y[m, co, b] = W[co, ci, m] * X[m, ci, b]
  23.956 ms (163 allocations: 7.92 KiB)
# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]
  30.556 ms (163 allocations: 7.92 KiB)
# Y[m, co, b] = W[m, ci, co] * X[m, ci, b]
  4.599 ms (163 allocations: 7.92 KiB)
# Y[m, co, b] = W[m, co, ci] * X[m, ci, b]
  4.486 ms (163 allocations: 7.92 KiB)
"""

if true
println("#==================#")
println("### Tullio Bilinear tests ###")
println("# with x/y[C, M, B]")
println("#==================#")

C1, C2, Co = 32, 32, 4
M = 1024
B = 100

X = CUDA.rand(C1, M, B)
Y = CUDA.rand(C2, M, B)
W = CUDA.rand(Co, C1, C2, M)

GC.gc()
W = reshape(W, (Co, C1, C2, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[co, c1, c2, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (Co, C2, C1, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[co, c2, c1, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C1, Co, C2, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c1, co, c2, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C2, Co, C1, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c2, co, c1, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C1, C2, Co, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c1, c2, co, m] * Y[c2, m, b]

GC.gc()
W = reshape(W, (C2, C1, Co, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c2, c1, co, m] * Y[c2, m, b]

# TODO benchmark bwd pass
# TODO benchmark on other GPUs

end

"""
##==================#
### Tullio Bilinear tests ###
# with x/y[C, M, B]
#==================#
  8.786 ms (168 allocations: 8.39 KiB)
  36.982 ms (170 allocations: 8.42 KiB)
  18.833 ms (168 allocations: 8.39 KiB)
  69.349 ms (170 allocations: 8.42 KiB)
  148.150 ms (170 allocations: 8.42 KiB)
  167.824 ms (170 allocations: 8.42 KiB)

"""

nothing
#
