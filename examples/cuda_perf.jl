#
using BenchmarkTools, CUDA
using FFTW
using Tullio, KernelAbstractions
using Zygote

CUDA.allowscalar(false)

function grad(f, args...)
    l, pb = Zygote.pullback(f, args...)
    pb(one.(l))
end

if false

C  = 16
Ns = 128, 128
K  = 100

dims = 1:2

x = CUDA.rand(Ns..., K, C)
y = CUDA.rand(Ns..., C, K)
z = CUDA.rand(C, Ns..., K)

println("### rFFT/irFFT dims (Nx, Ny, K, C) ###") # winner
_x = rfft(x, dims)
@btime CUDA.@sync rfft($x, $dims)
@btime CUDA.@sync irfft($_x, $Ns[1], $dims)

println("### rFFT/irFFT dims (Nx, Ny, C, K) ###")
_y = rfft(y, dims)
@btime CUDA.@sync rfft($y, $dims)
@btime CUDA.@sync irfft($_y, $Ns[1], $dims)

println("### permute (C, Nx, Ny, K) <-> (Nx, Ny, K, C) ###") # winner
perm1 = (2, 3, 4, 1) # bwd
perm2 = (4, 1, 2, 3) # fwd
@btime CUDA.@sync permutedims($z, $perm1)
@btime CUDA.@sync permutedims($x, $perm2)

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

if true
println("#==================#")
println("### Tullio tests ###")
println("# with x[C, M, B]")
println("#==================#")

Ci, Co = 32, 32
M = 1024
B = 100

X = CUDA.rand(Ci, M, B)
W = CUDA.rand(Co, Ci, M)

println("# Y[co, m, b] = W[co, ci, m] * X[ci, m, b]")
W = reshape(W, (Co, Ci, M))
f(W, X) = @tullio Y[co, m, b] := W[co, ci, m] * X[ci, m, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[ci, co, m] * X[ci, m, b]")
W = reshape(W, (Ci, Co, M))
f(W, X) = @tullio Y[co, m, b] := W[ci, co, m] * X[ci, m, b] # winner
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[co, m, ci] * X[ci, m, b]")
W = reshape(W, (Co, M, Ci))
f(W, X) = @tullio Y[co, m, b] := W[co, m, ci] * X[ci, m, b] # worst - do not run
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[ci, m, co] * X[ci, m, b]")
W = reshape(W, (Ci, M, Co))
f(W, X) = @tullio Y[co, m, b] := W[ci, m, co] * X[ci, m, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[m, ci, co] * X[ci, m, b]")
W = reshape(W, (M, Co, Ci))
f(W, X) = @tullio Y[co, m, b] := W[m, co, ci] * X[ci, m, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[m, co, ci] * X[ci, m, b]")
W = reshape(W, (M, Ci, Co))
f(W, X) = @tullio Y[co, m, b] := W[m, ci, co] * X[ci, m, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

end

"""
#==================#                                
### Tullio tests ###                                
# with x[C, M, B]                                   
#==================#                                
# Y[co, m, b] = W[co, ci, m] * X[ci, m, b]
  3.654 ms (161 allocations: 7.88 KiB)    
  75.145 ms (386 allocations: 19.14 KiB)            
                                                    
# Y[co, m, b] = W[ci, co, m] * X[ci, m, b]
  17.150 ms (161 allocations: 7.88 KiB) 
  24.229 ms (386 allocations: 19.14 KiB)
                                                                                                        
# Y[co, m, b] = W[co, m, ci] * X[ci, m, b]
  3.661 ms (161 allocations: 7.88 KiB)  
  156.142 ms (386 allocations: 19.14 KiB)                                                               
                                                    
# Y[co, m, b] = W[ci, m, co] * X[ci, m, b]
  43.198 ms (161 allocations: 7.88 KiB) 
  50.246 ms (386 allocations: 19.14 KiB)
                          
# Y[co, m, b] = W[m, ci, co] * X[ci, m, b]                                                              
  5.999 ms (161 allocations: 7.88 KiB)                                                                  
  37.816 ms (386 allocations: 19.14 KiB)
                                                    
# Y[co, m, b] = W[m, co, ci] * X[ci, m, b]                                                              
  7.591 ms (161 allocations: 7.88 KiB)             
  37.645 ms (386 allocations: 19.14 KiB)
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

println("# Y[m, co, b] = W[co, ci, m] * X[m, ci, b]")
@btime CUDA.@sync @tullio Y[m, co, b] := W[co, ci, m] * X[m, ci, b]

println("# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]")
W = reshape(W, (Ci, Co, M))
@btime CUDA.@sync @tullio Y[m, co, b] := W[ci, co, m] * X[m, ci, b]

println("# Y[m, co, b] = W[m, ci, co] * X[m, ci, b]")
W = reshape(W, (M, Co, Ci))
@btime CUDA.@sync @tullio Y[m, co, b] := W[m, co, ci] * X[m, ci, b]

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

if false
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

W = reshape(W, (Co, C1, C2, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[co, c1, c2, m] * Y[c2, m, b]

W = reshape(W, (Co, C2, C1, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[co, c2, c1, m] * Y[c2, m, b]

W = reshape(W, (C1, Co, C2, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c1, co, c2, m] * Y[c2, m, b]

W = reshape(W, (C2, Co, C1, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c2, co, c1, m] * Y[c2, m, b]

W = reshape(W, (C1, C2, Co, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c1, c2, co, m] * Y[c2, m, b]

W = reshape(W, (C2, C1, Co, M))
@btime CUDA.@sync @tullio Z[co, m, b] := X[c1, m, b] * W[c2, c1, co, m] * Y[c2, m, b]

W = reshape(W, (Co, C1, C2, M))
f = function(X, Y, W)
    @tullio Z1[co, c1, m, b] := W[co, c1, c2, m] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[co, c1, m, b] * X[c1, m, b]
    sum(Z2)
end
@btime CUDA.@sync $f($X, $Y, $W)

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

###
# Gradient computation
###

using Zygote

if false
    
# CUDA.@captured
# https://juliagpu.org/post/2021-06-10-cuda_3.3/#high-level_graph_apis
# https://github.com/JuliaGPU/CUDA.jl/blob/a8c55aed276892aeb7bbe5220448a5ca5922a9be/test/core/cudadrv.jl#L380-L395

C1, C2, Co = 32, 32, 4
M = 1024
B = 100

X = CUDA.rand(C1, M, B)
Y = CUDA.rand(C2, M, B)
W = CUDA.rand(Co, C1, C2, M)

#############
f = function(X, Y, W)
    @tullio Z1[co, c1, m, b] := W[co, c1, c2, m] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[co, c1, m, b] * X[c1, m, b]
    sum(Z2)
end

@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)
CUDA.@time f(X, Y, W)
CUDA.@time grad(f, X, Y, W)
println()

#############
f = function(X, Y, W)
    @tullio Z[co, m, b] := X[c1, m, b] * W[co, c1, c2, m] * Y[c2, m, b]
    sum(Z)
end

@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)
CUDA.@time f(X, Y, W)
CUDA.@time grad(f, X, Y, W)
println()

#############

W = reshape(W, (C1, C2, Co, M))
f = function(X, Y, W)
    @tullio Z1[c2, co, m, b] := W[c1, c2, co, m] * X[c1, m, b]
    @tullio Z2[co, m, b]     := Z1[c2, co, m, b] * Y[c2, m, b]
    sum(Z2)
end

@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)
CUDA.@time f(X, Y, W)
CUDA.@time grad(f, X, Y, W)
println()

#############
end

"""
  17.990 ms (338 allocations: 16.47 KiB)
  356.575 ms (851 allocations: 43.05 KiB)
  0.018423 seconds (343 CPU allocations: 16.750 KiB) (4 GPU allocations: 51.563 MiB, 0.18% memmgmt time)
  0.404347 seconds (22.02 k CPU allocations: 1.325 MiB) (9 GPU allocations: 144.125 MiB, 0.02% memmgmt time)

  14.002 ms (219 allocations: 10.94 KiB)
  1.177 s (526 allocations: 26.53 KiB)
  0.018853 seconds (223 CPU allocations: 11.156 KiB) (3 GPU allocations: 1.563 MiB, 0.18% memmgmt time)
  1.200023 seconds (617 CPU allocations: 31.109 KiB) (7 GPU allocations: 44.125 MiB, 0.00% memmgmt time)
"""

nothing
#
