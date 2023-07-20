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
y = CUDA.rand(Ns..., C, K) # winner

println("### rFFT/irFFT dims (Nx, Ny, K, C) ###")
_x = rfft(x, dims)
@btime CUDA.@sync rfft($x, $dims)
@btime CUDA.@sync irfft($_x, $Ns[1], $dims)

println("### rFFT/irFFT dims (Nx, Ny, C, K) ###")
_y = rfft(y, dims)
@btime CUDA.@sync rfft($y, $dims)
@btime CUDA.@sync irfft($_y, $Ns[1], $dims)

z = CUDA.rand(C, Ns..., K)
println("### permute (C, Nx, Ny, K) <-> (Nx, Ny, K, C) ###")
perm1 = (2, 3, 4, 1) # bwd
perm2 = (4, 1, 2, 3) # fwd
@btime CUDA.@sync permutedims($z, $perm1)
@btime CUDA.@sync permutedims($x, $perm2)

println("### permute (C, Nx, Ny, K) <-> (Nx, Ny, C, K) ###")
perm1 = (2, 3, 1, 4) # bwd
perm2 = (3, 1, 2, 4) # fwd
@btime CUDA.@sync permutedims($z, $perm1)
@btime CUDA.@sync permutedims($y, $perm2)

println()
println("### permute (C, Nx, Ny, K) <-> (Nx, Ny, C, K) ###")
f(X) = permutedims(X, perm1)
@btime CUDA.@sync $f($z)
@btime CUDA.@sync $grad($f, $z)

println()
f(X) = permutedims(X, perm2)
@btime CUDA.@sync $f($y)
@btime CUDA.@sync $grad($f, $y)

# println("FFT on PermutedDimsArray")
# @btime CUDA.@sync rfft(PermutedDimsArray($y, $perm_y), $dims) # very bad

end

"""
julia> include("examples/cuda_perf.jl")
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

### permute (C, Nx, Ny, K) <-> (Nx, Ny, C, K) ###
  2.021 ms (70 allocations: 4.03 KiB)
  5.334 ms (136 allocations: 7.64 KiB)

  1.859 ms (70 allocations: 4.03 KiB)
  5.378 ms (136 allocations: 7.64 KiB)
"""

nothing

Ci, Co = 32, 32
M = 1024
B = 100

if false
println("#==================#")
println("### Tullio Linear Test ###")
println("# with x[C, M, B]")
println("#==================#")

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
                                                    
# Y[co, m, b] = W[ci, co, m] * X[ci, m, b] # winner
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

if true
println("#==================#")
println("### Tullio Linear Test ###")
println("# with x[M, C, B]")
println("#==================#")

X = CUDA.rand(M, Ci, B)
W = CUDA.rand(Co, Ci, M)

println("# Y[m, co, b] = W[co, ci, m] * X[m, ci, b]")
f(W, X) = @tullio Y[m, co, b] := W[co, ci, m] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]")
W = reshape(W, (Ci, Co, M))
f(W, X) = @tullio Y[m, co, b] := W[ci, co, m] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[m, co, b] = W[m, ci, co] * X[m, ci, b]") # winner
W = reshape(W, (M, Co, Ci))
f(W, X) = @tullio Y[m, co, b] := W[m, co, ci] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[m, co, b] = W[m, co, ci] * X[m, ci, b]")
W = reshape(W, (M, Ci, Co))
f(W, X) = @tullio Y[m, co, b] := W[m, ci, co] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("### Tullio Linear Test with permuted return ###")
println()

println("# Y[co, m, b] = W[co, ci, m] * X[m, ci, b]")
W = reshape(W, (Co, Ci, M))
f(W, X) = @tullio Y[co, m, b] := W[co, ci, m] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[ci, co, m] * X[m, ci, b]")
W = reshape(W, (Ci, Co, M))
f(W, X) = @tullio Y[co, m, b] := W[ci, co, m] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[m, ci, co] * X[m, ci, b]")
W = reshape(W, (Co, M, Ci))
f(W, X) = @tullio Y[co, m, b] := W[co, m, ci] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

println()
println("# Y[co, m, b] = W[co, m, ci] * X[m, ci, b]")
W = reshape(W, (M, Ci, Co))
f(W, X) = @tullio Y[co, m, b] := W[m, ci, co] * X[m, ci, b]
@btime CUDA.@sync $f($W, $X)
@btime CUDA.@sync $grad($f, $W, $X)

end

"""
#==================#
### Tullio tests ###
# with x[M, C, B]
#==================#
# Y[m, co, b] = W[co, ci, m] * X[m, ci, b]
  23.629 ms (163 allocations: 7.91 KiB)
  132.129 ms (390 allocations: 19.20 KiB)
#
# Y[m, co, b] = W[ci, co, m] * X[m, ci, b]
  30.193 ms (163 allocations: 7.91 KiB)
  135.221 ms (390 allocations: 19.20 KiB)
#
# Y[m, co, b] = W[m, ci, co] * X[m, ci, b] # winner
  4.404 ms (163 allocations: 7.91 KiB)
  13.103 ms (390 allocations: 19.20 KiB)
#
# Y[m, co, b] = W[m, co, ci] * X[m, ci, b]
  4.367 ms (163 allocations: 7.91 KiB)
  13.194 ms (390 allocations: 19.20 KiB)

### Tullio Linear Test with permuted return ###

# Y[co, m, b] = W[co, ci, m] * X[m, ci, b]
  3.521 ms (161 allocations: 7.88 KiB)
  124.453 ms (388 allocations: 19.17 KiB)

# Y[co, m, b] = W[ci, co, m] * X[m, ci, b]
  15.001 ms (161 allocations: 7.88 KiB)
  152.818 ms (388 allocations: 19.17 KiB)

# Y[co, m, b] = W[m, ci, co] * X[m, ci, b]
  3.585 ms (161 allocations: 7.88 KiB)
  70.606 ms (388 allocations: 19.17 KiB)

# Y[co, m, b] = W[co, m, ci] * X[m, ci, b]
  8.453 ms (161 allocations: 7.88 KiB)
  19.000 ms (388 allocations: 19.17 KiB)
"""
#

C1, C2, Co = 32, 32, 4
M = 1024
B = 100

if false
println("#==================#")
println("### Tullio Bilinear tests (single call) ###")
println("# with x/y[C, M, B]")
println("#==================#")

X = CUDA.rand(C1, M, B)
Y = CUDA.rand(C2, M, B)
W = CUDA.rand(Co, C1, C2, M)

println("### M at end ###")

#######################
println("###")
W = reshape(W, (Co, C1, C2, M))
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[co, c1, c2, m] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
W = reshape(W, (Co, C2, C1, M))
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[co, c2, c1, m] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#######################
println("###")
W = reshape(W, (C1, Co, C2, M))
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[c1, co, c2, m] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
W = reshape(W, (C2, Co, C1, M))
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[c2, co, c1, m] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#######################
println("###")
W = reshape(W, (C1, C2, Co, M))
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[c1, c2, co, m] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
W = reshape(W, (C2, C1, Co, M))
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[c2, c1, co, m] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#######################

println("### M in front ###")

#####################
W = reshape(W, (M, C2, C1, Co))

println("###")
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[m, c2, c1, co] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[m, c1, c2, co] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#####################
W = reshape(W, (M, C2, Co, C1))

println("###")
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[m, c1, co, c2] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[m, c2, co, c1] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#####################
W = reshape(W, (M, Co, C1, C2))

println("###")
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[m, co, c2, c1] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
f(W, X, Y) = @tullio Z[co, m, b] := X[c1, m, b] * W[m, co, c1, c2] * Y[c2, m, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#####################
@show (CUDA.@allocated grad(f, W, X, Y)) / 1024 / 1024 # 44.125 mb
@show (CUDA.@allocated f(W, X, Y)) / 1024 / 1024       # 1.5625 mb

end

"""
#==================#
### Tullio Bilinear tests ###
# with x/y[C, M, B]
#==================#
### M at end ###
  9.777 ms (168 allocations: 8.38 KiB)
  1.181 s (425 allocations: 20.92 KiB)

  35.632 ms (170 allocations: 8.41 KiB)
  1.203 s (427 allocations: 20.95 KiB)

  19.013 ms (168 allocations: 8.38 KiB)
  1.325 s (425 allocations: 20.92 KiB)

  68.923 ms (170 allocations: 8.41 KiB)
  1.401 s (427 allocations: 20.95 KiB)

  147.998 ms (170 allocations: 8.41 KiB)
  1.464 s (427 allocations: 20.95 KiB)

  166.740 ms (171 allocations: 8.44 KiB)
  1.487 s (428 allocations: 20.98 KiB)

julia> (CUDA.@allocated f(W, X, Y)) / 1024 / 1024
1.5625

julia> (CUDA.@allocated grad(f, W, X, Y)) / 1024 / 1024
44.125
"""

if false
println("#==================#")
println("### Tullio Bilinear tests (single call) ###")
println("# with x/y[M, C, B]")
println("#==================#")

X = CUDA.rand(M, C1, B)
Y = CUDA.rand(M, C2, B)
W = CUDA.rand(Co, C1, C2, M)

println("### M at end ###")

#######################
W = reshape(W, (Co, C1, C2, M))

println("###")
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[co, c1, c2, m] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
W = reshape(W, (Co, C2, C1, M))
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[co, c2, c1, m] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#######################
W = reshape(W, (C1, Co, C2, M))

println("###")
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[c1, co, c2, m] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
W = reshape(W, (C2, Co, C1, M))
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[c2, co, c1, m] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#######################
W = reshape(W, (C1, C2, Co, M))

println("###")
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[c1, c2, co, m] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
W = reshape(W, (C2, C1, Co, M))
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[c2, c1, co, m] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#######################

println("### M in front ###")

#####################
W = reshape(W, (M, C2, C1, Co))

println("###")
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[m, c2, c1, co] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[m, c1, c2, co] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#####################
W = reshape(W, (M, C2, Co, C1))

println("###")
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[m, c1, co, c2] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[m, c2, co, c1] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#####################
W = reshape(W, (M, Co, C1, C2))

println("###")
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[m, co, c2, c1] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

println()
f(W, X, Y) = @tullio Z[m, co, b] := X[m, c1, b] * W[m, co, c1, c2] * Y[m, c2, b]
@btime CUDA.@sync $f($W, $X, $Y)
@btime CUDA.@sync $grad($f, $W, $X, $Y)

#####################
@show (CUDA.@allocated grad(f, W, X, Y)) / 1024 / 1024 # 44.125 mb
@show (CUDA.@allocated f(W, X, Y)) / 1024 / 1024       # 1.5625 mb

end

"""
#==================#                                
### Tullio Bilinear tests (single call) ###
# with x/y[M, C, B]                                 
#==================#                     
### M at end ###                                    
###                                                 
  8.068 ms (168 allocations: 8.38 KiB)              
  510.736 ms (423 allocations: 20.88 KiB)           
                                                    
  11.412 ms (168 allocations: 8.38 KiB)  
  513.668 ms (423 allocations: 20.88 KiB)           
###                                                 
  30.635 ms (168 allocations: 8.38 KiB)                                                                                                                                                                         
  536.812 ms (423 allocations: 20.88 KiB)                                                                                                                                                                       
                                                    
  50.129 ms (168 allocations: 8.38 KiB)             
  557.445 ms (423 allocations: 20.88 KiB)
###                                                 
  15.816 ms (168 allocations: 8.38 KiB)  
  525.191 ms (423 allocations: 20.88 KiB)
                                                    
  184.636 ms (168 allocations: 8.38 KiB) 
  703.622 ms (423 allocations: 20.88 KiB)                                                                                                                                                                       
### M in front ###                                                                                                                                                                                              
###                                   
  8.002 ms (168 allocations: 8.38 KiB)   
  336.836 ms (423 allocations: 20.88 KiB)
                                                    
  8.010 ms (168 allocations: 8.38 KiB)   
  336.548 ms (423 allocations: 20.88 KiB)
###                                   
  9.921 ms (168 allocations: 8.38 KiB)   
  339.172 ms (423 allocations: 20.88 KiB)                                                                                                                                                                       
                                                                                                                                                                                                                
  7.786 ms (168 allocations: 8.38 KiB)
  337.194 ms (423 allocations: 20.88 KiB)
###
  9.695 ms (168 allocations: 8.38 KiB)
  338.685 ms (423 allocations: 20.88 KiB)

  9.635 ms (168 allocations: 8.38 KiB)
  338.507 ms (423 allocations: 20.88 KiB)
(#= /home/vedantpu/.julia/dev/GeometryLearning.jl/examples/cuda_perf.jl:579 =# CUDA.@allocated(grad(f, W, X, Y)) / 1024) / 1024 = 44.125
(#= /home/vedantpu/.julia/dev/GeometryLearning.jl/examples/cuda_perf.jl:580 =# CUDA.@allocated(f(W, X, Y)) / 1024) / 1024 = 1.5625
"""

if false

println("#==================#")
println("### Tullio Bilinear tests (2 calls) ###")
println("# with x/y[C, M, B]")
println("#==================#")
    
# CUDA.@captured
# https://juliagpu.org/post/2021-06-10-cuda_3.3/#high-level_graph_apis
# https://github.com/JuliaGPU/CUDA.jl/blob/a8c55aed276892aeb7bbe5220448a5ca5922a9be/test/core/cudadrv.jl#L380-L395

X = CUDA.rand(C1, M, B)
Y = CUDA.rand(C2, M, B)
W = CUDA.rand(Co, C1, C2, M)

println("### M at end")

#############
println("###")
W = reshape(W, (Co, C1, C2, M))

function f(X, Y, W)
    @tullio Z1[co, c1, m, b] := W[co, c1, c2, m] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[co, c1, m, b] * X[c1, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W) # winner
    @tullio Z1[co, c2, m, b] := W[co, c1, c2, m] * X[c1, m, b]
    @tullio Z2[co, m, b]     := Z1[co, c2, m, b] * Y[c2, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############
println("###")
W = reshape(W, (C1, C2, Co, M))

function f(X, Y, W)
    @tullio Z1[c2, co, m, b] := W[c1, c2, co, m] * X[c1, m, b]
    @tullio Z2[co, m, b]     := Z1[c2, co, m, b] * Y[c2, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[c1, co, m, b] := W[c1, c2, co, m] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[c1, co, m, b] * X[c1, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############
println("###")
W = reshape(W, (C1, Co, C2, M))

function f(X, Y, W)
    @tullio Z1[c2, co, m, b] := W[c1, co, c2, m] * X[c1, m, b]
    @tullio Z2[co, m, b]     := Z1[c2, co, m, b] * Y[c2, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[c1, co, m, b] := W[c1, co, c2, m] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[c1, co, m, b] * X[c1, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############

println("###")
println("### M at front")

#############
println("###")
W = reshape(W, (M, Co, C1, C2))

function f(X, Y, W)
    @tullio Z1[co, c1, m, b] := W[m, co, c1, c2] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[co, c1, m, b] * X[c1, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[co, c2, m, b] := W[m, co, c1, c2] * X[c1, m, b]
    @tullio Z2[co, m, b]     := Z1[co, c2, m, b] * Y[c2, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############
println("###")
W = reshape(W, (M, C1, C2, Co))

function f(X, Y, W)
    @tullio Z1[c2, co, m, b] := W[m, c1, c2, co] * X[c1, m, b]
    @tullio Z2[co, m, b]     := Z1[c2, co, m, b] * Y[c2, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[c1, co, m, b] := W[m, c1, c2, co] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[c1, co, m, b] * X[c1, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############
println("###")
W = reshape(W, (M, C1, Co, C2))

function f(X, Y, W)
    @tullio Z1[c2, co, m, b] := W[m, c1, co, c2] * X[c1, m, b]
    @tullio Z2[co, m, b]     := Z1[c2, co, m, b] * Y[c2, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[c1, co, m, b] := W[m, c1, co, c2] * Y[c2, m, b]
    @tullio Z2[co, m, b]     := Z1[c1, co, m, b] * X[c1, m, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############

println("###")
(CUDA.@allocated f(X, Y, W)) / 1024 / 1024       # 51.5625 mb
(CUDA.@allocated grad(f, X, Y, W)) / 1024 / 1024 # 144.125 mb
#############
end

"""
#==================#                                
### Tullio Bilinear tests (2 calls) ###  
# with x/y[C, M, B]                                 
#==================#                    
### M at end                             
###                                      
  16.838 ms (287 allocations: 13.91 KiB)            
  340.882 ms (750 allocations: 37.44 KiB)
                                                    
  24.308 ms (287 allocations: 13.91 KiB)  # winner
  55.796 ms (750 allocations: 37.44 KiB)            
###                                      
  46.531 ms (287 allocations: 13.91 KiB)            
  83.318 ms (750 allocations: 37.44 KiB)
                                                    
  17.778 ms (287 allocations: 13.91 KiB) 
  211.185 ms (750 allocations: 37.44 KiB)
###                                     
  62.744 ms (287 allocations: 13.91 KiB) 
  100.269 ms (750 allocations: 37.44 KiB)
                                                    
  17.930 ms (287 allocations: 13.91 KiB) 
  355.943 ms (750 allocations: 37.44 KiB)
###                                     
### M at front                           
###                                      
  31.070 ms (287 allocations: 13.91 KiB)
  151.018 ms (750 allocations: 37.44 KiB)
                                                    
  43.090 ms (287 allocations: 13.91 KiB)
  142.949 ms (750 allocations: 37.44 KiB)
###                                      
  44.428 ms (287 allocations: 13.91 KiB)
  144.320 ms (750 allocations: 37.44 KiB)
                                                    
  32.818 ms (287 allocations: 13.91 KiB)
  150.499 ms (750 allocations: 37.44 KiB)
###    
  44.057 ms (287 allocations: 13.91 KiB)
  145.607 ms (750 allocations: 37.44 KiB)

  32.206 ms (287 allocations: 13.91 KiB)
  150.616 ms (750 allocations: 37.44 KiB)
###
"""

if true

println("#==================#")
println("### Tullio Bilinear tests (2 calls) ###")
println("# with x/y[M, C, B]")
println("#==================#")
    
X = CUDA.rand(M, C1, B)
Y = CUDA.rand(M, C2, B)
W = CUDA.rand(Co, C1, C2, M)

# println("### M at end")
#
# #############
# println("###")
# W = reshape(W, (Co, C1, C2, M))
#
# function f(X, Y, W)
#     @tullio Z1[m, co, c1, b] := W[co, c1, c2, m] * Y[m, c2, b]
#     @tullio Z2[m, co, b]     := Z1[m, co, c1, b] * X[m, c1, b]
# end
# @btime CUDA.@sync $f($X, $Y, $W)
# @btime CUDA.@sync $grad($f, $X, $Y, $W)
#
# println()
# function f(X, Y, W)
#     @tullio Z1[m, co, c2, b] := W[co, c1, c2, m] * X[m, c1, b]
#     @tullio Z2[m, co, b]     := Z1[m, co, c2, b] * Y[m, c2, b]
# end
# @btime CUDA.@sync $f($X, $Y, $W)
# @btime CUDA.@sync $grad($f, $X, $Y, $W)
#
# #############
# println("###")
# W = reshape(W, (C1, C2, Co, M))
#
# function f(X, Y, W)
#     @tullio Z1[c2, co, m, b] := W[c1, c2, co, m] * X[c1, m, b]
#     @tullio Z2[co, m, b]     := Z1[c2, co, m, b] * Y[c2, m, b]
# end
# @btime CUDA.@sync $f($X, $Y, $W)
# @btime CUDA.@sync $grad($f, $X, $Y, $W)
#
# println()
# function f(X, Y, W)
#     @tullio Z1[c1, co, m, b] := W[c1, c2, co, m] * Y[c2, m, b]
#     @tullio Z2[co, m, b]     := Z1[c1, co, m, b] * X[c1, m, b]
# end
# @btime CUDA.@sync $f($X, $Y, $W)
# @btime CUDA.@sync $grad($f, $X, $Y, $W)

#############
# println("###")
# W = reshape(W, (C1, Co, C2, M))
#
# function f(X, Y, W)
#     @tullio Z1[c2, co, m, b] := W[c1, co, c2, m] * X[c1, m, b]
#     @tullio Z2[co, m, b]     := Z1[c2, co, m, b] * Y[c2, m, b]
# end
# @btime CUDA.@sync $f($X, $Y, $W)
# @btime CUDA.@sync $grad($f, $X, $Y, $W)
#
# println()
# function f(X, Y, W)
#     @tullio Z1[c1, co, m, b] := W[c1, co, c2, m] * Y[c2, m, b]
#     @tullio Z2[co, m, b]     := Z1[c1, co, m, b] * X[c1, m, b]
# end
# @btime CUDA.@sync $f($X, $Y, $W)
# @btime CUDA.@sync $grad($f, $X, $Y, $W)

#############

println("###")
println("### M at front")

#############
println("###")
W = reshape(W, (M, Co, C1, C2))

function f(X, Y, W)
    @tullio Z1[m, co, c1, b] := W[m, co, c1, c2] * Y[m, c2, b]
    @tullio Z2[m, co, b]     := Z1[m, co, c1, b] * X[m, c1, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[m, co, c2, b] := W[m, co, c1, c2] * X[m, c1, b]
    @tullio Z2[m, co, b]     := Z1[m, co, c2, b] * Y[m, c2, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############
println("###")
W = reshape(W, (M, C1, C2, Co))

function f(X, Y, W)
    @tullio Z1[m, c2, co, b] := W[m, c1, c2, co] * X[m, c1, b]
    @tullio Z2[m, co, b]     := Z1[m, c2, co, b] * Y[m, c2, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[m, c1, co, b] := W[m, c1, c2, co] * Y[m, c2, b]
    @tullio Z2[m, co, b]     := Z1[m, c1, co, b] * X[m, c1, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############
println("###")
W = reshape(W, (M, C1, Co, C2))

function f(X, Y, W)
    @tullio Z1[m, c2, co, b] := W[m, c1, co, c2] * X[m, c1, b]
    @tullio Z2[m, co, b]     := Z1[m, c2, co, b] * Y[m, c2, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

println()
function f(X, Y, W)
    @tullio Z1[m, c1, co, b] := W[m, c1, co, c2] * Y[m, c2, b]
    @tullio Z2[m, co, b]     := Z1[m, c1, co, b] * X[m, c1, b]
end
@btime CUDA.@sync $f($X, $Y, $W)
@btime CUDA.@sync $grad($f, $X, $Y, $W)

#############

println("###")
@show (CUDA.@allocated f(X, Y, W)) / 1024 / 1024     
@show (CUDA.@allocated grad(f, X, Y, W)) / 1024 / 1024 
#############
end

"""
julia> include("examples/cuda_perf.jl")
#==================#                                
### Tullio Bilinear tests (2 calls) ###
# with x/y[M, C, B]                                 
#==================#                                
###                                                 
### M at front                                      
###                                                 
  19.213 ms (294 allocations: 14.02 KiB)
  56.433 ms (761 allocations: 37.61 KiB)

  18.873 ms (294 allocations: 14.02 KiB)
  56.466 ms (761 allocations: 37.61 KiB)
###                                                 
  19.101 ms (294 allocations: 14.02 KiB)
  56.750 ms (762 allocations: 37.78 KiB)

  19.569 ms (294 allocations: 14.02 KiB)
  56.835 ms (761 allocations: 37.61 KiB)
###                                                 
  19.278 ms (294 allocations: 14.02 KiB)
  56.990 ms (761 allocations: 37.61 KiB)

  19.262 ms (294 allocations: 14.02 KiB)
  56.374 ms (761 allocations: 37.61 KiB)
###                                                 
(CUDA.@allocated(f(X, Y, W)) / 1024) / 1024 = 51.5625 # mb
(CUDA.@allocated(grad(f, X, Y, W)) / 1024) / 1024 = 144.125 # mb


"""
#

nothing
#
