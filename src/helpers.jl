
#===========================================================#

function normalizedata(
    u::AbstractArray,
    μ::Union{Number, AbstractArray},
    σ::Union{Number, AbstractArray},
)
    (u .- μ) ./ σ
end

function unnormalizedata(
    u::AbstractArray,
    μ::Union{Number, AbstractArray},
    σ::Union{Number, AbstractArray},
)
    (u .* σ) .+ μ
end

#======================================================#

function loaddata(datafile::String; verbose::Bool = true)

    data = jldopen(datafile)
    x = data["x"]
    u = data["u"] # [Nx, Nb, Nt] or [out_dim, Nx, Nb, Nt]
    t = data["t"]
    mu = data["mu"]
    md_data = data["metadata"]

    close(data)

    @assert ndims(u) ∈ (3,4,)
    @assert x isa AbstractVecOrMat
    x = x isa AbstractVector ? reshape(x, 1, :) : x # (Dim, Npoints)

    if ndims(u) == 3 # [Nx, Nb, Nt]
        u = reshape(u, 1, size(u)...) # [1, Nx, Nb, Nt]
    end

    in_dim  = size(x, 1)
    out_dim = size(u, 1)

    if verbose
        println("input size $in_dim with $(size(x, 2)) points per trajectory.")
        println("output size $out_dim.")
    end

    @assert eltype(x) ∈ (Float32, Float64)
    @assert eltype(u) ∈ (Float32, Float64)

    mu = isnothing(mu) ? fill(nothing, size(u, 3)) |> Tuple : mu
    mu = isa(mu, AbstractArray) ? vec(mu) : mu

    if isa(mu[1], Number)
        mu = map(x -> [x], mu)
    end

    x, t, mu, u, md_data
end

function loadmodel(
    modelfile::String,
)
    model = jldopen(modelfile)
    NN, p, st = model["model"]
    metadata = model["metadata"]
    close(model)
    
    (NN, p, st), metadata
end

#======================================================#
function eval_model(
    model::NeuralROMs.AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractMatrix,
    ax; # (ComponentArray.Axis,)
    batchsize = 1,
    device = Lux.gpu_device(),
)
    us = []

    x = x |> device
    p = p |> device
    model = model |> device

    for i in axes(p, 2)
        q = ComponentArray(p[:, i], ax)
        u = model(x, q) |> Lux.cpu_device()
        # u = eval_model(model, x, q; batchsize, device)

        push!(us, u)
    end

    cat(us...; dims = 3)
end

function eval_model(
    model::NeuralROMs.AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractVector;
    batchsize = numobs(x) ÷ 100,
    device = Lux.gpu_device(),
)
    loader = MLUtils.DataLoader(x; batchsize, shuffle = false, partial = true)

    p = p |> device
    model = model |> device

    if device isa AbstractGPUDevice
        loader = DeviceIterator(device, loader)
    end

    y = ()
    for batch in loader
        yy = model(batch, p) |> Lux.cpu_device()
        y = (y..., yy)
    end

    hcat(y...)
end

function eval_model(
    model::NTuple{3, Any},
    x;
    batchsize = numobs(x) ÷ 100,
    device = Lux.gpu_device(),
)
    NN, p, st = model
    loader = MLUtils.DataLoader(x; batchsize, shuffle = false, partial = true)

    p, st = (p, st) |> device
	# st = Lux.testmode(st) # scalar indexing error with freeze_decoder

    if device isa AbstractGPUDevice
        loader = DeviceIterator(device, loader)
    end

    y = ()
    for batch in loader
        yy = NN(batch, p, st)[1] |> Lux.cpu_device()
        y = (y..., yy)
    end

    hcat(y...)
end

#======================================================#
"""
Input size `[out_dim, ...]`
"""
function normalize_u(
    u::AbstractArray{T,N},
    lims = nothing
) where{T,N}

    if isnothing(lims)
        dims = 2:N
        d  = prod(size(u)[dims])
        ū  = sum(u; dims) / d |> vec
        σu = sum(abs2, u .- ū; dims) / d .|> sqrt |> vec
    else
        @assert length(lims) == 2
        II = collect(Colon() for _ in 2:N)

        e  = collect(extrema(u[i, II...]) for i in axes(u, 1))
        mi = first.(e)
        ma = last.(e)

        σu = (ma - mi) ./ T(lims[2] - lims[1])
        ū  = T(0.5) * ((mi + ma) - σu * T(lims[1] + lims[2]))
    end

    u = normalizedata(u, ū, σu) # (u - ū) / σu

    u, ū, σu
end

const normalize_x = normalize_u

"""
t ∈ [0, T]
Input size `[Ntime]`.
"""
function normalize_t(
    t::AbstractVector{T},
    lims = nothing
) where{T}
    t, t̄, σt = normalize_u(t', lims)
    vec(t), t̄, σt
end
#======================================================#

