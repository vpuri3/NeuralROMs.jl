#
using NeuralROMs
using JLD2, TSne
using Random, Lux, NNlib, MLUtils
using Plots, ColorSchemes, LaTeXStrings

#======================================================#

function get_prob_grid(prob::NeuralROMs.AbstractPDEProblem)
    if prob isa Advection1D
        (128,)
    elseif prob isa BurgersViscous1D
        (1024,)
        # (2048,)
    elseif prob isa KuramotoSivashinsky1D
        (256,)
    elseif prob isa Advection2D
        (128,128,)
    elseif prob isa BurgersViscous2D
        (512,512,)
    else
        throw(ErrorException("Unsupported problem type."))
    end
end

#======================================================#

function cae_network(
    prob::NeuralROMs.AbstractPDEProblem,
    l::Integer,
    w::Integer,
    act,
)

    if prob isa Advection1D # 128 -> l -> 128
        encoder = Chain(
            Conv((8,), 1 => w, act; stride = 4, pad = 2), # /4
            Conv((8,), w => w, act; stride = 4, pad = 2), # /4
            Conv((8,), w => w, act; stride = 4, pad = 2), # /4
            Conv((2,), w => w, act; stride = 1, pad = 0), # /2
            flatten,
            Dense(w, l),
        )

        decoder = Chain(
            Dense(l, w, act),
            ReshapeLayer((1, w)),
            ConvTranspose((4,), w => w, act; stride = 1, pad = 0), # *4
            ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4
            ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4
            ConvTranspose((8,), w => 1     ; stride = 2, pad = 3), # *2
        )

        Chain(; encoder, decoder)

    elseif prob isa BurgersViscous1D # 1024 -> l -> 1024

        encoder = Chain(
            Conv((8,), 1 => w, act; stride = 4, pad = 2), # /4 = 256
            Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 64
            Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 16
            Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 4
            Conv((4,), w => w, act; stride = 1, pad = 0), # /4 = 1
            flatten,
            Dense(w, l),
        )
        
        decoder = Chain(
            Dense(l, w, act),
            ReshapeLayer((1, w)),
            ConvTranspose((4,), w => w, act; stride = 1, pad = 0), # *4 = 4
            ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 16
            ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 64
            ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 256
            ConvTranspose((8,), w => 1     ; stride = 4, pad = 2), # *4 = 1024
        )
        
        Chain(; encoder, decoder)

        # encoder = Chain( # 2048
        #     Conv((8,), 1 => w, act; stride = 4, pad = 2), # /4 = 512
        #     Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 128
        #     Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 32
        #     Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 8
        #     Conv((8,), w => w, act; stride = 1, pad = 0), # /4 = 1
        #     flatten,
        #     Dense(w, l),
        # )
        #
        # decoder = Chain(
        #     Dense(l, w, act),
        #     ReshapeLayer((1, w)),
        #     ConvTranspose((8,), w => w, act; stride = 1, pad = 0), # *8 = 8
        #     ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 16
        #     ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 64
        #     ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 256
        #     ConvTranspose((8,), w => 1     ; stride = 4, pad = 2), # *4 = 1024
        # )
        #
        # Chain(; encoder, decoder)

    elseif prob isa KuramotoSivashinsky1D # 256 -> l -> 256

        encoder = Chain(
            Conv((8,), 1 => w, act; stride = 4, pad = 2), # /4 = 256
            Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 16
            Conv((8,), w => w, act; stride = 4, pad = 2), # /4 = 4
            Conv((4,), w => w, act; stride = 1, pad = 0), # /4 = 1
            flatten,
            Dense(w, l),
        )

        decoder = Chain(
            Dense(l, w, act),
            ReshapeLayer((1, w)),
            ConvTranspose((4,), w => w, act; stride = 1, pad = 0), # *4 = 4
            ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 16
            ConvTranspose((8,), w => w, act; stride = 4, pad = 2), # *4 = 64
            ConvTranspose((8,), w => 1     ; stride = 4, pad = 2), # *4 = 256
        )
        
        Chain(; encoder, decoder)

    elseif prob isa Advection2D # (128, 128) -> l -> (128, 128)

        encoder = Chain(
            Conv((8, 8), 1  => w, act; stride = 4, pad = 2), # /4 = 32
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4 = 8
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4 = 2
            Conv((2, 2), w  => w, act; stride = 1, pad = 0), # /2 = 1
            flatten,
            Dense(w, l),
        )

        decoder = Chain(
            Dense(l, w, act),
            ReshapeLayer((1, 1, w)),
            ConvTranspose((4, 4), w => w, act; stride = 1, pad = 0), # *4 = 4
            ConvTranspose((8, 8), w => w, act; stride = 4, pad = 2), # *4 = 16
            ConvTranspose((8, 8), w => w, act; stride = 4, pad = 2), # *4 = 64
            ConvTranspose((8, 8), w => 1     ; stride = 2, pad = 3), # *2 = 128
        )
        
        Chain(; encoder, decoder)

    elseif prob isa BurgersViscous2D # (512, 512, 2) -> l -> (512, 512, 2)

        encoder = Chain(
            Conv((8, 8), 2  => w, act; stride = 4, pad = 2), # /4 = 128
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4 = 32
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4 = 8
            Conv((8, 8), w  => w, act; stride = 1, pad = 0), # /8 = 1
            flatten,
            Dense(w, l),
        )

        decoder = Chain(
            Dense(l, w, act),
            ReshapeLayer((1, 1, w)),
            ConvTranspose((8, 8), w => w, act; stride = 1, pad = 0), # *8 = 8
            ConvTranspose((8, 8), w => w, act; stride = 4, pad = 2), # *4 = 32
            ConvTranspose((8, 8), w => w, act; stride = 4, pad = 2), # *4 = 128
            ConvTranspose((8, 8), w => 2     ; stride = 4, pad = 2), # *4 = 512
        )

        Chain(; encoder, decoder)
    end
end

#======================================================#
function inr_decoder(l, h, wd, in_dim, out_dim)
    init_wt_in = scaled_siren_init(1f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    act = sin

    wi = l + in_dim
    wo = out_dim

    in_layer = Dense(wi, wd, act; init_weight = init_wt_in, init_bias)
    hd_layer = Dense(wd, wd, act; init_weight = init_wt_hd, init_bias)
    fn_layer = Dense(wd, wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

    Chain(in_layer, fill(hd_layer, h)..., fn_layer)
end

function convINR_network(
    prob::NeuralROMs.AbstractPDEProblem,
    l::Integer,
    h::Integer,
    we::Integer,
    wd::Integer,
    act,
)
    Ns = get_prob_grid(prob)
    in_dim  = length(Ns)
    out_dim = prob isa BurgersViscous2D ? 2 : 1

    encoder = cae_network(prob, l, we, act).layers.encoder
    decoder = inr_decoder(l, h, wd, in_dim, out_dim)

    ImplicitEncoderDecoder(encoder, decoder, Ns, out_dim)
end
#======================================================#

function loaddata(datafile::String)

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

    println("input size $in_dim with $(size(x, 2)) points per trajectory.")
    println("output size $out_dim.")

    @assert eltype(x) === Float32
    @assert eltype(u) === Float32

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
"""
Input size `[in_dim, Npoints]`
"""
function normalize_x(x::AbstractMatrix)

    x̄  = sum(x, dims = 2) / size(x, 2) |> vec
    σx = sum(abs2, x .- x̄, dims = 2) / size(x, 2) .|> sqrt |> vec
    x  = normalizedata(x, x̄, σx)

    x, x̄, σx
end

"""
Input size `[out_dim, ...]`
"""
function normalize_u(u::AbstractArray{T,N}) where{T} where{N}

    out_dim = size(u, 1)
    dims = 2:N
    den  = prod(size(u)[dims])

    ū  = sum(u; dims) / den |> vec
    σu = sum(abs2, u .- ū; dims) / den .|> sqrt |> vec
    u  = normalizedata(u, ū, σu)

    u, ū, σu
end

"""
Input size `[Ntime]`
"""
function normalize_t(t::AbstractVector)

    N = length(t)

    t̄  = sum(t) / N
    σt = sum(abs2, t .- t̄) / N .|> sqrt
    t  = normalizedata(t, t̄, σt)

    t, t̄, σt
end

#======================================================#
function displaymetadata(metadata::NamedTuple)
    println("METADATA:")
    println("ū, σu: $(metadata.ū), $(metadata.σu)")
    println("x̄, σx: $(metadata.x̄), $(metadata.σx)")
    println("Model README: ", metadata.readme)
    println("Data-metadata: ", metadata.md_data)
    println("train_args: ", metadata.train_args)
    println("Nx, _Ncodes, Ncodes_: $(metadata.Nx), $(metadata._Ns), $(metadata.Ns_)")
    nothing
end

#===========================================================#
function eval_model(
    model::NeuralROMs.AbstractNeuralModel,
    x::AbstractArray,
    p::AbstractArray;
    batchsize = numobs(x) ÷ 100,
    device = Lux.cpu_device(),
)
    loader = MLUtils.DataLoader(x; batchsize, shuffle = false, partial = true)

    p = p |> device
    model = model |> device

    if device isa Lux.LuxCUDADevice
        loader = CuIterator(loader)
    end

    y = ()
    for batch in loader
        yy = model(batch, p)
        y = (y..., yy)
    end

    hcat(y...) |> Lux.cpu_device()
end

function eval_model(
    model::NTuple{3, Any},
    x;
    batchsize = numobs(x) ÷ 100,
    device = Lux.cpu_device(),
)
    NN, p, st = model

    loader = MLUtils.DataLoader(x; batchsize, shuffle = false, partial = true)

    p, st = (p, st) |> device
    st = Lux.testmode(st)

    if device isa Lux.LuxCUDADevice
        loader = CuIterator(loader)
    end

    y = ()
    for batch in loader
        yy = NN(batch, p, st)[1]
        y = (y..., yy)
    end

    hcat(y...) |> Lux.cpu_device()
end

#======================================================#
function p_axes(NN;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    p = Lux.setup(copy(rng), NN)[1]
    p = ComponentArray(p)
    getaxes(p)
end

#===================================================#
function fieldplot(
    Xdata::AbstractArray,
    Tdata::AbstractArray,
    Udata::AbstractArray,
    Upred::AbstractArray, 
    grid::Tuple,
    outdir::String,
    prefix::String,
    case::Integer,
)
    in_dim  = size(Xdata, 1)
    out_dim = size(Udata, 1)

    linewidth = 2.0
    palette = :tab10

    for od in 1:out_dim
        up = Upred[od, :, :]
        ud = Udata[od, :, :]
        nr = sum(abs2, ud) / length(ud) |> sqrt
        er = (up - ud) / nr
        er = sum(abs2, er; dims = 1) / size(ud, 1) .|> sqrt |> vec

        Nx, Nt = size(Xdata, 2), length(Tdata)

        if in_dim == 1
            xd = vec(Xdata)

            Ixplt = LinRange(1, Nx, 32) .|> Base.Fix1(round, Int)
            Itplt = LinRange(1, Nt,  4) .|> Base.Fix1(round, Int)

            # u(x, t)
            plt = plot(;
                # title = "Ambient space evolution, case = $(case)",
                xlabel = L"x", ylabel = L"u(x,t)", legend = false,
            )
            plot!(plt, xd, up[:, Itplt]; linewidth, palette)
            scatter!(plt, xd[Ixplt], ud[Ixplt, Itplt]; w = 1, palette)
            png(plt, joinpath(outdir, "$(prefix)_u$(od)_case$(case)"))

            # anim = animate1D(Ud[:, It_data], Up[:, It_pred], vec(Xdata), Tdata[It_data];
            #                  w = 2, xlabel, ylabel, title)
            # gif(anim, joinpath(outdir, "train$(k).gif"); fps)

        elseif in_dim == 2
            xlabel = L"x"
            ylabel = L"y"
            zlabel = L"u$(od)(x, t)"

            kw = (; xlabel, ylabel, zlabel,)

            x_re = reshape(Xdata[1, :], grid)
            y_re = reshape(Xdata[2, :], grid)

            xline = x_re[:, 1]
            yline = x_re[1, :]

            upred_re = reshape(up, grid..., :)
            udata_re = reshape(ud, grid..., :)

            Itplt = LinRange(1, Nt, 5) .|> Base.Fix1(round, Int)

            for (i, idx) in enumerate(Itplt)
                up_re = upred_re[:, :, idx]
                ud_re = udata_re[:, :, idx]

                # p1 = plot()
                # p1 = meshplt(x_re, y_re, up_re; plt = p1, c=:black, w = 1.0, kw...,)
                # p1 = meshplt(x_re, y_re, up_re - ud_re; plt = p1, c=:red  , w = 0.2, kw...,)
                #
                # p2 = meshplt(x_re, y_re, ud_re - up_re; title = "error", kw...,)
                #
                # png(p1, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)"))
                # png(p2, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)_error"))

                p3 = heatmap(up_re) #; title = "u$(od)(x, y)")
                p4 = heatmap(abs.(up_re - ud_re)) #; title = "u$(od)(x, y)")

                png(p3, joinpath(outdir, "$(prefix)_u$(od)_$(case)_time_$(i)"))
                png(p4, joinpath(outdir, "$(prefix)_u$(od)_$(case)_time_$(i)_error"))

            end
        else
            throw(ErrorException("in_dim = $in_dim not supported."))
        end

        # e(t)
        plt = plot(;
            title = "Error evolution, case = $(case)",
            xlabel = L"Time ($s$)", ylabel = L"ε(t)", legend = false,
            yaxis = :log, ylims = (10^-5, 1.0),
        )

        plot!(plt, Tdata, er; linewidth, palette, ylabel = "ε(t)")
        png(plt, joinpath(outdir, "$(prefix)_e$(od)_case$(case)"))
    end
    
end

#======================================================#
function make_param_scatterplot(
    p::AbstractMatrix,
    t::AbstractVector,
    ; plt = plot(),
    kw...
)
    @assert size(p, 2) == length(t)

    kw_scatter = (; kw..., zcolor = t, markerstrokewidth = 0)

    if size(p, 1) == 1
        plot!(plt, vec(p), t; kw...,)
    elseif size(p, 1) == 2
        scatter!(plt, p[1,:], p[2,:]; kw_scatter...,)
    elseif size(p, 1) == 3
        scatter!(plt, p[1,:], p[2,:], p[3,:]; kw_scatter...,)
    else
        # TSNE
    end

    plt
end
#======================================================#

function make_optimizer(
    E::Integer,
    warmup::Bool,
    weightdecay = nothing,
)
    lrs = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    Nlrs = length(lrs)

    # Grokking (https://arxiv.org/abs/2201.02177)
    # Optimisers.Adam(lr, (0.9f0, 0.95f0)), # 0.999 (default), 0.98, 0.95
    # https://www.youtube.com/watch?v=IHikLL8ULa4&ab_channel=NeelNanda
    opts = if isnothing(weightdecay)
        Tuple(
            Optimisers.Adam(lr) for lr in lrs
        )
    else
        Tuple(
            OptimiserChain(
                Optimisers.Adam(lr),
                weightdecay,
            )
            for lr in lrs
        )
    end

    nepochs = (round.(Int, E / (Nlrs) * ones(Nlrs))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, Nlrs)...,)

    if warmup
        opt_warmup = if isnothing(weightdecay)
            Optimisers.Adam(1f-2)
        else
            OptimiserChain(Optimisers.Adam(1f-2), weightdecay,)
        end
        nepochs_warmup = 10
        schedule_warmup = Step(1f-2, 1f0, Inf32)
        early_stopping_warmup = true

        ######################
        opts = (opt_warmup, opts...,)
        nepochs = (nepochs_warmup, nepochs...,)
        schedules = (schedule_warmup, schedules...,)
        early_stoppings = (early_stopping_warmup, early_stoppings...,)
    end
    
    opts, nepochs, schedules, early_stoppings
end

#======================================================#

function ps_indices(NN, property::Symbol;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    p = Lux.setup(copy(rng), NN)[1]
    p = ComponentArray(p)
    idx = only(getaxes(p))[property].idx

    println("[ps_indices]: Passing $(length(idx)) / $(length(p)) $(property) parameters to IdxWeightDecay")

    idx
end

function ps_W_indices(
    NN,
    property::Union{Symbol, Nothing} = nothing;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    p = Lux.setup(copy(rng), NN)[1]
    p = ComponentArray(p)

    idx = Int32[]

    pprop = isnothing(property) ? p : getproperty(p, property)

    for lname in propertynames(pprop)
        w = getproperty(pprop, lname).weight # reshaped array

        @assert ndims(w) == 2

        i = if w isa Base.ReshapedArray
            only(w.parent.indices)
        elseif w isa SubArray
            w.indices
        end

        println("Grabbing weight indices from $(property) layer $(lname), size $(size(w)).")

        idx = vcat(idx, Int32.(i))
    end

    println("[ps_W_indices]: Passing $(length(idx)) / $(length(p)) $(property) parameters to IdxWeightDecay")

    idx
end

#======================================================#
