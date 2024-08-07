#
#======================================================#
# Gaussian
#======================================================#
function makemodelGaussian(
    data::NTuple{2,Any},
    train_params::NamedTuple,
    periods,
    metadata::NamedTuple,
    dir::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    device = Lux.gpu_device()
)

    in_dim  = size(data[1], 1)
    out_dim = size(data[2], 1)

    #--------------------------------------------#
    # get train params
    #--------------------------------------------#

    w = haskey(train_params, :w) ? train_params.w : 10

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    periodic = PeriodicEmbedding(1:in_dim, periods)
    periodic = NoOpLayer()

    decoder = begin
        i = if periodic isa PeriodicEmbedding
            2 * in_dim
        elseif periodic isa PeriodicLayer
            w
        elseif periodic isa NoOpLayer
            in_dim
        end

        o = out_dim

        GaussianLayer(i, o)
    end

    NN = Chain(; periodic, decoder)

    #-------------------------------------------#
    p, st = Lux.setup(rng, NN)
    p = ComponentArray(p)

    ST = nothing
    metadata = (;
        metadata..., 
        x̄ = metadata.x̄ * 0,
        ū = metadata.ū * 0,
        σx = metadata.σx * 0 .+ 1,
        σu = metadata.σu * 0 .+ 1,
    )

    p.decoder.x̄  .= -0.5f0
    p.decoder.σi .= 1 / 0.1f0

    #-------------------------------------------#
    model = NN, p, st
    jldsave(joinpath(dir, "model.jld2"); model, ST, metadata)
    #-------------------------------------------#

    model, ST, metadata
end

#======================================================#
# DNN
#======================================================#
function makemodelDNN(
    data,
    train_params,
    periods,
    metadata,
    dir;
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    device = Lux.gpu_device()
)
    in_dim  = size(data[1], 1)
    out_dim = size(data[2], 1)

    #--------------------------------------------#
    # get train params
    #--------------------------------------------#

    h = haskey(train_params, :h) ? train_params.h : 1
    w = haskey(train_params, :w) ? train_params.w : 10
    E = haskey(train_params, :E) ? train_params.E : 2100
    act = haskey(train_params, :act) ? train_params.act : sin

    γ = haskey(train_params, :γ) ? train_params.γ : 1f-2
    λ = haskey(train_params, :λ) ? train_params.λ : 0f-0

    _batchsize = haskey(train_params, :_batchsize) ? train_params._batchsize : nothing
    batchsize_ = haskey(train_params, :batchsize_) ? train_params.batchsize_ : nothing

    warmup = haskey(train_params, :warmup) ? train_params.warmup : true

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    # periodic = NoOpLayer()
    periodic = PeriodicEmbedding(1:in_dim, periods)
    # periodic = PeriodicLayer(w, periods)

    decoder = begin
        if act ∈ (sin, cos)
            init_wt_in = scaled_siren_init(1f1)
            init_wt_hd = scaled_siren_init(1f0)
            init_wt_fn = glorot_uniform
            init_bias = rand32
        else
            init_wt_in = glorot_uniform
            init_wt_hd = glorot_uniform
            init_wt_fn = glorot_uniform
            init_bias = zeros32
        end

        use_bias_fn = false

        i = if periodic isa PeriodicEmbedding
            2 * in_dim
        elseif periodic isa PeriodicLayer
            w
        elseif periodic isa NoOpLayer
            in_dim
        end

        o = out_dim

        in_layer = Dense(i, w, act; init_weight = init_wt_in, init_bias)
        hd_layer = Dense(w, w, act; init_weight = init_wt_hd, init_bias)
        fn_layer = Dense(w, o     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

        Chain(in_layer, fill(hd_layer, h)..., fn_layer)
    end

    NN = Chain(; periodic, decoder)

    #-------------------------------------------#
    # training hyper-params
    #-------------------------------------------#

    _batchsize = isnothing(_batchsize) ? numobs(data) ÷ 10 : _batchsize
    batchsize_ = isnothing(batchsize_) ? numobs(data) ÷ 1  : batchsize_

    lossfun = mse

    idx = ps_W_indices(NN, :decoder; rng)
    weightdecay = IdxWeightDecay(0f0, idx)
    opts, nepochs, schedules, early_stoppings = make_optimizer(E, warmup, weightdecay)

    #-------------------------------------------#

    train_args = (; h, w, E, λ, γ, _batchsize, batchsize_)
    metadata   = (; metadata..., train_args)

    display(NN)

    @time model, ST = train_model(NN, data; rng,
        _batchsize, batchsize_, weight_decays = γ,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    plot_training!(ST...) |> display

    model, ST, metadata
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

#===========================================================#

function ps_W_indices(
    NN,
    property::Union{Symbol, Nothing} = nothing;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    p = Lux.setup(copy(rng), NN)[1]
    p = ComponentArray(p)

    idx = Int32[]
    pprop = isnothing(property) ? p : getproperty(p, property)

    pNames = propertynames(pprop)
    pNum   = length(pNames)

    for i in 1:(pNum-1)
        lName = pNames[i]

        w = getproperty(pprop, lName).weight # reshaped array

        @assert ndims(w) == 2

        i = if w isa Base.ReshapedArray
            only(w.parent.indices)
        elseif w isa SubArray
            w.indices
        end

        println("[ps_W_indices]: Grabbing weight indices from [$i / $pNum] $(property) layer $(lName), size $(size(w)).")
        idx = vcat(idx, Int32.(i))
    end

    println("[ps_W_indices]: Passing $(length(idx)) / $(length(p)) $(property) parameters to IdxWeightDecay")

    idx
end
#===========================================================#
#
