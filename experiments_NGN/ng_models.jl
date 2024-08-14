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

    periodic = true

    # N = haskey(train_params, :N) ? train_params.N : 4 # num_gauss
    # f = haskey(train_params, :f) ? train_params.f : 4 # num_freqs
    # σmin = haskey(train_params, :σmin) ? train_params.σmin : 1e-4
    # train_freq = haskey(train_params, :train_freq) ? train_params.train_freq : true

    T = haskey(train_params, :T) ? train_params.T : Float32
    E = haskey(train_params, :E) ? train_params.E : 200
    exactIC = haskey(train_params, :exactIC) ? train_params.exactIC : (;)

    warmup = haskey(train_params, :warmup) ? train_params.warmup : false
    hessopt = haskey(train_params, :hessopt) ? train_params.hessopt : true

    _batchsize = haskey(train_params, :_batchsize) ? train_params._batchsize : 1
    batchsize_ = haskey(train_params, :batchsize_) ? train_params.batchsize_ : numobs(data)

    # TODO: increase batchsize during training
    # https://arxiv.org/abs/1711.00489

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    decoder = begin
        i = in_dim
        o = out_dim

        # case 1-4
        Ng = Nf = 1
        σmin = 1e-2
        σsplit = false
        train_freq = false
        periodic = false # comment out for case 4

        # # case 5-7
        # Ng = 2
        # Nf = 1
        # σmin = 1e-2
        # σsplit = true
        # train_freq = false
        # # periodic = false # comment out for case 8

        # # case 8
        # Ng = 4
        # Nf = 1
        # σmin = 1e-2
        # σsplit = true
        # train_freq = false
        # # periodic = false # comment out for case 8

        GaussianLayer1D(i, o, Ng, Nf; periodic, σmin, σsplit, train_freq)
    end

    NN = Chain(; decoder)

    #-------------------------------------------#
    model, ST, metadata = if !isempty(exactIC)
        @set! NN.decoder.periodic = false

        p, st = Lux.setup(rng, NN)
        p = ComponentArray(p)

        metadata = (;
            metadata..., 
            x̄ = metadata.x̄ * 0,
            ū = metadata.ū * 0,
            σx = metadata.σx * 0 .+ 1,
            σu = metadata.σu * 0 .+ 1,
        )

        p.decoder.c .= exactIC.c .|> T
        p.decoder.x̄ .= exactIC.x̄ .|> T
        p.decoder.σ .= exactIC.σ .|> T

        ST = nothing
        model = NN, p, st
        jldsave(joinpath(dir, "model.jld2"); model, ST, metadata)

        model, ST, metadata
    else
        #-------------------------------------------#
        lossfun = mse
        opts, nepochs, schedules, early_stoppings = make_optimizer_gaussian(E, warmup, hessopt)
        #-------------------------------------------#

        train_args = (; E, _batchsize, batchsize_)
        metadata   = (; metadata..., train_args)

        #----------------#
        # mess with initialization
        #----------------#
        p, st = Lux.setup(rng, NN)
        p = ComponentArray(p) .|> T
        ST = nothing
        model = NN, p, st
        #----------------#

        display(NN)

        @time model, ST = train_model(
            NN, data; rng, p, _batchsize, batchsize_,
            opts, nepochs, schedules, early_stoppings,
            device, dir, metadata, lossfun,
        )
        
        plot_training!(ST...) |> display

        @show model[2].decoder.b
        @show model[2].decoder.c
        @show model[2].decoder.x̄

        if σsplit
            @show model[2].decoder.w
            @show model[2].decoder.σl
            @show model[2].decoder.σr
        else
            @show model[2].decoder.σ
        end

        if train_freq
            @show model[2].decoder.ω
            @show model[2].decoder.ϕ
        else
            # @show model[3].decoder.ω
            # @show model[3].decoder.ϕ
        end

        @show length(model[2])
        
        model, ST, metadata
    end

    #-------------------------------------------#
    model, ST, metadata
end

#======================================================#
# MFN (Multiplicative Filter Networks)
#======================================================#
function makemodelMFN(
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

    T = haskey(train_params, :T) ? train_params.T : Float32
    MFNfilter = haskey(train_params, :MFNfilter) ? train_params.MFNfilter : :Fourier

    if MFNfilter === :Fourier
        h = haskey(train_params, :h) ? train_params.h : 3
        w = haskey(train_params, :w) ? train_params.w : 8
        E = haskey(train_params, :E) ? train_params.E : 2100
        γ = haskey(train_params, :γ) ? train_params.γ : 0f-4
    elseif MFNfilter === :Gabor
        h = haskey(train_params, :h) ? train_params.h : 5
        w = haskey(train_params, :w) ? train_params.w : 32
        E = haskey(train_params, :E) ? train_params.E : 2100
        γ = haskey(train_params, :γ) ? train_params.γ : 0f-2
    end

    warmup = haskey(train_params, :warmup) ? train_params.warmup : true
    _batchsize = haskey(train_params, :_batchsize) ? train_params._batchsize : 1
    batchsize_ = haskey(train_params, :batchsize_) ? train_params.batchsize_ : numobs(data)

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    # periodic = NoOpLayer()
    periodic = PeriodicEmbedding(1:in_dim, periods)

    decoder  = begin
        i = if periodic isa PeriodicEmbedding
            2 * in_dim
        elseif periodic isa PeriodicLayer
            w
        elseif periodic isa NoOpLayer
            in_dim
        end
        o = out_dim

        if MFNfilter === :Fourier
            FourierMFN(i, w, o, h)
        elseif MFNfilter === :Gabor
            GaborMFN(i, w, o, h)
        end
    end

    NN = Chain(; periodic, decoder)

    #-------------------------------------------#
    # training hyper-params
    #-------------------------------------------#

    lossfun = mse
    idx = mfn_W_indices(NN, :decoder; rng)
    weightdecay = IdxWeightDecay(0f0, idx)
    opts, nepochs, schedules, early_stoppings = make_optimizer_DNN(E, warmup, weightdecay)

    #-------------------------------------------#

    train_args = (; h, w, E, γ, _batchsize, batchsize_)
    metadata   = (; metadata..., train_args)

    display(NN)
    display(metadata)

    @time model, ST = train_model(NN, data; rng,
        _batchsize, batchsize_, weight_decays = γ,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    display(NN)
    display(metadata)

    plot_training!(ST...) |> display

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

    T = haskey(train_params, :T) ? train_params.T : Float32
    h = haskey(train_params, :h) ? train_params.h : 1
    w = haskey(train_params, :w) ? train_params.w : 10
    E = haskey(train_params, :E) ? train_params.E : 700
    γ = haskey(train_params, :γ) ? train_params.γ : 1f-4
    act = haskey(train_params, :act) ? train_params.act : sin

    warmup = haskey(train_params, :warmup) ? train_params.warmup : true
    _batchsize = haskey(train_params, :_batchsize) ? train_params._batchsize : 1
    batchsize_ = haskey(train_params, :batchsize_) ? train_params.batchsize_ : numobs(data)

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    # periodic = NoOpLayer()
    # periodic = PeriodicLayer(w, periods)
    periodic = PeriodicEmbedding(1:in_dim, periods)

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

    lossfun = mse
    idx = dnn_W_indices(NN, :decoder; rng)
    weightdecay = IdxWeightDecay(0f0, idx)
    opts, nepochs, schedules, early_stoppings = make_optimizer_DNN(E, warmup, weightdecay)

    #-------------------------------------------#

    train_args = (; h, w, E, γ, _batchsize, batchsize_)
    metadata   = (; metadata..., train_args)

    display(NN)
    display(metadata)

    @time model, ST = train_model(NN, data; rng,
        _batchsize, batchsize_, weight_decays = γ,
        opts, nepochs, schedules, early_stoppings,
        device, dir, metadata, lossfun,
    )

    display(NN)
    display(metadata)

    plot_training!(ST...) |> display

    model, ST, metadata
end
#======================================================#

function make_optimizer_gaussian(
    E::Integer,
    warmup::Bool,
    second_order::Bool = true,
)
    # LR ∈ [1e-5, 1e-3] with exponential decay.
    # Then second-order optimizer

    # Gradient descent / Adam aren't converging at all.

    lrs = (1f-3, 1f-4, 1f-5, 1f-6)
    Nlrs = length(lrs)

    opts = Tuple(Optimisers.Adam(lr) for lr in lrs)
    nepochs = (round.(Int, E / (Nlrs) * ones(Nlrs))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, Nlrs)...,)

    if warmup
        _opt = Optimisers.Adam(1f-2)
        _nepochs = 10
        _schedule = Step(1f-2, 1f0, Inf32)
        _early_stopping = true

        opts = (_opt, opts...,)
        nepochs = (_nepochs, nepochs...,)
        schedules = (_schedule, schedules...,)
        early_stoppings = (_early_stopping, early_stoppings...,)
    end

    if second_order
        opt_ = LBFGS()
        nepochs_ = E
        schedule_ = Step(0f-2, 1f0, Inf32)
        early_stopping_ = true

        ######################
        opts = (opts..., opt_)
        nepochs = (nepochs..., nepochs_)
        schedules = (schedules..., schedule_)
        early_stoppings = (early_stoppings..., early_stopping_)
    end

    opts, nepochs, schedules, early_stoppings
end
#======================================================#

function make_optimizer_DNN(
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

function mfn_W_indices(
    NN,
    property::Union{Symbol, Nothing} = nothing;
    rng::Random.AbstractRNG = Random.default_rng(),
)
    p = Lux.setup(copy(rng), NN)[1]
    p = ComponentArray(p)

    idx = Int32[]
    pprop = isnothing(property) ? p : getproperty(p, property) # MFN
    pprop = getproperty(pprop, :filters)

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

        println("[mfn_W_indices]: Grabbing weight indices from [$i / $pNum] $(property) layer $(lName), size $(size(w)).")
        idx = vcat(idx, Int32.(i))
    end

    println("[mfn_W_indices]: Passing $(length(idx)) / $(length(p)) $(property) parameters to IdxWeightDecay")

    idx
end

function dnn_W_indices(
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

        println("[dnn_W_indices]: Grabbing weight indices from [$i / $pNum] $(property) layer $(lName), size $(size(w)).")
        idx = vcat(idx, Int32.(i))
    end

    println("[dnn_W_indices]: Passing $(length(idx)) / $(length(p)) $(property) parameters to IdxWeightDecay")

    idx
end
#===========================================================#
#
