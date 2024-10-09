#
#======================================================#
# Any Kernelized implementation
#======================================================#
function makemodelKernel(
    data::NTuple{2,Any},
    train_params::NamedTuple,
    periods,
    metadata::NamedTuple,
    modeldir::String;
    rng::Random.AbstractRNG = Random.default_rng(),
    verbose::Bool = true,
    device = Lux.gpu_device()
)
	periodic = true
    in_dim  = size(data[1], 1)
    out_dim = size(data[2], 1)

    #--------------------------------------------#
    # get train params
    #--------------------------------------------#

    n = haskey(train_params, :n) ? train_params.N : 1
    N = haskey(train_params, :N) ? train_params.N : 5
    E = haskey(train_params, :E) ? train_params.E : 5000
    T = haskey(train_params, :T) ? train_params.T : Float32

	NN = TK1D(n, N; T)

    #-------------------------------------------#
	# set up training
    #-------------------------------------------#

	_batchsize = 32
	opt = Optimisers.Adam(1f-4)
	# sch = SinExp(1)

	# heuristics
	cb_start = 500
	cb_interval = 300 # Int(E // 1)
	cb_end = 4000

	cmin = 1f-4
	emax = 1f-4

	function per_point_error(NN, p, st, data)
		# per point L1 loss [1, Nx]
		x, yt = data
		yp = NN(x, p, st)[1]
		ex = yp - yt

		Nx = size(x, 2)

		# contribution of each kernal at every point [Nk, Nx]
		yk = NeuralROMs.evaluate_kernels(NN, p, st, x)
		yk = vcat(yk...)

		# per kernel error [Nk]
		E = sum(abs, (yk .* ex); dims = 2) / Nx |> vec
	end

	function cb_epoch(trainer, state, epoch) # loaders
		@unpack NN, p, st, opt_st = state
		@unpack fullbatch_freq = trainer.opt_args

		if ((epoch % cb_interval) == 0) & (epoch ≥ cb_start) & (epoch ≤ cb_end)
			#======================#
			# prune if
			#======================#
			# |c| < cmin
			# expanse (x̄ ± w) out of domain

			# mask_rm = @. st.mask * (abs(p.c) < cmin)
			# println("Pruning $(sum(mask_rm)) kernels.")
			# NN, p, st = NeuralROMs.prune_kernels(NN, p, st, mask_rm)

			#======================#
			# split based on pointwise error metric
			#======================#
			# E = per_point_error(NN, p, st, trainer.data._data)
			# @show E
			#
			# mask_split = 0
			# idx_split = findall(mask_split)

			# split
			NN, p, st, id1 = NeuralROMs.split_kernels(NN, p, st, NN.n:NN.n)

			#======================#
			# additional kernels in problem areas
			#======================#

			#======================#
			println("Number of Kernels: $(NN.n)")
		end

		if (epoch % fullbatch_freq) == 0
		end

		# return
		state = NeuralROMs.TrainState(NN, p, st, opt_st)
		state, false
	end

	function cb_batch(trainer, state, batch, loss, grad, epoch, ibatch)
		state, false
	end

	@time trainer = Trainer(
		NN, data; nepochs = E, _batchsize, opt, make_ca = true,
		print_stats = false, print_batch = false, print_config = false,
		fullbatch_freq = 100, cb_batch, cb_epoch,
		device,
	)

	state, ST = train!(trainer)

	#----------------------------------#
	# analyssi
	#----------------------------------#
	NN = state.NN
	p  = state.p
	st = state.st

	x, y = data

	yy = NN(data[1], p, st)[1]
	ys = NeuralROMs.evaluate_kernels(NN, p, st, data[1])
	ys = vcat(ys...)'

	plt = plot(; xlabel = "x", ylabel = "y", legend = false)
	plot!(plt, vec(x), ys; c = :black, w = 4)
	plot!(plt, vec(x), vec(yy); c = :red  , w = 4)

	imagefile = joinpath(modeldir, "img.png")
	display(plt)
	png(plt, imagefile)

	save_trainer(trainer, modeldir; metadata)

	#----------------------------------#
	# return
	#----------------------------------#
    train_args = (; E, _batchsize)
	metadata   = (; metadata..., train_args)

    (NN, p, st), ST, metadata
end

#======================================================#
# Tanh kernels
#======================================================#
function makemodelTanh(
    data::NTuple{2,Any},
    train_params::NamedTuple,
    periods,
    metadata::NamedTuple,
    modeldir::String;
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

    N = haskey(train_params, :N) ? train_params.N : 1
    E = haskey(train_params, :E) ? train_params.E : 200
    T = haskey(train_params, :T) ? train_params.T : Float32

    Nsplits = haskey(train_params, :Nsplits) ? train_params.Nsplits : 0
    Nboosts = haskey(train_params, :Nboosts) ? train_params.Nboosts : 0

    warmup = haskey(train_params, :warmup) ? train_params.warmup : false
    hessopt = haskey(train_params, :hessopt) ? train_params.hessopt : true

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    i = in_dim
    o = out_dim
    decoder = TanhKernel1D(i, o, N)

    NN = decoder
    #-------------------------------------------#

    lossfun = mse
    batchsize_ = numobs(data)
    opts, nepochs, schedules, early_stoppings, _batchsize = make_optimizer_gaussian(E, numobs(data), warmup, hessopt)

    #-------------------------------------------#
    train_args = (; E, _batchsize, batchsize_)
    metadata   = (; metadata..., train_args)

    #-------------------------------------------#
    # Training with progressive splitting
    #-------------------------------------------#
    p, st = Lux.setup(rng, NN)
    p = ComponentArray(p) .|> T
    ST = nothing
    
    for isplit in 0:Nsplits
        display(NN)
        dir = if iszero(Nsplits)
            modeldir
        else
            joinpath(modeldir, "split$(isplit)")
        end
    
        @time (NN, p, st), ST = train_model(
            NN, data; rng, p, st, _batchsize, batchsize_,
            opts, nepochs, schedules, early_stoppings,
            device, dir, metadata, lossfun,
        )
    
        @show p
        @show length(p)
        plot_training!(ST...) |> display
    
        if isplit != Nsplits
            NN, p, st = split_TanhKernel1D(NN, p, st; debug = true)
        end
    end

    # #-------------------------------------------#
    # # Training with progressive boosting
    # #-------------------------------------------#
    # models = ()
    # ST = nothing
    # _data = data
    #
    # for iboost in 0:Nboosts
    #     display(NN)
    #     dir = if iszero(Nboosts)
    #         modeldir
    #     else
    #         joinpath(modeldir, "boost$(iboost)")
    #     end
    #
    #     #----------------#
    #     p, st = Lux.setup(rng, NN)
    #     if iboost > 0
    #         p.c .= sqrt(sum(abs2, _data[2]) / length(_data[2]))
    #     end
    #     #----------------#
    #
    #     @time (NN, p, st), ST = train_model(
    #         NN, _data; rng, p, st, _batchsize, batchsize_,
    #         opts, nepochs, schedules, early_stoppings,
    #         device, dir, metadata, lossfun,
    #     )
    #     plot_training!(ST...) |> display
    #
    #     models = (models..., (NN, p, st))
    #     _data = _data[1], _data[2] - NN(_data[1], p, st)[1]
    # end
    #
    # NN, p, st = merge_TanhKernel1D(models; debug = true)
    # #-------------------------------------------#

    (NN, p, st), ST, metadata
end

#======================================================#
# Gaussian kernels
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

    Ng = haskey(train_params, :Ng) ? train_params.Ng : 4 # num_gauss
    Nf = haskey(train_params, :Nf) ? train_params.Nf : 4 # num_freqs
    σmin = haskey(train_params, :σmin) ? train_params.σmin : 1e-4
    σsplit = haskey(train_params, :σsplit) ? train_params.σsplit : true
    σinvert = haskey(train_params, :σinvert) ? train_params.σinvert : false
    train_freq = haskey(train_params, :train_freq) ? train_params.train_freq : true

    E = haskey(train_params, :E) ? train_params.E : 100
    T = haskey(train_params, :T) ? train_params.T : Float32
    exactIC = haskey(train_params, :exactIC) ? train_params.exactIC : (;)
    warmup = haskey(train_params, :warmup) ? train_params.warmup : false
    hessopt = haskey(train_params, :hessopt) ? train_params.hessopt : true

    #--------------------------------------------#
    # architecture
    #--------------------------------------------#

    i = in_dim
    o = out_dim

    decoder = begin
        # # AD1D case 1-4
        # Ng = Nf = 1
        # σmin = 1e-2
        # σsplit = false
        # train_freq = false
        # periodic = false # comment out for case 4

        # # AD1D case 5-7
        # Ng = 2
        # Nf = 1
        # σmin = 1e-2
        # σsplit = true
        # train_freq = false
        # # periodic = false # comment out for case 8

        # # AD1D case 8
        # Ng = 4
        # Nf = 1
        # σmin = 1e-2
        # σsplit = true
        # train_freq = false
        # # periodic = false # comment out for case 8

        # # Burg 1D
        # Ng = Nf = 1
        # σmin = 1f-6
        # σsplit = true
        # train_freq = false
        # periodic = false
        # σinvert = true

        # # Burg 1D
        # Ng = 4
        # Nf = 4
        # σmin = 1f-6
        # σsplit = true
        # train_freq = false
        # periodic = false
        # σinvert = true

        Gaussian1D(i, o, Ng, Nf; periodic, σmin, σsplit, σinvert, train_freq)
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
        batchsize_ = numobs(data)
        opts, nepochs, schedules, early_stoppings, _batchsize = make_optimizer_gaussian(E, numobs(data), warmup, hessopt)

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

        if σinvert
            if σsplit
                @show model[2].decoder.w
                @show model[2].decoder.σil
                @show model[2].decoder.σir
            else
                @show model[2].decoder.σi
            end
        else
            if σsplit
                @show model[2].decoder.w
                @show model[2].decoder.σl
                @show model[2].decoder.σr
            else
                @show model[2].decoder.σ
            end
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
    K::Integer, # numobs(data)
    warmup::Bool,
    second_order::Bool = true,
)
    # DON’T DECAY THE LEARNING RATE, INCREASE THE BATCH SIZE
    # https://arxiv.org/pdf/1711.00489

    # # OLD
    # lrs = (1f-3, 1f-4, 1f-5, 1f-6)
    # _batchsize = (1, 1, 1, 1)

    lrs = (1f-3, 1f-3, 1f-3, 1f-3)
    _batchsize = (1, 4, 16, 64)

    N = length(lrs)
    opts = Tuple(Optimisers.Adam(lr) for lr in lrs)
    nepochs = (round.(Int, E / (N) * ones(N))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, N)...,)

    if warmup
        _opt = Optimisers.Adam(1f-2)
        _nepochs = 10
        _schedule = Step(1f-2, 1f0, Inf32)
        _early_stopping = true

        opts = (_opt, opts...,)
        nepochs = (_nepochs, nepochs...,)
        schedules = (_schedule, schedules...,)
        early_stoppings = (_early_stopping, early_stoppings...,)
        _batchsize = (1, _batchsize...)
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
        _batchsize = (_batchsize..., K)
    end

    opts, nepochs, schedules, early_stoppings, _batchsize
end
#======================================================#

function make_optimizer_DNN(
    E::Integer,
    warmup::Bool,
    weightdecay = nothing,
)
    lrs = (1f-3, 5f-4, 2f-4, 1f-4, 5f-5, 2f-5, 1f-5,)
    N = length(lrs)

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

    nepochs = (round.(Int, E / (N) * ones(N))...,)
    schedules = Step.(lrs, 1f0, Inf32)
    early_stoppings = (fill(true, N)...,)

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
