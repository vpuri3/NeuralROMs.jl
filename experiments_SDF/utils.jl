#
#===========================================================#

clamp_vanilla(δ)  = x -> @. clamp(x, -δ, δ)
clamp_tanh(δ)     = x -> @. δ * tanh_fast(x)
clamp_sigmoid(δ)  = x -> @. δ * (2 * sigmoid_fast(x) - 1)
clamp_softsign(δ) = x -> @. δ * softsign(x)

#===========================================================#
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

function eval_model(
    model::NTuple{3, Any},
    x;
    batchsize = numobs(x) ÷ 50,
    device = Lux.gpu_device(),
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
#===========================================================#
#
