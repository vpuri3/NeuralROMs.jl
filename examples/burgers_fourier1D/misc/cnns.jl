NN = begin
    encoder = Chain(
        Conv((2,), 1  =>  8, act; stride = 2),
        Conv((2,), 8  => 16, act; stride = 2),
        Conv((2,), 16 => 32, act; stride = 2),
        Conv((2,), 32 =>  w, act; stride = 2),
        # BatchNorm(w),
        Conv((2,), w  =>  w, act; stride = 2),
        Conv((2,), w  =>  w, act; stride = 2),
        Conv((2,), w  =>  w, act; stride = 2),
        Conv((2,), w  =>  w, act; stride = 2),
        Conv((2,), w  =>  w, act; stride = 2),
        Conv((2,), w  =>  w, act; stride = 2),
        flatten,
        Dense(w, w, act),
        Dense(w, l),
    )

    decoder = Chain(
        Dense(l, w, act),
        Dense(w, w, act),
        ReshapeLayer((1, w)),
        ConvTranspose((2,), w  =>  w, act; stride = 2),
        ConvTranspose((2,), w  =>  w, act; stride = 2),
        ConvTranspose((2,), w  =>  w, act; stride = 2),
        ConvTranspose((2,), w  =>  w, act; stride = 2),
        ConvTranspose((2,), w  =>  w, act; stride = 2),
        ConvTranspose((2,), w  =>  w, act; stride = 2),
        # BatchNorm(w),
        ConvTranspose((2,), w  => 32, act; stride = 2),
        ConvTranspose((2,), 32 => 16, act; stride = 2),
        ConvTranspose((2,), 16 =>  8, act; stride = 2),
        ConvTranspose((2,), 8  =>  1     ; stride = 2),
    )

    Chain(encoder, decoder)
end

if false
    E = 200 # epochs
    w = 128 # width
    l = 64 # latent
    act = tanh # relu

    opt = Optimisers.Adam()
    batchsize  = 100
    batchsize_ = 200
    learning_rates = 1f-3 ./ (2 .^ (0:9))
    nepochs = E/10 * ones(10) .|> Int
    device = Lux.gpu_device()
    dir = joinpath(@__DIR__, "CAE_wide")

    NN = begin
        encoder = Chain(
            Conv((9,), 1 => w, act; stride = 5, pad = 0),
            Conv((9,), w => w, act; stride = 5, pad = 0),
            # BatchNorm(w, act),
            Conv((9,), w => w, act; stride = 5, pad = 0),
            Conv((7,), w => w, act; stride = 1, pad = 0),
            flatten,
            # Dense(w, w, act),
            Dense(w, l, act),
        )

        decoder = Chain(
            Dense(l, w, act),
            # Dense(w, w, act),
            ReshapeLayer((1, w)),
            ConvTranspose((7,), w => w, act; stride = 1, pad = 0),
            ConvTranspose((10,),w => w, act; stride = 5, pad = 0),
            # BatchNorm(w, act),
            ConvTranspose((9,), w => w, act; stride = 5, pad = 0),
            ConvTranspose((9,), w => 1,    ; stride = 5, pad = 0),
        )

        Chain(encoder, decoder)
    end

    model, ST = train_model(rng, NN, _data, data_, V, opt;
        batchsize, batchsize_, learning_rates, nepochs, dir,
        cbstep = 1, device)
    
    plot_training(ST...) |> display
end

