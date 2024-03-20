#
using Plots
using TSne
#======================================================#
function get_prob_grid(prob::GeometryLearning.AbstractPDEProblem)
    if prob isa Advection1D
        (128,)
    elseif prob isa BurgersViscous1D
        (1024,)
    elseif prob isa KuramotoSivashinsky1D
        (256,)
    elseif prob isa Advection2D
        (96,96,)
    elseif prob isa BurgersViscous2D
        (512,512,)
    else
        throw(ErrorException("Unsupported problem type."))
    end
end

#======================================================#

function cae_network(
    prob::GeometryLearning.AbstractPDEProblem,
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
            Conv((8, 8), 1  => w, act; stride = 4, pad = 2), # /4 = 24
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4 = 6
            Conv((6, 6), w  => w, act; stride = 1, pad = 0), # /6 = 1
            flatten,
            Dense(w, l),
        )

        decoder = Chain(
            Dense(l, w, act),
            ReshapeLayer((1, 1, w)),
            ConvTranspose((6, 6), w => w, act; stride = 1, pad = 0), # *2 = 2
            ConvTranspose((8, 8), w => w, act; stride = 4, pad = 2), # *4 = 8
            ConvTranspose((8, 8), w => 1     ; stride = 4, pad = 2), # *4 = 32
        )
        
        Chain(; encoder, decoder)

    elseif prob isa BurgersViscous2D # (512, 512) -> l -> (512, 512)

        encoder = Chain(
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4
            Conv((8, 8), w  => w, act; stride = 4, pad = 2), # /4
            Conv((2, 2), w  => w, act; stride = 1, pad = 0), # /2
            flatten,
            Dense(w, l),
        )

        decoder = Chain(
            ConvTranspose((4, 4), w => w, act; stride = 1, pad = 0), # *4 = 4
            ConvTranspose((8, 8), w => w, act; stride = 4, pad = 2), # *4 = 16
            ConvTranspose((8, 8), w => w, act; stride = 4, pad = 2), # *4 = 64
            ConvTranspose((8, 8), w => 1     ; stride = 4, pad = 2), # *4 = 256
        )

        Chain(; encoder, decoder)

    end
end

#======================================================#

function inr_decoder(l, h, w, in_dim, out_dim)
    init_wt_in = scaled_siren_init(3f1)
    init_wt_hd = scaled_siren_init(1f0)
    init_wt_fn = glorot_uniform

    init_bias = rand32 # zeros32
    use_bias_fn = false

    act = sin

    wi = l + in_dim
    wo = out_dim

    in_layer = Dense(wi, w , act; init_weight = init_wt_in, init_bias)
    hd_layer = Dense(w , w , act; init_weight = init_wt_hd, init_bias)
    fn_layer = Dense(w , wo     ; init_weight = init_wt_fn, init_bias, use_bias = use_bias_fn)

    Chain(in_layer, fill(hd_layer, h)..., fn_layer)
end

function convINR_network(
    prob::GeometryLearning.AbstractPDEProblem,
    l::Integer,
    h::Integer,
    we::Integer,
    wd::Integer,
    act,
)

    if prob isa Advection1D
        Ns = (128,)
        in_dim  = 1
        out_dim = 1

        wi = in_dim

        encoder = Chain(
            Conv((8,), wi  => we, act; stride = 4, pad = 2), # /4
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4
            Conv((2,), we  => we, act; stride = 1, pad = 0), # /2
            flatten,
            Dense(we, l),
        )

        decoder = inr_decoder(l, h, wd, in_dim, out_dim)
        
        ImplicitEncoderDecoder(encoder, decoder, Ns, out_dim)

    elseif prob isa KuramotoSivashinsky1D

        Ns = (256,)
        in_dim  = 1
        out_dim = 1

        wi = in_dim

        encoder = Chain(
            Conv((8,), wi  => we, act; stride = 4, pad = 2), # /4
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4
            Conv((4,), we  => we, act; stride = 1, pad = 0), # /4
            flatten,
            Dense(we, l),
        )

        decoder = inr_decoder(l, h, wd, in_dim, out_dim)
        
        ImplicitEncoderDecoder(encoder, decoder, Ns, out_dim)

    elseif prob isa BurgersViscous1D

        Ns = (1024,)
        in_dim  = 1
        out_dim = 1

        wi = in_dim

        encoder = Chain(
            Conv((8,), wi  => we, act; stride = 4, pad = 2), # /4 = 256
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4 = 64
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4 = 16
            Conv((8,), we  => we, act; stride = 4, pad = 2), # /4 = 4
            Conv((4,), we  => we, act; stride = 1, pad = 0), # /4 = 1
            flatten,
            Dense(we, l),
        )

        decoder = inr_decoder(l, h, wd, in_dim, out_dim)
        
        ImplicitEncoderDecoder(encoder, decoder, Ns, out_dim)

    elseif prob isa BurgersViscous2D

        Ns = (512, 512,)
        in_dim  = 1
        out_dim = 1

        wi = in_dim

        encoder = Chain(
            Conv((8,8), wi  => we, act; stride = 4, pad = 2), # /4 = 128
            Conv((8,8), we  => we, act; stride = 4, pad = 2), # /4 = 32
            Conv((8,8), we  => we, act; stride = 4, pad = 2), # /4 = 8
            Conv((8,8), we  => we, act; stride = 1, pad = 0), # /8 = 1
            flatten,
            Dense(we, l),
        )

        decoder = inr_decoder(l, h, wd, in_dim, out_dim)
        
        ImplicitEncoderDecoder(encoder, decoder, Ns, out_dim)

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

    kw = (; kw..., zcolor = t,)

    if size(p, 1) == 1
        scatter!(plt, vec(p); kw...,)
    elseif size(p, 1) == 2
        scatter!(plt, p[1,:], p[2,:]; kw...,)
    elseif size(p, 1) == 3
        scatter!(plt, p[1,:], p[2,:], p[3,:]; kw...,)
    else
        # do TSNE
    end

    plt
end
#======================================================#
