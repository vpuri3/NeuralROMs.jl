#
using GeometryLearning
using Plots, TSne, LaTeXStrings

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
    model::GeometryLearning.AbstractNeuralModel,
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

    hcat(y...)
end

function eval_model(
    model::NeuralEmbeddingModel,
    x::Tuple,
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
        yy = model(batch[1], p, batch[2])
        yy = reshape(yy, size(yy, 1), :)
        y = (y..., yy)
    end

    hcat(y...)
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
        nr = sum(abs2, ud) / length(ud)
        er = (up - ud) / nr
        er = sum(abs2, er; dims = 1) / size(ud, 1) |> vec

        Nx, Nt = length(Xdata), length(Tdata)

        if in_dim == 1
            xd = vec(Xdata)

            Ixplt = LinRange(1, Nx, 32) .|> Base.Fix1(round, Int)
            Itplt = LinRange(1, Nt,  4) .|> Base.Fix1(round, Int)

            # u(x, t)
            plt = plot(;
                title = "Ambient space evolution, case = $(case)",
                xlabel = L"x", ylabel = L"u(x,t)", legend = false,
            )
            plot!(plt, xd, up[:, Itplt]; linewidth, palette)
            scatter!(plt, xd[Ixplt], ud[Ixplt, Itplt]; w = 1, palette)
            png(plt, joinpath(outdir, "$(prefix)_u$(od)_case$(case)"))

            # e(t)
            plt = plot(;
                title = "Error evolution, case = $(case)",
                xlabel = L"Time ($s$)", ylabel = L"ε(t)", legend = false,
                yaxis = :log, ylims = (10^-9, 1.0),
            )
            plot!(plt, Tdata, er; linewidth, palette,)
            png(plt, joinpath(outdir, "$(prefix)_$(od)_case$(case)"))

            # anim = animate1D(Ud[:, It_data], Up[:, It_pred], vec(Xdata), Tdata[It_data];
            #                  w = 2, xlabel, ylabel, title)
            # gif(anim, joinpath(outdir, "train$(k).gif"); fps)

        elseif in_dim == 2
            xlabel = "x"
            ylabel = "y"
            zlabel = "u$(od)(x, t)"

            kw = (; xlabel, ylabel, zlabel,)

            x_re = reshape(Xdata[1, :], grid)
            y_re = reshape(Xdata[2, :], grid)

            upred_re = reshape(upred, out_dim, grid..., :)
            udata_re = reshape(udata, out_dim, grid..., :)

            Itplt = LinRange(1, Nt,  4) .|> Base.Fix1(round, Int)

            for i in eachindex(idx_pred)
                up_re = upred_re[:, :, i]
                ud_re = udata_re[:, :, i]

                # p1 = plot()
                # p1 = meshplt(x_re, y_re, up_re; plt = p1, c=:black, w = 1.0, kw...,)
                # p1 = meshplt(x_re, y_re, up_re - ud_re; plt = p1, c=:red  , w = 0.2, kw...,)
                #
                # p2 = meshplt(x_re, y_re, ud_re - up_re; title = "error", kw...,)
                #
                # png(p1, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)"))
                # png(p2, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)_error"))

                p3 = heatmap(up_re; title = "u$(od)(x, y)")
                p4 = heatmap(up_re - ud_re; title = "u$(od)(x, y)")

                png(p3, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)"))
                png(p4, joinpath(outdir, "train_u$(od)_$(k)_time_$(i)_error"))
            end
        else
            throw(ErrorException("in_dim = $in_dim not supported."))
        end

    end
    
end

#======================================================#
function paramplot(
    Tdata::AbstractArray,
    _p::AbstractArray,
    ps::AbstractArray, 
    outdir::String,
    prefix::String,
    case::Integer,
)
    linewidth = 2.0
    palette = :tab10

    # parameter evolution

    plt = plot(;
        title = "Parameter evolution, case $(case)",
        xlabel = L"Time ($s$)", ylabel = L"\tilde{u}(t)", legend = false,
    )
    plot!(plt, Tdata, ps'; linewidth, palette)
    png(plt, joinpath(outdir, "$(prefix)_p_case$(case)"))

    # parameter scatter plot

    plt = plot(; title = "Parameter scatter plot, case = $case")
    plt = make_param_scatterplot(ps, Tdata; plt,
        color = :blues, label = "Evolution" , markerstrokewidth = 0, cbar = false)

    png(plt, joinpath(outdir, "$(prefix)_p_scatter_case$(case)"))

    plt = make_param_scatterplot(_p, Tdata; plt,
        color = :reds , label = "Train data", markerstrokewidth = 0, cbar = false)

    png(plt, joinpath(outdir, "$(prefix)_p_scatter_case$(case)_"))
    
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
