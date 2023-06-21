#
"""
$SIGNATURES

"""
function model_setup(NN::Lux.AbstractExplicitLayer, data)

    x, ŷ = data

    function model(p, st)
        NN(x, p, st)[1]
    end

    function loss(p, st)
        y = NN(x, p, st)[1]

        mse(y, ŷ), st
    end

    function stats(p, st; io::Union{Nothing, IO} = stdout)
        y = model(p, st)
        Δy = y - ŷ

        R2 = rsquare(y, ŷ)
        MSE = mse(y, ŷ)

        meanAE = norm(Δy, 1) / length(ŷ)
        maxAE  = norm(Δy, Inf)

        rel   = Δy ./ ŷ
        meanRE = norm(rel, 1) / length(ŷ)
        maxRE  = norm(rel, Inf)

        if !isnothing(io)
            str = ""
            str *= string("R² score:       ", round(R2    , digits=8), "\n")
            str *= string("mean SQR error: ", round(MSE   , digits=8), "\n")
            str *= string("mean ABS error: ", round(meanAE, digits=8), "\n")
            str *= string("max  ABS error: ", round(maxAE , digits=8), "\n")
            str *= string("mean REL error: ", round(meanRE, digits=8), "\n")
            str *= string("max  REL error: ", round(maxRE , digits=8))

            println(io, str)
        end

        R2, MSE, meanAE, maxAE, meanRE, maxRE
    end

    model, loss, stats
end

"""
$SIGNATURES

"""
function callback(p, st; io::Union{Nothing, IO} = stdout,
                  _loss = nothing, _LOSS = nothing, _stats = nothing,
                  loss_ = nothing, LOSS_ = nothing, stats_ = nothing,
                  iter = nothing, step = 0, maxiter = 0, ITER = nothing,
                 )

    str = if !isnothing(iter)
        step = iszero(step) ? 10 : step

        if iter % step == 0 || iter == 1 || iter == maxiter
            "Iter $iter: "
        else
            return
        end
    else
        ""
    end

    # log iter
    if !isnothing(ITER) & !isnothing(iter)
        push!(ITER, iter)
    end

    # log training loss
    _l = if !isnothing(_loss)
        _l = _loss(p, st)[1]
        !isnothing(_LOSS) && push!(_LOSS, _l)
        _l
    else
        nothing
    end

    # log test loss
    l_ = if !isnothing(loss_)
        l_ = loss_(p, st)[1]
        !isnothing(LOSS_) && push!(LOSS_, l_)
        l_
    else
        nothing
    end

    isnothing(io) && return

    if !isnothing(_l)
        str *= string("TRAIN LOSS: ", round(_l, digits=8), " ")
    end

    if !isnothing(l_)
        str *= string("TEST LOSS: ", round(l_, digits=8), " ")
    end

    println(io, str)

    if !isnothing(io)
        if !isnothing(_stats)
            println(io, "### TRAIN STATS ###")
            _stats(p, st)
            println(io, "#=================#")
        end
        if !isnothing(stats_) 
            println(io, "### TEST  STATS ###")
            stats_(p, st)
            println(io, "#=================#")
        end
    end

    return
end

"""
$SIGNATURES

Train parameters `p` to minimize `loss` using optimization strategy `opt`.

# Arguments
- Loss signature: `loss(p, st) -> y, st`
- Callback signature: `cb(p, st iter, maxiter) -> nothing` 
"""
function train(loss, p, st, maxiter; opt = Optimisers.Adam(), cb = nothing)

    function grad(p, st)
        loss2 = Base.Fix2(loss, st)
        (l, st), pb = Zygote.pullback(loss2, p)
        gr = pb((one.(l), nothing))[1]

        l, gr, st
    end

    # print stats
    !isnothing(cb) && cb(p, st, 0, maxiter)

    # dry run
    grad(p, st)

    # init optimizer
    opt_st = Optimisers.setup(opt, p)

    for iter in 1:maxiter
        _, gr, st = grad(p, st)
        opt_st, p = Optimisers.update(opt_st, p, gr)

        !isnothing(cb) && cb(p, st, iter, maxiter)
    end

    p, st
end

function plot_training(ITER, _LOSS, LOSS_)
    z = findall(iszero, ITER)

    # fix ITER to account for multiple training loops
    if length(z) > 1
            for i in 2:length(z)-1
            idx =  z[i]:z[i+1] - 1
            ITER[idx] .+= ITER[z[i] - 1]
        end
        ITER[z[end]:end] .+= ITER[z[end] - 1]
    end

    plt = plot(title = "Training Plot", yaxis = :log,
               xlabel = "Epochs", ylabel = "Loss (MSE)")

    plot!(plt, ITER, _LOSS, w = 2.0, c = :green, label = "Train Dataset")
    plot!(plt, ITER, LOSS_, w = 2.0, c = :red, label = "Test Dataset")

    plt
end

"""
$SIGNATURES

"""
function visualize(V, _data, data_, NN, p, st; nsamples = 5)

    x, = points(V)

    _x, _ŷ = _data
    x_, ŷ_ = data_

    _y = NN(_x, p, st)[1]
    y_ = NN(x_, p, st)[1]

    N, K = size(_ŷ)

    I = rand(axes(_ŷ, 2), nsamples)
    n = 4
    ms = 4

    cmap = range(HSV(0,1,1), stop=HSV(-360,1,1), length = nsamples + 1)

    # Trajectory plots
    kw = (; legend = false, xlabel = "x", ylabel = "u(x)")

    _p0 = plot(;title = "Training Comparison, N=$N points, $K trajectories", kw...)
    p0_ = plot(;title = "Testing Comparison, N=$N points, $K trajectories", kw...)

    for i in 1:nsamples
        c = cmap[i]
        ii = I[i]

        kw_data = (; markersize = ms, c = c,)
        kw_pred = (; s = :solid, w = 2.0, c = c)

        # training
        __y = _y[:, ii]
        __ŷ = _ŷ[:, ii]
        scatter!(_p0, x[begin:n:end], __ŷ[begin:n:end]; kw_data...)
        plot!(_p0, x, __y; kw_pred...)

        # testing
        y__ = y_[:, ii]
        ŷ__ = ŷ_[:, ii]
        scatter!(p0_, x[begin:n:end], ŷ__[begin:n:end]; kw_data...)
        plot!(p0_, x, y__; kw_pred...)
    end

    _R2 = round(rsquare(_y, _ŷ), digits = 8)
    R2_ = round(rsquare(y_, ŷ_), digits = 8)

    kw = (; legend = false, xlabel = "u(x) (data)", ylabel = "û(x) (pred)", aspect_ratio = :equal)

    _p1 = plot(; title = "Training R² = $_R2", kw...)
    p1_ = plot(; title = "Testing R² = $R2_", kw...)

    scatter!(_p1, vec(_y), vec(_ŷ), ms = 2)
    scatter!(p1_, vec(y_), vec(ŷ_), ms = 2)

    _l = [extrema(_y)...]
    l_ = [extrema(y_)...]
    plot!(_p1, _l, _l, w = 4.0, c = :red)
    plot!(p1_, l_, l_, w = 4.0, c = :red)

    _p0, p0_, _p1, p1_
end

"""
    mse(ypred, ytrue)

Mean squared error
"""
mse(y, ŷ) = sum(abs2, ŷ - y) / length(ŷ)

"""
    rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)

Calculuate r2 (coefficient of determination) score.
"""
function rsquare(y, ŷ)
    @assert size(y) == size(ŷ)

    y = vec(y)
    ŷ = vec(ŷ)

    ȳ = sum(ŷ) / length(ŷ)   # mean
    MSE  = sum(abs2, ŷ  - y) # mse  (sum of squares of residuals)
    VAR  = sum(abs2, ŷ .- ȳ) # var  (sum of squares of data)

    rsq =  1 - MSE / (VAR + eps(eltype(y)))

    return rsq
end
#
