#
""" model """
function model_setup(NN::Lux.AbstractExplicitLayer, st)

    function model(p, x)
        NN(x, p, st)[1]
    end

    function loss(p, x, utrue)
        upred = model(p, x)

        mse(utrue, upred)
    end

    function stats(p, x, utrue)
        upred = model(p, x)
        udiff = upred - utrue

        meanAE = norm(udiff, 1) / length(utrue)
        maxAE  = norm(udiff, Inf)

        urel   = udiff ./ utrue
        meanRE = norm(urel, 1) / length(utrue)
        maxRE  = norm(urel, Inf)

        str = ""
        str *= string("meanAE: "  , round(meanAE, digits=8))
        str *= string(", maxAE: " , round(maxAE , digits=8))
        str *= string(", meanRE: ", round(meanRE, digits=8))
        str *= string(", maxRE: " , round(maxRE , digits=8))
        str *= "\n"

        meanAE, maxAE, meanRE, maxRE, str
    end

    model, loss, stats
end

function callback(p, io = stdout;
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

    if !isnothing(ITER) & !isnothing(iter)
        push!(ITER, iter)
    end

    _l = if !isnothing(_loss)
        _l = _loss(p)
        !isnothing(_LOSS) && push!(_LOSS, _l)
        _l
    else
        nothing
    end

    l_ = if !isnothing(loss_)
        l_ = loss_(p)
        !isnothing(LOSS_) && push!(LOSS_, l_)
        l_
    else
        nothing
    end

    if !isnothing(_l)
        str *= string("TRAIN LOSS: ", round(_l, digits=8), " ")
    end

    if !isnothing(l_)
        str *= string("TEST LOSS: ", round(l_, digits=8), " ")
    end

    println(io, str)

    if !isnothing(io)
        !isnothing(_stats) && print(io, "TRAIN STATS: ", _stats(p)[end])
        !isnothing(stats_) && print(io, "TEST  STATS: ", stats_(p)[end])
    end

    return
end

""" training """
function train(loss, p, maxiter; opt = Optimisers.Adam(), cb = nothing)

    # dry run
    l, pb = Zygote.pullback(loss, p)
    gr = pb(one.(l))[1]

    !isnothing(cb) && cb(p, 0, maxiter)

    # init optimizer
    opt_st = Optimisers.setup(opt, p)

    for iter in 1:maxiter
        l, pb = Zygote.pullback(loss, p)
        gr = pb(one.(l))[1]

        opt_st, p = Optimisers.update(opt_st, p, gr)

        !isnothing(cb) && cb(p, iter, maxiter)
    end

    p
end

function plot_training(ITER, _LOSS, LOSS_)
    # fix ITER
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

""" visualize """
function visualize(V, test, train, model, p; nsamples = 5)

    x, = points(V)[1]

    _x, _y = test
    x_, y_ = train

    _ŷ = model(p, _x)
    ŷ_ = model(p, x_)

    x, = points(V)

    N, K = size(_y)

    I = rand(axes(_y, 2), nsamples)
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
        scatter!(_p0, x[begin:n:end], __y[begin:n:end]; kw_data...)
        plot!(_p0, x, __ŷ; kw_pred...)

        # testing
        y__ = y_[:, ii]
        ŷ__ = ŷ_[:, ii]
        scatter!(p0_, x[begin:n:end], y__[begin:n:end]; kw_data...)
        plot!(p0_, x, ŷ__; kw_pred...)
    end

    _r2 = round(rsquare(_ŷ, _y), digits = 6)
    r2_ = round(rsquare(ŷ_, y_), digits = 6)

    println("TRAINING R2: $_r2")
    println("TESTING  R2: $r2_")

    kw = (; legend = false, xlabel = "u(x) (data)", ylabel = "û(x) (pred)", aspect_ratio = :equal)

    _p1 = plot(; title = "Training R² = $_r2", kw...)
    p1_ = plot(; title = "Testing R² = $r2_", kw...)

    scatter!(_p1, vec(_ŷ), vec(_y), ms = 2)
    scatter!(p1_, vec(ŷ_), vec(y_), ms = 2)

    _l = [extrema(_y)...]
    l_ = [extrema(y_)...]
    plot!(_p1, _l, _l, w = 4.0, c = :red)
    plot!(p1_, l_, l_, w = 4.0, c = :red)

    _p0, p0_, _p1, p1_
end

"""
    mse(ytrue, ypred)

Mean squared error
"""
mse(ypred, ytrue) = sum((ytrue - ypred).^2) / length(ytrue)

"""
    rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)

Calculuate r2 (coefficient of determination) score.
"""
function rsquare(ypred, ytrue)
    @assert size(ypred) == size(ytrue)

    ypred = vec(ypred)
    ytrue = vec(ytrue)

    mean = sum(ytrue) / length(ytrue)
    MSE  = sum((ytrue - ypred).^2) # sum of squares of residuals
    VAR  = sum((ytrue .- mean).^2) # sum of squares of data

    rsq =  1 - MSE / (VAR + eps(eltype(ypred)))

    return rsq
end
#
