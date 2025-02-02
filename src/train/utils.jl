#
#===============================================================#
# STATISTICS COMPUTATION
#===============================================================#
"""
    fullbatch_metric(NN, p, st, loader, lossfun, ismean) -> l

Only for callbacks. Enforce this by setting Lux.testmode

- `NN, p, st`: neural network
- `loader`: data loader
- `lossfun`: loss function: (x::Array, y::Array) -> l::Real
"""
function fullbatch_metric(
    NN::AbstractLuxLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    loader,
	lossfun,
)
    N = 0
    L = 0f0

    SK = nothing # stats keys
    SV = nothing # stats values

    st = Lux.testmode(st)

    for batch in loader
        l, _, stats = lossfun(NN, p, st, batch)

        if isnothing(SK)
            SK = keys(stats)
        end

        n = numobs(batch)
        N += n

        # compute mean stats
        if isnothing(SV)
            SV = values(stats) .* n
        else
            SV = SV .+ values(stats) .* n
        end

        L += l * n
    end

    SV   = SV ./ N
    loss = L   / N
    stats = NamedTuple{SK}(SV)

    loss, stats
end

#===============================================================#
"""
$SIGNATURES

"""
function statistics(
    NN::AbstractLuxLayer,
    p::Union{NamedTuple, AbstractVector},
    st::NamedTuple,
    loader,
)
    st = Lux.testmode(st)

    N     = 0
    SUM   = 0f0
    VAR   = 0f0
	SUMSQ = 0f0
    ABSER = 0f0
    SQRER = 0f0
    MAXER = 0f0

    for (x, ŷ) in loader
        y, _ = NN(x, p, st)
        Δy = y - ŷ

		# data
        N     += length(ŷ)
        SUM   += sum(ŷ)
		SUMSQ += sum(abs2, ŷ)

		# errors
        ABSER += sum(abs , Δy)
        SQRER += sum(abs2, Δy)
        MAXER  = max(MAXER, norm(Δy, Inf))
    end

	MEAN1 = SUM / N
	MEAN2 = sqrt(SUMSQ / N)

    for (x, _) in loader
        y, _ = NN(x, p, st)
        VAR += sum(abs2, y .- MEAN1) / N
    end

    MSE    = SQRER / N
    RMSE   = sqrt(MSE)
    meanAE = ABSER / N
    maxAE  = MAXER         # TODO - seems off
	meanRE = MSE / MEAN2
    maxRE  = maxAE / MEAN2 # TODO - seems off
    R2     = 1f0 - MSE / (VAR + eps(Float32))
    cbound = compute_cbound(NN, p, st)

	str = ""
	str *= string("MSE (mean SQR error): ", round(MSE   ; sigdigits=8), "\n")
	str *= string("RMSE (Root MSE):      ", round(RMSE  ; sigdigits=8), "\n")
	str *= string("MAE (mean ABS error): ", round(meanAE; sigdigits=8), "\n")
	str *= string("maxAE (max ABS error) ", round(maxAE ; sigdigits=8), "\n")
	str *= string("mean RELATIVE error:  ", round(meanRE; sigdigits=8), "\n")
	str *= string("max  RELATIVE error:  ", round(maxRE ; sigdigits=8), "\n")
	str *= string("R² score:             ", round(R2    ; sigdigits=8), "\n")
	str *= string("Lipschitz bound:      ", round(cbound; sigdigits=8))

	(; MSE, RMSE, meanAE, maxAE, meanRE, maxRE, R2, cbound), str
end

#===============================================================#
function plot_training!(EPOCH, TIME, _LOSS, LOSS_, _MSE, MSE_, _MAE, MAE_)
    z = findall(iszero, EPOCH)

    # fix EPOCH to account for multiple training loops
    if length(z) > 1
            for i in 2:length(z)-1
            idx =  z[i]:z[i+1] - 1
            EPOCH[idx] .+= EPOCH[z[i] - 1]
        end
        EPOCH[z[end]:end] .+= EPOCH[z[end] - 1]
    end

    plt = plot(
        title = "Training Plot", yaxis = :log,
        xlabel = "Epochs", ylabel = "Loss",
        yticks = (@. 10.0^(-20:10)),
    )

    # (; ribbon = (lower, upper))
    plot!(plt, EPOCH, _LOSS, w = 2.0, s = :solid, c = :red , label = "LOSS (Train)")
    plot!(plt, EPOCH, LOSS_, w = 2.0, s = :solid, c = :blue, label = "LOSS (Test)")

    if !isempty(_MSE)
        plot!(plt, EPOCH, _MSE, w = 2.0, s = :dash, c = :magenta, label = "MSE (Train)")
        plot!(plt, EPOCH, MSE_, w = 2.0, s = :dash, c = :cyan   , label = "MSE (Test)")
    end

    if !isempty(_MAE)
        plot!(plt, EPOCH, _MAE, w = 2.0, s = :dot, c = :magenta, label = "MAE (Train)")
        plot!(plt, EPOCH, MAE_, w = 2.0, s = :dot, c = :cyan   , label = "MAE (Test)")
    end

    vline!(plt, EPOCH[z[2:end]], c = :black, w = 2.0, label = nothing)

    plt
end
#===============================================================#
