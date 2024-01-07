var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = GeometryLearning","category":"page"},{"location":"#GeometryLearning","page":"Home","title":"GeometryLearning","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for GeometryLearning.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [GeometryLearning]","category":"page"},{"location":"#GeometryLearning.Atten","page":"Home","title":"GeometryLearning.Atten","text":"Attention Layer\n\nsingle layer model with no nonlinearity (single head linear attention)\n\nu = NN(f) q = Wq * u k = Wk * u v = Wv * u\n\nv = activation(q * k') * u\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.CosineTransform","page":"Home","title":"GeometryLearning.CosineTransform","text":"struct CosineTransform{D} <: GeometryLearning.AbstractTransform{D}\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.Diag","page":"Home","title":"GeometryLearning.Diag","text":"Diagonal layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.FourierTransform","page":"Home","title":"GeometryLearning.FourierTransform","text":"struct FourierTransform{D} <: GeometryLearning.AbstractTransform{D}\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OpConv","page":"Home","title":"GeometryLearning.OpConv","text":"Neural Operator convolution layer\n\nTODO OpConv design consierations\n\ncreate AbstractTransform interface\ninnitialize params Wre, Wimag if eltype(Transform) isn't isreal\n\nso that eltype(params) is always real\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OpConv-Union{Tuple{D}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}}} where D","page":"Home","title":"GeometryLearning.OpConv","text":"OpConv(ch_in, ch_out, modes; init, transform)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.OpConvBilinear","page":"Home","title":"GeometryLearning.OpConvBilinear","text":"Neural Operator bilinear convolution layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OpConvBilinear-Union{Tuple{D}, Tuple{Tuple{AbstractArray, AbstractArray}, Any, NamedTuple}} where D","page":"Home","title":"GeometryLearning.OpConvBilinear","text":"Extend OpConv to accept two inputs\n\nLike Lux.Bilinear in modal space\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.SplitRows","page":"Home","title":"GeometryLearning.SplitRows","text":"SplitRows\n\nSplit rows of ND array, into Tuple of ND arrays.\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.AutoDecoder-Tuple{LuxCore.AbstractExplicitLayer, Int64, Int64}","page":"Home","title":"GeometryLearning.AutoDecoder","text":"AutoDecoder\n\nAssumes input is (xyz, idx) of sizes [D, K], [1, K] respectively\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.HyperDecoder-Tuple{LuxCore.AbstractExplicitLayer, LuxCore.AbstractExplicitLayer, Int64, Int64}","page":"Home","title":"GeometryLearning.HyperDecoder","text":"HyperDecoder\n\nAssumes input is (xyz, idx) of sizes [D, K], [1, K] respectively\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.ImplicitEncoderDecoder-Union{Tuple{D}, Tuple{LuxCore.AbstractExplicitLayer, LuxCore.AbstractExplicitLayer, Tuple{Vararg{Int64, D}}, Integer}} where D","page":"Home","title":"GeometryLearning.ImplicitEncoderDecoder","text":"implicit_encoder_decoder\n\nComposition of a (possibly convolutional) encoder and an implicit neural network decoder.\n\nThe input array [Nx, Ny, C, B] or [C, Nx, Ny, B] is expected to contain XYZ coordinates in the last dim entries of the channel dimension which is dictated by channel_dim. The number of channels in the input array must match encoder_width + D, where encoder_width is the expected input width of your encoder. The encoder network is expected to work with whatever channel_dim, encoder_channels you choose.\n\nNOTE: channel_dim is set to 1. So the assumption is [C, Nx, Ny, B]\n\nThe coordinates are split and the remaining channels are passed to encoder which compresses each [:, :, :, 1] slice into a latent vector of length L. The output of the encoder is of size [L, B].\n\nWith a compressed mapping of each image, we are ready to apply the decoder mapping. The decoder is an implicit neural network which expects as input the concatenation of the latent vector and a query point. The decoder returns the value of the target field at that point.\n\nThe decoder is usually a deep neural network and expects the channel dimension to be the leading dimension. The decoder expects input with size of leading dimension L+dim, and returns an array with leading size out_dim.\n\nHere, we feed it an array of size [L+2, Nx, Ny, B], where the input Npoints equal to (Nx, Ny,) is the number of training points in each trajectory.\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.OpKernel-Union{Tuple{D}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}, Any}} where D","page":"Home","title":"GeometryLearning.OpKernel","text":"OpKernel(ch_in, ch_out, modes; ...)\nOpKernel(\n    ch_in,\n    ch_out,\n    modes,\n    activation;\n    transform,\n    init,\n    use_bias\n)\n\n\naccept data in shape (C, X1, ..., Xd, B)\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.PermutedBatchNorm-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.PermutedBatchNorm","text":"PermutedBatchNorm(c, num_dims)\n\n\nAssumes channel dimension is 1\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.__opconv-Union{Tuple{D}, Tuple{Any, Any, Tuple{Vararg{Int64, D}}}} where D","page":"Home","title":"GeometryLearning.__opconv","text":"__opconv(x, transform, modes)\n\n\nAccepts x [C, N1...Nd, B]. Returns x̂ [C, M, B] where M = prod(modes)\n\nOperations\n\napply transform to N1...Nd:       [K1...Kd, C, B] <- [K1...Kd, C, B]\ntruncate (discard high-freq modes): [M1...Md, C, B] <- [K1...Kd, C, B] where modes == (M1...Md)\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning._ntimes-Union{Tuple{D}, Tuple{AbstractMatrix, Union{Int64, Tuple{Vararg{Int64, D}}}}} where D","page":"Home","title":"GeometryLearning._ntimes","text":", FUNCCACHEPREFER_NONE     _ntimes(x, (Nx, Ny)): x [L, B] –> [L, Nx, Ny, B]\n\nMake Nx ⋅ Ny copies of the first dimension and store it in the following dimensions. Works for any (Nx, Ny, ...).\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.callback-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.callback","text":"callback(\n    p,\n    st;\n    io,\n    _loss,\n    loss_,\n    _stats,\n    stats_,\n    STATS,\n    epoch,\n    nepoch\n)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.forwarddiff_deriv1-Union{Tuple{T}, Tuple{Any, Union{AbstractArray{T}, T}}} where T","page":"Home","title":"GeometryLearning.forwarddiff_deriv1","text":"Based on SparseDiffTools.auto_jacvec\n\nMWE:\n\nf = x -> exp.(x)\nf = x -> x .^ 2\nx = [1.0, 2.0, 3.0, 4.0]\n\nforwarddiff_deriv1(f, x)\nforwarddiff_deriv2(f, x)\nforwarddiff_deriv4(f, x)\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.fullbatch_metric","page":"Home","title":"GeometryLearning.fullbatch_metric","text":"fullbatch_metric(NN, p, st, loader, lossfun, ismean) -> l\n\nOnly for callbacks. Enforce this by setting Lux.testmode\n\nNN, p, st: neural network\nloader: data loader\nlossfun: loss function: (x::Array, y::Array) -> l::Real\nismean: lossfun takes a mean\n\n\n\n\n\n","category":"function"},{"location":"#GeometryLearning.l2reg-Tuple{Any, Real}","page":"Home","title":"GeometryLearning.l2reg","text":"l2reg(lossfun, λ)(NN, p, st, batch) -> l, st, stats\n\nL2-Regularization\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.linear_nonlinear","page":"Home","title":"GeometryLearning.linear_nonlinear","text":"linear_nonlinear(split, nonlin, linear, bilinear)\nlinear_nonlinear(split, nonlin, linear, bilinear, project)\n\n\nif you have linear dependence on x1, and nonlinear on x2, then\n\nx1 → nonlin → y1 ↘\n                  bilinear → project → z\nx2 → linear → y2 ↗\n\nArguments\n\nCall nonlin as nonlin(x1, p, st)\nCall linear as linear(x2, p, st)\nCall bilin  as bilin((y1, y2), p, st)\n\n\n\n\n\n","category":"function"},{"location":"#GeometryLearning.mae-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.mae","text":"mae(ypred, ytrue) -> l\nmae(NN, p, st, batch) -> l, st, stats\n\nMean squared error\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.make_minconfig-NTuple{6, Any}","page":"Home","title":"GeometryLearning.make_minconfig","text":"early stopping based on mini-batch loss from test set https://github.com/jeffheaton/appdeeplearning/blob/main/t81558class034earlystop.ipynb\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.makecallback-Tuple{LuxCore.AbstractExplicitLayer, Union{CUDA.CuIterator, MLUtils.DataLoader}, Union{CUDA.CuIterator, MLUtils.DataLoader}, Any}","page":"Home","title":"GeometryLearning.makecallback","text":"makecallback(\n    NN,\n    _loader,\n    loader_,\n    lossfun;\n    STATS,\n    stats,\n    io\n)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.minibatch_metric-NTuple{5, Any}","page":"Home","title":"GeometryLearning.minibatch_metric","text":"minibatch_metric(NN, p, st, loader, lossfun, ismean) -> l\n\nOnly for callbacks. Enforce this by setting Lux.testmode\n\nNN, p, st: neural network\nloader: data loader\nlossfun: loss function: (x::Array, y::Array) -> l::Real\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.mse-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.mse","text":"mse(ypred, ytrue) -> l\nmse(NN, p, st, batch) -> l, st, stats\n\nMean squared error\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.opconv__-Union{Tuple{D}, Tuple{Any, Any, Tuple{Vararg{Int64, D}}, Any, Any}} where D","page":"Home","title":"GeometryLearning.opconv__","text":"opconv__(ŷ_tr, transform, modes, Ks, Ns)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.opconv_wt-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.opconv_wt","text":"opconv_wt(x, W)\n\n\nApply pointwise linear transform in mode space, i.e. no mode-mixing. Unique linear transform for each mode.\n\nOperations\n\nreshape: [Ci, M, B] <- [Ci, M1...Md, B] where M = prod(M1...Md)\napply weight\nreshape: [Co, M1...Md, B] <- [Co, M, B]\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.optimize","page":"Home","title":"GeometryLearning.optimize","text":"optimize(opt, NN, p, st, nepoch, _loader, loader_; ...)\noptimize(\n    opt,\n    NN,\n    p,\n    st,\n    nepoch,\n    _loader,\n    loader_,\n    __loader;\n    lossfun,\n    opt_st,\n    cb,\n    io,\n    early_stopping,\n    patience,\n    schedule,\n    kwargs...\n)\n\n\nTrain parameters p to minimize loss using optimization strategy opt.\n\nArguments\n\nLoss signature: loss(p, st) -> y, st\nCallback signature: cb(p, st epoch, nepoch) -> nothing \n\n\n\n\n\n","category":"function"},{"location":"#GeometryLearning.optimize-2","page":"Home","title":"GeometryLearning.optimize","text":"references\n\nhttps://docs.sciml.ai/Optimization/stable/tutorials/minibatch/ https://lux.csail.mit.edu/dev/tutorials/advanced/1_GravitationalWaveForm#training-the-neural-network\n\n\n\n\n\n","category":"function"},{"location":"#GeometryLearning.plot_1D_surrogate_steady-Tuple{CalculustCore.Spaces.AbstractSpace{<:Any, 1}, Tuple{Any, Any}, Tuple{Any, Any}, LuxCore.AbstractExplicitLayer, Any, Any}","page":"Home","title":"GeometryLearning.plot_1D_surrogate_steady","text":"plot_1D_surrogate_steady(\n    V,\n    _data,\n    data_,\n    NN,\n    p,\n    st;\n    nsamples,\n    dir,\n    format\n)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.pnorm-Tuple{Real}","page":"Home","title":"GeometryLearning.pnorm","text":"pnorm(p)(y, ŷ) -> l\npnorm(p)(NN, p, st, batch) -> l, st, stats\n\nP-Norm\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.rsquare-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.rsquare","text":"rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)\n\nCalculuate r2 (coefficient of determination) score.\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.statistics-Tuple{LuxCore.AbstractExplicitLayer, Union{NamedTuple, AbstractVector}, NamedTuple, Union{CUDA.CuIterator, MLUtils.DataLoader}}","page":"Home","title":"GeometryLearning.statistics","text":"statistics(NN, p, st, loader; io)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.train_model-Union{Tuple{M}, Tuple{LuxCore.AbstractExplicitLayer, Tuple{Any, Any}}, Tuple{LuxCore.AbstractExplicitLayer, Tuple{Any, Any}, Tuple{Any, Any}}} where M","page":"Home","title":"GeometryLearning.train_model","text":"train_model(NN, _data; ...)\ntrain_model(\n    NN,\n    _data,\n    data_;\n    rng,\n    _batchsize,\n    batchsize_,\n    __batchsize,\n    opts,\n    nepochs,\n    schedules,\n    dir,\n    name,\n    metadata,\n    io,\n    p,\n    st,\n    lossfun,\n    device,\n    early_stopping,\n    patience\n)\n\n\nArguments\n\nNN: Lux neural network\n_data: training data as (x, y). x may be an AbstractArray or a tuple of arrays\ndata_: testing data (same requirement as `_data)\n\nKeyword Arguments\n\nrng: random nunmber generator\n_batchsize/batchsize_: train/test batch size\nopts/nepochs: NTuple of optimizers, # epochs per optimizer\ncbstep: prompt callback function every cbstep epochs\ndir/name: directory to save model, plots, model name\nio: io for printing stats\np/st: initial model parameter, state. if nothing, initialized with Lux.setup(rng, NN)\n\n\n\n\n\n","category":"method"}]
}
