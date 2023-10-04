var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = GeometryLearning","category":"page"},{"location":"#GeometryLearning","page":"Home","title":"GeometryLearning","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for GeometryLearning.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [GeometryLearning]","category":"page"},{"location":"#GeometryLearning.Atten","page":"Home","title":"GeometryLearning.Atten","text":"Attention Layer\n\nsingle layer model with no nonlinearity (single head linear attention)\n\nu = NN(f) q = Wq * u k = Wk * u v = Wv * u\n\nv = activation(q * k') * u\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.CosineTransform","page":"Home","title":"GeometryLearning.CosineTransform","text":"struct CosineTransform{D} <: GeometryLearning.AbstractTransform{D}\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.Diag","page":"Home","title":"GeometryLearning.Diag","text":"Diagonal layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.FourierTransform","page":"Home","title":"GeometryLearning.FourierTransform","text":"struct FourierTransform{D} <: GeometryLearning.AbstractTransform{D}\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OpConv","page":"Home","title":"GeometryLearning.OpConv","text":"Neural Operator convolution layer\n\nTODO OpConv design consierations\n\ncreate AbstractTransform interface\ninnitialize params Wre, Wimag if eltype(Transform) isn't isreal\n\nso that eltype(params) is always real\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OpConv-Union{Tuple{D}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}}} where D","page":"Home","title":"GeometryLearning.OpConv","text":"OpConv(ch_in, ch_out, modes; init, transform)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.OpConvBilinear","page":"Home","title":"GeometryLearning.OpConvBilinear","text":"Neural Operator bilinear convolution layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OpConvBilinear-Union{Tuple{D}, Tuple{Tuple{AbstractArray, AbstractArray}, Any, NamedTuple}} where D","page":"Home","title":"GeometryLearning.OpConvBilinear","text":"Extend OpConv to accept two inputs\n\nLike Lux.Bilinear in modal space\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.SplitRows","page":"Home","title":"GeometryLearning.SplitRows","text":"SplitRows\n\nSplit rows of ND array, into Tuple of ND arrays.\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OpKernel-Union{Tuple{D}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}, Any}} where D","page":"Home","title":"GeometryLearning.OpKernel","text":"OpKernel(ch_in, ch_out, modes)\nOpKernel(\n    ch_in,\n    ch_out,\n    modes,\n    activation;\n    transform,\n    init,\n    use_bias\n)\n\n\naccept data in shape (C, X1, ..., Xd, B)\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.PermutedBatchNorm-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.PermutedBatchNorm","text":"PermutedBatchNorm(c, num_dims)\n\n\nAssumes channel dimension is 1\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.__opconv-Union{Tuple{D}, Tuple{Any, Any, Tuple{Vararg{Int64, D}}}} where D","page":"Home","title":"GeometryLearning.__opconv","text":"__opconv(x, transform, modes)\n\n\nAccepts x [C, N1...Nd, B]. Returns x̂ [C, M, B] where M = prod(modes)\n\nOperations\n\napply transform to N1...Nd:       [K1...Kd, C, B] <- [K1...Kd, C, B]\ntruncate (discard high-freq modes): [M1...Md, C, B] <- [K1...Kd, C, B] where modes == (M1...Md)\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.callback-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.callback","text":"callback(\n    p,\n    st;\n    io,\n    _loss,\n    _LOSS,\n    _stats,\n    loss_,\n    LOSS_,\n    stats_,\n    epoch,\n    step,\n    nepoch,\n    EPOCH\n)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.linear_nonlinear","page":"Home","title":"GeometryLearning.linear_nonlinear","text":"linear_nonlinear(split, nonlin, linear, bilinear)\nlinear_nonlinear(split, nonlin, linear, bilinear, project)\n\n\nif you have linear dependence on x1, and nonlinear on x2, then\n\nx1 → nonlin → y1 ↘\n                  bilinear → project → z\nx2 → linear → y2 ↗\n\nArguments\n\nCall nonlin as nonlin(x1, p, st)\nCall linear as linear(x2, p, st)\nCall bilin  as bilin((y1, y2), p, st)\n\n\n\n\n\n","category":"function"},{"location":"#GeometryLearning.mse-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.mse","text":"mse(ypred, ytrue)\n\nMean squared error\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.mse_norm-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.mse_norm","text":"mse_norm(ypred, ytrue)\n\nMean squared error where each trajectory is normalized to have unit norm trajectories of size\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.opconv__-Union{Tuple{D}, Tuple{Any, Any, Tuple{Vararg{Int64, D}}, Any, Any}} where D","page":"Home","title":"GeometryLearning.opconv__","text":"opconv__(ŷ_tr, transform, modes, Ks, Ns)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.opconv_wt-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.opconv_wt","text":"opconv_wt(x, W)\n\n\nApply pointwise linear transform in mode space, i.e. no mode-mixing. Unique linear transform for each mode.\n\nOperations\n\nreshape: [Ci, M, B] <- [Ci, M1...Md, B] where M = prod(M1...Md)\napply weight\nreshape: [Co, M1...Md, B] <- [Co, M, B]\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.optimize-NTuple{6, Any}","page":"Home","title":"GeometryLearning.optimize","text":"optimize(\n    NN,\n    p,\n    st,\n    _loader,\n    loader_,\n    nepochs;\n    lossfun,\n    opt,\n    opt_st,\n    cb_batch,\n    cb_epoch,\n    io,\n    patience\n)\n\n\nTrain parameters p to minimize loss using optimization strategy opt.\n\nArguments\n\nLoss signature: loss(p, st) -> y, st\nCallback signature: cb(p, st epoch, nepochs) -> nothing \n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.plot_1D_surrogate_steady-Tuple{CalculustCore.Spaces.AbstractSpace{<:Any, 1}, Tuple{Any, Any}, Tuple{Any, Any}, LuxCore.AbstractExplicitLayer, Any, Any}","page":"Home","title":"GeometryLearning.plot_1D_surrogate_steady","text":"plot_1D_surrogate_steady(\n    V,\n    _data,\n    data_,\n    NN,\n    p,\n    st;\n    nsamples,\n    dir,\n    format\n)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.rel_mse","page":"Home","title":"GeometryLearning.rel_mse","text":"rel_mse(ypred, ytrue)\n\nNormalize each trajectory to have unit norm, then take relative error\n\n\n\n\n\n","category":"function"},{"location":"#GeometryLearning.rsquare-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.rsquare","text":"rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)\n\nCalculuate r2 (coefficient of determination) score.\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.statistics-Tuple{LuxCore.AbstractExplicitLayer, Any, Any, Any}","page":"Home","title":"GeometryLearning.statistics","text":"statistics(NN, p, st, loader; mode, io)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.train_model","page":"Home","title":"GeometryLearning.train_model","text":"train_model(rng, NN, _data, data_, V)\ntrain_model(\n    rng,\n    NN,\n    _data,\n    data_,\n    V,\n    opt;\n    batchsize,\n    batchsize_,\n    learning_rates,\n    nepochs,\n    cbstep,\n    dir,\n    name,\n    nsamples,\n    io,\n    p,\n    st,\n    lossfun,\n    device,\n    make_plots,\n    patience,\n    metadata\n)\n\n\nArguments\n\nV: function space\nNN: Lux neural network\n_data: training data as (x, y). x may be an AbstractArray or a tuple of arrays\ndata_: testing data (same requirement as `_data)\n\nData arrays, x, y must be AbstractMatrix, or AbstractArray{T,3}. In the former case, the dimensions are assumed to be (points, batch), and (chs, points, batch) in the latter, where the points dimension is equal to length(V).\n\nKeyword Arguments\n\nopts: NTuple of optimizers\nnepochs: Number of epochs for each optimization cycle\ncbstep: prompt callback function every cbstep epochs\ndir: directory to save model, plots\nio: io for printing stats\n\n\n\n\n\n","category":"function"}]
}
