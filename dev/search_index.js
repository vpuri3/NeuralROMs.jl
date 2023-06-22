var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = GeometryLearning","category":"page"},{"location":"#GeometryLearning","page":"Home","title":"GeometryLearning","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for GeometryLearning.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [GeometryLearning]","category":"page"},{"location":"#GeometryLearning.Atten","page":"Home","title":"GeometryLearning.Atten","text":"Attention Layer\n\nsingle layer model with no nonlinearity (single head linear attention)\n\nu = NN(f) q = Wq * u k = Wk * u v = Wv * u\n\nv = activation(q * k') * u\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.Diag","page":"Home","title":"GeometryLearning.Diag","text":"Diagonal layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.FusionLayer","page":"Home","title":"GeometryLearning.FusionLayer","text":"make subtype of Lux.AbstractExplicitContainerLayer Basically PairwiseFusionLayer where connection is also a layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OperatorConv","page":"Home","title":"GeometryLearning.OperatorConv","text":"Neural Operator convolution layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.OperatorKernel-Union{Tuple{D}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}}} where D","page":"Home","title":"GeometryLearning.OperatorKernel","text":"OperatorKernel(\n    in_dims,\n    out_dims,\n    modes;\n    activation,\n    transform,\n    init\n)\n\n\naccept data in shape (C, X1, ..., Xd, B)\n\nTODO - Extend this to accept two inputs (optionally) For example if you have linear dependence on f, and nonlinear on ν, then\n\nν -> lifting -> OpKernel (nl) -> ... -> OpKernel -> project (lin) -> out\n                                           /||\n                                           forc\n\nwhere OpKernel(ν, f) will perform something like ν̂' * W * f̂ in fourier space.\n\noptions: Bilinear, Branch/ Parallel, PairwiseFusion\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.callback-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.callback","text":"callback(\n    p,\n    st;\n    io,\n    _loss,\n    _LOSS,\n    _stats,\n    loss_,\n    LOSS_,\n    stats_,\n    iter,\n    step,\n    maxiter,\n    ITER\n)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.model_setup-Tuple{LuxCore.AbstractExplicitLayer, Any}","page":"Home","title":"GeometryLearning.model_setup","text":"model_setup(NN, data)\n\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.mse-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.mse","text":"mse(ypred, ytrue)\n\nMean squared error\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.optimize-NTuple{4, Any}","page":"Home","title":"GeometryLearning.optimize","text":"optimize(loss, p, st, maxiter; opt, cb)\n\n\nTrain parameters p to minimize loss using optimization strategy opt.\n\nArguments\n\nLoss signature: loss(p, st) -> y, st\nCallback signature: cb(p, st iter, maxiter) -> nothing \n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.rsquare-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.rsquare","text":"rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)\n\nCalculuate r2 (coefficient of determination) score.\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.train_model-Union{Tuple{N}, Tuple{Random.AbstractRNG, LuxCore.AbstractExplicitLayer, Tuple{Any, Any}, Tuple{Any, Any}, CalculustCore.Spaces.AbstractSpace}} where N","page":"Home","title":"GeometryLearning.train_model","text":"train_model(\n    rng,\n    NN,\n    _data,\n    data_,\n    V;\n    opts,\n    maxiters,\n    cbstep,\n    dir,\n    name,\n    io\n)\n\n\nArguments\n\nV: function space\nNN: Lux neural network\n_data: training data as (x, y)\ndata_: testing  data as (x, y)\n\nData arrays, x, y must be AbstractMatrix, or AbstractArray{T,3}. In the former case, the dimensions are assumed to be (points, batch), and (chs, points, batch) in the latter, where the points dimension is equal to length(V).\n\nKeyword Arguments\n\nopts: NTuple of optimizers\nmaxiters: Number of iterations for each optimization cycle\ncbstep: prompt callback function every cbstep epochs\ndir: directory to save model, plots\nio: io for printing stats\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.visualize-Tuple{CalculustCore.Spaces.AbstractSpace{<:Any, 1}, Tuple{Any, Any}, Tuple{Any, Any}, LuxCore.AbstractExplicitLayer, Any, Any}","page":"Home","title":"GeometryLearning.visualize","text":"visualize(V, _data, data_, NN, p, st; nsamples)\n\n\n\n\n\n\n","category":"method"}]
}
