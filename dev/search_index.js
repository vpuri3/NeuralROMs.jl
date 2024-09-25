var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = NeuralROMs","category":"page"},{"location":"#NeuralROMs.jl","page":"Home","title":"NeuralROMs.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This repository implements machine learning (ML) based reduced order models (ROMs). Specifically, we introduce smooth neural field ROM (SNF-ROM) for solving advection dominated PDE problems.","category":"page"},{"location":"#Smooth-neural-field-ROM","page":"Home","title":"Smooth neural field ROM","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields  Vedant Puri, Aviral Prakash, Levent Burak Kara, Yongjie Jessica Zhang  Project page / Paper / Code / Slides / Talk","category":"page"},{"location":"#Abstract","page":"Home","title":"Abstract","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Reduced order modeling lowers the computational cost of solving PDEs by learning a low-dimensional spatial representation from data and dynamically evolving these representations using manifold projections of the governing equations. The commonly used linear subspace reduced-order models (ROMs) are often suboptimal for problems with a slow decay of Kolmogorov n-width, such as advection-dominated fluid flows at high Reynolds numbers. There has been a growing interest in nonlinear ROMs that use state-of-the-art representation learning techniques to accurately capture such phenomena with fewer degrees of freedom. We propose smooth neural field ROM (SNF-ROM), a nonlinear reduced order modeling framework that combines grid-free reduced representations with Galerkin projection. The SNF-ROM architecture constrains the learned ROM trajectories to a smoothly varying path, which proves beneficial in the dynamics evaluation when the reduced manifold is traversed in accordance with the governing PDEs. Furthermore, we devise robust regularization schemes to ensure the learned neural fields are smooth and differentiable. This allows us to compute physics-based dynamics of the reduced system nonintrusively with automatic differentiation and evolve the reduced system with classical time-integrators. SNF-ROM leads to fast offline training as well as enhanced accuracy and stability during the online dynamics evaluation. Numerical experiments reveal that SNF-ROM is able to accelerate the full-order computation by up to 199x. We demonstrate the efficacy of SNF-ROM on a range of advection-dominated linear and nonlinear PDE problems where we consistently outperform state-of-the-art ROMs.","category":"page"},{"location":"#Method","page":"Home","title":"Method","text":"","category":"section"},{"location":"#Offline-stage","page":"Home","title":"Offline stage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Capture-2024-05-28-171751)","category":"page"},{"location":"#Online-stage","page":"Home","title":"Online stage","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"(Image: Screenshot 2024-05-28 at 5 18 25 PM)","category":"page"},{"location":"","page":"Home","title":"Home","text":"ROMs are hybrid physics and databased methods that decouple the computation into two stages: an expensive offline stage and a cheap online stage. In the offline stage, a low-dimensional spatial representation is learned from simulation data by projecting the solution field snapshots onto a low-dimensional manifold that can faithfully capture the relevant features in the dataset. The online stage then involves evaluating the model at new parametric points by time-evolving the learned spatial representation following the governing PDE system with classical time integrators.","category":"page"},{"location":"","page":"Home","title":"Home","text":"SNF-ROM is a continuous neural field ROM that is nonintrusive by construction and eliminates the need for a fixed grid structure in the underlying data and the identification of associated spatial discretization for dynamics evaluation. There are two important features of SNF-ROM:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Constrained manifold formulation: SNF-ROM restricts the reduced trajectories to follow a regular, smoothly varying path. This behavior is achieved by directly modeling the ROM state vector as a simple, learnable function of problem parameters and time. Our numerical experiments reveal that this feature allows for larger time steps in the dynamics evaluation, where the reduced manifold is traversed in accordance with the governing PDEs.\nNeural field regularization: We formulate a robust network regularization approach encouraging smoothness in the learned neural fields. Consequently, the spatial derivatives of SNF representations match the true derivatives of the underlying signal. This feature allows us to calculate accurate spatial derivatives with the highly efficient forward mode automatic differentiation (AD) technique. Our studies indicate that precisely capturing spatial derivatives is crucial for an accurate dynamics prediction.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The confluence of these two features produces desirable effects on the dynamics evaluation, such as greater accuracy, robustness to hyperparameter choice, and robustness to numerical perturbations.","category":"page"},{"location":"#Citation","page":"Home","title":"Citation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"@misc{\n    puri2024snfrom,\n    title={{SNF-ROM}: {P}rojection-based nonlinear reduced order modeling with smooth neural fields},\n    author={Vedant Puri and Aviral Prakash and Levent Burak Kara and Yongjie Jessica Zhang},\n    year={2024},\n    eprint={2405.14890},\n    archivePrefix={arXiv},\n    primaryClass={physics.flu-dyn},\n}","category":"page"},{"location":"#Acknowledgements","page":"Home","title":"Acknowledgements","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The authors would like to acknowledge the support from the National Science Foundation (NSF) grant CMMI-1953323 and PA Manufacturing Fellows Initiative for the funds used towards this project. The research in this paper was also sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-20-2-0175. The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.","category":"page"},{"location":"API/","page":"API","title":"API","text":"","category":"page"},{"location":"API/","page":"API","title":"API","text":"Modules = [NeuralROMs]","category":"page"},{"location":"API/#NeuralROMs.CosineTransform","page":"API","title":"NeuralROMs.CosineTransform","text":"struct CosineTransform{D} <: NeuralROMs.AbstractTransform{D}\n\n\n\n\n\n","category":"type"},{"location":"API/#NeuralROMs.FourierTransform","page":"API","title":"NeuralROMs.FourierTransform","text":"struct FourierTransform{D} <: NeuralROMs.AbstractTransform{D}\n\n\n\n\n\n","category":"type"},{"location":"API/#NeuralROMs.GalerkinProjection","page":"API","title":"NeuralROMs.GalerkinProjection","text":"GalerkinProjection\n\noriginal: u' = f(u, t) ROM map : u = g(ũ)\n\n⟹  J(ũ) *  ũ' = f(ũ, t)\n\n⟹  ũ' = pinv(J)  (ũ) * f(ũ, t)\n\nsolve with timestepper ⟹  ũ' = f̃(ũ, t)\n\ne.g. (J*u)_n+1 - (J*u)_n = Δt * (f_n + f_n-1 + ...)\n\n\n\n\n\n","category":"type"},{"location":"API/#NeuralROMs.OpConv","page":"API","title":"NeuralROMs.OpConv","text":"Neural Operator convolution layer\n\nTODO OpConv design consierations\n\ncreate AbstractTransform interface\ninnitialize params Wre, Wimag if eltype(Transform) isn't isreal\n\nso that eltype(params) is always real\n\n\n\n\n\n","category":"type"},{"location":"API/#NeuralROMs.OpConv-Union{Tuple{D}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}}} where D","page":"API","title":"NeuralROMs.OpConv","text":"OpConv(ch_in, ch_out, modes; init, transform)\n\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.OpConvBilinear","page":"API","title":"NeuralROMs.OpConvBilinear","text":"Neural Operator bilinear convolution layer\n\n\n\n\n\n","category":"type"},{"location":"API/#NeuralROMs.OpConvBilinear-Union{Tuple{D}, Tuple{Tuple{AbstractArray, AbstractArray}, Any, NamedTuple}} where D","page":"API","title":"NeuralROMs.OpConvBilinear","text":"Extend OpConv to accept two inputs\n\nLike Lux.Bilinear in modal space\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.PeriodicLayer","page":"API","title":"NeuralROMs.PeriodicLayer","text":"x -> sin(π⋅x/L)\n\nWorks when input is symmetric around 0, i.e., x ∈ [-1, 1). If working with something like [0, 1], use cosines instead.\n\n\n\n\n\n","category":"type"},{"location":"API/#NeuralROMs.SplitRows","page":"API","title":"NeuralROMs.SplitRows","text":"SplitRows\n\nSplit rows of ND array, into Tuple of ND arrays.\n\n\n\n\n\n","category":"type"},{"location":"API/#NeuralROMs.AutoDecoder-Tuple{LuxCore.AbstractExplicitLayer, Int64, Int64}","page":"API","title":"NeuralROMs.AutoDecoder","text":"AutoDecoder\n\nAssumes input is (xyz, idx) of sizes [in_dim, K], [1, K] respectively\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.FlatDecoder-Tuple{LuxCore.AbstractExplicitLayer, LuxCore.AbstractExplicitLayer}","page":"API","title":"NeuralROMs.FlatDecoder","text":"FlatDecoder\n\nInput: (x, param) of sizes [x_dim, K], and [p_dim, K] respectively. Output: solution field u of size [out_dim, K].\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.HyperDecoder-Tuple{LuxCore.AbstractExplicitLayer, LuxCore.AbstractExplicitLayer, Int64, Int64}","page":"API","title":"NeuralROMs.HyperDecoder","text":"HyperDecoder\n\nAssumes input is (xyz, idx) of sizes [D, K], [1, K] respectively\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.ImplicitEncoderDecoder-Union{Tuple{D}, Tuple{LuxCore.AbstractExplicitLayer, LuxCore.AbstractExplicitLayer, Tuple{Vararg{Integer, D}}, Integer}} where D","page":"API","title":"NeuralROMs.ImplicitEncoderDecoder","text":"ImplicitEncoderDecoder\n\nComposition of a (possibly convolutional) encoder and an implicit neural network decoder.\n\nThe input array [Nx, Ny, C, B] or [C, Nx, Ny, B] is expected to contain XYZ coordinates in the last dim entries of the channel dimension which is dictated by channel_dim. The number of channels in the input array must match encoder_width + D, where encoder_width is the expected input width of your encoder. The encoder network is expected to work with whatever channel_dim, encoder_channels you choose.\n\nNOTE: channel_dim is set to 1. So the assumption is [C, Nx, Ny, B]\n\nThe coordinates are split and the remaining channels are passed to encoder which compresses each [:, :, :, 1] slice into a latent vector of length L. The output of the encoder is of size [L, B].\n\nWith a compressed representation of each image, we are ready to apply the decoder mapping. The decoder is an implicit neural network which expects as input the concatenation of the latent vector and a query point. The decoder returns the value of the target field at that point.\n\nThe decoder is usually a deep neural network and expects the channel dimension to be the leading dimension. The decoder expects input with size of leading dimension L+dim, and returns an array with leading size out_dim.\n\nHere, we feed it an array of size [L+2, Nx, Ny, B], where the input Npoints equal to (Nx, Ny,) is the number of training points in each trajectory.\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.OpKernel-Union{Tuple{D}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}}, Tuple{Int64, Int64, Tuple{Vararg{Int64, D}}, Any}} where D","page":"API","title":"NeuralROMs.OpKernel","text":"OpKernel(ch_in, ch_out, modes; ...)\nOpKernel(\n    ch_in,\n    ch_out,\n    modes,\n    activation;\n    transform,\n    init,\n    use_bias\n)\n\n\naccept data in shape (C, X1, ..., Xd, B)\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.PSNR-Tuple{AbstractArray, AbstractArray, Real}","page":"API","title":"NeuralROMs.PSNR","text":"PSNR(y, ŷ, maxval) --> -10 * log10(mse(y, ŷ) / maxval^2)\n\nPeak signal to noise ratio\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.PSNR-Tuple{Real}","page":"API","title":"NeuralROMs.PSNR","text":"PSNR(maxval)(NN, p, st, batch) --> PSNR\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.PermutedBatchNorm-Tuple{Any, Any}","page":"API","title":"NeuralROMs.PermutedBatchNorm","text":"PermutedBatchNorm(c, num_dims)\n\n\nAssumes channel dimension is 1\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.__opconv-Union{Tuple{D}, Tuple{Any, Any, Tuple{Vararg{Int64, D}}}} where D","page":"API","title":"NeuralROMs.__opconv","text":"__opconv(x, transform, modes)\n\n\nAccepts x [C, N1...Nd, B]. Returns x̂ [C, M, B] where M = prod(modes)\n\nOperations\n\napply transform to N1...Nd:       [K1...Kd, C, B] <- [K1...Kd, C, B]\ntruncate (discard high-freq modes): [M1...Md, C, B] <- [K1...Kd, C, B] where modes == (M1...Md)\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs._ntimes-Union{Tuple{D}, Tuple{AbstractMatrix, Union{Int64, Tuple{Vararg{Int64, D}}}}} where D","page":"API","title":"NeuralROMs._ntimes","text":", FUNCCACHEPREFER_NONE     _ntimes(x, (Nx, Ny)): x [L, B] –> [L, Nx, Ny, B]\n\nMake Nx ⋅ Ny copies of the first dimension and store it in the following dimensions. Works for any (Nx, Ny, ...).\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.callback-Tuple{Any, Any}","page":"API","title":"NeuralROMs.callback","text":"callback(\n    p,\n    st;\n    io,\n    _loss,\n    loss_,\n    _printstatistics,\n    printstatistics_,\n    STATS,\n    epoch,\n    nepoch,\n    notestdata\n)\n\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.codereg_autodecoder-Tuple{Any, Real}","page":"API","title":"NeuralROMs.codereg_autodecoder","text":"codereg_autodecoder(lossfun, σ; property)(NN, p, st, batch) -> l, st, stats\n\ncode regularized loss: lossfun(..) + 1/σ² ||ũ||₂²\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.elasticreg-Tuple{Any, Real, Real}","page":"API","title":"NeuralROMs.elasticreg","text":"elasticreg(lossfun, λ1, λ2)(NN, p, st, batch) -> l, st, stats\n\nElastic Regularization (L1 + L2)\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.forwarddiff_deriv1-Union{Tuple{T}, Tuple{Any, Union{AbstractArray{T}, T}}} where T","page":"API","title":"NeuralROMs.forwarddiff_deriv1","text":"Based on SparseDiffTools.auto_jacvec\n\nMWE:\n\nf = x -> exp.(x)\nf = x -> x .^ 2\nx = [1.0, 2.0, 3.0, 4.0]\n\nforwarddiff_deriv1(f, x)\nforwarddiff_deriv2(f, x)\nforwarddiff_deriv4(f, x)\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.fullbatch_metric-Tuple{LuxCore.AbstractExplicitLayer, Union{NamedTuple, AbstractVector}, NamedTuple, Union{CUDA.CuIterator, MLUtils.DataLoader}, Any}","page":"API","title":"NeuralROMs.fullbatch_metric","text":"fullbatch_metric(NN, p, st, loader, lossfun, ismean) -> l\n\nOnly for callbacks. Enforce this by setting Lux.testmode\n\nNN, p, st: neural network\nloader: data loader\nlossfun: loss function: (x::Array, y::Array) -> l::Real\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.get_state-Tuple{TimeIntegrator}","page":"API","title":"NeuralROMs.get_state","text":"returns t, p, u, f, f̃\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.interp_cubic-Tuple{Any, Any, Any}","page":"API","title":"NeuralROMs.interp_cubic","text":"Cubic hermite interpolation \n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.linear_nonlinear","page":"API","title":"NeuralROMs.linear_nonlinear","text":"linear_nonlinear(split, nonlin, linear, bilinear)\nlinear_nonlinear(split, nonlin, linear, bilinear, project)\n\n\nif you have linear dependence on x1, and nonlinear on x2, then\n\nx1 → nonlin → y1 ↘\n                  bilinear → project → z\nx2 → linear → y2 ↗\n\nArguments\n\nCall nonlin as nonlin(x1, p, st)\nCall linear as linear(x2, p, st)\nCall bilin  as bilin((y1, y2), p, st)\n\n\n\n\n\n","category":"function"},{"location":"API/#NeuralROMs.mae-Tuple{Any, Any}","page":"API","title":"NeuralROMs.mae","text":"mae(ypred, ytrue) -> l\nmae(NN, p, st, batch) -> l, st, stats\n\nMean squared error\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.mae_clamped-Tuple{Real}","page":"API","title":"NeuralROMs.mae_clamped","text":"mae_clamped(δ)(NN, p, st, batch) -> l, st, stats\n\nClamped mean absolute error\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.make_minconfig-NTuple{6, Any}","page":"API","title":"NeuralROMs.make_minconfig","text":"early stopping based on mini-batch loss from test set https://github.com/jeffheaton/appdeeplearning/blob/main/t81558class034earlystop.ipynb\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.makecallback-Tuple{LuxCore.AbstractExplicitLayer, Union{CUDA.CuIterator, MLUtils.DataLoader}, Union{CUDA.CuIterator, MLUtils.DataLoader}, Any}","page":"API","title":"NeuralROMs.makecallback","text":"makecallback(\n    NN,\n    _loader,\n    loader_,\n    lossfun;\n    STATS,\n    stats,\n    io,\n    cb_epoch,\n    notestdata\n)\n\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.merge_TanhKernel1D-Tuple{Any}","page":"API","title":"NeuralROMs.merge_TanhKernel1D","text":"Input should be: ((NN1, p1, st1), (NN2, p2, st2), ...) where each NN is a TanhKernel1D and p, st its parameters and states\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.mse-Tuple{Any, Any}","page":"API","title":"NeuralROMs.mse","text":"mse(ypred, ytrue) -> l\nmse(NN, p, st, batch) -> l, st, stats\n\nMean squared error\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.normalize_t-Union{Tuple{AbstractVector{T}}, Tuple{T}, Tuple{AbstractVector{T}, Any}} where T","page":"API","title":"NeuralROMs.normalize_t","text":"t ∈ [0, T] Input size [Ntime].\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.normalize_u-Union{Tuple{AbstractArray{T, N}}, Tuple{N}, Tuple{T}, Tuple{AbstractArray{T, N}, Any}} where {T, N}","page":"API","title":"NeuralROMs.normalize_u","text":"Input size [out_dim, ...]\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.opconv__-Union{Tuple{D}, Tuple{Any, Any, Tuple{Vararg{Int64, D}}, Any, Any}} where D","page":"API","title":"NeuralROMs.opconv__","text":"opconv__(ŷ_tr, transform, modes, Ks, Ns)\n\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.opconv_wt-Tuple{Any, Any}","page":"API","title":"NeuralROMs.opconv_wt","text":"opconv_wt(x, W)\n\n\nApply pointwise linear transform in mode space, i.e. no mode-mixing. Unique linear transform for each mode.\n\nOperations\n\nreshape: [Ci, M, B] <- [Ci, M1...Md, B] where M = prod(M1...Md)\napply weight\nreshape: [Co, M1...Md, B] <- [Co, M, B]\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.optimize","page":"API","title":"NeuralROMs.optimize","text":"optimize(opt, NN, p, st, nepoch, _loader, loader_; ...)\noptimize(\n    opt,\n    NN,\n    p,\n    st,\n    nepoch,\n    _loader,\n    loader_,\n    __loader;\n    lossfun,\n    opt_st,\n    cb,\n    io,\n    early_stopping,\n    patience,\n    schedule,\n    kwargs...\n)\n\n\nTrain parameters p to minimize loss using optimization strategy opt.\n\nArguments\n\nLoss signature: loss(p, st) -> y, st\nCallback signature: cb(p, st epoch, nepoch) -> nothing \n\n\n\n\n\n","category":"function"},{"location":"API/#NeuralROMs.optimize-2","page":"API","title":"NeuralROMs.optimize","text":"references\n\nhttps://docs.sciml.ai/Optimization/stable/tutorials/minibatch/ https://lux.csail.mit.edu/dev/tutorials/advanced/1_GravitationalWaveForm#training-the-neural-network\n\n\n\n\n\n","category":"function"},{"location":"API/#NeuralROMs.plot_1D_surrogate_steady-Tuple{CalculustCore.Spaces.AbstractSpace{<:Any, 1}, Tuple{Any, Any}, Tuple{Any, Any}, LuxCore.AbstractExplicitLayer, Any, Any}","page":"API","title":"NeuralROMs.plot_1D_surrogate_steady","text":"plot_1D_surrogate_steady(\n    V,\n    _data,\n    data_,\n    NN,\n    p,\n    st;\n    nsamples,\n    dir,\n    format\n)\n\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.pnorm-Tuple{Real}","page":"API","title":"NeuralROMs.pnorm","text":"pnorm(p)(y, ŷ) -> l\npnorm(p)(NN, p, st, batch) -> l, st, stats\n\nP-Norm\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.regularize_autodecoder-Union{Tuple{Any}, Tuple{T}} where T<:Real","page":"API","title":"NeuralROMs.regularize_autodecoder","text":"regularize_autodecoder(lossfun, σ, λ1, λ2, property)(NN, p, st, batch) -> l, st, stats\n\ncode reg loss, L1/L2 on decoder lossfun(..) + 1/σ² ||ũ||₂² + L1/L2 on decoder + Lipschitz reg. on decoder\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.regularize_decoder-Union{Tuple{Any}, Tuple{T}} where T<:Real","page":"API","title":"NeuralROMs.regularize_decoder","text":"regularize_decoder(lossfun, σ, λ1, λ2, property)(NN, p, st, batch) -> l, st, stats\n\ncode reg loss, L1/L2 on decoder lossfun(..) + 1/σ² ||ũ||₂² + L1/L2 on decoder + Lipschitz reg. on decoder\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.regularize_flatdecoder-Union{Tuple{Any}, Tuple{T}} where T<:Real","page":"API","title":"NeuralROMs.regularize_flatdecoder","text":"regularize_flatdecoder(lossfun, σ, λ1, λ2, property)(NN, p, st, batch) -> l, st, stats\n\nlossfun(..) + L2 (on hyper) + Lipschitz (on decoder)\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.rsquare-Tuple{Any, Any}","page":"API","title":"NeuralROMs.rsquare","text":"rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)\n\nCalculuate r2 (coefficient of determination) score.\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.statistics-Tuple{LuxCore.AbstractExplicitLayer, Union{NamedTuple, AbstractVector}, NamedTuple, Union{CUDA.CuIterator, MLUtils.DataLoader}}","page":"API","title":"NeuralROMs.statistics","text":"statistics(NN, p, st, loader)\n\n\n\n\n\n\n","category":"method"},{"location":"API/#NeuralROMs.train_model-Union{Tuple{M}, Tuple{LuxCore.AbstractExplicitLayer, Tuple{Any, Any}}, Tuple{LuxCore.AbstractExplicitLayer, Tuple{Any, Any}, Union{Nothing, Tuple{Any, Any}}}} where M","page":"API","title":"NeuralROMs.train_model","text":"train_model(NN, _data; ...)\ntrain_model(\n    NN,\n    _data,\n    data_;\n    rng,\n    _batchsize,\n    batchsize_,\n    __batchsize,\n    opts,\n    nepochs,\n    schedules,\n    early_stoppings,\n    patience_fracs,\n    weight_decays,\n    dir,\n    name,\n    metadata,\n    io,\n    p,\n    st,\n    lossfun,\n    device,\n    cb_epoch\n)\n\n\nArguments\n\nNN: Lux neural network\n_data: training data as (x, y). x may be an AbstractArray or a tuple of arrays\ndata_: testing data (same requirement as `_data)\n\nKeyword Arguments\n\nrng: random nunmber generator\n_batchsize/batchsize_: train/test batch size\nopts/nepochs: NTuple of optimizers, # epochs per optimizer\ncbstep: prompt callback function every cbstep epochs\ndir/name: directory to save model, plots, model name\nio: io for printing stats\np/st: initial model parameter, state. if nothing, initialized with Lux.setup(rng, NN)\n\n\n\n\n\n","category":"method"}]
}
