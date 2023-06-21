var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = GeometryLearning","category":"page"},{"location":"#GeometryLearning","page":"Home","title":"GeometryLearning","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for GeometryLearning.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [GeometryLearning]","category":"page"},{"location":"#GeometryLearning.Atten","page":"Home","title":"GeometryLearning.Atten","text":"Attention Layer\n\nsingle layer model with no nonlinearity (single head linear attention)\n\nu = NN(f) q = Wq * u k = Wk * u v = Wv * u\n\nv = activation(q * k') * u\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.Diag","page":"Home","title":"GeometryLearning.Diag","text":"Diagonal layer\n\n\n\n\n\n","category":"type"},{"location":"#GeometryLearning.model_setup-Tuple{LuxCore.AbstractExplicitLayer, Any}","page":"Home","title":"GeometryLearning.model_setup","text":"model_setup(NN, st)\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.mse-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.mse","text":"mse(ypred, ytrue)\n\nMean squared error\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.rsquare-Tuple{Any, Any}","page":"Home","title":"GeometryLearning.rsquare","text":"rsquare(ypred, ytrue) -> 1 - MSE(ytrue, ypred) / var(yture)\n\nCalculuate r2 (coefficient of determination) score.\n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.train-NTuple{4, Any}","page":"Home","title":"GeometryLearning.train","text":"train\n\nTrain parameters p to minimize loss using optimization strategy opt.\n\nArguments\n\nLoss signature: loss(p, st) -> y, st\nCallback signature: cb(p, st iter, maxiter) -> nothing \n\n\n\n\n\n","category":"method"},{"location":"#GeometryLearning.visualize-NTuple{6, Any}","page":"Home","title":"GeometryLearning.visualize","text":"visualize \n\n\n\n\n\n","category":"method"}]
}
