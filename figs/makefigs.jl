using GeometryLearning

srcdir = joinpath(pkgdir(GeometryLearning), "examples")
figdir = joinpath(pkgdir(GeometryLearning), "figs")

mkpath(figdir)

e1src = joinpath(srcdir, "advect_fourier1D")
e2src = joinpath(srcdir, "advect_fourier2D")
e3src = joinpath(srcdir, "burgers_fourier1D")
e4src = joinpath(srcdir, "burgers_fourier2D")
e5src = joinpath(srcdir, "ks_fourier1D")

e1dst = joinpath(figdir, "exp1")
e2dst = joinpath(figdir, "exp2")
e3dst = joinpath(figdir, "exp3")
e4dst = joinpath(figdir, "exp4")
e5dst = joinpath(figdir, "exp5")

mkpath(e1dst)
mkpath(e2dst)
mkpath(e3dst)
mkpath(e4dst)
mkpath(e5dst)

force = true

# EXP 1
cp(joinpath(e1src, "compare_er_case1.png"), joinpath(e1dst, "compare_er_case1.png"); force)
cp(joinpath(e1src, "compare_t0_case1.png"), joinpath(e1dst, "compare_t0_case1.png"); force)
cp(joinpath(e1src, "compare_t1_case1.png"), joinpath(e1dst, "compare_t1_case1.png"); force)

cp(joinpath(e1src, "model_CAE02", "results", "compare_p_scatter_case1.png"), joinpath(e1dst, "CAE-p.png"); force)
cp(joinpath(e1src, "model_SNL02", "results", "compare_p_scatter_case1.png"), joinpath(e1dst, "SNL-p.png"); force)
cp(joinpath(e1src, "model_SNW02", "results", "compare_p_scatter_case1.png"), joinpath(e1dst, "SNW-p.png"); force)

# EXP 2
cp(joinpath(e2src, "compare_er_case1.png"), joinpath(e2dst, "compare_er_case1.png"); force)
cp(joinpath(e2src, "compare_t0_case1.png"), joinpath(e2dst, "compare_t0_case1.png"); force)
cp(joinpath(e2src, "compare_t1_case1.png"), joinpath(e2dst, "compare_t1_case1.png"); force)

cp(joinpath(e2src, "data_advect", "heatmap_2.png"), joinpath(e2dst, "FOM-u-t0.png"); force)
cp(joinpath(e2src, "data_advect", "heatmap_6.png"), joinpath(e2dst, "FOM-u-t1.png"); force)

cp(joinpath(e2src, "model_PCA08", "results", "evolve_u1_1_time_2.png"), joinpath(e2dst, "PCA-u-t0.png"); force)
cp(joinpath(e2src, "model_PCA08", "results", "evolve_u1_1_time_4.png"), joinpath(e2dst, "PCA-u-t1.png"); force)
cp(joinpath(e2src, "model_PCA08", "results", "evolve_u1_1_time_2_error.png"), joinpath(e2dst, "PCA-e-t0.png"); force)
cp(joinpath(e2src, "model_PCA08", "results", "evolve_u1_1_time_4_error.png"), joinpath(e2dst, "PCA-e-t1.png"); force)

cp(joinpath(e2src, "model_CAE02", "results", "evolve_u1_1_time_2.png"), joinpath(e2dst, "CAE-u-t0.png"); force)
cp(joinpath(e2src, "model_CAE02", "results", "evolve_u1_1_time_4.png"), joinpath(e2dst, "CAE-u-t1.png"); force)
cp(joinpath(e2src, "model_CAE02", "results", "evolve_u1_1_time_2_error.png"), joinpath(e2dst, "CAE-e-t0.png"); force)
cp(joinpath(e2src, "model_CAE02", "results", "evolve_u1_1_time_4_error.png"), joinpath(e2dst, "CAE-e-t1.png"); force)

cp(joinpath(e2src, "model_SNL02", "results", "evolve_u1_1_time_2.png"), joinpath(e2dst, "SNL-u-t0.png"); force)
cp(joinpath(e2src, "model_SNL02", "results", "evolve_u1_1_time_4.png"), joinpath(e2dst, "SNL-u-t1.png"); force)
cp(joinpath(e2src, "model_SNL02", "results", "evolve_u1_1_time_2_error.png"), joinpath(e2dst, "SNL-e-t0.png"); force)
cp(joinpath(e2src, "model_SNL02", "results", "evolve_u1_1_time_4_error.png"), joinpath(e2dst, "SNL-e-t1.png"); force)

cp(joinpath(e2src, "model_SNW02", "results", "evolve_u1_1_time_2.png"), joinpath(e2dst, "SNW-u-t0.png"); force)
cp(joinpath(e2src, "model_SNW02", "results", "evolve_u1_1_time_4.png"), joinpath(e2dst, "SNW-u-t1.png"); force)
cp(joinpath(e2src, "model_SNW02", "results", "evolve_u1_1_time_2_error.png"), joinpath(e2dst, "SNW-e-t0.png"); force)
cp(joinpath(e2src, "model_SNW02", "results", "evolve_u1_1_time_4_error.png"), joinpath(e2dst, "SNW-e-t1.png"); force)

# EXP 3
cp(joinpath(e3src, "compare_er_case4.png"), joinpath(e3dst, "compare_er_case4.png"); force)
cp(joinpath(e3src, "compare_t0_case4.png"), joinpath(e3dst, "compare_t0_case4.png"); force)
cp(joinpath(e3src, "compare_t1_case4.png"), joinpath(e3dst, "compare_t1_case4.png"); force)

cp(joinpath(e3src, "compare_er_case5.png"), joinpath(e3dst, "compare_er_case5.png"); force)
cp(joinpath(e3src, "compare_t0_case5.png"), joinpath(e3dst, "compare_t0_case5.png"); force)
cp(joinpath(e3src, "compare_t1_case5.png"), joinpath(e3dst, "compare_t1_case5.png"); force)

cp(joinpath(e3src, "model_CAE02", "results", "train_p_scatter.png"), joinpath(e3dst, "CAE-p.png"); force)
cp(joinpath(e3src, "model_SNL02", "results", "train_p_scatter.png"), joinpath(e3dst, "SNL-p.png"); force)
cp(joinpath(e3src, "model_SNW02", "results", "train_p_scatter.png"), joinpath(e3dst, "SNW-p.png"); force)

# EXP 4
cp(joinpath(e4src, "compare_er_case1.png"), joinpath(e4dst, "compare_er_case1.png"); force)
cp(joinpath(e4src, "compare_t0_case1.png"), joinpath(e4dst, "compare_t0_case1.png"); force)
cp(joinpath(e4src, "compare_t1_case1.png"), joinpath(e4dst, "compare_t1_case1.png"); force)

cp(joinpath(e4src, "data_burgers2D", "heatmap_2.png"), joinpath(e4dst, "FOM-u-t0.png"); force)
cp(joinpath(e4src, "data_burgers2D", "heatmap_4.png"), joinpath(e4dst, "FOM-u-t1.png"); force)

cp(joinpath(e4src, "model_PCA08", "results", "evolve_u1_1_time_2.png"), joinpath(e4dst, "PCA-u-t0.png"); force)
cp(joinpath(e4src, "model_PCA08", "results", "evolve_u1_1_time_4.png"), joinpath(e4dst, "PCA-u-t1.png"); force)
cp(joinpath(e4src, "model_PCA08", "results", "evolve_u1_1_time_2_error.png"), joinpath(e4dst, "PCA-e-t0.png"); force)
cp(joinpath(e4src, "model_PCA08", "results", "evolve_u1_1_time_4_error.png"), joinpath(e4dst, "PCA-e-t1.png"); force)

cp(joinpath(e4src, "model_CAE02", "results", "evolve_u1_1_time_2.png"), joinpath(e4dst, "CAE-u-t0.png"); force)
cp(joinpath(e4src, "model_CAE02", "results", "evolve_u1_1_time_4.png"), joinpath(e4dst, "CAE-u-t1.png"); force)
cp(joinpath(e4src, "model_CAE02", "results", "evolve_u1_1_time_2_error.png"), joinpath(e4dst, "CAE-e-t0.png"); force)
cp(joinpath(e4src, "model_CAE02", "results", "evolve_u1_1_time_4_error.png"), joinpath(e4dst, "CAE-e-t1.png"); force)

cp(joinpath(e4src, "model_SNL02", "results", "evolve_u1_1_time_2.png"), joinpath(e4dst, "SNL-u-t0.png"); force)
cp(joinpath(e4src, "model_SNL02", "results", "evolve_u1_1_time_4.png"), joinpath(e4dst, "SNL-u-t1.png"); force)
cp(joinpath(e4src, "model_SNL02", "results", "evolve_u1_1_time_2_error.png"), joinpath(e4dst, "SNL-e-t0.png"); force)
cp(joinpath(e4src, "model_SNL02", "results", "evolve_u1_1_time_4_error.png"), joinpath(e4dst, "SNL-e-t1.png"); force)

cp(joinpath(e4src, "model_SNW02", "results", "evolve_u1_1_time_2.png"), joinpath(e4dst, "SNW-u-t0.png"); force)
cp(joinpath(e4src, "model_SNW02", "results", "evolve_u1_1_time_4.png"), joinpath(e4dst, "SNW-u-t1.png"); force)
cp(joinpath(e4src, "model_SNW02", "results", "evolve_u1_1_time_2_error.png"), joinpath(e4dst, "SNW-e-t0.png"); force)
cp(joinpath(e4src, "model_SNW02", "results", "evolve_u1_1_time_4_error.png"), joinpath(e4dst, "SNW-e-t1.png"); force)

# EXP 5
cp(joinpath(e5src, "compare_er_case1.png"), joinpath(e5dst, "compare_er_case1.png"); force)
cp(joinpath(e5src, "compare_t0_case1.png"), joinpath(e5dst, "compare_t0_case1.png"); force)
cp(joinpath(e5src, "compare_t1_case1.png"), joinpath(e5dst, "compare_t1_case1.png"); force)

#
nothing
