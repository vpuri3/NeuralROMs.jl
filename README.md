# NeuralROMs.jl

This repository implements machine learning (ML) based reduced order models (ROMs).
Specifically, we introduce [smooth neural field ROM (SNF-ROM)](https://arxiv.org/abs/2405.14890) for solving advection dominated PDE problems.

## SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields

### Abstract

Reduced order modeling lowers the computational cost of solving PDEs by learning a low-order spatial representation from data and dynamically evolving these representations using manifold projections of the governing equations.
While commonly used, linear subspace reduced-order models (ROMs) are often suboptimal for problems with a slow decay of Kolmogorov $n$-width, such as advection-dominated fluid flows at high Reynolds numbers.
There has been a growing interest in nonlinear ROMs that use state-of-the-art representation learning techniques to accurately capture such phenomena with fewer degrees of freedom.
We propose smooth neural field ROM (SNF-ROM), a nonlinear reduced modeling framework that combines grid-free reduced representations with Galerkin projection.
The SNF-ROM architecture constrains the learned ROM trajectories to a smoothly varying path, which proves beneficial in the dynamics evaluation when the reduced manifold is traversed in accordance with the governing PDEs.
Furthermore, we devise robust regularization schemes to ensure the learned neural fields are smooth and differentiable.
This allows us to compute physics-based dynamics of the reduced system nonintrusively with automatic differentiation and evolve the reduced system with classical time-integrators.
SNF-ROM leads to fast offline training as well as enhanced accuracy and stability during the online dynamics evaluation.
We demonstrate the efficacy of SNF-ROM on a range of advection-dominated linear and nonlinear PDE problems where we consistently outperform state-of-the-art ROMs.

### Method

ROMs are hybrid physics and databased methods that decouple the computation into two stages: an expensive offline stage and a cheap online stage. In
the offline stage, a low-dimensional spatial representation is learned from simulation data by projecting the solution
field snapshots onto a low-dimensional manifold that can faithfully capture the relevant features in the dataset. The
online stage then involves evaluating the model at new parametric points by time-evolving the learned spatial representation following the governing PDE system with classical time integrators.

SNF-ROM is a continuous neural field ROM that is nonintrusive by construction and eliminates the need for a fixed grid structure in the underlying
data and the identification of associated spatial discretization for dynamics evaluation. There are two important
features of SNF-ROM:
1. Constrained manifold formulation: SNF-ROM restricts the reduced trajectories to follow a regular, smoothly
varying path. This behavior is achieved by directly modeling the ROM state vector as a simple, learnable
function of problem parameters and time. Our numerical experiments reveal that this feature allows for larger
time steps in the dynamics evaluation, where the reduced manifold is traversed in accordance with the governing
PDEs.
2. Neural field regularization: We formulate a robust network regularization approach encouraging smoothness
in the learned neural fields. Consequently, the spatial derivatives of SNF representations match the true derivatives of the underlying signal. This feature allows us to calculate accurate spatial derivatives with the highly
efficient forward mode automatic differentiation (AD) technique. Our studies indicate that precisely capturing
spatial derivatives is crucial for an accurate dynamics prediction.

The confluence of these two features produces desirable effects on the dynamics evaluation, such as greater accuracy,
robustness to hyperparameter choice, and robustness to numerical perturbations.

##### Offline stage
![Capture-2024-05-28-171751](https://github.com/vpuri3/NeuralROMs.jl/assets/36345239/9656da99-de98-4ead-9ae6-37f935bffa33)

##### Online stage
![Screenshot 2024-05-28 at 5 18 25 PM](https://github.com/vpuri3/NeuralROMs.jl/assets/36345239/8bdd00d0-c1e0-4aea-9bfa-b014b5e1a86b)

### Setup and run

Download the code by cloning this Git repo.

```bash
$ git clone git@github.com:vpuri3/NeuralROMs.jl.git

$ cd NeuralROMs.jl
```

Start Julia and activate the environment.

```bash
$ julia
```

```julia
julia> import Pkg

julia> Pkg.activate() # switch environment

julia> Pkg.instantiate() # download environment
```

We show how to run the 1D Advection test case corresponding to Section 6.1 of the paper.
Each test case in Section 6 of the paper has a corresponding directory in `experiments_SNFROM/`.

```julia
julia> include("experiments_SNFROM/advect_fourier1D/datagen_advect1D.jl")
```

The script solves the 1D advection problem and stores the dataset as a JLD2 binary in
`experiments_SNFROM/advect_fourier1D/data_advect/`.
To train SNF-ROM, run

```julia
julia> include("experiments_SNFROM/advect_fourier1D/snf.jl")
```

### Code Structure

```bash
$ tree . -L 1 --filesfirst
.
├── CITATION.bib        # arXiv paper
├── LICENSE             # MIT License
├── Manifest.toml       # environment metadata
├── Project.toml        # environment spec
├── README.md           # this file
├── benchmarks          # internal benchmarking scripts
├── docs                # documentation (incomplete)
├── examples            # playground
├── experiments_SNFROM  # experiments in SNF-ROM paper Section 6
├── figs                # figures in SNF-ROM paper
├── src                 # source code
└── test                # test scripts (incomplete)

```

```bash
$ tree src/ -L 2 --filesfirst
.
├── autodiff.jl            # AD wrapper for 1-4th order derivatives
├── metrics.jl             # Loss functions
├── neuralgridmodel.jl     # Grid-dependent neural space discretization (e.g., CAE-ROM, POD-ROM)
├── neuralmodel.jl         # Neural field spatial discretization (e.g., C-ROM, SNF-ROM)
├── NeuralROMs.jl          # Main file: declares Julia module and imports relevant packages
├── nonlinleastsq.jl       # Nonlinear least square solve for LSPG and for initializing auto-decode.
├── optimisers.jl          # Modified weight decay optimisers based on Optimisers.jl
├── pdeproblems.jl         # PDE problem definitions du/dt = f(x, u, t, u', ...)
├── train.jl               # Training loop
├── utils.jl               # Miscalleneous utility functions
├── vis.jl                 # 1D/2D visualizations
├── dynamics               #
│   ├── evolve.jl          # Logic for dynamics evaluation
│   └── timeintegrator.jl  # Time integrator object definition
├── layers                 #
│   ├── basic.jl           # Basic layer definitions (e.g., PermuteLayer, HyperNet)
│   ├── encoder_decoder.jl # Encoder-decoder network definitions (auto-decode, CAE, C-ROM, SNF-ROM)
│   └── sdf.jl             # Layers for 3D shape encoding
└── operator               #
    ├── oplayers.jl        # Fourier neural operator kernel definitions
    └── transform.jl       # Spectral transforms for FNO
```

```bash
$ tree experiments_SNFROM/ -L 1 --filesfirst
experiments_SNFROM/
├── autodecode.jl     # Autodecode-ROM training and inference
├── cases.jl          # Experiment setup
├── compare.jl        # Comparison script
├── convAE.jl         # CAE-ROM training and inference
├── convINR.jl        # C-ROM training and inference
├── PCA.jl            # POD-ROM training and inference
├── smoothNF.jl       # SNF-ROM training and inference
├── advect_fourier1D  # Section 6.1
├── advect_fourier2D  # Section 6.2
├── burgers_fourier1D # Section 6.3
├── burgers_fourier2D # Section 6.4
└── ks_fourier1D      # Section 6.5
```

## Citing
```bib
@misc{
    puri2024snfrom,
    title={{SNF-ROM}: {P}rojection-based nonlinear reduced order modeling with smooth neural fields},
    author={Vedant Puri and Aviral Prakash and Levent Burak Kara and Yongjie Jessica Zhang},
    year={2024},
    eprint={2405.14890},
    archivePrefix={arXiv},
    primaryClass={physics.flu-dyn},
}
```
