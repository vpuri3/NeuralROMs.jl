# NeuralROMs.jl

This repository implements machine learning (ML) based reduced order models (ROMs).
Specifically, we introduce [smooth neural field ROM (SNF-ROM)](https://arxiv.org/abs/2405.14890) for solving advection dominated PDE problems.

## Smooth neural field ROM

> SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields
>
> [Vedant Puri](https://vpuri3.github.io/), [Aviral Prakash](https://scholar.google.com/citations?user=KgbgFP0AAAAJ&hl=en&oi=ao), [Levent Burak Kara](http://vdel.me.cmu.edu/), [Yongjie Jessica Zhang](https://www.meche.engineering.cmu.edu/faculty/zhang-computational-bio-modeling-lab.html)
>
> [Project page](https://vpuri3.github.io/NeuralROMs.jl/dev/) / [Paper](https://arxiv.org/abs/2405.14890)

#### Offline stage
![Capture-2024-05-28-171751](https://github.com/vpuri3/NeuralROMs.jl/assets/36345239/9656da99-de98-4ead-9ae6-37f935bffa33)

#### Online stage
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

julia> Pkg.activate(".") # switch environment

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
