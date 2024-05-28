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

### Setup and run

Download the code and 

```julia
julia> import Pkg

julia> Pkg.develop(url = "https://github.com/vpuri3/NeuralROMs.jl")

julia> cd(joinpath(DEPOT_PATH[1], "dev", "NeuralROMs.jl"))

julia> Pkg.activate() # activate the environment
```

We show how to run the 1D Advection test case corresponding to Section 6.1 of the paper.
Each test case in Section 6 of the paper has a corresponding directory in `examples/`.

```julia
julia> include("examples/advect_fourier1D/datagen_advect1D.jl")
```

The script solves the 1D advection problem and stores the dataset as a JLD2 binary in
`examples/advect_fourier1D/data_advect/`.
To train SNF-ROM, run

```julia
julia> include("examples/advect_fourier1D/snf.jl")
```

### Results

#### 1D Advection
#### 2D Advection
#### 1D Viscous Burgers (Re 10k)
#### 2D Viscous Burgers (Re 1k)
#### 1D Kuramoto-Sivashinsky

## Citing
```
@misc{puri2024snfrom,
      title={{SNF-ROM}: {P}rojection-based nonlinear reduced order modeling with smooth neural fields},
      author={Vedant Puri and Aviral Prakash and Levent Burak Kara and Yongjie Jessica Zhang},
      year={2024},
      eprint={2405.14890},
      archivePrefix={arXiv},
      primaryClass={physics.flu-dyn},
}
```
