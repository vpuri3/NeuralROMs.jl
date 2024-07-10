```@meta
CurrentModule = NeuralROMs
```

# NeuralROMs.jl

This repository implements machine learning (ML) based reduced order models (ROMs).
Specifically, we introduce [smooth neural field ROM (SNF-ROM)](https://arxiv.org/abs/2405.14890) for solving advection dominated PDE problems.

## Smooth neural field ROM

> SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields
>
> [Vedant Puri](https://vpuri3.github.io/), [Aviral Prakash](https://scholar.google.com/citations?user=KgbgFP0AAAAJ&hl=en&oi=ao), [Levent Burak Kara](http://vdel.me.cmu.edu/), [Yongjie Jessica Zhang](https://www.meche.engineering.cmu.edu/faculty/zhang-computational-bio-modeling-lab.html)
>
> [Project page](https://vpuri3.github.io/NeuralROMs.jl/dev/) / [Paper](https://arxiv.org/abs/2405.14890)

### Abstract

Reduced order modeling lowers the computational cost of solving PDEs by learning a low-dimensional spatial representation from data and dynamically evolving these representations using manifold projections of the governing equations.
The commonly used linear subspace reduced-order models (ROMs) are often suboptimal for problems with a slow decay of Kolmogorov n-width, such as advection-dominated fluid flows at high Reynolds numbers.
There has been a growing interest in nonlinear ROMs that use state-of-the-art representation learning techniques to accurately capture such phenomena with fewer degrees of freedom.
We propose smooth neural field ROM (SNF-ROM), a nonlinear reduced order modeling framework that combines grid-free reduced representations with Galerkin projection.
The SNF-ROM architecture constrains the learned ROM trajectories to a smoothly varying path, which proves beneficial in the dynamics evaluation when the reduced manifold is traversed in accordance with the governing PDEs.
Furthermore, we devise robust regularization schemes to ensure the learned neural fields are smooth and differentiable.
This allows us to compute physics-based dynamics of the reduced system nonintrusively with automatic differentiation and evolve the reduced system with classical time-integrators.
SNF-ROM leads to fast offline training as well as enhanced accuracy and stability during the online dynamics evaluation.
Numerical experiments reveal that SNF-ROM is able to accelerate the full-order computation by up to 199x.
We demonstrate the efficacy of SNF-ROM on a range of advection-dominated linear and nonlinear PDE problems where we consistently outperform state-of-the-art ROMs.

### Method

##### Offline stage
![Capture-2024-05-28-171751](https://github.com/vpuri3/NeuralROMs.jl/assets/36345239/9656da99-de98-4ead-9ae6-37f935bffa33)

##### Online stage
![Screenshot 2024-05-28 at 5 18 25 PM](https://github.com/vpuri3/NeuralROMs.jl/assets/36345239/8bdd00d0-c1e0-4aea-9bfa-b014b5e1a86b)

ROMs are hybrid physics and databased methods that decouple the computation into two stages: an expensive offline stage and a cheap online stage. In
the offline stage, a low-dimensional spatial representation is learned from simulation data by projecting the solution
field snapshots onto a low-dimensional manifold that can faithfully capture the relevant features in the dataset. The
online stage then involves evaluating the model at new parametric points by time-evolving the learned spatial representation following the governing PDE system with classical time integrators.

SNF-ROM is a continuous neural field ROM that is nonintrusive by construction and eliminates the need for a fixed grid structure in the underlying
data and the identification of associated spatial discretization for dynamics evaluation. There are two important
features of SNF-ROM:
1. Constrained manifold formulation: SNF-ROM restricts the reduced trajectories to follow a regular, smoothly varying path. This behavior is achieved by directly modeling the ROM state vector as a simple, learnable function of problem parameters and time. Our numerical experiments reveal that this feature allows for larger time steps in the dynamics evaluation, where the reduced manifold is traversed in accordance with the governing PDEs.
2. Neural field regularization: We formulate a robust network regularization approach encouraging smoothness in the learned neural fields. Consequently, the spatial derivatives of SNF representations match the true derivatives of the underlying signal. This feature allows us to calculate accurate spatial derivatives with the highly efficient forward mode automatic differentiation (AD) technique. Our studies indicate that precisely capturing spatial derivatives is crucial for an accurate dynamics prediction.

The confluence of these two features produces desirable effects on the dynamics evaluation, such as greater accuracy,
robustness to hyperparameter choice, and robustness to numerical perturbations.

## Citation

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

## Acknowledgements

The authors would like to acknowledge the support from the National Science Foundation (NSF) grant CMMI-1953323 and PA Manufacturing Fellows Initiative for the funds used towards this project.
The research in this paper was also sponsored by the Army Research Laboratory and was accomplished under Cooperative Agreement Number W911NF-20-2-0175.
The views and conclusions contained in this document are those of the authors and should not be interpreted as representing the official policies, either expressed or implied, of the Army Research Laboratory or the U.S. Government.
The U.S. Government is authorized to reproduce and distribute reprints for Government purposes notwithstanding any copyright notation herein.
