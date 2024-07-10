```@meta
CurrentModule = NeuralROMs
```

# NeuralROMs.jl

This repository implements machine learning (ML) based reduced order models (ROMs).
Specifically, we introduce [smooth neural field ROM (SNF-ROM)](https://arxiv.org/abs/2405.14890) for solving advection dominated PDE problems.

## SNF-ROM: Projection-based nonlinear reduced order modeling with smooth neural fields

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
