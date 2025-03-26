# Particle Filtering – Object Tracking Project

This repository presents the implementation of a Particle Filter for hidden state estimation in non-linear, non-Gaussian models.  
Developed during the *Statistical Filtering* course (MAT4501) at Télécom SudParis, the project is divided into two parts: a simulation benchmark (Kitagawa model) and a real-world face tracking application using video sequences.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Technical Description](#technical-description)
- [Code Breakdown](#code-breakdown)
- [Key Features](#key-features)
- [Learning Objectives](#learning-objectives)

---

## Project Overview

The particle filter is a sequential Monte Carlo method that approximates the posterior distribution of hidden states using a weighted set of samples (particles).

This project includes two core components:

1. **Kitagawa Model**  
   - A nonlinear system often used in econometrics to benchmark filtering algorithms.
   - Implemented from scratch using transition and observation models.
   - Analysis of estimation error, influence of noise, and number of particles.

2. **Face Tracking in Video**  
   - Tracking a face across a video sequence using color histograms as the observation model.
   - Particles represent candidate object locations.
   - Weights derived from histogram similarity with the target.
   - Visual results include bounding box, particles, and estimated trajectory.

---

## Technical Description

### State-Space Formulation

For both parts, the hidden state \(```math X_n ```\) evolves according to a transition model:

```math
X_n = f(X_{n-1}) + U_n
```

Observations \( Y_n \) are generated via:
```math
Y_n = g(X_n) + V_n
```

where \( U_n \) and \( V_n \) are random noise components.

- In the **Kitagawa model**, the functions `f` and `g` are nonlinear.
- In the **vision-based tracking**, \( X_n \) encodes the top-left position of the bounding box, and \( g(X_n) \) corresponds to the histogram of the color patch.

---

## Code Breakdown

### `exercice1.py`

Implements the particle filter on the **Kitagawa model**:

- Generates a true trajectory and observations
- Applies particle filtering step-by-step:
  - Propagation (transition)
  - Weighting (likelihood)
  - Resampling
- Plots:
  - True vs. estimated trajectory
  - Error curves
  - Comparison with varying number of particles


