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


---

### `exercice2.py`

Applies the particle filter to **image-based tracking**:

- Loads a video frame sequence
- Allows manual selection of initial tracking region
- Computes reference histogram for the object
- Propagates particles across frames using Gaussian noise
- Reweights particles based on histogram similarity
- Resamples and estimates the new object position
- Plots:
  - Bounding box (red)
  - Particle cloud (blue)
  - Frame-by-frame overlay of tracking results

---

### `exercice2_version_zoom.py`

Extension of `exercice2.py` that includes **scale adaptation** in the state vector:

- Augments state space with a scaling factor
- Adjusts bounding box dimensions accordingly
- Allows tracking with zoom effect (object moving closer/farther from camera)

---

## Key Features

- Works with both **synthetic and real-world data**
- Handles **nonlinear and non-Gaussian** filtering
- Implements **resampling** strategies
- Robust to partial occlusion and observation noise
- Modular code structure, reusable functions

---

## Learning Objectives

- Understand the principles of sequential Monte Carlo methods
- Implement a complete particle filter from scratch
- Apply filtering to nonlinear dynamic systems
- Use computer vision techniques for object tracking
- Evaluate the impact of key parameters: noise, number of particles, λ (likelihood sharpness)
