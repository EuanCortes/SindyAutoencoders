# SINDy Autoencoders for Dynamical Systems (PyTorch Implementation)

This project is inspired by Exercise 14.1 from the course textbook and builds upon the work of Champion et al. ([GitHub Repository](https://github.com/kpchamp/SindyAutoencoders)). Our objective is to explore **Sparse Identification of Nonlinear Dynamics (SINDy)** combined with autoencoders to model complex dynamical systems.

## 📌 Project Overview

This project focuses on three classic nonlinear dynamical systems:

- The **Lorenz system**
- A **Lotka-Volterra**
- The **nonlinear pendulum**

We aim to gain a deep understanding of how **SINDy Autoencoders** work and how they can be used to uncover interpretable dynamical models from high-dimensional data. While the original implementation is provided in TensorFlow, we re-implement the approach from scratch using **PyTorch** to ensure flexibility, transparency, and reproducibility.

## 🎯 Objectives

- 📖 **Learn and document**: Consolidate knowledge on SINDy Autoencoders and present it in an accessible, well-organized format.
- 🧠 **Autoencoding dynamics**: Train autoencoders to reduce the dimensionality of the input data, encoding it into a latent space.
- 🔍 **Sparse modeling**: Apply SINDy in the latent space to discover the most parsimonious model that describes the system’s dynamics.
- 🔁 **Reproduce and extend**: Validate results against the original work and experiment with custom modifications.

## 🛠️ Implementation Details

- Framework: **PyTorch**
- Key components:
  - Custom autoencoder architectures
  - SINDy integration in the latent space
  - Training and evaluation pipelines for dynamical systems


