# Advanced details

# Estimation
For the MM algorithm, the updates in each iteration are

```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} (\boldsymbol{I}_d \otimes \boldsymbol{X})]^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} \text{vec}\ \boldsymbol{Y} \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \boldsymbol{L}_i^{-(t)T}[\boldsymbol{L}_i^{(t)T}\boldsymbol{\Gamma}_i^{(t)}(\boldsymbol{R}^{(t)T}\boldsymbol{V}_i\boldsymbol{R}^{(t)})\boldsymbol{\Gamma}_i^{(t)}\boldsymbol{L}_i^{(t)}]^{1/2} \boldsymbol{L}_i^{-(t)},
\end{aligned}
```
where ``\boldsymbol{\Omega}^{(t)} = \sum_{i=1}^m \boldsymbol{\Gamma}_i^{(t)} \otimes \boldsymbol{V}_i`` and ``\boldsymbol{L}_i^{(t)}`` is the Cholesky factor of ``\boldsymbol{M}_i^{(t)} = (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)^T [(\boldsymbol{1}_d \boldsymbol{1}_d^T \otimes \boldsymbol{V}_i) \odot \boldsymbol{\Omega}^{-(t)}] (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)``, while ``\boldsymbol{R}^{(t)}`` is the ``n \times d`` matrix such that ``\text{vec}\ \boldsymbol{R}^{(t)} = \boldsymbol{\Omega}^{-(t)} \text{vec}(\boldsymbol{Y} - \boldsymbol{X} \boldsymbol{B}^{(t)})``.

For the EM algorithm, the updates in each iteration are

```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} (\boldsymbol{I}_d \otimes \boldsymbol{X})]^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} \text{vec}\ \boldsymbol{Y} \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \frac{1}{r_i} \boldsymbol{\Gamma}_i^{(t)} ( \boldsymbol{R}^{(t)T} \boldsymbol{V}_i \boldsymbol{R}^{(t)} - \boldsymbol{M}_i^{(t)} ) \boldsymbol{\Gamma}_i^{(t)} + \boldsymbol{\Gamma}_i^{(t)},
\end{aligned}
```
where ``r_i = \text{rank}(\boldsymbol{V}_i)``. As seen, the updates for mean effects ``\boldsymbol{B}`` are the same for these two algorithms.

In the setting of missing response, the adjusted MM updates in each interation are
```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} (\boldsymbol{I}_d \otimes \boldsymbol{X})]^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} \text{vec}\ \boldsymbol{Z}^{(t)} \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \boldsymbol{L}_i^{-(t)T}[\boldsymbol{L}_i^{(t)T}\boldsymbol{\Gamma}_i^{(t)}(\boldsymbol{R}^{*(t)T}\boldsymbol{V}_i\boldsymbol{R}^{*(t)} + \boldsymbol{M}_i^{*(t)})\boldsymbol{\Gamma}_i^{(t)}\boldsymbol{L}_i^{(t)}]^{1/2} \boldsymbol{L}_i^{-(t)},
\end{aligned}
```
where ``\boldsymbol{Z}^{(t)}`` is the completed response matrix from conditional mean and ``\boldsymbol{M}_i^{*(t)} = (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)^T [(\boldsymbol{1}_d \boldsymbol{1}_d^T \otimes \boldsymbol{V}_i) \odot (\boldsymbol{\Omega}^{-(t)} \boldsymbol{P}^T \boldsymbol{C}^{(t)}\boldsymbol{P}\boldsymbol{\Omega}^{-(t)})] (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)``, while ``\boldsymbol{R}^{*(t)}`` is the ``n \times d`` matrix such that ``\text{vec}\ \boldsymbol{R}^{*(t)} = \boldsymbol{\Omega}^{-(t)} \text{vec}(\boldsymbol{Z}^{(t)} - \boldsymbol{X} \boldsymbol{B}^{(t)})``. Additionally, ``\boldsymbol{P}`` is the ``nd \times nd`` permutation matrix such that ``\boldsymbol{P} \cdot \text{vec}\ \boldsymbol{Y} = \begin{bmatrix} \boldsymbol{y}_{\text{obs}} \\ \boldsymbol{y}_{\text{mis}} \end{bmatrix}``, where ``\boldsymbol{y}_{\text{obs}}`` and ``\boldsymbol{y}_{\text{mis}}`` are vectors of observed and missing response values, respectively, in column-major order, and the block matrix ``\boldsymbol{C}^{(t)}`` is ``\boldsymbol{0}`` except for a lower-right block consisting of conditional variance. As seen, the two MM updates for both mean effects and variance components are of similar form.

# Inference
Standard errors for our estimates are calculated using the Fisher information matrix, where

```math
\begin{aligned}
\text{E} \left[- \frac{\partial^2}{\partial(\text{vec}\ \boldsymbol{B})^T \partial(\text{vec}\ \boldsymbol{B})} \mathcal{L} \right] &= (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}) \\
\text{E} \left[ - \frac{\partial^2}{\partial (\text{vech}\ \boldsymbol{\Gamma}_i)^T \partial (\text{vec}\ \boldsymbol{B})} \mathcal{L} \right] &= \boldsymbol{0} \\
\text{E} \left[ - \frac{\partial^2}{\partial (\text{vech}\ \boldsymbol{\Gamma}_j)^T \partial (\text{vech}\ \boldsymbol{\Gamma}_i)} \mathcal{L} \right] &= \frac{1}{2} \boldsymbol{U}_i^T (\boldsymbol{\Omega}^{-1} \otimes \boldsymbol{\Omega}^{-1}) \boldsymbol{U}_j
\end{aligned}
```

and ``\boldsymbol{U}_i = (\boldsymbol{I}_d \otimes \boldsymbol{K}_{nd} \otimes \boldsymbol{I}_n) (\boldsymbol{I}_{d^2} \otimes \text{vec}\ \boldsymbol{V}_i) \boldsymbol{D}_{d}``. Here, ``\boldsymbol{K}_{nd}`` is the ``nd \times nd`` [commutation matrix](https://en.wikipedia.org/wiki/Commutation_matrix) and ``\boldsymbol{D}_{d}`` the ``d^2 \times \frac{d(d+1)}{2}`` [duplication matrix](https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices). ``\text{vech}\ \boldsymbol{\Gamma}_i`` creates an ``\frac{d(d+1)}{2} \times 1`` vector from ``\boldsymbol{\Gamma}_i`` by stacking its lower triangular part.