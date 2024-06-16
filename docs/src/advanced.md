# Advanced details

# Estimation
For the MM algorithm, the updates in each iteration are

```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} (\boldsymbol{I}_d \otimes \boldsymbol{X})]^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} \text{vec}\ \boldsymbol{Y} \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \boldsymbol{L}_i^{-(t)T}[\boldsymbol{L}_i^{(t)T}\boldsymbol{\Gamma}_i^{(t)}(\boldsymbol{R}^{(t)T}\boldsymbol{V}_i\boldsymbol{R}^{(t)})\boldsymbol{\Gamma}_i^{(t)}\boldsymbol{L}_i^{(t)}]^{1/2} \boldsymbol{L}_i^{-(t)},
\end{aligned}
```
where ``\boldsymbol{\Omega}^{(t)} = \sum_{i=1}^m \boldsymbol{\Gamma}_i^{(t)} \otimes \boldsymbol{V}_i`` and ``\boldsymbol{L}_i^{(t)}`` is the Cholesky factor of ``\boldsymbol{M}_i^{(t)} = (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)^T [(\boldsymbol{1}_d \boldsymbol{1}_d^T \otimes \boldsymbol{V}_i) \odot \boldsymbol{\Omega}^{-(t)}] (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)``, while ``\boldsymbol{R}^{(t)}`` is the ``n \times d`` matrix such that ``\text{vec}\ \boldsymbol{R}^{(t)} = \boldsymbol{\Omega}^{-(t)} \text{vec}(\boldsymbol{Y} - \boldsymbol{X} \boldsymbol{B}^{(t)})``. ``\odot`` denotes the Hadamard product.

For the EM algorithm, the updates in each iteration are

```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} (\boldsymbol{I}_d \otimes \boldsymbol{X})]^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} \text{vec}\ \boldsymbol{Y} \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \frac{1}{r_i} \boldsymbol{\Gamma}_i^{(t)} ( \boldsymbol{R}^{(t)T} \boldsymbol{V}_i \boldsymbol{R}^{(t)} - \boldsymbol{M}_i^{(t)} ) \boldsymbol{\Gamma}_i^{(t)} + \boldsymbol{\Gamma}_i^{(t)},
\end{aligned}
```
where ``r_i = \text{rank}(\boldsymbol{V}_i)``. As seen, the updates for mean effects ``\boldsymbol{B}`` are the same for these two algorithms.

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

# Special case: missing response
In the setting of missing response, the adjusted MM updates in each interation are
```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} (\boldsymbol{I}_d \otimes \boldsymbol{X})]^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} \text{vec}\ \boldsymbol{Z}^{(t)} \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \boldsymbol{L}_i^{-(t)T}[\boldsymbol{L}_i^{(t)T}\boldsymbol{\Gamma}_i^{(t)}(\boldsymbol{R}^{*(t)T}\boldsymbol{V}_i\boldsymbol{R}^{*(t)} + \boldsymbol{M}_i^{*(t)})\boldsymbol{\Gamma}_i^{(t)}\boldsymbol{L}_i^{(t)}]^{1/2} \boldsymbol{L}_i^{-(t)},
\end{aligned}
```
where ``\boldsymbol{Z}^{(t)}`` is the completed response matrix from conditional mean and ``\boldsymbol{M}_i^{*(t)} = (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)^T [(\boldsymbol{1}_d \boldsymbol{1}_d^T \otimes \boldsymbol{V}_i) \odot (\boldsymbol{\Omega}^{-(t)} \boldsymbol{P}^T \boldsymbol{C}^{(t)}\boldsymbol{P}\boldsymbol{\Omega}^{-(t)})] (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)``, while ``\boldsymbol{R}^{*(t)}`` is the ``n \times d`` matrix such that ``\text{vec}\ \boldsymbol{R}^{*(t)} = \boldsymbol{\Omega}^{-(t)} \text{vec}(\boldsymbol{Z}^{(t)} - \boldsymbol{X} \boldsymbol{B}^{(t)})``. Additionally, ``\boldsymbol{P}`` is the ``nd \times nd`` permutation matrix such that ``\boldsymbol{P} \cdot \text{vec}\ \boldsymbol{Y} = \begin{bmatrix} \boldsymbol{y}_{\text{obs}} \\ \boldsymbol{y}_{\text{mis}} \end{bmatrix}``, where ``\boldsymbol{y}_{\text{obs}}`` and ``\boldsymbol{y}_{\text{mis}}`` are vectors of observed and missing response values, respectively, in column-major order, and the block matrix ``\boldsymbol{C}^{(t)}`` is ``\boldsymbol{0}`` except for a lower-right block consisting of conditional variance. As seen, the two MM updates are of similar form.

# Special case: ``m = 2``
When there are ``m = 2`` variance components such that ``\boldsymbol{\Omega} = \boldsymbol{\Gamma}_1 \otimes \boldsymbol{V}_1 + \boldsymbol{\Gamma}_2 \otimes \boldsymbol{V}_2``, repeated inversion of the ``nd \times nd`` matrix ``\boldsymbol{\Omega}`` can be avoided and reduced to one ``d \times d`` generalized eigen-decomposition per iteration. Without loss of generality, if we assume ``\boldsymbol{V}_2`` to be positive definite, the generalized eigen-decomposition of the matrix pair ``(\boldsymbol{V}_1, \boldsymbol{V}_2)`` yields generalized eigenvalues ``\boldsymbol{d} = (d_1, \dots, d_n)^T`` and generalized eigenvectors ``\boldsymbol{U}`` such that ``\boldsymbol{U}^T \boldsymbol{V}_1 \boldsymbol{U} = \boldsymbol{D} = \text{diag}(\boldsymbol{d})`` and ``\boldsymbol{U}^T \boldsymbol{V}_2 \boldsymbol{U} = \boldsymbol{I}_n``. Similarly, if we let the generalized eigen-decomposition of ``(\boldsymbol{\Gamma}_1^{(t)}, \boldsymbol{\Gamma}_2^{(t)})`` be ``(\boldsymbol{\Lambda}^{(t)}, \boldsymbol{\Phi}^{(t)})`` such that ``\boldsymbol{\Phi}^{(t)T} \boldsymbol{\Gamma}_1^{(t)} \boldsymbol{\Phi}^{(t)} = \boldsymbol{\Lambda}^{(t)} = \text{diag}(\boldsymbol{\lambda^{(t)}})`` and ``\boldsymbol{\Phi}^{(t)T} \boldsymbol{\Gamma}_2^{(t)} \boldsymbol{\Phi}^{(t)} = \boldsymbol{I}_d``, then the MM updates in each iteration become

```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{\Phi}^{(t)T}\otimes \tilde{\boldsymbol{X}})^T (\boldsymbol{\Lambda}^{(t)} \otimes \boldsymbol{D} + \boldsymbol{I}_d \otimes \boldsymbol{I}_n)^{-1} (\boldsymbol{\Phi}^{(t)T}\otimes \tilde{\boldsymbol{X}})]^{-1} (\boldsymbol{\Phi}^{(t)T}\otimes \tilde{\boldsymbol{X}})^T (\boldsymbol{\Lambda}^{(t)} \otimes \boldsymbol{D} + \boldsymbol{I}_d \otimes \boldsymbol{I}_n)^{-1} \text{vec}(\tilde{\boldsymbol{Y}} \boldsymbol{\Phi}^{(t)}) \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \boldsymbol{L}_i^{-(t)T}[\boldsymbol{L}_i^{(t)T}\boldsymbol{N}_i^{(t)T}\boldsymbol{N}_i^{(t)}\boldsymbol{L}_i^{(t)}]^{1/2} \boldsymbol{L}_i^{-(t)},
\end{aligned}
```

where ``\tilde{\boldsymbol{X}} = \boldsymbol{U}^T \boldsymbol{X}``, ``\tilde{\boldsymbol{Y}} = \boldsymbol{U}^T \boldsymbol{Y}``, ``\boldsymbol{L}_1^{(t)}`` is the Cholesky factor of ``\boldsymbol{\Phi}^{(t)}\text{diag}(\text{tr}(\boldsymbol{D}(\lambda_k^{(t)}\boldsymbol{D} + \boldsymbol{I}_n)^{-1}), k = 1,\dots, d)\boldsymbol{\Phi}^{(t)T}``, ``\boldsymbol{L}_2^{(t)}`` is the Cholesky factor of ``\boldsymbol{\Phi}^{(t)}\text{diag}(\text{tr}((\lambda_k^{(t)}\boldsymbol{D} + \boldsymbol{I}_n)^{-1}), k = 1,\dots, d)\boldsymbol{\Phi}^{(t)T}``, ``\boldsymbol{N}_1^{(t)} = \boldsymbol{D}^{1/2}\{[(\tilde{\boldsymbol{Y}} - \tilde{\boldsymbol{X}}\boldsymbol{B})\boldsymbol{\Phi}^{(t)}]\oslash(\boldsymbol{d}\boldsymbol{\lambda}^{(t)T} + \boldsymbol{1}_n\boldsymbol{1}_d^T) \} \boldsymbol{\Lambda}^{(t)}\boldsymbol{\Phi}^{-(t)}``, and ``\boldsymbol{N}_2^{(t)} = \{[(\tilde{\boldsymbol{Y}} - \tilde{\boldsymbol{X}}\boldsymbol{B})\boldsymbol{\Phi}^{(t)}]\oslash(\boldsymbol{d}\boldsymbol{\lambda}^{(t)T} + \boldsymbol{1}_n\boldsymbol{1}_d^T) \} \boldsymbol{\Phi}^{-(t)}``. ``\oslash`` denotes the Hadamard quotient.

In this setting, the Fisher information matrix is equivalent to
```math
\begin{aligned}
\text{E} \left[- \frac{\partial^2}{\partial(\text{vec}\ \boldsymbol{B})^T \partial(\text{vec}\ \boldsymbol{B})} \mathcal{L} \right] =& (\boldsymbol{\Phi}^{T}\otimes \tilde{\boldsymbol{X}})^T (\boldsymbol{\Lambda} \otimes \boldsymbol{D} + \boldsymbol{I}_d \otimes \boldsymbol{I}_n)^{-1} (\boldsymbol{\Phi}^{T}\otimes \tilde{\boldsymbol{X}}) \\
\text{E} \left[ - \frac{\partial^2}{\partial (\text{vech} \ \boldsymbol{\Gamma}_i)^T \partial (\text{vec}\ \boldsymbol{B})} \mathcal{L} \right] =& \boldsymbol{0} \\
\text{E} \left[ - \frac{\partial^2}{\partial (\text{vech}\ \boldsymbol{\Gamma}_j)^T \partial (\text{vech}\ \boldsymbol{\Gamma}_i)} \mathcal{L} \right] =& \frac{1}{2} \boldsymbol{D}_d^T(\boldsymbol{\Phi}\otimes \boldsymbol{\Phi}) \text{diag}(\text{vec}(\boldsymbol{W}_{ij})) (\boldsymbol{\Phi}\otimes \boldsymbol{\Phi})^T\boldsymbol{D}_d,
\end{aligned}
```
where ``\boldsymbol{W}_{ij}`` is the ``d \times d`` matrix that has entries
```math
\begin{aligned}
(\boldsymbol{W}_{11})_{kl} = \text{tr}(\boldsymbol{D}^2(\boldsymbol{\lambda}_k \boldsymbol{D} + \boldsymbol{I}_n)^{-1}(\boldsymbol{\lambda}_l \boldsymbol{D} + \boldsymbol{I}_n)^{-1}) \\
(\boldsymbol{W}_{12})_{kl} = \text{tr}(\boldsymbol{D}(\boldsymbol{\lambda}_k \boldsymbol{D} + \boldsymbol{I}_n)^{-1}(\boldsymbol{\lambda}_l \boldsymbol{D} + \boldsymbol{I}_n)^{-1}) \\
(\boldsymbol{W}_{22})_{kl} = \text{tr}((\boldsymbol{\lambda}_k \boldsymbol{D} + \boldsymbol{I}_n)^{-1}(\boldsymbol{\lambda}_l \boldsymbol{D} + \boldsymbol{I}_n)^{-1}).
\end{aligned}
```