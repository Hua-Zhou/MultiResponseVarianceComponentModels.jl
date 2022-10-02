# Advanced details

# Estimation
For the MM algorithm, the updates in each iteration are

```math
\begin{aligned}
\text{vec}\ \boldsymbol{B}^{(t)} &= [(\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} (\boldsymbol{I}_d \otimes \boldsymbol{X})]^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-(t)} \text{vec}\ \boldsymbol{Y} \\
\boldsymbol{\Gamma}_i^{(t + 1)} &= \boldsymbol{L}_i^{-(t)T}[\boldsymbol{L}_i^{(t)T}(\boldsymbol{\Gamma}_i^{(t)}\boldsymbol{R}^{(t)T}\boldsymbol{V}_i\boldsymbol{R}^{(t)}\boldsymbol{\Gamma}_i^{(t)})\boldsymbol{L}_i^{(t)}]^{1/2} \boldsymbol{L}_i^{-(t)},
\end{aligned}
```
where ``\boldsymbol{\Omega}^{(t)} = \sum_{i=1}^m \boldsymbol{\Gamma}_i^{(t)} \otimes \boldsymbol{V}_i`` and ``\boldsymbol{L}_i^{(t)}`` is the Cholesky factor of ``(\boldsymbol{I}_d \otimes \boldsymbol{1}_n)^T [(\boldsymbol{1}_d \boldsymbol{1}_d^T \otimes \boldsymbol{V}_i) \odot \boldsymbol{\Omega}^{-(t)}] (\boldsymbol{I}_d \otimes \boldsymbol{1}_n)``, while ``\boldsymbol{R}^{(t)}`` is the ``n \times d`` matrix such that ``\text{vec}\ \boldsymbol{R}^{(t)} = \boldsymbol{\Omega}^{-(t)} \text{vec}(\boldsymbol{Y} - \boldsymbol{X} \boldsymbol{B}^{(t)})``.

# Inference
Standard errors for the parameters were estimated using the Fisher information matrix, where

```math
\begin{aligned}
\text{E} \left[- \frac{\partial^2}{\partial(\text{vec}\ \boldsymbol{B})^T \partial(\text{vec}\ \boldsymbol{B})} \mathcal{L} \right] &= (\boldsymbol{I}_d \otimes \boldsymbol{X}^T) \boldsymbol{\Omega}^{-1} (\boldsymbol{I}_d \otimes \boldsymbol{X}) \\
\text{E} \left[ - \frac{\partial^2}{\partial (\text{vech}\ \boldsymbol{L}_i)^T \partial (\text{vec}\ \boldsymbol{B})} \mathcal{L} \right] &= \boldsymbol{0} \\
\text{E} \left[ - \frac{\partial^2}{\partial (\text{vech}\ \mathbf{\Gamma}_j)^T \partial (\text{vech}\ \mathbf{\Gamma}_i)} \mathcal{L} \right] &= \frac{1}{2} \boldsymbol{U}_i^T (\boldsymbol{\Omega}^{-1} \otimes \boldsymbol{\Omega}^{-1}) \boldsymbol{U}_j
\end{aligned}
```

and ``\boldsymbol{U}_i = (\boldsymbol{I}_d \otimes \boldsymbol{K}_{nd} \otimes \boldsymbol{I}_n) (\boldsymbol{I}_{d^2} \otimes \text{vec}\ \boldsymbol{V}_i) \boldsymbol{D}_{d}``. Here, ``\boldsymbol{K}_{nd}`` is the ``nd \times nd`` [commutation matrix](https://en.wikipedia.org/wiki/Commutation_matrix) and ``\boldsymbol{D}_{d}`` the ``d^2 \times \frac{d(d+1)}{2}`` [duplication matrix](https://en.wikipedia.org/wiki/Duplication_and_elimination_matrices).