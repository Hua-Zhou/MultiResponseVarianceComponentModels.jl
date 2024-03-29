```@meta
CurrentModule = MultiResponseVarianceComponentModels
```

# MultiResponseVarianceComponentModels

[__MultiResponseVarianceComponentModels.jl__](https://github.com/hua-zhou/MultiResponseVarianceComponentModels.jl) is a package for fitting and testing multivariate response variance components linear mixed models of form

```math
\text{vec}\ \boldsymbol{Y} \sim \mathcal{N}(\text{vec}(\boldsymbol{X} \boldsymbol{B}), \sum_{i=1}^m \boldsymbol{\Gamma}_i \otimes \boldsymbol{V}_i),
```

where ``\boldsymbol{Y}`` and ``\boldsymbol{X}`` are ``n \times d`` response and  ``n \times p`` predictor matrices, respectively, and ``\boldsymbol{V}_1, \ldots, \boldsymbol{V}_m`` are ``m`` known ``n \times n`` positive semidefinite matrices. ``\text{vec}\ \boldsymbol{Y}`` creates an ``nd \times 1`` vector from ``\boldsymbol{Y}`` by stacking its columns and ``\otimes`` denotes the Kronecker product. The parameters of the model include ``p \times d`` mean effects ``\boldsymbol{B}`` and ``d \times d`` variance components (``\boldsymbol{\Gamma}_1, \dots, \boldsymbol{\Gamma}_m``), which [MultiResponseVarianceComponentModels.jl](https://github.com/hua-zhou/MultiResponseVarianceComponentModels.jl) estimates through either [minorization-maximization (MM)](https://en.wikipedia.org/wiki/MM_algorithm) or [expectation–maximization (EM)](https://en.wikipedia.org/wiki/Expectation–maximization_algorithm) algorithms.

!!! note

    [MultiResponseVarianceComponentModels.jl](https://github.com/hua-zhou/MultiResponseVarianceComponentModels.jl) is not suitable for biobank-scale data. We recommend using this package for datasets of size up to ``n \cdot d \approx 50000``. This package also currently works for balanced data without any missing data.

# Installation

To use [MultiResponseVarianceComponentModels.jl](https://github.com/hua-zhou/MultiResponseVarianceComponentModels.jl), type:
```julia
using Pkg
Pkg.add(url = "https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl.git")
```

This documentation was built using [Documenter.jl](https://github.com/JuliaDocs/Documenter.jl).

```@example
using Dates # hide
println("Documentation built $(Dates.now()) with Julia $(VERSION)") # hide
```