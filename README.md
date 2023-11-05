<p align="center"><img width="100%" style="border-radius: 5px;" src="docs/src/assets/logo.svg"></p>

# MultiResponseVarianceComponentModels
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://hua-zhou.github.io/MultiResponseVarianceComponentModels.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://hua-zhou.github.io/MultiResponseVarianceComponentModels.jl/stable)
[![CI](https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl/workflows/CI/badge.svg)](https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl/actions)
[![Codecov](https://codecov.io/gh/Hua-Zhou/MultiResponseVarianceComponentModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Hua-Zhou/MultiResponseVarianceComponentModels.jl)

MultiResponseVarianceComponentModels.jl is a <a href="https://julialang.org"><img src="https://julialang.org/assets/infra/julia.ico" width="10em"> Julia </a>package that allows fitting and testing multivariate response variance components linear mixed models of form 

<p align="center"><img width="70%" style="border-radius: 5px;" src="docs/src/assets/MRVC.png"></p>

## Installation
```julia
julia> ]
pkg> add https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl.git
```
## Documentation
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](http://hua-zhou.github.io/MultiResponseVarianceComponentModels.jl/dev)

## Examples
```julia
using MultiResponseVarianceComponentModels, LinearAlgebra, Random
const MRVC = MultiResponseVarianceComponentModel
# simulation
Random.seed!(1234)
n = 1_000  # n of observations
d = 4      # n of responses
p = 10     # n of covariates
m = 3      # n of variance components
X = rand(n, p)
B = rand(p, d) 
V = [zeros(n, n) for _ in 1:m] # kernel matrices
Σ = [zeros(d, d) for _ in 1:m] # variance components
for i in 1:m
    Vi = randn(n, n)
    copy!(V[i], Vi' * Vi)
    Σi = randn(d, d)
    copy!(Σ[i], Σi' * Σi)
end
Ω = zeros(n * d, n * d) # overall nd-by-nd covariance matrix Ω
for i = 1:m
    Ω += kron(Σ[i], V[i])
end
Ωchol = cholesky(Ω)
Y = X * B + reshape(Ωchol.L * randn(n * d), n, d)
# maximum likelihood estimation
model = MRVC(Y, X, V)
@time fit!(model, verbose = true) # ~ 30 seconds
# residual maximum likelihood estimation
@time fit!(model, reml = true, verbose = true)

model.Σ
reduce(hcat, [hcat(vec(Σ[i]), vec(model.Σ[i])) for i in 1:m])
model.Σcov # sampling variance by inverse of Fisher information matrix
diag(model.Σcov) # (binomial(d, 2) + d) * m variance/covariance parameters
model.logl
```
## References
- H. Zhou, L. Hu, J. Zhou, and K. Lange: **MM algorithms for variance components models** (2019) ([link](https://doi.org/10.1080/10618600.2018.1529601))
- M. Kim: **Gene regulation in the human brain and the biological mechanisms underlying psychiatric disorders** (2022) ([link](https://escholarship.org/uc/item/9v08q5f7))

## See also
- J. Kim, J. Shen, A. Wang, D.V. Mehrotra, S. Ko, J.J. Zhou, and H. Zhou: **VCSEL: Prioritizing SNP-set by penalized variance component selection** (2021) ([link](http://doi.org/10.1214/21-aoas1491))