<p align="center"><img width="100%" style="border-radius: 5px;" src="docs/src/assets/logo.svg"></p>

# MultiResponseVarianceComponentModels
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://hua-zhou.github.io/MultiResponseVarianceComponentModels.jl/dev)
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](http://hua-zhou.github.io/MultiResponseVarianceComponentModels.jl/stable)
[![CI](https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl/workflows/CI/badge.svg)](https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl/actions)
[![Codecov](https://codecov.io/gh/Hua-Zhou/MultiResponseVarianceComponentModels.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/Hua-Zhou/MultiResponseVarianceComponentModels.jl)

MultiResponseVarianceComponentModels.jl is a <a href="https://julialang.org"><img src="https://julialang.org/assets/infra/julia.ico" width="10em"> Julia </a>package that allows fitting and testing multivariate response variance components linear mixed models of form 

<p align="center"><img width="80%" style="border-radius: 5px;" src="docs/src/assets/MRVC.png"></p>

## Installation
```julia
julia> ]
pkg> add https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl.git
```
## Documentation
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](http://hua-zhou.github.io/MultiResponseVarianceComponentModels.jl/dev)

## Examples
```julia
using MultiResponseVarianceComponentModels, LinearAlgebra, Random

# simulation
begin
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
end

# maximum likelihood estimation
model = MRVCModel(Y, X, V)
@timev fit!(model) # ~ 30 seconds

# residual maximum likelihood estimation
model = MRVCModel(Y, X, V; reml = true)
@timev fit!(model) # ~ 30 seconds

# variance components estimates
model.Σ
# comparison of true values and estimates
reduce(hcat, [hcat(vech(Σ[i]), vech(model.Σ[i])) for i in 1:m])
# sampling variance by inverse of Fisher information matrix
model.Σcov
diag(model.Σcov) # m * (binomial(d, 2) + d) parameters
# log-likelihood
model.logl
```
## References
- <u>H. Zhou, L. Hu, J. Zhou, and K. Lange: **MM algorithms for variance components models** (2019) ([link](https://doi.org/10.1080/10618600.2018.1529601))</u>
- <u>M. Kim: **Gene regulation in the human brain and the biological mechanisms underlying psychiatric disorders** (2022) ([link](https://escholarship.org/uc/item/9v08q5f7))</u>

## See also
- J. Kim, J. Shen, A. Wang, D.V. Mehrotra, S. Ko, J.J. Zhou, and H. Zhou: **VCSEL: Prioritizing SNP-set by penalized variance component selection** (2021) ([link](http://doi.org/10.1214/21-aoas1491))
- L. Hu, W. Lu, J. Zhou, and H. Zhou: **MM algorithms for variance component estimation and selection in logistic linear mixed models** (2019) ([link](http://doi.org/10.5705/ss.202017.0220))
- J.J. Zhou, T. Hu, D. Qiao, M.H. Cho, and H. Zhou: **Boosting gene mapping power and efficiency with efficient exact variance component tests of single nucleotide polymorphism sets** (2016) ([link](http://doi.org/10.1534/genetics.116.190454))