# MultiResponseVarianceComponentModels

## Installation
```julia
julia> ]
pkg> add https://github.com/Hua-Zhou/MultiResponseVarianceComponentModels.jl.git
```
## Examples
```julia
using Pkg
Pkg.activate(@__DIR__)
Pkg.add("MultiResponseVarianceComponentModels")

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
@time fit!(model, verbose = true) # ~ 143 seconds
# residual maximum likelihood estimation
@time fit!(model, reml = true, verbose = true)

model.Σ
reduce(hcat,[hcat(vec(Σ[i]), vec(model.Σ[i])) for i in 1:m])
model.Σcov # sampling variance by inverse of Fisher information matrix
diag(model.Σcov) # (binomial(d, 2) + d) * m variance/covariance parameters
model.logl
```
## References
> Zhou et al., (2019). MM Algorithms For Variance Components Models. Journal of Computational and Graphical Statistics, 28(2): 350–361, https://doi.org/10.1080/10618600.2018.1529601.