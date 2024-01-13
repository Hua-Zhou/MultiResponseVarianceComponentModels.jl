# Examples

# Simulate data
```@repl 1
using MultiResponseVarianceComponentModels, LinearAlgebra, Random
Random.seed!(6789)
n = 1_000;  # n of observations
d = 4;      # n of responses
p = 10;     # n of covariates
m = 5;      # n of variance components
X = rand(n, p);
B = rand(p, d)
V = [zeros(n, n) for _ in 1:m]; # kernel matrices
Σ = [zeros(d, d) for _ in 1:m]; # variance components
for i in 1:m
    Vi = randn(n, n)
    copy!(V[i], Vi' * Vi)
    Σi = randn(d, d)
    copy!(Σ[i], Σi' * Σi)
end
Ω = zeros(n * d, n * d); # overall nd-by-nd covariance matrix Ω
for i = 1:m
    Ω += kron(Σ[i], V[i])
end
Ωchol = cholesky(Ω);
Y = X * B + reshape(Ωchol.L * randn(n * d), n, d)
```

!!! note

    In the case of heritability and genetic correlation analyses, one can use classic genetic relationship matrices (GRMs) for ``\boldsymbol{V}_i``'s, which in turn can be constructed using [__SnpArrays.jl__](https://github.com/OpenMendel/SnpArrays.jl).

# Maximum likelihood estimation
```@repl 1
model = MultiResponseVarianceComponentModel(Y, X, V)
@timev fit!(model, verbose = true)
```

For residual maximum likelihood estimation, you can instead type:
```julia
@timev fit!(model, reml = true, verbose = true)
```

Then variance components and mean effects estimates can be accessed through
```@repl 1
model.Σ
model.B
hcat(vec(B), vec(model.B))
reduce(hcat, [hcat(vec(Σ[i]), vec(model.Σ[i])) for i in 1:m])
```

# Standard errors
Sampling variance and covariance of these estimates are
```@repl 1
model.Σcov
model.Bcov
```
Corresponding standard error of these estimates are
```@repl 1
sqrt.(diag(model.Σcov))
sqrt.(diag(model.Bcov))
```