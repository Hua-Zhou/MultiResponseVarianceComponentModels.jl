# Examples

# Simulate data
```@repl 1
using MultiResponseVarianceComponentModels, LinearAlgebra, Random
Random.seed!(6789)
n = 1_000;  # number of observations
d = 4;      # number of responses
p = 10;     # number of covariates
m = 5;      # number of variance components
X = rand(n, p);
B = rand(p, d);
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
    kron_axpy!(Σ[i], V[i], Ω) # Ω = Σ[1]⊗V[1] + ... + Σ[m]⊗V[m]
end
Ωchol = cholesky(Ω);
Y = X * B + reshape(Ωchol.L * randn(n * d), n, d)
```

!!! note

    In the case of heritability and genetic correlation analyses, one can use classic genetic relationship matrices (GRMs) for ``\boldsymbol{V}_i``'s, which in turn can be constructed using [__SnpArrays.jl__](https://github.com/OpenMendel/SnpArrays.jl).

# Maximum likelihood estimation
```@repl 1
model = MRVCModel(Y, X, V)
@timev fit!(model)
```

Then variance components and mean effects estimates can be accessed through
```@repl 1
model.Σ
model.B
hcat(vec(B), vec(model.B)) # comparison of true values and estimates
reduce(hcat, [hcat(vech(Σ[i]), vech(model.Σ[i])) for i in 1:m]) # comparison of true values and estimates
```

# Standard errors
Sampling variance and covariance of these estimates are
```@repl 1
model.Σcov
model.Bcov
```
Corresponding standard error of these estimates are
```@repl 1
sqrt.(diag(model.Σcov)) # m * (binomial(d, 2) + d) parameters
sqrt.(diag(model.Bcov))
```

# Residual maximum likelihood estimation
For REML estimation, you can instead:
```@repl 1
model = MRVCModel(Y, X, V; reml = true)
@timev fit!(model)
```

Variance components and mean effects estimates and their standard errors can be accessed through:
```@repl 1
model.Σ
model.B_reml # note model.B_reml not model.B
hcat(vec(B), vec(model.B_reml))
reduce(hcat, [hcat(vech(Σ[i]), vech(model.Σ[i])) for i in 1:m])
model.Σcov
model.Bcov_reml # note model.Bcov_reml not model.Bcov
sqrt.(diag(model.Σcov))
sqrt.(diag(model.Bcov_reml))
```

# Estimation only
Calculating standard errors can be memory-consuming, so you could instead forego such calculation via:
```julia
model = MRVCModel(Y, X, V; se = false)
@timev fit!(model)
```

# Special case: missing response
You can also fit data with missing response. For example:
```julia
Y_miss = Matrix{Union{eltype(Y), Missing}}(missing, size(Y))
copy!(Y_miss, Y)
Y_miss[rand(1:length(Y_miss), n)] .= missing

model = MRVCModel(Y_miss, X, V; se = false)
@timev fit!(model)
```

# Special case: ``m = 2``
When there are __two__ variance components, you can accelerate fitting by avoiding large matrix inversion per iteration. To illustrate this, you can first simulate data as done previously but with larger ``nd`` and ``m = 2``.
```@repl 1
function simulate(n, d, p, m)
    X = rand(n, p)
    B = rand(p, d)
    V = [zeros(n, n) for _ in 1:m]
    Σ = [zeros(d, d) for _ in 1:m]
    for i in 1:m
        Vi = randn(n, n)
        mul!(V[i], transpose(Vi), Vi)
        Σi = randn(d, d)
        mul!(Σ[i], transpose(Σi), Σi)
    end
    Y = X * B
    for i in 1:m
        Z = randn(n, d)
        Σchol = cholesky(Σ[i])
        Vchol = cholesky(V[i])
        rmul!(Z, transpose(Σchol.L))
        lmul!(Vchol.L, Z)
        Y .= Y .+ Z
    end
    Y, X, V, B, Σ
end
Y, X, V, B, Σ = simulate(5_000, 4, 10, 2)
```

Then you can fit data as follows:
```@repl 1
model = MRTVCModel(Y, X, V)
@timev fit!(model)
reduce(hcat, [hcat(vech(Σ[i]), vech(model.Σ[i])) for i in 1:2])
```