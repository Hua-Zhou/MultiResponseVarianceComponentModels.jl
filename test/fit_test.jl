module FitTest

using MultiResponseVarianceComponentModels
using BenchmarkTools, LinearAlgebra, Profile, Random, StatsBase, Test
import LinearAlgebra: copytri!

const MRVC = MultiResponseVarianceComponentModels

rng = MersenneTwister(123)

n, p, d, m = 855, 3, 4, 3
# design matrix, including intercept
X = [ones(n) randn(rng, n, p - 1)]
# V[1] is an AR1(ρ) matrix, with entries ρ^|i-j|
# V[2] has entries i * (n - j + 1) for j ≥ i, then scaled to be a correlation matrix
# V[3] is identity
ρ = 0.5
V = Vector{Matrix{Float64}}(undef, m)
V[1] = [ρ^abs(i - j) for i in 1:n, j in 1:n]
V[2] = [j ≥ i ? i * (n - j + 1) : j * (n - i + 1) for i in 1:n, j in 1:n]
StatsBase.cov2cor!(V[2], [sqrt(V[2][i, i]) for i in 1:n])
V[3] = Matrix(UniformScaling(1.0), n, n)
# true parameter values
Β_true = 2rand(p, d) # uniform on [0, 2]
Σ_true = [
    Matrix(UniformScaling(0.2), d, d), 
    Matrix(UniformScaling(0.2), d, d),
    Matrix(UniformScaling(0.6), d, d)
]
Ω_true = zeros(n * d, n * d)
for k in 1:m
    Ω_true .+= kron(Σ_true[k], V[k])
end
y = vec(X * Β_true) + cholesky(Symmetric(Ω_true)).L * randn(rng, n * d)
Y = reshape(y, n, d)

mrvc = MultiResponseVarianceComponentModel(Y, X, V)

@testset "log-likelihood at the truth" begin
    copyto!(mrvc.Β, Β_true)
    update_res!(mrvc)
    for k in 1:m
        copyto!(mrvc.Σ[k], Σ_true[k])
    end
    update_Ω!(mrvc)
    @show loglikelihood!(mrvc)
    bm = @benchmark loglikelihood!($mrvc)
    display(bm); println()
    @test allocs(bm) == 0
end

@testset "fit!" begin
    @time MRVC.fit!(mrvc, verbose = true)
    println("Btrue:")
    display(Β_true)
    println()
    println("B̂:")
    display(mrvc.Β)
    println()
    for k in 1:m
        println("Σ_true[$k]:")
        display(Σ_true[k])
        println()
        println("Σ̂[$k]:")
        display(mrvc.Σ[k])
        println()
    end
end

# @testset "profile fit!" begin
#     Profile.clear()
#     @profile MRVC.fit!(mrvc, maxiter=20)
#     Profile.print(format=:flat)
# end

end
