module EigenTest

using MultiResponseVarianceComponentModels
using BenchmarkTools, LinearAlgebra, Profile, Random, StatsBase, Test

const MRVCModels = MultiResponseVarianceComponentModels
rng = MersenneTwister(456)

n, p, d, m = 855, 3, 4, 2
X = [ones(n) randn(rng, n, p - 1)] # design matrix including intercept
# V[1] has entries i * (n - j + 1) for j ≥ i, then scaled to be a correlation matrix
# V[2] is identity
V = Vector{Matrix{Float64}}(undef, m)
V = Vector{Matrix{Float64}}(undef, m)
V[1] = [j ≥ i ? i * (n - j + 1) : j * (n - i + 1) for i in 1:n, j in 1:n]
StatsBase.cov2cor!(V[1], [sqrt(V[1][i, i]) for i in 1:n])
V[2] = Matrix(UniformScaling(1.0), n, n)
# true parameter values
B_true = 2 * rand(p, d) # uniform on [0, 2]
Σ_true = [
    Matrix(UniformScaling(0.2), d, d), 
    Matrix(UniformScaling(0.6), d, d)
    ]
Ω_true = zeros(n * d, n * d)
for k in 1:m
    Ω_true .+= kron(Σ_true[k], V[k])
end
y = vec(X * B_true) + cholesky(Symmetric(Ω_true)).L * randn(rng, n * d)
Y = reshape(y, n, d)

@testset "constructor two component" begin
    model2 = MRTVCModel(Y, X[:, 1], V)
    model2 = MRTVCModel(Y[:, 1], X, V)
    model2 = MRTVCModel(Y[:, 1], X[:, 1], V)
    model2 = MRTVCModel(Y, V)
    model2 = MRTVCModel(Y[:, 1], V)
    model2 = MRTVCModel(Y, X, V, se = false)
    model2 = MRTVCModel(Y, X, V)
end

model2 = MRTVCModel(Y, X, V)
model  = MRVCModel(Y, X, V)

@testset "fit! two component by MLE with MM" begin
    MRVCModels.fit!(model2, algo = :MM, verbose = false, maxiter = 100)
    MRVCModels.fit!(model,  algo = :MM, verbose = false, maxiter = 100)
    println("||B̂_MRTVCModel - B̂_MRVCModel||       = $(norm(model2.B - model.B))")
    for k in 1:m
        println("||Σ̂[$k]_MRTVCModel - Σ̂[$k]_MRVCModel|| = $(norm(model2.Σ[k] - model.Σ[k]))")
    end
    println("||logl_MRTVCModel - logl_MRVCModel|| = $(abs2(model2.logl[1] - model.logl[1]))")
    println("||Bcov_MRTVCModel - Bcov_MRVCModel|| = $(norm(model2.Bcov - model.Bcov))")
    println("||Σcov_MRTVCModel - Σcov_MRVCModel|| = $(norm(model2.Σcov - model.Σcov))")
    println("||B_true - B̂||       = $(norm(B_true - model.B))")
    for k in 1:m
        println("||Σ_true[$k] - Σ̂[$k]|| = $(norm(Σ_true[k] - model.Σ[k]))")
    end
    # @test norm(model2.B - model.B) < 2e-4 # 6.986221757875547e-5
    # @test norm(model2.Σ[1] - model.Σ[1]) < 2e-4 # 0.0001772074443345503
    # @test norm(model2.Σ[2] - model.Σ[2]) < 2e-4 # 1.6274369068536495e-5
    # @test abs2(model2.logl[1] - model.logl[1]) < 2e-4 # 4.7903903529347e-8
    # @test norm(model2.Bcov - model.Bcov) < 2e-4 # 4.423117207077623e-5
    # @test norm(model2.Σcov - model.Σcov) < 2e-4 # 9.284043626500212e-6
end

model2 = MRTVCModel(Y, X, V, reml = true)
model  = MRVCModel(Y, X, V, reml = true)

@testset "fit! two component by REML with MM" begin
    MRVCModels.fit!(model2, algo = :MM, verbose = false, maxiter = 100)
    MRVCModels.fit!(model,  algo = :MM, verbose = false, maxiter = 100)
    println("||B̂_MRTVCModel - B̂_MRVCModel||       = $(norm(model2.B_reml - model.B_reml))")
    for k in 1:m
        println("||Σ̂[$k]_MRTVCModel - Σ̂[$k]_MRVCModel|| = $(norm(model2.Σ[k] - model.Σ[k]))")
    end
    println("||logl_MRTVCModel - logl_MRVCModel|| = $(abs2(model2.logl[1] - model.logl[1]))")
    println("||Bcov_MRTVCModel - Bcov_MRVCModel|| = $(norm(model2.Bcov_reml - model.Bcov_reml))")
    println("||Σcov_MRTVCModel - Σcov_MRVCModel|| = $(norm(model2.Σcov - model.Σcov))")
    println("||B_true - B̂||       = $(norm(B_true - model.B_reml))")
    for k in 1:m
        println("||Σ_true[$k] - Σ̂[$k]|| = $(norm(Σ_true[k] - model.Σ[k]))")
    end
    # @test norm(model2.B_reml - model.B_reml) < 9e-14 # 8.886934507200367e-14
    # @test norm(model2.Σ[1] - model.Σ[1]) < 9e-14 # 1.9800578221407427e-14
    # @test norm(model2.Σ[2] - model.Σ[2]) < 9e-14 # 2.9834222365790734e-15
    # @test abs2(model2.logl[1] - model.logl[1]) < 9e-14 # 1.987301421658649e-22
    # @test norm(model2.Bcov_reml - model.Bcov_reml) < 9e-14 # 5.04991691887946e-15
    # @test norm(model2.Σcov - model.Σcov) < 9e-14 # 1.0311308785032728e-15
end

end