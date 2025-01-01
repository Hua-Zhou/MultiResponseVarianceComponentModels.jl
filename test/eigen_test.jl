module EigenTest

using MultiResponseVarianceComponentModels
using BenchmarkTools, LinearAlgebra, Profile, Random, StatsBase, Test

const MRVCModels = MultiResponseVarianceComponentModels
Random.seed!(456)

n, p, d, m = 855, 3, 4, 2
X = [ones(n) randn(n, p - 1)] # design matrix including intercept
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
y = vec(X * B_true) + cholesky(Symmetric(Ω_true)).L * randn(n * d)
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
    # @test norm(model2.B - model.B) ≈ 0.00014716893555402874
    # @test norm(model2.Σ[1] - model.Σ[1]) ≈ 0.00016413437480828665
    # @test norm(model2.Σ[2] - model.Σ[2]) ≈ 1.959162421647704e-5
    # @test abs2(model2.logl[1] - model.logl[1]) ≈ 1.1554151166406821e-7
    # @test norm(model2.Bcov - model.Bcov) ≈ 4.381181547309963e-5
    # @test norm(model2.Σcov - model.Σcov) ≈ 7.364691385280541e-6
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
    # @test norm(model2.B_reml - model.B_reml) ≈ 1.0372715743775087e-14
    # @test norm(model2.Σ[1] - model.Σ[1]) ≈ 1.5009812336408224e-14
    # @test norm(model2.Σ[2] - model.Σ[2]) ≈ 1.914067895379648e-15
    # @test abs2(model2.logl[1] - model.logl[1]) ≈ 5.169878828456423e-24
    # @test norm(model2.Bcov_reml - model.Bcov_reml) ≈ 3.3256512190621786e-15
    # @test norm(model2.Σcov - model.Σcov) ≈ 8.216977255740531e-16
end

model2 = MRTVCModel(Y, X, V)
model  = MRVCModel(Y, X, V)

@testset "fit! two component by MLE with EM" begin
    MRVCModels.fit!(model2, algo = :EM, maxiter = 500)
    MRVCModels.fit!(model,  algo = :EM, maxiter = 500)
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
    # @test norm(model2.B - model.B) ≈ 7.349610254339635e-6
    # @test norm(model2.Σ[1] - model.Σ[1]) ≈ 1.1257198865106154e-5
    # @test norm(model2.Σ[2] - model.Σ[2]) ≈ 1.012150518153817e-6
    # @test abs2(model2.logl[1] - model.logl[1]) ≈ 1.19209171506449e-8
    # @test norm(model2.Bcov - model.Bcov) ≈ 2.9426471195867426e-6
    # @test norm(model2.Σcov - model.Σcov) ≈ 5.009462139695597e-7
end

model2 = MRTVCModel(Y, X, V, reml = true)
model  = MRVCModel(Y, X, V, reml = true)

@testset "fit! two component by REML with EM" begin
    MRVCModels.fit!(model2, algo = :EM, maxiter = 500)
    MRVCModels.fit!(model,  algo = :EM, maxiter = 500)
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
    # @test norm(model2.B_reml - model.B_reml) ≈ 9.777844544033894e-15
    # @test norm(model2.Σ[1] - model.Σ[1]) ≈ 1.1994372626858826e-14
    # @test norm(model2.Σ[2] - model.Σ[2]) ≈ 1.7232460901564156e-15
    # @test abs2(model2.logl[1] - model.logl[1]) ≈ 1.0132962503774589e-23
    # @test norm(model2.Bcov_reml - model.Bcov_reml) ≈ 2.9216826995302145e-15
    # @test norm(model2.Σcov - model.Σcov) ≈ 5.890258777783223e-16
end

end