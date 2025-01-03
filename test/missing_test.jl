module MissingTest

using MultiResponseVarianceComponentModels
using BenchmarkTools, LinearAlgebra, Profile, Random, StatsBase, Test

const MRVCModels = MultiResponseVarianceComponentModels
Random.seed!(789)

n, p, d, m = 855, 3, 4, 3
X = [ones(n) randn(n, p - 1)] # design matrix including intercept
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
B_true = 2 * rand(p, d) # uniform on [0, 2]
Σ_true = [
    Matrix(UniformScaling(0.2), d, d), 
    Matrix(UniformScaling(0.2), d, d),
    Matrix(UniformScaling(0.6), d, d)
    ]
Ω_true = zeros(n * d, n * d)
for k in 1:m
    Ω_true .+= kron(Σ_true[k], V[k])
end
y = vec(X * B_true) + cholesky(Symmetric(Ω_true)).L * randn(n * d)
Y = reshape(y, n, d)

Y_miss = Matrix{Union{eltype(Y), Missing}}(missing, size(Y))
copy!(Y_miss, Y)
Y_miss[rand(1:length(Y_miss), n)] .= missing

@testset "permute" begin
    Y = reshape(1:16, 4, 4)
    Y_miss = Matrix{Union{Float64, Missing}}(missing, size(Y))
    copy!(Y_miss, Y)
    Y_miss[[1, 5, 6, 15, 16]] .= missing
    P, invP, n_miss, Y_imputed = permute(Y_miss)
    @test P == [2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 1, 5, 6, 15, 16]
    @test invP == [12, 1, 2, 3, 13, 14, 4, 5, 6, 7, 8, 9, 10, 11, 15, 16]
    @test n_miss == 5
    @test Y_imputed == [3.0 7.5 9.0 13.0; 2.0 7.5 10.0 14.0; 3.0 7.0 11.0 13.5; 4.0 8.0 12.0 13.5]
end

@testset "fit! missing response with MM" begin
    model = MRVCModel(Y_miss, X, V; se = false)
    # @timev MultiResponseVarianceComponentModels.fit!(model)
end

end