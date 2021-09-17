module MVCalculusTest

using MultiResponseVarianceComponentModels
using BenchmarkTools, LinearAlgebra, Random, Test
import LinearAlgebra: copytri!

Random.seed!(123)

@testset "kron_axpy!" begin
    m, n, p, q = 3, 4, 5, 6
    A = randn(m, n)
    X = randn(p, q)
    Y = zeros(m * p, n * q)
    kron_axpy!(A, X, Y)
    @test norm(Y - kron(A, X)) ≈ 0.0
    bm = @benchmark kron_axpy!($A, $X, $Y) setup=(fill!($Y, 0))
    @test allocs(bm) == 0
    display(bm); println()
end

@testset "kron_reduction! (asymmetric case)" begin
    n, q, p, r = 2, 3, 4, 5
    X = randn(n, q)
    Y = randn(p, r)
    M = kron(X, Y)
    # calculate gradient wrt X using kron_reduction!
    dX1 = zero(X)
    kron_reduction!(M, Y, dX1)
    # alternative way to calculate gradient wrt X
    dX2 = zero(X)
    for j in 1:q, i in 1:n
        dX2[i, j] = dot(Y, M[(i-1)*p+1:i*p, (j-1)*r+1:j*r])
    end
    @test norm(dX1 - dX2) < 1e-8
    bm = @benchmark kron_reduction!($M, $Y, $dX1)
    display(bm); println()
end

@testset "kron_reduction! (symmetric case)" begin
    n, q, p, r = 3, 3, 5, 5
    X = randn(n, q)
    Y = randn(p, r)
    copytri!(X, 'U')
    copytri!(Y, 'U')
    M = kron(X, Y)
    # calculate gradient wrt X using kron_reduction!
    dX1 = zero(X)
    kron_reduction!(M, Y, dX1, true)
    # alternative way to calculate gradient wrt X
    dX2 = zero(X)
    for j in 1:q, i in 1:n
        dX2[i, j] = dot(Y, M[(i-1)*p+1:i*p, (j-1)*r+1:j*r])
    end
    @test norm(dX1 - dX2) < 1e-8
    bm = @benchmark kron_reduction!($M, $Y, $dX1, true)
    @test allocs(bm) == 0
    display(bm); println()
end

end