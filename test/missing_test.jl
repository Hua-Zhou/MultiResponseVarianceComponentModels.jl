module MissingTest

using MultiResponseVarianceComponentModels
using Test

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

end

