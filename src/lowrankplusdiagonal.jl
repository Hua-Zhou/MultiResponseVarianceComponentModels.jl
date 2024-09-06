struct LowRankPlusDiagonal{T} <: VarCompStructure{T}
    # parameters
    F             :: Matrix{T}
    Ψ             :: Vector{T}
    pardim        :: Int
    # working arrays
    data          :: Matrix{T}
    datarank      :: Int
    storage_nd_nd :: Matrix{T}
    # Constants
    V             :: Matrix{T}
    Vrank         :: Int
end

function LowRankPlusDiagonal(F::Matrix{T}, Ψ::Vector{T}, V::Matrix{T}) where {T<:BlasReal}
    n = LinearAlgebra.checksquare(V)
    d, r = size(F)
    nd = n * d
    (length(Ψ) == d) || throw(DimensionMismatch())
    data = Matrix{T}(undef, d, d)
    BLAS.syrk!('L', 'N', one(T), F, zero(T), data)
    copytri!(data, 'L')
    @inbounds for i in 1:d
        data[i, i] += Ψ[i]
    end
    # dimension here is the number of elements in augmented matrix [F ⋮ Ψ]
    pardim = d * (r + 1)
    # Here, `datarank` is not the rank of Σ = FFᵀ + Ψ, but the number of columns in F
    datarank = r
    storage_nd_nd = Matrix{T}(undef, nd, nd)
    # avoid evaluation of rank(V), only provide it if needed
    Vrank = n
    return LowRankPlusDiagonal{T}(F, Ψ, pardim, data, datarank, storage_nd_nd, V, Vrank)
end

function LowRankPlusDiagonal(d::Int, r::Int, V::Matrix{T}) where {T<:BlasReal}
    F = Matrix{T}(undef, d, r)
    fill!(F, zero(T))
    dg = Vector{T}(one(T), d)
    LowRankPlusDiagonal(F, dg, V)
end

function LowRankPlusDiagonal(Σ::Matrix{T}, r::Int, V::Matrix{T}) where {T<:BlasReal}
    # Best rank-r approximation to Σ
    d = LinearAlgebra.checksquare(Σ)
    # by convention, `eigen` is normally smallest to largest
    Σ_eig = eigen(Σ)
    reverse!(Σ_eig.values)
    reverse!(Σ_eig.vectors; dims=2)
    F = Matrix{T}(undef, d, r)
    for j in 1:r, 
        sqrtλ = sqrt(Σ_eig.values[j])
        for i in 1:d
            F[i, j] = sqrtλ * Σ_eig.vectors[i, j]
        end
    end
    dg = Vector{T}(undef, d)
    for j in 1:r, i in 1:d
        if j == 1
            dg[i] = Σ[i,i]
        end
        dg[i] += -abs2(F[i, j])
    end
    LowRankPlusDiagonal(F, dg, V)
end