abstract type VarCompStructure{T} end
# TODO: Should the covariance structure also contain V[k]?
struct Unstructured{T} <: VarCompStructure{T}
    Σ :: Matrix{T}
end

function Unstructured(Σ::Matrix{T}) where {T<:BlasReal}
    Unstructured{T}(Σ)
end

function Unstructured(d::Int, ::Type{T}) where {T<:BlasReal}
    Unstructured{T}(Matrix{T}(undef, d, d))
end

struct LowRankPlusDiagonal{T} <: VarCompStructure{T}
    Σ    :: Matrix{T}
    F    :: Matrix{T}
    Ψ    :: Diagonal{T, Vector{T}}
end

function LowRankPlusDiagonal(F::Matrix{T}, ψ::Vector{T}) where {T<:BlasReal}
    d, r = size(F)
    length(ψ) == d || throw("ψ must be length $d")
    Σ = Matrix{T}(undef, d, d)
    BLAS.syrk!('L', 'N', one(T), F, zero(T), Σ)
    @inbounds for i in eachindex(Ψ)
        Σ[i, i] += ψ[i]
    end
    copytri!(Σ, 'L')
    LowRankPlusDiagonal{T}(Σ, F, Diagonal(ψ))
end

function LowRankPlusDiagonal(d::Int, r::Int, ::Type{T}) where {T<:BlasReal}
    LowRankPlusDiagonal{T}(
        Matrix{T}(undef, d, d), 
        Matrix{T}(undef, d, r),
        Diagonal{T}(undef, d))
end

struct LowRank{T} <: VarCompStructure{T}
    Σ    :: Matrix{T}
    F    :: Matrix{T}
end

function LowRank(F::Matrix{T}) where {T<:BlasReal}
    d, r = size(F)
    Σ = Matrix{T}(undef, d, d)
    BLAS.syrk!('L', 'N', one(T), F, zero(T), Σ)
    copytri!(Σ, 'L')
    LowRank{T}(Σ, F)
end

function LowRank(d::Int, r::Int, ::Type{T}) where {T<:BlasReal}
    LowRank{T}(
        Matrix{T}(undef, d, d), 
        Matrix{T}(undef, d, r)
        )
end