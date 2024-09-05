# Abstract type to describe generic variance component structure
abstract type VarCompStructure{T<:BlasReal} <: AbstractMatrix{T} end

# Define methods for AbstractArray interface
Base.size(S::VarCompStructure) = size(S.Σ)
Base.getindex(S::VarCompStructure, i::Int) = getindex(S.Σ, i)
Base.getindex(S::VarCompStructure, I::Vararg{Int,N}) where {N<:Integer} = getindex(S.Σ, I)
Base.length(S::VarCompStructure) = length(S.Σ)

# Define methods for StridedArray interface
Base.strides(S::VarCompStructure) = strides(S.Σ)
Base.unsafe_convert(::Type{Ptr{T}}, S::VarCompStructure) = S.Σ
Base.elsize(S::VarCompStructure) = Base.elsize(S.Σ)
Base.stride(S::VarCompStructure, i::Int) = stride(S.Σ, i)

struct Unstructured{T} <: VarCompStructure{T}
    Σ::Matrix{T}
    L::Matrix{T}
    function Unstructured(Σ::Matrix{T}) where {T<:BlasReal}
        L = similar(Σ)
        LAPACK.potrf!('L', copyto!(L, Σ))
        tril!(L)
        return Unstructured{T}(Σ, L)
    end
end

function Unstructured(::Type{T}, d::Int) where {T<:BlasReal}
    Σ = Matrix{T}(undef, d, d)
    L = similar(Σ)
    Unstructured{T}(Σ, L)
end
