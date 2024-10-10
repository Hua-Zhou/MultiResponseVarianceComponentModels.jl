# Abstract type to describe generic variance component structure
abstract type VarCompStructure{T<:BlasReal} <: DenseMatrix{T} end

# Define methods for AbstractArray interface
# Required
Base.size(S::VarCompStructure) = size(S.Σ)
Base.getindex(S::VarCompStructure, i::Int) = getindex(S.Σ, i)
Base.getindex(S::VarCompStructure, i::Int, j::Int) = getindex(S.Σ, i, j)
# Extras
Base.IndexStyle(::Type{<:VarCompStructure}) = Base.IndexLinear()
Base.setindex!(S::VarCompStructure{T}, val::T, i::Int) where {T} = (S.Σ[i] = val)
Base.setindex!(S::VarCompStructure{T}, val::T, i::Int, j::Int) where {T} = (S.Σ[i, j] = val)
Base.length(S::VarCompStructure) = length(S.Σ)
Base.axes(S::VarCompStructure) = Base.axes(S.Σ)

Base.collect(S::VarCompStructure) = S.Σ

# Define methods for StridedArray interface
Base.strides(S::VarCompStructure) = strides(S.Σ)
function Base.unsafe_convert(::Type{Ptr{T}}, S::VarCompStructure{T}) where {T}
    Base.unsafe_convert(Ptr{T}, S.Σ)
end
Base.elsize(::Type{<:VarCompStructure{T}}) where {T} = sizeof(T)
Base.stride(S::VarCompStructure, i::Int) = stride(S.Σ, i)

# Basic Structure
# struct Type{T} <: VarCompStructure{T}
#     # parameters
#     par_1         :: Matrix{T}
#       ⋮
#     pardim        :: Int
#     # working arrays
#     # `Σ` stores Σ in Minsoo's code/paper and Γ in my manuscript
#     Σ          :: Matrix{T}
#     Σrank      :: Int
#     storage_nd_nd :: Matrix{T}
#     # Constants
#     V             :: Matrix{T}
#     Vrank         :: Int
# end