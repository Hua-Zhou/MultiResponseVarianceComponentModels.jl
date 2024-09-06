# Abstract type to describe generic variance component structure
abstract type VarCompStructure{T<:BlasReal} <: DenseMatrix{T} end

# Define methods for AbstractArray interface
# Required
Base.size(S::VarCompStructure) = size(S.data)
Base.getindex(S::VarCompStructure, i::Int) = getindex(S.data, i)
Base.getindex(S::VarCompStructure, i::Int, j::Int) = getindex(S.data, i, j)
# Extras
Base.IndexStyle(::Type{<:VarCompStructure}) = Base.IndexLinear()
Base.setindex!(S::VarCompStructure{T}, val::T, i::Int) where {T} = (S.data[i] = val)
Base.setindex!(S::VarCompStructure{T}, val::T, i::Int, j::Int) where {T} = (S.data[i, j] = val)
Base.length(S::VarCompStructure) = length(S.data)
Base.axes(S::VarCompStructure) = Base.axes(S.data)
Base.dataids(S::VarCompStructure) = Base.dataids(S.data)

Base.collect(S::VarCompStructure) = S.data

# Define methods for StridedArray interface
Base.strides(S::VarCompStructure) = strides(S.data)
function Base.unsafe_convert(::Type{Ptr{T}}, S::VarCompStructure{T}) where {T}
    Base.unsafe_convert(Ptr{T}, S.data)
end
Base.elsize(::Type{<:VarCompStructure{T}}) where {T} = sizeof(T)
Base.stride(S::VarCompStructure, i::Int) = stride(S.data, i)

# Basic Structure
# struct Type{T} <: VarCompStructure{T}
#     # parameters
#     par_1         :: Matrix{T}
#       ⋮
#     pardim        :: Int
#     # working arrays
#     # `data` stores Σ in Minsoo's code/paper and Γ in my manuscript
#     data          :: Matrix{T}
#     datarank      :: Int
#     storage_nd_nd :: Matrix{T}
#     # Constants
#     V             :: Matrix{T}
#     Vrank         :: Int
# end