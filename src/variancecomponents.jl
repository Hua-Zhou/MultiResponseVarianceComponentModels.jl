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
#     # `Σ` stores Σ in Minsoo's code/paper ≡ Γ in my manuscript
#     Σ          :: Matrix{T}
#     Σrank      :: Int
#     storage_nd_nd :: Matrix{T}
#     # Constants
#     V             :: Matrix{T}
#     Vrank         :: Int
# end

"""
    update_M!(VC::VarCompStructure, Ωinv::Matrix)

Updates the term M.
"""

# Shared Methods
function update_M!(VC::VarCompStructure{T}, Ωinv::Matrix{T}) where {T}
    kron_reduction!(Ωinv, VC.V, VC.storage_dd_1; sym = true)
end

"""
    update_N!(VC::VarCompStructure, Ωinv_R::Matrix)

Updates the term N = ̃R̃ᵀVR̃ where R̃ = vec⁻¹(Ω⁻¹ vec(R)), storing R̃ᵀVR̃ in `VC.storage_dd_2`
"""

function update_N!(VC::VarCompStructure{T}, Ωinv_R::Matrix{T}) where {T}
    n, d = size(Ωinv_R)
    # take an n × d slice out of working array
    VΩinv_R = view(VC.storage_nn, 1:n, 1:d)
    BLAS.symm!('L', 'L', one(T), VC.V, Ωinv_R, zero(T), VΩinv_R)
    mul!(VC.storage_dd_2, transpose(Ωinv_R), VΩinv_R)
    return VC.storage_dd_2
end