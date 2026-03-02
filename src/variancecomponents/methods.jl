# Abstract type to describe generic variance component structure
abstract type VarCompStructure{T<:BlasReal} <: DenseMatrix{T} end

# Define methods for AbstractArray interface
# Required
Base.size(VC::VarCompStructure) = size(VC.Σ)
Base.getindex(VC::VarCompStructure, i::Int) = getindex(VC.Σ, i)
Base.getindex(VC::VarCompStructure, i::Int, j::Int) = getindex(VC.Σ, i, j)
# Extras
Base.IndexStyle(::Type{<:VarCompStructure}) = Base.IndexLinear()
function Base.setindex!(VC::VarCompStructure{T}, val::T, i::Int) where {T}
    return (VC.Σ[i] = val)
end
function Base.setindex!(
    VC::VarCompStructure{T}, val::T, i::Int, j::Int
) where {T}
    return (VC.Σ[i, j] = val)
end
Base.length(VC::VarCompStructure) = length(VC.Σ)
Base.axes(VC::VarCompStructure) = Base.axes(VC.Σ)

Base.collect(VC::VarCompStructure) = VC.Σ

# Define methods for StridedArray interface
Base.strides(VC::VarCompStructure) = strides(VC.Σ)
function Base.unsafe_convert(::Type{Ptr{T}}, VC::VarCompStructure{T}) where {T}
    return Base.unsafe_convert(Ptr{T}, VC.Σ)
end
Base.elsize(::Type{<:VarCompStructure{T}}) where {T} = sizeof(T)
Base.stride(VC::VarCompStructure, i::Int) = stride(VC.Σ, i)

# Basic Structure
# struct Type{T} <: VarCompStructure{T}
#     # parameters
#     par_1         :: Matrix{T}
#       ⋮
#     # working arrays
#     # `Σ` stores Σ in Minsoo's code/paper ≡ Γ in my manuscript
#     Σ          :: Matrix{T}
#     Σrank      :: Int
#     storage_nd_nd :: Matrix{T}
#     # Constants
#     V             :: Matrix{T}
#     Vrank         :: Int
#     pardim        :: Int
# end

function LinearAlgebra.mul!(
    dest::AbstractVector{T},
    VC::VarCompStructure{T},
    b::AbstractVector{T},
    alpha::T,
    beta::T,
) where {T}
    d = size(VC.Σ, 1)
    # Note (Σ ⊗ V) b = vec(V B Σ) where b = vec(B) and B is n by d
    storage_nd_1 = view(VC.storage_nn, :, 1:d)
    storage_nd_2 = view(VC.storage_nn, :, (d + 1):(2d))
    BLAS.symm!(
        'L', 'L', one(T), VC.V, copyto!(storage_nd_1, b), zero(T), storage_nd_2
    )
    # overwrite storage_nd_1 with current value of dest
    # α * vec(V B Σ) * β * dest
    iszero(beta) ? copyto!(storage_nd_1, dest) : nothing
    BLAS.symm!('L', 'R', alpha, VC.Σ, storage_nd_2, beta, storage_nd_1)
    return copyto!(dest, storage_nd_1)
end

# should be the most efficent method for n >> d
# only uses level3 BLAS, main overhead from data movement
function LinearAlgebra.mul!(
    dest::AbstractMatrix{T},
    VC::VarCompStructure{T},
    B::AbstractMatrix{T},
    alpha::T,
    beta::T,
) where {T}
    n = size(VC.V, 1)
    d = size(VC.Σ, 1)
    p = size(B, 2)
    dp = d * p
    # B is nd by p, we reshape it to n by dp, then apply the same logic as the 
    # vector case, then reshape to np by d, and right multiply by Σ
    storage_n_dp_1 = copyto!(view(VC.storage_nn, :, 1:dp), B)
    storage_n_dp_2 = view(VC.storage_nn, :, (dp + 1):(2dp))
    BLAS.symm!('L', 'L', one(T), VC.V, storage_n_dp_1, zero(T), storage_n_dp_2)
    # Blockwise reshape
    for k in 1:p
        colidx_1   = ((k - 1) * d + 1):(k * d)
        colidx_2   = (dp + (k - 1) * d + 1):(2dp)
        storage_nd = view(VC.storage_nn, :, colidx_1)
        V_Bk       = view(VC.storage_nn, :, colidx_2)
        iszero(beta) ? copyto!(storage_nd, view(dest, :, k)) : nothing
        BLAS.symm!('L', 'R', alpha, VC.Σ, V_Bk, beta, storage_nd)
        copyto!(view(dest, :, k), storage_nd)
    end
    return dest
end

function LinearAlgebra.mul!(
    dest::AbstractVecOrMat{T}, VC::VarCompStructure{T}, b::AbstractVecOrMat{T}
) where {T}
    return mul!(dest, VC, b, one(T), zero(T))
end

"""
    update_M!(M::AbstractMatrix{T}, VC::VarCompStructure{T}, Ωinv::AbstractMatrix{T})

Updates the term M.
"""

# Shared Methods
function update_M!(
    M::AbstractMatrix{T}, VC::VarCompStructure{T}, Ωinv::AbstractMatrix{T}
) where {T}
    return kron_reduction!(Ωinv, VC.V, M; sym=true)
end

function update_M!(VC::VarCompStructure{T}, Ωinv::AbstractMatrix{T}) where {T}
    return update_M!(VC.storage_dd_1, VC, Ωinv)
end

"""
    update_N!(N::AbstractMatrix{T}, VC::VarCompStructure{T}, Ωinv_R::AbstractMatrix{T})

Updates the term N = R̃ᵀVR̃ where R̃ = vec⁻¹(Ω⁻¹ vec(R))
"""

function update_N!(
    N::AbstractMatrix{T},
    VC::VarCompStructure{T},
    Ωinv_R::AbstractMatrix{T},
    storage_n_d::AbstractMatrix{T},
) where {T}
    n, d = size(Ωinv_R)
    # take an n by d slice
    V_Ωinv_R = storage_n_d
    BLAS.symm!('L', 'L', one(T), VC.V, Ωinv_R, zero(T), V_Ωinv_R)
    mul!(N, transpose(Ωinv_R), V_Ωinv_R)
    return N
end

function update_N!(VC::VarCompStructure{T}, Ωinv_R::AbstractMatrix{T}) where {T}
    storage_n_d = view(VC.storage_nn, :, axes(Ωinv_R, 2))
    return update_N!(VC.storage_dd_2, VC, Ωinv_R, storage_n_d)
end

function _update_ΣNΣ!(
    C::AbstractMatrix{T},
    N::AbstractMatrix{T},
    Σ::AbstractMatrix{T},
    storage_dd::AbstractMatrix{T},
) where {T}
    BLAS.symm!('L', 'L', one(T), N, Σ, zero(T), storage_dd)
    BLAS.symm!('L', 'L', one(T), Σ, storage_dd, zero(T), C)
    return copytri!(C, 'L')
end