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

function update_fisher_inf!(VC::VarCompStructure{T}, Ωinv::Matrix{T}) where {T}
    n = size(VC.storage_nn, 1)
    d = size(VC, 1)
    fill!(VC.storage_nd_nd, zero(T))
    for j in 1:d, i in j:d
        jstart = (j - 1) * n + 1
        jstop = j * n
        istart = (i - 1) * n + 1
        istop = i * n
        Ωinv_ij = view(Ωinv, istart:istop, jstart:jstop)
        out = view(VC.storage_nd_nd, istart:istop, jstart:jstop)
        BLAS.symm!('R', 'L', one(T), VC.V, Ωinv_ij, zero(T), out)
    end
    Ωinv_V = VC.storage_nd_nd
    ℱ = VC.storage_dd_dd
    for l in 1:d
        for k in l:d
            lstart = (l - 1) * n + 1
            lstop = l * n
            kstart = (k - 1) * n + 1
            kstop = k * n
            Ωinv_kl = view(Ωinv_V, kstart:kstop, lstart:lstop)
            for j in 1:d
                for i in j:d
                    jstart = (j - 1) * n + 1
                    jstop = j * n
                    istart = (i - 1) * n + 1
                    istop = i * n
                    Ωinv_ij = view(Ωinv_V, istart:istop, jstart:jstop)
                    cidx = (l - 1) * d + j
                    ridx = (k - 1) * d + i
                    ℱ[ridx, cidx] = dot(transpose(Ωinv_ij), Ωinv_kl)
                end
            end
            copytri!(view(ℱ, ((k-1) * d + 1):(k*d), ((l-1) * d + 1):(l * d)), 'L')
        end
    end
    return ℱ ./= 2
end