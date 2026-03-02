
"""
    ΩMatrix{T, K, VC}

Lazy representation of Ω = Σ[1]⊗V[1] + Σ[2]⊗V[2] + … + Σ[K]⊗V[K].

Types:
- T: element type (e.g. Float64)
- K: number of variance components
- VC: tuple type of variance component structures

Fields:
- `varcomp`: tuple of variance component structures (holding Σ and V)
- `factor`: pre-allocated matrix for Cholesky factor or full Ω
- `factorized`: Ref{Bool} to track if factorization is up to date
- `n`: sample size (dimension of V)
- `d`: response dimension (dimension of Σ)
- `storage_n_d`: pre-allocated matrix for intermediate computations

"""
struct ΩMatrix{T<:BlasReal,K,VC<:Tuple{Vararg{VarCompStructure{T},K}}}
    # Tuple of variance components (holds Σ[i] and V[i] for i=1..K)
    varcomp::VC
    # Factorization storage
    factor::Matrix{T}
    # state tracking (Ref for mutability in immutable struct)
    evaluated::Ref{Bool}
    factorized::Ref{Bool}
    # Dimensions
    n::Int                   # sample size
    d::Int                   # response dimension
    # storage for intermediate operations
    storage_n_d::Matrix{T}
end

# Methods interface for compatibility
Base.size(Ω::ΩMatrix{T}) where {T} = (Ω.n * Ω.d, Ω.n * Ω.d)
Base.eltype(::ΩMatrix{T}) where {T} = T

"""
    ΩMatrix(varcomp::Tuple{Vararg{VarCompStructure{T}, K}})

Construct lazy Ω matrix from tuple of variance component structures.

- T: element type (e.g. Float64)
- K: number of variance components
- VC: tuple type of variance component structures
"""
function ΩMatrix(
    varcomp::VC
) where {T,K,VC<:Tuple{Vararg{VarCompStructure{T},K}}}
    n = size(first(varcomp).V, 1)
    d = size(first(varcomp).Σ, 1)
    nd = n * d
    factor = Matrix{T}(undef, nd, nd)
    storage_n_d = Matrix{T}(undef, n, d)
    return ΩMatrix{T,K,VC}(
        varcomp, factor, Ref(false), Ref(false), n, d, storage_n_d
    )
end

# Helper functions to avoid [] syntax everywhere
@inline is_factorized(Ω::ΩMatrix{T}) where {T} = Ω.factorized[]
@inline set_factorized!(Ω::ΩMatrix{T}, val::Bool) where {T} =
    (Ω.factorized[] = val)

@inline is_evaluated(Ω::ΩMatrix{T}) where {T} = Ω.evaluated[]
@inline set_evaluated!(Ω::ΩMatrix{T}, val::Bool) where {T} =
    (Ω.evaluated[] = val)

@generated function finalize!(vc::VC) where {VC<:Tuple}
    K = length(VC.parameters)
    updates = [:(finalize!(vc[$i])) for i in 1:K]
    quote
        $(updates...)
    end
end

function finalize!(Ω::ΩMatrix{T}) where {T}
    finalize!(Ω.varcomp)
    return nothing
end

"""
    _kron_sum!(dest::Matrix, vc::Tuple)

Generated function to efficiently compute `dest` = Σ[1]⊗V[1] + ... + Σ[K]⊗V[K].
Uses metaprogramming to unroll the loop over K variance components
"""
@generated function _kron_sum!(dest::Matrix{T}, vc::VC) where {T,VC<:Tuple}
    K = length(VC.parameters)
    ops = [:(kron_axpy!(vc[$i].Σ, vc[$i].V, dest)) for i in 1:K]
    quote
        $(Expr(:meta, :propagate_inbounds))
        begin
            $(ops...)
        end
        return dest
    end
end

"""
    ___materialize!(vc::Tuple)

Unexported generated function to evaluate each Σ from variance component
structures. Shouldn't ever need to be called if update_Σ! is implemented 
correctly for each structure.
"""
@generated function ___materialize!(vc::VC) where {VC<:Tuple}
    K = length(VC.parameters)
    ops = [:(materialize!(vc[$i])) for i in 1:K]
    quote
        $(ops...)
        return dest
    end
end

"""
    materialize!(dest::Matrix, Ω::ΩMatrix)

Materialize Ω = Σ[1]⊗V[1] + ... + Σ[K]⊗V[K] into preallocated matrix `dest`. 
"""
function materialize!(dest::Matrix{T}, Ω::ΩMatrix{T}) where {T}
    if is_evaluated(Ω) == false
        fill!(dest, zero(T))
        _kron_sum!(dest, Ω.varcomp)
    end
    return dest
end

function materialize!(Ω::ΩMatrix{T}) where {T}
    materialize!(Ω.factor, Ω)
    # set_evaluated!(Ω, true)
    return Ω.factor
end

"""
    factorize!(Ω::ΩMatrix)

Compute Cholesky factorization Ω = LLᵀ in-place, only if not already factorized. 
Returns Ω with updated factorization state.
"""
function factorize!(Ω::ΩMatrix{T}) where {T}
    if is_evaluated(Ω) == false
        throw("Ω needs to be materialized before factorization")
    else
        # is_evaluated(Ω) == true
        if is_factorized(Ω) == false
            _, info = LAPACK.potrf!('L', Ω.factor)
            info > 0 && throw(PosDefException(info))
            set_factorized!(Ω, true)
        end
    end
    return Ω.factor
end

"""
    ldiv!(dest::Matrix, Ω::ΩMatrix, B::Matrix)

Solve ΩX = B directly using Cholesky factorization, modifying Ω in-place.
"""
function LinearAlgebra.ldiv!(
    dest::AbstractVecOrMat{T}, Ω::ΩMatrix{T}, B::AbstractVecOrMat{T}
) where {T}
    factorize!(Ω)
    LAPACK.potrs!('L', Ω.factor, copyto!(dest, B))
    return dest
end

"""
    _mul!(dest::AbstractVector{T}, vc::Tuple, x::AbstractVector{T})

Generated function to efficiently compute `dest` = (Σ[1]⊗V[1] + ... + Σ[K]⊗V[K]) x.
Uses metaprogramming to unroll the loop over K variance components
"""
@generated function _mul!(
    dest::AbstractVector{T}, vc::VC, x::AbstractVector{T}
) where {T,VC<:Tuple}
    K = length(VC.parameters)
    ops = [:(mul!(dest, vc[$i], x, one(T), one(T))) for i in 1:K]
    quote
        $(Expr(:meta, :propagate_inbounds))
        $(ops...)
        return dest
    end
end

function LinearAlgebra.mul!(
    dest::AbstractVector{T}, Ω::ΩMatrix{T}, x::AbstractVector{T}
) where {T}
    fill!(dest, zero(T))
    _mul!(dest, Ω.varcomp, x)
    return dest
end

"""
    logdet!(Ω::ΩMatrix)

Compute log determinant of `Ω` by direct Cholesky factorization, modifying `Ω` in-place.
"""
function logdet(Ω::ΩMatrix{T}) where {T}
    if is_factorized(Ω) == true
        return _logdet_tri(Ω.factor) * 2
    else
        throw("Ω needs to be factorized to compute log determinant")
    end
end

"""
    _logdet_tri(L::Matrix)

Compute log determinant for a triangular matrix `L`.
"""
function _logdet_tri(L::Matrix{T}) where {T}
    res = zero(T)
    @inbounds for i in axes(L, 1)
        res += log(L[i, i])
    end
    return res
end

@generated function update_M!(
    vc::VC, Ωinv::AbstractMatrix{T}
) where {T,VC<:Tuple}
    K = length(VC.parameters)
    ops = [:(update_M!(vc[$i], Ωinv)) for i in 1:K]
    quote
        $(ops...)
    end
end

@generated function update_N!(
    vc::VC, Ωinv_R::AbstractMatrix{T}
) where {T,VC<:Tuple}
    K = length(VC.parameters)
    ops = [:(update_N!(vc[$i], Ωinv_R)) for i in 1:K]
    quote
        $(ops...)
    end
end

@generated function update_Σ!(vc::VC, algo::Val) where {VC<:Tuple}
    K = length(VC.parameters)
    ops = [:(update_Σ!(vc[$i], algo)) for i in 1:K]
    quote
        $(ops...)
    end
end

function update_Σ!(vc::Tuple; algo::Symbol=:MM)
    update_Σ!(vc, Val(algo))
    return nothing
end