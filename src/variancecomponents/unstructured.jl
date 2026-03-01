struct Unstructured{T} <: VarCompStructure{T}
    # parameters
    L             :: Matrix{T}
    # working arrays
    Σ             :: Matrix{T}
    # storage_nd_nd :: Matrix{T}
    storage_nn    :: Matrix{T}
    storage_dd_1  :: Matrix{T}
    storage_dd_2  :: Matrix{T}
    storage_dd_3  :: Matrix{T}
    # Constants
    Σrank         :: Int
    V             :: Matrix{T}
    Vrank         :: Int
    pardim        :: Int
end

function Unstructured(Σ::Matrix{T}, V::Matrix{T}) where {T}
    n = LinearAlgebra.checksquare(V)
    d = LinearAlgebra.checksquare(Σ)
    nd = n * d
    Σ = copy(Σ)
    L = similar(Σ)
    LAPACK.potrf!('L', copyto!(L, Σ))
    tril!(L)
    pardim = ◺(d)
    Σrank = rank(Σ)
    # storage_nd_nd = Matrix{T}(undef, nd, nd)
    storage_nn = Matrix{T}(undef, n, n)
    storage_dd_1 = Matrix{T}(undef, d, d)
    storage_dd_2 = similar(storage_dd_1)
    storage_dd_3 = similar(storage_dd_1)
    # avoid evaluation of rank(V), only provide it if needed
    Vrank = n
    return Unstructured{T}(L, Σ, 
                        #    storage_nd_nd, 
                           storage_nn,
                           storage_dd_1, 
                           storage_dd_2, 
                           storage_dd_3, 
                           Σrank,
                           V, 
                           Vrank,
                           pardim
                           )
end

function Unstructured(d::Int, V::Matrix{T}) where {T}
    # initialize with identity, overwritten later
    Σ = Matrix{T}(UniformScaling{T}(1), d, d)
    # use constructor already defined
    Unstructured(Σ, V)
end

function initialize!(VC::Unstructured{T}) where {T}
    LAPACK.potrf!('L', copyto!(VC.L, VC.Σ))
    tril!(VC.L)
    return nothing
end

function initialize!(VC::Unstructured{T}, S::Matrix{T}) where {T}
    LAPACK.potrf!('L', copyto!(VC.L, S))
    tril!(VC.L)
    return nothing
end

function materialize!(VC::Unstructured{T}) where {T}
    BLAS.syrk!('L', 'N', one(T), VC.L, zero(T), VC.Σ)
    copytri!(VC.Σ, 'L')
    return nothing
end

function update_Σ!(VC::Unstructured{T}, ::Val{:MM}; tol=T(1e-6)) where {T}
    d = size(VC, 1)
    L = VC.L
    Σ = VC.Σ
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    C = VC.storage_dd_3
    storage_dd = view(VC.storage_nn, 1:d, 1:d)
    # Assume M = tr[Ωinv ⊗ V] already stored
    # Lower Cholesky factor L_M, M = L_M L_Mᵀ
    L_M, info = LAPACK.potrf!('L', M)
    info > 0 && throw(PosDefException(info))
    # Assume N = R̃ᵀVR̃ where R̃ = vec⁻¹(Ωinv vec(R)) stored
    # C = LᵀΣNΣL
    # assume current estimate of Σ(n) is stored in `VC.Σ`
    _update_ΣNΣ!(C, N, Σ, storage_dd)
    BLAS.trmm!('L', 'L', 'T', 'N', one(T), L_M, C)
    BLAS.trmm!('R', 'L', 'N', 'N', one(T), L_M, C)
    # Need symmetric square root of C, C small so syevr! should be fine
    # max eigenvalue of C is upper bounded by infinity norm
    λ_upper_bound = opnorm(C, Inf)
    # LAPACK.syevr! is allocating, which is annoying
    vals, vecs = LAPACK.syevr!('V', 'A', 'L', C, 
                               zero(T), 
                               λ_upper_bound, 
                               1, d, 
                               tol)
    for j in 1:d
        vj = view(vecs, :, j)
        if vals[j] > 0
            quadroot::T = sqrt(sqrt(vals[j]))
            vj .*= quadroot
        else
            fill!(vj, zero(T))
        end
    end
    BLAS.syrk!('L', 'N', one(T), vecs, zero(T), Σ)
    copytri!(Σ, 'L')
    # Right multiply by L_M⁻¹
    BLAS.trsm!('R', 'L', 'N', 'N', one(T), L_M, Σ)
    # Left multiply by L_M⁻ᵀ
    BLAS.trsm!('L', 'L', 'T', 'N', one(T), L_M, Σ)
    # Solve for parameters L
    LAPACK.potrf!('L', copyto!(L, Σ))
    tril!(L)
    return Σ
end

function update_Σ!(VC::Unstructured{T}, ::Val{:EM}) where {T}
    d = size(VC, 1)
    L = VC.L
    Σ = VC.Σ
    M = VC.storage_dd_1
    N = VC.storage_dd_2
    C = VC.storage_dd_3
    C .= (N .- M) ./ VC.Σrank
    mul!(M, C, Σ)
    for j in 1:d
        M[j, j] += 1
    end
    mul!(Σ, copyto!(N, Σ), M)
    # Project to 𝕊
    transpose!(C, Σ)
    Σ .= (Σ .+ C) ./ 2
    # Solve for parameters L
    LAPACK.potrf!('L', copyto!(L, Σ))
    tril!(L)
    return Σ
end

function finalize!(VC::Unstructured{T}) where {T}
    return nothing
end